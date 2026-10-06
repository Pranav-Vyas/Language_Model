"""Decoder-only transformer language model (Phase 2, full forward pass).

Duplicated verbatim into every language tree; never imports from a sibling.
Built only from nn.Linear / nn.Embedding / nn.LayerNorm / nn.Dropout and raw
tensor ops -- no nn.Transformer*, no HuggingFace classes, no fused attention.

FORWARD PASS, END TO END
------------------------
    ids (B, T)
      -> token embedding + learned positional embedding      (B, T, d)
      -> embedding dropout
      -> N x [ pre-norm attention residual, pre-norm FFN residual ]
      -> final LayerNorm                                     (B, T, d)
      -> LM head (weight-tied to the token embedding)        (B, T, vocab)
      -> cross-entropy against ids shifted by one

WHY A FINAL LayerNorm
---------------------
In a pre-norm stack every sub-layer normalises its *input*, so nothing
normalises the output of the last block -- the residual stream leaves the
stack unnormalised and its scale has grown with depth. Without a final norm
the LM head sees inputs whose variance depends on how many layers happen to be
stacked. This layer is not optional bookkeeping; omitting it destabilises
training in exactly the deep stacks pre-norm exists to enable.

WEIGHT TYING
------------
The LM head reuses the token-embedding matrix as its projection (Press & Wolf,
2017). Both map between the same two spaces -- vocabulary and d_model -- just
in opposite directions, so learning two independent matrices for one
relationship wastes capacity.

The saving here is exact and large: vocab x d_model = 16,000 x 384 =
6,144,000 parameters, roughly a quarter of the ~25M budget, redirected into
depth. Reported explicitly because the spec asks that tying be stated and its
saving explained.

The tied matrix is used transposed: embedding lookup is a row select from
(vocab, d), while the head needs (d, vocab), and F.linear(x, W) computes
x @ W.T -- so passing the same tensor gives the correct orientation.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from block import TransformerBlock
from embeddings import TokenAndPositionEmbedding


@dataclass
class ModelConfig:
    """Every architectural knob, logged beside each checkpoint.

    Defaults are the locked Phase-2 configuration. Depth over width: at a
    fixed ~25M budget, 10 layers at d_model=384 beats 5 at 512 for language
    modelling, and gives a richer early-vs-late contrast for the attention
    analysis the spec requires.
    """

    vocab_size: int = 16_000
    d_model: int = 384
    n_layers: int = 10
    n_heads: int = 6          # d_k = 384 / 6 = 64, the usual per-head width
    d_ff: int = 1_536         # 4 x d_model
    max_seq_len: int = 512
    emb_dropout: float = 0.1
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1
    tie_weights: bool = True
    # False drops the positional table entirely (optional no-PE ablation).
    # Defaults True, so every pre-ablation checkpoint reloads unchanged.
    use_positional_embeddings: bool = True
    lang_code: str = ""       # stamped into checkpoints; blocks cross-language loads

    def to_dict(self) -> dict:
        return asdict(self)


class TransformerLM(nn.Module):
    """Decoder-only causal language model."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = TokenAndPositionEmbedding(
            cfg.vocab_size, cfg.d_model, cfg.max_seq_len, cfg.emb_dropout,
            use_positional=cfg.use_positional_embeddings)

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.max_seq_len,
                             cfg.attn_dropout, cfg.resid_dropout)
            for _ in range(cfg.n_layers)
        ])

        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            # Shared storage, not a copy: one tensor, two uses.
            self.lm_head.weight = self.embed.token_emb.weight

        self.apply(self._init_weights)

        # GPT-2's scaled init for layers that write into the residual stream.
        # With N blocks each adding to the stream, the accumulated variance
        # grows like N; scaling those projections by 1/sqrt(2N) keeps the
        # stream's scale roughly constant with depth.
        for name, p in self.named_parameters():
            if name.endswith("w_o.weight") or name.endswith("fc_out.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * cfg.n_layers) ** 0.5)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, ids: torch.Tensor, targets: torch.Tensor | None = None,
                return_attention: bool = False):
        """Run the stack; optionally return loss and per-layer attention.

        ids     : (B, T) int64
        targets : (B, T) int64 or None. Already shifted by the caller -- the
                  data loader hands (x, y) where y[t] == x[t+1].
        returns : logits (B, T, vocab), loss or None, attentions or None
        """
        x = self.embed(ids)                                   # (B, T, d)

        attentions = [] if return_attention else None
        for blk in self.blocks:
            if return_attention:
                x, w = blk(x, return_attention=True)           # w: (B, h, T, T)
                attentions.append(w)
            else:
                x = blk(x)

        x = self.ln_f(x)                                      # (B, T, d)
        logits = self.lm_head(x)                              # (B, T, vocab)

        loss = None
        if targets is not None:
            # Flatten batch and time: cross_entropy wants (N, C) against (N,).
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,   # lets a caller mask positions if needed
            )

        return (logits, loss, attentions) if return_attention else (logits, loss)

    # -- parameter accounting ------------------------------------------------

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count parameters, counting a tied matrix ONCE.

        `nn.Module.parameters()` already de-duplicates shared tensors, so a
        tied head is not double counted. Stated explicitly because a naive
        sum over `named_parameters()` would count it twice and overstate the
        model by 6.1M -- a quarter of the budget.
        """
        seen = set()
        total = 0
        for p in self.parameters():
            if trainable_only and not p.requires_grad:
                continue
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total

    def parameter_breakdown(self) -> dict[str, int]:
        """Where the parameters actually sit -- the depth/width justification."""
        cfg = self.cfg
        emb_tok = cfg.vocab_size * cfg.d_model
        emb_pos = cfg.max_seq_len * cfg.d_model if cfg.use_positional_embeddings else 0
        per_attn = 4 * cfg.d_model * cfg.d_model            # W_q, W_k, W_v, W_o
        per_ffn = 2 * cfg.d_model * cfg.d_ff + cfg.d_ff + cfg.d_model
        per_ln = 2 * 2 * cfg.d_model                        # two LayerNorms, weight+bias
        per_block = per_attn + per_ffn + per_ln
        head = 0 if cfg.tie_weights else cfg.vocab_size * cfg.d_model
        return {
            "token_embedding": emb_tok,
            "position_embedding": emb_pos,
            "per_block": per_block,
            "all_blocks": per_block * cfg.n_layers,
            "final_layernorm": 2 * cfg.d_model,
            "lm_head": head,
            "tied_saving": emb_tok if cfg.tie_weights else 0,
            "total_counted": self.num_parameters(),
        }
