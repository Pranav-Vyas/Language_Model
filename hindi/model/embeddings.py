"""Token and positional embeddings (Phase 2, stage 1 of the forward pass).

Duplicated verbatim into every language tree; never imports from a sibling.

WHY POSITIONS MUST BE INJECTED AT ALL
-------------------------------------
Self-attention is a *set* operation: it computes a weighted sum over all
positions, and that sum is permutation-symmetric. "कुत्ते ने आदमी को काटा" and
"आदमी ने कुत्ते को काटा" would produce identical representations without a
position signal. Order is not implicit anywhere in the attention mechanism,
so it has to be added to the input.

WHY LEARNED ABSOLUTE, AND WHAT IT COSTS
---------------------------------------
Choice: a trainable `nn.Embedding(max_seq_len, d_model)` indexed by position.

The honest trade-off, stated because the spec asks how the scheme bounds
sequence length:

  * A learned table has exactly `max_seq_len` rows. Position `max_seq_len`
    does not exist -- not "degrades", *does not exist*. The model therefore
    has a HARD context limit of `max_seq_len` tokens, and `forward` asserts
    it rather than letting an index error surface deep in a backward pass.
  * Sinusoidal encodings can be evaluated at any position, but the model was
    never trained on those values, so quality still degrades out of range --
    "computable" is not "generalises".
  * Relative schemes (RoPE, ALiBi) are what actually generalise past the
    trained length, because attention then depends on the *offset* between
    two tokens rather than their absolute indices.

We take learned absolute because it is the simplest scheme to explain end to
end, and because we are not claiming length generalisation: context is fixed
at 512 for training and evaluation alike. The cost is `max_seq_len x d_model`
= 512 x 384 = 196,608 parameters, under 1% of the budget.

THE ABLATION SWITCH (`use_positional`)
--------------------------------------
`use_positional=False` removes the table entirely -- it is not zeroed, not
frozen, not multiplied by zero: `self.pos_emb` is never constructed, so the
196,608 parameters genuinely leave the model and the count drops. This backs
the optional no-positional-embeddings ablation.

What survives is worth stating, because it is the whole reason the ablated
model does not simply collapse: a *causal* decoder still leaks position
through its own mask. Token 0 attends over 1 position, token 50 over 51, so
"how much context is visible" is itself a positional signal the stack can
read off. Removing this table removes the *explicit* position signal, not
every position signal.

ADDED, NOT CONCATENATED
-----------------------
`x = token_emb + pos_emb`. Concatenating would grow d_model and force every
downstream matrix to widen. Adding keeps the width fixed and lets the model
learn to separate "what this token is" from "where it sits" inside the same
vector space -- the positional signal should mean "second position", not
"this specific word at position two".
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TokenAndPositionEmbedding(nn.Module):
    """Map token ids to vectors and add a learned position signal.

    Shapes
    ------
    input   ids : (B, T)          int64, values in [0, vocab_size)
    output      : (B, T, d_model) float
    """

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int, dropout: float,
                 use_positional: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.use_positional = use_positional

        self.token_emb = nn.Embedding(vocab_size, d_model)
        # Constructed only when used, so the ablated model is genuinely 196,608
        # parameters lighter rather than carrying a dead tensor.
        self.pos_emb = nn.Embedding(max_seq_len, d_model) if use_positional else None
        # Embedding dropout: the spec lists it as its own stage. It regularises
        # the input representation itself, before any attention has run.
        self.drop = nn.Dropout(dropout)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # Fail loudly at the boundary rather than deep inside a kernel.
        if ids.dim() != 2:
            raise ValueError(f"expected ids of shape (B, T), got {tuple(ids.shape)}")
        B, T = ids.shape
        if T > self.max_seq_len:
            # Still enforced in the ablated model: the causal mask buffer is
            # allocated at max_seq_len, so the bound is real either way -- only
            # the reason for it changes.
            reason = ("Learned absolute positions have no row for that index -- this "
                      "is the hard context bound of this positional scheme, not a "
                      "soft degradation."
                      if self.use_positional else
                      "No positional table here, but the causal mask is still "
                      "allocated at max_seq_len, so the bound holds.")
            raise ValueError(
                f"sequence length {T} exceeds max_seq_len {self.max_seq_len}. {reason}")

        tok = self.token_emb(ids)                                  # (B, T, d)
        if self.pos_emb is None:                                   # ablation path
            return self.drop(tok)                                  # (B, T, d)
        pos_ids = torch.arange(T, device=ids.device)               # (T,)
        pos = self.pos_emb(pos_ids).unsqueeze(0)                   # (1, T, d) -> broadcasts over B
        return self.drop(tok + pos)                                # (B, T, d)
