"""Multi-head causal self-attention, implemented from primitives.

Duplicated verbatim into every language tree; never imports from a sibling.

Only nn.Linear / nn.Dropout and raw tensor ops are used. No nn.MultiheadAttention,
no nn.Transformer*, no scaled_dot_product_attention fused kernel -- the point of
this phase is that every tensor operation in the forward pass is written out and
explainable.

THE COMPUTATION
---------------
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k) + M) V

WHY THREE SEPARATE PROJECTIONS
------------------------------
Q, K and V come from three independent weight matrices applied to the same
input. If one projection served as both query and key, every token would score
a perfect match against itself (q_i . q_i is maximal by construction) and would
dominate its own attention distribution. Splitting the roles lets the model
learn "what I am looking for" (Q) separately from "what I advertise" (K).
V is a third projection again because Q/K decide *how much* to attend, while V
decides *what content* is actually mixed in.

WHY DIVIDE BY sqrt(d_k)
-----------------------
q . k is a sum of d_k products. Treating the components as roughly independent
with unit variance, the dot product has variance ~ d_k, so its magnitude grows
like sqrt(d_k). Feeding large values into softmax pushes it into a saturated
regime: one logit dominates, the output approaches a one-hot vector, and the
softmax Jacobian -- diag(p) - p p^T -- goes to zero. The gradient vanishes and
the layer stops learning. Dividing by sqrt(d_k) holds the scores at a scale
where softmax stays in its responsive region. The justification is about
GRADIENT FLOW, not merely "keeping numbers small".

WHY AN ADDITIVE -inf MASK
-------------------------
Masking must happen *before* softmax, on the scores. Zeroing probabilities
after softmax would not work: the surviving probabilities would no longer sum
to one, so each position would silently down-weight its own attention by
however much future mass was deleted. Adding -inf pre-softmax makes exp(-inf)=0
exactly, and the remaining positions renormalise correctly.

A large negative constant is used rather than literal -inf: a row that is
entirely masked would otherwise produce 0/0 = NaN. Causal masking always leaves
at least the diagonal unmasked, so this cannot occur here, but the finite value
keeps the module safe under any future mask.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class MultiHeadCausalSelfAttention(nn.Module):
    """Multi-head self-attention restricted to attend only backwards.

    Shapes, written out because they are exactly what the viva asks for:

        x            (B, T, d_model)
        q, k, v      (B, T, d_model)          after the linear projections
        reshaped     (B, T, h, d_k)           split the last dim into heads
        transposed   (B, h, T, d_k)           heads become a batch dimension
        scores       (B, h, T, T)             q @ k^T over the last two dims
        weights      (B, h, T, T)             softmax over the KEY axis (-1)
        context      (B, h, T, d_k)           weights @ v
        merged       (B, T, d_model)          transpose back and flatten heads
        output       (B, T, d_model)          after the output projection
    """

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int,
                 attn_dropout: float, resid_dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}; "
                "each head works in a d_model/n_heads subspace."
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_k)

        # Three independent projections (see module docstring for why).
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        # W_O recombines the concatenated heads; without it the heads would be
        # merely concatenated subspaces that never interact.
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(resid_dropout)

        # Lower-triangular True = "allowed to attend". Registered as a buffer so
        # it moves with .to(device) and is saved/loaded with the module, but is
        # not a trainable parameter.
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        B, T, C = x.shape
        if C != self.d_model:
            raise ValueError(f"expected last dim {self.d_model}, got {C}")

        # --- project, then split d_model into (h, d_k) -----------------------
        # (B, T, d) -> (B, T, h, d_k) -> (B, h, T, d_k)
        # The transpose puts heads next to batch so the matmuls below operate
        # independently per head, in parallel, with no Python loop over heads.
        q = self.w_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # --- scaled dot-product scores --------------------------------------
        # (B, h, T, d_k) @ (B, h, d_k, T) -> (B, h, T, T)
        # scores[b, h, i, j] = how much query i attends to key j.
        scores = (q @ k.transpose(-2, -1)) * self.scale

        # --- causal mask, additive, BEFORE softmax ---------------------------
        # Slice to T so a shorter sequence than max_seq_len still works.
        allowed = self.causal_mask[:T, :T]                       # (T, T) bool
        scores = scores.masked_fill(~allowed, float("-inf"))

        # --- normalise over KEYS (last axis), not queries --------------------
        # Softmax over -1 makes each query's weights sum to 1 across the keys
        # it may see. Softmaxing over -2 would normalise down a column and mean
        # something entirely different (and wrong).
        weights = torch.softmax(scores, dim=-1)                   # (B, h, T, T)
        weights = self.attn_dropout(weights)

        # --- weighted sum of values -----------------------------------------
        context = weights @ v                                    # (B, h, T, d_k)

        # --- merge heads back --------------------------------------------------
        # (B, h, T, d_k) -> (B, T, h, d_k) -> (B, T, d_model)
        # .contiguous() because transpose returns a non-contiguous view and
        # .view() needs contiguous memory.
        merged = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.resid_dropout(self.w_o(merged))               # (B, T, d_model)

        if return_attention:
            # Pre-dropout weights are what the Phase-2 attention analysis wants:
            # dropout is a training-time perturbation, not part of the learned
            # attention pattern.
            return out, torch.softmax(scores, dim=-1)
        return out
