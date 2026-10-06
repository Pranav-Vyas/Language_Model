"""One pre-norm transformer block (Phase 2, stage 3).

Duplicated verbatim into every language tree; never imports from a sibling.

PRE-NORM vs POST-NORM -- AND WHY WE CHOSE PRE-NORM
--------------------------------------------------
Post-norm, as published in Vaswani et al. (2017):

    y = LayerNorm(x + Sublayer(x))

Pre-norm, the modern default (GPT-2 onward):

    y = x + Sublayer(LayerNorm(x))

The difference is not cosmetic. In pre-norm the residual path from input to
output is an *unmodified* identity: nothing is normalised on the way through,
so a gradient flowing backwards reaches early layers without being repeatedly
rescaled. In post-norm every residual addition is followed by a LayerNorm, so
the signal is transformed at every depth; the output variance grows with depth
and training a deep stack requires a learning-rate warmup to stay stable.

We use PRE-NORM because it trains reliably without warmup, which matters here:
10 layers on a fixed compute budget with no room to debug a diverging run. We
still use a warmup in the schedule, but as insurance rather than a necessity.

Each sub-layer keeps its own LayerNorm -- they normalise different things
(the block input, and the attention-updated stream) and cannot be shared.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from attention import MultiHeadCausalSelfAttention
from ffn import FeedForward


class TransformerBlock(nn.Module):
    """Attention sub-layer then FFN sub-layer, each pre-normed and residual.

    Shapes: (B, T, d_model) in, (B, T, d_model) out -- a block never changes
    the width, which is what makes the stack composable.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int,
                 attn_dropout: float, resid_dropout: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadCausalSelfAttention(
            d_model, n_heads, max_seq_len, attn_dropout, resid_dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, resid_dropout)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        # Sub-layer 1: normalise a copy, attend, add back to the untouched stream.
        if return_attention:
            attn_out, attn_w = self.attn(self.ln_1(x), return_attention=True)
        else:
            attn_out, attn_w = self.attn(self.ln_1(x)), None
        x = x + attn_out

        # Sub-layer 2: same pattern with the feed-forward network.
        x = x + self.ffn(self.ln_2(x))

        return (x, attn_w) if return_attention else x
