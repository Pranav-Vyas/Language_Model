"""Position-wise feed-forward network (Phase 2, block sub-layer 2).

Duplicated verbatim into every language tree; never imports from a sibling.

WHY THE BLOCK NEEDS THIS AT ALL
-------------------------------
Attention output is a weighted *average* of value vectors -- a linear
recombination of things already present in the sequence. Stacking attention
alone would therefore compose linear maps (softmax aside) and could not build
genuinely new features. The FFN is where the per-token nonlinear computation
happens: it is applied identically and independently at every position, which
is what "position-wise" means.

SHAPE: WIDEN, ACTIVATE, PROJECT BACK
------------------------------------
    (B, T, d_model) -> (B, T, 4*d_model) -> GELU -> (B, T, d_model)

The inner dimension is larger than d_model (4x is the convention from the
original paper and still standard). The expansion gives the nonlinearity room:
projecting into a wider space, applying GELU there, then compressing back lets
the layer represent functions that a same-width map could not.

At d_model=384 this is 384*1536 + 1536*384 = 1,179,648 weights per block --
the FFN holds roughly twice the parameters of the attention sub-layer, which
is why total parameter count is dominated by 12*d^2 per block.

WHY GELU
--------
GELU is smooth everywhere, unlike ReLU's kink at zero, and it gates inputs by
their magnitude rather than hard-thresholding. It is the activation used by
GPT-family models and is permitted by the spec ("GELU or SwiGLU"). SwiGLU
would perform comparably but needs a third projection, spending parameters we
would rather give to depth.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Two linear layers with a GELU between them, applied per position.

    Shapes
    ------
    input  : (B, T, d_model)
    hidden : (B, T, d_ff)      d_ff > d_model
    output : (B, T, d_model)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        if d_ff <= d_model:
            raise ValueError(
                f"d_ff={d_ff} must exceed d_model={d_model}; the spec requires "
                "an inner dimension larger than the model dimension."
            )
        self.fc_in = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.fc_out = nn.Linear(d_ff, d_model)
        # Dropout on the sub-layer output, before it re-enters the residual
        # stream (the spec asks for dropout "in the appropriate places":
        # attention probabilities and sub-layer outputs).
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc_out(self.act(self.fc_in(x))))
