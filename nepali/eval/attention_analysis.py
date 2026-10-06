"""Attention analysis: capture, entropy, distance (T2.14).

Named `attention_analysis` rather than `attention` on purpose: this tree's
`model/attention.py` is on the same sys.path, and two modules called
`attention` resolve to whichever directory was inserted first -- which failed
as `ImportError: cannot import name MultiHeadCausalSelfAttention`. A distinct
name removes the ambiguity rather than depending on path ordering.

Duplicated verbatim into every language tree; never imports from a sibling.

Computation only -- every function here returns numbers. Heatmaps are drawn in
`viz.py`, so a metric can be tested without rendering anything.

This module is reused UNCHANGED in Phase 3 to compare pretrained against
finetuned checkpoints, which is why it is a module with a stable return shape
rather than notebook cells.

WHAT IS MEASURED, AND WHY EACH ONE
----------------------------------
For every layer L and head h we capture the post-softmax attention matrix
`A[L][h]` of shape (T, T), where `A[i, j]` is how much query position i
attends to key position j. Causal masking means `A[i, j] = 0` for j > i, and
each row sums to 1.

  * **Entropy** `H_i = -sum_j A[i,j] log A[i,j]`, averaged over queries.
    Low entropy = the head commits to a few positions (a "sharp" or
    positional head); high entropy = it spreads attention broadly (an
    averaging head). Raw entropy is confounded by position -- query i can only
    attend to i+1 keys, so early rows are forced to be low-entropy. We
    therefore also report entropy NORMALISED by log(i+1), the entropy of a
    uniform distribution over the allowed keys, which makes heads comparable
    across positions and across sequence lengths.

  * **Mean attention distance** `sum_j A[i,j] * (i - j)`, averaged over
    queries. Small values mean the head looks at recent tokens (local,
    n-gram-like); large values mean it reaches far back. This is the direct
    evidence for the spec's "local vs long-range" discussion.

  * **Previous-token fraction** `A[i, i-1]`: isolates the induction-style
    heads that simply copy from one step back, which mean distance alone can
    hide (a head that mixes "previous token" with "far back" can average to a
    middling distance while doing neither).

  * **Attention to position 0**: transformer LMs commonly develop an
    "attention sink" on the first token, dumping probability mass there when a
    head has nothing to attend to. It is reported separately so it does not
    silently inflate the mean-distance statistic.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch


@dataclass
class HeadStats:
    """Per-head attention statistics on one probe input."""

    layer: int
    head: int
    entropy_nats: float
    entropy_normalised: float
    mean_distance: float
    prev_token_frac: float
    first_token_frac: float

    def to_dict(self) -> dict:
        return asdict(self)


@torch.no_grad()
def capture_attention(model, ids: torch.Tensor) -> list[torch.Tensor]:
    """Run the model and return per-layer attention.

    ids     : (B, T)
    returns : list of `n_layers` tensors, each (B, heads, T, T)

    The model returns PRE-dropout softmax weights: dropout is a training-time
    perturbation, not part of the learned attention pattern, and the model is
    in eval() here anyway.
    """
    model.eval()
    _, _, attentions = model(ids, return_attention=True)
    return attentions


def head_statistics(attn: torch.Tensor, layer: int,
                    eps: float = 1e-12) -> list[HeadStats]:
    """Reduce one layer's attention to per-head numbers.

    attn : (B, heads, T, T) -- averaged over the batch, since these are
           corpus-level properties of a head rather than of one sentence.
    """
    if attn.dim() != 4:
        raise ValueError(f"expected (B, h, T, T), got {tuple(attn.shape)}")
    a = attn.float().mean(dim=0)                        # (h, T, T)
    h_count, T, _ = a.shape

    # Row entropy. 0*log0 is 0, so eps only guards the log.
    ent = -(a * (a + eps).log()).sum(dim=-1)            # (h, T)

    # Uniform-over-allowed-keys entropy for each query row: query i may attend
    # to i+1 keys, so the maximum possible entropy is log(i+1).
    allowed = torch.arange(1, T + 1, device=a.device, dtype=a.dtype)
    max_ent = allowed.log().clamp(min=eps)              # (T,)

    # Distance: query i attending key j is a lookback of (i - j).
    idx = torch.arange(T, device=a.device, dtype=a.dtype)
    dist = idx.view(-1, 1) - idx.view(1, -1)            # (T, T), >=0 where allowed
    mean_dist = (a * dist.clamp(min=0)).sum(dim=-1)     # (h, T)

    prev = torch.zeros(h_count, T, device=a.device, dtype=a.dtype)
    if T > 1:
        prev[:, 1:] = a[:, torch.arange(1, T), torch.arange(0, T - 1)]
    first = a[:, :, 0]                                  # (h, T)

    # Row 0 is excluded from every average: it can only attend to itself, so
    # its entropy is 0 and its distance is 0 by construction, not by learning.
    sl = slice(1, T) if T > 1 else slice(0, T)
    out = []
    for head in range(h_count):
        out.append(HeadStats(
            layer=layer,
            head=head,
            entropy_nats=float(ent[head, sl].mean()),
            entropy_normalised=float((ent[head, sl] / max_ent[sl]).mean()),
            mean_distance=float(mean_dist[head, sl].mean()),
            prev_token_frac=float(prev[head, sl].mean()),
            first_token_frac=float(first[head, sl].mean()),
        ))
    return out


@torch.no_grad()
def content_dependence(model, ids: torch.Tensor, n_shuffles: int = 3,
                       seed: int = 1337) -> dict[tuple[int, int], float]:
    """How much does a head's attention depend on WHICH tokens are present?

    The spec asks which heads look local/positional and which look
    content-based. Reading that off a heatmap is guesswork, so it is measured:
    re-run the model with the token identities randomly permuted *within the
    sequence* and compare each head's attention matrix to the original.

    Permuting rather than replacing keeps the length, the position indices and
    the token distribution identical, so the ONLY thing that changes is which
    token sits where. A purely positional head (e.g. "always attend one step
    back") is unaffected and scores ~0. A content-based head re-routes its
    attention and scores high.

    Returns {(layer, head): mean absolute change in attention weight}, averaged
    over shuffles and over allowed (lower-triangular) entries only -- masked
    entries are 0 in both runs and would otherwise dilute the score.
    """
    base = capture_attention(model, ids)
    T = ids.size(1)
    tril = torch.tril(torch.ones(T, T, dtype=torch.bool, device=ids.device))
    n_allowed = int(tril.sum())

    g = torch.Generator(device="cpu").manual_seed(seed)
    totals: dict[tuple[int, int], float] = {}
    for _ in range(n_shuffles):
        perm = torch.stack([ids[b][torch.randperm(T, generator=g)]
                            for b in range(ids.size(0))]).to(ids.device)
        shuf = capture_attention(model, perm)
        for layer, (a0, a1) in enumerate(zip(base, shuf)):
            d = (a0.float() - a1.float()).abs()          # (B, h, T, T)
            d = (d * tril).sum(dim=(-2, -1)) / n_allowed  # (B, h)
            d = d.mean(dim=0)                             # (h,)
            for head in range(d.numel()):
                totals[(layer, head)] = totals.get((layer, head), 0.0) + float(d[head])
    return {k: v / n_shuffles for k, v in totals.items()}


def analyse(model, ids: torch.Tensor) -> tuple[list[HeadStats], list[torch.Tensor]]:
    """Capture attention and reduce it. Returns (stats, raw attention)."""
    attns = capture_attention(model, ids)
    stats: list[HeadStats] = []
    for layer, a in enumerate(attns):
        stats.extend(head_statistics(a, layer))
    return stats, attns


def summarise_by_layer(stats: list[HeadStats]) -> dict[int, dict]:
    """Layer-level means, for the early-vs-late contrast the spec asks about."""
    by: dict[int, list[HeadStats]] = {}
    for s in stats:
        by.setdefault(s.layer, []).append(s)
    return {
        layer: {
            "n_heads": len(rows),
            "entropy_nats": sum(r.entropy_nats for r in rows) / len(rows),
            "entropy_normalised": sum(r.entropy_normalised for r in rows) / len(rows),
            "mean_distance": sum(r.mean_distance for r in rows) / len(rows),
            "prev_token_frac": sum(r.prev_token_frac for r in rows) / len(rows),
            "first_token_frac": sum(r.first_token_frac for r in rows) / len(rows),
        }
        for layer, rows in sorted(by.items())
    }
