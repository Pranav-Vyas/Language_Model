"""Autoregressive sampling: greedy, temperature, and top-p (T2.13).

Duplicated verbatim into every language tree; never imports from a sibling.

EVALUATION PROTOCOL
-------------------
The model is an unconditional LM, so "generation quality" needs a reference to
score against. We use held-out documents the model has never seen:

    prompt     = first `prompt_tokens` tokens of a test document
    reference  = the next `max_new_tokens` tokens of that SAME document
    hypothesis = what the model generates from the prompt

That gives every generation exactly one reference continuation, which is what
BLEU / chrF / ROUGE-L need. Its limitation is discussed at length in
gen_metrics.py: one reference badly under-counts the space of valid
continuations, which is precisely why these metrics are weak here.

NO KV CACHE, ON PURPOSE
-----------------------
Each step re-runs the full forward pass over the sequence so far, which is
O(n^2) across a generation instead of O(n). A KV cache would be faster but
adds a second code path whose correctness would have to be proven separately
against this one; at evaluation scale (hundreds of short continuations) the
cost is seconds. Simplicity that can be explained in the viva beats speed we
do not need.

CONTEXT CROPPING
----------------
Learned absolute positions have no row beyond `max_seq_len`, so once the
sequence reaches that length the input is cropped to its last `max_seq_len`
tokens. This is a hard property of the positional scheme, not a tunable.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus filtering: keep the smallest set of tokens with cumulative
    probability >= top_p, mask the rest to -inf.

    Included as a DIAGNOSTIC, not because the spec asks for it. The course
    notes (Holtzman et al.) measure greedy decoding at 73.66% repetition
    against 0.28% for human text, and nucleus sampling at 0.36%. The spec
    mandates greedy and temperature 0.5/1.0/1.5, so those are what we report --
    top-p is run alongside so the repetition numbers we are about to see can be
    attributed to the decoding strategy rather than to the model.

    Shapes: logits (B, V) -> (B, V)
    """
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative = probs.cumsum(dim=-1)
    # Keep everything up to and including the token that crosses top_p.
    remove = cumulative - probs > top_p
    sorted_logits[remove] = float("-inf")
    out = torch.full_like(logits, float("-inf"))
    return out.scatter(-1, sorted_idx, sorted_logits)


@torch.no_grad()
def generate(model, prompt_ids: torch.Tensor, max_new_tokens: int, *,
             temperature: float = 1.0, greedy: bool = False,
             top_p: float | None = None, eos_id: int | None = None,
             max_seq_len: int = 512) -> torch.Tensor:
    """Continue each row of `prompt_ids` by up to `max_new_tokens`.

    Shapes
    ------
    prompt_ids : (B, P)  int64
    returns    : (B, P + N) int64, N <= max_new_tokens

    `greedy=True` takes the argmax and ignores temperature, rather than
    emulating greedy with a tiny temperature -- a near-zero temperature
    overflows softmax and is not the same operation.
    """
    model.eval()
    ids = prompt_ids.clone()
    finished = torch.zeros(ids.size(0), dtype=torch.bool, device=ids.device)

    for _ in range(max_new_tokens):
        # Crop to the positional table's hard bound (see module docstring).
        window = ids[:, -max_seq_len:]
        logits, _ = model(window)
        next_logits = logits[:, -1, :]                      # (B, V)

        if greedy:
            next_id = next_logits.argmax(dim=-1, keepdim=True)
        else:
            if temperature <= 0:
                raise ValueError("temperature must be > 0; use greedy=True instead")
            next_logits = next_logits / temperature
            if top_p is not None:
                next_logits = top_p_filter(next_logits, top_p)
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)

        # A finished row keeps emitting EOS so the batch stays rectangular;
        # the trailing EOS run is stripped by the caller.
        if eos_id is not None:
            next_id = torch.where(finished.unsqueeze(1),
                                  torch.full_like(next_id, eos_id), next_id)
            finished = finished | (next_id.squeeze(1) == eos_id)
        ids = torch.cat([ids, next_id], dim=1)
        if eos_id is not None and bool(finished.all()):
            break
    return ids


def strip_after_eos(ids: list[int], eos_id: int) -> list[int]:
    """Truncate at the first EOS; a continuation ends where the model says it does."""
    return ids[:ids.index(eos_id)] if eos_id in ids else ids
