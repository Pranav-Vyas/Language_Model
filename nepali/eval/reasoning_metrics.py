"""Reasoning-task metrics: exact match, closed-set accuracy, heuristics (T3.7).

Duplicated verbatim into every language tree; never imports from a sibling.

The training loop (`train/finetune.py`) imports its in-training evaluation from
HERE, so the held-out accuracy printed while a model trains and the figure in
the final report are computed by literally the same function. Two
implementations of "exact match" would be two chances to disagree.

THREE NUMBERS, BECAUSE ONE WOULD BE UNINTERPRETABLE
--------------------------------------------------
1. EXACT MATCH, greedy decoding. The headline the spec asks for. Greedy because
   a reasoning answer is a closed-form span: the degeneration problem that made
   Phase 2 sample at several temperatures applies to open-ended text, not here.

2. CLOSED-SET ACCURACY. The model is handed the candidates and we ask which it
   finds most probable as the answer. This is what makes the PRETRAINED
   baseline meaningful: a pretrained model has never seen the "उत्तर:" answer
   format, so its exact match is ~0 whatever it knows, and "0 -> 0.68" would
   say nothing. Closed-set has chance at exactly 1/k and asks the real question:
   did any comparative knowledge exist before finetuning?

   It also isolates COPYING from COMPARING. Exact match requires the model to
   produce the right name itself; closed-set supplies the names. A model that
   scores far higher on closed-set than on exact match understands the relation
   but cannot reliably copy the entity out of context -- which is the specific
   failure the induction probe found in Model L.

3. HEURISTIC BASELINES -- first-mentioned, last-mentioned, chance. A score that
   merely matches "always name the last entity mentioned" is not evidence of
   reasoning, so these travel with every accuracy rather than in a footnote.

CLOSED-SET SCORING IS ONE TOKEN, NO LENGTH NORMALISATION
--------------------------------------------------------
Every candidate is a single token (enforced by tests/test_reasoning_data.py) and
the answer prefix "▁उत्तर :" tokenizes identically ahead of every candidate --
verified on all 14,124 test candidates in both languages. So scoring is just the
log-probability of one next token, and the usual length-normalisation choice,
which can quietly favour short or long candidates, does not arise.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch

ANSWER_PREFIX = " उत्तर:"


def load_rows(path: str | Path, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def answer_of(s: str) -> str:
    """Extract the entity from a generated span.

    The generation may be a bare answer (" उत्तर: राम") or a reasoning trace that
    ends in one. The entity is the FIRST word after the first colon -- taking
    everything after it would swallow any text the model keeps emitting.

    No colon means the model never produced an answer at all (a truncated trace,
    say). The whole span is returned so it cannot equal a bare entity name;
    returning its first word instead would let a cut-off trace that happens to
    begin with the right name score as correct.
    """
    s = s.strip()
    if ":" not in s:
        return s
    tail = s.split(":", 1)[1].split()
    return tail[0] if tail else ""


@torch.no_grad()
def greedy_answers(model, sp, rows, device, *, max_new: int = 12,
                   batch_size: int = 128, autocast_ctx=None) -> list[str]:
    """Greedily decode the answer span for each row. Returns decoded strings.

    RIGHT-padded, and this is not a style choice -- left-padding silently
    destroys the measurement for this architecture. Two reasons:

      * `forward` takes no attention mask, so any pad token BEFORE real text is
        attended to as if it were content.
      * positions are LEARNED ABSOLUTE, so left-padding shifts every real token
        to a position index the model was never trained to see there.

    Measured on 120 training examples where the teacher-forced answer
    probability is 1.0000: batch_size=1 (no padding) scored 1.000, batch_size=8
    scored 0.583 and batch_size=60 scored 0.517. The model was perfect; the
    padding was doing the damage.

    With right-padding the real tokens keep positions 0..n-1 and the causal mask
    guarantees position t never attends to a pad at t' > t, so a batched result
    is identical to running each row alone -- asserted in the test suite.

    Shapes: x (B, T) right-padded; `lengths` (B,) tracks each row's true length,
    and the next-token logit is read at index length-1 rather than at -1.
    """
    model.eval()
    eos, pad = sp.eos_id(), sp.pad_id()
    out: list[str] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        enc = [sp.encode(r["prompt"]) for r in chunk]
        B = len(enc)
        width = max(len(e) for e in enc) + max_new
        width = min(width, model.cfg.max_seq_len)
        x = torch.full((B, width), pad, dtype=torch.long)
        lengths = torch.tensor([len(e) for e in enc], dtype=torch.long)
        for i, e in enumerate(enc):
            x[i, :len(e)] = torch.tensor(e, dtype=torch.long)
        x = x.to(device)
        lengths = lengths.to(device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        gen = [[] for _ in enc]
        for _ in range(max_new):
            cur = int(lengths.max())
            if cur >= width:
                break
            if autocast_ctx is not None:
                with autocast_ctx():
                    logits, _ = model(x[:, :cur])
            else:
                logits, _ = model(x[:, :cur])
            # each row's own last real position, not a shared -1
            idx = (lengths - 1).clamp(min=0)
            nxt = logits[torch.arange(B, device=device), idx].argmax(dim=-1)
            for i in range(B):
                if done[i]:
                    continue
                t = int(nxt[i])
                if t == eos:
                    done[i] = True
                else:
                    gen[i].append(t)
                    pos = int(lengths[i])
                    if pos < width:
                        x[i, pos] = t
                        lengths[i] = pos + 1
            if bool(done.all()):
                break
        out.extend(sp.decode(g) for g in gen)
    return out


@torch.no_grad()
def closed_set_predictions(model, sp, rows, device, *, batch_size: int = 128,
                           autocast_ctx=None) -> list[str]:
    """For each row, the candidate the model rates most probable as the answer.

    Scores log P(candidate | prompt + "▁उत्तर :"). Every candidate is one token
    and shares that prefix, so one forward pass per row scores all candidates:
    read the next-token distribution at the row's last real position and index
    it with each candidate id.

    RIGHT-padded for the same reason as `greedy_answers` -- see that docstring.
    Left-padding corrupts attention and shifts learned absolute positions.
    """
    model.eval()
    pad = sp.pad_id()
    prefix = sp.encode(ANSWER_PREFIX)
    preds: list[str] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        enc = [sp.encode(r["prompt"]) + prefix for r in chunk]
        B = len(enc)
        width = min(max(len(e) for e in enc), model.cfg.max_seq_len)
        x = torch.full((B, width), pad, dtype=torch.long)
        for i, e in enumerate(enc):
            e = e[:width]
            x[i, :len(e)] = torch.tensor(e, dtype=torch.long)
        x = x.to(device)
        idx = torch.tensor([min(len(e), width) - 1 for e in enc], device=device)
        if autocast_ctx is not None:
            with autocast_ctx():
                logits, _ = model(x)
        else:
            logits, _ = model(x)
        last = logits[torch.arange(B, device=device), idx]          # (B, V)
        logp = torch.log_softmax(last.float(), dim=-1)
        for i, r in enumerate(chunk):
            cand_ids = []
            for c in r["candidates"]:
                ids = sp.encode(c)
                if len(ids) != 1:
                    raise ValueError(f"candidate {c!r} is {len(ids)} tokens; "
                                     "closed-set scoring assumes single-token names")
                cand_ids.append(ids[0])
            preds.append(r["candidates"][int(logp[i, cand_ids].argmax())])
    return preds


def heuristics(rows: list[dict]) -> dict:
    """Surface-heuristic accuracies. A real result must clear all of these."""
    n = len(rows)
    return {
        "chance": sum(1 / len(r["candidates"]) for r in rows) / n,
        "first_mention": sum(r["answer"] == r["first_mention"] for r in rows) / n,
        "last_mention": sum(r["answer"] == r["last_mention"] for r in rows) / n,
    }


def breakdown(rows: list[dict], correct: list[bool], key: str) -> dict:
    """Accuracy grouped by one row field (pattern, relation, polarity, ...)."""
    tot, hit = defaultdict(int), defaultdict(int)
    for r, ok in zip(rows, correct):
        k = str(r[key])
        tot[k] += 1
        hit[k] += int(ok)
    return {k: {"n": tot[k], "accuracy": hit[k] / tot[k]} for k in sorted(tot)}


def exact_match(model, sp, rows, device, **kw) -> dict:
    """Exact-match accuracy plus heuristics. Kept for the training loop."""
    preds = greedy_answers(model, sp, rows, device, **kw)
    n = len(rows)
    hit = sum(1 for p, r in zip(preds, rows) if answer_of(p) == r["answer"])
    return {"n": n, "exact_match": hit / n, **heuristics(rows),
            "well_formed": sum(1 for p, r in zip(preds, rows)
                               if answer_of(p) in r["candidates"]) / n}


def evaluate(model, sp, rows, device, *, max_new: int = 12, batch_size: int = 128,
             autocast_ctx=None) -> dict:
    """Full evaluation of one model on one split: EM, closed-set, heuristics,
    and per-pattern / per-relation breakdowns of both accuracies."""
    gen = greedy_answers(model, sp, rows, device, max_new=max_new,
                         batch_size=batch_size, autocast_ctx=autocast_ctx)
    em_pred = [answer_of(g) for g in gen]
    em_ok = [p == r["answer"] for p, r in zip(em_pred, rows)]
    cs_pred = closed_set_predictions(model, sp, rows, device, batch_size=batch_size,
                                     autocast_ctx=autocast_ctx)
    cs_ok = [p == r["answer"] for p, r in zip(cs_pred, rows)]
    n = len(rows)
    return {
        "n": n,
        "exact_match": sum(em_ok) / n,
        "closed_set": sum(cs_ok) / n,
        "well_formed": sum(1 for p, r in zip(em_pred, rows) if p in r["candidates"]) / n,
        **heuristics(rows),
        "by_pattern": {"exact_match": breakdown(rows, em_ok, "pattern"),
                       "closed_set": breakdown(rows, cs_ok, "pattern")},
        "by_held_out_relation": {"exact_match": breakdown(rows, em_ok, "held_out_relation"),
                                 "closed_set": breakdown(rows, cs_ok, "held_out_relation")},
        "by_rank_distance": {"exact_match": breakdown(rows, em_ok, "rank_distance"),
                             "closed_set": breakdown(rows, cs_ok, "rank_distance")},
        # Polarity separates "understands the relation" from "can invert it":
        # a negative question ("who is SHORTER?") asks the model to flip a
        # premise stated with the positive adjective.
        "by_polarity": {"exact_match": breakdown(rows, em_ok, "polarity"),
                        "closed_set": breakdown(rows, cs_ok, "polarity")},
        "samples": [{"prompt": r["prompt"], "gold": r["answer"], "em_pred": e,
                     "cs_pred": c, "pattern": r["pattern"]}
                    for r, e, c in list(zip(rows, em_pred, cs_pred))[:40]],
    }
