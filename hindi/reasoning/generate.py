"""Generate the synthetic reasoning dataset (T3.4).

Duplicated verbatim into every language tree; never imports from a sibling.

    python hindi/reasoning/generate.py --split train --n 60000
    python hindi/reasoning/generate.py --split test  --n 6000
    python hindi/reasoning/generate.py --split ood   --n 2000

GROUND TRUTH IS CONSTRUCTED, NEVER LABELLED
-------------------------------------------
Each example starts from a hidden total order over k entities. The premises are
*rendered from* that order and the answer is *read off* it, so the label cannot
disagree with the text. Nothing is post-hoc annotated, which is the whole point
of generating rather than downloading a benchmark.

THREE ANTI-LEAKAGE AXES
-----------------------
1. Entity names   — `entities.py` holds disjoint pools per split.
2. Relations      — `templates.HELD_OUT_RELATIONS` never appear in train.
3. Hop count      — the 4-hop pattern (P6) is generated ONLY for the `ood`
                    split, so it tests length generalisation: train on ≤2 hops,
                    evaluate on 3.

DEFEATING SURFACE HEURISTICS BY CONSTRUCTION
--------------------------------------------
A model that always answers the first-mentioned entity should score at chance,
otherwise a high accuracy would mean nothing. Three mechanisms enforce that:
  * premise order is shuffled, so chain order is not presentation order;
  * question polarity is balanced, so the same premises ask both "who is
    tallest" and "who is shortest" and the answers differ;
  * premises themselves are randomly stated with the positive adjective or with
    the negative adjective and reversed arguments ("A is taller than B" vs
    "B is shorter than A") — the same fact, two surfaces.
`first_mention` and `last_mention` are recorded on every example so the
heuristic baselines can actually be computed rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import entities as E          # noqa: E402
import templates as T         # noqa: E402

TREE = HERE.parent
REPO = TREE.parent

# Train mix. Biased toward multi-hop: the inferred:atomic ratio is the knob that
# drives generalisation speed on comparison tasks, per the grokking literature.
# P5 (a 3-hop CHAIN question) is absent from the default arm; it exists only
# for the curriculum arm. The held-out hop count is 4, carried by P6.
# v2 mix. Rebalanced toward P3, which v1 measured as the weakest GENUINE
# pattern (0.416 against 0.333 chance) and which is the assignment's own first
# example -- "given A > B and B > C, which is smallest / largest?". The budget
# comes from P1, the easiest pattern, and from P4's former free ride.
TRAIN_MIX = {"P1": 0.20, "P2": 0.28, "P3": 0.35, "P4": 0.17}
# P4 takes FOUR entities in v2, not three. With a 3-chain A > B > C the
# second-most and the second-least are both B, and B is precisely the entity
# named in both premises -- so "say the repeated word" answered every v1 P4
# example correctly (verified 960/960) with no comparison at all, and the model
# duly scored 0.972 with polarity making no difference. With four entities the
# second-most (B) and second-least (C) differ, polarity matters, and the
# repeated-entity rule no longer identifies a unique answer.
#
# Consequence, stated plainly: P4 is now a 3-hop pattern, so 3 hops enter
# TRAINING and the length-generalisation probe moves out to 4 hops (P6).
PATTERN_ENTITIES = {"P1": 2, "P2": 3, "P3": 3, "P4": 4, "P5": 4, "P6": 5}

# CURRICULUM ARM (--curriculum). NOTE: the numbers in this comment are from the
# v1 dataset, when the default mix trained on <=2 hops and probed 3. Since P4
# was widened to four entities, the DEFAULT arm trains on 1-3 hops and probes 4,
# so both arms now probe 4 hops and they differ only in whether a 3-hop *chain*
# question (P5) joins the 3-hop *ordinal* one (P4) in training.
# The v1 measurement that motivated this arm: 0.219 against a 0.250 chance floor
# learned "apply the comparison at most twice", not the recursive rule. The
# length-generalisation literature trains across a RANGE of lengths so the
# rule has to be applied a variable number of times, then extrapolates one
# step beyond. This mix puts 3-hop INTO training and moves the probe to
# 4-hop. It is a SEPARATE arm, not a replacement: the default mix stays
# exactly as it was so the original finding remains reproducible.
CURRICULUM_MIX = {"P1": 0.20, "P2": 0.30, "P3": 0.15, "P4": 0.15, "P5": 0.20}


def _render_premises(chain: list[str], rel: dict, rng: random.Random) -> list[str]:
    """Adjacent facts of the hidden order, each surfaced one of two ways.

    `chain` is ranked high-to-low, so chain[i] exceeds chain[i+1].
    """
    out = []
    for hi, lo in zip(chain, chain[1:]):
        if rng.random() < 0.5:
            out.append(T.premise(hi, lo, rel["pos"]))     # "hi is more than lo"
        else:
            out.append(T.premise(lo, hi, rel["neg"]))     # "lo is less than hi"
    rng.shuffle(out)                                       # order != chain order
    return out


def make_example(pattern: str, rel_key: str, names: list[str],
                 rng: random.Random) -> dict:
    """Build one example plus every field the evaluation needs. Pure w.r.t. rng."""
    rel = T.relation(rel_key)
    k = PATTERN_ENTITIES[pattern]
    chain = rng.sample(names, k)          # chain[0] is the greatest
    premises = _render_premises(chain, rel, rng)
    positive = rng.random() < 0.5         # question polarity
    adj = rel["pos"] if positive else rel["neg"]
    variant = rng.randrange(3)

    if pattern == "P1":
        x, y = chain[0], chain[1]
        if rng.random() < 0.5:
            x, y = y, x                   # mention order independent of rank
        q = T.question_pair(x, y, adj, variant, rng.random())
        gold = chain[0] if positive else chain[1]
        candidates, distance = [x, y], 1
    elif pattern == "P2":
        # The UNSTATED pair: no premise mentions these two together.
        x, y = chain[0], chain[-1]
        if rng.random() < 0.5:
            x, y = y, x
        q = T.question_pair(x, y, adj, variant, rng.random())
        gold = chain[0] if positive else chain[-1]
        candidates, distance = [x, y], k - 1
    elif pattern == "P3":
        q = T.question_superlative(adj, variant)
        gold = chain[0] if positive else chain[-1]
        candidates, distance = list(chain), k - 1
    elif pattern == "P4":
        # Ordinal: second-most is chain[1]; second-least is chain[-2].
        q = T.question_ordinal(adj, variant)
        gold = chain[1] if positive else chain[-2]
        candidates, distance = list(chain), 1
    elif pattern in ("P5", "P6"):
        q = T.question_superlative(adj, variant)
        gold = chain[0] if positive else chain[-1]
        candidates, distance = list(chain), k - 1
    else:
        raise ValueError(f"unknown pattern {pattern}")

    # Stored candidate order must carry NO information. `chain` is ranked
    # high-to-low, so leaving candidates in chain order would make
    # candidates[0] always the maximum -- the shortcut audit duly scored
    # "first_candidate" at ~0.5 on v1. The model never sees this list (it is
    # metrics-only), but an ordered list invites a future analysis to exploit
    # it by accident.
    rng.shuffle(candidates)

    prompt = " ".join(premises) + " " + q
    mentioned = [w for w in prompt.split() if w in set(chain)]

    return {
        "pattern": pattern,
        "relation": rel_key,
        "polarity": "pos" if positive else "neg",
        "n_entities": k,
        "n_hops": k - 1,
        "prompt": prompt,
        "answer": gold,
        "target_answer_only": T.answer(gold),
        "target_with_trace": " " + T.trace_chain(chain, rel, positive)
                             + T.answer(gold),
        # Evaluation support -------------------------------------------------
        "candidates": candidates,          # closed-set scoring
        "chain": chain,                    # hidden total order, high -> low
        "rank_distance": distance,         # symbolic distance effect
        "first_mention": mentioned[0] if mentioned else None,
        "last_mention": mentioned[-1] if mentioned else None,
    }


def build(split: str, n: int, seed: int, curriculum: bool = False) -> list[dict]:
    """Generate `n` examples for one split.

    Deterministic in (split, n, seed, curriculum). With `curriculum=True`
    training spans 1-3 hops and the ood split becomes 4-hop, so the probe
    still tests a length never trained -- one step beyond the longest seen,
    which is how the length-generalisation literature sets these up.
    """
    rng = random.Random(f"{seed}:{split}")
    names = E.pool("train" if split == "train" else
                   "val" if split == "val" else "test")

    train_rels = [k for k, *_ in T.RELATIONS if k not in T.HELD_OUT_RELATIONS]
    base_mix = CURRICULUM_MIX if curriculum else TRAIN_MIX
    # Training now reaches 3 hops via P4, so the probe is 4 hops in both arms:
    # it must always sit one hop beyond anything seen in training.
    ood_pattern = "P6"
    if split == "train":
        rels, mix = train_rels, base_mix
    elif split == "val":
        # Val mirrors train's distribution so it measures the same thing.
        rels, mix = train_rels, base_mix
    elif split == "test":
        # Test covers ALL relations, including the two never trained.
        rels, mix = [k for k, *_ in T.RELATIONS], base_mix
    elif split == "ood":
        rels, mix = [k for k, *_ in T.RELATIONS], {ood_pattern: 1.0}
    else:
        raise ValueError(split)

    patterns = list(mix)
    weights = [mix[p] for p in patterns]
    rows = []
    for i in range(n):
        pat = rng.choices(patterns, weights)[0]
        rel = rng.choice(rels)
        ex = make_example(pat, rel, names, rng)
        ex["id"] = f"{split}-{i:06d}"
        ex["held_out_relation"] = rel in T.HELD_OUT_RELATIONS
        rows.append(ex)
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Generate the reasoning dataset.")
    ap.add_argument("--split", required=True,
                    choices=["train", "val", "test", "ood"])
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--curriculum", action="store_true",
                    help="train on 1-3 hops and probe 4-hop, instead of the "
                         "default 1-2 hops probing 3-hop. Writes to a separate "
                         "directory so the default dataset is never overwritten.")
    args = ap.parse_args(argv)

    rows = build(args.split, args.n, args.seed, curriculum=args.curriculum)
    default_dir = TREE / ("reasoning_data_curriculum" if args.curriculum
                          else "reasoning_data")
    out_dir = Path(args.out_dir) if args.out_dir else default_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.split}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"[{args.split}] {len(rows):,} examples -> {out}")
    print(f"  patterns : {dict(Counter(r['pattern'] for r in rows))}")
    print(f"  polarity : {dict(Counter(r['polarity'] for r in rows))}")
    print(f"  relations: {len(set(r['relation'] for r in rows))} distinct")
    fm = sum(r["answer"] == r["first_mention"] for r in rows) / len(rows)
    lm = sum(r["answer"] == r["last_mention"] for r in rows) / len(rows)
    print(f"  first-mention baseline {fm:.3f} | last-mention baseline {lm:.3f}")
    print(f"  example: {rows[0]['prompt']}")
    print(f"           {rows[0]['target_answer_only']}")


if __name__ == "__main__":
    main()
