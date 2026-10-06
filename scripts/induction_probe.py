"""Measure in-context copying (induction) in a pretrained checkpoint (Phase 3).

Language-agnostic tooling: it loads one tree's checkpoint at a time and holds no
language data of its own.

    python scripts/induction_probe.py

WHY THIS EXISTS
---------------
Phase 3's reasoning task needs one specific ability: copy the correct entity out
of the prompt. Model L failed at it in a very particular way -- on held-out
examples 68% of its answers were TRAINING-pool names that never appeared in the
prompt at all, against 1% on training examples. That is not a reasoning error,
it is a failure to copy, so the model fell back on memorising the training names.

Copying from context is the classic induction-head behaviour, and it can be
measured directly on the PRETRAINED checkpoint, before any finetuning. The probe
feeds a random token sequence twice -- [A B C ... A B C ...] -- and asks whether
the model predicts the second copy. A model with induction heads does; one
without falls back on unigram priors.

Measuring it on the pretrained model matters: it locates the cause in Phase 2,
upstream of anything Phase 3 did, and it is what turns "Model L is worse at
reasoning" from a bare observation into an explanation.

TWO CONTROLS, BOTH NECESSARY
----------------------------
1. HYPERPARAMETERS. Model H's headline checkpoint was tuned (peak LR 3e-3,
   dropout 0.0) while Model L's was not (6e-4, dropout 0.1), so a raw H-vs-L
   comparison confounds language with recipe. The untuned Model H BASELINE is
   therefore probed too, at the identical recipe to Model L.
2. TOKEN SAMPLING. Sampling ids uniformly would favour whichever vocabulary has
   better-trained rare embeddings -- a tokenizer artifact, not a copying result.
   `--frequency-sampled` instead draws from each corpus's own token frequency
   distribution, so each model copies tokens it genuinely sees often.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent

# (label, tree, lang_code, checkpoint, recipe) -- recipe recorded so the
# hyperparameter control is visible in the output rather than implied.
CHECKPOINTS = [
    ("hindi-tuned", "hindi", "hi",
     "kaggle_tuned_out/link1/checkpoints/hindi/step_8732.pt", "3e-3 / dropout 0.0"),
    ("hindi-baseline", "hindi", "hi",
     "kaggle_out2/link1/checkpoints/hindi/step_8732.pt", "6e-4 / dropout 0.1"),
    ("nepali", "nepali", "ne",
     "kaggle_out2/link1/checkpoints/nepali/step_13547.pt", "6e-4 / dropout 0.1"),
    ("nepali-matched", "nepali", "ne",
     "kaggle_out2/link1/checkpoints/nepali/step_8732.pt", "6e-4 / dropout 0.1"),
]


def load_model(tree: str, code: str, path: Path):
    """Import the tree's OWN model code and load its checkpoint.

    Modules of the same name are evicted first: the two trees deliberately hold
    separate copies of `transformer`/`checkpoint`, and a stale import would
    silently evaluate one language's checkpoint with the other's code.
    """
    for mod in ("transformer", "checkpoint"):
        sys.modules.pop(mod, None)
    for sub in ("model", "train"):
        p = str(REPO / tree / sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    from checkpoint import load_checkpoint            # noqa: E402
    from transformer import ModelConfig, TransformerLM  # noqa: E402

    ck = load_checkpoint(path, expected_lang_code=code, map_location="cpu")
    cfg = ModelConfig(**{k: v for k, v in ck["model_cfg"].items()
                         if k in ModelConfig.__dataclass_fields__})
    model = TransformerLM(cfg)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, ck


def common_tokens(tree: str, n: int = 4000) -> list[int]:
    """The `n` most frequent real tokens in this tree's own tokenized shard."""
    shard = REPO / tree / "data" / "tokenized" / "val.bin"
    arr = np.memmap(shard, dtype=np.uint16, mode="r")[:2_000_000]
    cnt = Counter(int(t) for t in arr[::7])
    return [t for t, _ in cnt.most_common(n) if t > 3]


@torch.no_grad()
def induction_accuracy(model, vocab_pool, *, seq_len: int = 24, trials: int = 150,
                       seed: int = 1337) -> dict:
    """Fraction of second-copy tokens predicted correctly. Pure given inputs."""
    rng = random.Random(seed)
    hits = total = 0
    reciprocal_ranks = []
    for _ in range(trials):
        seq = [rng.choice(vocab_pool) for _ in range(seq_len)]
        ids = torch.tensor([seq + seq], dtype=torch.long)
        logits, _ = model(ids)
        # Only positions inside the SECOND copy: the first copy is unpredictable
        # by construction, so scoring it would just add noise.
        for i in range(seq_len, 2 * seq_len - 1):
            gold = int(ids[0, i + 1])
            row = logits[0, i]
            hits += int(row.argmax()) == gold
            reciprocal_ranks.append(1.0 / (int((row > row[gold]).sum()) + 1))
            total += 1
    return {"accuracy": hits / total, "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
            "positions_scored": total}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Induction-head probe.")
    ap.add_argument("--seq-len", type=int, default=24)
    ap.add_argument("--trials", type=int, default=150)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--uniform", action="store_true",
                    help="sample token ids uniformly instead of by corpus "
                         "frequency (kept as the sensitivity check)")
    ap.add_argument("--out", default=str(REPO / "report" / "phase3" / "induction_probe.json"))
    args = ap.parse_args(argv)

    results = []
    for label, tree, code, rel, recipe in CHECKPOINTS:
        path = REPO / rel
        if not path.exists():
            print(f"[skip] {label}: {rel} not present locally", flush=True)
            continue
        model, ck = load_model(tree, code, path)
        pool = (list(range(300, 15000)) if args.uniform else common_tokens(tree))
        m = induction_accuracy(model, pool, seq_len=args.seq_len,
                               trials=args.trials, seed=args.seed)
        row = {"label": label, "tree": tree, "recipe": recipe,
               "checkpoint": rel, "step": int(ck["step"]),
               "val_loss": ck.get("val_loss"),
               "sampling": "uniform" if args.uniform else "corpus-frequency",
               **m}
        results.append(row)
        print(f"{label:16s} [{recipe:18s}] induction {m['accuracy']:.3f}  "
              f"mrr {m['mrr']:.3f}  (val_loss {ck.get('val_loss'):.4f})", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(args), "results": results},
                              indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
