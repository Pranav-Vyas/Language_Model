"""Pretrained-vs-finetuned reasoning evaluation for this tree (T3.7).

Duplicated verbatim into every language tree; never imports from a sibling.

    python hindi/eval/run_reasoning_eval.py \
        --pretrained kaggle_tuned_out/link1/checkpoints/hindi/step_8732.pt \
        --finetuned "kaggle_fin_out/sft/hindi_*/step_*.pt"

Every model is scored with the SAME rows, the SAME decoding and the SAME
closed-set procedure, so the pretrained-vs-finetuned difference is attributable
to the finetune and nothing else. Results go to
report/phase3/<tree>/reasoning_eval.json.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics as st
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
TREE = HERE.parent
REPO = TREE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(TREE / "model"))
sys.path.insert(0, str(TREE / "train"))

from checkpoint import load_checkpoint                    # noqa: E402
from reasoning_metrics import evaluate, load_rows         # noqa: E402
from transformer import ModelConfig, TransformerLM        # noqa: E402

LANG = {"hindi": "hi", "nepali": "ne"}[TREE.name]


def load_model(path: str | Path, device):
    """Rebuild a model from its checkpoint's OWN stored config."""
    ck = load_checkpoint(path, expected_lang_code=LANG, map_location=device)
    cfg = ModelConfig(**{k: v for k, v in ck["model_cfg"].items()
                         if k in ModelConfig.__dataclass_fields__})
    model = TransformerLM(cfg).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, ck


def summarise(results: list[dict]) -> dict:
    """Mean and half-range across seeds for the headline accuracies."""
    out = {}
    for key in ("exact_match", "closed_set", "well_formed"):
        v = [r[key] for r in results]
        out[key] = {"mean": st.mean(v), "half_range": (max(v) - min(v)) / 2,
                    "per_seed": v}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Pretrained vs finetuned reasoning eval.")
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--finetuned", required=True, help="glob for finetuned checkpoints")
    ap.add_argument("--data-dir", default=str(TREE / "reasoning_data"))
    ap.add_argument("--splits", default="val,test,ood")
    ap.add_argument("--n", type=int, default=None,
                    help="cap rows per split (default: the full split)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(
        model_file=str(TREE / "tokenizer" / f"{LANG}_bpe_16000.model"))
    device = torch.device("cpu")
    torch.manual_seed(1337)

    splits = [s for s in args.splits.split(",") if s]
    rows = {s: load_rows(Path(args.data_dir) / f"{s}.jsonl", limit=args.n) for s in splits}
    print(f"[data] { {s: len(r) for s, r in rows.items()} }", flush=True)

    ft_paths = sorted(glob.glob(args.finetuned))
    if not ft_paths:
        raise SystemExit(f"no finetuned checkpoints match {args.finetuned}")

    models = [("pretrained", args.pretrained)] + [
        (f"finetuned/{Path(p).parent.name}", p) for p in ft_paths]

    report = {"lang_code": LANG, "splits": {s: len(r) for s, r in rows.items()},
              "models": {}}
    for label, path in models:
        t0 = time.time()
        model, ck = load_model(path, device)
        per_split = {}
        for s in splits:
            per_split[s] = evaluate(model, sp, rows[s], device, batch_size=args.batch_size)
            r = per_split[s]
            print(f"  {label:44s} {s:4s} EM {r['exact_match']:.3f}  "
                  f"closed-set {r['closed_set']:.3f}  chance {r['chance']:.3f}  "
                  f"last-mention {r['last_mention']:.3f}  wf {r['well_formed']:.3f}",
                  flush=True)
        report["models"][label] = {"checkpoint": str(path), "step": int(ck["step"]),
                                   "stage": ck.get("train_cfg", {}).get("stage", "pretrain"),
                                   "results": per_split}
        print(f"    ({time.time() - t0:.0f}s)", flush=True)

    # Seed aggregate for the finetuned models, per split.
    ft = {k: v for k, v in report["models"].items() if k.startswith("finetuned/")}
    report["finetuned_aggregate"] = {
        s: summarise([v["results"][s] for v in ft.values()]) for s in splits}

    out = Path(args.out) if args.out else (
        REPO / "report" / "phase3" / TREE.name / "reasoning_eval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
