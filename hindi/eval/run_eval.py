"""Run the full Phase-2 evaluation for this tree and write every artifact.

Duplicated verbatim into every language tree; never imports from a sibling.

    python <lang>/eval/run_eval.py --checkpoint <ckpt.pt>

Produces, under report/phase2/<lang>/:
    lm_metrics.json        loss / PPL / BPB on val and test        (T2.12)
    generation.json        per-setting scores + sample outputs     (T2.13)
    generation_samples.md  readable generations for the report     (T2.13)
    attention.json         per-head + per-layer statistics         (T2.14)
    *.png                  loss curves, heatmaps, head grids       (T2.15)

PROMPTS START AT DOCUMENT BOUNDARIES
------------------------------------
Prompts are taken from the *test* shard immediately after an EOS token, so a
prompt begins where a real document begins rather than in the middle of an
unrelated one. A mid-document prompt would ask the model to continue text
whose beginning it never saw, and the reference continuation would then be
unfairly hard to match.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from attention_analysis import (analyse, content_dependence,           # noqa: E402
                                summarise_by_layer)
from common import (TREE, autocast_for, load_model_from_checkpoint,   # noqa: E402
                    load_tokenizer, pick_device, seed_everything,
                    shard_path, write_json)
from generate import generate, strip_after_eos                        # noqa: E402
from gen_metrics import score_generations                             # noqa: E402
from lm_metrics import evaluate_split                                 # noqa: E402
import viz                                                            # noqa: E402

# Spec-mandated settings, plus top-p as a diagnostic (see generate.py).
DECODING = [
    ("greedy",       dict(greedy=True)),
    ("temp_0.5",     dict(temperature=0.5)),
    ("temp_1.0",     dict(temperature=1.0)),
    ("temp_1.5",     dict(temperature=1.5)),
    ("top_p_0.95",   dict(temperature=1.0, top_p=0.95)),
]


def build_prompts(sp, shard: Path, n: int, prompt_tokens: int, ref_tokens: int,
                  seed: int = 1337) -> tuple[list[list[int]], list[str]]:
    """Prompt/reference pairs beginning at document starts in the test shard."""
    toks = np.memmap(shard, dtype=np.uint16, mode="r")
    eos = sp.eos_id()
    need = prompt_tokens + ref_tokens
    # Scan a bounded prefix; the test split has tens of thousands of documents,
    # so we never need to walk the whole shard to find n starts.
    limit = min(len(toks), 4_000_000)
    window = np.asarray(toks[:limit])
    starts = (np.flatnonzero(window == eos) + 1)
    starts = starts[starts + need < limit]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(starts, size=min(n, len(starts)), replace=False)
    chosen.sort()

    prompts, refs = [], []
    for s in chosen:
        seg = window[s:s + need].astype(np.int64)
        if eos in seg[:prompt_tokens]:          # document too short to prompt from
            continue
        p = seg[:prompt_tokens].tolist()
        r = strip_after_eos(seg[prompt_tokens:].tolist(), eos)
        if len(r) < 8:                          # reference too short to score
            continue
        prompts.append(p)
        refs.append(sp.decode(r))
    return prompts, refs


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Phase-2 evaluation suite.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-windows", type=int, default=None,
                    help="cap LM-metric windows (smoke); omit to score fully")
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--gen-batch", type=int, default=25)
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["lm", "gen", "attn", "plots"])
    args = ap.parse_args(argv)

    # Check metric dependencies BEFORE any compute. The first Kaggle run spent
    # ~7 minutes generating 197 continuations per model and only then hit
    # `ModuleNotFoundError: sacrebleu` while scoring them. A dependency that is
    # missing should cost a second, not a generation pass.
    if "gen" not in args.skip:
        try:
            import sacrebleu  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                f"generation scoring needs sacrebleu ({exc}). Install it, or "
                "pass --skip gen to run the other stages.") from exc

    seed_everything()
    device = pick_device(prefer_cuda=not args.cpu)
    model, cfg, meta = load_model_from_checkpoint(args.checkpoint, device)
    sp = load_tokenizer(cfg.vocab_size)
    actx = autocast_for(device)
    lang = meta["lang_code"]
    out = Path(args.out_dir) if args.out_dir else Path("report/phase2") / TREE.name
    out.mkdir(parents=True, exist_ok=True)

    print(f"=== {TREE.name} ({lang}) · {Path(args.checkpoint).name} · step "
          f"{meta['step']:,} · {model.num_parameters():,} params · {device} ===")

    # ---------------- T2.12 : loss / PPL / BPB --------------------------
    if "lm" not in args.skip:
        rows = []
        for split in ("val", "test"):
            p = shard_path(split)
            if not p.exists():
                continue
            r = evaluate_split(model, sp, p, cfg.max_seq_len, device, actx,
                               batch_size=args.batch_size,
                               max_windows=args.max_windows, split=split,
                               checkpoint=Path(args.checkpoint).name)
            r.lang_code = lang
            rows.append(r)
            print(f"  [lm] {split:5s} loss {r.mean_nats_per_token:.4f}  "
                  f"ppl {r.perplexity:8.2f}  BPB {r.bits_per_byte:.4f}  "
                  f"({r.tokens_scored:,} tokens, {r.bytes_utf8:,} bytes)")
        write_json({"lang_code": lang, "checkpoint": meta,
                    "note": ("PPL is per-token and NOT comparable across "
                             "languages; BPB is the cross-language comparator."),
                    "results": [r.to_dict() for r in rows]},
                   out / "lm_metrics.json")

    # ---------------- T2.13 : generation --------------------------------
    if "gen" not in args.skip:
        prompts, refs = build_prompts(sp, shard_path("test"), args.n_prompts,
                                      args.prompt_tokens, args.new_tokens)
        print(f"  [gen] {len(prompts)} prompt/reference pairs "
              f"({args.prompt_tokens} prompt tokens, up to {args.new_tokens} new)")
        gen_rows, samples = [], {}
        for name, kw in DECODING:
            hyps = []
            for b0 in range(0, len(prompts), args.gen_batch):
                chunk = prompts[b0:b0 + args.gen_batch]
                pid = torch.tensor(chunk, dtype=torch.long, device=device)
                with actx():
                    outk = generate(model, pid, args.new_tokens,
                                    eos_id=sp.eos_id(),
                                    max_seq_len=cfg.max_seq_len, **kw)
                for row in outk[:, pid.size(1):].tolist():
                    hyps.append(sp.decode(strip_after_eos(row, sp.eos_id())))
            prompt_texts = [sp.decode(p) for p in prompts[:len(hyps)]]
            m = score_generations(hyps, refs[:len(hyps)], name,
                                  prompts=prompt_texts)
            gen_rows.append(m.to_dict())
            samples[name] = [{"prompt": prompt_texts[i],
                              "generated": hyps[i], "reference": refs[i]}
                             for i in range(min(10, len(hyps)))]
            # Every hypothesis is kept, so qualitative analysis can be redone
            # later without re-running generation on a GPU.
            (out / f"hypotheses_{name}.json").write_text(
                json.dumps({"setting": name, "hypotheses": hyps,
                            "references": refs[:len(hyps)],
                            "prompts": prompt_texts}, ensure_ascii=False, indent=1),
                encoding="utf-8")
            print(f"  [gen] {name:12s} BLEU-4 {m.bleu4:6.2f}  chrF {m.chrf:5.2f}  "
                  f"chrF++ {m.chrf_pp:5.2f}  ROUGE-L {m.rouge_l_f:.4f}  "
                  f"rep4 {m.repetition_4gram:.3f}  D-1 {m.distinct_1:.3f}  "
                  f"D-2 {m.distinct_2:.3f}")
        write_json({"lang_code": lang, "checkpoint": meta,
                    "protocol": {"prompt_tokens": args.prompt_tokens,
                                 "max_new_tokens": args.new_tokens,
                                 "n_prompts": len(prompts),
                                 "source": "test split, prompts start at document boundaries"},
                    "results": gen_rows, "samples": samples},
                   out / "generation.json")

        lines = [f"# {TREE.name} — generation samples\n",
                 f"Checkpoint `{Path(args.checkpoint).name}` (step {meta['step']:,}). "
                 f"Prompt = first {args.prompt_tokens} tokens of a held-out test "
                 f"document; reference = its true continuation.\n"]
        for name, rows_ in samples.items():
            lines.append(f"\n## {name}\n")
            for i, s in enumerate(rows_, 1):
                lines.append(f"**Sample {i} — prompt**\n\n> {s['prompt']}\n")
                lines.append(f"**Generated**\n\n> {s['generated']}\n")
                lines.append(f"**Reference (true continuation)**\n\n> {s['reference']}\n")
        (out / "generation_samples.md").write_text("\n".join(lines), encoding="utf-8")

    # ---------------- T2.14 : attention ---------------------------------
    if "attn" not in args.skip:
        probes, _ = build_prompts(sp, shard_path("test"), 8, 64, 8, seed=7)
        ids = torch.tensor(probes, dtype=torch.long, device=device)
        stats, attns = analyse(model, ids)
        summary = summarise_by_layer(stats)
        # Content-dependence separates positional heads from content-based ones
        # (see attention_analysis.content_dependence).
        cdep = content_dependence(model, ids)
        per_head = []
        for st in stats:
            d = st.to_dict()
            d["content_dependence"] = cdep.get((st.layer, st.head))
            per_head.append(d)
        # The spec requires heatmaps to "include example sentences in the
        # model's language", so the probe text is stored alongside the numbers
        # rather than living only as tick labels inside a PNG.
        probe_texts = [sp.decode(pr) for pr in probes]
        write_json({"lang_code": lang, "checkpoint": meta,
                    "n_probe_sequences": ids.size(0), "seq_len": ids.size(1),
                    "probe_texts": probe_texts,
                    "probe_pieces": [[sp.id_to_piece(i) for i in pr] for pr in probes],
                    "per_head": per_head,
                    "per_layer": {str(k): v for k, v in summary.items()}},
                   out / "attention.json")
        # Readable companion so the sentences are visible without opening JSON.
        pl = [f"# {TREE.name} — attention probe sentences\n",
              f"Checkpoint `{Path(args.checkpoint).name}`, {len(probe_texts)} held-out "
              f"test passages of {ids.size(1)} tokens each. Heatmaps in this "
              f"directory visualise probe 1.\n"]
        for i, t in enumerate(probe_texts, 1):
            pl.append(f"\n**Probe {i}**\n\n> {t}\n")
        (out / "attention_probes.md").write_text("\n".join(pl), encoding="utf-8")
        early, late = 0, cfg.n_layers - 1
        print(f"  [attn] layer {early}: ent_norm "
              f"{summary[early]['entropy_normalised']:.3f}  dist "
              f"{summary[early]['mean_distance']:.1f}  prev "
              f"{summary[early]['prev_token_frac']:.3f}")
        print(f"  [attn] layer {late}: ent_norm "
              f"{summary[late]['entropy_normalised']:.3f}  dist "
              f"{summary[late]['mean_distance']:.1f}  prev "
              f"{summary[late]['prev_token_frac']:.3f}")

        if "plots" not in args.skip:
            toks = [sp.id_to_piece(i) for i in probes[0]]
            for layer in (early, late):
                a = attns[layer][0].float().cpu().numpy()      # (h, T, T)
                viz.plot_attention_heatmap(
                    a, layer, list(range(min(3, a.shape[0]))), toks, lang,
                    out / f"attention_layer{layer}.png")
            sd = per_head
            viz.plot_head_grid(sd, "entropy_normalised", "normalised entropy",
                               f"{lang}: attention entropy per head",
                               out / "attention_entropy_grid.png")
            viz.plot_head_grid(sd, "mean_distance", "mean distance (tokens)",
                               f"{lang}: mean attention distance per head",
                               out / "attention_distance_grid.png")
            viz.plot_layer_profile({int(k): v for k, v in summary.items()}, lang,
                                   out / "attention_layer_profile.png")
            viz.plot_head_grid(sd, "content_dependence",
                               "mean |Δ attention| under token shuffle",
                               f"{lang}: content-dependence per head "
                               f"(0 = purely positional)",
                               out / "attention_content_grid.png")

    # ---------------- T2.15 : loss curves -------------------------------
    if "plots" not in args.skip:
        hist_p = Path("report/phase2/training") / f"{TREE.name}_train_history.jsonl"
        val_p = Path("report/phase2/training") / f"{TREE.name}_val_curve.json"
        # Only plot the curve into a directory that IS this tree's run. A
        # milestone checkpoint (e.g. nepali at Hindi's token budget) is a point
        # on the parent run, not a run of its own -- writing the parent's full
        # curve there produced a byte-identical file implying a 13,547-step
        # run for a checkpoint that stops at 8,732.
        own_run = out.name == TREE.name
        if own_run and hist_p.exists() and val_p.exists():
            hist = [json.loads(l) for l in hist_p.read_text().splitlines() if l.strip()]
            valc = json.loads(val_p.read_text())
            viz.plot_loss_curves(hist, valc, lang, out / "loss_curves.png")
        gp = out / "generation.json"
        if gp.exists():
            rows = json.loads(gp.read_text())["results"]
            viz.plot_decoding_comparison(rows, "repetition_4gram",
                                         "4-gram repetition rate",
                                         f"{lang}: repetition by decoding setting",
                                         out / "generation_repetition.png")
            viz.plot_decoding_comparison(rows, "distinct_2", "Distinct-2",
                                         f"{lang}: Distinct-2 by decoding setting",
                                         out / "generation_distinct2.png")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
