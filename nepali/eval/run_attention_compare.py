"""Phase-3 attention re-analysis: pretrained vs finetuned (T3.10).

Duplicated verbatim into every language tree; never imports from a sibling.

    python hindi/eval/run_attention_compare.py \
        --pretrained kaggle_tuned_out/link1/checkpoints/hindi/step_8732.pt \
        --finetuned "kaggle_v2_out/sft/hindi_*/step_*.pt"

The spec asks to re-run the Phase-2 attention toolkit on the finetuned
checkpoints and compare. This driver imports that toolkit unchanged
(`attention_analysis.py`) so the two phases' numbers are produced by the same
code, and runs it on TWO probe sets:

  corpus     held-out natural text, built with EXACTLY the Phase-2 call
             (`build_prompts(..., 8, 64, 8, seed=7)`), so the finetuned numbers
             are directly comparable to the committed Phase-2 `attention.json`.
             Answers "did finetuning damage the language model's attention?"

  reasoning  held-out reasoning prompts. Answers "what did finetuning build?"

Why the reasoning probes are all one pattern at one exact token length:
`head_statistics` averages over the batch, which is only meaningful if query
position i means the same thing in every row. Padding to a common length is
NOT an option here -- this model has learned absolute positional embeddings and
`forward()` takes no attention mask, so a pad token is attended to like any
other and would silently corrupt every statistic. (Same root cause as the
right-padding bug fixed in `reasoning_metrics.py`.) Selecting rows that already
share a length avoids the problem instead of papering over it.

Beyond the Phase-2 statistics this adds one reasoning-specific measurement,
`answer_entity_attention`: at the final prompt position -- the position whose
logits produce the answer -- what share of the attention mass that lands on
candidate-entity tokens lands on the GOLD entity. Its baseline is not 1/k but
the occurrence-weighted uniform share, because entities repeat different
numbers of times in a chain (the pivot of a 3-entity chain appears twice).

Outputs report/phase3/<tree>/attention_compare.json plus figures.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TREE = HERE.parent
REPO = TREE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(TREE / "model"))
sys.path.insert(0, str(TREE / "train"))

import viz                                                    # noqa: E402
from attention_analysis import (analyse, content_dependence,  # noqa: E402
                                summarise_by_layer)
from checkpoint import load_checkpoint                        # noqa: E402
from common import load_tokenizer, shard_path, write_json     # noqa: E402
from reasoning_metrics import load_rows                       # noqa: E402
from run_eval import build_prompts                            # noqa: E402
from transformer import ModelConfig, TransformerLM            # noqa: E402

LANG = {"hindi": "hi", "nepali": "ne"}[TREE.name]


def load_model(path: str | Path, device):
    """Rebuild a model from its checkpoint's OWN stored config.

    `expected_lang_code` is the independence gate: both models have identical
    architecture and vocab size, so the wrong checkpoint would load silently.
    """
    ck = load_checkpoint(path, expected_lang_code=LANG, map_location=device)
    cfg = ModelConfig(**{k: v for k, v in ck["model_cfg"].items()
                         if k in ModelConfig.__dataclass_fields__})
    model = TransformerLM(cfg).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model


# ---------------------------------------------------------------- probes ---

def corpus_probes(sp, n: int, length: int, seed: int = 7) -> list[list[int]]:
    """Phase-2's own probe construction, called with Phase-2's own arguments."""
    prompts, _ = build_prompts(sp, shard_path("test"), n, length, 8, seed=seed)
    return prompts[:n]


def reasoning_probes(sp, data_path: Path, pattern: str, n: int,
                     seed: int = 7) -> tuple[list[list[int]], list[dict]]:
    """Held-out reasoning prompts of one pattern, all at the modal length.

    Returns (token ids, the source rows) so the answer-attention measurement
    can look up each row's candidates and gold answer.
    """
    rows = load_rows(data_path)
    rows = [r for r in rows if r["pattern"] == pattern]
    if not rows:
        raise SystemExit(f"no rows with pattern={pattern} in {data_path}")

    enc = {id(r): sp.encode(r["prompt"]) for r in rows}
    lengths = [len(enc[id(r)]) for r in rows]
    modal = max(set(lengths), key=lengths.count)
    keep = [r for r in rows if len(enc[id(r)]) == modal]

    rng = random.Random(seed)
    rng.shuffle(keep)
    keep = keep[:n]
    if len(keep) < n:
        raise SystemExit(f"only {len(keep)} rows at modal length {modal}, need {n}")

    # The ANSWER-POSITION probes are the prompt plus the answer prefix, i.e.
    # everything the target contains before the entity itself (" उत्तर: ").
    # That matters: the query whose logits emit the entity is the ':' token,
    # NOT the last token of the prompt, which emits the answer-marker word two
    # steps earlier. Measuring at the prompt's last token asks where the model
    # looks while deciding to start answering, not while choosing WHO.
    # Encoded whole rather than by concatenating encodings, so any merge across
    # the join is handled by the tokenizer instead of assumed away.
    ans_enc = []
    for r in keep:
        prefix = r["target_answer_only"].rsplit(r["answer"], 1)[0]
        ans_enc.append(sp.encode(r["prompt"] + prefix))
    widths = {len(e) for e in ans_enc}
    if len(widths) != 1:
        raise SystemExit(f"answer-position probes have mixed lengths {sorted(widths)}")
    return [enc[id(r)] for r in keep], keep, ans_enc


# ------------------------------------------------- answer attention ---------

def _entity_positions(sp, ids: list[int], name: str) -> list[int]:
    """Token positions in `ids` occupied by `name`.

    v2 entity pools are single-token by construction, but the subsequence
    search is kept so a multi-token name degrades to a correct (if slower)
    answer rather than a silently empty one.
    """
    ent = sp.encode(name)
    if not ent:
        return []
    if len(ent) == 1:
        return [i for i, t in enumerate(ids) if t == ent[0]]
    out = []
    for i in range(len(ids) - len(ent) + 1):
        if ids[i:i + len(ent)] == ent:
            out.extend(range(i, i + len(ent)))
    return out


@torch.no_grad()
def answer_entity_attention(model, sp, probes: list[list[int]],
                            rows: list[dict], device) -> dict:
    """Where does the answer position look?

    For every (layer, head), the share of the attention mass falling on
    candidate-entity tokens that falls on the gold entity, measured at the
    final position of `prompt + answer prefix` -- the query whose logits
    actually emit the entity (the ':' of " उत्तर: ").

    Also returns `uniform_share`: the share that same query would produce by
    spreading its mass evenly over every candidate-entity token. That, not
    1/k, is the correct chance line, because entities repeat unevenly.
    """
    from attention_analysis import capture_attention

    ids = torch.tensor(probes, dtype=torch.long, device=device)
    attns = capture_attention(model, ids)                # list[(B, h, T, T)]
    B, T = ids.shape
    n_layers, n_heads = len(attns), attns[0].size(1)

    # Per-row token positions for the gold entity and for all candidates.
    gold_pos, cand_pos, uniform = [], [], []
    for b, row in enumerate(rows):
        g = _entity_positions(sp, probes[b], row["answer"])
        c = []
        for name in row["candidates"]:
            c.extend(_entity_positions(sp, probes[b], name))
        c = sorted(set(c))
        gold_pos.append(sorted(set(g)))
        cand_pos.append(c)
        uniform.append(len(set(g)) / len(c) if c else float("nan"))

    per_head, by_layer = [], {}
    for layer, a in enumerate(attns):
        last = a[:, :, T - 1, :].float()                 # (B, h, T)
        shares = torch.full((B, n_heads), float("nan"))
        for b in range(B):
            if not cand_pos[b] or not gold_pos[b]:
                continue
            tot = last[b][:, cand_pos[b]].sum(dim=-1)    # (h,)
            gld = last[b][:, gold_pos[b]].sum(dim=-1)    # (h,)
            shares[b] = torch.where(tot > 0, gld / tot,
                                    torch.full_like(tot, float("nan")))
        col = shares.nanmean(dim=0)                      # (h,)
        for head in range(n_heads):
            per_head.append({"layer": layer, "head": head,
                             "gold_share": float(col[head])})
        by_layer[layer] = float(col.nanmean())

    finite = [u for u in uniform if u == u]
    return {
        "per_head": per_head,
        "by_layer": by_layer,
        "uniform_share": st.mean(finite) if finite else float("nan"),
        "mean_candidates_per_row": st.mean(len(c) for c in cand_pos),
        "n_layers": n_layers, "n_heads": n_heads, "n_rows": B, "seq_len": T,
    }


# ------------------------------------------------------------ analysis -----

def analyse_one(model, ids: torch.Tensor) -> tuple[list[dict], list[torch.Tensor]]:
    """Phase-2 per-head statistics plus content dependence, as plain dicts."""
    stats, attns = analyse(model, ids)
    cdep = content_dependence(model, ids)
    out = []
    for s in stats:
        d = s.to_dict()
        d["content_dependence"] = cdep.get((s.layer, s.head))
        out.append(d)
    return out, attns


METRICS = ("entropy_normalised", "entropy_nats", "mean_distance",
           "prev_token_frac", "first_token_frac", "content_dependence")


def aggregate_seeds(per_seed: list[list[dict]]) -> list[dict]:
    """Mean and half-range per (layer, head) across finetuning seeds.

    The three seeds differ only in finetuning randomness, so spread across
    them is the noise floor any pretrained-vs-finetuned delta must clear.
    """
    keys = [(d["layer"], d["head"]) for d in per_seed[0]]
    out = []
    for i, (layer, head) in enumerate(keys):
        row = {"layer": layer, "head": head}
        for m in METRICS:
            v = [seed[i][m] for seed in per_seed]
            row[m] = st.mean(v)
            row[f"{m}_half_range"] = (max(v) - min(v)) / 2
        out.append(row)
    return out


def deltas(pre: list[dict], post: list[dict]) -> list[dict]:
    """finetuned - pretrained, per head."""
    idx = {(d["layer"], d["head"]): d for d in pre}
    out = []
    for d in post:
        base = idx[(d["layer"], d["head"])]
        row = {"layer": d["layer"], "head": d["head"]}
        for m in METRICS:
            row[m] = d[m] - base[m]
            row[f"{m}_half_range"] = d.get(f"{m}_half_range", 0.0)
        out.append(row)
    return out


def layer_summary(stats: list[dict]) -> dict:
    """Layer means for every metric, including content dependence.

    `summarise_by_layer` predates the content-dependence measurement, so it is
    called for the Phase-2 metrics and content dependence is folded in here.
    """
    from attention_analysis import HeadStats
    base = summarise_by_layer([HeadStats(**{k: v for k, v in d.items()
                                            if k in HeadStats.__dataclass_fields__})
                               for d in stats])
    for layer in base:
        rows = [d for d in stats if d["layer"] == layer]
        base[layer]["content_dependence"] = st.mean(r["content_dependence"] for r in rows)
    return base


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Pretrained vs finetuned attention (T3.10).")
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--finetuned", required=True, help="glob for finetuned checkpoints")
    ap.add_argument("--data-dir", default=str(TREE / "reasoning_data"))
    ap.add_argument("--pattern", default="P3",
                    help="reasoning pattern to probe (P3 = 3-entity transitive)")
    ap.add_argument("--n-corpus", type=int, default=8)
    ap.add_argument("--corpus-len", type=int, default=64)
    ap.add_argument("--n-reasoning", type=int, default=32)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    ap.add_argument("--figdir", default=None)
    args = ap.parse_args(argv)

    out_path = Path(args.out or REPO / "report" / "phase3" / TREE.name /
                    "attention_compare.json")
    figdir = Path(args.figdir or out_path.parent / "figures")
    figdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")     # deterministic, and these probes are tiny
    torch.manual_seed(1337)
    sp = load_tokenizer()

    ft_paths = sorted(glob.glob(args.finetuned))
    if not ft_paths:
        raise SystemExit(f"no finetuned checkpoints match {args.finetuned}")
    print(f"[ckpt] pretrained={args.pretrained}")
    for p in ft_paths:
        print(f"[ckpt] finetuned  ={p}")

    # ---- probe sets -------------------------------------------------------
    cids = corpus_probes(sp, args.n_corpus, args.corpus_len, seed=args.seed)
    rids, rrows, aids = reasoning_probes(sp, Path(args.data_dir) / "test.jsonl",
                                         args.pattern, args.n_reasoning,
                                         seed=args.seed)
    probesets = {
        "corpus": (torch.tensor(cids, dtype=torch.long, device=device), None, None),
        "reasoning": (torch.tensor(rids, dtype=torch.long, device=device), rids, rrows),
    }
    print(f"[probe] corpus {len(cids)}x{len(cids[0])} tokens | "
          f"reasoning {len(rids)}x{len(rids[0])} tokens | "
          f"answer-position {len(aids)}x{len(aids[0])} tokens, "
          f"pattern={args.pattern}")

    models = [("pretrained", args.pretrained)] + [
        (f"finetuned/{Path(p).parent.name}", p) for p in ft_paths]

    results: dict = {
        "tree": TREE.name, "lang_code": LANG,
        "pretrained": str(args.pretrained), "finetuned": [str(p) for p in ft_paths],
        "pattern": args.pattern, "seed": args.seed,
        "probe_texts": {
            "corpus": [sp.decode(p) for p in cids],
            "reasoning": [r["prompt"] for r in rrows[:8]],
        },
        "probesets": {},
    }

    raw_attn: dict = {}      # (probeset, model_label) -> list[(B,h,T,T)]
    stats: dict = {}         # (probeset, model_label) -> list[dict]
    ans: dict = {}           # model_label -> answer-attention dict

    for label, path in models:
        model = load_model(path, device)
        for pname, (ids, id_lists, rows) in probesets.items():
            s, attns = analyse_one(model, ids)
            stats[(pname, label)] = s
            if label == "pretrained" or label == models[1][0]:
                raw_attn[(pname, label)] = attns    # heatmaps: one model each
            print(f"  [{label}] {pname}: "
                  f"ent_norm {st.mean(d['entropy_normalised'] for d in s):.3f} "
                  f"dist {st.mean(d['mean_distance'] for d in s):.2f} "
                  f"cdep {st.mean(d['content_dependence'] for d in s):.5f}", flush=True)
        ans[label] = answer_entity_attention(model, sp, aids, rrows, device)
        print(f"  [{label}] answer-position gold share "
              f"{st.mean(ans[label]['by_layer'].values()):.3f} "
              f"(uniform {ans[label]['uniform_share']:.3f})", flush=True)
        del model

    # ---- aggregate across finetuning seeds, then diff ---------------------
    ft_labels = [l for l, _ in models if l != "pretrained"]
    for pname in probesets:
        pre = stats[(pname, "pretrained")]
        post = aggregate_seeds([stats[(pname, l)] for l in ft_labels])
        dlt = deltas(pre, post)
        results["probesets"][pname] = {
            "n_probes": int(probesets[pname][0].size(0)),
            "seq_len": int(probesets[pname][0].size(1)),
            "pretrained": {"per_head": pre, "by_layer": layer_summary(pre)},
            "finetuned": {"per_head": post, "by_layer": layer_summary(post),
                          "n_seeds": len(ft_labels)},
            "delta": {"per_head": dlt},
        }

    # Answer attention: mean across finetuning seeds, kept per layer.
    results["answer_entity_attention"] = {
        "uniform_share": ans["pretrained"]["uniform_share"],
        "mean_candidates_per_row": ans["pretrained"]["mean_candidates_per_row"],
        "pretrained": ans["pretrained"]["by_layer"],
        "finetuned": {
            layer: st.mean(ans[l]["by_layer"][layer] for l in ft_labels)
            for layer in ans["pretrained"]["by_layer"]
        },
        "finetuned_per_head": aggregate_answer_heads([ans[l] for l in ft_labels]),
        "pretrained_per_head": ans["pretrained"]["per_head"],
    }

    write_json(results, out_path)
    print(f"  -> {out_path}")

    make_figures(results, raw_attn, sp, cids, rids, models, figdir, args)
    print(f"  -> {figdir}")


def aggregate_answer_heads(per_seed: list[dict]) -> list[dict]:
    """Mean gold-share per (layer, head) across finetuning seeds."""
    keys = per_seed[0]["per_head"]
    out = []
    for i, k in enumerate(keys):
        v = [s["per_head"][i]["gold_share"] for s in per_seed]
        out.append({"layer": k["layer"], "head": k["head"],
                    "gold_share": st.mean(v),
                    "gold_share_half_range": (max(v) - min(v)) / 2})
    return out


def make_figures(results, raw_attn, sp, cids, rids, models, figdir, args) -> None:
    """Every figure this task produces. No metric is computed in here."""
    lang = TREE.name
    ft_label = models[1][0]
    n_layers = max(int(l) for l in results["probesets"]["corpus"]["pretrained"]["by_layer"])
    early, late = 1, n_layers        # layer 0 is degenerate (attends only to itself)

    for pname in ("corpus", "reasoning"):
        blk = results["probesets"][pname]
        toks = [sp.id_to_piece(i) for i in (cids if pname == "corpus" else rids)[0]]

        # Heatmaps: one early + one late layer, pretrained and finetuned.
        for label, short in ((("pretrained"), "pretrained"), (ft_label, "finetuned")):
            attns = raw_attn.get((pname, label))
            if attns is None:
                continue
            for layer in (early, late):
                a = attns[layer][0].float().cpu().numpy()
                viz.plot_attention_heatmap(
                    a, layer, list(range(min(3, a.shape[0]))), toks,
                    f"{lang} {short} ({pname})",
                    figdir / f"attn_{pname}_layer{layer}_{short}.png")

        # Layer profiles, both models on one axis.
        for metric, ylab in (("entropy_normalised", "normalised entropy (1.0 = uniform)"),
                             ("mean_distance", "mean attention distance (tokens)"),
                             ("content_dependence", "mean |Δ attention| under token shuffle")):
            viz.plot_layer_profile_compare(
                {"pretrained": {int(k): v for k, v in blk["pretrained"]["by_layer"].items()},
                 "finetuned": {int(k): v for k, v in blk["finetuned"]["by_layer"].items()}},
                metric, ylab,
                f"{lang} ({pname} probes): {metric.replace('_', ' ')} by depth",
                figdir / f"attn_{pname}_profile_{metric}.png")

        # Per-head deltas.
        for metric, ylab in (("entropy_normalised", "Δ normalised entropy"),
                             ("mean_distance", "Δ mean distance (tokens)"),
                             ("content_dependence", "Δ content dependence")):
            viz.plot_head_delta_grid(
                blk["delta"]["per_head"], metric, ylab,
                f"{lang} ({pname}): {metric.replace('_', ' ')}, finetuned − pretrained",
                figdir / f"attn_{pname}_delta_{metric}.png")

    # Answer-position attention.
    aa = results["answer_entity_attention"]
    rows = []
    for label, key in (("pretrained", "pretrained"), ("finetuned", "finetuned")):
        for layer, share in aa[key].items():
            rows.append({"model": label, "layer_label": str(layer), "share": share})
    viz.plot_answer_attention(
        rows, aa["uniform_share"],
        f"{lang}: attention on the gold entity at the answer position "
        f"({args.pattern} probes)",
        figdir / "attn_answer_gold_share.png")


if __name__ == "__main__":
    main()
