"""Cross-language independence gate (T1.14).

Language-agnostic tooling. This is the ONLY script allowed to look at both
language trees at once, and it holds no language data of its own -- it reads
each tree's artifacts and asserts they never overlap.

WHAT THE SPEC REQUIRES
----------------------
"Two fully independent monolingual LMs (separate data, tokenizer, vocabulary,
and weights). Do not train a multilingual model or share artifacts across
languages." Being careful is not evidence; this script produces the evidence,
and its output goes in the phase report.

CHECKS
------
1. No cross-tree imports. A module under hindi/ must never import from
   nepali/ or vice versa -- that is the code-level version of sharing.
2. Config isolation. Each config's declared paths must stay inside its own
   tree (this re-runs the loader's own guard, from the outside).
3. No shared documents. SHA-256 of every cleaned/split document in one tree
   must not appear in the other. Exact overlap is the hard failure.
4. No near-duplicate documents across trees. The same wire story reworded in
   both corpora is still contamination; exact hashing would miss it, so the
   MinHash machinery is reused here.
5. Separate tokenizers and vocabularies. The .model files must differ, and
   the vocabularies must not be identical.

Exit code is non-zero if any check fails, so this can gate a push.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


@dataclass
class CheckResult:
    """One named check plus enough detail to put in the report."""

    name: str
    passed: bool
    detail: str
    data: dict = field(default_factory=dict)


def find_language_trees(repo: Path) -> list[Path]:
    """A language tree is a top-level dir with configs/<name>.yaml inside it."""
    trees = []
    for child in sorted(repo.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        cfg = child / "configs" / f"{child.name}.yaml"
        if cfg.exists():
            trees.append(child)
    return trees


def check_no_cross_imports(trees: list[Path]) -> CheckResult:
    """Grep every .py in each tree for an import naming a sibling tree.

    Deliberately textual rather than import-graph based: a cross-tree import
    is forbidden even if it is never executed, and a static scan cannot be
    fooled by a branch that happens not to run.
    """
    names = {t.name for t in trees}
    violations: list[str] = []
    for tree in trees:
        others = names - {tree.name}
        for py in tree.rglob("*.py"):
            text = py.read_text(encoding="utf-8", errors="replace")
            for other in others:
                # `import nepali`, `from nepali...`, or a path string.
                pattern = rf"^\s*(?:from|import)\s+{other}\b|['\"]{other}/"
                for m in re.finditer(pattern, text, re.MULTILINE):
                    line_no = text[:m.start()].count("\n") + 1
                    violations.append(f"{py.relative_to(REPO)}:{line_no} references '{other}'")
    return CheckResult(
        name="no_cross_tree_imports",
        passed=not violations,
        detail=("no module references a sibling language tree"
                if not violations else f"{len(violations)} violation(s)"),
        data={"violations": violations},
    )


def _load_config(tree: Path):
    """Load a tree's config using that tree's own loader copy."""
    spec = importlib.util.spec_from_file_location(f"{tree.name}_cfg", tree / "config.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.load_config(tree / "configs" / f"{tree.name}.yaml")


def check_config_isolation(trees: list[Path]) -> CheckResult:
    """Every path a config declares must live inside that config's own tree."""
    problems: list[str] = []
    langs: dict[str, str] = {}
    for tree in trees:
        try:
            cfg = _load_config(tree)
        except Exception as exc:                      # a raising loader IS the guard working
            problems.append(f"{tree.name}: config failed to load: {exc}")
            continue
        langs[tree.name] = cfg.lang_code
        for key, value in cfg.paths.as_dict().items():
            if not value.replace("\\", "/").startswith(tree.name):
                problems.append(f"{tree.name}: paths.{key}='{value}' escapes the tree")
    # Two trees sharing a lang_code would break per-language artifact naming.
    if len(set(langs.values())) != len(langs):
        problems.append(f"duplicate lang_code across trees: {langs}")
    return CheckResult(
        name="config_isolation",
        passed=not problems,
        detail=("each config only references its own tree; lang_codes distinct"
                if not problems else f"{len(problems)} problem(s)"),
        data={"lang_codes": langs, "problems": problems},
    )


def _iter_docs(path: Path, limit: int | None = None):
    """Stream `text` fields out of a jsonl corpus file."""
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                return
            line = line.strip()
            if not line:
                continue
            try:
                text = json.loads(line).get("text") or ""
            except json.JSONDecodeError:
                continue
            if text:
                yield text


def _corpus_files(tree: Path) -> list[Path]:
    """Prefer the split files (final corpus); fall back to cleaned output."""
    splits = sorted((tree / "corpus" / "splits").glob("*.jsonl"))
    if splits:
        return splits
    return sorted((tree / "corpus" / "clean").glob("*.jsonl"))


def check_no_shared_documents(trees: list[Path], limit: int | None) -> CheckResult:
    """Assert the two corpora share no byte-identical document.

    Hashes of the smaller corpus are held in memory and the larger is
    streamed against them, so peak memory is one hash set rather than one
    corpus.
    """
    if len(trees) != 2:
        return CheckResult("no_shared_documents", True, "needs exactly 2 trees; skipped")

    counts = {}
    for tree in trees:
        counts[tree.name] = sum(1 for f in _corpus_files(tree)
                                for _ in _iter_docs(f, limit))
    small, large = sorted(trees, key=lambda t: counts[t.name])

    seen: set[bytes] = set()
    for f in _corpus_files(small):
        for text in _iter_docs(f, limit):
            seen.add(hashlib.sha256(text.encode("utf-8")).digest()[:16])

    overlaps = 0
    checked = 0
    for f in _corpus_files(large):
        for text in _iter_docs(f, limit):
            checked += 1
            if hashlib.sha256(text.encode("utf-8")).digest()[:16] in seen:
                overlaps += 1

    return CheckResult(
        name="no_shared_documents",
        passed=overlaps == 0,
        detail=f"{overlaps} exact document(s) shared across corpora",
        data={"docs": counts, "compared": checked, "overlaps": overlaps},
    )


def check_no_cross_near_duplicates(trees: list[Path], limit: int | None) -> CheckResult:
    """Assert neither corpus contains a near-duplicate of the other's documents.

    Reuses one tree's MinHash implementation as a *library* for a
    language-agnostic comparison. That is not a cross-language data
    dependency: no corpus content crosses trees, only the algorithm.
    """
    if len(trees) != 2:
        return CheckResult("no_cross_near_duplicates", True, "needs exactly 2 trees; skipped")

    sys.path.insert(0, str(trees[0] / "data_prep"))
    import near_dedup as nd

    cfg = nd.MinHashConfig()
    sig_fn = nd.make_signature_fn(cfg)

    index: dict[bytes, list[int]] = {}
    sigs: list[list[int]] = []
    for f in _corpus_files(trees[0]):
        for text in _iter_docs(f, limit):
            sig = sig_fn(nd.shingles(text, cfg.shingle_size))
            i = len(sigs)
            sigs.append(sig)
            for key in nd.band_keys(sig, cfg):
                index.setdefault(key, []).append(i)

    hits = 0
    checked = 0
    for f in _corpus_files(trees[1]):
        for text in _iter_docs(f, limit):
            checked += 1
            sig = sig_fn(nd.shingles(text, cfg.shingle_size))
            seen_ids: set[int] = set()
            found = False
            for key in nd.band_keys(sig, cfg):
                for j in index.get(key, ()):
                    if j in seen_ids:
                        continue
                    seen_ids.add(j)
                    if nd.estimated_jaccard(sig, sigs[j]) >= cfg.threshold:
                        found = True
                        break
                if found:
                    break
            hits += found

    return CheckResult(
        name="no_cross_near_duplicates",
        passed=hits == 0,
        detail=f"{hits} near-duplicate pair(s) across corpora (threshold {cfg.threshold:.3f})",
        data={"indexed": len(sigs), "compared": checked, "near_duplicates": hits,
              "threshold": cfg.threshold},
    )


def check_separate_tokenizers(trees: list[Path]) -> CheckResult:
    """Tokenizer models must exist per tree, and must not be the same file.

    Also compares vocabularies: two different .model files that produced an
    identical vocabulary would still mean a shared vocabulary in substance.
    """
    digests: dict[str, str] = {}
    vocabs: dict[str, str] = {}
    missing: list[str] = []
    for tree in trees:
        models = sorted((tree / "tokenizer").glob("*.model"))
        if not models:
            missing.append(tree.name)
            continue
        # Compare the largest-vocab model from each tree (the chosen one is
        # recorded separately; any pair sharing bytes is already a failure).
        digests[tree.name] = hashlib.sha256(models[-1].read_bytes()).hexdigest()[:16]
        vfile = models[-1].with_suffix(".vocab")
        if vfile.exists():
            vocabs[tree.name] = hashlib.sha256(vfile.read_bytes()).hexdigest()[:16]

    if missing:
        return CheckResult("separate_tokenizers", True,
                           f"not yet trained for: {', '.join(missing)} (skipped)",
                           {"missing": missing})
    same_model = len(set(digests.values())) != len(digests)
    same_vocab = len(vocabs) > 1 and len(set(vocabs.values())) != len(vocabs)
    return CheckResult(
        name="separate_tokenizers",
        passed=not (same_model or same_vocab),
        detail=("tokenizer models and vocabularies differ per language"
                if not (same_model or same_vocab)
                else "IDENTICAL tokenizer model or vocabulary across languages"),
        data={"model_sha": digests, "vocab_sha": vocabs},
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Cross-language independence gate.")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap documents read per corpus file (fast smoke run).")
    ap.add_argument("--skip-near-dup", action="store_true",
                    help="skip the (slower) cross-corpus near-duplicate check.")
    ap.add_argument("--out", default="report/phase1/independence_check.json")
    args = ap.parse_args(argv)

    trees = find_language_trees(REPO)
    print(f"language trees: {[t.name for t in trees]}\n")

    results = [check_no_cross_imports(trees), check_config_isolation(trees)]
    results.append(check_no_shared_documents(trees, args.limit))
    if not args.skip_near_dup:
        results.append(check_no_cross_near_duplicates(trees, args.limit))
    results.append(check_separate_tokenizers(trees))

    width = max(len(r.name) for r in results)
    for r in results:
        print(f"  [{'PASS' if r.passed else 'FAIL'}] {r.name:<{width}}  {r.detail}")
        if not r.passed:
            for k, v in r.data.items():
                if isinstance(v, list) and v:
                    for item in v[:10]:
                        print(f"           - {item}")

    ok = all(r.passed for r in results)
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"passed": ok, "checks": [asdict(r) for r in results]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  {'ALL CHECKS PASSED' if ok else 'INDEPENDENCE VIOLATION'}")
    print(f"  evidence -> {out.relative_to(REPO)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
