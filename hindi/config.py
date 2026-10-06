"""Config schema + loader for a single, self-contained language tree.

This file is duplicated *verbatim* into every language directory (``hindi/``,
``nepali/``). It is deliberately NOT shared: the locked project rule is that a
language tree never imports from a sibling tree, so each carries its own copy.

The core job here is the **independence guard** (`_assert_paths_are_local`):
a config is only allowed to point at paths inside *its own* language directory.
A Hindi config that names a ``nepali/...`` path raises immediately, which is
the cheapest possible enforcement of "separate data / tokenizer / vocab / weights".
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# The two accounting buckets every source must declare. The manual/downloaded
# split is a graded Phase-1 statistic, so the value is validated, not free-text.
VALID_BUCKETS = {"manual", "downloaded"}

# Locked decision: SentencePiece BPE only. Guard against a config drifting to
# unigram/word/char by mistake.
VALID_TOKENIZER_TYPES = {"bpe"}


@dataclass(frozen=True)
class SourceSpec:
    """One corpus source and how to fetch/account for it.

    `bucket` decides manual-vs-downloaded accounting; `name` is the per-source
    label carried on every document (e.g. ``wiki``, ``hplt``, ``ocr:bhikhari``).
    The optional fields describe *where* the data comes from and are only read
    by the collectors that need them.
    """

    name: str
    bucket: str
    wiki_lang: str | None = None      # Wikipedia API host code, e.g. "hi", "ne"
    hf_dataset: str | None = None     # HuggingFace dataset id
    hf_config: str | None = None      # HF config name, e.g. "verified"
    hf_subdir: str | None = None      # sub-directory within the dataset, e.g. "hin", "nep"
    notes: str | None = None


@dataclass(frozen=True)
class Paths:
    """All filesystem locations for this language, relative to the repo root.

    Every value MUST live inside the language's own top-level directory
    (`root`). That invariant is enforced in `_assert_paths_are_local`.
    """

    root: str          # the language directory itself, e.g. "hindi"
    corpus_dir: str    # gitignored artifact root
    raw_dir: str       # collected-but-uncleaned documents
    clean_dir: str     # after the cleaning pipeline
    dedup_dir: str     # after exact + near dedup
    splits_dir: str    # train/val/test manifests + text
    tokenizer_dir: str # committed .model/.vocab live here
    stats_dir: str     # computed statistics (JSON/CSV) for the report
    manifest: str      # provenance manifest (one row per document)

    def as_dict(self) -> dict[str, str]:
        return {
            "root": self.root, "corpus_dir": self.corpus_dir, "raw_dir": self.raw_dir,
            "clean_dir": self.clean_dir, "dedup_dir": self.dedup_dir,
            "splits_dir": self.splits_dir, "tokenizer_dir": self.tokenizer_dir,
            "stats_dir": self.stats_dir, "manifest": self.manifest,
        }


@dataclass(frozen=True)
class DataCfg:
    """Corpus-collection targets and the ordered list of sources."""

    target_tokens: int          # RAW collection goal, pre-cleaning
    manual_fraction_min: float  # spec floor: >= 0.20
    sources: tuple[SourceSpec, ...]
    # The spec's actual requirement: "~500M training tokens per language".
    # Kept separate from `target_tokens` because that one is the raw haul we
    # aim to collect so cleaning has a buffer to spend. Conflating the two
    # silently breaks the manual-share denominator: measuring the manual
    # fraction against the raw goal rather than the training target reports
    # the diluted number as if it were the graded one.
    training_target_tokens: int = 500_000_000
    chars_per_token: float = 4.0  # used to SIZE bulk downloads before a real tokenizer
                                   # exists; override per-language once measured (e.g.
                                   # Nepali's Sangraha text measured at 4.988, not the
                                   # generic default). Not used for final reported token
                                   # counts once real text is in hand -- those come from
                                   # actual word/char counts, same as every other source.


@dataclass(frozen=True)
class TokenizerCfg:
    """SentencePiece BPE settings and the vocab-size sweep."""

    model_type: str                 # must be "bpe"
    vocab_sizes: tuple[int, ...]    # sweep candidates; winner chosen on fertility/UNK
    character_coverage: float       # e.g. 0.9995 for Devanagari
    byte_fallback: bool             # decode-safety for unseen bytes


@dataclass(frozen=True)
class SplitCfg:
    """Document-level split ratios and seed."""

    train: float
    val: float
    test: float


@dataclass(frozen=True)
class Config:
    """Fully-validated configuration for one language tree."""

    lang_code: str      # short/Wikipedia code, e.g. "hi", "ne"
    lang_name: str      # directory name + human label, e.g. "hindi", "nepali"
    script: str         # e.g. "Devanagari"
    seed: int
    paths: Paths
    data: DataCfg
    tokenizer: TokenizerCfg
    split: SplitCfg
    source_path: Path = field(compare=False, default=Path("."))  # where it loaded from


def _expected_root_from_location(config_path: Path) -> str:
    """Infer which language tree a config belongs to, from its own location.

    A config lives at ``<repo>/<lang>/configs/<lang>.yaml``. The language tree
    root is therefore the parent of the ``configs/`` directory. We use this to
    cross-check ``paths.root`` — the config cannot lie about which tree it is in,
    because its filesystem location is ground truth.
    """
    configs_dir = config_path.parent
    if configs_dir.name != "configs":
        raise ValueError(
            f"config must live in a 'configs/' directory, got: {config_path}"
        )
    return configs_dir.parent.name


def _assert_paths_are_local(paths: Paths, expected_root: str) -> None:
    """The independence guard: every path stays inside this language's own tree.

    Rejects absolute paths, parent-directory escapes (``..``), and any path that
    does not begin with ``<expected_root>/`` (or equal it). This is what makes it
    physically impossible for a Hindi config to reference a Nepali artifact.
    """
    if paths.root != expected_root:
        raise ValueError(
            f"paths.root='{paths.root}' does not match the tree this config lives "
            f"in ('{expected_root}'). A config may only describe its own language."
        )
    for key, value in paths.as_dict().items():
        p = Path(value)
        if p.is_absolute():
            raise ValueError(f"paths.{key}='{value}' must be relative to the repo root.")
        if ".." in p.parts:
            raise ValueError(f"paths.{key}='{value}' must not contain '..' (no escaping the tree).")
        # Normalise separators, then require the root prefix.
        norm = value.replace(os.sep, "/")
        if norm != expected_root and not norm.startswith(expected_root + "/"):
            raise ValueError(
                f"paths.{key}='{value}' escapes this language tree; every path must "
                f"start with '{expected_root}/'. This is the independence guard."
            )


def _build_sources(raw: list[dict]) -> tuple[SourceSpec, ...]:
    """Validate and construct the ordered source list."""
    sources: list[SourceSpec] = []
    seen_names: set[str] = set()
    for i, s in enumerate(raw):
        name = s.get("name")
        bucket = s.get("bucket")
        if not name:
            raise ValueError(f"data.sources[{i}] is missing 'name'.")
        if name in seen_names:
            raise ValueError(f"duplicate source name '{name}' in data.sources.")
        seen_names.add(name)
        if bucket not in VALID_BUCKETS:
            raise ValueError(
                f"data.sources[{i}] bucket='{bucket}' invalid; must be one of {sorted(VALID_BUCKETS)}."
            )
        sources.append(SourceSpec(
            name=name, bucket=bucket,
            wiki_lang=s.get("wiki_lang"), hf_dataset=s.get("hf_dataset"),
            hf_config=s.get("hf_config"), hf_subdir=s.get("hf_subdir"),
            notes=s.get("notes"),
        ))
    if not sources:
        raise ValueError("data.sources must list at least one source.")
    return tuple(sources)


def load_config(config_path: str | os.PathLike) -> Config:
    """Load, validate, and return a `Config` for one language tree.

    Raises ``ValueError`` on any structural problem or independence violation.
    Validation is strict on purpose: a bad config should fail here, loudly, and
    never silently point training at the wrong corpus.
    """
    config_path = Path(config_path).resolve()
    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path}: top-level YAML must be a mapping.")

    paths = Paths(**raw["paths"])
    expected_root = _expected_root_from_location(config_path)
    _assert_paths_are_local(paths, expected_root)

    # Cross-check identity fields against the tree location.
    lang_name = raw["lang_name"]
    if lang_name != expected_root:
        raise ValueError(
            f"lang_name='{lang_name}' must equal the tree directory '{expected_root}'."
        )

    data_raw = raw["data"]
    data = DataCfg(
        target_tokens=int(data_raw["target_tokens"]),
        manual_fraction_min=float(data_raw["manual_fraction_min"]),
        sources=_build_sources(data_raw["sources"]),
        training_target_tokens=int(data_raw.get("training_target_tokens", 500_000_000)),
        chars_per_token=float(data_raw.get("chars_per_token", 4.0)),
    )
    if not 0.0 < data.manual_fraction_min <= 1.0:
        raise ValueError("data.manual_fraction_min must be in (0, 1].")
    if data.manual_fraction_min < 0.20:
        raise ValueError("data.manual_fraction_min is below the spec floor of 0.20.")

    tok_raw = raw["tokenizer"]
    tokenizer = TokenizerCfg(
        model_type=tok_raw["model_type"],
        vocab_sizes=tuple(int(v) for v in tok_raw["vocab_sizes"]),
        character_coverage=float(tok_raw["character_coverage"]),
        byte_fallback=bool(tok_raw["byte_fallback"]),
    )
    if tokenizer.model_type not in VALID_TOKENIZER_TYPES:
        raise ValueError(
            f"tokenizer.model_type='{tokenizer.model_type}' invalid; must be 'bpe' (locked decision)."
        )
    if not tokenizer.vocab_sizes or any(v <= 0 for v in tokenizer.vocab_sizes):
        raise ValueError("tokenizer.vocab_sizes must be a non-empty list of positive ints.")

    split_raw = raw["split"]
    split = SplitCfg(
        train=float(split_raw["train"]), val=float(split_raw["val"]), test=float(split_raw["test"]),
    )
    if abs((split.train + split.val + split.test) - 1.0) > 1e-6:
        raise ValueError("split ratios (train+val+test) must sum to 1.0.")

    return Config(
        lang_code=raw["lang_code"], lang_name=lang_name, script=raw["script"],
        seed=int(raw["seed"]), paths=paths, data=data, tokenizer=tokenizer,
        split=split, source_path=config_path,
    )
