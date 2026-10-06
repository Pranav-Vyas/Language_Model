"""Config-loader tests: real configs load, and the independence guard fires.

Run: python -m pytest tests/test_config.py  (or run this file directly).
Each language tree has its own identical copy of config.py; we import each and
check it validates its own config and rejects a cross-language path.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load_module(lang: str):
    """Import <lang>/config.py under a unique module name (they are twins)."""
    spec = importlib.util.spec_from_file_location(f"{lang}_config", REPO / lang / "config.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_real_configs_load():
    """Both shipped configs validate and carry the expected identity."""
    hi = _load_module("hindi").load_config(REPO / "hindi/configs/hindi.yaml")
    ne = _load_module("nepali").load_config(REPO / "nepali/configs/nepali.yaml")

    assert hi.lang_code == "hi" and hi.lang_name == "hindi"
    assert ne.lang_code == "ne" and ne.lang_name == "nepali"
    assert hi.tokenizer.model_type == "bpe" and ne.tokenizer.model_type == "bpe"

    # Every path stays inside its own tree.
    assert all(p.startswith("hindi") for p in hi.paths.as_dict().values())
    assert all(p.startswith("nepali") for p in ne.paths.as_dict().values())

    # Manual sources exist and clear the quota floor.
    assert {s.name for s in hi.data.sources} >= {"wiki", "sangraha"}
    assert {s.name for s in ne.data.sources} >= {"wiki", "news", "sangraha"}
    assert hi.data.manual_fraction_min >= 0.20 and ne.data.manual_fraction_min >= 0.20

    # Nepali's chars_per_token is the measured reference value, not the generic default.
    assert ne.data.chars_per_token == 4.988
    assert hi.data.chars_per_token == 4.0  # hindi.yaml doesn't set one -> generic default


def test_independence_guard_rejects_cross_language_path():
    """A Hindi config pointing at a nepali/ path must raise."""
    cfg = _load_module("hindi")
    with tempfile.TemporaryDirectory() as d:
        configs_dir = Path(d) / "hindi" / "configs"
        configs_dir.mkdir(parents=True)
        bad = configs_dir / "hindi.yaml"
        bad.write_text((REPO / "hindi/configs/hindi.yaml").read_text(encoding="utf-8")
                       .replace("tokenizer_dir: hindi/tokenizer",
                                "tokenizer_dir: nepali/tokenizer"),
                       encoding="utf-8")
        try:
            cfg.load_config(bad)
        except ValueError as e:
            assert "independence guard" in str(e) or "escapes" in str(e)
        else:
            raise AssertionError("cross-language path was NOT rejected — guard failed")


def test_independence_guard_rejects_reverse_direction():
    """And a Nepali config pointing at a hindi/ path must raise too."""
    cfg = _load_module("nepali")
    with tempfile.TemporaryDirectory() as d:
        configs_dir = Path(d) / "nepali" / "configs"
        configs_dir.mkdir(parents=True)
        bad = configs_dir / "nepali.yaml"
        bad.write_text((REPO / "nepali/configs/nepali.yaml").read_text(encoding="utf-8")
                       .replace("raw_dir: nepali/corpus/raw",
                                "raw_dir: hindi/corpus/raw"),
                       encoding="utf-8")
        try:
            cfg.load_config(bad)
        except ValueError as e:
            assert "independence guard" in str(e) or "escapes" in str(e)
        else:
            raise AssertionError("reverse cross-language path was NOT rejected")


if __name__ == "__main__":
    test_real_configs_load()
    test_independence_guard_rejects_cross_language_path()
    test_independence_guard_rejects_reverse_direction()
    print("OK: configs load and independence guard fires.")
