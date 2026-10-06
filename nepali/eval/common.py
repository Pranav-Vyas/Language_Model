"""Shared plumbing for this tree's evaluation modules.

Duplicated verbatim into every language tree; never imports from a sibling.

Every entry point here goes through `load_model_from_checkpoint`, which
enforces the language gate. That matters more at evaluation time than during
training: the two models have identical architecture and identical vocab size
(16,000), so loading Nepali weights into a Hindi model raises nothing and
produces perfectly plausible metrics for the wrong model.
"""

from __future__ import annotations

import contextlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

TREE = Path(__file__).resolve().parent.parent
for _sub in ("model", "train"):
    _p = str(TREE / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from checkpoint import load_checkpoint            # noqa: E402
from transformer import ModelConfig, TransformerLM  # noqa: E402


def lang_code() -> str:
    """This tree's language code, read from its own model config."""
    import yaml
    cfg = yaml.safe_load((TREE / "configs" / "model.yaml").read_text(encoding="utf-8"))
    return cfg["lang_code"]


def seed_everything(seed: int = 1337) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(prefer_cuda: bool = True) -> torch.device:
    return torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")


def autocast_for(device: torch.device):
    """bf16 autocast on CUDA; a no-op elsewhere (bf16 is slower than fp32 on CPU)."""
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext


def load_tokenizer(vocab_size: int = 16000):
    """This tree's own SentencePiece model. Never a sibling's."""
    import sentencepiece as spm
    path = TREE / "tokenizer" / f"{lang_code()}_bpe_{vocab_size}.model"
    if not path.exists():
        raise FileNotFoundError(f"tokenizer not found: {path}")
    return spm.SentencePieceProcessor(model_file=str(path))


def load_model_from_checkpoint(ckpt_path: str | Path, device: torch.device):
    """Rebuild the model recorded in a checkpoint and load its weights.

    The architecture comes from the checkpoint's own stored config rather than
    from the current YAML, so evaluating an older checkpoint cannot silently
    use today's hyperparameters.
    """
    expected = lang_code()
    ckpt = load_checkpoint(ckpt_path, expected_lang_code=expected, map_location=device)

    stored = dict(ckpt["model_cfg"])
    cfg = ModelConfig(**{k: v for k, v in stored.items()
                         if k in ModelConfig.__dataclass_fields__})
    model = TransformerLM(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    meta = {"step": int(ckpt["step"]), "val_loss": ckpt.get("val_loss"),
            "lang_code": ckpt["lang_code"], "path": str(ckpt_path)}
    return model, cfg, meta


def shard_path(split: str) -> Path:
    return TREE / "data" / "tokenized" / f"{split}.bin"


def write_json(obj, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
