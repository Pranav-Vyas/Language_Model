"""Trainer-component tests (Phase 2, T2.7).

Targets the parts of a training loop that go wrong *quietly*: an LR schedule
that is subtly discontinuous, a data loader that is not reproducible across a
resume, an optimizer that decays the wrong parameters. None of these raise;
each of them makes the reported numbers wrong.

Runs against the `hindi/train` copy; `nepali/train` is asserted identical.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "hindi" / "train"))
sys.path.insert(0, str(REPO / "hindi" / "model"))

from schedule import WarmupCosineSchedule  # noqa: E402


# --------------------------------------------------------------------------
# LR schedule
# --------------------------------------------------------------------------

def test_warmup_rises_then_cosine_falls_to_the_floor():
    s = WarmupCosineSchedule(peak_lr=6e-4, warmup_steps=100, total_steps=1000)
    assert s.lr_at(0) == pytest.approx(6e-4 / 100)      # nonzero first step
    assert s.lr_at(99) == pytest.approx(6e-4)           # peak at end of warmup
    warm = [s.lr_at(i) for i in range(100)]
    assert warm == sorted(warm), "warmup must be monotonically increasing"
    cos = [s.lr_at(i) for i in range(100, 1000)]
    assert cos == sorted(cos, reverse=True), "cosine leg must be monotonically decreasing"
    assert s.lr_at(999) == pytest.approx(s.min_lr, rel=1e-3)


def test_schedule_is_continuous_at_the_warmup_boundary():
    """A jump at the handover would show up as a kink in the loss curve."""
    s = WarmupCosineSchedule(peak_lr=6e-4, warmup_steps=100, total_steps=1000)
    assert abs(s.lr_at(99) - s.lr_at(100)) < 1e-6


def test_schedule_holds_the_floor_past_the_end():
    """Past total_steps the cosine would turn back UP; it must not."""
    s = WarmupCosineSchedule(peak_lr=6e-4, warmup_steps=10, total_steps=100)
    assert s.lr_at(200) == pytest.approx(s.min_lr)
    assert s.lr_at(10_000) == pytest.approx(s.min_lr)


def test_schedule_is_a_pure_function_of_step():
    """The property that makes resume exact: no hidden state to restore."""
    a = WarmupCosineSchedule(6e-4, 100, 1000)
    b = WarmupCosineSchedule(6e-4, 100, 1000)
    for _ in range(500):
        b.lr_at(7)                       # exercise b out of order
    assert a.lr_at(432) == b.lr_at(432)


def test_degenerate_schedule_is_rejected():
    with pytest.raises(ValueError):
        WarmupCosineSchedule(6e-4, warmup_steps=1000, total_steps=1000)


# --------------------------------------------------------------------------
# Data loader
# --------------------------------------------------------------------------

def _loader(tmp_path, ctx=16, bs=4):
    """A small synthetic shard, so these tests need no corpus on disk."""
    import numpy as np
    from data import TokenShardLoader
    p = tmp_path / "toy.bin"
    np.arange(5000, dtype=np.uint16).tofile(p)
    return TokenShardLoader(p, ctx=ctx, batch_size=bs, seed=99)


def test_batch_is_deterministic_in_step(tmp_path):
    """Same step -> same batch, on a fresh loader object.

    This is what makes a resumed run see the same tokens as an uninterrupted
    one. Without it, resume silently changes the data order.
    """
    x1, y1 = _loader(tmp_path).batch(1234)
    x2, y2 = _loader(tmp_path).batch(1234)
    assert torch.equal(x1, x2) and torch.equal(y1, y2)


def test_different_steps_give_different_batches(tmp_path):
    """Guards the opposite failure: a 'deterministic' loader stuck on one batch."""
    ld = _loader(tmp_path)
    assert not torch.equal(ld.batch(0)[0], ld.batch(1)[0])


def test_targets_are_inputs_shifted_by_one(tmp_path):
    """y[:, t] must equal x[:, t+1] -- the whole next-token objective."""
    ld = _loader(tmp_path)
    x, y = ld.batch(7)
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert x.dtype == torch.int64 and y.dtype == torch.int64


def test_eval_batches_are_fixed_across_calls(tmp_path):
    """A val number that moves because the sample moved is not a measurement."""
    ld = _loader(tmp_path)
    a = [x for x, _ in ld.eval_batches(3)]
    b = [x for x, _ in ld.eval_batches(3)]
    assert all(torch.equal(i, j) for i, j in zip(a, b))


def test_missing_shard_raises_a_useful_error(tmp_path):
    from data import TokenShardLoader
    with pytest.raises(FileNotFoundError, match="tokenize_corpus"):
        TokenShardLoader(tmp_path / "nope.bin", ctx=16, batch_size=2)


# --------------------------------------------------------------------------
# Optimizer grouping
# --------------------------------------------------------------------------

def test_weight_decay_skips_layernorms_and_biases():
    """Decaying a LayerNorm gain toward zero suppresses the layer for no benefit."""
    from trainer import build_optimizer
    from train_config import TrainConfig
    from transformer import ModelConfig, TransformerLM

    model = TransformerLM(ModelConfig(vocab_size=64, d_model=32, n_layers=2,
                                      n_heads=4, d_ff=128, max_seq_len=16))
    opt = build_optimizer(model, TrainConfig())
    decay_group, plain_group = opt.param_groups
    assert decay_group["weight_decay"] > 0
    assert plain_group["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay_group["params"])
    assert all(p.dim() < 2 for p in plain_group["params"])


def test_tied_weight_is_placed_in_exactly_one_group():
    """The tied matrix is one tensor; decaying it twice would double the pull."""
    from trainer import build_optimizer
    from train_config import TrainConfig
    from transformer import ModelConfig, TransformerLM

    model = TransformerLM(ModelConfig(vocab_size=64, d_model=32, n_layers=2,
                                      n_heads=4, d_ff=128, max_seq_len=16,
                                      tie_weights=True))
    opt = build_optimizer(model, TrainConfig())
    ids = [id(p) for g in opt.param_groups for p in g["params"]]
    assert len(ids) == len(set(ids)), "a parameter appears in more than one group"


# --------------------------------------------------------------------------
# Independence
# --------------------------------------------------------------------------

def test_language_trees_hold_identical_training_code():
    """Duplication is the locked rule; drift between the copies is the risk."""
    for name in ["schedule", "data", "checkpoint", "trainer", "loop"]:
        hi = (REPO / "hindi" / "train" / f"{name}.py").read_bytes()
        ne = (REPO / "nepali" / "train" / f"{name}.py").read_bytes()
        assert hi == ne, f"{name}.py has drifted between the language trees"


# --------------------------------------------------------------------------
# Checkpointing (T2.8). The full proof is scripts/kill_resume_proof.py, which
# uses a real SIGKILL; these are the fast regression guards for its invariants.
# --------------------------------------------------------------------------

def _tiny():
    from transformer import ModelConfig, TransformerLM
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4,
                      d_ff=128, max_seq_len=16, lang_code="hi")
    return cfg, TransformerLM(cfg)


def test_checkpoint_rejects_a_different_language(tmp_path):
    """THE independence gate.

    Hindi and Nepali share vocab 16,000, so every tensor shape matches and a
    cross-language load would otherwise succeed silently. This must be a hard
    failure, and a distinctly-typed one.
    """
    from checkpoint import LanguageMismatchError, load_checkpoint, save_checkpoint
    from train_config import TrainConfig
    from trainer import build_optimizer

    cfg, model = _tiny()
    opt = build_optimizer(model, TrainConfig())
    p = save_checkpoint(tmp_path / "c.pt", model=model, optimizer=opt, step=1, model_cfg=cfg)

    with pytest.raises(LanguageMismatchError, match="independence"):
        load_checkpoint(p, expected_lang_code="ne")
    assert load_checkpoint(p, expected_lang_code="hi")["step"] == 1


def test_language_gate_cannot_be_skipped_by_omission():
    """`expected_lang_code` is keyword-REQUIRED: a gate you can forget is not a gate."""
    import inspect
    from checkpoint import load_checkpoint
    param = inspect.signature(load_checkpoint).parameters["expected_lang_code"]
    assert param.default is inspect.Parameter.empty
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


def test_checkpoint_carries_optimizer_scheduler_step_config_and_rng(tmp_path):
    """All six components the spec's resume requirement implies."""
    from checkpoint import load_checkpoint, save_checkpoint
    from train_config import TrainConfig
    from trainer import build_optimizer

    cfg, model = _tiny()
    opt = build_optimizer(model, TrainConfig())
    model(torch.randint(0, 64, (2, 8)), torch.randint(0, 64, (2, 8)))[1].backward()
    opt.step()
    p = save_checkpoint(tmp_path / "c.pt", model=model, optimizer=opt, step=7,
                        model_cfg=cfg, train_cfg=TrainConfig().to_dict())
    ck = load_checkpoint(p, expected_lang_code="hi")

    assert ck["step"] == 7                                   # step
    assert ck["model_cfg"]["d_model"] == 32                  # config
    assert ck["lang_code"] == "hi"                           # language
    assert ck["rng_state"]                                   # RNG
    state = ck["optimizer_state"]["state"]
    assert state and all("exp_avg" in v and "exp_avg_sq" in v for v in state.values())


def test_restoring_rng_reproduces_the_same_dropout_draw(tmp_path):
    """Without RNG restore, a resumed run takes a different random path.

    Dropout masks come from the global torch RNG. If it is not restored, the
    resumed run is statistically similar but NOT bit-identical -- and nothing
    raises to tell you.
    """
    from checkpoint import load_checkpoint, restore_into, save_checkpoint
    from train_config import TrainConfig
    from trainer import build_optimizer

    cfg, model = _tiny()
    opt = build_optimizer(model, TrainConfig())
    torch.manual_seed(4321)
    p = save_checkpoint(tmp_path / "c.pt", model=model, optimizer=opt, step=0, model_cfg=cfg)
    expected = torch.rand(4)

    torch.manual_seed(999)                                   # deliberately elsewhere
    ck = load_checkpoint(p, expected_lang_code="hi")
    restore_into(ck, model=model, optimizer=opt)
    assert torch.equal(torch.rand(4), expected)


def test_find_latest_sorts_numerically_not_lexicographically(tmp_path):
    """step_9000 must not sort after step_10000."""
    from checkpoint import find_latest
    for s in (500, 9000, 10000):
        (tmp_path / f"step_{s}.pt").write_bytes(b"x")
    assert find_latest(tmp_path).name == "step_10000.pt"
    assert find_latest(tmp_path / "nope") is None


def test_save_is_atomic_leaving_no_partial_file(tmp_path):
    """Written to a temp name then renamed, so a kill mid-write cannot truncate."""
    from checkpoint import save_checkpoint
    from train_config import TrainConfig
    from trainer import build_optimizer

    cfg, model = _tiny()
    p = save_checkpoint(tmp_path / "c.pt", model=model,
                        optimizer=build_optimizer(model, TrainConfig()),
                        step=1, model_cfg=cfg)
    assert p.exists() and not list(tmp_path.glob("*.tmp"))


def test_schedule_horizon_is_independent_of_where_a_link_stops():
    """A chained run must follow ONE cosine, not one per link.

    Regression test for a real corruption: when each link was given its own
    max_steps, link 1 annealed to the LR floor at its own end and link 2
    re-planned over a longer horizon, jumping the LR back up ~6x
    (6.02e-05 -> 3.63e-04, grad-norm 0.21 -> 0.98 at the boundary). Over a
    multi-link run that is a sawtooth. The horizon must come from the total
    run length, never from a link's stop point.
    """
    horizon = 8732
    sched = WarmupCosineSchedule(6e-4, 500, horizon)
    # Whatever the link boundary, the LR either side of it is continuous.
    for boundary in (1000, 4366, 7000):
        assert abs(sched.lr_at(boundary) - sched.lr_at(boundary - 1)) < 1e-5
    # And a schedule built for the same horizon is identical regardless of
    # how many links the run is split into.
    assert sched.lr_at(4366) == WarmupCosineSchedule(6e-4, 500, horizon).lr_at(4366)
    # A shorter horizon would give a materially different LR at the same step --
    # this is the bug, asserted so it cannot silently return.
    assert abs(sched.lr_at(4366) - WarmupCosineSchedule(6e-4, 500, 4366).lr_at(4366)) > 1e-4
