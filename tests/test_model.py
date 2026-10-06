"""Correctness tests for the decoder-only transformer (Phase 2, T2.5).

These target the failures that are SILENT: a model with a broken causal mask
trains happily, produces a falling loss curve, and scores plausible
perplexity -- while having cheated by reading the future. Nothing raises. The
only way to catch it is to test the property directly.

The suite runs against the `hindi/model` copy. The `nepali/model` copy is
asserted byte-identical, so testing one tests both without importing across
language trees (which would itself be an independence violation).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "hindi" / "model"))

from transformer import ModelConfig, TransformerLM  # noqa: E402


def tiny_cfg(**over) -> ModelConfig:
    """Small config so tests run in seconds on CPU."""
    base = dict(vocab_size=97, d_model=32, n_layers=2, n_heads=4, d_ff=128,
                max_seq_len=16, emb_dropout=0.0, attn_dropout=0.0,
                resid_dropout=0.0)
    base.update(over)
    return ModelConfig(**base)


# ---------------------------------------------------------------------------
# THE causal test the spec asks for
# ---------------------------------------------------------------------------

def test_future_token_cannot_change_past_logits():
    """Perturb token t+1; assert logits at every position <= t are unchanged.

    This is the spec's explicit requirement ("show that changing token t+1
    does not change the logits at position t"). It is the direct test of
    autoregressive validity: if it fails, every perplexity number the model
    ever produces is meaningless because it was computed with lookahead.

    Uses eval() so dropout is off -- otherwise the two runs would differ for
    reasons unrelated to masking.
    """
    torch.manual_seed(0)
    model = TransformerLM(tiny_cfg()).eval()

    ids = torch.randint(0, 97, (2, 12))
    with torch.no_grad():
        base, _ = model(ids)

    t = 6                                   # perturb position t+1 = 7
    modified = ids.clone()
    # Force a genuinely different token id at t+1.
    modified[:, t + 1] = (modified[:, t + 1] + 43) % 97
    assert not torch.equal(ids[:, t + 1], modified[:, t + 1])

    with torch.no_grad():
        after, _ = model(modified)

    # Positions 0..t must be bit-identical, not merely close.
    assert torch.equal(base[:, : t + 1, :], after[:, : t + 1, :]), (
        "logits at or before position t changed when token t+1 was modified -- "
        "the causal mask is leaking future information"
    )
    # And the change must actually have taken effect somewhere, otherwise the
    # test would pass on a model that ignores its input entirely.
    assert not torch.equal(base[:, t + 1, :], after[:, t + 1, :]), (
        "logits at t+1 did not change; the perturbation had no effect at all"
    )


def test_attention_weights_are_strictly_lower_triangular():
    """Every attention weight above the diagonal must be exactly zero.

    Complements the perturbation test by inspecting the mechanism rather than
    its behaviour: -inf pre-softmax must yield exp(-inf) = 0 above the
    diagonal, and each row must still sum to 1 after renormalisation.
    """
    torch.manual_seed(0)
    model = TransformerLM(tiny_cfg()).eval()
    ids = torch.randint(0, 97, (1, 10))

    with torch.no_grad():
        _, _, attns = model(ids, return_attention=True)

    for layer, w in enumerate(attns):
        B, h, T, T2 = w.shape
        assert T == T2 == 10
        upper = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        assert torch.all(w[..., upper] == 0), f"layer {layer} attends to the future"
        rows = w.sum(dim=-1)
        assert torch.allclose(rows, torch.ones_like(rows), atol=1e-5), (
            f"layer {layer} attention rows do not sum to 1 after masking"
        )


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

def test_forward_shapes():
    cfg = tiny_cfg()
    model = TransformerLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (3, 11))
    logits, loss = model(ids)
    assert logits.shape == (3, 11, cfg.vocab_size)
    assert loss is None

    targets = torch.randint(0, cfg.vocab_size, (3, 11))
    logits, loss = model(ids, targets)
    assert loss.ndim == 0 and loss.item() > 0


def test_attention_head_reshape_is_a_true_split():
    """(B,T,d) -> (B,h,T,d_k) must partition d_model, not overlap or drop it."""
    cfg = tiny_cfg()
    assert cfg.d_model == cfg.n_heads * (cfg.d_model // cfg.n_heads)
    model = TransformerLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 9))
    with torch.no_grad():
        _, _, attns = model(ids, return_attention=True)
    assert attns[0].shape == (2, cfg.n_heads, 9, 9)


def test_sequence_longer_than_context_is_rejected():
    """Learned absolute positions have no row past max_seq_len -- fail loudly."""
    cfg = tiny_cfg(max_seq_len=8)
    model = TransformerLM(cfg).eval()
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        model(torch.randint(0, cfg.vocab_size, (1, 9)))


# ---------------------------------------------------------------------------
# Weight tying and parameter accounting
# ---------------------------------------------------------------------------

def test_weight_tying_shares_storage_and_is_counted_once():
    cfg = tiny_cfg(tie_weights=True)
    model = TransformerLM(cfg)
    assert model.lm_head.weight is model.embed.token_emb.weight

    untied = TransformerLM(tiny_cfg(tie_weights=False))
    saving = untied.num_parameters() - model.num_parameters()
    assert saving == cfg.vocab_size * cfg.d_model, (
        "tying should save exactly vocab x d_model parameters"
    )


def test_gradient_flows_to_every_parameter():
    """A parameter with no gradient is silently dead weight."""
    cfg = tiny_cfg()
    model = TransformerLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    _, loss = model(ids, ids)
    loss.backward()
    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and (p.grad is None or torch.all(p.grad == 0))]
    assert not missing, f"parameters received no gradient: {missing}"


# ---------------------------------------------------------------------------
# The learning test: can it actually fit?
# ---------------------------------------------------------------------------

def test_overfits_a_tiny_batch():
    """Drive loss to ~0 on one batch.

    If a model cannot memorise a single batch, no amount of data or compute
    will help -- the bug is in the architecture or the loss wiring. This is
    the cheapest possible check that the whole forward/backward path is
    connected correctly.
    """
    torch.manual_seed(0)
    cfg = tiny_cfg(vocab_size=23, d_model=64, n_layers=2, max_seq_len=16)
    model = TransformerLM(cfg).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    ids = torch.randint(0, cfg.vocab_size, (2, 12))
    x, y = ids[:, :-1], ids[:, 1:]

    first = None
    for step in range(300):
        opt.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()

    assert loss.item() < 0.05, (
        f"failed to overfit a tiny batch: {first:.3f} -> {loss.item():.3f}"
    )


# ---------------------------------------------------------------------------
# Independence: the two model copies must not drift apart
# ---------------------------------------------------------------------------

def test_language_trees_hold_identical_model_code():
    """hindi/model and nepali/model must stay byte-identical copies.

    They are duplicated deliberately (no shared src/), so the only way they
    stay in sync is a test that fails when one is edited without the other.
    """
    for name in ("embeddings.py", "attention.py", "ffn.py", "block.py", "transformer.py"):
        hi = (REPO / "hindi" / "model" / name).read_bytes()
        ne = (REPO / "nepali" / "model" / name).read_bytes()
        assert hi == ne, f"{name} differs between hindi/ and nepali/"


# ---------------------------------------------------------------------------
# Bonus ablation — no positional embeddings
#
# The claim the ablation rests on is that removing the table removes the
# EXPLICIT position signal but not every position signal, because a causal
# decoder leaks position through its own mask. These tests pin down both
# halves of that claim so the report's explanation is backed by executable
# evidence rather than by argument.
# ---------------------------------------------------------------------------

def test_no_pe_drops_exactly_the_position_table():
    """Ablation removes max_seq_len x d_model params -- not zeroed, absent."""
    std = tiny_cfg()
    nope = tiny_cfg(use_positional_embeddings=False)
    m_std, m_nope = TransformerLM(std), TransformerLM(nope)

    delta = m_std.num_parameters() - m_nope.num_parameters()
    assert delta == std.max_seq_len * std.d_model

    # Absent, not merely unused: a frozen or zeroed table would still be in
    # the state dict and would still be restored on load.
    assert m_nope.embed.pos_emb is None
    assert not any("pos_emb" in n for n, _ in m_nope.named_parameters())
    assert not any("pos_emb" in k for k in m_nope.state_dict())


def test_positional_embeddings_default_to_on():
    """Every pre-ablation checkpoint must reload with its positions intact."""
    assert ModelConfig().use_positional_embeddings is True


def test_no_pe_single_layer_is_order_blind():
    """One no-PE layer: permuting the prefix cannot change the last position.

    With a single layer the last position's output is
        FFN(e_last + sum_i softmax(q . k_i) v_i)
    over the SET of visible tokens. A sum over a set does not depend on the
    order of that set, so [A, B, Q] and [B, A, Q] must agree exactly at the
    final position. The standard model disagrees, because the positional table
    makes e_A-at-0 a different vector from e_A-at-1.
    """
    torch.manual_seed(0)
    ids_ab = torch.tensor([[3, 7, 11]])
    ids_ba = torch.tensor([[7, 3, 11]])          # prefix permuted, last token fixed

    nope = TransformerLM(tiny_cfg(n_layers=1, use_positional_embeddings=False)).eval()
    with torch.no_grad():
        a, _ = nope(ids_ab)
        b, _ = nope(ids_ba)
    # Last position only: earlier positions legitimately differ (different tokens).
    assert torch.allclose(a[0, -1], b[0, -1], atol=1e-5), \
        "no-PE single layer must be invariant to prefix order at the last position"

    torch.manual_seed(0)
    std = TransformerLM(tiny_cfg(n_layers=1)).eval()
    with torch.no_grad():
        a, _ = std(ids_ab)
        b, _ = std(ids_ba)
    assert (a[0, -1] - b[0, -1]).abs().max() > 1e-3, \
        "control: the standard model MUST see prefix order"


def test_no_pe_regains_order_sensitivity_with_depth():
    """Stacking restores order sensitivity even with no positional table.

    This is why the ablated model does not simply collapse to a bag of words.
    Layer 1's outputs at positions 0 and 1 are computed from different visible
    prefixes ({A} vs {B}), so layer 2 reads a different key/value set for the
    two orderings and the final logits diverge. Position information is
    reconstructed from the causal mask, one layer removed.
    """
    torch.manual_seed(0)
    model = TransformerLM(tiny_cfg(n_layers=2, use_positional_embeddings=False)).eval()
    with torch.no_grad():
        a, _ = model(torch.tensor([[3, 7, 11]]))
        b, _ = model(torch.tensor([[7, 3, 11]]))
    assert (a[0, -1] - b[0, -1]).abs().max() > 1e-6, \
        "two no-PE layers should already distinguish prefix order via the mask"


def test_no_pe_model_is_still_causal():
    """Removing positions must not weaken the causal guarantee."""
    torch.manual_seed(0)
    model = TransformerLM(tiny_cfg(use_positional_embeddings=False)).eval()
    ids = torch.randint(0, 97, (1, 8))
    with torch.no_grad():
        base, _ = model(ids)
        tampered = ids.clone()
        tampered[0, 5] = (tampered[0, 5] + 1) % 97      # perturb position 5
        after, _ = model(tampered)
    # Positions 0..4 precede the edit and must be bit-identical.
    assert torch.equal(base[0, :5], after[0, :5]), \
        "a future token changed a past logit in the ablated model"
    assert not torch.equal(base[0, 5], after[0, 5]), "position 5 should change"


def test_tokenizer_folds_nukta_forms_identically():
    """Precomposed and NFC-decomposed nukta must reach the model as one token.

    NFC decomposes U+095C into U+0921 + U+093C, so the cleaned corpus holds
    only the decomposed form while many input methods emit the precomposed
    one. SentencePiece's own normaliser folds them, which is why hand-written
    text can be mixed with corpus text safely. Asserted so a future tokenizer
    change cannot break that silently.
    """
    spm = pytest.importorskip("sentencepiece")
    for tree, code in (("hindi", "hi"), ("nepali", "ne")):
        path = REPO / tree / "tokenizer" / f"{code}_bpe_16000.model"
        if not path.exists():
            pytest.skip(f"{path} not present")
        sp = spm.SentencePieceProcessor(model_file=str(path))
        pre = "बड़ा"                       # ba + DDDHA + aa
        deco = "बड़ा"                # ba + DDA + NUKTA + aa
        assert sp.encode(pre) == sp.encode(deco), f"{tree}: nukta forms diverge"
