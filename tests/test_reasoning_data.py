"""Correctness tests for the synthetic reasoning dataset (Phase 3).

The failure mode these target is SILENT. If the premise renderer got a polarity
backwards -- emitting "A is shorter than B" for the fact A > B -- every example
would still look well-formed, the finetune would run, and the model would learn
a consistently inverted relation. Nothing would raise. The only way to catch it
is to parse the RENDERED TEXT back into an order and re-derive the answer
independently of the chain the generator read its label from.

Both language trees are exercised; they are separate datasets and a bug could
live in either template file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def load_tree(tree: str):
    """Import one tree's reasoning modules in isolation.

    The two trees have modules of the SAME name (`templates`, `entities`,
    `generate`) by design -- they are independent copies, not a shared package.
    Stale entries are evicted so the second import does not silently return the
    first tree's modules.
    """
    for m in ("templates", "entities", "generate"):
        sys.modules.pop(m, None)
    path = str(REPO / tree / "reasoning")
    sys.path.insert(0, path)
    import entities, generate, templates      # noqa: E402
    return templates, entities, generate, path


TREES = ["hindi", "nepali"]


@pytest.mark.parametrize("tree", TREES)
def test_answer_is_rederivable_from_rendered_text(tree):
    """Parse premises back into an order; the label must follow from it."""
    T, E, G, path = load_tree(tree)
    try:
        rows = G.build("train", 800, seed=1337)
        POS = {r[1] for r in T.RELATIONS}
        NEG = {r[2] for r in T.RELATIONS}

        def parse_facts(prompt, chain):
            # Premises end with '।'; the trailing question does not, and it also
            # names entities -- parsing it as a premise was a real bug in the
            # first version of this check.
            cut = prompt.rfind("।")
            body = prompt[:cut + 1] if cut != -1 else ""
            facts = set()
            for sent in body.split("।"):
                toks = sent.split()
                ents = [t for t in toks if t in set(chain)]
                adj = next((t for t in toks if t in POS or t in NEG), None)
                if len(ents) == 2 and adj is not None:
                    a, b = ents
                    facts.add((a, b) if adj in POS else (b, a))
            return facts

        def topo(facts, chain):
            """Kahn's algorithm; None unless the order is unique and total."""
            nodes = set(chain)
            indeg = {n: 0 for n in nodes}
            succ = {n: set() for n in nodes}
            for g, l in facts:
                if l not in succ[g]:
                    succ[g].add(l)
                    indeg[l] += 1
            order, avail = [], [n for n in nodes if indeg[n] == 0]
            while avail:
                if len(avail) > 1:
                    return None
                n = avail.pop()
                order.append(n)
                for m in succ[n]:
                    indeg[m] -= 1
                    if indeg[m] == 0:
                        avail.append(m)
            return order if len(order) == len(nodes) else None

        for r in rows:
            facts = parse_facts(r["prompt"], r["chain"])
            assert len(facts) == r["n_hops"], f"{r['id']}: {facts}"
            order = topo(facts, r["chain"])
            assert order == r["chain"], f"{r['id']}: {order} != {r['chain']}"

            pos_q = r["polarity"] == "pos"
            if r["pattern"] in ("P1", "P2"):
                c = r["candidates"]
                exp = min(c, key=order.index) if pos_q else max(c, key=order.index)
            elif r["pattern"] in ("P3", "P5"):
                exp = order[0] if pos_q else order[-1]
            else:
                exp = order[1] if pos_q else order[-2]
            assert exp == r["answer"], f"{r['id']} {r['pattern']}/{r['polarity']}"
    finally:
        sys.path.remove(path)


@pytest.mark.parametrize("tree", TREES)
def test_entity_pools_are_split_disjoint(tree):
    """A name shared across splits would void the entity holdout."""
    T, E, G, path = load_tree(tree)
    try:
        pools = {k: set(v) for k, v in E.SPLITS.items()}
        assert pools["train"] & pools["val"] == set()
        assert pools["train"] & pools["test"] == set()
        assert pools["val"] & pools["test"] == set()
    finally:
        sys.path.remove(path)


def test_entity_pools_are_independent_but_may_share_names():
    """Independence does NOT require name-disjoint pools, and enforcing it hurt.

    An earlier version asserted the two pools shared no name at all. That is not
    what the independence constraint asks for -- separate corpora, tokenizers,
    weights and data files already satisfy it -- and enforcing it pushed the
    Nepali pool into common nouns (mountain, sky, fire) because most South Asian
    given names are shared across both languages. That produced absurd prompts
    and a spurious 0.79-vs-0.21 gap between the models.

    What still must hold: the two pools are built and stored SEPARATELY, and are
    the same size so pool size cannot confound the H-vs-L comparison.
    """
    sizes = {}
    for tree in TREES:
        T, E, G, path = load_tree(tree)
        try:
            sizes[tree] = {k: len(v) for k, v in E.SPLITS.items()}
        finally:
            sys.path.remove(path)
    assert sizes["hindi"] == sizes["nepali"], sizes


@pytest.mark.parametrize("tree", TREES)
def test_held_out_relations_never_appear_in_train(tree):
    """Anti-leakage axis 2: two relations are test-only, by construction."""
    T, E, G, path = load_tree(tree)
    try:
        train = G.build("train", 500, seed=7)
        assert T.HELD_OUT_RELATIONS
        assert not any(r["relation"] in T.HELD_OUT_RELATIONS for r in train)
        test = G.build("test", 500, seed=7)
        assert any(r["relation"] in T.HELD_OUT_RELATIONS for r in test)
    finally:
        sys.path.remove(path)


@pytest.mark.parametrize("tree", TREES)
def test_ood_hop_count_is_strictly_beyond_training(tree):
    """Anti-leakage axis 3: the probe must be LONGER than anything trained.

    Asserted as an invariant rather than a fixed number. The first version hard
    coded "ood == 3 hops, train max 2" and broke the moment P4 was widened to
    four entities to remove a shortcut -- the constant was right for one dataset
    version, the invariant is right for all of them.
    """
    T, E, G, path = load_tree(tree)
    try:
        train_hops = {r["n_hops"] for r in G.build("train", 800, seed=7)}
        ood_hops = {r["n_hops"] for r in G.build("ood", 200, seed=7)}
        assert len(ood_hops) == 1, f"ood should be one hop count, got {ood_hops}"
        assert min(ood_hops) > max(train_hops), (
            f"ood {ood_hops} must exceed train {train_hops}")
    finally:
        sys.path.remove(path)


@pytest.mark.parametrize("tree", TREES)
def test_ordinal_pattern_is_not_solvable_by_naming_the_repeated_entity(tree):
    """P4 must not be answerable without comparing anything.

    v1's P4 used three entities, where second-most and second-least are both the
    middle -- which is exactly the entity appearing in both premises. "Say the
    repeated word" answered 100% of them and the model scored 0.972 without
    reasoning. Every label was correct, so no other test could catch it; the
    TASK was degenerate, not the data.
    """
    from collections import Counter
    T, E, G, path = load_tree(tree)
    try:
        rows = [r for r in G.build("train", 3000, seed=7) if r["pattern"] == "P4"]
        assert rows, "no P4 examples generated"
        hits = 0
        for r in rows:
            cut = r["prompt"].rfind("।")
            body = r["prompt"][:cut + 1] if cut != -1 else r["prompt"]
            counts = Counter(w for w in body.split() if w in set(r["chain"]))
            hits += counts.most_common(1)[0][0] == r["answer"]
        rate = hits / len(rows)
        # The bar is DETERMINISM, not "no structure at all". Asking for a
        # non-extreme rank necessarily makes the answer one of the doubly
        # mentioned middle entities, so this rule legitimately reaches ~0.5 on a
        # 4-chain and choosing between them still needs the chain direction.
        # What must never recur is v1's situation, where the rule identified a
        # UNIQUE answer and scored 1.000, leaving nothing to reason about.
        assert rate < 0.75, (
            f"{tree}: 'name the repeated entity' scores {rate:.3f} on P4 -- "
            "near-deterministic, so the pattern is degenerate again")
        # and polarity must actually change the answer
        assert any(r["chain"][1] != r["chain"][-2] for r in rows)
    finally:
        sys.path.remove(path)


@pytest.mark.parametrize("tree", TREES)
def test_generation_is_deterministic(tree):
    """Same (split, n, seed) must reproduce the dataset exactly."""
    T, E, G, path = load_tree(tree)
    try:
        assert G.build("train", 100, seed=99) == G.build("train", 100, seed=99)
    finally:
        sys.path.remove(path)


@pytest.mark.parametrize("tree", TREES)
def test_surface_heuristics_stay_near_chance(tree):
    """A first/last-mention guesser must not beat chance, or scores are hollow."""
    T, E, G, path = load_tree(tree)
    try:
        rows = G.build("train", 3000, seed=1337)
        chance = sum(1 / len(r["candidates"]) for r in rows) / len(rows)
        for key in ("first_mention", "last_mention"):
            acc = sum(r["answer"] == r[key] for r in rows) / len(rows)
            assert acc < chance + 0.05, f"{tree} {key}={acc:.3f} vs chance {chance:.3f}"
    finally:
        sys.path.remove(path)


@pytest.mark.parametrize("tree", TREES)
def test_examples_tokenize_without_unk(tree):
    """An UNK in the finetuning data would test the tokenizer, not reasoning."""
    spm = pytest.importorskip("sentencepiece")
    T, E, G, path = load_tree(tree)
    try:
        code = {"hindi": "hi", "nepali": "ne"}[tree]
        mp = REPO / tree / "tokenizer" / f"{code}_bpe_16000.model"
        if not mp.exists():
            pytest.skip(f"{mp} missing")
        sp = spm.SentencePieceProcessor(model_file=str(mp))
        for r in G.build("train", 300, seed=5):
            for field in ("prompt", "target_answer_only", "target_with_trace"):
                ids = sp.encode(r[field])
                assert sp.unk_id() not in ids, f"{tree} UNK in {field}: {r[field]}"
    finally:
        sys.path.remove(path)


@pytest.mark.parametrize("tree", TREES)
def test_every_entity_name_is_a_single_token(tree):
    """Guards a confound that already bit once and was invisible.

    The first entity pools were hand-picked without checking tokenization: 83%
    of TRAIN names were single tokens against 36% of TEST names, so held-out
    examples were systematically harder to SPELL, independent of the reasoning.
    A finetuned model duly garbled multi-token names it had to copy
    (संतोष -> सोस), and the held-out accuracy that resulted was measuring the
    wrong thing entirely.

    Requiring one token everywhere makes copying an entity a single-step
    operation, so the metric reflects the comparison rather than the spelling.
    """
    spm = pytest.importorskip("sentencepiece")
    T, E, G, path = load_tree(tree)
    try:
        code = {"hindi": "hi", "nepali": "ne"}[tree]
        mp = REPO / tree / "tokenizer" / f"{code}_bpe_16000.model"
        if not mp.exists():
            pytest.skip(f"{mp} missing")
        sp = spm.SentencePieceProcessor(model_file=str(mp))
        for split, names in E.SPLITS.items():
            bad = {n: sp.encode(n, out_type=str) for n in names
                   if len(sp.encode(n)) != 1}
            assert not bad, f"{tree}/{split} multi-token names: {bad}"
    finally:
        sys.path.remove(path)


def test_entity_pool_sizes_match_across_languages():
    """H-vs-L must differ only in language, so pool sizes are held equal.

    A larger entity vocabulary on one side would be a second uncontrolled
    variable in the comparison the final report has to make.
    """
    sizes = {}
    for tree in TREES:
        T, E, G, path = load_tree(tree)
        try:
            sizes[tree] = {k: len(v) for k, v in E.SPLITS.items()}
        finally:
            sys.path.remove(path)
    assert sizes["hindi"] == sizes["nepali"], sizes


@pytest.mark.parametrize("tree", TREES)
def test_batched_decoding_matches_unbatched(tree):
    """Batching must not change a single prediction.

    This caught a bug that silently corrupted every reported accuracy. The
    evaluator LEFT-padded batches, but `forward` takes no attention mask and the
    positions are learned ABSOLUTE -- so padding both let the model attend to
    pad tokens and shifted every real token to a position index it was never
    trained on. On training examples where the teacher-forced answer probability
    was 1.0000, greedy decoding scored 1.000 unbatched, 0.583 at batch 8 and
    0.517 at batch 60. The model was perfect; the padding destroyed the
    measurement, and because training and evaluation shared the function the
    error was invisible in every cross-check.

    Right-padding is correct here: real tokens keep positions 0..n-1 and a
    causal mask means position t never attends to a pad at t' > t.
    """
    torch = pytest.importorskip("torch")
    spm = pytest.importorskip("sentencepiece")
    T, E, G, path = load_tree(tree)
    try:
        code = {"hindi": "hi", "nepali": "ne"}[tree]
        mp = REPO / tree / "tokenizer" / f"{code}_bpe_16000.model"
        if not mp.exists():
            pytest.skip("tokenizer missing")
        sys.path.insert(0, str(REPO / tree / "eval"))
        sys.path.insert(0, str(REPO / tree / "model"))
        for m in ("reasoning_metrics", "transformer"):
            sys.modules.pop(m, None)
        from reasoning_metrics import closed_set_predictions, greedy_answers
        from transformer import ModelConfig, TransformerLM

        sp = spm.SentencePieceProcessor(model_file=str(mp))
        torch.manual_seed(0)
        # A small random model is enough: the property under test is about
        # padding and positions, not about the weights being any good.
        cfg = ModelConfig(vocab_size=16000, d_model=64, n_layers=2, n_heads=4,
                          d_ff=256, max_seq_len=128, emb_dropout=0.0,
                          attn_dropout=0.0, resid_dropout=0.0, lang_code=code)
        model = TransformerLM(cfg).eval()
        rows = G.build("train", 24, seed=3)
        dev = torch.device("cpu")
        assert greedy_answers(model, sp, rows, dev, batch_size=1) == \
               greedy_answers(model, sp, rows, dev, batch_size=16)
        assert closed_set_predictions(model, sp, rows, dev, batch_size=1) == \
               closed_set_predictions(model, sp, rows, dev, batch_size=16)
    finally:
        sys.path.remove(path)
