"""Surface realisations for the reasoning dataset — Nepali (T3.3).

Duplicated verbatim into every language tree; never imports from a sibling.
The Nepali and Hindi template sets were written INDEPENDENTLY from each
language's own corpus evidence — neither is a translation of the other.

EVERY CONSTRUCTION HERE WAS COUNTED IN OUR OWN PRETRAINING CORPUS FIRST
----------------------------------------------------------------------
`mine_constructions.py` measured each candidate over 40,000 training documents
before it was allowed into a template. Counts are recorded on each relation
below so the report shows the evidence rather than asserting naturalness.
Candidates REJECTED on that evidence: दुब्लो ("thin", 10 occurrences total),
गह्रौँ and हलुको (13 and 12) — attested too rarely to be how this corpus talks.

THE COMPARATIVE FRAME
---------------------
Nepali marks the standard of comparison with भन्दा, which FOLLOWS the standard:

    राम श्याम भन्दा अग्लो छ।      "Ram is taller than Shyam."
    [subject] [standard] भन्दा [adj] छ।

Attested directly in our corpus, e.g. "तामा भन्दा हल्का" ("lighter than
copper"). The superlative is सबैभन्दा ("than all"), and the ordinal superlative
stacks दोस्रो on it — दोस्रो सबैभन्दा ठूलो, "second biggest" — a form found in
our own corpus, not invented here.

A NOTE ON THE COMPARATIVE FRAME'S RARITY VERSUS HINDI
-----------------------------------------------------
Nepali's भन्दा-frame counts run consistently below Hindi's से-frame counts for
the same attribute (ठूलो 179 vs बड़ा 353; अग्लो 27 vs लंबा 40), even though the
Nepali corpus is the larger of the two. The bare adjectives are well attested
(ठूलो 7,693), so the model knows the words; it is the explicit comparative
construction that is rarer. That is a genuine property of the lower-resource
corpus and is carried into the Phase-3 report rather than smoothed over.

WHY EVERY RELATION CARRIES AN ANTONYM
-------------------------------------
So the SAME premises can be asked in both directions: given "A is taller than
B", "who is taller?" answers A but "who is shorter?" answers B. Without that, a
model could score well by always naming the first-mentioned entity. Combined
with premise-order shuffling in the generator, this is what keeps the
first-mention and last-mention baselines near chance.
"""

from __future__ import annotations

# (key, positive adj, negative adj, attribute gloss, corpus counts)
# `se` = occurrences of "भन्दा <adj>" (the than-frame), `bare` = the adjective
# anywhere. Both measured over this tree's own train split.
RELATIONS = [
    # key         +adj       -adj        gloss         +se  +bare   -se  -bare
    ("height",    "अग्लो",   "होचो",      "height",      27,   923,    6,    99),
    ("size",      "ठूलो",    "सानो",      "size",       179,  7693,   36,  2617),
    ("weight",    "भारी",    "हलुका",     "weight",       2,   896,    1,    80),
    ("price",     "महँगो",   "सस्तो",     "price",        2,   186,    6,   267),
    ("age",       "पुरानो",  "नयाँ",      "age",         39,  2300,    8,  8682),
    ("strength",  "बलियो",   "कमजोर",     "strength",     3,   682,    6,   656),
    ("thickness", "मोटो",    "पातलो",     "thickness",    1,   124,    2,   299),
    # --- held out of TRAIN entirely; test-only (anti-leakage axis 2) ---------
    ("wealth",    "धनी",     "गरिब",      "wealth",       4,   574,    0,   208),
    ("speed",     "छिटो",    "ढिलो",      "speed",       28,   726,    8,   330),
]

# Relations the model never sees in training. The SAME two attributes are held
# out in the Hindi tree, so the H-vs-L comparison is structurally matched even
# though the adjectives themselves are unrelated words.
HELD_OUT_RELATIONS = {"wealth", "speed"}

BY_KEY = {r[0]: r for r in RELATIONS}


def relation(key: str) -> dict:
    """Look up one relation as a dict, including its corpus evidence."""
    k, pos, neg, gloss, p_se, p_bare, n_se, n_bare = BY_KEY[key]
    return {"key": k, "pos": pos, "neg": neg, "gloss": gloss,
            "pos_se": p_se, "pos_bare": p_bare, "neg_se": n_se, "neg_bare": n_bare}


# --------------------------------------------------------------------------
# Surface realisations.
#
# Delimiters are limited to । ? : — each is a SINGLE token under our tokenizer
# and abundant in the pretraining corpus. Newline is deliberately absent: it
# does NOT survive tokenization ("क\nख" -> ['▁क','▁ख']), so a newline-delimited
# format would carry no structure at all. The spec also freezes the tokenizer,
# so no new special token is available even in principle.
# --------------------------------------------------------------------------

def premise(a: str, b: str, adj: str) -> str:
    """`A B भन्दा ADJ छ।` — A exceeds B on the attribute named by `adj`."""
    return f"{a} {b} भन्दा {adj} छ।"


# Question forms, several per pattern so the model cannot key on one string.
# Ordered and WEIGHTED, not uniformly alternated. Native-speaker review (10 Sep)
# judged both मध्ये कुन and मध्ये को well-formed here -- every entity in this
# dataset is a person, which is exactly the context where को ("who") is
# idiomatic -- but asked that मध्ये कुन be the generic default and that the
# alternation be controlled rather than uncontrolled. Hence explicit weights.
Q_PAIR = [
    "{x} र {y} मध्ये कुन {adj} छ?",   # generic "which of A and B" — the default
    "{x} र {y} मध्ये को {adj} छ?",    # person-oriented "who" — valid: entities are people
    "{x} कि {y}, को {adj} छ?",
]
Q_PAIR_WEIGHTS = [0.5, 0.3, 0.2]
Q_SUPERLATIVE = [
    "सबैभन्दा {adj} को हो?",
    "यीमध्ये सबैभन्दा {adj} को हो?",
    "यी सबैमा {adj} को हो?",
]
# ONE form only, and deliberately so. Native-speaker review (10 Sep) declined to
# approve the stacked superlative "दोस्रो सबैभन्दा {adj}" for mechanical use
# across all nine relations, and the corpus agrees: "दोस्रो ठूलो" occurs 145
# times against 38 for "दोस्रो सबैभन्दा ठूलो", and real usage takes exactly this
# shape -- "दोस्रो अग्लो हिमाल", "दोस्रो लामो नदी". The rejected second variant
# ("...दोस्रो नम्बरमा...") rested on नम्बरमा, attested only 16 times.
#
# Hindi does the OPPOSITE, and that is not an inconsistency: Hindi's corpus
# prefers the stacked form by ~18x (दूसरा सबसे बड़ा 299 vs दूसरा बड़ा 17). The two
# languages genuinely diverge here, which is why the template sets were written
# independently rather than one being translated from the other.
#
# Cost: P4 has a single phrasing where other patterns have two or three. Shipping
# one reviewed form beats shipping a second unreviewed one.
Q_ORDINAL = [
    "दोस्रो {adj} को हो?",
]
ANSWER_PREFIX = " उत्तर:"


def question_pair(x: str, y: str, adj: str, variant: int, u: float = None) -> str:
    """Pick a phrasing. `u` is a uniform draw in [0,1); when given, the weighted
    distribution above is used so मध्ये कुन dominates as the generic form."""
    if u is None:
        return Q_PAIR[variant % len(Q_PAIR)].format(x=x, y=y, adj=adj)
    acc = 0.0
    for form, w in zip(Q_PAIR, Q_PAIR_WEIGHTS):
        acc += w
        if u < acc:
            return form.format(x=x, y=y, adj=adj)
    return Q_PAIR[-1].format(x=x, y=y, adj=adj)


def question_superlative(adj: str, variant: int) -> str:
    return Q_SUPERLATIVE[variant % len(Q_SUPERLATIVE)].format(adj=adj)


def question_ordinal(adj: str, variant: int) -> str:
    return Q_ORDINAL[variant % len(Q_ORDINAL)].format(adj=adj)


def answer(name: str) -> str:
    """Target span. Leading space: SentencePiece marks word starts with ▁."""
    return f"{ANSWER_PREFIX} {name}"


def trace_chain(chain: list[str], rel: dict, positive: bool) -> str:
    """Reasoning trace: restate the chain, then state the derived relation.

    Used for the chain-of-thought arm of the CoT-vs-answer-only A/B. Traces are
    generated from the SAME ground truth as the answer, never post-hoc, so a
    trace can never disagree with its own label.

    THE TRACE IS STATED IN THE QUESTION'S OWN POLARITY.
    `chain` is ranked high-to-low. For a question asking who is *most* X the
    trace walks it in that direction; for one asking who is *least* X the chain
    is walked in reverse using the antonym, so the final line of the trace names
    the answer directly. Rendering every trace in the positive direction would
    make a "who is weakest" question end on "...therefore D is stronger than A",
    forcing an unstated inversion to reach the answer — which would handicap the
    chain-of-thought arm of the A/B for a reason that has nothing to do with
    whether traces help.
    """
    walk = list(chain) if positive else list(reversed(chain))
    adj = rel["pos"] if positive else rel["neg"]
    steps = ", र ".join(f"{a} {b} भन्दा {adj} छ" for a, b in zip(walk, walk[1:]))
    return f"{steps}, त्यसैले {walk[0]} {walk[-1]} भन्दा {adj} छ।"
