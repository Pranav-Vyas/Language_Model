"""Surface realisations for the reasoning dataset — Hindi (T3.2).

Duplicated verbatim into every language tree; never imports from a sibling.
The Hindi and Nepali template sets were written INDEPENDENTLY from each
language's own corpus evidence — neither is a translation of the other.

EVERY CONSTRUCTION HERE WAS COUNTED IN OUR OWN PRETRAINING CORPUS FIRST
----------------------------------------------------------------------
`mine_constructions.py` measured each candidate over 40,000 training documents
before it was allowed into a template. Counts are recorded on each relation
below so the report can show the evidence rather than assert naturalness. Two
candidates were REJECTED on that evidence: उथला ("shallow", 1 occurrence in the
than-frame) and संकरा ("narrow", 0) — they are simply not how this corpus talks.

THE COMPARATIVE FRAME
---------------------
Hindi marks the standard of comparison with the ablative से:

    राम श्याम से लंबा है।        "Ram is taller than Shyam."
    [subject] [standard] से [adj] है।

Superlative uses सबसे ("than all"), and the ordinal superlative stacks दूसरा on
top of it — दूसरा सबसे लंबा, "second tallest" — a form found in the corpus
(e.g. "दूसरा सबसे लंबा है" in a Wikipedia sports article), not invented here.

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
# `se` = occurrences of "से <adj>" (the than-frame), `bare` = the adjective
# anywhere. Both measured over 40,000 documents of THIS tree's train split.
RELATIONS = [
    # key         +adj        -adj        gloss          +se  +bare   -se  -bare
    ("height",    "लंबा",     "छोटा",      "height",       40,  2592,  140,  6058),
    ("size",      "बड़ा",      "छोटा",      "size",        353, 13626,  140,  6058),
    ("weight",    "भारी",     "हल्का",     "weight",      101,  2509,   31,   678),
    ("price",     "महंगा",    "सस्ता",     "price",        10,   325,   15,   344),
    ("age",       "पुराना",   "नया",       "age",          48,  3100,  105,  7202),
    ("strength",  "मजबूत",    "कमजोर",     "strength",     70,  1908,   76,   826),
    ("thickness", "मोटा",     "पतला",      "thickness",     7,   546,   12,   585),
    # --- held out of TRAIN entirely; test-only (anti-leakage axis 2) ---------
    ("wealth",    "अमीर",     "गरीब",      "wealth",       13,   968,   19,   707),
    ("speed",     "तेज",      "धीमा",      "speed",        51,  1603,    6,   184),
]

# Relations the model never sees in training. Generalising to these requires
# reading the FRAME ("X Y से ___ है" ⇒ X exceeds Y) rather than memorising which
# specific adjective means "greater".
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
    """`A B से ADJ है।` — A exceeds B on the attribute named by `adj`."""
    return f"{a} {b} से {adj} है।"


# Question forms, several per pattern so the model cannot key on one string.
Q_PAIR = [
    "{x} और {y} में {adj} कौन है?",
    "{x} और {y} में से {adj} कौन है?",
    "{x} या {y}, {adj} कौन है?",
]
Q_SUPERLATIVE = [
    "सबसे {adj} कौन है?",
    "इनमें सबसे {adj} कौन है?",
    "इन सबमें {adj} कौन है?",
]
Q_ORDINAL = [
    "दूसरा सबसे {adj} कौन है?",
    "{adj} होने में दूसरे नंबर पर कौन है?",
]
ANSWER_PREFIX = " उत्तर:"


# Hindi's three pair-forms are all equally idiomatic, so they stay uniform.
# The `u` argument exists for signature parity with the Nepali copy, where a
# native-speaker review asked for a weighted default; it is unused here.
Q_PAIR_WEIGHTS = None


def question_pair(x: str, y: str, adj: str, variant: int, u: float = None) -> str:
    return Q_PAIR[variant % len(Q_PAIR)].format(x=x, y=y, adj=adj)


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
    steps = ", और ".join(f"{a} {b} से {adj} है" for a, b in zip(walk, walk[1:]))
    return f"{steps}, इसलिए {walk[0]} {walk[-1]} से {adj} है।"
