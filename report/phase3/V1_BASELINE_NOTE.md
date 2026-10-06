# v1 results, preserved for the before/after comparison

`*_v1.json` and `sft_baseline_v1/` are the FIRST production run, kept unchanged
so the v2 dataset revision can be compared against a fixed reference.

> **Every accuracy on this page was produced by the buggy left-padding decoder**
> (see `README.md` §5.1) and is therefore wrong in absolute terms. They are kept
> because the *argument* below — that v1's P4 was solvable without reasoning —
> rests on the structure of the data, not on the decoder, and because a
> correction is only checkable against what it replaced. Do not quote these
> numbers as results.

## Why v1 is superseded

v1's P4 (ordinal, "second most X") used **three** entities. In a 3-chain
A > B > C the second-most and the second-least are BOTH B, and B is exactly the
entity that appears in both premises. Verified on all 960 P4 test examples:

* answer == the entity mentioned in both premises: **100.0%**
* second-most == second-least: **100.0%**

So P4 was solvable by "name the repeated word", with no comparison at all. The
model duly scored 0.972 on it (0.990 negative / 0.969 positive -- polarity made
no difference, which is the signature of a shortcut rather than reasoning), and
at 16% of the data this inflated the v1 headline.

Excluding P4, v1 Hindi is **0.636 against 0.461 chance**, not the 0.690 headline.

## What changes in v2

1. P4 uses **four** entities, so second-most (B) != second-least (C) and the
   repeated-entity rule no longer picks a unique answer.
2. Training rebalanced toward P3 -- the weakest genuine pattern (0.416 vs 0.333
   chance) and the assignment's own first example, "which is smallest/largest?".
3. Larger entity pools, to weaken the answer prior over a small name set.

Fixing P4 necessarily brings 3-hop chains into training, so the OOD probe moves
from 3 hops to 4. P1/P2/P3 are structurally unchanged and directly comparable
across v1 and v2; P4 and OOD are not, and are reported as changed.

**The v2 headline may be LOWER than v1's.** That is the intended outcome: v1's
number included a 16% slice the model solved without reasoning.
