"""Learning-rate schedule: linear warmup then cosine decay.

Duplicated verbatim into every language tree; never imports from a sibling.

WHY WARMUP AT ALL, GIVEN WE CHOSE PRE-NORM
------------------------------------------
This is a viva question, so the answer is written down rather than implied.

Pre-norm (`x + Sublayer(LayerNorm(x))`) exists precisely so that deep stacks
train *without* warmup: the residual path is an unmodified identity, so
gradients reach layer 0 without being rescaled at every depth, and the
output variance does not grow with depth the way post-norm's does. Post-norm
genuinely needs warmup to survive its first few hundred steps. We do not.

We keep warmup anyway, as **insurance, not necessity**. The reason is
specific: at step 0 the Adam second-moment estimate `v` is near zero and
poorly conditioned, so the very first updates are effectively divided by a
noisy estimate of the gradient scale. That is an *optimizer* instability,
independent of where the LayerNorms sit. A few hundred warmup steps cost
well under 1% of the run and remove a failure mode we would otherwise only
discover by watching a loss curve diverge on rented GPU time.

So: pre-norm removes the *architectural* need for warmup; warmup remains
worthwhile for the *optimizer*. Both halves of that sentence matter.

THE SHAPE
---------
    lr(t) = peak * t / warmup                       for t < warmup
    lr(t) = min_lr + 0.5*(peak-min_lr)*(1+cos(pi*p))  for t >= warmup
            where p = (t - warmup) / (total - warmup)

Cosine rather than step decay because it anneals smoothly to `min_lr` with no
hand-tuned drop points, and because a run that is killed and resumed lands on
exactly the same LR it would have had -- the schedule is a pure function of
the step number, holding no internal state to restore. That property is what
makes resume exact rather than approximate.
"""

from __future__ import annotations

import math


class WarmupCosineSchedule:
    """Stateless LR schedule: `lr(step)` is a pure function of `step`.

    Holding no mutable state is deliberate. A stateful scheduler must have its
    state serialised into every checkpoint and restored on resume, and a
    mismatch there is silent -- training continues at the wrong LR. Here the
    step number (already in the checkpoint) is sufficient to reconstruct the
    LR exactly, so there is nothing extra to get wrong.
    """

    def __init__(self, peak_lr: float, warmup_steps: int, total_steps: int,
                 min_lr_ratio: float = 0.1):
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if total_steps <= warmup_steps:
            raise ValueError(
                f"total_steps ({total_steps}) must exceed warmup_steps "
                f"({warmup_steps}); otherwise the cosine leg has no length."
            )
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = peak_lr * min_lr_ratio

    def lr_at(self, step: int) -> float:
        """Learning rate for a 0-indexed optimizer step."""
        if step < self.warmup_steps:
            # step+1 so step 0 gets a nonzero LR; a literal 0 would waste the
            # first update entirely.
            return self.peak_lr * (step + 1) / self.warmup_steps
        if step >= self.total_steps:
            # Past the planned end (e.g. training extended by hand) hold the
            # floor rather than letting the cosine turn back upward.
            return self.min_lr
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.peak_lr - self.min_lr) * cosine

    def apply(self, optimizer, step: int) -> float:
        """Set the LR on every param group and return the value used."""
        lr = self.lr_at(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr
