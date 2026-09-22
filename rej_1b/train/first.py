"""Training the first model: one step, and the loop over a dataset."""
from __future__ import annotations

from typing import Iterable, Iterator, Optional

import torch
from tqdm import tqdm

from ..model import RejRNM
from .data import DEFAULT_LOG_EVERY, compute_lm_loss


def train_step(
    model: RejRNM,
    batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    carry_passes: int = 1,
) -> float:
    """Single training step. Returns loss value.

    Teacher forcing runs a whole sequence in one parallel pass, so nothing in
    an ordinary step is a previous step and the cross-step carry never fires:
    its gate would receive no gradient and the model would learn to ignore a
    channel it was never shown. `carry_passes > 1` runs the same batch again
    with the concept state the previous pass ended on, which is what puts the
    fusion on the gradient path. The loss is the mean over passes.

    The carried state is detached between passes. Backpropagating through the
    carry would grow the graph with every pass for a signal that arrives at the
    next pass's input, and one forward and one backward per pass is what makes
    a mixed schedule affordable.
    """
    if carry_passes < 1:
        raise ValueError("carry_passes must be at least 1")

    carry = model.config.carry_concept_state
    if carry_passes > 1 and not carry:
        raise ValueError(
            "carry_passes > 1 requires carry_concept_state=True on the model config"
        )

    model.train()
    batch = batch.to(device)
    optimizer.zero_grad()

    state: torch.Tensor | None = None
    pass_losses = []
    for _ in range(carry_passes):
        outputs = model(batch, concept_state=state, return_concept_state=carry)
        pass_losses.append(compute_lm_loss(outputs["logits"], batch))
        if carry:
            state = outputs["concept_state"].detach()

    loss = torch.stack(pass_losses).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


def train(
    model: RejRNM,
    dataset: Iterable[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_steps: int,
    log_every: int = DEFAULT_LOG_EVERY,
    save_every: int | None = None,
    save_fn=None,
    carry_passes: int = 1,
    carry_fraction: float = 1.0,
) -> List[float]:
    """Train a RejRNM model on a token dataset.

    Args:
        model: the model to train.
        dataset: iterator yielding token-id batches.
        optimizer: optimizer.
        device: device.
        num_steps: total training steps.
        log_every: how often to print loss.
        save_every: how often to call save_fn(step).
        save_fn: optional callable(step) invoked for checkpointing.
        carry_passes: passes per step on the steps that use the cross-step
            concept carry. Requires carry_concept_state on the model config.
        carry_fraction: share of steps that run `carry_passes` instead of one.
            A mixed schedule is deliberate: every step paying for a second
            forward and backward doubles the cost of training, and the carry
            has to be learned, not made the only regime the model ever sees.

    Returns:
        list of loss values per step.
    """
    if not 0.0 <= carry_fraction <= 1.0:
        raise ValueError("carry_fraction must be between 0.0 and 1.0")

    # One carried step every `carry_interval` steps. Zero means never.
    carry_interval = 0 if carry_fraction <= 0.0 else max(1, round(1.0 / carry_fraction))

    model.to(device)
    losses = []
    iterator = iter(dataset)

    pbar = tqdm(range(num_steps), desc="Training")
    for step in pbar:
        try:
            batch = next(iterator)
        except StopIteration:
            break

        # Integer arithmetic rather than a float accumulator, so a resumed run
        # puts the carried steps in the same places as the first one.
        use_carry = (
            carry_passes > 1
            and carry_interval > 0
            and (step + 1) % carry_interval == 0
        )
        loss = train_step(
            model,
            batch,
            optimizer,
            device,
            carry_passes=carry_passes if use_carry else 1,
        )
        losses.append(loss)
        pbar.set_postfix({"loss": f"{loss:.4f}"})

        if log_every > 0 and (step + 1) % log_every == 0:
            print(f"Step {step + 1}/{num_steps} | loss: {loss:.4f}")

        if save_every is not None and save_fn is not None and (step + 1) % save_every == 0:
            save_fn(step + 1)

    return losses

