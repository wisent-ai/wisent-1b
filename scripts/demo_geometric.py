"""Tiny geometric demo for ReyRNMv2.

Trains a tiny v2 model on two opposite completions controlled by the
magnitude of the "truthfulness" concept, then shows generation under
magnitude, direction, uncertainty, and select controls.
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rey_1b.config import rey_tiny_v2_config
from rey_1b.model_v2 import ReyRNMv2
from rey_1b.tokenizer import ReyTokenizer
from rey_1b.generate import generate_v2
from rey_1b.utils import get_device


def build_controlled_dataset(tokenizer, n_examples: int = 400):
    """Build a dataset where truthfulness magnitude steers the answer."""
    prompt = "the sky is "
    pos_target = "blue"
    neg_target = "gray"

    prompt_ids = tokenizer.encode(prompt)
    pos_ids = tokenizer.encode(pos_target)
    neg_ids = tokenizer.encode(neg_target)
    eos = [tokenizer.eos_token_id]

    sequences = []
    controls = []
    for i in range(n_examples):
        if i % 2 == 0:
            seq = prompt_ids + pos_ids + eos
            ctrl = {"truthfulness": 2.0}
        else:
            seq = prompt_ids + neg_ids + eos
            ctrl = {"truthfulness": -2.0}
        sequences.append(torch.tensor(seq, dtype=torch.long))
        controls.append(ctrl)
    return sequences, controls


def collate_with_controls(batch, pad_token_id: int, n_named: int, concept_names: list):
    """Collate token sequences and control vectors for v2."""
    tokens, controls = zip(*batch)
    max_len = max(len(t) for t in tokens)
    padded = torch.full((len(tokens), max_len), pad_token_id, dtype=torch.long)
    for i, seq in enumerate(tokens):
        padded[i, : len(seq)] = seq

    control_matrix = torch.zeros(len(tokens), n_named, dtype=torch.float32)
    for i, ctrl in enumerate(controls):
        for name, value in ctrl.items():
            control_matrix[i, concept_names.index(name)] = value

    return padded, {"magnitude": control_matrix}


def controlled_train_step_v2(model, batch_tokens, batch_controls, optimizer, device):
    """One v2 training step with concept controls injected."""
    model.train()
    batch_tokens = batch_tokens.to(device)
    for k in batch_controls:
        batch_controls[k] = batch_controls[k].to(device)
    optimizer.zero_grad()

    outputs = model(batch_tokens, controls=batch_controls)
    logits = outputs["logits"]
    lm_loss = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        batch_tokens[:, 1:].reshape(-1),
        ignore_index=0,
    )
    kl_loss = outputs.get("kl_loss", torch.tensor(0.0, device=device))
    geometry_loss = outputs.get("geometry_loss", torch.tensor(0.0, device=device))
    loss = (
        lm_loss
        + model.config.kl_weight * kl_loss
        + model.config.geometry_weight * geometry_loss
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {
        "total_loss": loss.item(),
        "lm_loss": lm_loss.item(),
        "kl_loss": kl_loss.item(),
        "geometry_loss": geometry_loss.item(),
    }


def main():
    device = get_device(preferred="cpu")
    print(f"Demo running on {device}")

    config = rey_tiny_v2_config()
    config.use_concept_alignment = False
    config.use_geometry_regularization = False
    config.use_titan_manifold = False
    config.kl_weight = 1e-4
    tokenizer = ReyTokenizer(vocab_size=config.vocab_size)

    sequences, controls = build_controlled_dataset(tokenizer, n_examples=400)
    dataset = list(zip(sequences, controls))
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=lambda batch: collate_with_controls(
            batch,
            pad_token_id=tokenizer.pad_token_id,
            n_named=config.n_named_concepts,
            concept_names=config.named_concepts,
        ),
    )

    model = ReyRNMv2(config).to(device)
    print(f"Demo model parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    print("\nTraining tiny geometric model with truthfulness controls...")

    num_steps = 300
    iterator = iter(dataloader)
    losses = []
    for step in range(num_steps):
        try:
            batch_tokens, batch_controls = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch_tokens, batch_controls = next(iterator)

        metrics = controlled_train_step_v2(model, batch_tokens, batch_controls, optimizer, device)
        losses.append(metrics)
        if (step + 1) % 75 == 0:
            avg_total = sum(m["total_loss"] for m in losses[-75:]) / len(losses[-75:])
            avg_kl = sum(m["kl_loss"] for m in losses[-75:]) / len(losses[-75:])
            avg_geo = sum(m["geometry_loss"] for m in losses[-75:]) / len(losses[-75:])
            print(
                f"Step {step + 1}/{num_steps} | "
                f"avg total: {avg_total:.4f} | avg KL: {avg_kl:.4f} | avg geometry: {avg_geo:.4f}"
            )

    prompt = "the sky is "
    print("\n--- Greedy generation (no controls) ---")
    out = generate_v2(
        model, tokenizer, prompt,
        max_new_tokens=10, do_sample=False,
        eos_token_id=tokenizer.eos_token_id, device=device,
    )
    print(repr(out.text))

    print("\n--- truthfulness magnitude +2.0 ---")
    out = generate_v2(
        model, tokenizer, prompt,
        controls={"magnitude": {"truthfulness": 2.0}},
        max_new_tokens=10, do_sample=False,
        eos_token_id=tokenizer.eos_token_id, device=device,
    )
    print(repr(out.text))

    print("\n--- truthfulness magnitude -2.0 ---")
    out = generate_v2(
        model, tokenizer, prompt,
        controls={"magnitude": {"truthfulness": -2.0}},
        max_new_tokens=10, do_sample=False,
        eos_token_id=tokenizer.eos_token_id, device=device,
    )
    print(repr(out.text))

    print("\n--- direction push in subspace ---")
    out = generate_v2(
        model, tokenizer, prompt,
        controls={"direction": {"truthfulness": [1.0, -1.0, 0.5, 0.0]}},
        max_new_tokens=10, do_sample=False,
        eos_token_id=tokenizer.eos_token_id, device=device,
    )
    print(repr(out.text))

    print("\n--- uncertainty increase ---")
    out = generate_v2(
        model, tokenizer, prompt,
        controls={"uncertainty": {"truthfulness": 2.0}},
        max_new_tokens=10, do_sample=True,
        eos_token_id=tokenizer.eos_token_id, device=device,
    )
    print(repr(out.text))

    print("\n--- select suppression ---")
    out = generate_v2(
        model, tokenizer, prompt,
        controls={"select": {"truthfulness": -10.0}},
        max_new_tokens=10, do_sample=False,
        eos_token_id=tokenizer.eos_token_id, device=device,
    )
    print(repr(out.text))

    print("\nDemo complete.")


if __name__ == "__main__":
    main()
