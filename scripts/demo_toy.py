"""Small end-to-end demo of Wisent-1B on synthetic controlled data.

This demo shows the key architectural claim: by injecting concept controls
during training, the model learns to map a named concept to a behavior, and
that same control can be applied at inference time to steer generation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from wisent_1b.config import wisent_tiny_config
from wisent_1b.generate import generate
from wisent_1b.model import WisentRNM
from wisent_1b.tokenizer import WisentTokenizer
from wisent_1b.utils import get_device


def build_controlled_dataset(tokenizer, n_examples: int = 400):
    """Build a dataset where truthfulness controls the next token.

    Input is always "the sky is ".
    - truthfulness=+2.0 -> target "blue"
    - truthfulness=-2.0 -> target "green"
    """
    prompt = "the sky is "
    pos_target = "blue"   # 4 chars, "truthful" continuation
    neg_target = "gray"   # 4 chars, "untruthful" continuation

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
    """Collate token sequences and control vectors."""
    tokens, controls = zip(*batch)
    max_len = max(len(t) for t in tokens)
    padded = torch.full((len(tokens), max_len), pad_token_id, dtype=torch.long)
    for i, seq in enumerate(tokens):
        padded[i, : len(seq)] = seq

    control_matrix = torch.zeros(len(tokens), n_named, dtype=torch.float32)
    for i, ctrl in enumerate(controls):
        for name, value in ctrl.items():
            control_matrix[i, concept_names.index(name)] = value

    return padded, control_matrix


def controlled_train_step(model, batch_tokens, batch_controls, optimizer, device):
    """One training step with concept controls injected."""
    model.train()
    batch_tokens = batch_tokens.to(device)
    batch_controls = batch_controls.to(device)
    optimizer.zero_grad()

    outputs = model(batch_tokens, controls=batch_controls)
    logits = outputs["logits"]

    # Standard next-token prediction loss.
    loss = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        batch_tokens[:, 1:].reshape(-1),
        ignore_index=0,  # pad id
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


def main():
    # Use CPU for the tiny demo: it is fast enough and avoids MPS overhead.
    device = torch.device("cpu")
    print(f"Demo running on {device}")

    # Slightly larger than the unit-test tiny config so concept control is stable.
    config = wisent_tiny_config()
    config.d_model = 128
    config.n_layers = 4
    config.n_heads = 4
    config.n_concepts = 8
    config.d_concept = 64

    tokenizer = WisentTokenizer(vocab_size=config.vocab_size)

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

    model = WisentRNM(config).to(device)
    print(f"Demo model parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    print("\nTraining demo model with truthfulness controls...")

    num_steps = 200
    iterator = iter(dataloader)
    losses = []
    for step in range(num_steps):
        try:
            batch_tokens, batch_controls = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch_tokens, batch_controls = next(iterator)

        loss = controlled_train_step(model, batch_tokens, batch_controls, optimizer, device)
        losses.append(loss)
        if (step + 1) % 50 == 0:
            avg_loss = sum(losses[-50:]) / len(losses[-50:])
            print(f"Step {step + 1}/{num_steps} | avg loss: {avg_loss:.4f}")

    prompt = "the sky is "
    print("\n--- Greedy generation (no controls) ---")
    out = generate(
        model, tokenizer, prompt,
        max_new_tokens=10, do_sample=False,
        eos_token_id=tokenizer.eos_token_id, device=device,
    )
    print(repr(out.text))

    print("\n--- Greedy generation with truthfulness=+2.0 ---")
    out = generate(
        model, tokenizer, prompt,
        controls={"truthfulness": 2.0},
        max_new_tokens=10, do_sample=False,
        eos_token_id=tokenizer.eos_token_id, device=device,
    )
    print(repr(out.text))

    print("\n--- Greedy generation with truthfulness=-2.0 ---")
    out = generate(
        model, tokenizer, prompt,
        controls={"truthfulness": -2.0},
        max_new_tokens=10, do_sample=False,
        eos_token_id=tokenizer.eos_token_id, device=device,
    )
    print(repr(out.text))

    print("\nDemo complete.")


if __name__ == "__main__":
    main()
