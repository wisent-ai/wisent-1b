"""CLI entry point for pretraining Rey-1B."""
from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from rey_1b.config import ReyConfig
from rey_1b.model import ReyRNM
from rey_1b.tokenizer import ReyTokenizer
from rey_1b.train import TokenDataset, collate_fn, train
from rey_1b.utils import get_device, save_checkpoint


def load_text_corpus(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    parser = argparse.ArgumentParser(description="Pretrain a Rey-1B model.")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file.")
    parser.add_argument("--data", type=str, required=True, help="Path to text corpus.")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Checkpoint dir.")
    parser.add_argument("--seq_length", type=int, default=256, help="Sequence length.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of training steps.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda/mps).")
    parser.add_argument("--save_every", type=int, default=500, help="Checkpoint frequency.")
    args = parser.parse_args()

    config = ReyConfig.from_json(args.config)
    tokenizer = ReyTokenizer(vocab_size=config.vocab_size)

    print(f"Loading data from {args.data}")
    text = load_text_corpus(args.data)
    token_ids = tokenizer.encode(text)
    dataset = TokenDataset([token_ids], seq_length=args.seq_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id=tokenizer.pad_token_id),
    )

    device = get_device(args.device)
    print(f"Using device: {device}")

    model = ReyRNM(config).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def save_fn(step: int):
        path = save_checkpoint(model, optimizer, step, args.output_dir)
        print(f"Saved checkpoint to {path}")

    train(
        model=model,
        dataset=dataloader,
        optimizer=optimizer,
        device=device,
        num_steps=args.num_steps,
        log_every=50,
        save_every=args.save_every,
        save_fn=save_fn,
    )

    final_path = save_checkpoint(model, optimizer, args.num_steps, args.output_dir)
    print(f"Training complete. Final checkpoint: {final_path}")

    # Save config alongside checkpoint for easy loading.
    config.save_json(os.path.join(args.output_dir, "config.json"))


if __name__ == "__main__":
    main()
