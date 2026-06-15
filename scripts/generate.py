"""CLI entry point for controlled generation with Wisent-1B."""
from __future__ import annotations

import argparse
import json

from wisent_1b.generate import generate
from wisent_1b.tokenizer import WisentTokenizer
from wisent_1b.utils import get_device, load_checkpoint


def parse_controls(controls_str: str | None) -> dict:
    """Parse 'truthfulness=1.2,refusal=-0.5' into a dict."""
    if not controls_str:
        return {}
    result = {}
    for part in controls_str.split(","):
        name, value = part.split("=")
        result[name.strip()] = float(value.strip())
    return result


def main():
    parser = argparse.ArgumentParser(description="Generate text with Wisent-1B.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint.")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt.")
    parser.add_argument("--controls", type=str, default=None, help="Concept controls, e.g. truthfulness=1.2,refusal=-0.5")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Tokens to generate.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling.")
    parser.add_argument("--top_p", type=float, default=None, help="Nucleus sampling.")
    parser.add_argument("--trace", action="store_true", help="Return concept trace.")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda/mps).")
    args = parser.parse_args()

    device = get_device(args.device)
    model = load_checkpoint(args.checkpoint, device=device)
    tokenizer = WisentTokenizer(vocab_size=model.config.vocab_size)

    controls = parse_controls(args.controls)
    if controls:
        print(f"Applying controls: {json.dumps(controls, indent=2)}")

    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        controls=controls or None,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=tokenizer.eos_token_id,
        return_concept_trace=args.trace,
        device=device,
    )

    print("\n--- Generated text ---")
    print(output.text)

    if args.trace and output.concept_trace:
        print("\n--- Concept trace (last-layer norms) ---")
        for name, trace in output.concept_trace.items():
            # trace is a list of (1, d_concept) tensors, one per generated token.
            norms = [t.squeeze().norm().item() for t in trace]
            avg = sum(norms) / len(norms) if norms else 0.0
            print(f"  {name}: avg norm = {avg:.4f}")


if __name__ == "__main__":
    main()
