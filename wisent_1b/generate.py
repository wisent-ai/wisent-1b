"""Controlled generation for Wisent-1B."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from .model import WisentRNM
from .tokenizer import WisentTokenizer


@dataclass
class GenerationOutput:
    """Output of controlled generation."""

    text: str
    token_ids: List[int]
    concept_trace: Optional[Dict[str, List[torch.Tensor]]] = None


def _controls_to_tensor(
    controls: Optional[Dict[str, float]],
    named_concepts: List[str],
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Convert a dict of named concept controls to a float tensor."""
    if controls is None or len(controls) == 0:
        return None
    vec = torch.zeros(len(named_concepts), dtype=torch.float32, device=device)
    for name, value in controls.items():
        if name not in named_concepts:
            raise ValueError(
                f"Unknown control concept '{name}'. Available: {named_concepts}"
            )
        vec[named_concepts.index(name)] = float(value)
    return vec.unsqueeze(0)


def _aggregate_concept_trace(
    trace: List[torch.Tensor],
    named_concepts: List[str],
) -> Dict[str, List[torch.Tensor]]:
    """Return per-named-concept trace across layers."""
    return {
        name: [layer_state[:, idx, :].detach().cpu() for layer_state in trace]
        for idx, name in enumerate(named_concepts)
    }


def generate(
    model: WisentRNM,
    tokenizer: WisentTokenizer,
    prompt: str,
    controls: Optional[Dict[str, float]] = None,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    do_sample: bool = True,
    eos_token_id: Optional[int] = None,
    return_concept_trace: bool = False,
    device: torch.device | str = "cpu",
) -> GenerationOutput:
    """Generate text with explicit concept controls.

    Args:
        model: a WisentRNM model.
        tokenizer: a WisentTokenizer.
        prompt: input text.
        controls: mapping from named concept to scalar magnitude.
        max_new_tokens: number of tokens to generate.
        temperature: sampling temperature.
        top_k: if set, restrict sampling to top-k tokens.
        top_p: if set, restrict sampling to nucleus.
        do_sample: if False, use greedy decoding.
        eos_token_id: token id that stops generation.
        return_concept_trace: if True, return per-layer concept states.
        device: device to run on.

    Returns:
        GenerationOutput with generated text, token ids, and optional concept trace.
    """
    model.eval()
    device = torch.device(device)
    model = model.to(device)

    input_ids = tokenizer.encode(prompt)
    generated = list(input_ids)

    controls_t = _controls_to_tensor(controls, model.named_concept_labels, device)

    concept_trace_layers: List[torch.Tensor] = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            inp = torch.tensor([generated[-model.config.max_position_embeddings:]], device=device)
            outputs = model(
                inp,
                controls=controls_t,
                return_concept_trace=return_concept_trace,
            )
            logits = outputs["logits"]
            if return_concept_trace and outputs["concept_trace"] is not None:
                concept_trace_layers.append(outputs["concept_trace"][-1])

            next_token_logits = logits[:, -1, :] / temperature

            if top_k is not None and top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float("-inf")

            if top_p is not None and top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float("-inf")

            probs = F.softmax(next_token_logits, dim=-1)

            if do_sample:
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = torch.argmax(probs, dim=-1).item()

            generated.append(next_token)

            if eos_token_id is not None and next_token == eos_token_id:
                break

    text = tokenizer.decode(generated, skip_special_tokens=True)

    concept_trace = None
    if return_concept_trace:
        concept_trace = _aggregate_concept_trace(
            concept_trace_layers, model.named_concept_labels
        )

    return GenerationOutput(
        text=text,
        token_ids=generated,
        concept_trace=concept_trace,
    )
