# Wisent-1B

A reference implementation of **Wisent-1B**, a Representation-Native Language Model (RNM) with an explicit concept stream.

> **Key idea:** concepts are not a post-hoc decomposition of hidden states; they are a separate computational state that reads from tokens, updates itself across layers, and writes back into generation.

## What's inside

- `wisent_1b/model.py` — `WisentRNM` and `WisentLayer` implementing the dual-stream architecture.
- `wisent_1b/config.py` — `WisentConfig`, plus `wisent_1b_config()` and `wisent_tiny_config()`.
- `wisent_1b/generate.py` — controlled generation with named concept controls and concept-trace output.
- `wisent_1b/train.py` — causal language-modeling training utilities.
- `wisent_1b/control.py` — lightweight helpers for concept alignment and control fine-tuning.
- `scripts/demo_toy.py` — end-to-end demo showing concept control on synthetic data.
- `scripts/train.py` / `scripts/generate.py` — CLI entry points.
- `tests/` — unit tests.

## Install

```bash
cd wisent-1b
pip install -e .
```

## Quick demo

Run the toy demo to see a tiny Wisent model learn that `truthfulness=+2.0` and `truthfulness=-2.0` produce different continuations for the same prompt:

```bash
python scripts/demo_toy.py
```

Expected output (approximate):

```text
--- Greedy generation (no controls) ---
'the sky is blue'

--- Greedy generation with truthfulness=+2.0 ---
'the sky is blue'

--- Greedy generation with truthfulness=-2.0 ---
'the sky is gray'
```

## Architecture overview

Each `WisentLayer` maintains two streams:

1. **Token stream** — standard causal self-attention over tokens.
2. **Concept stream** — `K` concept slots of dimension `d_concept`.

Per layer:

```
tokens  ← tokens + CausalSelfAttn(tokens)
concepts ← concepts + CrossAttn(concepts → tokens)
concepts ← concepts + SelfAttn(concepts)
concepts ← concepts + ConceptFFN(concepts)
tokens  ← tokens + gate * CrossAttn(tokens → concepts)
tokens  ← tokens + TokenFFN(tokens)
```

The first `n_named_concepts` slots are exposed as the named control plane, e.g. `truthfulness`, `uncertainty`, `refusal`, `code_mode`. The remaining slots are latent concept dimensions.

Controls are applied in two ways:

1. **Direct token-level control embedding** — a stable bootstrap path that guarantees concept controls reach the token stream from the first layer.
2. **Concept-stream scaling** — named concept embeddings are scaled by the scalar control magnitudes at the input to the concept stream.

This dual-path design keeps training stable while preserving the representation-native concept stream.

## Training

### Pretraining

```bash
python scripts/train.py \
  --config configs/wisent_1b.json \
  --data corpus.txt \
  --output_dir checkpoints \
  --num_steps 10000 \
  --batch_size 8 \
  --seq_length 512
```

### Controlled generation

```bash
python scripts/generate.py \
  --checkpoint checkpoints/checkpoint_step_10000.pt \
  --prompt "Explain quantum computing." \
  --controls "truthfulness=1.5,refusal=-0.5" \
  --max_new_tokens 100 \
  --trace
```

### Python API

```python
from wisent_1b import WisentRNM, WisentTokenizer, generate, wisent_1b_config

config = wisent_1b_config()
model = WisentRNM(config)
tokenizer = WisentTokenizer(vocab_size=config.vocab_size)

out = generate(
    model,
    tokenizer,
    prompt="The capital of France is",
    controls={"truthfulness": 1.2, "uncertainty": -0.3},
    max_new_tokens=20,
    return_concept_trace=True,
)

print(out.text)
print(out.concept_trace["truthfulness"])  # per-layer, per-token concept state
```

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -v
```

(The environment may have conflicting pytest plugins; disabling autoload avoids unrelated import errors.)

## Status

This is a reference implementation of the architecture described in `research/wisent-1b/neurips_2024.tex`. It contains no pretrained 1B weights — only the model definition, training code, and a working toy demo. Scaling to 1B+ parameters requires the data pipeline and compute described in the paper.

## Citation

```bibtex
@article{wisent2025,
  title={Wisent-1B: A Representation-Native Language Model with Explicit Concept Control},
  author={Bartoszcze, Lukasz and Towarek, Jakub},
  year={2025}
}
```
