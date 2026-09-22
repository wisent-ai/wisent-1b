<!-- wisent-banner:start -->
<p align="center">
  <img src="assets/readme-banner.webp" alt="wisent-1b by Wisent" width="100%">
</p>
<!-- wisent-banner:end -->

<!-- wisent-readme-signals:start -->
[![Source](https://img.shields.io/badge/GitHub-Source-181717?logo=github)](https://github.com/wisent-ai/wisent-1b) [![Issues](https://img.shields.io/badge/GitHub-Issues-181717?logo=github)](https://github.com/wisent-ai/wisent-1b/issues) [![Wisent](https://img.shields.io/badge/Wisent-Website-0B0B0B)](https://wisent.com) [![Discord](https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white)](https://discord.gg/qRjpkthq54) [![LinkedIn](https://img.shields.io/badge/LinkedIn-Follow-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/company/wisent-ai/) [![X](https://img.shields.io/badge/X-Follow-000000?logo=x&logoColor=white)](https://x.com/wisentai) [![Enterprise](https://img.shields.io/badge/Enterprise-Book%20a%20call-0B0B0B?logo=calendly)](https://calendly.com/lbartoszcze)
<!-- wisent-readme-signals:end -->

# Rej-1B

A Concept-First Model. Interpretable and Steerable by Design.

Machine learning interpretability tries to untangle computations present after
training but struggles due to superposition, concept entanglement and
steerability misidentification. The Rej family of models uses concepts as a
separate element of the model architecture. It reads from tokens, updates itself
over layers and activations and writes back into the generation stream. Every
token can be assigned a specific score and be manipulated into desired states.

Representation-Native Models. Designed for Human Control.

> **Key idea:** concepts are not a post-hoc decomposition of hidden states; they are a separate computational state that reads from tokens, updates itself across layers, and writes back into generation.

Documentation: [Rej-1B model architecture and runtime](https://wisent.com/docs/models/wisent-1b)

## What's inside

- `rej_1b/model/` — `RejRNM` and `RejLayer` implementing the dual-stream architecture.
- `rej_1b/model_v2/` — `RejRNMv2`, an advanced geometry-native version (subspaces, probabilistic concepts, non-linear cells, manifold decoder).
- `rej_1b/config.py` — `RejConfig` / `RejConfigV2`, plus factory helpers.
- `rej_1b/generate/` — controlled generation for v1 (`generate`) and v2 (`generate_v2`).
- `rej_1b/train/` — causal language-modeling training utilities for v1 and v2.
- `rej_1b/control.py` — lightweight helpers for concept alignment and control fine-tuning.
- `scripts/demo_toy.py` — end-to-end demo of v1 concept control on synthetic data.
- `scripts/demo_geometric.py` — end-to-end demo of v2 geometric concept control.
- `rej_1b/cli/` — the two console scripts, `rej-1b-train` and `rej-1b-generate`.
- `tests/` — unit tests.

## Install

```bash
cd wisent-1b
pip install -e .
```

## Quick demo

Run the toy demo to see a tiny Rej model learn that `truthfulness=+2.0` and `truthfulness=-2.0` produce different continuations for the same prompt:

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


## Documentation

- [Architecture](docs/architecture.md) — the dual stream, the concept state,
  and the geometric version that bakes subspaces into the model.
- [Training](docs/training.md) — pretraining, the aligned and multilingual
  steps, and the commands that run them.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -v
```

(The environment may have conflicting pytest plugins; disabling autoload avoids unrelated import errors.)

## Status

This is a reference implementation of the architecture described in the manuscript at [wisent-ai/wisent-1b-paper](https://github.com/wisent-ai/wisent-1b-paper) (`neurips_2024.tex`). It contains no pretrained 1B weights — only the model definition, training code, and a working toy demo. Scaling to 1B+ parameters requires the data pipeline and compute described in the paper.

## Citation

```bibtex
@article{rej2025,
  title={Rej-1B: A Representation-Native Language Model with Explicit Concept Control},
  author={Bartoszcze, Lukasz and Towarek, Jakub},
  year={2025}
}
```