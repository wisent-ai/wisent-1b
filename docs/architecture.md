# Architecture

## Architecture overview

Each `RejLayer` maintains two streams:

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

`gate` on the write-back line is `concept_to_token_gate`, one scalar per layer,
**initialized to zero**. The token stream therefore starts as a standard
transformer and the gate grows only if the concept stream helps prediction.

That has a consequence worth knowing before measuring anything: on a freshly
constructed model the concept stream cannot reach the token stream at all, so
**no concept intervention changes any output**. Setting `truthfulness=+2.0` on
an untrained `RejRNM` and reading identical logits is the gate at zero, not a
broken control plane. Controllability is measurable once training has opened
the gates, which is why `scripts/demo_toy.py` trains before it steers.

The first `n_named_concepts` slots are exposed as the named control plane, e.g. `truthfulness`, `uncertainty`, `refusal`, `code_mode`. The remaining slots are latent concept dimensions.

Controls are applied in two ways:

1. **Direct token-level control embedding** — a stable bootstrap path that guarantees concept controls reach the token stream from the first layer.
2. **Concept-stream scaling** — named concept embeddings are scaled by the scalar control magnitudes at the input to the concept stream.

This dual-path design keeps training stable while preserving the representation-native concept stream.

### Cross-step concept state

The block above carries the concept state across **depth**. Across generation
**steps** it carries nothing: `_build_initial_concepts` rebuilds the stream from
the learned concept embeddings on every step, so a `K x d_concept` state the
model has just computed is discarded at the step boundary and only the sampled
token survives it. A state that advances one layer per update is a deeper
feedforward computation, not a state that tracks anything over time.

`carry_concept_state` closes that channel. The final concept state of a step is
fused into the initial state of the next one by a gated linear unit over the
pair:

```
c_0(t+1) = c_init(t+1) + sigmoid(W_g [c_init, c_L(t)]) * W_v [c_init, c_L(t)]
```

`W_v` is zero-initialized, so the carry contributes exactly nothing until it is
trained: turning the flag on for a checkpoint written without it changes no
logit. Verified on the tiny config — carried and uncarried forward passes agree
to 0.0 at initialization, and diverge once `W_v` is non-zero and the per-layer
`concept_to_token_gate` is open.

```python
from rej_1b.config import RejConfig, rej_tiny_config

cfg = RejConfig.from_dict({**rej_tiny_config().to_dict(), "carry_concept_state": True})
```

`generate` threads the state through the decoding loop by itself when the flag
is set. Training needs one extra thing: teacher forcing runs a whole sequence in
one parallel pass, so nothing in an ordinary step is a *previous* step and the
fusion never fires — its gradient is `None`. `train_step(..., carry_passes=2)`
runs the batch again with the state the first pass ended on, which is what puts
the fusion on the gradient path, and `train(..., carry_passes=2,
carry_fraction=0.25)` mixes those steps in one in four rather than paying for a
second forward and backward on every step.

The carried state is detached between passes. `RejRNMv2` refuses the flag rather
than ignoring it: its concept state is a Gaussian over subspace coordinates, so
carrying it is a distribution to propagate and needs its own fusion rule.

## RejRNMv2: geometric concepts (advanced)

`RejRNMv2` bakes geometry into the architecture itself, rather than applying scalar steering after training:

- **Subspace concepts** — each concept is a rank-`r` subspace (`basis` + `centroid`) instead of a single vector.
- **Probabilistic concept state** — each concept carries a Gaussian `N(mean, std²)` in subspace coordinates, regularized by a KL term during training.
- **Non-linear concept cells** — MLP-based read/update/write dynamics replace linear cross-attention.
- **Input-dependent router** — each token is assigned a relevance distribution over concepts.
- **Manifold decoder** — subspace coordinates are decoded through a non-linear MLP before being written back to tokens.
- **TITAN-style steering manifold** — each named concept owns multiple learned directions in subspace coordinates; an input-dependent intensity network combines them per layer.
- **Geometry-aware regularization** — biprojection-style loss keeps updated concept embeddings on their subspace manifold.
- **Concept alignment head** — a small head predicts injected control magnitudes from concept states, trained with contrastive supervision.
- **Language-invariant concept objective** — optional loss that aligns concept embeddings of parallel sentences across languages.
- **Control perturbation training** — random control magnitudes are injected during LM training so the model learns a smooth control surface.
- **Geometric controls** — four control modes:
  - `magnitude`: scale concept means.
  - `direction`: add a vector in subspace coordinates.
  - `uncertainty`: increase/decrease concept std.
  - `select`: soft-mask concept activation.

Run the geometric demo:

```bash
python scripts/demo_geometric.py
```

### Python API (v2)

```python
from rej_1b import RejRNMv2, RejTokenizer, generate_v2, rej_tiny_v2_config

config = rej_tiny_v2_config()
model = RejRNMv2(config)
tokenizer = RejTokenizer(vocab_size=config.vocab_size)

out = generate_v2(
    model,
    tokenizer,
    prompt="The sky is",
    controls={
        "magnitude": {"truthfulness": 2.0},
        "direction": {"truthfulness": [1.0, -0.5, 0.0, 0.0]},
    },
    max_new_tokens=20,
    return_concept_trace=True,
)

print(out.text)
print(out.concept_trace["truthfulness"])  # per-layer subspace mean
```

