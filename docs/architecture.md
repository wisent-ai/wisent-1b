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
an untrained `RejRnm` and reading identical logits is the gate at zero, not a
broken control plane. Controllability is measurable once training has opened
the gates, which is why the quick start in the README trains before it steers.

The first slots of the concept stream are the named control plane, one per
entry of `named_concepts` in the configuration, e.g. `truthfulness`,
`uncertainty`, `refusal`, `code_mode`. The remaining slots are latent concept
dimensions. The two concept attentions use `concept_heads` heads and the concept
feed-forward is `concept_intermediate_size` wide; the configuration states both.

Controls are applied in two ways:

1. **Direct token-level control embedding** — a stable bootstrap path that guarantees concept controls reach the token stream from the first layer.
2. **Concept-stream scaling** — named concept embeddings are scaled by the scalar control magnitudes at the input to the concept stream.

This dual-path design keeps training stable while preserving the representation-native concept stream.

### Cross-step concept state

The block above carries the concept state across **depth**. Across generation
**steps** it carries nothing by default: the concept stream is rebuilt from the
learned concept embeddings on every step, so a `K x d_concept` state the model
has just computed is discarded at the step boundary and only the sampled token
survives it. A state that advances one layer per update is a deeper feedforward
computation, not a state that tracks anything over time.

`carry_concept_state` closes that channel. The final concept state of a step is
fused into the initial state of the next one by a gated linear unit over the
pair:

```
c_0(t+1) = c_init(t+1) + sigmoid(W_g [c_init, c_L(t)]) * W_v [c_init, c_L(t)]
```

`W_v` starts at zero, so the carry contributes exactly nothing until it is
trained: a carried and an uncarried forward pass agree at initialisation, and
diverge once `W_v` is non-zero and the per-layer `concept_to_token_gate` is open.

Set `"carry_concept_state": true` in the configuration file. `rej-1b generate`
threads the state through the decoding loop by itself and prints
`cross-step concept carry: on`. Training needs one extra thing: teacher forcing
runs a whole sequence in one parallel pass, so nothing in an ordinary step is a
*previous* step and the fusion never fires. A carried step runs the batch again
with the state the first pass ended on, which puts the fusion on the gradient
path. `rej-1b train --carry-passes 2 --carry-every 4` carries one step in four
with two passes. A configuration with the carry and no carry schedule is
refused, because the carry would never receive a gradient.

The carried state is detached between passes. `RejRnmV2` refuses the flag rather
than ignoring it: its concept state is a Gaussian over subspace coordinates, so
carrying it is a distribution to propagate and needs its own fusion rule.

## RejRNMv2: geometric concepts (advanced)

`RejRNMv2` bakes geometry into the architecture itself, rather than applying scalar steering after training:

- **Subspace concepts**: each concept is a rank-`r` subspace (`basis` + `centroid`) instead of a single vector. The basis is orthonormalised by modified Gram–Schmidt. It spans the same subspace as a QR factorisation, and a row may differ from Householder QR's by its sign.
- **Probabilistic concept state** — each concept carries a Gaussian `N(mean, std²)` in subspace coordinates, regularized by a KL term during training.
- **Non-linear concept cells** — MLP-based read/update/write dynamics replace linear cross-attention.
- **Input-dependent router** — each token is assigned a relevance distribution over concepts.
- **Manifold decoder** — subspace coordinates are decoded through a non-linear MLP before being written back to tokens.
- **TITAN-style steering manifold** — each named concept owns multiple learned directions in subspace coordinates; an input-dependent intensity network combines them per layer.
- **Geometry-aware regularization** — biprojection-style loss keeps updated concept embeddings on their subspace manifold.
- **Concept alignment head** — a small head predicts injected control magnitudes from concept states, trained with contrastive supervision.
- **Language-invariant concept objective** — optional loss that aligns concept embeddings of parallel sentences across languages.
- **Control perturbation training**: random control magnitudes, drawn from N(0, scale²) with the scale the run states (`--perturbation-scale`), are injected during LM training so the model learns a smooth control surface.
- **Geometric controls**: four control modes, each enabled by its flag under `allowed_controls` in the configuration:
  - `magnitude` (`--control name=value`): scale concept means by (1 + value).
  - `direction` (`--direction name=v1,v2,...`): add a vector in subspace coordinates.
  - `uncertainty` (`--uncertainty name=value`): add to the concept's log standard deviation.
  - `select` (`--select name=value`): multiply the concept's mean by sigmoid(value).

Generate with geometric controls from a v2 checkpoint:

```bash
rej-1b generate --checkpoint checkpoints/checkpoint_step_1000 \
  --prompt "The sky is" --max-new-tokens 20 \
  --control truthfulness=2.0 --direction truthfulness=1.0,-0.5,0.0,0.0 \
  --trace
```

`--trace` prints the mean norm of each named concept's last-layer state over
the generated steps. `--deterministic` uses each concept's mean instead of a
draw from its Gaussian.

