# Training

## Training

### Pretraining

```bash
python scripts/train.py \
  --config configs/rej_1b.json \
  --data corpus.txt \
  --output_dir checkpoints \
  --num_steps 10000 \
  --batch_size 8 \
  --seq_length 512
```

Add the cross-step concept carry by setting `carry_concept_state` in the config
JSON. `--carry_passes` then defaults to 2 and `--carry_fraction` to 0.25, and
the run prints the schedule it chose:

```bash
python scripts/train.py \
  --config configs/rej_1b.json \
  --data corpus.txt \
  --output_dir checkpoints \
  --carry_passes 2 \
  --carry_fraction 0.25
```

`--carry_passes 1` with the flag set is refused rather than run: the fusion
would never be called and never receive gradient, so the model would carry a
module it cannot learn.

### Controlled generation

```bash
python scripts/generate.py \
  --checkpoint checkpoints/checkpoint_step_10000.pt \
  --prompt "Explain quantum computing." \
  --controls "truthfulness=1.5,refusal=-0.5" \
  --max_new_tokens 100 \
  --trace
```

### v2 training APIs

```python
from rej_1b import RejRNMv2, rej_tiny_v2_config
from rej_1b.train import train_v2, train_v2_aligned, train_v2_multilingual

config = rej_tiny_v2_config()
config.use_concept_alignment = True
config.use_titan_manifold = True
config.use_geometry_regularization = True
model = RejRNMv2(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# LM pretraining with random control perturbations.
train_v2(model, token_batches, optimizer, device, num_steps=1000,
         perturb_controls=True, perturbation_scale=1.0)

# Concept-alignment training with (tokens, control_magnitudes) batches.
train_v2_aligned(model, aligned_batches, optimizer, device, num_steps=1000)

# Multilingual concept-alignment with parallel sentences.
config.use_language_invariant_concepts = True
train_v2_multilingual(model, parallel_batches, optimizer, device, num_steps=1000)
```

### Python API

```python
from rej_1b import RejRNM, RejTokenizer, generate, rej_1b_config

config = rej_1b_config()
model = RejRNM(config)
tokenizer = RejTokenizer(vocab_size=config.vocab_size)

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

