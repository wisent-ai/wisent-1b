# Training

Every setting a run depends on is stated by the run:

- The model's shape, dropout and loss weights come from its configuration file.
- The batch, sequence length, step count and learning rate come from flags.
- AdamW's beta1, beta2, epsilon and weight decay are candle's declared defaults
  unless `--beta1`, `--beta2`, `--eps` and `--weight-decay` state them.
- Gradients are clipped only with `--max-grad-norm`.

`train` prints one JSON line per step to standard output. It saves a
checkpoint every `--save-every` steps, if given, and always at the end.
Training stops after `--num-steps` steps or after one pass over the data,
whichever comes first.

## Pretraining

```bash
rej-1b train --config configs/rej_1b.json --data corpus.txt \
  --seq-length 512 --batch-size 8 --num-steps 10000 --learning-rate 3e-4 \
  --save-every 500 --output-dir checkpoints
```

The corpus is cut into sequences of `--seq-length + 1` ids, every
`--seq-length` ids. The sequences are shuffled and padded into batches with the
tokenizer's pad id.

With `"carry_concept_state": true` in the configuration, the run needs a carry
schedule:

```bash
rej-1b train --config my_carry_config.json --data corpus.txt \
  --seq-length 512 --batch-size 8 --num-steps 10000 --learning-rate 3e-4 \
  --carry-passes 2 --carry-every 4 --output-dir checkpoints
```

Without a schedule the run is refused, because the fusion would never be called
and never receive a gradient. `--carry-passes 1` is refused for the same reason.

## Training v2

A v2 configuration (`configs/rej_1b_v2.json`) trains the same way on `--data`.
`--perturbation-scale S` adds random magnitude controls drawn from N(0, S²) to
every step. The total loss is the language-model loss plus `kl_weight` times
the KL term plus `geometry_weight` times the geometry term.

Concept-alignment training reads JSON Lines. Each line is a text and the
magnitude of each named concept it shows; a concept a line does not name has
magnitude zero:

```json
{"text": "The capital of France is Paris.", "controls": {"truthfulness": 1.0}}
```

```bash
rej-1b train --config configs/rej_1b_v2.json --aligned aligned.jsonl \
  --batch-size 8 --num-steps 1000 --learning-rate 3e-4 --output-dir checkpoints
```

This adds `alignment_weight` times the mean squared error between the
alignment head's prediction and the line's magnitudes. It needs
`use_concept_alignment`.

Language-invariant training reads the same lines with a `parallel` translation:

```json
{"text": "The sky is blue.", "parallel": "Niebo jest niebieskie.", "controls": {}}
```

```bash
rej-1b train --config my_multilingual_config.json --multilingual parallel.jsonl \
  --batch-size 8 --num-steps 1000 --learning-rate 3e-4 --output-dir checkpoints
```

This adds `language_invariant_weight` times the mean squared error between the
two texts' pooled concept embeddings. It needs `use_language_invariant_concepts`.

## Controlled generation

```bash
rej-1b generate --checkpoint checkpoints/checkpoint_step_10000 \
  --prompt "Explain quantum computing." \
  --control truthfulness=1.5 --control refusal=-0.5 \
  --max-new-tokens 100 --trace
```

`--max-new-tokens` is required. Generation stops early at the tokenizer's
end-of-sequence id.

Sampling settings:

- Without `--temperature`, `--top-k` or `--top-p`, tokens are drawn from the
  model's distribution as it is.
- `--greedy` takes the most likely token at every step.

## Tokenizers

Without `--tokenizer`, the model uses the character tokenizer. It holds
`<PAD>`, `<EOS>`, `<UNK>`, printable ASCII, newline, tab and the printable
Latin-1 supplement, cut to the model's `vocab_size`.

`--tokenizer tokenizer.json` reads a Hugging Face tokenizer instead:

- `--eos-token` names the token that ends generation.
- `--pad-token` names the token that pads a batch.
- A tokenizer with more ids than the model has embeddings is refused.

## Refusals

- **A configuration missing a field:** refused with the field's name.
- **A control naming a concept the model does not have:** refused with the
  names it does have.
- **A v2 control on a first-model checkpoint, or the reverse:** refused.
- **`--aligned` or `--multilingual` with a first-model configuration:**
  refused.
- **A batch that needs padding when the tokenizer has no pad token:** refused.

## Library

The crate exposes the same operations:

- **Building and saving:** `checkpoint::Loaded` builds, loads and saves a model.
- **Training:** `train::train` and `train::train_v2` run training with a
  `train::Trainer`.
- **Generation:** `generate::generate` and `generate::generate_v2` decode.
- **Concepts:** `train::ConceptProbe` trains a probe on a named concept slot,
  and `train::ControlFineTuner` fine-tunes under random controls of a stated
  scale.
