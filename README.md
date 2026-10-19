# Autotrain

A Python library for completion-only supervised fine-tuning with LoRA. The
initial implementation uses Hugging Face Transformers and PEFT, masks prompt
tokens, rejects overlength examples, and resumes optimizer and RNG state from
verified checkpoints. This repository is private for now.

```sh
pip install .
python -m unittest discover -s tests
```

```python
from autotrain import Trainer

trainer = Trainer("./run", recipe={
    "id": "my-sft-v1",
    "model": "Qwen/Qwen3-0.6B",
    "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
    "seed": 42,
    "rank": 2,
    "learning_rate": 0.0002,
    "max_steps": 50,
    "sequence_length": 2048,
    "gradient_accumulation": 4,
    "checkpoint_steps": 5,
    "early_stopping": {"patience": 3, "threshold": 0.0001, "min_steps": 10},
    "validation_examples": 256,
    "shuffle": True,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 5,
})
adapter = trainer.train(
    "train.jsonl", "validation.jsonl",
    emit=print,
    save_checkpoint=lambda step, path: print(step, path),
)
trainer.verify(adapter)
```

Datasets are conversational JSONL rows with `prompt` and `completion` message
arrays. The current recipe accepts text, a final user prompt, and one assistant
completion. Checkpoint callbacks receive a local archive path; copy or upload it
before returning because the next checkpoint reuses that path. Pass the saved
archive as `checkpoint=` to resume. Set `trainer.stop` to request cancellation
between microbatches. The adapter archive path is returned by `train`.

`load_adapter` and `generate` support local verification and comparison. The
model revision and recipe are supplied by the caller, never read from deployment
configuration. This library does not manage HTTP servers, object storage,
workflows, scheduling, GPU provisioning, or container images.

The CPU tests use a tiny synthetic Qwen model and assert exact adapter weights
across uninterrupted and resumed training. They do not qualify live GPU training.

The caller chooses the model and pins its revision in the recipe. For an image/text
model used with text-only examples, set `model_task="image_text_to_text"`; the
default remains `causal_lm`. `target_modules` selects the LoRA projections, and
`attention` selects the Transformers attention implementation. These are model
settings, not infrastructure settings.

For models that exceed one GPU, set `gpu_count` to the number already assigned to
the process. The trainer balances complete decoder layers across those visible
GPUs, keeps 15% of their memory for activations, and rejects CPU/disk offload.
`gradient_checkpointing=True` reduces activation memory. Inputs follow the
embedding device, and checkpoints preserve RNG state for every visible GPU. The
caller owns GPU assignment, capacity admission and process lifetime.

CPU tests cover both the original Qwen dependency versions and the GLM-compatible
runtime, including finite-loss LoRA backward through GLM's mixed attention and MoE
layers. Full model sizing, throughput and multi-GPU recovery require qualification
on the caller's target hardware.

With `early_stopping`, validation runs at checkpoint intervals and at the final
step. Training stops after `patience` checks without an absolute loss improvement
larger than `threshold`, once `min_steps` is reached. The returned adapter always
comes from the lowest validation-loss checkpoint, including when training reaches
the hard step limit. Validation measures completion-token-weighted loss on a fixed
seeded sample of up to `validation_examples` examples; use a representative,
leakage-free validation set. The test set belongs to final evaluation and must not
be supplied here. This follows the validation/patience/best-checkpoint pattern in
[Transformers](https://huggingface.co/docs/transformers/en/main_classes/callback),
within this library's custom PyTorch loop; it does not instantiate HF `Trainer` or
TRL `SFTTrainer`.

Optimizer checkpoints load through CPU so parameter state follows each parameter’s
GPU and Adam step counters stay portable across device placements.

Checkpoints contain the current optimizer, scheduler, RNG, early stopping counters
and a copy of the best adapter. Resuming preserves the sample and shuffled epoch
order and will not take additional optimizer steps if stopping was already reached.
Both training and validation file hashes are checked on resume. Recipes without
these optional settings retain fixed-step training and constant learning rate.

These settings are starting points, not tuned hyperparameters. Rank, learning rate,
effective batch size, evaluation frequency and total training budget need validation
for each model and dataset. Fifty steps is a small qualification budget, not a
universal production SFT duration. Low validation loss is a selection signal;
generated-answer quality and regressions still need a separate held-out evaluation.

## Review gate

Every pull request to `main` needs the required `AI review passed` status, set by
`.github/workflows/ai-review-gate.yml`. It passes once both AI reviewers, Codex
(`chatgpt-codex-connector[bot]`) and Graphite (the `Graphite / AI Reviews` check),
have reviewed at least one commit of the PR. No human approval is required, but
every review thread must be resolved and `Tests passed` must be green. If Codex
did not run or failed, comment `@codex review` on the PR; for Graphite, re-run
the AI review from the check's Details link. To re-evaluate a PR by hand, run
`gh workflow run ai-review-gate.yml -f pr=<number>`.
