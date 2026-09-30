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
    "rank": 8,
    "learning_rate": 0.0002,
    "max_steps": 50,
    "sequence_length": 2048,
    "gradient_accumulation": 4,
    "checkpoint_steps": 10,
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
