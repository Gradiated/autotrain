import math
import random
import shutil
import threading
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    get_scheduler,
)

from .data import digest, example_tokens, pack, read_examples, unpack
from .stopping import EarlyStopping


class Trainer:
    def __init__(self, root, recipe, device="cuda"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.recipe = dict(recipe)
        self.device = device
        self.model = None
        self.tokenizer = None
        self.stop = threading.Event()

    def load(self):
        if self.model is not None:
            return
        torch.manual_seed(self.recipe["seed"])
        random.seed(self.recipe["seed"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.recipe["model"],
            revision=self.recipe["revision"],
            trust_remote_code=False,
        )
        gpu_count = self.recipe.get("gpu_count", 1)
        if self.device == "cuda" and torch.cuda.device_count() < gpu_count:
            raise ValueError("The recipe requires more GPUs than were reserved")
        loader = (
            AutoModelForImageTextToText
            if self.recipe.get("model_task") == "image_text_to_text"
            else AutoModelForCausalLM
        )
        placement = {}
        if self.device == "cuda" and gpu_count > 1:
            placement = {
                "device_map": "balanced",
                "max_memory": {
                    **{
                        index: int(
                            torch.cuda.get_device_properties(index).total_memory * 0.85
                        )
                        for index in range(gpu_count)
                    },
                    "cpu": 0,
                },
            }
        base = loader.from_pretrained(
            self.recipe["model"],
            revision=self.recipe["revision"],
            trust_remote_code=False,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            attn_implementation=self.recipe.get("attention", "sdpa"),
            **placement,
        )
        if not placement:
            base = base.to(self.device)
        elif any(device in ("cpu", "disk") for device in base.hf_device_map.values()):
            raise ValueError("The model does not fit the reserved GPU memory")
        self.model = get_peft_model(
            base,
            LoraConfig(
                r=self.recipe["rank"],
                lora_alpha=16,
                lora_dropout=0,
                target_modules=self.recipe.get(
                    "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
                ),
                task_type="CAUSAL_LM",
            ),
        )
        if self.recipe.get("gradient_checkpointing", False):
            self.model.enable_input_require_grads()
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    @property
    def input_device(self):
        return self.model.get_input_embeddings().weight.device

    def check(self):
        if self.stop.is_set():
            raise InterruptedError("Training stopped")

    def load_adapter(self, archive):
        self.check()
        self.load()
        directory = self.root / "loaded-adapter"
        shutil.rmtree(directory, ignore_errors=True)
        unpack(archive, directory)
        if "verified" not in self.model.peft_config:
            self.model.load_adapter(
                directory, adapter_name="verified", is_trainable=False
            )
        self.model.set_adapter("verified")
        self.model.eval()

    def verify(self, archive):
        self.load_adapter(archive)
        with torch.no_grad():
            ids = torch.tensor([[self.tokenizer.eos_token_id]], device=self.device)
            if not torch.isfinite(self.model(input_ids=ids).logits).all():
                raise ValueError("Adapter produced non-finite logits")

    def generate(self, messages, adapted, max_tokens=256, max_prompt_tokens=None):
        self.check()
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.input_device)
        limit = (
            max_prompt_tokens
            if max_prompt_tokens is not None
            else self.recipe["sequence_length"]
        )
        if inputs.input_ids.shape[1] > limit:
            raise ValueError("Evaluation prompt exceeds sequence limit")
        started = time.monotonic()
        with (
            nullcontext() if adapted else self.model.disable_adapter(),
            torch.no_grad(),
        ):
            output = self.model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False
            )
        suffix = output[0, inputs.input_ids.shape[1] :]
        return {
            "text": self.tokenizer.decode(suffix, skip_special_tokens=True),
            "tokens": len(suffix),
            "seconds": time.monotonic() - started,
        }

    def save_adapter(self, directory):
        shutil.rmtree(directory, ignore_errors=True)
        self.model.save_pretrained(
            directory,
            safe_serialization=True,
            selected_adapters=[self.model.active_adapter],
        )
        nested = directory / self.model.active_adapter
        if nested.is_dir():
            for path in nested.iterdir():
                shutil.move(str(path), directory / path.name)
            nested.rmdir()

    def evaluate(self, examples):
        self.model.eval()
        loss = 0.0
        tokens = 0
        with torch.no_grad():
            for item in examples:
                self.check()
                count = sum(label != -100 for label in item["labels"][1:])
                batch = {
                    key: torch.tensor([value], device=self.input_device)
                    for key, value in item.items()
                }
                loss += self.model(**batch).loss.item() * count
                tokens += count
        self.model.train()
        if not tokens or not math.isfinite(loss):
            raise ValueError("Invalid validation loss")
        return loss / tokens

    def train(
        self,
        dataset,
        validation,
        *,
        checkpoint=None,
        emit=lambda event: None,
        save_checkpoint=lambda step, path: None,
    ):
        self.check()
        self.load()
        self.check()
        dataset = Path(dataset)
        dataset_sha256 = digest(dataset)
        examples = read_examples(dataset)
        data = [
            tokens
            for item in examples
            if (
                tokens := example_tokens(
                    self.tokenizer, item, self.recipe["sequence_length"]
                )
            )
        ]
        validation_sha256 = digest(validation)
        validation_examples = read_examples(validation)
        limit = self.recipe.get("validation_examples", 64)
        if type(limit) is not int or limit <= 0:
            raise ValueError("validation_examples must be a positive integer")
        validation_examples = (
            random.Random(self.recipe["seed"]).sample(
                validation_examples, min(limit, len(validation_examples))
            )
            if "validation_examples" in self.recipe
            else validation_examples[:limit]
        )
        validation = [
            tokens
            for item in validation_examples
            if (
                tokens := example_tokens(
                    self.tokenizer, item, self.recipe["sequence_length"]
                )
            )
        ]
        if not validation:
            raise ValueError("No validation examples fit the sequence limit")
        if not data:
            raise ValueError("No examples fit the sequence limit")
        emit(
            {
                "type": "trainer_started",
                "gpu": {
                    "type": torch.cuda.get_device_name()
                    if self.device == "cuda"
                    else "cpu",
                    "count": self.recipe.get("gpu_count", 1),
                },
                "recipe": self.recipe["id"],
            }
        )
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.recipe["learning_rate"],
        )
        stopping = (
            EarlyStopping(**self.recipe["early_stopping"])
            if self.recipe.get("early_stopping") is not None
            else None
        )
        best = self.root / "best-adapter"
        state = None
        step = 0
        if checkpoint:
            archive = checkpoint
            directory = self.root / "resume"
            unpack(archive, directory)
            self.model.load_adapter(
                directory, adapter_name="resumed", is_trainable=True
            )
            self.model.set_adapter("resumed")
            optimizer = torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=self.recipe["learning_rate"],
            )
            state = torch.load(
                directory / "optimizer.pt", map_location=self.device, weights_only=True
            )
            if (
                state["dataset_sha256"] != dataset_sha256
                or state["recipe"] != self.recipe
                or state.get("validation_sha256", validation_sha256)
                != validation_sha256
            ):
                raise ValueError("Checkpoint does not match dataset and recipe")
            optimizer.load_state_dict(state["optimizer"])
            if stopping:
                stopping.restore(state["early_stopping"])
                if stopping.step != state["step"]:
                    raise ValueError(
                        "Early stopping state does not match checkpoint step"
                    )
                shutil.copytree(directory / "best-adapter", best, dirs_exist_ok=True)
            step = state["step"]
            torch.set_rng_state(state["rng"].cpu())
            if self.device == "cuda":
                if "cuda_rng_all" in state:
                    torch.cuda.set_rng_state_all(
                        [value.cpu() for value in state["cuda_rng_all"]]
                    )
                else:
                    torch.cuda.set_rng_state(state["cuda_rng"].cpu())
        if not 0 <= step <= self.recipe["max_steps"]:
            raise ValueError("Invalid checkpoint step")
        scheduler = None
        if "lr_scheduler_type" in self.recipe:
            scheduler = get_scheduler(
                self.recipe["lr_scheduler_type"],
                optimizer,
                num_warmup_steps=self.recipe.get("warmup_steps", 0),
                num_training_steps=self.recipe["max_steps"],
            )
            if state:
                scheduler.load_state_dict(state["scheduler"])
                for group, rate in zip(optimizer.param_groups, scheduler.get_last_lr()):
                    group["lr"] = rate
        self.model.train()
        self.model.config.use_cache = False
        accumulation = self.recipe["gradient_accumulation"]
        emit({"type": "trainer_stage", "stage": "training"})
        order = []
        epoch = -1
        while step < self.recipe["max_steps"] and not (stopping and stopping.stopped):
            self.check()
            started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            loss = 0.0
            tokens = 0
            for micro in range(accumulation):
                self.check()
                position = step * accumulation + micro
                current_epoch, index = divmod(position, len(data))
                if current_epoch != epoch:
                    order = list(range(len(data)))
                    if self.recipe.get("shuffle", False):
                        random.Random(self.recipe["seed"] + current_epoch).shuffle(
                            order
                        )
                    epoch = current_epoch
                example = data[order[index]]
                batch = {
                    key: torch.tensor([value], device=self.input_device)
                    for key, value in example.items()
                }
                result = self.model(**batch).loss / accumulation
                if not torch.isfinite(result):
                    raise ValueError("Non-finite training loss")
                result.backward()
                loss += result.item()
                tokens += len(example["input_ids"])
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            learning_rate = optimizer.param_groups[0]["lr"]
            optimizer.step()
            if scheduler:
                scheduler.step()
            step += 1
            seconds = max(time.monotonic() - started, 0.001)
            emit(
                {
                    "type": "trainer_progress",
                    "step": step,
                    "total_steps": self.recipe["max_steps"],
                    "train_loss": loss,
                    "tokens_per_second": tokens / seconds,
                    "eta_seconds": seconds * (self.recipe["max_steps"] - step),
                    "metrics": {"learning_rate": learning_rate},
                }
            )
            if (
                step % self.recipe["checkpoint_steps"] == 0
                or step == self.recipe["max_steps"]
            ):
                self.check()
                validation_loss = self.evaluate(validation)
                if stopping and stopping.observe(step, validation_loss):
                    self.save_adapter(best)
                emit(
                    {
                        "type": "note",
                        "level": "info",
                        "message": f"Validation loss at step {step}: {validation_loss:.6f}",
                    }
                )
                emit(
                    {
                        "type": "trainer_observation",
                        "name": "validation",
                        "message": f"Validated step {step} on {len(validation)} examples",
                        "step": step,
                        "metrics": {
                            "validation_loss": validation_loss,
                            **(
                                {
                                    "best_step": stopping.best_step,
                                    "bad_checks": stopping.bad_checks,
                                }
                                if stopping
                                else {}
                            ),
                        },
                    }
                )
                directory = self.root / "checkpoint"
                self.save_adapter(directory)
                if stopping:
                    shutil.copytree(best, directory / "best-adapter")
                torch.save(
                    {
                        "step": step,
                        "recipe": self.recipe,
                        "dataset_sha256": dataset_sha256,
                        "validation_sha256": validation_sha256,
                        "optimizer": optimizer.state_dict(),
                        **({"scheduler": scheduler.state_dict()} if scheduler else {}),
                        **({"early_stopping": stopping.state()} if stopping else {}),
                        "rng": torch.get_rng_state(),
                        "cuda_rng_all": torch.cuda.get_rng_state_all()
                        if self.device == "cuda"
                        else [],
                    },
                    directory / "optimizer.pt",
                )
                archive = self.root / "checkpoint.tar.gz"
                pack(directory, archive)
                if step % self.recipe["checkpoint_steps"] == 0:
                    save_checkpoint(step, archive)
        directory = self.root / "adapter"
        if stopping:
            shutil.rmtree(directory, ignore_errors=True)
            shutil.copytree(best, directory)
            emit(
                {
                    "type": "note",
                    "level": "info",
                    "message": (
                        f"Training {'stopped early' if stopping.stopped else 'reached the step limit'} "
                        f"at step {step}; selected step {stopping.best_step} "
                        f"with validation loss {stopping.best_loss:.6g}"
                    ),
                }
            )
        else:
            self.save_adapter(directory)
        archive = self.root / "adapter.tar.gz"
        pack(directory, archive)
        emit({"type": "trainer_finished", "adapter_sha256": digest(archive)})
        return archive
