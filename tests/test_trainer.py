import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from test_data import Tokenizer
from transformers import Qwen3Config, Qwen3ForCausalLM

from autotrain import Trainer
from autotrain.data import unpack


class TinyTrainer(Trainer):
    def load(self):
        torch.manual_seed(self.recipe["seed"])
        self.tokenizer = Tokenizer()
        model = Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=256,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                max_position_embeddings=128,
                use_cache=False,
            )
        )
        self.model = get_peft_model(
            model,
            LoraConfig(
                r=self.recipe["rank"],
                lora_alpha=16,
                lora_dropout=0,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                task_type="CAUSAL_LM",
            ),
        )


class TrainerTests(unittest.TestCase):
    def test_checkpoint_resume_matches_uninterrupted_sft(self):
        torch.set_num_threads(1)
        recipe = {
            "id": "test-sft",
            "seed": 42,
            "rank": 2,
            "learning_rate": 0.0002,
            "max_steps": 2,
            "checkpoint_steps": 1,
            "gradient_accumulation": 1,
            "sequence_length": 64,
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "dataset.jsonl"
            dataset.write_text(
                '{"prompt":[{"role":"user","content":"hi"}],"completion":[{"role":"assistant","content":"answer"}]}\n'
            )
            checkpoints = {}

            def save_checkpoint(step, path):
                destination = root / f"step-{step}.tar.gz"
                destination.write_bytes(path.read_bytes())
                checkpoints[step] = destination

            events = []
            full = TinyTrainer(root / "full", recipe, device="cpu").train(
                dataset,
                dataset,
                emit=events.append,
                save_checkpoint=save_checkpoint,
            )
            self.assertEqual(
                [
                    event["step"]
                    for event in events
                    if event["type"] == "trainer_progress"
                ],
                [1, 2],
            )

            def interrupt(step, path):
                save_checkpoint(step, path)
                if step == 1:
                    raise InterruptedError("simulate lost host")

            with self.assertRaises(InterruptedError):
                TinyTrainer(root / "interrupted", recipe, device="cpu").train(
                    dataset,
                    dataset,
                    save_checkpoint=interrupt,
                )
            resumed_events = []
            resumed = TinyTrainer(root / "resumed", recipe, device="cpu").train(
                dataset,
                dataset,
                checkpoint=checkpoints[1],
                emit=resumed_events.append,
            )
            self.assertEqual(
                [
                    event["step"]
                    for event in resumed_events
                    if event["type"] == "trainer_progress"
                ],
                [2],
            )
            unpack(full, root / "full-adapter")
            unpack(resumed, root / "resumed-adapter")
            expected = load_file(root / "full-adapter/adapter_model.safetensors")
            actual = load_file(root / "resumed-adapter/adapter_model.safetensors")
            self.assertEqual(expected.keys(), actual.keys())
            for name, tensor in expected.items():
                torch.testing.assert_close(tensor, actual[name], rtol=0, atol=0)

    def test_early_stopping_publishes_best_weights_and_survives_resume(self):
        torch.set_num_threads(1)
        recipe = {
            "id": "test-early-stopping",
            "seed": 42,
            "rank": 2,
            "learning_rate": 0.0002,
            "max_steps": 8,
            "checkpoint_steps": 1,
            "gradient_accumulation": 1,
            "sequence_length": 64,
            "early_stopping": {"patience": 2, "threshold": 0.001, "min_steps": 1},
            "lr_scheduler_type": "cosine",
            "warmup_steps": 1,
            "shuffle": True,
            "validation_examples": 256,
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "dataset.jsonl"
            dataset.write_text(
                "".join(
                    '{"prompt":[{"role":"user","content":"hi"}],"completion":[{"role":"assistant","content":"'
                    + answer
                    + '"}]}\n'
                    for answer in ["one", "two", "three"]
                )
            )
            checkpoints = {}

            def save(step, archive):
                destination = root / f"step-{step}.tar.gz"
                destination.write_bytes(archive.read_bytes())
                checkpoints[step] = destination

            events = []
            with patch.object(
                TinyTrainer, "evaluate", side_effect=[1.0, 0.5, 0.7, 0.8]
            ):
                full = TinyTrainer(root / "full", recipe, "cpu").train(
                    dataset,
                    dataset,
                    emit=events.append,
                    save_checkpoint=save,
                )
            self.assertEqual(
                [e["step"] for e in events if e["type"] == "trainer_progress"],
                [1, 2, 3, 4],
            )
            unpack(full, root / "full-adapter")
            unpack(checkpoints[2], root / "best-checkpoint")
            expected = load_file(root / "best-checkpoint/adapter_model.safetensors")
            actual = load_file(root / "full-adapter/adapter_model.safetensors")
            for name in expected:
                torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0)

            for checkpoint_step, losses, steps in [(3, [0.8], [4]), (4, [], [])]:
                resumed_events = []
                resumed_checkpoints = {}

                def save_resumed(step, archive, captured=resumed_checkpoints):
                    destination = root / f"resumed-checkpoint-{step}.tar.gz"
                    destination.write_bytes(archive.read_bytes())
                    captured[step] = destination

                with patch.object(TinyTrainer, "evaluate", side_effect=losses):
                    resumed = TinyTrainer(
                        root / f"resumed-{checkpoint_step}", recipe, "cpu"
                    ).train(
                        dataset,
                        dataset,
                        checkpoint=checkpoints[checkpoint_step],
                        emit=resumed_events.append,
                        save_checkpoint=save_resumed,
                    )
                self.assertEqual(
                    [
                        e["step"]
                        for e in resumed_events
                        if e["type"] == "trainer_progress"
                    ],
                    steps,
                )
                if checkpoint_step == 3:
                    unpack(checkpoints[4], root / "full-current")
                    unpack(resumed_checkpoints[4], root / "resumed-current")
                    full_current = load_file(
                        root / "full-current/adapter_model.safetensors"
                    )
                    resumed_current = load_file(
                        root / "resumed-current/adapter_model.safetensors"
                    )
                    for name in full_current:
                        torch.testing.assert_close(
                            full_current[name], resumed_current[name], rtol=0, atol=0
                        )
                unpack(resumed, root / f"adapter-{checkpoint_step}")
                weights = load_file(
                    root / f"adapter-{checkpoint_step}/adapter_model.safetensors"
                )
                for name in expected:
                    torch.testing.assert_close(
                        expected[name], weights[name], rtol=0, atol=0
                    )

            dataset.write_text(dataset.read_text().replace("one", "changed"))
            with self.assertRaisesRegex(ValueError, "does not match"):
                TinyTrainer(root / "mismatched", recipe, "cpu").train(
                    dataset,
                    dataset,
                    checkpoint=checkpoints[3],
                )

    def test_validation_is_weighted_by_completion_tokens(self):
        trainer = TinyTrainer(
            tempfile.gettempdir() + "/autotrain-validation-test",
            {"seed": 42, "rank": 2},
            "cpu",
        )
        trainer.load()
        examples = [
            {"input_ids": [1, 2, 3], "labels": [-100, -100, 3]},
            {"input_ids": [1, 2, 3, 4], "labels": [-100, 2, 3, 4]},
        ]
        from types import SimpleNamespace

        with patch.object(
            trainer.model,
            "forward",
            side_effect=[
                SimpleNamespace(loss=torch.tensor(2.0)),
                SimpleNamespace(loss=torch.tensor(4.0)),
            ],
        ):
            self.assertEqual(trainer.evaluate(examples), 3.5)


if __name__ == "__main__":
    unittest.main()
