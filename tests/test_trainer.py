import tempfile
import unittest
from pathlib import Path

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
            "rank": 8,
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


if __name__ == "__main__":
    unittest.main()
