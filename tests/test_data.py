import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from autotrain.data import example_tokens, pack, unpack


class Tokenizer:
    eos_token = "!"

    def apply_chat_template(self, messages, **kwargs):
        return "prompt>"

    def encode(self, text, **kwargs):
        return list(text.encode())


class DataTests(unittest.TestCase):
    def test_only_the_target_contributes_to_loss(self):
        result = example_tokens(
            Tokenizer(),
            {
                "prompt": [{"role": "user", "content": "hi"}],
                "completion": [{"role": "assistant", "content": "answer"}],
            },
            100,
        )
        self.assertEqual(result["labels"], [-100] * 7 + list(b"answer!"))
        self.assertEqual(result["input_ids"], list(b"prompt>answer!"))

    def test_does_not_truncate_targets(self):
        self.assertIsNone(
            example_tokens(
                Tokenizer(),
                {
                    "prompt": [{"role": "user", "content": "hi"}],
                    "completion": [{"role": "assistant", "content": "answer"}],
                },
                10,
            )
        )

    def test_rejects_invalid_completions(self):
        for completion in (
            [],
            [{"role": "user", "content": "answer"}],
            [{"role": "assistant", "content": " "}],
            [{"role": "assistant", "content": "a"}] * 2,
        ):
            with self.subTest(completion=completion), self.assertRaises(ValueError):
                example_tokens(
                    Tokenizer(),
                    {
                        "prompt": [{"role": "user", "content": "hi"}],
                        "completion": completion,
                    },
                    100,
                )

    def test_roundtrip_and_reject_path_traversal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            (source / "adapter_config.json").write_text("{}")
            archive = root / "adapter.tar.gz"
            pack(source, archive)
            unpack(archive, root / "output")
            self.assertEqual((root / "output/adapter_config.json").read_text(), "{}")
            for name in ("../outside", "/absolute"):
                with self.subTest(name=name):
                    with tarfile.open(archive, "w:gz") as target:
                        member = tarfile.TarInfo(name)
                        member.size = 1
                        target.addfile(member, io.BytesIO(b"x"))
                    with self.assertRaises(ValueError):
                        unpack(archive, root / "output")


if __name__ == "__main__":
    unittest.main()
