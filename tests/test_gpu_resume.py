import unittest
from unittest.mock import patch

import torch
from accelerate import dispatch_model
import test_trainer


class SplitGpuTrainer(test_trainer.TinyTrainer):
    def __init__(self, root, recipe, device="cpu"):
        super().__init__(root, recipe, device="cuda")

    def load(self):
        super().load()
        self.model = dispatch_model(
            self.model,
            device_map={
                "base_model.model.model.embed_tokens": 0,
                "base_model.model.model.layers.0": 1,
                "base_model.model.model.norm": 1,
                "base_model.model.lm_head": 1,
            },
        )


class MultiGpuResumeTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.device_count() >= 2, "Requires two CUDA GPUs")
    def test_checkpoint_resume_matches_uninterrupted_training(self):
        with patch("test_trainer.TinyTrainer", SplitGpuTrainer):
            test_trainer.TrainerTests().test_checkpoint_resume_matches_uninterrupted_sft()
