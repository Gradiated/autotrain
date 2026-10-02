import unittest

import torch
from peft import LoraConfig, get_peft_model

try:
    from transformers import Glm5NextConfig, Glm5NextForConditionalGeneration
except ImportError:
    Glm5NextConfig = None


@unittest.skipIf(Glm5NextConfig is None, "GLM requires Transformers 5.17 or later")
class GlmTests(unittest.TestCase):
    def test_mixed_attention_moe_backward_updates_only_lora(self):
        torch.set_num_threads(1)
        config = Glm5NextConfig(
            text_config={
                "vocab_size": 256,
                "hidden_size": 32,
                "intermediate_size": 64,
                "moe_intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "n_routed_experts": 2,
                "n_shared_experts": 1,
                "num_experts_per_tok": 1,
                "q_lora_rank": 8,
                "kv_lora_rank": 8,
                "qk_nope_head_dim": 16,
                "qk_rope_head_dim": 0,
                "v_head_dim": 16,
                "linear_num_heads": 2,
                "linear_head_dim": 16,
                "index_n_heads": 2,
                "index_head_dim": 16,
                "index_topk": 4,
                "index_kpool": 1,
                "hc_mult": 2,
                "hc_sinkhorn_iters": 3,
                "layer_types": ["linear_attention", "deepseek_sparse_attention"],
                "mlp_layer_types": ["dense", "sparse"],
                "indexer_types": ["full", "full"],
                "max_position_embeddings": 64,
                "use_cache": False,
                "pad_token_id": 0,
            },
            vision_config={
                "depth": 1,
                "hidden_size": 32,
                "num_heads": 2,
                "intermediate_size": 64,
            },
        )
        model = Glm5NextForConditionalGeneration(config)
        model = get_peft_model(
            model,
            LoraConfig(
                r=2,
                lora_alpha=16,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "q_a_proj",
                    "q_b_proj",
                    "kv_a_proj_with_mqa",
                    "kv_b_proj",
                ],
                task_type="CAUSAL_LM",
            ),
        )
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        trainable = {
            name: value
            for name, value in model.named_parameters()
            if value.requires_grad
        }
        self.assertTrue(trainable)
        self.assertTrue(all("lora_" in name for name in trainable))
        before = {name: value.detach().clone() for name, value in trainable.items()}
        optimizer = torch.optim.AdamW(trainable.values(), lr=0.001)
        tokens = torch.tensor([[5, 8, 11, 12]])
        loss = model(input_ids=tokens, labels=tokens).loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(
            any(
                value.grad is not None and value.grad.abs().sum() > 0
                for value in trainable.values()
            )
        )
        optimizer.step()
        self.assertTrue(
            any(
                not torch.equal(value, before[name])
                for name, value in trainable.items()
            )
        )
