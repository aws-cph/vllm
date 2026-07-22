# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

import torch

from vllm.platforms.interface import Platform, PlatformEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None


def _dummy_topk_softmax(
    topk_weights,
    topk_ids,
    token_expert_indices,
    gating_output,
    renormalize=False,
):
    """Pure-PyTorch top-k routing, standing in for a native OOT kernel."""
    scores = torch.softmax(gating_output.float(), dim=-1)
    weights, ids = torch.topk(scores, k=topk_weights.shape[1], dim=-1)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-20)
    topk_weights.copy_(weights.to(topk_weights.dtype))
    topk_ids.copy_(ids.to(topk_ids.dtype))
    return topk_weights, topk_ids


class DummyPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name = "DummyDevice"
    device_type: str = "privateuseone"
    dispatch_key: str = "PrivateUse1"

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        vllm_config.compilation_config.custom_ops = ["all"]

    @classmethod
    def get_moe_topk_func(cls, scoring_func: str):
        # OOT platforms with no CUDA lowering supply a traceable top-k impl.
        if scoring_func == "softmax":
            return _dummy_topk_softmax
        return None

    @classmethod
    def moe_forward_oot(cls, layer, x, topk_weights, topk_ids):
        # OOT experts kernel: loop over experts using the stacked weights the
        # oracle leaves on the layer (moe_kernel is None for OOT).
        num_experts = layer.w13_weight.shape[0]
        topk_weights = topk_weights.to(x.dtype)
        out = torch.zeros_like(x)
        for e in range(num_experts):
            mask = (topk_ids == e).to(x.dtype)
            weight = (topk_weights * mask).sum(dim=-1, keepdim=True)
            gate_up = x @ layer.w13_weight[e].t()
            gate, up = gate_up.chunk(2, dim=-1)
            hidden = torch.nn.functional.silu(gate) * up
            out = out + (hidden @ layer.w2_weight[e].t()) * weight
        return out

    def get_attn_backend_cls(
        self,
        backend_name,
        head_size,
        dtype,
        kv_cache_dtype,
        block_size,
        use_mla,
        has_sink,
        use_sparse,
        use_mm_prefix,
    ):
        return "vllm_add_dummy_platform.dummy_attention_backend.DummyAttentionBackend"  # noqa E501
