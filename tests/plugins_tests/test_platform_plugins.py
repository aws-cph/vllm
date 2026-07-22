# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.plugins import load_general_plugins


def test_platform_plugins():
    # simulate workload by running an example
    import runpy

    current_file = __file__
    import os

    example_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(current_file))),
        "examples",
        "basic/offline_inference/basic.py",
    )
    runpy.run_path(example_file)

    # check if the plugin is loaded correctly
    from vllm.platforms import _init_trace, current_platform

    assert current_platform.device_name == "DummyDevice", (
        f"Expected DummyDevice, got {current_platform.device_name}, "
        "possibly because current_platform is imported before the plugin"
        f" is loaded. The first import:\n{_init_trace}"
    )


def test_oot_custom_op(default_vllm_config, monkeypatch: pytest.MonkeyPatch):
    # simulate workload by running an example
    load_general_plugins()
    from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

    layer = RotaryEmbedding(16, 16, 16, 16, True, torch.float16)
    assert layer.__class__.__name__ == "DummyRotaryEmbedding", (
        f"Expected DummyRotaryEmbedding, got {layer.__class__.__name__}, "
        "possibly because the custom op is not registered correctly."
    )
    assert hasattr(layer, "addition_config"), (
        "Expected DummyRotaryEmbedding to have an 'addition_config' attribute, "
        "which is set by the custom op."
    )


def test_oot_moe_hooks_wired():
    """The OOT MoE hooks on the loaded dummy platform are consumed by the
    fused-MoE stack: the router picks up the native top-k, and the unquantized
    method routes the (kernel-less) OOT backend to moe_forward_oot."""
    load_general_plugins()

    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
        dispatch_topk_softmax_func,
        dispatch_topk_sigmoid_func,
        vllm_topk_sigmoid,
    )
    from vllm.platforms import current_platform

    assert current_platform.device_name == "DummyDevice"

    # softmax is overridden by the dummy platform; sigmoid falls back to default
    softmax_impl = dispatch_topk_softmax_func(use_rocm_aiter=False)
    assert softmax_impl is current_platform.get_moe_topk_func("softmax")
    assert softmax_impl is not None
    assert dispatch_topk_sigmoid_func(use_rocm_aiter=False) is vllm_topk_sigmoid

    # moe_forward_oot runs the platform's expert loop end-to-end.
    class _Layer:
        pass

    layer = _Layer()
    num_experts, hidden, inter = 2, 4, 6
    layer.w13_weight = torch.randn(num_experts, 2 * inter, hidden)
    layer.w2_weight = torch.randn(num_experts, hidden, inter)

    x = torch.randn(3, hidden)
    topk_ids = torch.tensor([[0], [1], [0]])
    topk_weights = torch.ones(3, 1)

    out = current_platform.moe_forward_oot(layer, x, topk_weights, topk_ids)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
