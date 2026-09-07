import logging
import os

import torch

import vllm_gaudi.extension.ops as hpu_ops
from vllm_gaudi import envs
import vllm.model_executor.model_loader.utils as hpu_utils
import vllm.model_executor.model_loader.base_loader as base_loader
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.attention import (Attention, MLAAttention)
from vllm.model_executor.model_loader.reload import set_torchao_reload_attrs

logger = logging.getLogger(__name__)


def _cache_hpu_block_fp8_weight(
    module: torch.nn.Module,
    model_dtype: torch.dtype,
) -> int:
    quant_method = getattr(module, "quant_method", None)
    if (
        not getattr(quant_method, "block_quant", False)
        or not hasattr(module, "weight")
        or not hasattr(module, "weight_scale_inv")
    ):
        return 0

    block_size = getattr(
        getattr(quant_method, "quant_config", None),
        "weight_block_size",
        None,
    )
    if block_size is None:
        return 0

    original_m = getattr(module, "_hpu_orig_M", None)
    original_n = getattr(module, "_hpu_orig_N", None)
    if original_m is None or original_n is None:
        return 0

    dequant_weight = hpu_ops.dequant_block_fp8_weight_naive(
        module.weight.detach(),
        module.weight_scale_inv.detach(),
        block_size,
        dtype=model_dtype,
        original_M=original_m,
        original_N=original_n,
        do_unpad=True,
    )
    transposed_weight = torch.empty(
        (original_n, original_m),
        dtype=model_dtype,
        device=module.weight.device,
    )
    transposed_weight.copy_(dequant_weight.T)
    buffer_name = "_hpu_block_fp8_weight_dequant_transposed"
    if buffer_name in module._buffers:
        module._buffers[buffer_name] = transposed_weight
    else:
        module.register_buffer(
            buffer_name, transposed_weight, persistent=False
        )
    return (
        transposed_weight.numel()
        * transposed_weight.element_size()
    )


def _cache_hpu_shared_expert_block_fp8_weight(
    module_name: str,
    module: torch.nn.Module,
    model_dtype: torch.dtype,
) -> int:
    if "shared_experts" not in module_name.split("."):
        return 0
    return _cache_hpu_block_fp8_weight(module, model_dtype)


def _cache_hpu_dsv4_attention_block_fp8_weight(
    module_name: str,
    module: torch.nn.Module,
    model_dtype: torch.dtype,
) -> int:
    cache_bf16 = envs.VLLM_HPU_DSV4_BF16_ATTN_WEIGHT_CACHE
    cache_native_fp8 = envs.VLLM_HPU_DSV4_NATIVE_FP8_ATTN_FRONTEND
    if not cache_bf16 and not cache_native_fp8:
        return 0
    is_wo_a = module_name.endswith("attn.wo_a")
    is_native_frontend = module_name.endswith(
        ("attn.fused_wqa_wkv", "attn.wq_b")
    )
    if not module_name.endswith(
        (
            "attn.fused_wqa_wkv",
            "attn.wq_b",
            "attn.wo_a",
            "attn.wo_b",
        )
    ):
        return 0
    cached_bytes = 0
    bf16_cached_bytes = 0
    if cache_bf16:
        if is_wo_a:
            quant_method = getattr(module, "quant_method", None)
            block_size = getattr(
                getattr(quant_method, "quant_config", None),
                "weight_block_size",
                None,
            )
            original_m = getattr(module, "_hpu_orig_M", None)
            original_n = getattr(module, "_hpu_orig_N", None)
            n_local_groups = getattr(module, "bmm_batch_size", None)
            if (
                block_size is not None
                and original_m is not None
                and original_n is not None
                and n_local_groups is not None
                and original_m % n_local_groups == 0
            ):
                o_lora_rank = original_m // n_local_groups
                bmm_weight = hpu_ops.dequant_block_fp8_weight_naive(
                    module.weight.detach(),
                    module.weight_scale_inv.detach(),
                    block_size,
                    dtype=model_dtype,
                    original_M=original_m,
                    original_N=original_n,
                    do_unpad=True,
                ).view(n_local_groups, o_lora_rank, original_n)
                buffer_name = "_hpu_dsv4_wo_a_bmm_weight"
                if buffer_name in module._buffers:
                    module._buffers[buffer_name] = bmm_weight
                else:
                    module.register_buffer(
                        buffer_name, bmm_weight, persistent=False
                    )
                bf16_cached_bytes = (
                    bmm_weight.numel() * bmm_weight.element_size()
                )
        else:
            bf16_cached_bytes = _cache_hpu_block_fp8_weight(module, model_dtype)
        cached_bytes += bf16_cached_bytes
    if cache_native_fp8 and is_native_frontend:
        quant_method = getattr(module, "quant_method", None)
        block_size = getattr(
            getattr(quant_method, "quant_config", None),
            "weight_block_size",
            None,
        )
        original_m = getattr(module, "_hpu_orig_M", None)
        original_n = getattr(module, "_hpu_orig_N", None)
        if block_size is not None and original_m is not None and original_n is not None:
            dequant_weight = hpu_ops.dequant_block_fp8_weight_naive(
                module.weight.detach(),
                module.weight_scale_inv.detach(),
                block_size,
                dtype=model_dtype,
                original_M=original_m,
                original_N=original_n,
                do_unpad=True,
            )
            fp8_weight, fp8_scale = hpu_ops.dynamic_quant(dequant_weight)
            fp8_weight = fp8_weight.contiguous()
            fp8_scale = fp8_scale.squeeze(-1).float().contiguous()
            for name, value in (
                ("_hpu_dsv4_native_fp8_weight", fp8_weight),
                ("_hpu_dsv4_native_fp8_weight_scale", fp8_scale),
            ):
                if name in module._buffers:
                    module._buffers[name] = value
                else:
                    module.register_buffer(name, value, persistent=False)
            cached_bytes += (
                fp8_weight.numel() * fp8_weight.element_size()
                + fp8_scale.numel() * fp8_scale.element_size()
            )
    if bf16_cached_bytes:
        module._hpu_dsv4_bf16_attention_cache = True
    return cached_bytes


def _cache_hpu_router_weight(module: torch.nn.Module) -> None:
    if (
        not getattr(module, "allow_router_gemm", False)
        or not hasattr(module, "weight")
    ):
        return

    weight_fp32 = module.weight.detach().to(torch.float32)
    buffer_name = "_hpu_router_weight_fp32"
    if buffer_name in module._buffers:
        module._buffers[buffer_name] = weight_fp32
    else:
        module.register_buffer(
            buffer_name, weight_fp32, persistent=False
        )


def _cache_hpu_dsv4_compressor_norm_weight(
    module_name: str,
    module: torch.nn.Module,
) -> int:
    if (
        not module_name.endswith("compressor.norm")
        or not hasattr(module, "weight")
        or module.weight.numel() not in (128, 512)
    ):
        return 0

    weight_fp32 = module.weight.detach().to(torch.float32).contiguous()
    buffer_name = "_hpu_dsv4_weight_fp32"
    if buffer_name in module._buffers:
        module._buffers[buffer_name] = weight_fp32
    else:
        module.register_buffer(
            buffer_name, weight_fp32, persistent=False
        )
    return weight_fp32.numel() * weight_fp32.element_size()


def hpu_process_weights_after_loading(model, model_config, target_device):
    """Gaudi override: accept device strings (e.g., "hpu")."""
    target_device = torch.device(target_device)
    is_deepseek_v4 = getattr(getattr(model_config, "hf_config", None), "model_type", None) == "deepseek_v4"
    debug_shared_fp8_cache = (
        os.getenv("VLLM_HPU_SHARED_FP8_CACHE_DEBUG") == "1"
    )
    if debug_shared_fp8_cache:
        logger.warning("HPU post-load weight processing started")
    shared_expert_cache_count = 0
    shared_expert_cache_bytes = 0
    dsv4_attention_cache_count = 0
    dsv4_attention_cache_bytes = 0
    dsv4_compressor_norm_cache_count = 0
    dsv4_compressor_norm_cache_bytes = 0
    for module_name, module in model.named_modules():
        cached_bytes = _cache_hpu_dsv4_compressor_norm_weight(module_name, module) if is_deepseek_v4 else 0
        if cached_bytes:
            dsv4_compressor_norm_cache_count += 1
            dsv4_compressor_norm_cache_bytes += cached_bytes
        quant_method = getattr(module, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            #with device_loading_context(module, target_device):
            quant_method.process_weights_after_loading(module)
            if (
                debug_shared_fp8_cache
                and getattr(quant_method, "block_quant", False)
            ):
                logger.warning(
                    "HPU block-FP8 module name=%s weight_shape=%s "
                    "orig=(%s,%s)",
                    module_name,
                    tuple(module.weight.shape),
                    getattr(module, "_hpu_orig_M", None),
                    getattr(module, "_hpu_orig_N", None),
                )
            cached_bytes = (
                _cache_hpu_shared_expert_block_fp8_weight(
                    module_name,
                    module,
                    model_config.dtype,
                )
            ) if is_deepseek_v4 else 0
            if cached_bytes:
                shared_expert_cache_count += 1
                shared_expert_cache_bytes += cached_bytes
            cached_bytes = (
                _cache_hpu_dsv4_attention_block_fp8_weight(
                    module_name,
                    module,
                    model_config.dtype,
                )
            ) if is_deepseek_v4 else 0
            if cached_bytes:
                dsv4_attention_cache_count += 1
                dsv4_attention_cache_bytes += cached_bytes

    if shared_expert_cache_count:
        log_cache_summary = (
            logger.warning
            if debug_shared_fp8_cache
            else logger.info
        )
        log_cache_summary(
            "Cached %d shared-expert block-FP8 weights "
            "(%.1f MiB) in %s",
            shared_expert_cache_count,
            shared_expert_cache_bytes / (1024 * 1024),
            model_config.dtype,
        )
    if dsv4_attention_cache_count:
        logger.info(
            "Cached %d DeepSeek V4 attention decode weights "
            "(%.1f MiB, model dtype %s)",
            dsv4_attention_cache_count,
            dsv4_attention_cache_bytes / (1024 * 1024),
            model_config.dtype,
        )
    if dsv4_compressor_norm_cache_count:
        logger.info(
            "Cached %d DeepSeek V4 compressor RMSNorm weights "
            "(%.1f KiB) in FP32",
            dsv4_compressor_norm_cache_count,
            dsv4_compressor_norm_cache_bytes / 1024,
        )

    if is_deepseek_v4:
        for _, module in model.named_modules():
            _cache_hpu_router_weight(module)

    # Initialize post-load attention weights for both Attention and MLA.
    for _, module in model.named_modules():
        if isinstance(module, (Attention, MLAAttention)) and hasattr(module, "process_weights_after_loading"):
            #with device_loading_context(module, target_device):
            module.process_weights_after_loading(model_config.dtype)

    if model_config.quantization == "torchao":
        set_torchao_reload_attrs(model, model_config)


hpu_utils.process_weights_after_loading = hpu_process_weights_after_loading
base_loader.process_weights_after_loading = hpu_process_weights_after_loading
