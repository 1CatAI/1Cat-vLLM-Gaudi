# SPDX-License-Identifier: Apache-2.0
"""Prepared V4.1 stage program shared by normal loading, compile and replay."""

import json
from itertools import count
from pathlib import Path
from types import FunctionType, MethodType

import torch
from torch import nn
import torch.nn.functional as F
from vllm_gaudi import envs as gaudi_envs

from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut
from vllm_gaudi.ops.deepseek_v41_attention import CSA2Attention, CSA2SharedState
from vllm_gaudi.ops.deepseek_v41_math import (
    engram_update,
    prefill_engram_update,
    hc_post,
    hc_pre,
    prefill_hc_input,
    quantize_activation,
    final_collapse_rms_norm,
    rms_norm,
    unpack_swa,
)
from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
from vllm_gaudi.ops.deepseek_v41_shared_experts import shared_expert_moe
from vllm_gaudi.ops.deepseek_v41_weights import canonical_hash
from vllm_gaudi.ops.deepseek_v41_trace import prefill_scope
from vllm_gaudi.ops.deepseek_v41_native_trace import prefill_span
from vllm_gaudi.ops.deepseek_v41_prefill_regions import (
    clear_prefill_function_regions,
    clear_prefill_regions,
    prefill_function_region,
)


@prefill_span("mhc_post")
@prefill_function_region
def _prefill_hc_post(value, residual, post, comb):
    if gaudi_envs.VLLM_HPU_DSV41_PREFILL_MHC_POST and value.device.type == "hpu":
        return torch.ops.custom_op.custom_deepseek_v41_prefill_mhc_post_gaudi2(
            value.contiguous(), residual.contiguous(), post.contiguous(), comb.contiguous()
        )
    return hc_post(value, residual, post, comb)


@prefill_function_region
def _prefill_combine(output, shared_out):
    return (output.float() + shared_out.float()).to(output.dtype)


@prefill_function_region
def _prefill_router(value, weight, bias, image_bias, image_mask):
    logits = torch.ops.custom_op.custom_deepseek_v41_bf16_linear_f32_gaudi2(value.contiguous(), weight)
    return torch.ops.custom_op.custom_deepseek_v41_router_logits_top6_gaudi2(
        logits.contiguous(), bias, image_bias, image_mask.contiguous()
    )


@prefill_function_region
def _prefill_shared_expert(value, gate_up_weight, down_weight, down_bias, quantize_down):
    # Keep both BF16 boundaries explicit when the compiler combines the
    # projection, activation and downstream projection into one region.
    projected = F.linear(quantize_activation(value), gate_up_weight)
    projected = (
        torch.ops.custom_op.custom_deepseek_v41_bf16_identity_gaudi2(projected.reshape(1, -1))
        .reshape(projected.shape)
        .float()
    )
    gate, up = projected.chunk(2, dim=-1)
    middle = (F.silu(gate.clamp(max=10.0)) * up.clamp(-10.0, 10.0)).to(value.dtype)
    middle = torch.ops.custom_op.custom_deepseek_v41_bf16_identity_gaudi2(middle.reshape(1, -1)).reshape(middle.shape)
    if quantize_down:
        middle = quantize_activation(middle)
    return F.linear(middle, down_weight, down_bias)


def linear(value, layer):
    if getattr(layer, "dense_fp8_direct_input", False):
        from vllm_gaudi.ops.deepseek_v41_qkv import direct_dense_fp8
        return direct_dense_fp8(value, layer.weight, layer.channel_scale)
    if hasattr(layer, "scale"):
        value = quantize_activation(value)
    if getattr(layer, "dense_fp8", False):
        return torch.ops.custom_op.custom_deepseek_v41_dense_fp8_gaudi2(
            value.contiguous(), layer.weight, layer.channel_scale
        )
    return F.linear(value, layer.weight, getattr(layer, "bias", None))


def _weight_tree(specs):
    root = nn.Module()
    dtypes = {
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "U8": torch.uint8,
        "I8": torch.int8,
        "I16": torch.int16,
        "I32": torch.int32,
        "I64": torch.int64,
        "F8_E4M3": torch.bfloat16,
    }
    for name, spec in specs.items():
        fields = name.split(".")
        node = root
        for field in fields[:-1]:
            if field not in node._modules:
                node.add_module(field, nn.Module())
            node = node._modules[field]
        node.register_buffer(fields[-1], torch.empty(spec["shape"], dtype=dtypes[spec["dtype"]], device="meta"))
    return root


def load_weight_tree(
    shard,
    tree,
    device,
    specs=None,
    *,
    woa_sidecar=None,
    woa_layers=(),
    expert_n256_layers=None,
    dense_sidecar=None,
    dense_config=None,
    engram_sidecar=None,
):
    n256 = gaudi_envs.VLLM_HPU_DSV41_EXPERT_N256 or gaudi_envs.VLLM_HPU_DSV41_EXPERT_N256_FP8
    from vllm_gaudi.ops.deepseek_v41_dense_fp8 import projection_prefix, INPUT_PROJECTIONS

    dense_names = {
        projection_prefix(layer, projection) + "weight"
        for projection, layers in (dense_config or {}).items()
        if projection != "version"
        for layer in layers
    }
    for name, spec in (shard.specs if specs is None else specs).items():
        layer = int(name.split(".")[1]) if name.startswith("layers.") else None
        selected_n256 = n256 and layer is not None and (expert_n256_layers is None or layer in expert_n256_layers)
        if engram_sidecar is not None and name in {"layers.1.engram.wkv.weight", "layers.14.engram.wkv.weight"}:
            module = tree.get_submodule(name.rpartition(".")[0])
            module.weight = engram_sidecar.tensor(name, device)
            channel = engram_sidecar.tensor(name.removesuffix("weight") + "channel_scale", device)
            if "channel_scale" in module._buffers:
                module.channel_scale = channel
            else:
                module.register_buffer("channel_scale", channel, False)
            module.dense_fp8 = True
            continue
        if name in dense_names:
            module = tree.get_submodule(name.rpartition(".")[0])
            module.weight = dense_sidecar.tensor(name, device)
            channel = dense_sidecar.tensor(name.removesuffix("weight") + "channel_scale", device)
            if "channel_scale" in module._buffers:
                module.channel_scale = channel
            else:
                module.register_buffer("channel_scale", channel, False)
            module.dense_fp8 = True
            module.dense_fp8_direct_input = any(
                name == projection_prefix(layer, p) + "weight" for p in INPUT_PROJECTIONS
            )
            continue
        if woa_sidecar is not None and name.endswith(".attn.wo_a.weight") and layer in woa_layers:
            module = tree.get_submodule(name.rpartition(".")[0])
            module.weight = woa_sidecar.tensor(name, device)
            channel = woa_sidecar.tensor(name.removesuffix("weight") + "channel_scale", device)
            if "channel_scale" in module._buffers:
                module.channel_scale = channel
            else:
                module.register_buffer("channel_scale", channel, False)
            continue
        if selected_n256 and ".ffn.experts." in name:
            if name.endswith("_s16"):
                continue
            if name.endswith("_q16"):
                from vllm_gaudi.ops.deepseek_v41_expert_n256 import load_projection

                module_name, _, attribute = name.rpartition(".")
                module = tree.get_submodule(module_name)
                projection = attribute.removesuffix("_q16")
                # Retire recipes and release the prior storage before reload.
                # Keep the channel scale under a private name.  Some vLLM
                # quantization modules expose ``w13_channel``/``w2_channel``
                # as boolean capability flags; using that generic name here
                # makes the Python schema binder treat the flag as the FP8
                # tensor.  The native N256 schema requires two real tensors
                # followed by the bool ``normal`` argument.
                for suffix in ("_q16", "_s16", "_fp8_channel"):
                    if projection + suffix not in module._buffers:
                        module.register_buffer(projection + suffix, None, False)
                    else:
                        setattr(module, projection + suffix, None)
                q, scales, channel = load_projection(shard, name.removesuffix("_q16"), device)
                for suffix, value in (("_q16", q), ("_s16", scales), ("_fp8_channel", channel)):
                    setattr(module, projection + suffix, value)
                if not isinstance(channel, torch.Tensor):
                    raise TypeError(f"{name} FP8 channel preparation returned {type(channel).__name__}")
                continue
        value = shard.dense(name, device) if spec["dtype"] == "F8_E4M3" else shard.tensor(name, device)
        # The MLA output projection is consumed as [groups, K, N] by the
        # final einsum. The checkpoint/rank file stores each TP shard in
        # [groups, N, K], which made Synapse insert a 32 MiB DRAM transpose
        # for every captured attention graph. Prepare this one static weight
        # once at load time so the steady graph can feed MME in its native
        # layout. Keep the switch opt-in because old manifests and ordinary
        # prefill graphs still use the checkpoint layout.
        if gaudi_envs.VLLM_HPU_DSV41_PRETRANSPOSE_ATTN and name.endswith(".wo_a.weight") and value.ndim == 2:
            if value.shape[0] % 4:
                raise ValueError(f"MLA wo_a rows are not divisible by 4: {name} {tuple(value.shape)}")
            value = value.reshape(4, value.shape[0] // 4, value.shape[1]).transpose(-1, -2).contiguous()
        if (
            (name.endswith("ffn.gate.weight") and not gaudi_envs.VLLM_HPU_DSV41_BF16_ROUTER_GATE)
            or name.endswith("confidence_head.proj.weight")
            or (name == "head.weight" and not gaudi_envs.VLLM_HPU_DSV41_BF16_LM_HEAD)
            or name == "mtp.2.markov_head.head.weight"
            or (".compressor.w" in name and ".weight" in name and int(name.split(".")[1]) in (2, 8, 14))
        ):
            value = value.float()
        module_name, _, attribute = name.rpartition(".")
        setattr(tree.get_submodule(module_name), attribute, value)
    shard.check_identity()


class PreparedInput(nn.Module):
    """Compile the complete PP0 text input chain with its TP reduction."""

    def __init__(self, embedding, tp_rank, reduce):
        super().__init__()
        self.embedding, self.tp_rank, self.reduce = embedding, tp_rank, reduce

    def forward(self, input_ids):
        ids = input_ids.masked_fill(input_ids == 129265, 129264)
        width = self.embedding.weight.shape[0]
        local = ids.long() - self.tp_rank * width
        valid = (local >= 0) & (local < width)
        value = F.embedding(local.masked_fill(~valid, 0), self.embedding.weight)
        value = self.reduce(value.masked_fill(~valid.unsqueeze(-1), 0))
        residual = value.unsqueeze(1).expand(-1, 4, -1).contiguous()
        pre = torch.zeros(input_ids.numel(), 4, dtype=torch.float32, device=input_ids.device)
        pre[:, 0] = 1
        return residual, pre


class PreparedMoE(nn.Module):
    # Keep ordinary vLLM prompt transactions large (the scheduler still uses
    # max_num_batched_tokens=8192), but cap each HPU N256/attention working
    # tile at 128 tokens.  This is a graph/resource tile, not a C1/C6 decode
    # loop: the full prompt transaction and its token order remain owned by
    # the scheduler, while paged CSA2 state carries the causal prefix between
    # tiles.  A larger tile can exceed the small amount of free HBM left after
    # a 1M KV allocation and force PT_DEVMEM defragmentation while live
    # tensors are still referenced.
    # N256's qualified large-M schemas start at M128.  Keep this at the
    # smallest registered bucket: M64 falls outside the native contract, while
    # larger buckets provide no measured throughput gain and raise peak HBM.
    # The scheduler transaction itself remains the normal C8192 chunk.
    N256_PREFILL_TILE = 128

    def __init__(self, weights, topk, normal_scales, lookup, reduce, tensor_parallel_size=2):
        super().__init__()
        self.weights, self.topk = weights, topk
        self.tensor_parallel_size = tensor_parallel_size
        self.normal_scales, self.reduce = normal_scales, reduce
        self.register_buffer("lookup", lookup, False)
        self.router_top6 = gaudi_envs.VLLM_HPU_DSV41_ROUTER_TOP6
        self.expert_k128 = gaudi_envs.VLLM_HPU_DSV41_EXPERT_K128
        self.fp8 = False
        self.register_buffer("fp8_w13_scale", None, False)
        self.register_buffer("fp8_w2_scale", None, False)
        self.n256 = topk == 6 and (gaudi_envs.VLLM_HPU_DSV41_EXPERT_N256 or gaudi_envs.VLLM_HPU_DSV41_EXPERT_N256_FP8)
        self.n256_fp8 = self.n256 and gaudi_envs.VLLM_HPU_DSV41_EXPERT_N256_FP8
        if tensor_parallel_size == 4 and not self.n256_fp8:
            raise ValueError("TP4 requires the optimized N256 FP8 MoE; generic MoE is not a serving fallback")
        if tensor_parallel_size == 4 and gaudi_envs.VLLM_HPU_DSV41_PREFILL_MXFP4:
            raise ValueError("TP4 prefill requires the V4.1 clipped, FP32-routed prepared MoE path")
        self.prefill_mxfp4 = self.n256 and gaudi_envs.VLLM_HPU_DSV41_PREFILL_MXFP4
        self.prefill_grouped = self.n256 and gaudi_envs.VLLM_HPU_DSV41_PREFILL_GROUPED
        from vllm_gaudi.ops.deepseek_v41_prefill_plan import validate_prefill_plan_config

        validate_prefill_plan_config(n256=self.n256)
        self.n256_fused = self.n256_fp8 and gaudi_envs.VLLM_HPU_DSV41_EXPERT_FUSED_QUANT
        self.n256_fused_reduce = self.n256_fused and gaudi_envs.VLLM_HPU_DSV41_EXPERT_FUSED_REDUCE
        self.concurrent_moe_rows = gaudi_envs.VLLM_HPU_DSV41_CONCURRENT_MOE_ROWS
        self.batch_prefetch_w2 = gaudi_envs.VLLM_HPU_DSV41_BATCH_EXPERT_PREFETCH_W2
        self.batch_route_pack = gaudi_envs.VLLM_HPU_DSV41_BATCH_ROUTE_PACK
        self.batch_expert_reuse = gaudi_envs.VLLM_HPU_DSV41_BATCH_EXPERT_REUSE
        self.batch_w13_horizontal = gaudi_envs.VLLM_HPU_DSV41_BATCH_W13_HORIZONTAL
        self.batch_expert_direct_finalize = gaudi_envs.VLLM_HPU_DSV41_BATCH_EXPERT_DIRECT_FINALIZE
        self.batch_expert_transpose_mme = gaudi_envs.VLLM_HPU_DSV41_BATCH_EXPERT_TRANSPOSE_MME
        if self.batch_expert_transpose_mme:
            if (
                not self.batch_w13_horizontal
                or not self.n256_fused_reduce
                or self.batch_expert_direct_finalize
                or self.batch_prefetch_w2
                or self.batch_expert_reuse
                or self.batch_route_pack
                or self.concurrent_moe_rows
            ):
                raise ValueError("Transposed expert MME requires the unchanged horizontal fused-reduce parent")
            if not hasattr(torch.ops.custom_op, "custom_deepseek_v41_expert_n256_moe_horizontal_transpose_fp8_gaudi2"):
                raise RuntimeError("Transposed expert MME native operator is unavailable")
        if self.batch_expert_direct_finalize and not (self.batch_w13_horizontal and self.n256_fused_reduce):
            raise ValueError("Batch direct finalize requires horizontal N256 and ordered fused reduction")
        if self.batch_expert_direct_finalize and not hasattr(
            torch.ops.custom_op, "custom_deepseek_v41_expert_n256_moe_horizontal_finalize_fp8_gaudi2"
        ):
            raise RuntimeError("Batch expert finalize native operator is unavailable")
        if self.batch_prefetch_w2 and not self.n256_fused_reduce:
            raise ValueError("Batch W2 prefetch requires the N256 fused expert reduction")
        if self.concurrent_moe_rows not in (0, 1, 4, 8, 16):
            raise ValueError("Concurrent MoE rows must be disabled, direct 1, or grouped 4/8/16")
        if self.concurrent_moe_rows:
            if not self.n256_fused_reduce or not self.normal_scales or gaudi_envs.VLLM_HPU_DSV41_DSPARK:
                raise ValueError("Concurrent MoE requires ordinary qualified N256 fused FP8 decode")
            if not hasattr(torch.ops.custom_op, "custom_deepseek_v41_grouped_n256_fp8_gaudi2"):
                raise RuntimeError("Concurrent MoE native operator is unavailable; no fallback was executed")
        # Feature-tiled activation/quantization increased complete-chain latency.
        self.feature_silu = False
        self.router_bf16_gate = gaudi_envs.VLLM_HPU_DSV41_BF16_ROUTER_GATE
        self.shared_gate_up = gaudi_envs.VLLM_HPU_DSV41_SHARED_GATE_UP
        self.register_buffer("shared_gate_up_weight", None, False)
        self.register_buffer("shared_gate_up_channel", None, False)
        self.register_buffer("shared_down_weight", None, False)
        if topk == 6 and gaudi_envs.VLLM_HPU_DSV41_EXPERT_FUSED_QUANT and not self.n256_fp8:
            raise ValueError("Fused expert quantization requires N256 FP8 experts")
        if self.n256 and (gaudi_envs.VLLM_HPU_DSV41_SHARED_C6_EXPERTS or gaudi_envs.VLLM_HPU_DSV41_INDEXED_MOE):
            raise ValueError("N256 prepared storage excludes the old shared/indexed expert layouts")
        if gaudi_envs.VLLM_HPU_DSV41_EXPERT_COORD_PIPELINE and (
            not self.expert_k128 or gaudi_envs.VLLM_HPU_DSV41_FP8_DECODE
        ):
            raise ValueError("V4.1 expert coordinate pipeline requires the K128 BF16 decoder")
        if self.expert_k128 and gaudi_envs.VLLM_HPU_DSV41_FP8_DECODE:
            raise ValueError("V4.1 K128 BF16 and FP8 decode must be selected independently")

    def release_shared_gate_up_weight(self):
        self.shared_gate_up_weight = None
        self.shared_gate_up_channel = None
        self.shared_down_weight = None

    def prepare_shared_gate_up_weight(self):
        if not self.shared_gate_up:
            return
        shared = self.weights.shared_experts
        from vllm_gaudi.ops.deepseek_v41_qkv import concatenate_static_weights

        fp8 = [getattr(p, "dense_fp8_direct_input", False) for p in (shared.w1, shared.w3, shared.w2)]
        if any(fp8) and not all(fp8):
            raise ValueError("Shared FP8 requires all three projections")
        if all(fp8):
            width = shared.w1.weight.shape[0]
            padded = (width + 127) // 128 * 128
            device = shared.w1.weight.device

            def pad_gate(projection):
                weight = projection.weight.cpu()
                return torch.cat((weight, torch.zeros(padded - width, weight.shape[1], dtype=weight.dtype)))

            self.shared_gate_up_weight = concatenate_static_weights(pad_gate(shared.w1), pad_gate(shared.w3)).to(device)
            scales = torch.cat((F.pad(shared.w1.channel_scale.cpu(), (0, padded - width), value=1),
                                F.pad(shared.w3.channel_scale.cpu(), (0, padded - width), value=1)), dim=1)
            self.shared_gate_up_channel = scales.bfloat16().reshape(1, padded * 2 // 256, 256).to(device)
            down = shared.w2.weight.cpu()
            down_padding = torch.zeros(down.shape[0], padded - width, dtype=down.dtype)
            self.shared_down_weight = torch.cat((down, down_padding), dim=1).to(device)
            shared.w2.weight = torch.empty(down.shape, dtype=down.dtype, device="meta")
        else:
            self.shared_gate_up_weight = concatenate_static_weights(shared.w1.weight, shared.w3.weight).contiguous()
        for projection in (shared.w1, shared.w3):
            source = projection.weight
            projection.weight = torch.empty(source.shape, dtype=source.dtype, device="meta")

    def shared_expert(self, value):
        shared = self.weights.shared_experts
        if self.shared_gate_up_channel is not None:
            q, sx = torch.ops.custom_op.custom_deepseek_v41_dense_quant_gaudi2(value.contiguous())
            product = torch.ops.hpu.fp8_gemm_v2(q, False, self.shared_gate_up_weight, True, None,
                                               torch.float32, None, None, None, False)
            rows = value.shape[0]
            ids = torch.zeros((1, rows), dtype=torch.int32, device=value.device)
            route = torch.ones((1, rows), dtype=torch.float32, device=value.device)
            middle, scale = torch.ops.custom_op.custom_deepseek_v41_shared_silu_quant_gaudi2(
                product.reshape(rows, 1, -1), ids, sx, self.shared_gate_up_channel, route)
            return torch.ops.hpu.fp8_gemm_v2(middle.reshape(rows, -1), False, self.shared_down_weight, True,
                                           None, torch.bfloat16, scale.reshape(rows, 1), shared.w2.channel_scale,
                                           None, False)
        if self.shared_gate_up:
            if self.shared_gate_up_weight is None:
                raise RuntimeError("Shared gate/up weight was not prepared before execution")
            projected = F.linear(quantize_activation(value), self.shared_gate_up_weight).float()
            gate, up = projected.chunk(2, dim=-1)
            gate = gate.clamp(max=10.0)
            up = up.clamp(-10.0, 10.0)
        else:
            gate = linear(value, shared.w1).float().clamp(max=10.0)
            up = linear(value, shared.w3).float().clamp(-10.0, 10.0)
        return linear((F.silu(gate) * up).to(value.dtype), shared.w2)

    def _can_use_indexed(self, value, ids, routing):
        """Static contract for the direct Q16 TPC candidate.

        The candidate is deliberately limited to the routed V4.1 experts. DSpark
        MTP uses a different (128 expert/top-3) checkpoint and must continue to
        use the prepared BF16/MME compound op until it gets its own kernel.
        """
        experts = self.weights.experts
        return (
            gaudi_envs.VLLM_HPU_DSV41_INDEXED_MOE
            and self.topk == 6
            and value.device.type == "hpu"
            and value.dtype == torch.bfloat16
            and value.ndim == 2
            and 1 <= value.shape[0] <= 6
            and value.shape[-1] == 5120
            and ids.device == value.device
            and ids.dtype == torch.int64
            and ids.ndim == 2
            and ids.shape[-1] == 6
            and routing.device == value.device
            and routing.dtype == torch.float32
            and routing.shape == ids.shape
            and experts.w13_q16.shape == (384, 18, 163840)
            and experts.w2_q16.shape == (384, 40, 36864)
            and experts.w13_s16.shape == (384, 18, 20480)
            and experts.w2_s16.shape == (384, 40, 4608)
        )

    def _forward_indexed(self, value, ids, routing):
        """Compute six routed experts without a decoded-weight tensor.

        Q16's N-major layout lets one TPC program accumulate a 128-row output
        tile in registers. The two custom calls emit only gate/up and down
        activations; BF16 boundaries remain in the same places as the prepared
        MME path before FP32 SwiGLU and ordered router accumulation.
        """
        experts = self.weights.experts
        gate_up = torch.ops.custom_op.custom_deepseek_v41_mxfp4_indexed_fc1_bf16_gaudi2(
            value, ids.to(torch.int32), experts.w13_q16, experts.w13_s16, self.lookup, self.normal_scales
        )
        intermediate = gate_up.shape[-1] // 2
        gate = gate_up[..., :intermediate].float().clamp(max=10.0)
        up = gate_up[..., intermediate:].float().clamp(-10.0, 10.0)
        # The reference path applies router weights before the BF16 boundary
        # feeding W2.  Keep that ordering so the direct kernel is only a data
        # path change, not a numerical/rounding change.
        routed_input = F.silu(gate) * up * routing.float().unsqueeze(-1)
        routed_input = routed_input.to(value.dtype)
        down = torch.ops.custom_op.custom_deepseek_v41_mxfp4_indexed_fc2_bf16_gaudi2(
            routed_input, ids.to(torch.int32), experts.w2_q16, experts.w2_s16, self.lookup, self.normal_scales
        )
        result = down.float()[:, 0]
        for expert in range(1, 6):
            result = result + down.float()[:, expert]
        return result.to(value.dtype)

    def _forward_n256_fp8(self, value, ids, routing, *, ordinary_decode=False, prequant=None, shared=None):
        """Run the resident FP8 N256 body for decode and bounded prefill."""
        experts = self.weights.experts
        if ordinary_decode and (self.batch_expert_reuse or self.batch_route_pack) and value.shape[0] > 1:
            if prequant is not None or shared is not None:
                raise ValueError("Batch expert reuse has an independent quantization/finalization contract")
            from vllm_gaudi.ops.deepseek_v41_grouped_decode import grouped_decode

            return grouped_decode(
                value,
                ids.to(torch.int32),
                routing.float(),
                experts.w13_q16,
                experts.w2_q16,
                experts.w13_s16,
                experts.w2_s16,
                self.lookup,
                experts.w13_fp8_channel,
                experts.w2_fp8_channel,
                rows=1,
                reuse_weights=True,
                native_pack=self.batch_route_pack,
            )
        if ordinary_decode and self.concurrent_moe_rows and value.shape[0] == 64:
            from vllm_gaudi.ops.deepseek_v41_grouped_decode import grouped_decode

            return grouped_decode(
                value,
                ids.to(torch.int32),
                routing.float(),
                experts.w13_q16,
                experts.w2_q16,
                experts.w13_s16,
                experts.w2_s16,
                self.lookup,
                experts.w13_fp8_channel,
                experts.w2_fp8_channel,
                rows=self.concurrent_moe_rows,
            )

        def one_tile(tile_value, tile_ids, tile_routing, tile_prequant, tile_shared):
            operands = (
                tile_value,
                tile_ids.to(torch.int32),
                tile_routing.float(),
                experts.w13_q16,
                experts.w2_q16,
                experts.w13_s16,
                experts.w2_s16,
                self.lookup,
            )
            use_fused = self.n256_fused and (tile_value.shape[0] <= 6 or ordinary_decode)
            # C1 keeps the same public scheduler/state contract as every
            # other bucket.  Inside the expert compound node, finish W2
            # directly from its FP32 accumulator so the six BF16 route rows
            # are rounded and reduced in routing order without an HBM
            # intermediate.  C2+ continues to use the normal fused body.
            # W2 weights depend only on the selected expert IDs.  The
            # qualified C1 plan emits their decode before the independent
            # W13 MME so TPC preparation can overlap matrix execution.
            # Both N256 FP8 schemas consume the per-output channel scales;
            # the fused variant additionally folds the SwiGLU/quant boundary.
            channel13 = experts.w13_fp8_channel
            channel2 = experts.w2_fp8_channel
            if not isinstance(channel13, torch.Tensor) or not isinstance(channel2, torch.Tensor):
                raise TypeError("N256 FP8 channel scales were not loaded as tensors")
            if self.tensor_parallel_size == 4 and use_fused and tile_prequant is None:
                from vllm_gaudi.ops.deepseek_v41_expert_n256 import run_fused_decode

                return run_fused_decode(
                    *operands, channel13, channel2, bool(self.normal_scales), direct_finalize=self.n256_fused_reduce
                )
            if tile_prequant is not None:
                if not use_fused:
                    raise ValueError("Prequantized N256 input requires the fused expert body")
                quantized, activation_scale = tile_prequant
                if tile_shared is not None:
                    if not self.n256_fused_reduce or tile_value.shape[0] != 1:
                        raise ValueError("Shared finalize requires the C1 prequant direct-finalize path")
                    op = torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_prequant_direct_finalize_shared_prefetch_w2_fp8_gaudi2
                    if self.feature_silu:
                        op = torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_prequant_shared_feature_silu_fp8_gaudi2
                    return op(
                        *operands,
                        channel13,
                        channel2,
                        quantized,
                        activation_scale,
                        tile_shared,
                        bool(self.normal_scales),
                    )
                op = (
                    torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_prequant_direct_finalize_prefetch_w2_fp8_gaudi2
                    if self.n256_fused_reduce and tile_value.shape[0] == 1
                    else torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_prequant_horizontal_fp8_gaudi2
                    if self.n256_fused_reduce and ordinary_decode and self.batch_w13_horizontal
                    else torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_prequant_fused_reduce_fp8_gaudi2
                    if self.n256_fused_reduce and ordinary_decode
                    else torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_prequant_fused_fp8_gaudi2
                )
                return op(*operands, channel13, channel2, quantized, activation_scale, bool(self.normal_scales))
            op = (
                torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_direct_finalize_prefetch_w2_fp8_gaudi2
                if self.n256_fused_reduce and tile_value.shape[0] == 1
                else torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_horizontal_transpose_fp8_gaudi2
                if ordinary_decode and self.batch_expert_transpose_mme
                else torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_horizontal_finalize_fp8_gaudi2
                if ordinary_decode and self.batch_expert_direct_finalize
                else torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_horizontal_fp8_gaudi2
                if ordinary_decode and self.batch_w13_horizontal and self.n256_fused_reduce
                else torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_batch_prefetch_w2_fp8_gaudi2
                if ordinary_decode and self.batch_prefetch_w2
                else torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_fused_reduce_fp8_gaudi2
                if self.n256_fused_reduce and ordinary_decode
                else torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_fused_fp8_gaudi2
                if use_fused
                else torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_fp8_gaudi2
            )
            return op(*operands, channel13, channel2, bool(self.normal_scales))

        tokens = int(value.shape[0])
        if tokens <= self.N256_PREFILL_TILE:
            return one_tile(value, ids, routing, prequant, shared)
        if shared is not None:
            raise ValueError("Shared finalize cannot cross N256 prefill tiles")
        pieces = []
        for begin in range(0, tokens, self.N256_PREFILL_TILE):
            end = min(tokens, begin + self.N256_PREFILL_TILE)
            tile_prequant = None if prequant is None else (prequant[0][begin:end], prequant[1][begin:end])
            pieces.append(one_tile(value[begin:end], ids[begin:end], routing[begin:end], tile_prequant, None))
        return torch.cat(pieces, dim=0)

    def _router_logits(self, value, prefill_tokens=0):
        tokens = value.shape[0]
        if prefill_tokens:
            from vllm_gaudi.ops.deepseek_v41_prefill_capacity import MAX_PREFILL_TOKENS

            if not tokens <= prefill_tokens <= MAX_PREFILL_TOKENS:
                raise ValueError("Halo router requires the original bounded prefill row count")
        if prefill_tokens > tokens:
            # A changed M shape can alter the MME's FP32 accumulation order.
            # Keep the original router geometry and suffix row placement;
            # tiny gate differences otherwise change FP32 routing weights and
            # amplify through the remaining decoder layers. Other prompt rows
            # have no dependency on these row-independent gate projections.
            original = value.new_zeros((prefill_tokens, value.shape[-1]))
            original[-tokens:].copy_(value)
            value = original
        logits = (
            torch.ops.custom_op.custom_deepseek_v41_bf16_linear_f32_gaudi2(value.contiguous(), self.weights.gate.weight)
            if self.router_bf16_gate
            else F.linear(value.float(), self.weights.gate.weight)
        )
        return logits[-tokens:] if prefill_tokens > tokens else logits

    @prefill_span("moe")
    @prefill_scope("moe")
    def forward(
        self,
        value,
        image_mask,
        ready_outputs=(),
        *,
        fp8_decode=False,
        ordinary_decode=False,
        decode=False,
        prequant=None,
        prefill_router_tokens=0,
        prefill_sequence=False,
    ):
        decode = decode or ordinary_decode
        if decode and prefill_router_tokens:
            raise ValueError("Decoder halo router padding is confined to prefill")
        w = self.weights
        gate_logits = self._router_logits(value, prefill_router_tokens)
        if self.router_top6 and decode:
            if self.topk != 6 or gate_logits.shape[1] != 384:
                raise ValueError("Native V4.1 Router requires 384 experts and top6")
            # Keep the deployed Gaudi2 softplus/sqrt instruction chains and
            # the ordered top-6 reduction in one TPC consumer.  This avoids
            # materialising all 384 scores while preserving the stock
            # BF16-gate -> FP32-score result bit for bit.
            ids, routing = torch.ops.custom_op.custom_deepseek_v41_router_logits_top6_gaudi2(
                gate_logits.contiguous(), w.gate.bias, w.gate.bias_vl, image_mask.contiguous()
            )
        else:
            scores = F.softplus(gate_logits).sqrt()
            if self.router_top6:
                # Large-M prefill retains the already-qualified native
                # score-to-top6 path.  Falling through to torch.topk here
                # changes the parent's tie/order contract and was the cause
                # of the long-prompt output divergence in the decode-only
                # fused-router candidate.
                ids, routing = torch.ops.custom_op.custom_deepseek_v41_router_top6_gaudi2(
                    scores, w.gate.bias, w.gate.bias_vl, image_mask
                )
            else:
                bias = torch.where(image_mask.unsqueeze(-1), w.gate.bias_vl, w.gate.bias)
                ids = torch.topk(scores + bias, self.topk, dim=-1, sorted=True).indices
                routing = scores.gather(1, ids)
                routing = routing / (routing.sum(-1, keepdim=True) + 1e-20) * 1.5
            del scores
        del gate_logits
        experts = w.experts
        # The C1 prequant path can keep the exact BF16 routed/shared consumer
        # in the same native graph. The shared branch still computes from the
        # normal request-slot input and remains independently schedulable;
        # only its final FP32 addition and BF16 rounding move into the routed
        # compound node. Wider batches keep the common batch implementation.
        fused_shared = self.n256_fp8 and self.n256_fused_reduce and prequant is not None and value.shape[0] == 1
        shared_out = self.shared_expert(value) if fused_shared else None
        if self.prefill_grouped and value.shape[0] > 6 and not ordinary_decode:
            from vllm_gaudi.ops.deepseek_v41_grouped_prefill import run_grouped_prefill

            output = run_grouped_prefill(
                value,
                ids,
                routing,
                experts.w13_q16,
                experts.w2_q16,
                experts.w13_s16,
                experts.w2_s16,
                self.lookup,
                self.normal_scales,
                experts.w13_fp8_channel,
                experts.w2_fp8_channel,
            )
        elif self.prefill_mxfp4 and value.shape[0] > 6 and not ordinary_decode:
            # Large-M prompt work has a different reuse regime from C1.  The
            # decode-oriented N256 compound kernel rereads and converts all
            # routed weights once per small token tile.  Restore one bounded
            # expert range at a time and let Habana's stock MXFP4 FusedMoE
            # reuse it across the complete scheduler transaction.  The
            # restored tensors are recipe temporaries; N256 remains the only
            # resident expert allocation and the C1 path below is unchanged.
            from vllm_gaudi.ops.deepseek_v41_prefill_moe import run_q16_prefill_moe

            output = run_q16_prefill_moe(
                value, ids, routing, experts.w13_q16, experts.w2_q16, experts.w13_s16, experts.w2_s16
            )
        elif self.n256:
            # The N256 FP8 compound node is the only large-M implementation
            # qualified on this prepared layout.  It accepts C1..C8192; the
            # outer stage wrapper tiles a normal vLLM scheduler transaction at
            # N256_PREFILL_TILE so the node's conversion workspace is released
            # before the next tile.  This preserves the regular
            # max_num_batched_tokens=8192 contract without restoring a second
            # checkpoint-order weight layout or falling into the BF16 N256
            # dequant node, which is not registered for the current Gaudi
            # runtime shape profile (M128 generic failure).
            if self.n256_fp8:
                output = self._forward_n256_fp8(
                    value, ids, routing, ordinary_decode=ordinary_decode, prequant=prequant, shared=shared_out
                )
            else:
                operands = (
                    value,
                    ids.to(torch.int32),
                    routing.float(),
                    experts.w13_q16,
                    experts.w2_q16,
                    experts.w13_s16,
                    experts.w2_s16,
                    self.lookup,
                )
                output = torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_bf16_gaudi2(
                    *operands, self.normal_scales
                )
        elif self.fp8 and fp8_decode:
            if value.shape[0] != 1:
                raise ValueError("The legacy FP8 expert path requires C1")
            output = torch.ops.custom_op.custom_deepseek_v41_mxfp4_prepared_moe_fp8_gaudi2(
                value,
                ids.to(torch.int32),
                routing.float(),
                experts.w13_q16,
                experts.w2_q16,
                experts.w13_s16,
                experts.w2_s16,
                self.lookup,
                self.fp8_w13_scale,
                self.fp8_w2_scale,
                self.normal_scales,
            )
        elif self.expert_k128 and self.topk == 6 and value.shape[0] == 1:
            namespace = torch.ops.custom_op
            name = "custom_deepseek_v41_mxfp4_prepared_moe_k128_bf16_gaudi2"
            op = (
                getattr(namespace, name)
                if hasattr(namespace, name)
                else namespace.custom_deepseek_v41_mxfp4_k128_moe_bf16_gaudi2
            )
            output = op(
                value,
                ids.to(torch.int32),
                routing.float(),
                experts.w13_q16,
                experts.w2_q16,
                experts.w13_s16,
                experts.w2_s16,
                self.lookup,
                self.normal_scales,
            )
        elif (
            gaudi_envs.VLLM_HPU_DSV41_SHARED_C6_EXPERTS
            and self.topk == 6
            and value.device.type == "hpu"
            and 2 <= value.shape[0] <= 6
        ):
            output = shared_expert_moe(value, ids, routing.float(), experts, self.lookup, self.normal_scales)
        elif self._can_use_indexed(value, ids, routing):
            output = self._forward_indexed(value, ids, routing)
        else:
            op = torch.ops.custom_op.custom_deepseek_v41_mxfp4_prepared_moe_bf16_gaudi2
            if (
                gaudi_envs.VLLM_HPU_DSV41_TILED_EXPERT_DECODE
                and self.topk == 6
                and value.device.type == "hpu"
                and 2 <= value.shape[0] <= 6
            ):
                op = torch.ops.custom_op.custom_deepseek_v41_mxfp4_k128_moe_bf16_gaudi2
            if (
                gaudi_envs.VLLM_HPU_DSV41_W13_N512
                and self.topk == 6
                and value.device.type == "hpu"
                and 2 <= value.shape[0] <= 6
            ):
                op = torch.ops.custom_op.custom_deepseek_v41_mxfp4_n512_moe_bf16_gaudi2
            output = op(
                value,
                ids.to(torch.int32),
                routing.float(),
                experts.w13_q16,
                experts.w2_q16,
                experts.w13_s16,
                experts.w2_s16,
                self.lookup,
                self.normal_scales,
            )
        if shared_out is None:
            shared_out = self.shared_expert(value)
            partial = (
                _prefill_combine(output, shared_out)
                if gaudi_envs.VLLM_HPU_DSV41_PREFILL_REGIONS and not decode and value.shape[0] > 6
                else (output.float() + shared_out.float()).to(value.dtype)
            )
        else:
            partial = output
        if prefill_sequence:
            if decode or ready_outputs or self.tensor_parallel_size != 4:
                raise ValueError("Token-owned MoE output is restricted to TP4 prefill")
            from vllm_gaudi.ops.deepseek_v41_prefill_sequence_state import reduce_owned_tokens

            return reduce_owned_tokens(partial, self.reduce)
        return self.reduce(partial, ready_outputs=ready_outputs) if ready_outputs else self.reduce(partial)


class PreparedDecoderLayer(nn.Module):
    def __init__(
        self,
        weights,
        config,
        layer,
        shared,
        normal_scales,
        lookup,
        reduce,
        all_gather,
        device,
        tensor_parallel_size=2,
        collect_target_state=False,
    ):
        super().__init__()
        self.weights, self.layer = weights, layer
        self.draft = layer >= config["num_hidden_layers"]
        self.collect_target_state = collect_target_state and layer in (37, 38, 39)
        self.eps, self.hc_eps, self.iterations = config["rms_norm_eps"], config["hc_eps"], config["hc_sinkhorn_iters"]
        self.batch_main_fusions = gaudi_envs.VLLM_HPU_DSV41_BATCH_MAIN_FUSIONS
        self.batch_mhc_fusion = self.batch_main_fusions
        self.batch_ffn_fusion = self.batch_main_fusions or gaudi_envs.VLLM_HPU_DSV41_BATCH_C1_NUMERICS
        self.batch_control_reuse = gaudi_envs.VLLM_HPU_DSV41_MHC_BATCH_REUSE
        self.batch_control_prefetch = gaudi_envs.VLLM_HPU_DSV41_MHC_CONTROL_PREFETCH
        self.mhc_control_rrms = gaudi_envs.VLLM_HPU_DSV41_MHC_CONTROL_RRMS
        self.decode_attention_norm_quant = False
        # Full prompt state already owns token rows. Reuse that ownership for
        # replicated Q/KV inputs on layers without a full hidden-state consumer.
        self.sequence_qkv_input = tensor_parallel_size == 4
        self.mhc_post_collapse = tensor_parallel_size == 4 and hasattr(
            torch.ops.custom_op, "custom_deepseek_v41_mhc_post_collapse_gaudi2"
        )
        # Retained for diagnosis: the complete TP4 chain regresses with the
        # inter-layer handoff even when every numerical gate is exact.
        self.mhc_interlayer_collapse = False
        self.register_buffer("hc_attn_fn_packed", None, False)
        self.register_buffer("hc_ffn_fn_packed", None, False)
        if shared.length > 512:
            from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2Attention

            self.attention = PagedCSA2Attention(
                weights.attn,
                config,
                layer,
                shared,
                linear,
                reduce,
                all_gather,
                device,
                tensor_parallel_size=tensor_parallel_size,
            )
        else:
            self.attention = CSA2Attention(
                weights.attn, config, layer, shared, linear, reduce, device, tensor_parallel_size=tensor_parallel_size
            )
        self.moe = PreparedMoE(
            weights.ffn,
            config["num_experts_per_tok"],
            normal_scales,
            lookup,
            reduce,
            tensor_parallel_size=tensor_parallel_size,
        )
        self.moe.layer = layer
        self.all_gather = all_gather

    @staticmethod
    def _pack_mhc_control_weight(weight):
        if weight.dtype != torch.float32 or weight.shape != (24, 20480):
            raise ValueError(
                f"mHC control/RRMS preparation requires FP32 [24,20480], got {weight.dtype} {tuple(weight.shape)}"
            )
        # The fused kernel reconstructs linear BF16 lanes before its MACs, so
        # it consumes the checkpoint's original K order and does not need a
        # second lane-permuted resident copy.
        return weight.contiguous()

    def prepare_mhc_control_weights(self):
        self.release_mhc_control_weights()
        if not self.mhc_control_rrms:
            return
        self.hc_attn_fn_packed = self._pack_mhc_control_weight(self.weights.hc_attn_fn)
        self.hc_ffn_fn_packed = self._pack_mhc_control_weight(self.weights.hc_ffn_fn)

    def release_mhc_control_weights(self):
        self.hc_attn_fn_packed = None
        self.hc_ffn_fn_packed = None

    @prefill_span("layer")
    def forward(
        self,
        residual,
        pre_mix,
        positions,
        image_mask,
        engram_rows=None,
        *,
        fp8_decode=False,
        decode=False,
        prefill_router_tokens=0,
        prefill_sequence=False,
        selected_main=None,
        decode_metadata=None,
        collapse_handoff=None,
        publish_collapse=False,
    ):
        w = self.weights
        collapsed_attention = None if collapse_handoff is None else collapse_handoff.pop(self.layer, None)
        if hasattr(w, "engram") and collapsed_attention is not None:
            raise RuntimeError("An Engram update invalidates a preceding residual collapse")
        prefill = (
            gaudi_envs.VLLM_HPU_DSV41_PREFILL_REGIONS
            and not decode
            and not self.draft
            and residual.shape[0] > 6
            and hasattr(self.attention, "_prefill_attention")
        )
        # The fused update depends on the four mHC streams, not TP heads.
        # TP4 needs it even before the other prefill regions are enabled:
        # the eager broadcast otherwise materializes [T,4,4,5120] FP32.
        fused_post = prefill or (self.moe.tensor_parallel_size == 4 and residual.shape[0] > 6)
        post_update = _prefill_hc_post if fused_post else hc_post
        if prefill_sequence:
            if not prefill or self.moe.tensor_parallel_size != 4 or residual.shape[0] * 4 != positions.numel():
                raise ValueError("TP4 prompt layer requires matching token-owned residuals and full positions")
            from vllm.distributed import get_tp_group
            from vllm_gaudi.ops.deepseek_v41_prefill_sequence_state import (
                exchange_engram_tokens,
                gather_tokens,
                sequence_hc_input,
                token_owner,
            )

            group = get_tp_group()
            owned = token_owner(positions.numel(), group.rank_in_group)
        if hasattr(w, "engram"):
            if engram_rows is None:
                raise RuntimeError("Engram layer requires its completed host gather and DMA generation")
            if prefill_sequence:
                packet = exchange_engram_tokens(engram_rows, group=group.device_group)
                rows = packet if packet.dtype == torch.bfloat16 else unpack_swa(packet, 256)
                local_rows = None
            else:
                local_rows = engram_rows if engram_rows.dtype == torch.bfloat16 else unpack_swa(engram_rows, 256)
                rows = self.all_gather(local_rows, dim=1)
            kv = linear(rows.flatten(1), w.engram.wkv)
            update = prefill_engram_update if prefill else engram_update
            active_mask = ~image_mask[owned] if prefill_sequence else ~image_mask
            residual = update(residual, kv, w.engram.q_weight, w.engram.k_weight, active_mask, self.eps)
            del kv, rows, local_rows
        target_state = residual.mean(1) if self.collect_target_state else None
        if target_state is not None and prefill_sequence:
            target_state = gather_tokens(target_state.contiguous(), group=group.device_group)
        compiled_input = prefill and gaudi_envs.VLLM_HPU_DSV41_PREFILL_MHC_INPUT
        requires_full_input = self.attention.owns_kv or self.attention.owns_index if prefill_sequence else False
        sequence_qkv = (
            prefill_sequence
            and getattr(self, "sequence_qkv_input", False)
            and not requires_full_input
            and self.attention._fused_qkv_weight is not None
        )
        input_prequant = None
        if prefill_sequence:
            new_pre, post, comb, value = sequence_hc_input(
                residual,
                pre_mix,
                w.hc_attn_fn,
                w.hc_attn_scale,
                w.hc_attn_base,
                w.attn_norm.weight,
                self.eps,
                self.hc_eps,
                self.iterations,
                self.hc_attn_fn_packed,
                group=group.device_group,
                gather=requires_full_input or not sequence_qkv,
            )
        elif compiled_input:
            collapsed, new_pre, post, comb, value = prefill_hc_input(
                residual,
                pre_mix,
                w.hc_attn_fn,
                w.hc_attn_scale,
                w.hc_attn_base,
                w.attn_norm.weight,
                self.eps,
                self.hc_eps,
                self.iterations,
                self.hc_attn_fn_packed,
            )
            del collapsed
        else:
            value, new_pre, post, comb = hc_pre(
                residual,
                pre_mix,
                w.hc_attn_fn,
                w.hc_attn_scale,
                w.hc_attn_base,
                self.eps,
                self.hc_eps,
                self.iterations,
                packed_fn=self.hc_attn_fn_packed,
                prefill=fused_post,
                collapsed_input=collapsed_attention,
            )
            if (
                self.decode_attention_norm_quant and decode and not self.draft
                and value.dtype == torch.bfloat16 and 1 <= value.shape[0] <= 6
                and "fused_qkv_channel" in self.attention._buffers
            ):
                value, quantized, activation_scale = (
                    torch.ops.custom_op.custom_deepseek_v41_attention_norm_quant_gaudi2(
                        value.contiguous(), w.attn_norm.weight, self.eps
                    )
                )
                input_prequant = quantized, activation_scale
            else:
                value = rms_norm(value, w.attn_norm.weight, self.eps)
            if collapsed_attention is not None and input_prequant is None:
                # The generic attention graph consumes an FP32 collapse,
                # then rounds the normalized row before block quantization.
                value = value.to(residual.dtype)
        schedule = gaudi_envs.VLLM_HPU_DSV41_MHC_SCHEDULE and not self.draft and 1 <= value.shape[0] <= 6
        attention_kwargs = {"decode": decode}
        if input_prequant is not None:
            attention_kwargs["input_prequant"] = input_prequant
        if selected_main is not None and getattr(self.attention, "shared_main_mla", False):
            attention_kwargs["selected_main"] = selected_main
        if decode_metadata is not None and getattr(self.attention, "shared_decode_metadata", False):
            attention_kwargs["decode_metadata"] = decode_metadata
        if prefill_sequence:
            value = self.attention(
                value, positions, decode=False, prefill_sequence=True, prefill_qkv_sequence=sequence_qkv
            )
        elif schedule:
            value = self.attention(value, positions, ready_outputs=(post, comb), **attention_kwargs)
        else:
            value = (
                self.attention.draft(value, positions)
                if self.draft
                else self.attention(value, positions, **attention_kwargs)
            )
        collapsed_ffn = None
        if self.mhc_post_collapse and decode and not self.draft and value.shape[0] <= 2 and value.device.type == "hpu":
            # Keep forty independently scheduled feature tiles. The residual
            # BF16 boundary is retained inside the fused producer; the next
            # control/RRMS and FFN norm/quant remain independent consumers.
            residual, collapsed_ffn = torch.ops.custom_op.custom_deepseek_v41_mhc_post_collapse_gaudi2(
                value.contiguous(), residual.contiguous(), post.contiguous(), comb.contiguous(), new_pre.contiguous()
            )
        else:
            residual = post_update(value, residual, post, comb)
        del value, post, comb
        if prefill_sequence:
            pre_mix, post, comb, value = sequence_hc_input(
                residual,
                new_pre,
                w.hc_ffn_fn,
                w.hc_ffn_scale,
                w.hc_ffn_base,
                w.ffn_norm.weight,
                self.eps,
                self.hc_eps,
                self.iterations,
                self.hc_ffn_fn_packed,
                group=group.device_group,
            )
        elif compiled_input:
            collapsed, pre_mix, post, comb, value = prefill_hc_input(
                residual,
                new_pre,
                w.hc_ffn_fn,
                w.hc_ffn_scale,
                w.hc_ffn_base,
                w.ffn_norm.weight,
                self.eps,
                self.hc_eps,
                self.iterations,
                self.hc_ffn_fn_packed,
            )
            del collapsed
        else:
            value, pre_mix, post, comb = hc_pre(
                residual,
                new_pre,
                w.hc_ffn_fn,
                w.hc_ffn_scale,
                w.hc_ffn_base,
                self.eps,
                self.hc_eps,
                self.iterations,
                packed_fn=self.hc_ffn_fn_packed,
                prefill=fused_post,
                collapsed_input=collapsed_ffn,
            )
        # Keep the normalized BF16 row in the TPC register file while forming
        # the exact FP8 operand consumed by N256.  This is an internal B1/B2
        # implementation choice under the normal scheduler and request-state
        # contract; larger decode batches and prefill retain the generic path.
        # The B1/B2 outputs were qualified bit-for-bit against the separate
        # RMSNorm and dynamic-quant nodes before this became the default.
        if decode and value.shape[0] <= 2 and self.moe.n256_fp8 and self.moe.n256_fused:
            normalized, quantized, activation_scale = torch.ops.custom_op.custom_deepseek_v41_ffn_norm_quant_gaudi2(
                value.contiguous(), w.ffn_norm.weight, self.eps
            )
            value = self.moe(
                normalized,
                image_mask,
                ready_outputs=(post, comb) if schedule else (),
                fp8_decode=fp8_decode,
                decode=decode,
                prequant=(quantized, activation_scale),
            )
        else:
            value = self.moe(
                value if compiled_input else rms_norm(value, w.ffn_norm.weight, self.eps),
                image_mask,
                ready_outputs=(post, comb) if schedule else (),
                fp8_decode=fp8_decode,
                decode=decode,
                prefill_router_tokens=prefill_router_tokens,
                prefill_sequence=prefill_sequence,
            )
        if (
            publish_collapse
            and collapse_handoff is not None
            and self.mhc_interlayer_collapse
            and decode
            and not self.draft
            and value.shape[0] <= 2
            and value.device.type == "hpu"
        ):
            # The group only publishes to the immediately following layer,
            # without an intervening Engram update. Both consumers receive
            # the same BF16 residual and FP32 collapse as separate layer recipes.
            residual, collapsed = torch.ops.custom_op.custom_deepseek_v41_mhc_post_collapse_f32_gaudi2(
                value.contiguous(), residual.contiguous(), post.contiguous(), comb.contiguous(), pre_mix.contiguous()
            )
            collapse_handoff[self.layer + 1] = collapsed
        else:
            residual = post_update(value, residual, post, comb)
        return residual, pre_mix, target_state

    def forward_batch(
        self,
        residual,
        pre_mix,
        positions,
        image_mask,
        engram_rows,
        slots,
        pages,
        selected,
        candidates,
        main_ready,
        index_ready,
    ):
        """One ordinary decode input per request, with explicit state owners."""
        if self.draft or self.collect_target_state:
            raise RuntimeError("Request batching excludes speculative state collection")
        w = self.weights
        active = (slots >= 0) & (positions >= 0)
        if hasattr(w, "engram"):
            if engram_rows is None:
                raise RuntimeError("Batched Engram requires a completed request-owned packet")
            local = engram_rows if engram_rows.dtype == torch.bfloat16 else unpack_swa(engram_rows, 256)
            rows = self.all_gather(local, dim=1)
            kv = linear(rows.flatten(1), w.engram.wkv)
            residual = engram_update(residual, kv, w.engram.q_weight, w.engram.k_weight, active & ~image_mask, self.eps)
        value, new_pre, post, comb = hc_pre(
            residual,
            pre_mix,
            w.hc_attn_fn,
            w.hc_attn_scale,
            w.hc_attn_base,
            self.eps,
            self.hc_eps,
            self.iterations,
            request_batch=True,
            batch_control_reuse=self.batch_control_reuse,
            batch_control_prefetch=self.batch_control_prefetch,
            packed_fn=self.hc_attn_fn_packed if self.batch_mhc_fusion else None,
        )
        schedule = gaudi_envs.VLLM_HPU_DSV41_MHC_SCHEDULE
        result = self.attention.forward_batch(
            rms_norm(value, w.attn_norm.weight, self.eps, request_batch=True),
            positions,
            slots,
            pages,
            selected,
            candidates,
            main_ready,
            index_ready,
            ready_outputs=(post, comb) if schedule else (),
        )
        value, selected, candidates, main_ready, index_ready = result
        residual = hc_post(value, residual, post, comb)
        value, pre_mix, post, comb = hc_pre(
            residual,
            new_pre,
            w.hc_ffn_fn,
            w.hc_ffn_scale,
            w.hc_ffn_base,
            self.eps,
            self.hc_eps,
            self.iterations,
            request_batch=True,
            batch_control_reuse=self.batch_control_reuse,
            batch_control_prefetch=self.batch_control_prefetch,
            packed_fn=self.hc_ffn_fn_packed if self.batch_mhc_fusion else None,
        )
        # Reuse the production norm/quant producer for ordinary request rows.
        # The grouped B64 specialization owns its own activation preparation;
        # do not compute an unused FP8 operand on that path.
        prequant = None
        if (
            self.batch_ffn_fusion
            and self.moe.n256_fused
            and not (value.shape[0] == 64 and self.moe.concurrent_moe_rows)
        ):
            value, quantized, scale = torch.ops.custom_op.custom_deepseek_v41_ffn_norm_quant_gaudi2(
                value.contiguous(), w.ffn_norm.weight, self.eps
            )
            prequant = quantized, scale
        else:
            value = rms_norm(value, w.ffn_norm.weight, self.eps, request_batch=True)
        value = self.moe(
            value, image_mask, ready_outputs=(post, comb) if schedule else (), ordinary_decode=True, prequant=prequant
        )
        residual = hc_post(value, residual, post, comb)
        residual = residual.masked_fill(~active[:, None, None], 0)
        pre_mix = pre_mix.masked_fill(~active[:, None], 0)
        return residual, pre_mix, selected, candidates, main_ready, index_ready


class PrefillInput:
    """Transfer a prompt's initial residual into the layer loop exactly once.

    Outer Python/module call frames retain the empty owner, so they do not
    keep the initial four-stream activation alive after layer zero consumes
    it. This changes tensor lifetime only; embedding and Engram overlap, the
    complete scheduler chunk, and each layer's BF16 boundaries are preserved.
    """

    def __init__(self, residual, pre_mix):
        self.residual, self.pre_mix = residual, pre_mix

    def take(self):
        if self.residual is None:
            raise RuntimeError("Prefill input has already been consumed")
        result = self.residual, self.pre_mix
        self.residual = self.pre_mix = None
        return result


class PreparedStage(nn.Module):
    def __init__(
        self,
        directory,
        pp_rank,
        tp_rank,
        reduce,
        all_gather,
        device,
        max_length=512,
        *,
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        dspark=None,
        prefill_tokens=8192,
    ):
        super().__init__()
        from vllm_gaudi.ops.deepseek_v41_prefill_regions import validate_prefill_region_config

        validate_prefill_region_config()
        self.shard = PreparedV41Shard(directory, pp_rank, tp_rank)
        if (
            getattr(self.shard, "tensor_parallel_size", tensor_parallel_size) != tensor_parallel_size
            or getattr(self.shard, "pipeline_parallel_size", pipeline_parallel_size) != pipeline_parallel_size
        ):
            raise ValueError("Prepared shard topology differs from the runtime topology")
        self.config = json.loads((Path(directory) / "config.json").read_text())
        config = self.config["text_config"]
        self.pp_rank, self.tp_rank, self.length = pp_rank, tp_rank, max_length
        self.pipeline_parallel_size = pipeline_parallel_size
        self.is_last_stage = pp_rank == pipeline_parallel_size - 1
        # Set for each request chunk by the worker. The experimental suffix
        # path must never infer finality from a static graph shape.
        self.prefill_halo_mode = "full"
        self.reduce, self.all_gather = reduce, all_gather
        self.dspark = gaudi_envs.VLLM_HPU_DSV41_DSPARK if dspark is None else bool(dspark)
        if tensor_parallel_size == 4 and self.dspark:
            raise ValueError("V4.1 TP4 does not yet support DSpark")
        self.runtime_indexer = gaudi_envs.VLLM_HPU_DSV41_RUNTIME_INDEXER
        if self.runtime_indexer and (self.dspark or max_length <= 512):
            raise ValueError("Runtime CSA2 indexer requires paged ordinary decode")
        self.bf16_head = gaudi_envs.VLLM_HPU_DSV41_BF16_LM_HEAD
        if self.dspark and (self.bf16_head or gaudi_envs.VLLM_HPU_DSV41_BF16_ROUTER_GATE):
            raise ValueError("BF16 projection candidates require ordinary C1 decode")
        if self.dspark and gaudi_envs.VLLM_HPU_DSV41_SHARED_GATE_UP:
            raise ValueError("Shared gate/up candidate requires ordinary C1 decode")
        if self.dspark and gaudi_envs.VLLM_HPU_DSV41_MHC_GATES_FUSED:
            raise ValueError("Fused mHC gates require ordinary C1 decode")
        self.woa_config = {"version": 1, "layers": []}
        self.dense_config = {"version": 1, "wq_b": [], "wo_b": []}
        self.engram_fp8 = gaudi_envs.VLLM_HPU_DSV41_ENGRAM_FP8 and pp_rank == 0
        if gaudi_envs.VLLM_HPU_DSV41_ENGRAM_FP8 and self.dspark:
            raise ValueError("Engram FP8 requires ordinary execution")
        if gaudi_envs.VLLM_HPU_DSV41_ATTN_DENSE_FP8:
            from vllm_gaudi.ops.deepseek_v41_dense_fp8 import precision_config

            if self.dspark:
                raise ValueError("Attention dense FP8 requires ordinary C1 decode")
            self.dense_config = precision_config(gaudi_envs.VLLM_HPU_DSV41_ATTN_DENSE_FP8_CONFIG)
        if gaudi_envs.VLLM_HPU_DSV41_WO_A_FP8:
            from vllm_gaudi.ops.deepseek_v41_woa_fp8 import layer_selection

            if self.dspark:
                raise ValueError("wo_a FP8 requires ordinary C1 decode")
            self.woa_config = layer_selection(gaudi_envs.VLLM_HPU_DSV41_WO_A_FP8_CONFIG)
        self.woa_output_roundtrip = gaudi_envs.VLLM_HPU_DSV41_WOA_OUTPUT_ROUNDTRIP
        if self.woa_output_roundtrip:
            if self.dspark or not gaudi_envs.VLLM_HPU_DSV41_QUANT_ROUNDTRIP:
                raise ValueError("Fused wo_a output roundtrip requires ordinary quantized execution")
            if set(self.woa_config["layers"]) != set(self.dense_config["wo_b"]):
                raise ValueError("Fused wo_a output roundtrip requires matching wo_a and wo_b FP8 layers")
        self.expert_n256 = gaudi_envs.VLLM_HPU_DSV41_EXPERT_N256 or gaudi_envs.VLLM_HPU_DSV41_EXPERT_N256_FP8
        self.expert_fused_quant = gaudi_envs.VLLM_HPU_DSV41_EXPERT_FUSED_QUANT
        self.expert_fused_reduce = gaudi_envs.VLLM_HPU_DSV41_EXPERT_FUSED_REDUCE
        if self.expert_fused_quant and not gaudi_envs.VLLM_HPU_DSV41_EXPERT_N256_FP8:
            raise ValueError("Fused expert quantization requires N256 FP8 experts")
        if self.expert_fused_reduce and not (gaudi_envs.VLLM_HPU_DSV41_EXPERT_N256_FP8 and self.expert_fused_quant):
            raise ValueError("Fused expert finalize requires N256 FP8 and fused quantization")
        if self.expert_n256 and (
            gaudi_envs.VLLM_HPU_DSV41_FP8_DECODE or gaudi_envs.VLLM_HPU_DSV41_EXPERT_COORD_PIPELINE
        ):
            raise ValueError("N256 experts use their own layout and decoder")
        self.expert_n256_config = {"routed_experts": []}
        if self.expert_n256:
            from vllm_gaudi.ops.deepseek_v41_fp8 import precision_config

            self.expert_n256_config = precision_config(gaudi_envs.VLLM_HPU_DSV41_FP8_CONFIG)
            if not self.expert_n256_config["routed_experts"]:
                self.expert_n256_config["routed_experts"] = list(range(40))
        self.fp8_decode = gaudi_envs.VLLM_HPU_DSV41_FP8_DECODE or gaudi_envs.VLLM_HPU_DSV41_EXPERT_N256_FP8
        if gaudi_envs.VLLM_HPU_DSV41_FP8_DECODE and (self.dspark or not gaudi_envs.VLLM_HPU_DSV41_GRAPH_REPLAY):
            raise ValueError("Legacy V4.1 FP8 decode requires ordinary C1 replay")
        self.weight_specs = {
            name: spec for name, spec in self.shard.specs.items() if self.dspark or not name.startswith("mtp.")
        }
        self.weights = _weight_tree(self.weight_specs)
        self.tensor_parallel_size = tensor_parallel_size
        ranges = self.shard.manifest.get("pp_layer_ranges", [[0, 20], [20, 40]])
        self.start, self.stop = ranges[pp_rank]
        if max_length > 512:
            from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2SharedState

            self.shared = PagedCSA2SharedState(
                config,
                self.start,
                self.stop,
                device,
                max_length,
                prefill_tokens=prefill_tokens,
                tensor_parallel_size=tensor_parallel_size,
            )
        else:
            self.shared = CSA2SharedState(config, self.start, self.stop, device, max_length)
        lookup = mxfp4_bf16_lut(torch.device(device))
        self.layers = nn.ModuleList()
        for layer in range(self.start, self.stop):
            normal = self.shard.manifest["normal_scales"][f"layers.{layer}.ffn.experts"][tp_rank]
            self.layers.append(
                PreparedDecoderLayer(
                    self.weights.layers.get_submodule(str(layer)),
                    config,
                    layer,
                    self.shared,
                    normal,
                    lookup,
                    reduce,
                    all_gather,
                    device,
                    tensor_parallel_size,
                    collect_target_state=self.dspark,
                )
            )
            # Lightweight loader/ownership tests replace the decoder layer
            # with a shell that deliberately has no attention module.
            attention = getattr(self.layers[-1], "attention", None)
            if attention is not None:
                attention.prefill_tp_rank = tp_rank
        self.generation, self.loaded = 0, False
        self.runtime_precision = {
            "experts": (
                "MXFP4/N256 -> FP8 decode MME; grouped BF16 prefill MME"
                if self.expert_n256 and self.fp8_decode
                else "MXFP4 -> BF16 SRAM -> BF16 MME"
            ),
            "dense": "E4M3FN/block32 -> prepared BF16; block32 activation quantization",
            "mla_wo_a": (
                "pretransposed [groups,K,N] BF16"
                if gaudi_envs.VLLM_HPU_DSV41_PRETRANSPOSE_ATTN
                else "checkpoint [groups,N,K] BF16"
            ),
            "mHC_router": (
                "FP32 mHC; BF16 router operands with FP32 logits"
                if gaudi_envs.VLLM_HPU_DSV41_BF16_ROUTER_GATE
                else "FP32"
            ),
            "communication": "BF16",
        }
        self.draft = PreparedDraft(self, lookup, device) if self.dspark and pp_rank == 1 else None

    def load_prepared(self, device):
        if self.loaded:
            self.invalidate()
        else:
            self._invalidate_prefill_regions()
        for layer in self.layers:
            moe = getattr(layer, "moe", None)
            if moe is not None:
                moe.release_shared_gate_up_weight()
            release = getattr(layer, "release_mhc_control_weights", None)
            if release is not None:
                release()
        sidecar = None
        dense_sidecar = None
        engram_sidecar = None
        if gaudi_envs.VLLM_HPU_DSV41_ATTN_DENSE_FP8:
            from vllm_gaudi.ops.deepseek_v41_dense_fp8 import DenseFP8Sidecar

            dense_sidecar = DenseFP8Sidecar(gaudi_envs.VLLM_HPU_DSV41_ATTN_DENSE_FP8_SIDECAR, self.shard)
        if self.engram_fp8:
            from vllm_gaudi.ops.deepseek_v41_engram_fp8 import EngramFP8Sidecar

            engram_sidecar = EngramFP8Sidecar(gaudi_envs.VLLM_HPU_DSV41_ENGRAM_FP8_SIDECAR, self.shard)
        if gaudi_envs.VLLM_HPU_DSV41_WO_A_FP8:
            from vllm_gaudi.ops.deepseek_v41_woa_fp8 import WoaFP8Sidecar

            sidecar = WoaFP8Sidecar(gaudi_envs.VLLM_HPU_DSV41_WO_A_FP8_SIDECAR, self.shard)
        load_weight_tree(
            self.shard,
            self.weights,
            device,
            self.weight_specs,
            woa_sidecar=sidecar,
            woa_layers=self.woa_config["layers"],
            expert_n256_layers=self.expert_n256_config["routed_experts"],
            dense_sidecar=dense_sidecar,
            dense_config=self.dense_config,
            engram_sidecar=engram_sidecar,
        )
        for layer in self.layers:
            prepare = getattr(layer, "prepare_mhc_control_weights", None)
            if prepare is not None:
                prepare()
        if dense_sidecar is not None:
            self.runtime_precision["attention_dense_fp8"] = {
                "config": self.dense_config,
                "weight_fingerprint": dense_sidecar.fingerprint,
            }
        if engram_sidecar is not None:
            self.runtime_precision["engram_fp8"] = {
                "layers": [1, 14],
                "weight_fingerprint": engram_sidecar.fingerprint,
                "max_host_chunk_bytes": engram_sidecar.max_host_chunk_bytes,
            }
        for layer in self.layers:
            moe = getattr(layer, "moe", None)
            if moe is not None:
                moe.prepare_shared_gate_up_weight()
        for layer in self.layers:
            attention = getattr(layer, "attention", None)
            if attention is None:
                continue
            attention.prepare_qkv_input_weight()
            attention.prepare_compressor_input_weight()
            attention.woa_fp8 = layer.layer in self.woa_config["layers"]
            attention.woa_output_roundtrip = self.woa_output_roundtrip and attention.woa_fp8
            if attention.woa_output_roundtrip:
                wo_b = attention.weights.wo_b
                if not (
                    getattr(wo_b, "dense_fp8", False) and hasattr(wo_b, "scale") and hasattr(wo_b, "channel_scale")
                ):
                    raise ValueError("Fused wo_a output roundtrip requires group32 and dense FP8 wo_b")
            if gaudi_envs.VLLM_HPU_DSV41_PREPARED_OUTPUT:
                attention.prepare_output_weight()
        if sidecar is not None:
            self.runtime_precision["wo_a_fp8"] = {
                "config": self.woa_config,
                "weight_fingerprint": sidecar.fingerprint,
            }
        if self.expert_n256:
            from vllm_gaudi.ops.deepseek_v41_expert_n256 import COMPACT_FINGERPRINT, COMPACT_LAYOUT, FINGERPRINT, LAYOUT

            runtime_shard = getattr(self.shard, "_n256_runtime_shard", None)
            compact_expert_scales = (
                runtime_shard.layout == COMPACT_LAYOUT if runtime_shard is not None else self.tensor_parallel_size == 4
            )
            enabled_layers = set(self.expert_n256_config["routed_experts"])
            for layer in self.layers:
                enabled = layer.layer in enabled_layers
                layer.moe.n256 = enabled
                layer.moe.n256_fp8 = enabled and gaudi_envs.VLLM_HPU_DSV41_EXPERT_N256_FP8
                layer.moe.n256_fused = layer.moe.n256_fp8 and self.expert_fused_quant
                layer.moe.n256_fused_reduce = layer.moe.n256_fused and self.expert_fused_reduce
            self.runtime_precision["expert_n256"] = {
                "config": self.expert_n256_config,
                "layout": COMPACT_LAYOUT if compact_expert_scales else LAYOUT,
                "layout_fingerprint": COMPACT_FINGERPRINT if compact_expert_scales else FINGERPRINT,
                "prepared_rank_fingerprint": (
                    self.shard._n256_runtime_shard.fingerprint if hasattr(self.shard, "_n256_runtime_shard") else None
                ),
                "c1_c6": "FP8xFP8" if gaudi_envs.VLLM_HPU_DSV41_EXPERT_N256_FP8 else "BF16xBF16",
                "prefill": (
                    "expert-grouped BF16 BMM over resident N256; ordered weighted SwiGLU"
                    if gaudi_envs.VLLM_HPU_DSV41_PREFILL_GROUPED
                    else "stock MXFP4 bridge"
                    if gaudi_envs.VLLM_HPU_DSV41_PREFILL_MXFP4
                    else "FP8 N256 compound MME in bounded 128-token tiles"
                ),
                "fused_quant": gaudi_envs.VLLM_HPU_DSV41_EXPERT_FUSED_QUANT,
                "fused_reduce": self.expert_fused_reduce,
            }
            self.runtime_precision["experts"] = (
                "N256 MXFP4; C1-C6 "
                + self.runtime_precision["expert_n256"]["c1_c6"]
                + " MME; prefill: "
                + self.runtime_precision["expert_n256"]["prefill"]
            )
        elif gaudi_envs.VLLM_HPU_DSV41_FP8_DECODE:
            from vllm_gaudi.ops.deepseek_v41_fp8 import FP8Sidecar, precision_config

            config = precision_config(gaudi_envs.VLLM_HPU_DSV41_FP8_CONFIG)
            expert_sidecar = FP8Sidecar(gaudi_envs.VLLM_HPU_DSV41_FP8_SIDECAR, self.shard)
            for layer in self.layers:
                enabled = layer.layer in config["routed_experts"]
                layer.moe.fp8 = enabled
                for projection in ("w13", "w2"):
                    value = (
                        expert_sidecar.tensor(
                            f"layers.{layer.layer}.ffn.experts.{projection}_fp8_channel_scale", device
                        )
                        if enabled
                        else None
                    )
                    setattr(layer.moe, f"fp8_{projection}_scale", value)
            self.runtime_precision["fp8"] = {
                "config": config,
                "weight_fingerprint": expert_sidecar.fingerprint,
                "route": "MXFP4 -> E4M3 SRAM -> FP8xFP8 MME -> FP32 scale -> BF16",
            }
        self.runtime_precision["selected_mla_mme"] = gaudi_envs.VLLM_HPU_DSV41_MLA_MME
        self.runtime_precision["router_selection"] = (
            "FP32 scores / native top6 / smallest-ID ties" if gaudi_envs.VLLM_HPU_DSV41_ROUTER_TOP6 else "torch.topk"
        )
        self.runtime_precision["router_gate"] = (
            "BF16xBF16 MME -> FP32 logits" if gaudi_envs.VLLM_HPU_DSV41_BF16_ROUTER_GATE else "FP32 MME"
        )
        self.runtime_precision["head"] = "BF16xBF16 MME -> FP32" if self.bf16_head else "FP32 MME"
        self.runtime_precision["attention_input"] = (
            "fused wq_a+wkv BF16 MME / one activation quantization"
            if gaudi_envs.VLLM_HPU_DSV41_QKV_FUSED_INPUT
            else "separate wq_a/wkv"
        )
        self.runtime_precision["compressor_input"] = (
            "fused ratio-2 FP32 wkv+wgate MME"
            if gaudi_envs.VLLM_HPU_DSV41_COMPRESSOR_FUSED_INPUT
            else "separate ratio-2 wkv/wgate MME"
        )
        self.runtime_precision["mla"] = (
            "shared-KV BF16 QK / FP32 softmax and PV / BF16 output v1"
            if gaudi_envs.VLLM_HPU_DSV41_MLA_MME
            else "TPC online softmax"
        )
        if gaudi_envs.VLLM_HPU_DSV41_MLA_BF16_PV:
            self.runtime_precision["mla"] = "shared BF16 KV / BF16 exp-PV / FP32 denominator and accumulator v1"
        self.runtime_precision["attention_kv_first"] = gaudi_envs.VLLM_HPU_DSV41_ATTN_KV_FIRST
        self.runtime_precision["attention_fused_norm"] = gaudi_envs.VLLM_HPU_DSV41_ATTN_FUSED_NORM
        self.runtime_precision["attention_kv_norm_rope"] = (
            "exact C1 compound TPC; batch-generic split path for B2+"
            if (gaudi_envs.VLLM_HPU_DSV41_ATTN_FUSED_NORM and gaudi_envs.VLLM_HPU_DSV41_NATIVE_ROPE)
            else "split RMSNorm and RoPE"
        )
        self.runtime_precision["q_scale_rope"] = gaudi_envs.VLLM_HPU_DSV41_Q_SCALE_ROPE
        self.runtime_precision["native_rope"] = {
            "enabled": gaudi_envs.VLLM_HPU_DSV41_NATIVE_ROPE,
            "request_batch_rows": 64,
            "other_rows": 6,
            "arithmetic": "fp32-multiply-second-term-fma-bf16",
        }
        self.runtime_precision["local_index_query_layers"] = [
            layer.layer
            for layer in self.layers
            if getattr(getattr(layer, "attention", None), "tp4_local_index_queries", False)
        ]
        self.runtime_precision["woa_output_roundtrip"] = self.woa_output_roundtrip
        self.runtime_precision["batch_full_index_mme"] = gaudi_envs.VLLM_HPU_DSV41_BATCH_FULL_INDEX_MME
        self.runtime_precision["batch_index_tiled_keys"] = gaudi_envs.VLLM_HPU_DSV41_BATCH_INDEX_TILED_KEYS
        self.runtime_precision["batch_packed_mla"] = gaudi_envs.VLLM_HPU_DSV41_BATCH_PACKED_MLA
        self.runtime_precision["batch_packed_mla_sram"] = gaudi_envs.VLLM_HPU_DSV41_BATCH_PACKED_MLA_SRAM
        self.runtime_precision["batch_packed_mla_vector"] = gaudi_envs.VLLM_HPU_DSV41_BATCH_PACKED_MLA_VECTOR
        self.runtime_precision["batch_kernel_policy_version"] = 1
        self.runtime_precision["batch_main_fusions"] = gaudi_envs.VLLM_HPU_DSV41_BATCH_MAIN_FUSIONS
        self.runtime_precision["batch_c1_numerics"] = gaudi_envs.VLLM_HPU_DSV41_BATCH_C1_NUMERICS
        self.runtime_precision["mhc_batch_reuse"] = gaudi_envs.VLLM_HPU_DSV41_MHC_BATCH_REUSE
        self.runtime_precision["mhc_control_prefetch"] = gaudi_envs.VLLM_HPU_DSV41_MHC_CONTROL_PREFETCH
        self.runtime_precision["batch_compressor_pair"] = gaudi_envs.VLLM_HPU_DSV41_BATCH_COMPRESSOR_PAIR
        self.runtime_precision["batch_compressor_gather"] = gaudi_envs.VLLM_HPU_DSV41_BATCH_COMPRESSOR_GATHER
        self.runtime_precision["batch_expert_prefetch_w2"] = gaudi_envs.VLLM_HPU_DSV41_BATCH_EXPERT_PREFETCH_W2
        self.runtime_precision["batch_route_pack"] = gaudi_envs.VLLM_HPU_DSV41_BATCH_ROUTE_PACK
        self.runtime_precision["batch_expert_reuse"] = gaudi_envs.VLLM_HPU_DSV41_BATCH_EXPERT_REUSE
        self.runtime_precision["batch_w13_horizontal"] = gaudi_envs.VLLM_HPU_DSV41_BATCH_W13_HORIZONTAL
        self.runtime_precision["batch_expert_direct_finalize"] = gaudi_envs.VLLM_HPU_DSV41_BATCH_EXPERT_DIRECT_FINALIZE
        self.runtime_precision["batch_expert_transpose_mme"] = gaudi_envs.VLLM_HPU_DSV41_BATCH_EXPERT_TRANSPOSE_MME
        self.runtime_precision["shared_gate_up"] = gaudi_envs.VLLM_HPU_DSV41_SHARED_GATE_UP
        self.runtime_precision["concurrent_moe"] = {
            "rows": gaudi_envs.VLLM_HPU_DSV41_CONCURRENT_MOE_ROWS,
            "version": 1,
            "groups_per_island": 16,
            "batches": [64],
        }
        self.runtime_precision["mhc_gates_fused"] = gaudi_envs.VLLM_HPU_DSV41_MHC_GATES_FUSED
        self.runtime_precision["reindex_bounded_plan"] = {
            "enabled": gaudi_envs.VLLM_HPU_DSV41_REINDEX_BOUNDED_PLAN,
            "version": 2,
            "tile_rows": 2048,
            "maximum_tiles": 8,
            "whole_pool_after_tiles": 4,
            "whole_pool_tag": 9,
        }
        self.precision_fingerprint = canonical_hash(self.runtime_precision)
        self.loaded = True
        self.generation += 1

    def _invalidate_prefill_regions(self):
        from vllm_gaudi.ops.deepseek_v41_prefill_plan import invalidate_prefill_plans

        invalidate_prefill_plans()
        clear_prefill_function_regions()
        for layer in self.layers:
            clear_prefill_regions(layer)
            attention = getattr(layer, "attention", None)
            if attention is not None:
                clear_prefill_regions(attention)

    def invalidate(self):
        # The execution owner must close its recipes before reloading or
        # migrating buffers. A live plan cannot retain addresses from this tree.
        if getattr(self, "replay_owner", None) is not None:
            self.replay_owner.close()
        invalidate_index_mirror = getattr(getattr(self, "shared", None), "invalidate_index_mirror", None)
        if invalidate_index_mirror is not None:
            invalidate_index_mirror()
        self._invalidate_prefill_regions()
        for layer in self.layers:
            attention = getattr(layer, "attention", None)
            if attention is not None:
                attention.invalidate_qkv_input_weight()
                attention.invalidate_compressor_input_weight()
                invalidate_queries = getattr(attention, "invalidate_tp4_index_query_weights", None)
                if invalidate_queries is not None:
                    invalidate_queries()
            moe = getattr(layer, "moe", None)
            if moe is not None:
                moe.release_shared_gate_up_weight()
            layer.release_mhc_control_weights()
        self.loaded = False
        self.generation += 1

    def embed(self, input_ids, *, draft=False):
        embedding = self.weights.mtp.embed if draft else self.weights.embed
        per_rank = embedding.weight.shape[0]
        local = input_ids.long() - self.tp_rank * per_rank
        valid = (local >= 0) & (local < per_rank)
        values = F.embedding(local.masked_fill(~valid, 0), embedding.weight)
        return self.reduce(values.masked_fill(~valid.unsqueeze(-1), 0))

    def _forward_impl(self, residual, pre_mix, positions, input_ids, engram_rows):
        if isinstance(residual, PrefillInput):
            residual, pre_mix = residual.take()
        retire = (
            torch.hpu.Event()
            if getattr(self, "tensor_parallel_size", 2) == 4
            and residual.device.type == "hpu"
            and residual.shape[0] > 8192
            else None
        )
        from vllm_gaudi.ops.deepseek_v41_prefill_sequence_state import (
            can_sequence_prefill_state,
            gather_tokens,
            retire_prefill_layer,
            token_owner,
        )

        sequence_state = can_sequence_prefill_state(self, positions.numel())
        layer_options = {"prefill_sequence": True} if sequence_state else {}
        if sequence_state:
            from vllm.distributed import get_tp_group

            group = get_tp_group()
            owned = token_owner(positions.numel(), group.rank_in_group)
            residual, pre_mix = residual[owned].clone(), pre_mix[owned].clone()
        workspace = getattr(self.shared, "prefill_main_workspace", None)
        if workspace is not None and positions.numel() > 6:
            self.shared.prefill_kv_generation += 1
            workspace.begin(self.shared.prefill_kv_generation)
        image_mask = (input_ids == 129264) | (input_ids == 129265)
        target_states = []
        for layer in self.layers:
            rows = engram_rows[0 if layer.layer == 1 else 1] if layer.layer in (1, 14) else None
            residual, pre_mix, target = layer(residual, pre_mix, positions, image_mask, rows, **layer_options)
            if target is not None:
                target_states.append(target)
            retire_prefill_layer(retire)
        if sequence_state:
            residual = gather_tokens(residual, group=group.device_group)
            pre_mix = gather_tokens(pre_mix.contiguous(), group=group.device_group)
        if not self.is_last_stage:
            return residual, pre_mix, None
        value = final_collapse_rms_norm(
            residual, pre_mix, self.weights.norm.weight, self.config["text_config"]["rms_norm_eps"]
        )
        return value, pre_mix, torch.cat(target_states, -1) if target_states else None

    def _forward_prefill_halo(self, residual, pre_mix, positions, input_ids, mode, engram_rows=()):
        """Keep layers through the global source complete and bound decoder rows.

        Prefix-only chunks publish layer-20 global KV. A full chunk before a
        short final block rebuilds the local caches. The final chunk retains
        the full dependency halo for the last output and for
        every decoder layer's trailing SWA cache. The source's candidate rows
        are compacted after its last consumer, before the suffix Reindex layers.
        The scheduler selects this only for the qualified text-prompt geometry.
        """
        tp4 = getattr(self, "tensor_parallel_size", 2) == 4
        supported = (tp4 and self.pp_rank == 0 and self.start in (0, 20)) or (
            not tp4 and self.pp_rank == 1 and self.start == 20
        )
        if not supported or self.dspark or self.stop != 40 or len(self.layers) != 40 - self.start:
            raise RuntimeError("Decoder prefill halo requires complete layers through source20 and decoder39")
        if mode not in ("prefix_only", "final"):
            raise RuntimeError("Decoder prefill halo requires an explicit request phase")
        if isinstance(residual, PrefillInput):
            residual, pre_mix = residual.take()
        retire = torch.hpu.Event() if tp4 and residual.device.type == "hpu" and positions.numel() > 8192 else None
        from vllm_gaudi.ops.deepseek_v41_prefill_sequence_state import (
            can_sequence_prefill_state,
            gather_tokens,
            replicate_owned_tail,
            retire_prefill_layer,
            token_owner,
        )

        sequence_state = can_sequence_prefill_state(self, positions.numel())
        layer_options = {"prefill_sequence": True} if sequence_state else {}
        if sequence_state:
            from vllm.distributed import get_tp_group

            group = get_tp_group()
            owned = token_owner(positions.numel(), group.rank_in_group)
            residual, pre_mix = residual[owned].clone(), pre_mix[owned].clone()
        workspace = getattr(self.shared, "prefill_main_workspace", None)
        if workspace is not None:
            self.shared.prefill_kv_generation += 1
            workspace.begin(self.shared.prefill_kv_generation)
        image_mask = (input_ids == 129264) | (input_ids == 129265)
        source_index = 20 - self.start
        for layer in self.layers[: source_index + 1]:
            layer_id = getattr(layer, "layer", -1)
            if layer_id in (1, 14):
                rows = engram_rows[0 if layer_id == 1 else 1] if engram_rows else None
                residual, pre_mix, target = layer(residual, pre_mix, positions, image_mask, rows, **layer_options)
            else:
                residual, pre_mix, target = layer(residual, pre_mix, positions, image_mask, **layer_options)
            if target is not None:
                raise RuntimeError("Decoder halo cannot suppress DSpark target-state rows")
            retire_prefill_layer(retire)
        if mode == "prefix_only":
            # The runner neither samples nor requests logits for this prompt
            # chunk. Return a defined tensor for its existing stage contract.
            if sequence_state:
                return (
                    gather_tokens(residual[:, 0, :].contiguous(), group=group.device_group),
                    gather_tokens(pre_mix.contiguous(), group=group.device_group),
                    None,
                )
            return residual[:, 0, :].contiguous(), pre_mix, None

        from vllm_gaudi.ops.deepseek_v41_decoder_halo import decoder_halo_rows

        retained = decoder_halo_rows(len(self.layers) - source_index - 1)
        cut = max(0, positions.numel() - retained)
        if cut:
            pool = self.shared.candidate_pool
            if pool is None or pool.shape[0] < positions.numel():
                raise RuntimeError("Decoder halo has no complete layer-20 candidate rows")
            # Layer 20 published [0,T) in transaction-local row order. The
            # Reindex consumer takes its own [0,retained) row interval.
            pool[:retained].copy_(pool[cut : cut + retained].clone())
            selection = self.shared.topk["20"].indices
            if selection.shape[0] < positions.numel():
                raise RuntimeError("Decoder halo has no complete layer-20 selection rows")
            selection[:retained].copy_(selection[cut : cut + retained].clone())
            if sequence_state:
                residual = replicate_owned_tail(residual, retained, group=group.device_group)
                pre_mix = replicate_owned_tail(pre_mix, retained, group=group.device_group)
                positions, image_mask = positions[cut:], image_mask[cut:]
            else:
                residual, pre_mix, positions, image_mask = (v[cut:] for v in (residual, pre_mix, positions, image_mask))
        elif sequence_state:
            residual = gather_tokens(residual, group=group.device_group)
            pre_mix = gather_tokens(pre_mix.contiguous(), group=group.device_group)
        for layer in self.layers[source_index + 1 :]:
            if tp4 and cut:
                residual, pre_mix, target = layer(
                    residual, pre_mix, positions, image_mask, prefill_router_tokens=cut + retained
                )
            else:
                residual, pre_mix, target = layer(residual, pre_mix, positions, image_mask)
            if target is not None:
                raise RuntimeError("Decoder halo cannot suppress DSpark target-state rows")
        value = final_collapse_rms_norm(
            residual, pre_mix, self.weights.norm.weight, self.config["text_config"]["rms_norm_eps"]
        )
        if cut:
            # The vLLM sampler still indexes the last row of the original
            # scheduler transaction. Prefix rows are deliberately invalid.
            full = value.new_zeros((cut + retained, value.shape[-1]))
            full[cut:].copy_(value)
            full_pre = pre_mix.new_zeros((cut + retained, *pre_mix.shape[1:]))
            full_pre[cut:].copy_(pre_mix)
            return full, full_pre, None
        return value, pre_mix, None

    def forward(self, residual, pre_mix, positions, input_ids, engram_rows):
        """Run normal vLLM prefill with bounded internal token tiles.

        The scheduler still submits one ordinary prompt transaction (up to
        ``max_num_batched_tokens``).  A large transaction is split only at the
        model execution boundary so the HPU does not retain every layer's
        temporary attention/MoE workspace for the whole prompt.  Each tile
        passes through all layers before the next tile; paged CSA2 state makes
        the causal prefix visible to the following tile, so this preserves the
        normal prefill ordering and does not turn the request into C1/C6.
        """
        if isinstance(residual, PrefillInput):
            if self.tensor_parallel_size != 4 or not (
                gaudi_envs.VLLM_HPU_DSV41_PREFILL_GROUPED or gaudi_envs.VLLM_HPU_DSV41_PREFILL_MXFP4
            ):
                raise RuntimeError("Owned prefill requires the full-chunk TP4 expert path")
            if (
                gaudi_envs.VLLM_HPU_DSV41_PREFILL_DECODER_HALO
                and positions.numel() > 6
                and self.prefill_halo_mode != "full"
            ):
                return self._forward_prefill_halo(
                    residual, None, positions, input_ids, self.prefill_halo_mode, engram_rows
                )
            return self._forward_impl(residual, None, positions, input_ids, engram_rows)
        tile = PreparedMoE.N256_PREFILL_TILE
        tokens = residual.shape[0]
        if (
            gaudi_envs.VLLM_HPU_DSV41_PREFILL_DECODER_HALO
            and (self.pp_rank == 1 or getattr(self, "tensor_parallel_size", 2) == 4)
            and tokens > 6
            and self.prefill_halo_mode != "full"
        ):
            return self._forward_prefill_halo(
                residual, pre_mix, positions, input_ids, self.prefill_halo_mode, engram_rows
            )
        if tokens > 6 and (gaudi_envs.VLLM_HPU_DSV41_PREFILL_GROUPED or gaudi_envs.VLLM_HPU_DSV41_PREFILL_MXFP4):
            # Both implementations bound expert workspace internally and
            # reuse each weight over this complete scheduler transaction.
            # Splitting the stage into C128 first defeats that reuse. Paged
            # attention owns causal masks and its separate workspace tiles.
            return self._forward_impl(residual, pre_mix, positions, input_ids, engram_rows)
        if tokens <= tile:
            return self._forward_impl(residual, pre_mix, positions, input_ids, engram_rows)

        # Keep only one tile's graph values alive.  Holding every tile in a
        # Python list defeats the bounded-workspace contract: HPU tensors keep
        # their producer allocations live until the final ``cat`` and the
        # allocator then tries (and cannot) defragment the 1M-context pool.
        full_value = full_pre = full_target = None
        for start in range(0, tokens, tile):
            stop = min(start + tile, tokens)
            rows = tuple(row[start:stop] for row in engram_rows) if engram_rows else engram_rows
            value, local_pre, target = self._forward_impl(
                residual[start:stop],
                pre_mix[start:stop],
                positions[start:stop],
                input_ids[start:stop],
                rows,
            )
            if full_value is None:
                full_value = torch.empty((tokens, *value.shape[1:]), dtype=value.dtype, device=value.device)
                full_pre = torch.empty((tokens, *local_pre.shape[1:]), dtype=local_pre.dtype, device=local_pre.device)
                if target is not None:
                    full_target = torch.empty((tokens, *target.shape[1:]), dtype=target.dtype, device=target.device)
            full_value[start:stop].copy_(value)
            full_pre[start:stop].copy_(local_pre)
            if target is not None:
                full_target[start:stop].copy_(target)
            # The copies are the only values that must survive the tile.  Drop
            # the producer outputs before synchronizing so Synapse can reclaim
            # its large MME/TPC workspaces instead of attempting defragmentation
            # while those allocations are still referenced.
            # Same-stream ordering makes the next tile observe the copy.  A
            # device-wide synchronize here asks PT_DEVMEM to defragment while
            # the just-submitted producer graph is still in flight; defer the
            # only required synchronization to the PP/response boundary.
            del value, local_pre, target, rows
        return full_value, full_pre, full_target

    def _head_projection(self, hidden):
        if self.bf16_head:
            return torch.ops.custom_op.custom_deepseek_v41_bf16_linear_f32_gaudi2(
                hidden.contiguous(), self.weights.head.weight
            )
        return F.linear(hidden.float(), self.weights.head.weight)

    def logits(self, hidden):
        if not self.is_last_stage:
            raise RuntimeError("Only the final PP stage owns the output head")
        local = self._head_projection(hidden)
        return self.all_gather(local, dim=-1)

    def sample_greedy(self, hidden):
        from vllm_gaudi.ops.deepseek_v41_sampling import (
            local_greedy_candidate,
            select_greedy_candidate,
        )

        maximum = 64 if gaudi_envs.VLLM_HPU_DSV41_BATCH_DECODE else 1
        if not self.is_last_stage or not 1 <= hidden.shape[0] <= maximum:
            raise ValueError("Ordinary greedy head exceeds the configured final-stage request batch")
        local = self._head_projection(hidden)
        candidates = self.all_gather(local_greedy_candidate(local, self.tp_rank), dim=-1)
        return select_greedy_candidate(candidates)

    def sample_greedy_token(self, hidden):
        return self.sample_greedy(hidden).to(torch.int32)

    def sample_greedy_commit(self, hidden, record):
        selected = self.sample_greedy(hidden).reshape(1).to(torch.int32)
        updated = torch.cat((record[:1] + 1, torch.ones_like(record[1:3]), selected))
        record.copy_(updated)
        return record


class PreparedGreedyTail(nn.Module):
    """Reuse the ordinary greedy head inside the decoder replay."""

    _head_projection = PreparedStage._head_projection
    sample_greedy = PreparedStage.sample_greedy
    sample_greedy_token = PreparedStage.sample_greedy_token

    def __init__(self, stage):
        super().__init__()
        self.weights = nn.Module()
        self.weights.head = stage.weights.head
        self.bf16_head = stage.bf16_head
        from vllm_gaudi.ops.deepseek_v41_replay import stage_collectives

        self.tp_rank = stage.tp_rank
        _, self.all_gather = stage_collectives(stage.tp_rank, True,
                                              getattr(stage, "tensor_parallel_size", 2),
                                              native_fp32_gather=True)
        self.is_last_stage = True

    def forward(self, hidden):
        return self.sample_greedy_token(hidden)


class PreparedLayerGroup(nn.Module):
    """Bound FX dependency closure without changing the stage tensor program."""

    def __init__(self, stage, start, stop, *, pp_wire_input=False, fused_text_io=False,
                 fp8_decode=False, decode=False, replay_tail=False):
        super().__init__()
        self.fp8_decode = fp8_decode
        self.decode = decode
        self.preserve_layer_rounding = getattr(stage, "tensor_parallel_size", 2) == 4
        self.pp_wire_input = pp_wire_input and start == 0
        self.text_input = fused_text_io and stage.pp_rank == 0 and start == 0
        is_last_stage = getattr(stage, "is_last_stage", stage.pp_rank == 1)
        self.wire_output = fused_text_io and stage.pp_rank == 0 and not is_last_stage and stop == len(stage.layers)
        self.embedding = stage.weights.embed if self.text_input else None
        self.tp_rank = getattr(stage, "tp_rank", 0)
        self.reduce = getattr(stage, "reduce", None)
        self.layers = nn.ModuleList(list(stage.layers[start:stop]))
        self.final = is_last_stage and stop == len(stage.layers)
        self.norm = stage.weights.norm if self.final else None
        self.greedy_tail = PreparedGreedyTail(stage) if self.final and replay_tail else None
        self.eps = stage.config["text_config"]["rms_norm_eps"]
        self.native_input = None

    def _forward(self, residual, pre_mix, positions, input_ids, engram_rows, *, fp8_decode=False, decode=False):
        if self.text_input:
            mapped = input_ids.masked_fill(input_ids == 129265, 129264)
            per_rank = self.embedding.weight.shape[0]
            local = mapped.long() - self.tp_rank * per_rank
            valid = (local >= 0) & (local < per_rank)
            values = F.embedding(local.masked_fill(~valid, 0), self.embedding.weight)
            values = self.reduce(values.masked_fill(~valid.unsqueeze(-1), 0))
            residual = values.unsqueeze(1).expand(-1, 4, -1).contiguous()
            pre_mix = torch.tensor([1.0, 0.0, 0.0, 0.0], device=values.device).expand(values.shape[0], 4)
        if self.pp_wire_input:
            from vllm_gaudi.ops.deepseek_v41_pp_wire import decode_pp_wire

            residual, pre_mix = decode_pp_wire(residual)
        image_mask = (input_ids == 129264) | (input_ids == 129265)
        target_states = []
        # Purely local graph values: no cross-token cache or request state.
        selected_main = {} if self.preserve_layer_rounding and decode and residual.shape[0] == 1 else None
        decode_metadata = None
        if (
            self.preserve_layer_rounding
            and decode
            and any(
                getattr(getattr(layer, "attention", None), "shared_decode_metadata", False) for layer in self.layers
            )
        ):
            from vllm_gaudi.ops.deepseek_v41_decode_metadata import prepare_decode_metadata

            decode_metadata = prepare_decode_metadata(positions)
        collapse_handoff = {} if self.preserve_layer_rounding and decode and residual.shape[0] <= 2 else None
        for index, layer in enumerate(self.layers):
            rows = engram_rows[0 if layer.layer == 1 else 1] if layer.layer in (1, 14) else None
            collapse_kwargs = {}
            if collapse_handoff is not None and getattr(layer, "mhc_interlayer_collapse", False):
                following = self.layers[index + 1] if index + 1 < len(self.layers) else None
                publish = (
                    following is not None
                    and following.layer == layer.layer + 1
                    and getattr(following, "mhc_interlayer_collapse", False)
                    and not getattr(layer, "draft", False)
                    and not getattr(following, "draft", False)
                    and not hasattr(following.weights, "engram")
                )
                collapse_kwargs = {"collapse_handoff": collapse_handoff, "publish_collapse": publish}
            residual, pre_mix, target = layer(
                residual,
                pre_mix,
                positions,
                image_mask,
                rows,
                fp8_decode=fp8_decode,
                decode=decode,
                selected_main=selected_main,
                decode_metadata=decode_metadata,
                **collapse_kwargs,
            )
            if target is not None:
                target_states.append(target)
            if self.preserve_layer_rounding and decode and layer is not self.layers[-1]:
                if collapse_handoff is not None and layer.layer + 1 in collapse_handoff:
                    # The opaque native producer has already materialized
                    # the exact BF16 boundary, including its collapse.
                    continue
                # Preserve the same BF16 residual boundary as separate layer
                # recipes; cross-layer fusion otherwise changes cached KV.
                shape = residual.shape
                residual = torch.ops.custom_op.custom_deepseek_v41_bf16_identity_gaudi2(
                    residual.reshape(1, -1).contiguous()
                ).reshape(shape)
        if gaudi_envs.VLLM_HPU_DSV41_MHC_GATES_FUSED:
            # The last layer's pre gate is a narrow view into one packed
            # kernel output. Materialize it at the recipe boundary because
            # prepared replay cannot bind a partial-storage alias.
            pre_mix = pre_mix.clone()
        if self.wire_output:
            from vllm_gaudi.ops.deepseek_v41_pp_wire import encode_pp_wire

            return encode_pp_wire(residual, pre_mix), None, None
        if not self.final:
            return residual, pre_mix, None
        value = final_collapse_rms_norm(residual, pre_mix, self.norm.weight, self.eps)
        aux = torch.cat(target_states, -1) if target_states else None
        if self.greedy_tail is not None and decode and value.shape[0] == 1:
            return value, pre_mix, aux, self.greedy_tail(value)
        return value, pre_mix, aux

    def forward(self, residual, pre_mix, positions, input_ids, engram_rows):
        # ``native`` describes the compiled/replay-capable stage, not the
        # request shape.  Treat only C1-C6 as decode work; larger scheduler
        # transactions must use the normal paged prefill attention path even
        # when they enter through the native stage wrapper.
        decode = self.decode and residual.shape[0] <= 6
        return self._forward(
            residual, pre_mix, positions, input_ids, engram_rows, fp8_decode=self.fp8_decode, decode=decode
        )

    def native_forward(self, residual, pre_mix, positions, input_ids, engram_rows):
        if self.native_input is not None:
            residual, pre_mix = self.native_input(input_ids)
        return self(residual, pre_mix, positions, input_ids, engram_rows)


_compile_entry_ids = count()


def _compile_group(group, *, native, backend="hpu_backend", tp4_owner=None):
    if tp4_owner is not None:
        if native:
            raise ValueError("Prepared TP4 export does not use the TP2 native peer path")
        from vllm_gaudi.compilation.deepseek_v41_prepared import PreparedTP4Group

        return PreparedTP4Group(group, tp4_owner, backend)
    # Dynamo caches variants by code object. Each static layer group/bucket
    # owns its entry so legitimate preparations cannot exhaust another group.
    method = group.native_forward if native else group.forward
    function = method.__func__
    name = f"{function.__name__}_v41_{next(_compile_entry_ids)}"
    entry = FunctionType(
        function.__code__.replace(co_name=name), function.__globals__, name, function.__defaults__, function.__closure__
    )
    entry.__kwdefaults__ = function.__kwdefaults__
    entry.__module__ = function.__module__
    return torch.compile(MethodType(entry, group), backend=backend, fullgraph=True, dynamic=False)


class CompiledStage:
    def __init__(
        self,
        stage,
        *,
        native=False,
        pp_wire_input=False,
        fused_text_io=False,
        native_input=False,
        group_size=4,
        prepared_tp4=False,
        native_tp4=False,
        replay_tail=False,
    ):
        legacy_fp8 = getattr(stage, "fp8_decode", False) and not getattr(stage, "expert_n256", False)
        if native_input and (not native or stage.pp_rank != 0 or stage.dspark or legacy_fp8):
            raise ValueError("Native input capture requires ordinary BF16 PP0 decode")
        if native_input and (pp_wire_input or fused_text_io):
            raise ValueError("Native input capture has a single PP0 input owner")
        backend = "hpu_backend"
        if not native and getattr(stage, "tensor_parallel_size", 2) == 4:
            from vllm_gaudi.compilation.deepseek_v41_tp4 import make_backend

            backend = make_backend(native_group_owner=stage if prepared_tp4 and native_tp4 else None)
        # Keep the scheduler transaction large (normal vLLM chunked prefill),
        # but remember whether this stage owns compiled entries.  Very large
        # M inputs are dispatched through the eager group method below so
        # each bounded N256 tile can release its workspace before the next
        # tile.  C1/decode and <=512-token prefill retain the compiled graph.
        self.native = bool(native)
        self.audit = {"calls": 0, "tokens": 0, "groups": 0}
        if native and gaudi_envs.VLLM_HPU_DSV41_TP_MHC_OVERLAP:
            if stage.dspark or (stage.fp8_decode and not stage.expert_n256):
                raise ValueError("TP/mHC overlap requires BF16 boundaries and a qualified C1 expert layout")
            from vllm_gaudi.compilation.deepseek_v41_overlap import make_backend

            backend = (make_backend(static_int32=True)
                       if getattr(stage, 'decode_static_int32', False) else make_backend())
        if group_size < 1 or len(stage.layers) % group_size:
            raise ValueError(f"Invalid V4.1 compiled layer group size {group_size} for {len(stage.layers)} layers")
        self.groups = tuple(
            PreparedLayerGroup(
                stage,
                start,
                start + group_size,
                pp_wire_input=pp_wire_input,
                fused_text_io=fused_text_io,
                fp8_decode=getattr(stage, "fp8_decode", False),
                decode=native or getattr(stage, "tensor_parallel_size", 2) == 4,
                replay_tail=replay_tail,
            )
            for start in range(0, len(stage.layers), group_size)
        )
        if native_input:
            self.groups[0].native_input = PreparedInput(stage.weights.embed, stage.tp_rank, stage.reduce)
        tp4_owner = stage if not native and prepared_tp4 and getattr(stage, "tensor_parallel_size", 2) == 4 else None
        compile_options = {"native": native, "backend": backend}
        if tp4_owner is not None:
            compile_options["tp4_owner"] = tp4_owner
        self.chunks = tuple(_compile_group(group, **compile_options) for group in self.groups)
        self.owner = stage
        self.prefix_groups = (
            next(
                (index for index, group in enumerate(self.groups) if any(layer.layer == 14 for layer in group.layers)),
                0,
            )
            if getattr(stage, "tensor_parallel_size", 2) == 4
            else 0
        )
        self.prefix_generation = None

    def prefix_ready(self, search_length):
        return self.prefix_groups > 0 and self.prefix_generation == self._prefix_key(search_length)

    def _prefix_key(self, search_length):
        from vllm_gaudi.ops.deepseek_v41_index_mirror import index_mirror_execution_mode

        return (
            self.owner.generation,
            getattr(self.owner, "precision_fingerprint", None),
            search_length,
            getattr(self.owner, "decode_token_bound", None),
            index_mirror_execution_mode(self.owner, search_length),
        )

    def prefix(self, hidden, pre_mix, positions, input_ids, engram):
        if not self.prefix_ready(self.owner.search_length) or hidden.shape[0] != 1:
            raise RuntimeError("TP4 prefix requires a prepared C1 generation and search geometry")
        hidden, pre_mix, _ = self._run_group_range(0, self.prefix_groups, hidden, pre_mix, positions, input_ids, engram)
        return hidden, pre_mix

    def suffix(self, hidden, pre_mix, positions, input_ids, engram):
        if not self.prefix_ready(self.owner.search_length):
            raise RuntimeError("TP4 prefix generation changed before its consumer")
        hidden, pre_mix, aux = self._run_group_range(
            self.prefix_groups, len(self.chunks), hidden, pre_mix, positions, input_ids, engram
        )
        self.audit["calls"] += 1
        self.audit["tokens"] += 1
        self.audit["groups"] += len(self.chunks)
        return hidden, pre_mix, aux

    def _run_group_range(self, start, stop, hidden, pre_mix, positions, input_ids, engram):
        from vllm_gaudi.ops.deepseek_v41_native_trace import annotations_enabled, scope

        for index in range(start, stop):
            group, chunk = self.groups[index], self.chunks[index]
            if annotations_enabled():
                first, last = group.layers[0].layer, group.layers[-1].layer
                with scope(f"v41::compiled::layers{first}-{last}::C{hidden.shape[0]}"):
                    hidden, pre_mix, aux = chunk(hidden, pre_mix, positions, input_ids, engram)
            else:
                hidden, pre_mix, aux = chunk(hidden, pre_mix, positions, input_ids, engram)
        return hidden, pre_mix, aux

    def __call__(self, hidden, pre_mix, positions, input_ids, engram):
        stock_prefill = hidden.shape[0] > 6 and (
            gaudi_envs.VLLM_HPU_DSV41_PREFILL_MXFP4 or gaudi_envs.VLLM_HPU_DSV41_PREFILL_GROUPED
        )
        large_prefill = hidden.shape[0] > PreparedMoE.N256_PREFILL_TILE
        if stock_prefill:
            # Preserve the complete scheduler chunk through the stage. TP4
            # dispatches bounded prepared N256 plans inside each MoE layer;
            # their temporary expert buffers retire before the next layer.
            # Keep prefill outside the compiled decode layer groups.
            aux = None
            for group in self.groups:
                method = group.native_forward if self.native else group.forward
                hidden, pre_mix, aux = method(hidden, pre_mix, positions, input_ids, engram)
            return hidden, pre_mix, aux
        if large_prefill:
            # A single compiled four-layer graph would retain every N256
            # tile's temporary MME/TPC buffers until the graph completed.
            # Run the *whole stage* for one bounded tile before advancing to
            # the next tile.  Advancing all groups for a tile is required for
            # causal CSA2/Engram state: running group-by-group over the full
            # prompt would make layer 1 observe all future tokens and also
            # retain the large attention workspace.  The scheduler still owns
            # one normal C8192 transaction; this is only an execution-memory
            # tile at the model boundary.
            tile = PreparedMoE.N256_PREFILL_TILE
            full_hidden = full_pre = full_aux = None
            token_count = hidden.shape[0]
            for start in range(0, token_count, tile):
                stop = min(start + tile, token_count)
                tile_hidden = hidden[start:stop]
                tile_pre = pre_mix[start:stop]
                tile_positions = positions[start:stop]
                tile_ids = input_ids[start:stop]
                tile_engram = tuple(row[start:stop] for row in engram) if engram else engram
                aux = None
                for group in self.groups:
                    method = group.native_forward if self.native else group.forward
                    tile_hidden, tile_pre, aux = method(tile_hidden, tile_pre, tile_positions, tile_ids, tile_engram)
                if full_hidden is None:
                    full_hidden = torch.empty(
                        (token_count, *tile_hidden.shape[1:]), dtype=tile_hidden.dtype, device=tile_hidden.device
                    )
                    full_pre = torch.empty(
                        (token_count, *tile_pre.shape[1:]), dtype=tile_pre.dtype, device=tile_pre.device
                    )
                    if aux is not None:
                        full_aux = torch.empty((token_count, *aux.shape[1:]), dtype=aux.dtype, device=aux.device)
                full_hidden[start:stop].copy_(tile_hidden)
                full_pre[start:stop].copy_(tile_pre)
                if aux is not None:
                    full_aux[start:stop].copy_(aux)
                # Release each tile's producer graph before the synchronization
                # boundary.  Otherwise the old list-based implementation kept
                # all prompt tiles live and exhausted the contiguous HBM left
                # by a 1M-token KV pool.
                del tile_hidden, tile_pre, aux, tile_engram, tile_positions, tile_ids
                # Keep tile execution ordered on the same HPU stream.  The
                # runner synchronizes at the actual PP/response boundary;
                # synchronizing here can trigger defragmentation while this
                # tile's producer graph is still referenced by the stream.
            return full_hidden, full_pre, full_aux
        self.audit["calls"] += 1
        self.audit["tokens"] += hidden.shape[0]
        self.audit["groups"] += len(self.chunks)
        from vllm_gaudi.ops.deepseek_v41_native_trace import annotations_enabled, scope

        if annotations_enabled():
            for group, chunk in zip(self.groups, self.chunks, strict=True):
                first, last = group.layers[0].layer, group.layers[-1].layer
                with scope(f"v41::compiled::layers{first}-{last}::C{hidden.shape[0]}"):
                    hidden, pre_mix, aux = chunk(hidden, pre_mix, positions, input_ids, engram)
        else:
            for chunk in self.chunks:
                hidden, pre_mix, aux = chunk(hidden, pre_mix, positions, input_ids, engram)
        if (
            input_ids.numel() == 1
            and input_ids.dtype == torch.int32
            and len(engram) == 2
            and engram[0].dtype == torch.bfloat16
            and getattr(self.owner, "tensor_parallel_size", 2) == 4
        ):
            self.prefix_generation = self._prefix_key(self.owner.search_length)
        return hidden, pre_mix, aux


class PreparedDraft(nn.Module):
    """Three DSpark layers and checkpoint heads on the final PP stage.

    Model math follows vLLM #56214 e47aa780 and the checkpoint reference.
    The draft owns an embedding shard under PP, following #53577 d2b1b735.
    Accepted-prefix ownership stays with the caller's verify transaction.
    """

    def __init__(self, stage, lookup, device):
        super().__init__()
        self.weights = stage.weights.mtp
        self.output_head = stage.weights.head
        self.tp_rank, self.reduce, self.all_gather = stage.tp_rank, stage.reduce, stage.all_gather
        config = dict(stage.config["text_config"])
        config["num_experts_per_tok"] = config["dspark_num_experts_per_tok"]
        self.eps, self.noise = config["rms_norm_eps"], config["dspark_noise_token_id"]
        self.layers = nn.ModuleList()
        for index in range(3):
            normal = stage.shard.manifest["normal_scales"][f"mtp.{index}.ffn.experts"][self.tp_rank]
            self.layers.append(
                PreparedDecoderLayer(
                    self.weights.get_submodule(str(index)),
                    config,
                    index + 40,
                    stage.shared,
                    normal,
                    lookup,
                    self.reduce,
                    self.all_gather,
                    device,
                )
            )
        self.register_buffer("offsets", torch.arange(5, device=device, dtype=torch.int32), False)

    def _embed(self, token_ids, embedding):
        width = embedding.weight.shape[0]
        local = token_ids.long() - self.tp_rank * width
        valid = (local >= 0) & (local < width)
        value = F.embedding(local.masked_fill(~valid, 0), embedding.weight)
        return self.reduce(value.masked_fill(~valid.unsqueeze(-1), 0))

    def insert_context(self, target_states, positions, valid_count=None):
        first = self.weights.get_submodule("0")
        main_value = rms_norm(linear(target_states, first.main_proj), first.main_norm.weight, self.eps)
        return tuple(layer.attention.insert_context(main_value, positions, valid_count) for layer in self.layers)

    def _forward_hidden(self, next_token, positions):
        ids = torch.where(self.offsets == 0, next_token.reshape(1), self.noise)
        value = self.embed_input_ids(ids)
        residual = value.unsqueeze(1).expand(-1, 4, -1).contiguous()
        pre = F.one_hot(torch.zeros_like(ids, dtype=torch.int64), 4).float()
        image_mask = torch.zeros_like(ids, dtype=torch.bool)
        for layer in self.layers:
            residual, pre, _ = layer(residual, pre, positions, image_mask)
        hidden = (residual.float() * pre.unsqueeze(-1)).sum(1).to(value.dtype)
        return hidden

    def forward(self, next_token, positions):
        hidden = self._forward_hidden(next_token, positions)
        return hidden, self.compute_logits(hidden)

    def embed_input_ids(self, input_ids):
        return self._embed(input_ids, self.weights.embed)

    def compute_logits(self, hidden):
        last = self.weights.get_submodule("2")
        normalized = rms_norm(hidden, last.norm.weight, self.eps)
        return self.all_gather(F.linear(normalized.float(), self.output_head.weight), dim=-1)

    def compute_logits_local(self, hidden):
        """Compute draft logits on the local vocab shard only."""
        last = self.weights.get_submodule("2")
        normalized = rms_norm(hidden, last.norm.weight, self.eps)
        return F.linear(normalized.float(), self.output_head.weight)

    def _global_argmax(self, local_logits):
        """Reduce a vocab-sharded argmax with O(rows) TP communication.

        Each rank owns a contiguous, equally sized vocab range.  Gathering the
        local ``(max_value, global_id)`` pair preserves the full-logits
        argmax tie order while transferring four scalars per row instead of
        the complete vocabulary.  ``all_gather`` is intentionally used here,
        rather than the BF16-only native peer exchange, because logits are
        FP32 and this path must also work before native replay is enabled.
        """
        from vllm_gaudi.ops.deepseek_v41_verify import vocab_parallel_argmax

        return vocab_parallel_argmax(local_logits, self.tp_rank, self.all_gather)

    def sample_greedy(self, first_token, hidden, logits):
        """Reference sampler for the legacy full-vocabulary draft path."""
        return self._sample_greedy(first_token, hidden, logits, local=False)

    def sample_greedy_local(self, first_token, hidden, logits):
        """Draft sampler for the vocab-sharded device verify graph."""
        return self._sample_greedy(first_token, hidden, logits, local=True)

    def _sample_greedy(self, first_token, hidden, logits, *, local):
        last = self.weights.get_submodule("2")
        previous = first_token.reshape(1)
        token_ids, scores = [], []
        for index in range(5):
            markov = self._embed(previous, last.markov_head.embed)
            bias = F.linear(markov.float(), last.markov_head.head.weight)
            if not local:
                # Keep the compatibility path bit-for-bit equivalent: its
                # caller owns full-vocabulary base logits.
                bias = self.all_gather(bias, dim=-1)
            scores_logits = logits[index : index + 1] + bias
            previous = self._global_argmax(scores_logits) if local else scores_logits.argmax(dim=-1)
            confidence_input = torch.cat((hidden[index : index + 1], markov), dim=-1).float()
            scores.append(linear(confidence_input, last.confidence_head.proj).reshape(1))
            token_ids.append(previous)
        return torch.cat(token_ids), torch.cat(scores)

    def verify_prefix(self, target_hidden, proposed_ids, metadata, target_states, target_positions):
        """Verify the accepted prefix and publish a commit-only record.

        This is intentionally separate from :meth:`draft_from_prefix`. PP0
        only needs ``committed`` and the accepted output ids; making the
        commit record before the three draft layers lets the PP collective
        overlap draft execution instead of waiting for it.
        """
        from vllm_gaudi.ops.deepseek_v41_verify import (
            encode_record_wire,
            pack_record,
            verify_control_from_target,
        )

        target = self._global_argmax(F.linear(target_hidden.float(), self.output_head.weight))
        target, output, committed, output_count, anchor, draft_enabled, status = verify_control_from_target(
            target, proposed_ids, metadata
        )
        self.insert_context(target_states, target_positions, committed)
        placeholder = torch.full_like(proposed_ids, -1)
        commit_record = pack_record(
            metadata, committed, output_count, output, placeholder, torch.zeros_like(draft_enabled), status
        )
        return (
            target,
            output,
            committed,
            output_count,
            anchor,
            draft_enabled,
            status,
            commit_record,
            encode_record_wire(commit_record),
        )

    def draft_from_prefix(
        self, metadata, target_positions, output, committed, output_count, anchor, draft_enabled, status
    ):
        """Run draft layers after the prefix commit has been enqueued."""
        from vllm_gaudi.ops.deepseek_v41_verify import encode_record_wire, pack_record

        next_position = target_positions[0].to(torch.int32) + committed.to(torch.int32)
        draft_positions = next_position + self.offsets
        hidden, logits = self.forward_local(anchor.reshape(1), draft_positions)
        draft_ids, confidence = self.sample_greedy_local(anchor.reshape(1), hidden, logits)
        record = pack_record(metadata, committed, output_count, output, draft_ids, draft_enabled, status)
        return record, encode_record_wire(record), confidence

    def verify_and_propose(self, target_hidden, proposed_ids, metadata, target_states, target_positions):
        """C6 verify, committed context insert and draft control in one graph.

        ``target_states`` and ``target_positions`` are always six rows.  The
        device-side valid count in ``metadata`` masks padded rows before the
        context cache is updated.  Keeping the draft call in this entry avoids
        the old forward -> synchronize -> sampling submission chain.
        """
        from vllm_gaudi.ops.deepseek_v41_verify import (
            encode_record_wire,
            pack_record,
            verify_control_from_target,
        )

        # Keep the head projection and its reduction in the control graph.
        # Copying [C6, hidden] is sufficient; a full target logits tensor never
        # leaves this compiled entry.
        target = self._global_argmax(F.linear(target_hidden.float(), self.output_head.weight))
        target, output, committed, output_count, anchor, draft_enabled, status = verify_control_from_target(
            target, proposed_ids, metadata
        )
        self.insert_context(target_states, target_positions, committed)
        next_position = target_positions[0].to(torch.int32) + committed.to(torch.int32)
        draft_positions = next_position + self.offsets
        # The draft head is vocab-parallel too.  Keep it local until the five
        # greedy ids have been reduced, avoiding six full-vocabulary gathers
        # inside every C6 transaction.
        hidden, logits = self.forward_local(anchor.reshape(1), draft_positions)
        draft_ids, confidence = self.sample_greedy_local(anchor.reshape(1), hidden, logits)
        record = pack_record(metadata, committed, output_count, output, draft_ids, draft_enabled, status)
        # Keep the integer record for the ring/diagnostic contract, while the
        # direct PP collective consumes an exact byte-valued BF16 wire.
        return record, encode_record_wire(record), target, confidence

    def forward_local(self, next_token, positions):
        """Draft forward returning local-vocab logits for graph verification."""
        hidden = self._forward_hidden(next_token, positions)
        return hidden, self.compute_logits_local(hidden)
