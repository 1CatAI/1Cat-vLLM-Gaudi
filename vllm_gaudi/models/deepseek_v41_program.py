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
    hc_post,
    hc_pre,
    quantize_activation,
    rms_norm,
    unpack_swa,
)
from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
from vllm_gaudi.ops.deepseek_v41_shared_experts import shared_expert_moe
from vllm_gaudi.ops.deepseek_v41_weights import canonical_hash


def linear(value, layer):
    if hasattr(layer, "scale"):
        value = quantize_activation(value)
    if getattr(layer, "dense_fp8", False):
        return torch.ops.custom_op.custom_deepseek_v41_dense_fp8_gaudi2(value.contiguous(), layer.weight,
                                                                        layer.channel_scale)
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
        "F8_E4M3": torch.bfloat16
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


def load_weight_tree(shard,
                     tree,
                     device,
                     specs=None,
                     *,
                     woa_sidecar=None,
                     woa_layers=(),
                     expert_n256_layers=None,
                     dense_sidecar=None,
                     dense_config=None):
    n256 = gaudi_envs.VLLM_HPU_DSV41_EXPERT_N256 or gaudi_envs.VLLM_HPU_DSV41_EXPERT_N256_FP8
    dense_names = {
        f"layers.{layer}.attn.{projection}.weight"
        for projection, layers in (dense_config or {}).items() if projection != "version" for layer in layers
    }
    for name, spec in (shard.specs if specs is None else specs).items():
        layer = int(name.split(".")[1]) if name.startswith("layers.") else None
        selected_n256 = n256 and layer is not None and (expert_n256_layers is None or layer in expert_n256_layers)
        if name in dense_names:
            module = tree.get_submodule(name.rpartition(".")[0])
            module.weight = dense_sidecar.tensor(name, device)
            channel = dense_sidecar.tensor(name.removesuffix("weight") + "channel_scale", device)
            if "channel_scale" in module._buffers:
                module.channel_scale = channel
            else:
                module.register_buffer("channel_scale", channel, False)
            module.dense_fp8 = True
            continue
        if (woa_sidecar is not None and name.endswith(".attn.wo_a.weight") and layer in woa_layers):
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
                for suffix in ("_q16", "_s16", "_channel"):
                    if projection + suffix not in module._buffers:
                        module.register_buffer(projection + suffix, None, False)
                    else:
                        setattr(module, projection + suffix, None)
                q, scales, channel = load_projection(shard, name.removesuffix("_q16"), device)
                for suffix, value in (("_q16", q), ("_s16", scales), ("_channel", channel)):
                    setattr(module, projection + suffix, value)
                continue
        value = shard.dense(name, device) if spec["dtype"] == "F8_E4M3" else shard.tensor(name, device)
        # The MLA output projection is consumed as [groups, K, N] by the
        # final einsum. The checkpoint/rank file stores each TP shard in
        # [groups, N, K], which made Synapse insert a 32 MiB DRAM transpose
        # for every captured attention graph. Prepare this one static weight
        # once at load time so the steady graph can feed MME in its native
        # layout. Keep the switch opt-in because old manifests and ordinary
        # prefill graphs still use the checkpoint layout.
        if (gaudi_envs.VLLM_HPU_DSV41_PRETRANSPOSE_ATTN and name.endswith(".wo_a.weight") and value.ndim == 2):
            if value.shape[0] % 4:
                raise ValueError(f"MLA wo_a rows are not divisible by 4: {name} {tuple(value.shape)}")
            value = value.reshape(4, value.shape[0] // 4, value.shape[1]).transpose(-1, -2).contiguous()
        if ((name.endswith("ffn.gate.weight") and not gaudi_envs.VLLM_HPU_DSV41_BF16_ROUTER_GATE)
                or name.endswith("confidence_head.proj.weight")
                or (name == "head.weight" and not gaudi_envs.VLLM_HPU_DSV41_BF16_LM_HEAD)
                or name == "mtp.2.markov_head.head.weight"
                or (".compressor.w" in name and ".weight" in name and int(name.split(".")[1]) in (2, 8, 14))):
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

    def __init__(self, weights, topk, normal_scales, lookup, reduce):
        super().__init__()
        self.weights, self.topk = weights, topk
        self.normal_scales, self.reduce = normal_scales, reduce
        self.register_buffer("lookup", lookup, False)
        self.router_top6 = gaudi_envs.VLLM_HPU_DSV41_ROUTER_TOP6
        self.expert_k128 = gaudi_envs.VLLM_HPU_DSV41_EXPERT_K128
        self.fp8 = False
        self.register_buffer("fp8_w13_scale", None, False)
        self.register_buffer("fp8_w2_scale", None, False)
        self.n256 = topk == 6 and (gaudi_envs.VLLM_HPU_DSV41_EXPERT_N256 or gaudi_envs.VLLM_HPU_DSV41_EXPERT_N256_FP8)
        self.n256_fp8 = self.n256 and gaudi_envs.VLLM_HPU_DSV41_EXPERT_N256_FP8
        self.n256_fused = self.n256_fp8 and gaudi_envs.VLLM_HPU_DSV41_EXPERT_FUSED_QUANT
        self.n256_fused_reduce = self.n256_fused and gaudi_envs.VLLM_HPU_DSV41_EXPERT_FUSED_REDUCE
        self.router_bf16_gate = gaudi_envs.VLLM_HPU_DSV41_BF16_ROUTER_GATE
        self.shared_gate_up = gaudi_envs.VLLM_HPU_DSV41_SHARED_GATE_UP
        self.register_buffer("shared_gate_up_weight", None, False)
        if topk == 6 and gaudi_envs.VLLM_HPU_DSV41_EXPERT_FUSED_QUANT and not self.n256_fp8:
            raise ValueError("Fused expert quantization requires N256 FP8 experts")
        if self.n256 and (gaudi_envs.VLLM_HPU_DSV41_SHARED_C6_EXPERTS or gaudi_envs.VLLM_HPU_DSV41_INDEXED_MOE):
            raise ValueError("N256 prepared storage excludes the old shared/indexed expert layouts")
        if (gaudi_envs.VLLM_HPU_DSV41_EXPERT_COORD_PIPELINE
                and (not self.expert_k128 or gaudi_envs.VLLM_HPU_DSV41_FP8_DECODE)):
            raise ValueError("V4.1 expert coordinate pipeline requires the K128 BF16 decoder")
        if self.expert_k128 and gaudi_envs.VLLM_HPU_DSV41_FP8_DECODE:
            raise ValueError("V4.1 K128 BF16 and FP8 decode must be selected independently")

    def release_shared_gate_up_weight(self):
        self.shared_gate_up_weight = None

    def prepare_shared_gate_up_weight(self):
        if not self.shared_gate_up:
            return
        shared = self.weights.shared_experts
        self.shared_gate_up_weight = torch.cat((shared.w1.weight, shared.w3.weight), dim=0).contiguous()
        for projection in (shared.w1, shared.w3):
            source = projection.weight
            projection.weight = torch.empty(source.shape, dtype=source.dtype, device="meta")

    def shared_expert(self, value):
        shared = self.weights.shared_experts
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
        return (gaudi_envs.VLLM_HPU_DSV41_INDEXED_MOE and self.topk == 6 and value.device.type == "hpu"
                and value.dtype == torch.bfloat16 and value.ndim == 2 and 1 <= value.shape[0] <= 6
                and value.shape[-1] == 5120 and ids.device == value.device and ids.dtype == torch.int64
                and ids.ndim == 2 and ids.shape[-1] == 6 and routing.device == value.device
                and routing.dtype == torch.float32 and routing.shape == ids.shape
                and experts.w13_q16.shape == (384, 18, 163840) and experts.w2_q16.shape == (384, 40, 36864)
                and experts.w13_s16.shape == (384, 18, 20480) and experts.w2_s16.shape == (384, 40, 4608))

    def _forward_indexed(self, value, ids, routing):
        """Compute six routed experts without a decoded-weight tensor.

        Q16's N-major layout lets one TPC program accumulate a 128-row output
        tile in registers. The two custom calls emit only gate/up and down
        activations; BF16 boundaries remain in the same places as the prepared
        MME path before FP32 SwiGLU and ordered router accumulation.
        """
        experts = self.weights.experts
        gate_up = torch.ops.custom_op.custom_deepseek_v41_mxfp4_indexed_fc1_bf16_gaudi2(
            value, ids.to(torch.int32), experts.w13_q16, experts.w13_s16, self.lookup, self.normal_scales)
        intermediate = gate_up.shape[-1] // 2
        gate = gate_up[..., :intermediate].float().clamp(max=10.0)
        up = gate_up[..., intermediate:].float().clamp(-10.0, 10.0)
        # The reference path applies router weights before the BF16 boundary
        # feeding W2.  Keep that ordering so the direct kernel is only a data
        # path change, not a numerical/rounding change.
        routed_input = F.silu(gate) * up * routing.float().unsqueeze(-1)
        routed_input = routed_input.to(value.dtype)
        down = torch.ops.custom_op.custom_deepseek_v41_mxfp4_indexed_fc2_bf16_gaudi2(
            routed_input, ids.to(torch.int32), experts.w2_q16, experts.w2_s16, self.lookup, self.normal_scales)
        result = down.float()[:, 0]
        for expert in range(1, 6):
            result = result + down.float()[:, expert]
        return result.to(value.dtype)

    def forward(self, value, image_mask, ready_outputs=(), *, fp8_decode=False):
        w = self.weights
        gate_logits = (torch.ops.custom_op.custom_deepseek_v41_bf16_linear_f32_gaudi2(
            value.contiguous(), w.gate.weight) if self.router_bf16_gate else F.linear(value.float(), w.gate.weight))
        scores = F.softplus(gate_logits).sqrt()
        if self.router_top6:
            if self.topk != 6 or scores.shape[1] != 384:
                raise ValueError("Native V4.1 Router requires 384 experts and top6")
            ids, routing = torch.ops.custom_op.custom_deepseek_v41_router_top6_gaudi2(
                scores, w.gate.bias, w.gate.bias_vl, image_mask)
        else:
            bias = torch.where(image_mask.unsqueeze(-1), w.gate.bias_vl, w.gate.bias)
            ids = torch.topk(scores + bias, self.topk, dim=-1, sorted=True).indices
            routing = scores.gather(1, ids)
            routing = routing / (routing.sum(-1, keepdim=True) + 1e-20) * 1.5
        experts = w.experts
        if self.n256:
            operands = (value, ids.to(torch.int32), routing.float(), experts.w13_q16, experts.w2_q16, experts.w13_s16,
                        experts.w2_s16, self.lookup)
            if self.n256_fp8 and fp8_decode:
                if not 1 <= value.shape[0] <= 6:
                    raise ValueError("N256 FP8 native decode requires C1-C6")
                # The finalize kernel owns a C1-only ordered top-6 reduction.
                # C2-C6 prompt/verify batches retain the same fused FP8 expert
                # body and use the established reduction outside that kernel.
                op = (torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_fused_reduce_fp8_gaudi2
                      if self.n256_fused_reduce and value.shape[0] == 1 else
                      torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_fused_fp8_gaudi2
                      if self.n256_fused else torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_fp8_gaudi2)
                output = op(*operands, experts.w13_channel, experts.w2_channel, self.normal_scales)
            else:
                output = torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_bf16_gaudi2(
                    *operands, self.normal_scales)
        elif self.fp8 and fp8_decode:
            if value.shape[0] != 1:
                raise ValueError("The legacy FP8 expert path requires C1")
            output = torch.ops.custom_op.custom_deepseek_v41_mxfp4_prepared_moe_fp8_gaudi2(
                value, ids.to(torch.int32), routing.float(), experts.w13_q16, experts.w2_q16, experts.w13_s16,
                experts.w2_s16, self.lookup, self.fp8_w13_scale, self.fp8_w2_scale, self.normal_scales)
        elif self.expert_k128 and self.topk == 6 and value.shape[0] == 1:
            namespace = torch.ops.custom_op
            name = "custom_deepseek_v41_mxfp4_prepared_moe_k128_bf16_gaudi2"
            op = (getattr(namespace, name)
                  if hasattr(namespace, name) else namespace.custom_deepseek_v41_mxfp4_k128_moe_bf16_gaudi2)
            output = op(value, ids.to(torch.int32), routing.float(), experts.w13_q16, experts.w2_q16, experts.w13_s16,
                        experts.w2_s16, self.lookup, self.normal_scales)
        elif (gaudi_envs.VLLM_HPU_DSV41_SHARED_C6_EXPERTS and self.topk == 6 and value.device.type == "hpu"
              and 2 <= value.shape[0] <= 6):
            output = shared_expert_moe(value, ids, routing.float(), experts, self.lookup, self.normal_scales)
        elif self._can_use_indexed(value, ids, routing):
            output = self._forward_indexed(value, ids, routing)
        else:
            op = torch.ops.custom_op.custom_deepseek_v41_mxfp4_prepared_moe_bf16_gaudi2
            if (gaudi_envs.VLLM_HPU_DSV41_TILED_EXPERT_DECODE and self.topk == 6 and value.device.type == "hpu"
                    and 2 <= value.shape[0] <= 6):
                op = torch.ops.custom_op.custom_deepseek_v41_mxfp4_k128_moe_bf16_gaudi2
            if (gaudi_envs.VLLM_HPU_DSV41_W13_N512 and self.topk == 6 and value.device.type == "hpu"
                    and 2 <= value.shape[0] <= 6):
                op = torch.ops.custom_op.custom_deepseek_v41_mxfp4_n512_moe_bf16_gaudi2
            output = op(value, ids.to(torch.int32), routing.float(), experts.w13_q16, experts.w2_q16, experts.w13_s16,
                        experts.w2_s16, self.lookup, self.normal_scales)
        shared_out = self.shared_expert(value)
        partial = (output.float() + shared_out.float()).to(value.dtype)
        return self.reduce(partial, ready_outputs=ready_outputs) if ready_outputs else self.reduce(partial)


class PreparedDecoderLayer(nn.Module):

    def __init__(self,
                 weights,
                 config,
                 layer,
                 shared,
                 normal_scales,
                 lookup,
                 reduce,
                 all_gather,
                 device,
                 collect_target_state=False):
        super().__init__()
        self.weights, self.layer = weights, layer
        self.draft = layer >= config["num_hidden_layers"]
        self.collect_target_state = collect_target_state and layer in (37, 38, 39)
        self.eps, self.hc_eps, self.iterations = config["rms_norm_eps"], config["hc_eps"], config["hc_sinkhorn_iters"]
        if shared.length > 512:
            from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2Attention
            self.attention = PagedCSA2Attention(weights.attn, config, layer, shared, linear, reduce, all_gather, device)
        else:
            self.attention = CSA2Attention(weights.attn, config, layer, shared, linear, reduce, device)
        self.moe = PreparedMoE(weights.ffn, config["num_experts_per_tok"], normal_scales, lookup, reduce)
        self.all_gather = all_gather

    def forward(self, residual, pre_mix, positions, image_mask, engram_rows=None, *, fp8_decode=False):
        w = self.weights
        if hasattr(w, "engram"):
            if engram_rows is None:
                raise RuntimeError("Engram layer requires its completed host gather and DMA generation")
            local_rows = engram_rows if engram_rows.dtype == torch.bfloat16 else unpack_swa(engram_rows, 256)
            rows = self.all_gather(local_rows, dim=1)
            kv = linear(rows.flatten(1), w.engram.wkv)
            residual = engram_update(residual, kv, w.engram.q_weight, w.engram.k_weight, ~image_mask, self.eps)
        target_state = residual.mean(1) if self.collect_target_state else None
        value, new_pre, post, comb = hc_pre(residual, pre_mix, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base, self.eps,
                                            self.hc_eps, self.iterations)
        value = rms_norm(value, w.attn_norm.weight, self.eps)
        schedule = (gaudi_envs.VLLM_HPU_DSV41_MHC_SCHEDULE and not self.draft and 1 <= value.shape[0] <= 6)
        if schedule:
            value = self.attention(value, positions, ready_outputs=(post, comb))
        else:
            value = self.attention.draft(value, positions) if self.draft else self.attention(value, positions)
        residual = hc_post(value, residual, post, comb)
        value, pre_mix, post, comb = hc_pre(residual, new_pre, w.hc_ffn_fn, w.hc_ffn_scale, w.hc_ffn_base, self.eps,
                                            self.hc_eps, self.iterations)
        value = self.moe(rms_norm(value, w.ffn_norm.weight, self.eps),
                         image_mask,
                         ready_outputs=(post, comb) if schedule else (),
                         fp8_decode=fp8_decode)
        return hc_post(value, residual, post, comb), pre_mix, target_state


class PreparedStage(nn.Module):

    def __init__(self, directory, pp_rank, tp_rank, reduce, all_gather, device, max_length=512, *, dspark=None):
        super().__init__()
        self.shard = PreparedV41Shard(directory, pp_rank, tp_rank)
        self.config = json.loads((Path(directory) / "config.json").read_text())
        config = self.config["text_config"]
        self.pp_rank, self.tp_rank, self.length = pp_rank, tp_rank, max_length
        self.reduce, self.all_gather = reduce, all_gather
        self.dspark = (gaudi_envs.VLLM_HPU_DSV41_DSPARK if dspark is None else bool(dspark))
        self.bf16_head = gaudi_envs.VLLM_HPU_DSV41_BF16_LM_HEAD
        if self.dspark and (self.bf16_head or gaudi_envs.VLLM_HPU_DSV41_BF16_ROUTER_GATE):
            raise ValueError("BF16 projection candidates require ordinary C1 decode")
        if self.dspark and gaudi_envs.VLLM_HPU_DSV41_SHARED_GATE_UP:
            raise ValueError("Shared gate/up candidate requires ordinary C1 decode")
        if self.dspark and gaudi_envs.VLLM_HPU_DSV41_MHC_GATES_FUSED:
            raise ValueError("Fused mHC gates require ordinary C1 decode")
        self.woa_config = {"version": 1, "layers": []}
        self.dense_config = {"version": 1, "wq_b": [], "wo_b": []}
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
        self.expert_n256 = (gaudi_envs.VLLM_HPU_DSV41_EXPERT_N256 or gaudi_envs.VLLM_HPU_DSV41_EXPERT_N256_FP8)
        self.expert_fused_quant = gaudi_envs.VLLM_HPU_DSV41_EXPERT_FUSED_QUANT
        self.expert_fused_reduce = gaudi_envs.VLLM_HPU_DSV41_EXPERT_FUSED_REDUCE
        if self.expert_fused_quant and not gaudi_envs.VLLM_HPU_DSV41_EXPERT_N256_FP8:
            raise ValueError("Fused expert quantization requires N256 FP8 experts")
        if self.expert_fused_reduce and not (gaudi_envs.VLLM_HPU_DSV41_EXPERT_N256_FP8 and self.expert_fused_quant):
            raise ValueError("Fused expert finalize requires N256 FP8 and fused quantization")
        if self.expert_n256 and (gaudi_envs.VLLM_HPU_DSV41_FP8_DECODE
                                 or gaudi_envs.VLLM_HPU_DSV41_EXPERT_COORD_PIPELINE):
            raise ValueError("N256 experts use their own layout and decoder")
        self.expert_n256_config = {"routed_experts": []}
        if self.expert_n256:
            from vllm_gaudi.ops.deepseek_v41_fp8 import precision_config
            self.expert_n256_config = precision_config(gaudi_envs.VLLM_HPU_DSV41_FP8_CONFIG)
            if not self.expert_n256_config["routed_experts"]:
                self.expert_n256_config["routed_experts"] = list(range(40))
        self.fp8_decode = (gaudi_envs.VLLM_HPU_DSV41_FP8_DECODE or gaudi_envs.VLLM_HPU_DSV41_EXPERT_N256_FP8)
        if (gaudi_envs.VLLM_HPU_DSV41_FP8_DECODE and (self.dspark or not gaudi_envs.VLLM_HPU_DSV41_GRAPH_REPLAY)):
            raise ValueError("Legacy V4.1 FP8 decode requires ordinary C1 replay")
        self.weight_specs = {
            name: spec
            for name, spec in self.shard.specs.items() if self.dspark or not name.startswith("mtp.")
        }
        self.weights = _weight_tree(self.weight_specs)
        self.start, self.stop = (0, 20) if pp_rank == 0 else (20, 40)
        if max_length > 512:
            from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2SharedState
            self.shared = PagedCSA2SharedState(config, self.start, self.stop, device, max_length)
        else:
            self.shared = CSA2SharedState(config, self.start, self.stop, device, max_length)
        lookup = mxfp4_bf16_lut(torch.device(device))
        self.layers = nn.ModuleList()
        for layer in range(self.start, self.stop):
            normal = self.shard.manifest["normal_scales"][f"layers.{layer}.ffn.experts"][tp_rank]
            self.layers.append(
                PreparedDecoderLayer(self.weights.layers.get_submodule(str(layer)),
                                     config,
                                     layer,
                                     self.shared,
                                     normal,
                                     lookup,
                                     reduce,
                                     all_gather,
                                     device,
                                     collect_target_state=self.dspark))
        self.generation, self.loaded = 0, False
        self.runtime_precision = {
            "experts":
            "MXFP4 -> BF16 SRAM -> BF16 MME",
            "dense":
            "E4M3FN/block32 -> prepared BF16; block32 activation quantization",
            "mla_wo_a": ("pretransposed [groups,K,N] BF16"
                         if gaudi_envs.VLLM_HPU_DSV41_PRETRANSPOSE_ATTN else "checkpoint [groups,N,K] BF16"),
            "mHC_router":
            "FP32",
            "communication":
            "BF16"
        }
        self.draft = PreparedDraft(self, lookup, device) if self.dspark and pp_rank == 1 else None

    def load_prepared(self, device):
        if self.loaded:
            self.invalidate()
        for layer in self.layers:
            moe = getattr(layer, "moe", None)
            if moe is not None:
                moe.release_shared_gate_up_weight()
        sidecar = None
        dense_sidecar = None
        if gaudi_envs.VLLM_HPU_DSV41_ATTN_DENSE_FP8:
            from vllm_gaudi.ops.deepseek_v41_dense_fp8 import DenseFP8Sidecar
            dense_sidecar = DenseFP8Sidecar(gaudi_envs.VLLM_HPU_DSV41_ATTN_DENSE_FP8_SIDECAR, self.shard)
        if gaudi_envs.VLLM_HPU_DSV41_WO_A_FP8:
            from vllm_gaudi.ops.deepseek_v41_woa_fp8 import WoaFP8Sidecar
            sidecar = WoaFP8Sidecar(gaudi_envs.VLLM_HPU_DSV41_WO_A_FP8_SIDECAR, self.shard)
        load_weight_tree(self.shard,
                         self.weights,
                         device,
                         self.weight_specs,
                         woa_sidecar=sidecar,
                         woa_layers=self.woa_config["layers"],
                         expert_n256_layers=self.expert_n256_config["routed_experts"],
                         dense_sidecar=dense_sidecar,
                         dense_config=self.dense_config)
        if dense_sidecar is not None:
            self.runtime_precision["attention_dense_fp8"] = {
                "config": self.dense_config,
                "weight_fingerprint": dense_sidecar.fingerprint,
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
            if gaudi_envs.VLLM_HPU_DSV41_PREPARED_OUTPUT:
                attention.prepare_output_weight()
        if sidecar is not None:
            self.runtime_precision["wo_a_fp8"] = {
                "config": self.woa_config,
                "weight_fingerprint": sidecar.fingerprint,
            }
        if self.expert_n256:
            from vllm_gaudi.ops.deepseek_v41_expert_n256 import FINGERPRINT, LAYOUT
            enabled_layers = set(self.expert_n256_config["routed_experts"])
            for layer in self.layers:
                enabled = layer.layer in enabled_layers
                layer.moe.n256 = enabled
                layer.moe.n256_fp8 = (enabled and gaudi_envs.VLLM_HPU_DSV41_EXPERT_N256_FP8)
                layer.moe.n256_fused = (layer.moe.n256_fp8 and self.expert_fused_quant)
                layer.moe.n256_fused_reduce = (layer.moe.n256_fused and self.expert_fused_reduce)
            self.runtime_precision["expert_n256"] = {
                "config": self.expert_n256_config,
                "layout": LAYOUT,
                "layout_fingerprint": FINGERPRINT,
                "c1_c6": "FP8xFP8" if gaudi_envs.VLLM_HPU_DSV41_EXPERT_N256_FP8 else "BF16xBF16",
                "prefill": "BF16 from retained original scales",
                "fused_quant": gaudi_envs.VLLM_HPU_DSV41_EXPERT_FUSED_QUANT,
                "fused_reduce": self.expert_fused_reduce,
            }
            self.runtime_precision["experts"] = ("MXFP4 -> FP8 SRAM -> FP8xFP8 MME for native C1-C6; BF16 prefill"
                                                 if gaudi_envs.VLLM_HPU_DSV41_EXPERT_N256_FP8 else
                                                 "N256 MXFP4 -> BF16 SRAM -> BF16 MME")
        elif gaudi_envs.VLLM_HPU_DSV41_FP8_DECODE:
            from vllm_gaudi.ops.deepseek_v41_fp8 import FP8Sidecar, precision_config
            config = precision_config(gaudi_envs.VLLM_HPU_DSV41_FP8_CONFIG)
            expert_sidecar = FP8Sidecar(gaudi_envs.VLLM_HPU_DSV41_FP8_SIDECAR, self.shard)
            for layer in self.layers:
                enabled = layer.layer in config["routed_experts"]
                layer.moe.fp8 = enabled
                for projection in ("w13", "w2"):
                    value = expert_sidecar.tensor(f"layers.{layer.layer}.ffn.experts.{projection}_fp8_channel_scale",
                                                  device) if enabled else None
                    setattr(layer.moe, f"fp8_{projection}_scale", value)
            self.runtime_precision["fp8"] = {
                "config": config,
                "weight_fingerprint": expert_sidecar.fingerprint,
                "route": "MXFP4 -> E4M3 SRAM -> FP8xFP8 MME -> FP32 scale -> BF16",
            }
        self.runtime_precision["selected_mla_mme"] = gaudi_envs.VLLM_HPU_DSV41_MLA_MME
        self.runtime_precision["router_selection"] = ("FP32 scores / native top6 / smallest-ID ties"
                                                      if gaudi_envs.VLLM_HPU_DSV41_ROUTER_TOP6 else "torch.topk")
        self.runtime_precision["router_gate"] = ("BF16xBF16 MME -> FP32 logits"
                                                 if gaudi_envs.VLLM_HPU_DSV41_BF16_ROUTER_GATE else "FP32 MME")
        self.runtime_precision["head"] = ("BF16xBF16 MME -> FP32" if self.bf16_head else "FP32 MME")
        self.runtime_precision["attention_input"] = ("fused wq_a+wkv BF16 MME / one activation quantization" if
                                                     gaudi_envs.VLLM_HPU_DSV41_QKV_FUSED_INPUT else "separate wq_a/wkv")
        self.runtime_precision["compressor_input"] = ("fused ratio-2 FP32 wkv+wgate MME"
                                                      if gaudi_envs.VLLM_HPU_DSV41_COMPRESSOR_FUSED_INPUT else
                                                      "separate ratio-2 wkv/wgate MME")
        self.runtime_precision["mla"] = ("shared-KV BF16 QK / FP32 softmax and PV / BF16 output v1"
                                         if gaudi_envs.VLLM_HPU_DSV41_MLA_MME else "TPC online softmax")
        if gaudi_envs.VLLM_HPU_DSV41_MLA_BF16_PV:
            self.runtime_precision["mla"] = "shared BF16 KV / BF16 exp-PV / FP32 denominator and accumulator v1"
        self.runtime_precision["attention_kv_first"] = gaudi_envs.VLLM_HPU_DSV41_ATTN_KV_FIRST
        self.runtime_precision["attention_fused_norm"] = gaudi_envs.VLLM_HPU_DSV41_ATTN_FUSED_NORM
        self.runtime_precision["q_scale_rope"] = gaudi_envs.VLLM_HPU_DSV41_Q_SCALE_ROPE
        self.runtime_precision["shared_gate_up"] = gaudi_envs.VLLM_HPU_DSV41_SHARED_GATE_UP
        self.runtime_precision["mhc_gates_fused"] = gaudi_envs.VLLM_HPU_DSV41_MHC_GATES_FUSED
        self.precision_fingerprint = canonical_hash(self.runtime_precision)
        self.loaded = True
        self.generation += 1

    def invalidate(self):
        # The execution owner must close its recipes before reloading or
        # migrating buffers. A live plan cannot retain addresses from this tree.
        if getattr(self, "replay_owner", None) is not None:
            self.replay_owner.close()
        for layer in self.layers:
            attention = getattr(layer, "attention", None)
            if attention is not None:
                attention.invalidate_qkv_input_weight()
                attention.invalidate_compressor_input_weight()
            moe = getattr(layer, "moe", None)
            if moe is not None:
                moe.release_shared_gate_up_weight()
        self.loaded = False
        self.generation += 1

    def embed(self, input_ids, *, draft=False):
        embedding = self.weights.mtp.embed if draft else self.weights.embed
        per_rank = embedding.weight.shape[0]
        local = input_ids.long() - self.tp_rank * per_rank
        valid = (local >= 0) & (local < per_rank)
        values = F.embedding(local.masked_fill(~valid, 0), embedding.weight)
        return self.reduce(values.masked_fill(~valid.unsqueeze(-1), 0))

    def forward(self, residual, pre_mix, positions, input_ids, engram_rows):
        image_mask = (input_ids == 129264) | (input_ids == 129265)
        target_states = []
        for layer in self.layers:
            rows = engram_rows[0 if layer.layer == 1 else 1] if layer.layer in (1, 14) else None
            residual, pre_mix, target = layer(residual, pre_mix, positions, image_mask, rows)
            if target is not None:
                target_states.append(target)
        if self.pp_rank == 0:
            return residual, pre_mix, None
        value = (residual.float() * pre_mix.unsqueeze(-1)).sum(1).to(residual.dtype)
        value = rms_norm(value, self.weights.norm.weight, self.config["text_config"]["rms_norm_eps"])
        return value, pre_mix, torch.cat(target_states, -1) if target_states else None

    def _head_projection(self, hidden):
        if self.bf16_head:
            return torch.ops.custom_op.custom_deepseek_v41_bf16_linear_f32_gaudi2(hidden.contiguous(),
                                                                                  self.weights.head.weight)
        return F.linear(hidden.float(), self.weights.head.weight)

    def logits(self, hidden):
        if self.pp_rank != 1:
            raise RuntimeError("Only the final PP stage owns the output head")
        local = self._head_projection(hidden)
        return self.all_gather(local, dim=-1)

    def sample_greedy(self, hidden):
        from vllm_gaudi.ops.deepseek_v41_sampling import (
            local_greedy_candidate,
            select_greedy_candidate,
        )
        if self.pp_rank != 1 or hidden.shape[0] != 1:
            raise ValueError("Ordinary greedy head requires one token on PP1")
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


class PreparedLayerGroup(nn.Module):
    """Bound FX dependency closure without changing the stage tensor program."""

    def __init__(self,
                 stage,
                 start,
                 stop,
                 *,
                 pp_wire_input=False,
                 fused_text_io=False,
                 fp8_decode=False):
        super().__init__()
        self.fp8_decode = fp8_decode
        self.pp_wire_input = pp_wire_input and start == 0
        self.text_input = fused_text_io and stage.pp_rank == 0 and start == 0
        self.wire_output = fused_text_io and stage.pp_rank == 0 and stop == len(stage.layers)
        self.embedding = stage.weights.embed if self.text_input else None
        self.tp_rank = getattr(stage, "tp_rank", 0)
        self.reduce = getattr(stage, "reduce", None)
        self.layers = nn.ModuleList(list(stage.layers[start:stop]))
        self.final = stage.pp_rank == 1 and stop == len(stage.layers)
        self.norm = stage.weights.norm if self.final else None
        self.eps = stage.config["text_config"]["rms_norm_eps"]
        self.native_input = None

    def _forward(self, residual, pre_mix, positions, input_ids, engram_rows, *, fp8_decode=False):
        if self.text_input:
            mapped = input_ids.masked_fill(input_ids == 129265, 129264)
            per_rank = self.embedding.weight.shape[0]
            local = mapped.long() - self.tp_rank * per_rank
            valid = (local >= 0) & (local < per_rank)
            values = F.embedding(local.masked_fill(~valid, 0), self.embedding.weight)
            values = self.reduce(values.masked_fill(~valid.unsqueeze(-1), 0))
            residual = values.unsqueeze(1).expand(-1, 4, -1).contiguous()
            pre_mix = torch.tensor([1., 0., 0., 0.], device=values.device).expand(values.shape[0], 4)
        if self.pp_wire_input:
            from vllm_gaudi.ops.deepseek_v41_pp_wire import decode_pp_wire
            residual, pre_mix = decode_pp_wire(residual)
        image_mask = (input_ids == 129264) | (input_ids == 129265)
        target_states = []
        for layer in self.layers:
            rows = engram_rows[0 if layer.layer == 1 else 1] if layer.layer in (1, 14) else None
            residual, pre_mix, target = layer(residual,
                                              pre_mix,
                                              positions,
                                              image_mask,
                                              rows,
                                              fp8_decode=fp8_decode)
            if target is not None:
                target_states.append(target)
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
        value = (residual.float() * pre_mix.unsqueeze(-1)).sum(1).to(residual.dtype)
        value = rms_norm(value, self.norm.weight, self.eps)
        return value, pre_mix, torch.cat(target_states, -1) if target_states else None

    def forward(self, residual, pre_mix, positions, input_ids, engram_rows):
        return self._forward(residual, pre_mix, positions, input_ids, engram_rows, fp8_decode=self.fp8_decode)

    def native_forward(self, residual, pre_mix, positions, input_ids, engram_rows):
        if self.native_input is not None:
            residual, pre_mix = self.native_input(input_ids)
        return self(residual, pre_mix, positions, input_ids, engram_rows)


_compile_entry_ids = count()


def _compile_group(group, *, native, backend="hpu_backend"):
    # Dynamo caches variants by code object. Each static layer group/bucket
    # owns its entry so legitimate preparations cannot exhaust another group.
    method = group.native_forward if native else group.forward
    function = method.__func__
    name = f"{function.__name__}_v41_{next(_compile_entry_ids)}"
    entry = FunctionType(function.__code__.replace(co_name=name), function.__globals__, name, function.__defaults__,
                         function.__closure__)
    entry.__kwdefaults__ = function.__kwdefaults__
    entry.__module__ = function.__module__
    return torch.compile(MethodType(entry, group), backend=backend, fullgraph=True, dynamic=False)


class CompiledStage:

    def __init__(self, stage, *, native=False, pp_wire_input=False, fused_text_io=False, native_input=False):
        legacy_fp8 = getattr(stage, "fp8_decode", False) and not getattr(stage, "expert_n256", False)
        if native_input and (not native or stage.pp_rank != 0 or stage.dspark or legacy_fp8):
            raise ValueError("Native input capture requires ordinary BF16 PP0 decode")
        if native_input and (pp_wire_input or fused_text_io):
            raise ValueError("Native input capture has a single PP0 input owner")
        backend = "hpu_backend"
        if native and gaudi_envs.VLLM_HPU_DSV41_TP_MHC_OVERLAP:
            if stage.dspark or (stage.fp8_decode and not stage.expert_n256):
                raise ValueError("TP/mHC overlap requires BF16 boundaries and a qualified C1 expert layout")
            from vllm_gaudi.compilation.deepseek_v41_overlap import make_backend
            backend = make_backend()
        self.groups = tuple(
            PreparedLayerGroup(stage,
                               start,
                               start + 4,
                               pp_wire_input=pp_wire_input,
                               fused_text_io=fused_text_io,
                               fp8_decode=native and stage.fp8_decode)
            for start in range(0, 20, 4))
        if native_input:
            self.groups[0].native_input = PreparedInput(stage.weights.embed, stage.tp_rank, stage.reduce)
        self.chunks = tuple(_compile_group(group, native=native, backend=backend) for group in self.groups)

    def __call__(self, hidden, pre_mix, positions, input_ids, engram):
        for chunk in self.chunks:
            hidden, pre_mix, aux = chunk(hidden, pre_mix, positions, input_ids, engram)
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
                PreparedDecoderLayer(self.weights.get_submodule(str(index)), config, index + 40, stage.shared, normal,
                                     lookup, self.reduce, self.all_gather, device))
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
            scores_logits = logits[index:index + 1] + bias
            previous = (self._global_argmax(scores_logits) if local else scores_logits.argmax(dim=-1))
            confidence_input = torch.cat((hidden[index:index + 1], markov), dim=-1).float()
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
            target, proposed_ids, metadata)
        self.insert_context(target_states, target_positions, committed)
        placeholder = torch.full_like(proposed_ids, -1)
        commit_record = pack_record(metadata, committed, output_count, output, placeholder,
                                    torch.zeros_like(draft_enabled), status)
        return (target, output, committed, output_count, anchor, draft_enabled, status, commit_record,
                encode_record_wire(commit_record))

    def draft_from_prefix(self, metadata, target_positions, output, committed, output_count, anchor, draft_enabled,
                          status):
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
            target, proposed_ids, metadata)
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
