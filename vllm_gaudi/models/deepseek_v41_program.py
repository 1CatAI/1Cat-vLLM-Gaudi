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
from vllm_gaudi.ops.deepseek_v41_weights import canonical_hash


def linear(value, layer):
    if hasattr(layer, "scale"):
        value = quantize_activation(value)
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


def load_weight_tree(shard, tree, device, specs=None, *, woa_sidecar=None, woa_layers=(), expert_n256_layers=()):
    converted = set()
    for name, spec in (shard.specs if specs is None else specs).items():
        if name in converted:
            continue
        if (name.startswith("layers.") and ".ffn.experts." in name and name.endswith(("_q16", "_s16"))
                and int(name.split(".")[1]) in expert_n256_layers):
            from vllm_gaudi.ops.deepseek_v41_expert_n256 import load_projection
            prefix = name[:-4]
            module_name, _, projection = prefix.rpartition(".")
            module = tree.get_submodule(module_name)
            # A closed recipe must release its former device allocations before
            # reload prepares a replacement. A failed reload is not executable.
            for attribute in (projection + "_q16", projection + "_s16", projection + "_n256_channel"):
                previous = getattr(module, attribute, None)
                if previous is not None:
                    setattr(module, attribute, torch.empty(previous.shape, dtype=previous.dtype, device="meta"))
            q, planes, channel = load_projection(shard, prefix, device)
            setattr(module, projection + "_q16", q)
            setattr(module, projection + "_s16", planes)
            key = projection + "_n256_channel"
            if key in module._buffers:
                setattr(module, key, channel)
            else:
                module.register_buffer(key, channel, False)
            converted.update((prefix + "_q16", prefix + "_s16"))
            continue
        if name.endswith(".attn.wo_a.weight") and int(name.split(".")[1]) in woa_layers:
            module = tree.get_submodule(name.rpartition(".")[0])
            module.weight = woa_sidecar.tensor(name, device)
            channel = woa_sidecar.tensor(name.removesuffix("weight") + "channel_scale", device)
            if "channel_scale" not in module._buffers:
                module.register_buffer("channel_scale", channel, False)
            else:
                module.channel_scale = channel
            continue
        value = shard.dense(name, device) if spec["dtype"] == "F8_E4M3" else shard.tensor(name, device)
        if (name.endswith("ffn.gate.weight") or name.endswith("confidence_head.proj.weight")
                or name == "mtp.2.markov_head.head.weight"
                or (name == "head.weight" and not gaudi_envs.VLLM_HPU_DSV41_BF16_LM_HEAD)
                or (".compressor.w" in name and ".weight" in name and int(name.split(".")[1]) in (2, 8, 14))):
            value = value.float()
        module_name, _, attribute = name.rpartition(".")
        setattr(tree.get_submodule(module_name), attribute, value)
    shard.check_identity()


class PreparedInput(nn.Module):
    """Compile the complete PP0 input chain, including its embedding reduction."""

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
        self.fp8 = False
        self.n256 = False
        self.n256_fused = gaudi_envs.VLLM_HPU_DSV41_EXPERT_FUSED_QUANT
        self.router_top6 = gaudi_envs.VLLM_HPU_DSV41_ROUTER_TOP6
        self.expert_k128 = gaudi_envs.VLLM_HPU_DSV41_EXPERT_K128
        if gaudi_envs.VLLM_HPU_DSV41_EXPERT_COORD_PIPELINE and (not self.expert_k128
                                                                or gaudi_envs.VLLM_HPU_DSV41_FP8_DECODE):
            raise ValueError("V4.1 expert coordinate pipeline requires the K128 BF16 decoder")
        if self.expert_k128 and gaudi_envs.VLLM_HPU_DSV41_FP8_DECODE:
            raise ValueError("V4.1 K128 BF16 and FP8 decode must be selected independently")
        self.register_buffer("fp8_w13_scale", None, False)
        self.register_buffer("fp8_w2_scale", None, False)

    def forward(self, value, image_mask, *, fp8_decode=False):
        w = self.weights
        scores = F.softplus(F.linear(value.float(), w.gate.weight)).sqrt()
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
            if fp8_decode:
                op = (torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_fused_fp8_gaudi2
                      if self.n256_fused else torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_fp8_gaudi2)
                output = op(
                    value, ids.to(torch.int32), routing.float(), experts.w13_q16, experts.w2_q16, experts.w13_s16,
                    experts.w2_s16, self.lookup, experts.w13_n256_channel, experts.w2_n256_channel, self.normal_scales)
            else:
                output = torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_bf16_gaudi2(
                    value, ids.to(torch.int32), routing.float(), experts.w13_q16, experts.w2_q16, experts.w13_s16,
                    experts.w2_s16, self.lookup, self.normal_scales)
        elif self.fp8 and fp8_decode:
            if value.shape[0] != 1:
                raise ValueError("V4.1 FP8 expert replay requires C1")
            output = torch.ops.custom_op.custom_deepseek_v41_mxfp4_prepared_moe_fp8_gaudi2(
                value, ids.to(torch.int32), routing.float(), experts.w13_q16, experts.w2_q16, experts.w13_s16,
                experts.w2_s16, self.lookup, self.fp8_w13_scale, self.fp8_w2_scale, self.normal_scales)
        elif self.expert_k128 and value.shape[0] == 1:
            if self.topk != 6:
                raise ValueError("V4.1 K128 decode requires top6")
            output = torch.ops.custom_op.custom_deepseek_v41_mxfp4_prepared_moe_k128_bf16_gaudi2(
                value, ids.to(torch.int32), routing.float(), experts.w13_q16, experts.w2_q16, experts.w13_s16,
                experts.w2_s16, self.lookup, self.normal_scales)
        else:
            output = torch.ops.custom_op.custom_deepseek_v41_mxfp4_prepared_moe_bf16_gaudi2(
                value, ids.to(torch.int32), routing.float(), experts.w13_q16, experts.w2_q16, experts.w13_s16,
                experts.w2_s16, self.lookup, self.normal_scales)
        shared = w.shared_experts
        gate = linear(value, shared.w1).float().clamp(max=10.0)
        up = linear(value, shared.w3).float().clamp(-10.0, 10.0)
        shared_out = linear((F.silu(gate) * up).to(value.dtype), shared.w2)
        return self.reduce((output.float() + shared_out.float()).to(value.dtype))


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
        self.attention = CSA2Attention(weights.attn, config, layer, shared, linear, reduce, device)
        self.moe = PreparedMoE(weights.ffn, config["num_experts_per_tok"], normal_scales, lookup, reduce)
        self.all_gather = all_gather

    def forward(self, residual, pre_mix, positions, image_mask, engram_rows=None, *, fp8_decode=False):
        w = self.weights
        if hasattr(w, "engram"):
            if engram_rows is None:
                raise RuntimeError("Engram layer requires its completed host gather and DMA generation")
            rows = self.all_gather(unpack_swa(engram_rows, 256), dim=1)
            kv = linear(rows.flatten(1), w.engram.wkv)
            residual = engram_update(residual, kv, w.engram.q_weight, w.engram.k_weight, ~image_mask, self.eps)
        target_state = residual.mean(1) if self.collect_target_state else None
        value, new_pre, post, comb = hc_pre(residual, pre_mix, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base, self.eps,
                                            self.hc_eps, self.iterations)
        value = rms_norm(value, w.attn_norm.weight, self.eps)
        value = self.attention.draft(value, positions) if self.draft else self.attention(value, positions)
        residual = hc_post(value, residual, post, comb)
        value, pre_mix, post, comb = hc_pre(residual, new_pre, w.hc_ffn_fn, w.hc_ffn_scale, w.hc_ffn_base, self.eps,
                                            self.hc_eps, self.iterations)
        value = self.moe(rms_norm(value, w.ffn_norm.weight, self.eps), image_mask, fp8_decode=fp8_decode)
        return hc_post(value, residual, post, comb), pre_mix, target_state


class PreparedStage(nn.Module):

    def __init__(self, directory, pp_rank, tp_rank, reduce, all_gather, device, max_length=512, *, dspark=True):
        super().__init__()
        self.shard = PreparedV41Shard(directory, pp_rank, tp_rank)
        self.config = json.loads((Path(directory) / "config.json").read_text())
        config = self.config["text_config"]
        self.pp_rank, self.tp_rank, self.length = pp_rank, tp_rank, max_length
        self.reduce, self.all_gather = reduce, all_gather
        self.dspark = bool(dspark)
        if self.dspark and gaudi_envs.VLLM_HPU_DSV41_MLA_MME:
            raise ValueError("MME MLA candidate requires ordinary C1 decode")
        if self.dspark and gaudi_envs.VLLM_HPU_DSV41_QKV_FUSED_INPUT:
            raise ValueError("QKV fused input candidate requires ordinary C1 decode")
        self.bf16_head = gaudi_envs.VLLM_HPU_DSV41_BF16_LM_HEAD
        if self.dspark and (self.bf16_head or gaudi_envs.VLLM_HPU_DSV41_ROUTER_TOP6):
            raise ValueError("Projection candidates require DSpark disabled")
        self.woa_config = {"version": 1, "layers": []}
        if gaudi_envs.VLLM_HPU_DSV41_WO_A_FP8:
            from vllm_gaudi.ops.deepseek_v41_woa_fp8 import layer_selection
            if self.dspark:
                raise ValueError("wo_a FP8 requires DSpark disabled")
            self.woa_config = layer_selection(gaudi_envs.VLLM_HPU_DSV41_WO_A_FP8_CONFIG)
        self.expert_n256 = gaudi_envs.VLLM_HPU_DSV41_EXPERT_N256_FP8
        self.expert_fused_quant = gaudi_envs.VLLM_HPU_DSV41_EXPERT_FUSED_QUANT
        if self.expert_fused_quant and not self.expert_n256:
            raise ValueError("Fused expert scale/SwiGLU/quantization requires the N256 FP8 path")
        if self.expert_n256 and (gaudi_envs.VLLM_HPU_DSV41_FP8_DECODE
                                 or gaudi_envs.VLLM_HPU_DSV41_EXPERT_COORD_PIPELINE):
            raise ValueError("N256 experts require their own weight layout and decoder")
        self.expert_n256_config = {"routed_experts": []}
        if self.expert_n256:
            from vllm_gaudi.ops.deepseek_v41_fp8 import precision_config
            self.expert_n256_config = precision_config(gaudi_envs.VLLM_HPU_DSV41_FP8_CONFIG)
        self.fp8_decode = gaudi_envs.VLLM_HPU_DSV41_FP8_DECODE or self.expert_n256
        if self.fp8_decode and (self.dspark or not gaudi_envs.VLLM_HPU_DSV41_GRAPH_REPLAY):
            raise ValueError("V4.1 FP8 decode requires ordinary C1 native stage replay")
        self.weight_specs = {
            name: spec
            for name, spec in self.shard.specs.items() if self.dspark or not name.startswith("mtp.")
        }
        self.weights = _weight_tree(self.weight_specs)
        self.start, self.stop = (0, 20) if pp_rank == 0 else (20, 40)
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
            "experts": "MXFP4 -> BF16 SRAM -> BF16 MME",
            "dense": "E4M3FN/block32 -> prepared BF16; block32 activation quantization",
            "mHC_router": "FP32",
            "communication": "BF16"
        }
        self.draft = PreparedDraft(self, lookup, device) if self.dspark and pp_rank == 1 else None

    def load_prepared(self, device):
        if self.loaded:
            self.invalidate()
        if gaudi_envs.VLLM_HPU_DSV41_QKV_FUSED_INPUT:
            for layer in self.layers:
                attention = getattr(layer, "attention", None)
                if attention is not None:
                    attention.invalidate_qkv_input_weight()
        sidecar = None
        if gaudi_envs.VLLM_HPU_DSV41_WO_A_FP8:
            from vllm_gaudi.ops.deepseek_v41_woa_fp8 import WoaFP8Sidecar
            sidecar = WoaFP8Sidecar(gaudi_envs.VLLM_HPU_DSV41_WO_A_FP8_SIDECAR, self.shard)
        load_weight_tree(self.shard,
                         self.weights,
                         device,
                         self.weight_specs,
                         woa_sidecar=sidecar,
                         woa_layers=self.woa_config["layers"],
                         expert_n256_layers=self.expert_n256_config["routed_experts"])
        if gaudi_envs.VLLM_HPU_DSV41_QKV_FUSED_INPUT:
            for layer in self.layers:
                attention = getattr(layer, "attention", None)
                if attention is not None:
                    attention.prepare_qkv_input_weight()
        for layer in self.layers:
            # Keep lightweight contract doubles usable in loader tests. Real
            # decoder layers always expose ``attention``; a test double may
            # intentionally model only weight ownership and should not be
            # forced to construct the full attention module.
            attention = getattr(layer, "attention", None)
            if attention is not None:
                attention.woa_fp8 = layer.layer in self.woa_config["layers"]
        if sidecar is not None:
            self.runtime_precision["wo_a_fp8"] = {"config": self.woa_config, "weight_fingerprint": sidecar.fingerprint}

        if gaudi_envs.VLLM_HPU_DSV41_PREPARED_OUTPUT:
            for layer in self.layers:
                attention = getattr(layer, "attention", None)
                if attention is not None:
                    attention.prepare_output_weight()
        if self.expert_n256:
            from vllm_gaudi.ops.deepseek_v41_expert_n256 import FINGERPRINT, LAYOUT
            for layer in self.layers:
                layer.moe.n256 = layer.layer in self.expert_n256_config["routed_experts"]
            self.runtime_precision["expert_n256"] = {
                "config": self.expert_n256_config,
                "fused_swiglu_quant": self.expert_fused_quant,
                "layout": LAYOUT,
                "fingerprint": FINGERPRINT,
                "source": self.shard.manifest["model_revision"],
                "source_plan": self.shard.manifest["plan_fingerprint"],
                "route": "MXFP4 -> FP8 SRAM -> FP8xFP8 MME -> fused FP32 scale -> BF16",
                "compatibility": "same Q16/original scale plane -> BF16 SRAM -> BF16 MME"
            }
        elif self.fp8_decode:
            from vllm_gaudi.ops.deepseek_v41_fp8 import FP8Sidecar, precision_config
            config = precision_config(gaudi_envs.VLLM_HPU_DSV41_FP8_CONFIG)
            sidecar = FP8Sidecar(gaudi_envs.VLLM_HPU_DSV41_FP8_SIDECAR, self.shard)
            for layer in self.layers:
                enabled = layer.layer in config["routed_experts"]
                layer.moe.fp8 = enabled
                for projection in ("w13", "w2"):
                    value = sidecar.tensor(f"layers.{layer.layer}.ffn.experts.{projection}_fp8_channel_scale",
                                           device) if enabled else None
                    setattr(layer.moe, f"fp8_{projection}_scale", value)
            self.runtime_precision["fp8"] = {
                "config": config,
                "weight_fingerprint": sidecar.fingerprint,
                "route": "MXFP4 -> E4M3 SRAM -> FP8xFP8 MME -> FP32 scale -> BF16"
            }
        self.runtime_precision["router_selection"] = ("FP32 scores / native top6 / smallest-ID ties"
                                                      if gaudi_envs.VLLM_HPU_DSV41_ROUTER_TOP6 else "torch.topk")
        self.runtime_precision["head"] = "BF16xBF16 MME -> FP32" if self.bf16_head else "FP32 MME"
        self.runtime_precision["attention_input"] = (
            "fused wq_a+wkv BF16 MME / one activation quantization"
            if gaudi_envs.VLLM_HPU_DSV41_QKV_FUSED_INPUT else "separate wq_a/wkv")
        self.runtime_precision["mla"] = ("shared-KV BF16 QK / FP32 softmax and PV / BF16 output v1"
                                         if gaudi_envs.VLLM_HPU_DSV41_MLA_MME else "TPC online softmax")
        self.precision_fingerprint = canonical_hash(self.runtime_precision)
        self.loaded = True
        self.generation += 1

    def invalidate(self):
        # The execution owner must close its recipes before reloading or
        # migrating buffers. A live plan cannot retain addresses from this tree.
        if getattr(self, "replay_owner", None) is not None:
            self.replay_owner.close()
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
        from vllm_gaudi.ops.deepseek_v41_sampling import local_greedy_candidate, select_greedy_candidate
        if self.pp_rank != 1 or hidden.shape[0] != 1:
            raise ValueError("Ordinary greedy head requires one token on the final PP stage")
        local = self._head_projection(hidden)
        candidates = self.all_gather(local_greedy_candidate(local, self.tp_rank), dim=-1)
        return select_greedy_candidate(candidates)

    def sample_greedy_commit(self, hidden, record):
        selected = self.sample_greedy(hidden).reshape(1).to(torch.int32)
        updated = torch.cat((record[:1] + 1, torch.ones_like(record[1:3]), selected))
        record.copy_(updated)
        return record


class PreparedLayerGroup(nn.Module):
    """Bound FX dependency closure without changing the stage tensor program."""

    def __init__(self, stage, start, stop, *, fp8_decode=False):
        super().__init__()
        self.layers = nn.ModuleList(list(stage.layers[start:stop]))
        self.fp8_decode = fp8_decode
        self.final = stage.pp_rank == 1 and stop == len(stage.layers)
        self.norm = stage.weights.norm if self.final else None
        self.eps = stage.config["text_config"]["rms_norm_eps"]

    def forward(self, residual, pre_mix, positions, input_ids, engram_rows):
        image_mask = (input_ids == 129264) | (input_ids == 129265)
        target_states = []
        for layer in self.layers:
            rows = engram_rows[0 if layer.layer == 1 else 1] if layer.layer in (1, 14) else None
            residual, pre_mix, target = layer(residual,
                                              pre_mix,
                                              positions,
                                              image_mask,
                                              rows,
                                              fp8_decode=self.fp8_decode)
            if target is not None:
                target_states.append(target)
        if not self.final:
            return residual, pre_mix, None
        value = (residual.float() * pre_mix.unsqueeze(-1)).sum(1).to(residual.dtype)
        value = rms_norm(value, self.norm.weight, self.eps)
        return value, pre_mix, torch.cat(target_states, -1) if target_states else None

    def native_forward(self, residual, pre_mix, positions, input_ids, engram_rows):
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

    def __init__(self, stage, *, native=False):
        backend = "hpu_backend"
        if native and gaudi_envs.VLLM_HPU_DSV41_TP_MHC_OVERLAP:
            if stage.dspark or (stage.fp8_decode and not stage.expert_n256):
                raise ValueError("TP/mHC overlap requires BF16 boundaries and a qualified C1 expert layout")
            from vllm_gaudi.compilation.deepseek_v41_overlap import make_backend
            backend = make_backend()
        self.groups = tuple(
            PreparedLayerGroup(stage, start, start + 4, fp8_decode=native and stage.fp8_decode)
            for start in range(0, 20, 4))
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

    def insert_context(self, target_states, positions):
        first = self.weights.get_submodule("0")
        main_value = rms_norm(linear(target_states, first.main_proj), first.main_norm.weight, self.eps)
        return tuple(layer.attention.insert_context(main_value, positions) for layer in self.layers)

    def forward(self, next_token, positions):
        ids = torch.where(self.offsets == 0, next_token.reshape(1), self.noise)
        value = self.embed_input_ids(ids)
        residual = value.unsqueeze(1).expand(-1, 4, -1).contiguous()
        pre = F.one_hot(torch.zeros_like(ids, dtype=torch.int64), 4).float()
        image_mask = torch.zeros_like(ids, dtype=torch.bool)
        for layer in self.layers:
            residual, pre, _ = layer(residual, pre, positions, image_mask)
        hidden = (residual.float() * pre.unsqueeze(-1)).sum(1).to(value.dtype)
        return hidden, self.compute_logits(hidden)

    def embed_input_ids(self, input_ids):
        return self._embed(input_ids, self.weights.embed)

    def compute_logits(self, hidden):
        last = self.weights.get_submodule("2")
        normalized = rms_norm(hidden, last.norm.weight, self.eps)
        return self.all_gather(F.linear(normalized.float(), self.output_head.weight), dim=-1)

    def sample_greedy(self, first_token, hidden, logits):
        last = self.weights.get_submodule("2")
        previous = first_token.reshape(1)
        token_ids, scores = [], []
        for index in range(5):
            markov = self._embed(previous, last.markov_head.embed)
            bias = self.all_gather(F.linear(markov.float(), last.markov_head.head.weight), dim=-1)
            previous = torch.argmax(logits[index:index + 1] + bias, dim=-1)
            confidence_input = torch.cat((hidden[index:index + 1], markov), dim=-1).float()
            scores.append(linear(confidence_input, last.confidence_head.proj).reshape(1))
            token_ids.append(previous)
        return torch.cat(token_ids), torch.cat(scores)
