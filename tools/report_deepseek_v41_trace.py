# SPDX-License-Identifier: Apache-2.0
"""Attribute all recorded engines within common four-rank device periods.

Tensor and pre-graph dependency joins reuse the archived V4 reporting method.
Missing invocation boundaries remain unknown, especially for EDMA descriptors.
"""

import argparse
import bisect
import collections
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import json
from pathlib import Path
import statistics

from analyze_deepseek_v41_trace import symbols, union
from map_deepseek_v41_trace_contracts import graph_nodes, tensor
from deepseek_v41_trace_accounting import device_partition, host_accounting, merged


def io(node, prefix):
    return [
        tensor(v)
        for k, v in sorted(node["attrs"].items(),
                           key=lambda pair: (int(pair[0].split(":")[-1]) if pair[0].startswith(prefix) else -1))
        if k.startswith(prefix)
    ]


def classify(node, kernel, inputs, outputs):
    name = node.lower()
    if kernel == "unresolved_mme_recipe0":
        return "矩阵计算未归因", "SDK recipeId=0 的完整 MME 区间；源矩阵、精度与物理调用边界尚未恢复"
    if kernel.startswith("TPC_SPU_"):
        return "TPC 区间未归因", "TPC 硬件起止已配对；该 context 尚未关联到编译 GUID/张量"
    if "deepseek_v41_expert_n256_silu_quant" in kernel:
        return "路由专家", "W13 结果缩放/BF16 舍入、clamp/SwiGLU、路由加权与 W2 输入 FP8 量化"
    if "deepseek_v41_expert_n256_scale_reduce" in kernel:
        if inputs and inputs[0].get("dtype") in ("float32", "f32"):
            return "路由专家", "W2 结果缩放、逐路 BF16 舍入与六路有序归约，直接写最终 BF16 输出"
        return "路由专家", "六路已缩放 BF16 专家结果按路由顺序作 FP32 归约，再舍入 BF16"
    native_roles = {
        "deepseek_v41_compressor_pair": ("CSA2", "双帧压缩、门控合并及历史状态更新"),
        "deepseek_v41_final_collapse_norm": ("输出头", "最终四流 residual collapse 与 RMSNorm"),
        "deepseek_v41_fp4_pack_g16": ("量化", "主 KV 的 FP4 编码与 group16 E4M3 scale"),
        "deepseek_v41_fp4_pack_g32": ("量化", "FP4 编码与 group32 UE8M0 scale"),
        "deepseek_v41_fp4_roundtrip_g32": ("量化", "group32 FP4 舍入与 BF16 恢复"),
        "deepseek_v41_quant_roundtrip": ("量化", "模型规定的激活量化/恢复边界；所属投影见源节点"),
        "deepseek_v41_kv_norm_rope": ("Attention", "KV RMSNorm 与 RoPE 融合"),
        "deepseek_v41_qnorm_quant": ("Attention", "Q 输入 RMSNorm 与 FP8 激活量化"),
        "deepseek_v41_router_logits_top6": ("Router", "FP32 logits 的 softplus/sqrt、偏置选择及 top6 路由归一化"),
        "deepseek_v41_selected_mla_gather": ("Attention", "选中 MLA 行的 KV 读取与实际矩阵输入准备"),
        "deepseek_v41_paged_mla_gather": ("Attention", "按选中行直接解码 FP8/FP4 KV，生成 BF16 QK 输入、"
                                           "BF16 舍入后的 FP32 PV 输入和 mask"),
        "deepseek_v41_selected_mla_softmax": ("Attention", "选中 MLA 行的 softmax/sink 与概率准备"),
        "deepseek_v41_index_keys": ("CSA2", "物理 page 查找与 FP4 index key 解码，供多头 MME 复用"),
        "deepseek_v41_prefill_main_decode": ("CSA2", "prefill 主 KV page 查找与 FP4 解码，共享 BF16 工作区"),
        "deepseek_v41_prefill_topk": ("CSA2", "prefill 选择阈值/有序候选发射，具体阶段见 GUID"),
        "deepseek_v41_prefill_sparse_kv": ("Attention", "prefill 稀疏 KV 选择与矩阵操作数准备"),
        "deepseek_v41_prefill_sparse_exp": ("Attention", "prefill softmax/sink 统计与 BF16 指数权重"),
        "deepseek_v41_prefill_sparse_normalize": ("Attention", "prefill PV 结果归一化与 BF16 输出"),
        "deepseek_v41_prefill_flash_kv": ("Attention", "Flash prefill 的 KV 合并/解码准备"),
        "deepseek_v41_prefill_flash_mask": ("Attention", "Flash prefill 可见性与 causal mask 准备"),
        "deepseek_v41_dynamic_quant": ("量化", "动态激活 scale 归约与 FP8 量化"),
    }
    for guid, role in native_roles.items():
        if guid in kernel:
            return role
    if "deepseek_v41_prefill_bmm_f32" in name:
        return "Attention", "prefill QK/PV 分块矩阵计算；转置、实际形状与 dtype 见合同"
    if "deepseek_v4_mhc_" in kernel:
        return "mHC", "四流 gate/collapse/post 数据准备，具体阶段见 GUID"
    if "deepseek_v41_engram_hash_gather" in kernel or "device_engram_" in name:
        return "Engram", "设备端 token history/hash、TP shard 行读取及 BF16 staging 填充"
    if "_prefill_engram_tile" in name or "engram_update" in name:
        return "Engram", "分块 FP32 Q/K 归一化、门控与四流 residual 更新；最终 BF16 舍入"
    if "deepseek_v41_mhc_gates" in kernel:
        return "mHC", "mHC gate、Sinkhorn 输入及 residual mixing 系数准备"
    if "deepseek_v41_ffn_norm_quant" in kernel:
        return "MoE 准备", "FFN RMSNorm、动态 FP8 scale 与激活量化融合"
    if any(part in kernel for part in ("deepseek_v41_index_scores", "deepseek_v41_index_threshold",
                                       "deepseek_v41_index_emit")):
        return "CSA2", "CSA2 Reindex 候选打分、阈值选择及紧凑候选槽输出"
    if "deepseek_v41_q_scale_rope" in kernel:
        return "Attention", "wq_b FP32 结果缩放、BF16 舍入及 Q RoPE 融合"
    if any(part in name for part in ("deepseek_v41_q_projection_rope", "prefill_q_projection",
                                       "q_norm_projection_rope")):
        return "Attention", "wq_b 原生 FP8 投影与 RoPE；实际操作数见合同"
    if "deepseek_v41_mla_shared_kv" in kernel:
        return "Attention", "MLA 候选 KV 一次 BF16 准备，共用于 QK/PV"
    if "deepseek_v41_mla_exp_bf16" in kernel:
        return "Attention", "MLA FP32 softmax/sink 统计、BF16 指数权重及 FP32 分母"
    if "deepseek_v41_mla_normalize_bf16" in kernel:
        return "Attention", "MLA FP32 PV 结果归一化、BF16 输出"
    if any(part in name for part in ("deepseek_v41_selected_mla_mme", "deepseek_v41_paged_mla_mme",
                                     "deepseek_v41_paged_mla_direct_mme")):
        if kernel.lower().replace("_", "") in ("gemm", "batchgemm") and inputs:
            # hpu_dsv41_selected_mla_pt2.cpp builds BF16 QK, then an
            # FP32 probability x FP32 V product. PostGraph may fuse the final
            # BF16 cast into that product; report its actual output contract.
            role = "PV" if inputs[0]["dtype"] in ("float32", "f32") else "QK"
            return "Attention", "选中/分页 MLA " + role + " 矩阵计算；实际 A/B/输出精度见合同"
        return "Attention", "选中/分页 MLA 数据准备与最终 BF16 舍入"
    if "deepseek_v41_mla_bf16_pv" in name:
        if kernel in ("GEMM", "BatchGemm", "gemm", "batch_gemm"):
            output = outputs[0].get("shape", []) if outputs else []
            ambiguous = inputs and inputs[0].get("shape", [])[-1:] == [512] and output[-1:] == [512]
            role = ("QK/PV 尚待转置描述符关联" if ambiguous else
                    "PV" if output and output[-1] == 512 else "QK" if output else "QK/PV 尚待形状关联")
            return "Attention", "MLA " + role + " 矩阵计算；实际操作数见合同"
        return "Attention", "MLA 内部准备/别名/搬运"
    if "deepseek_v41_attention_norm" in kernel:
        width = inputs[0]["shape"][-1] if inputs else None
        projection = "Q" if width == 1280 else "KV" if width == 512 else "Q/KV"
        return "Attention", projection + " RMSNorm：FP32 行归约与权重乘法，BF16 输出"
    if "deepseek_v41_mla_gather" in kernel:
        return "Attention", "MLA 候选 KV 共享读取、BF16 K/FP32 V 准备及有效 mask"
    if "deepseek_v41_mla_softmax" in kernel:
        return "Attention", "MLA FP32 scale/softmax/sink，概率保持 FP32"
    if "deepseek_v41_mla_mme" in name:
        if kernel in ("GEMM", "BatchGemm", "gemm", "batch_gemm"):
            if not inputs:
                return "Attention", "MLA 矩阵计算，操作数尚未关联"
            return "Attention", ("MLA QK：BF16×BF16，FP32 结果"
                                 if inputs[0]["dtype"] == "bf16" else "MLA PV：FP32 概率×FP32 V，FP32 累加")
        return "Attention", "MLA 内部转换、矩阵输入准备及数据搬运"
    if "deepseek_v41_router_top6" in kernel:
        return "Router", "text/image bias 选择、六次最大值选择、原分数归一化"
    if "deepseek_v41_woa_stage" in kernel:
        return "Attention", "wo_a FP8 权重 HBM→SRAM 分块搬运"
    if "deepseek_v41_woa_quant" in kernel:
        return "Attention", "wo_a 分组激活最大值、二次幂 scale、FP8 量化"
    if "deepseek_v41_woa_scale" in kernel:
        return "Attention", "wo_a FP32 结果乘通道/激活 scale 并转 BF16"
    if "deepseek_v41_woa_fp8" in name and kernel in ("GEMM", "BatchGemm", "gemm", "batch_gemm"):
        return "Attention", "wo_a 分组输出 BMM"
    if "deepseek_v41_dense_quant" in kernel:
        k = inputs[0]["shape"][-1] if inputs else None
        if k == 6144:
            return "Engram", "wkv 激活最大值、二次幂 scale 与 FP8 量化"
        projection = "wq_b" if k == 1280 else "wo_b" if k in (2048, 4096) else "未关联投影"
        return "Attention", projection + " 全 K 激活最大值、二次幂 scale 与 FP8 量化"
    if "deepseek_v41_dense_scale" in kernel:
        n = inputs[0]["shape"][-1] if inputs else None
        if n == 25600:
            return "Engram", "wkv FP32 结果乘激活/通道 scale 并舍入 BF16"
        projection = "wq_b" if n in (8192, 16384) else "wo_b" if n == 5120 else "未关联投影"
        return "Attention", projection + " FP32 矩阵结果乘两类 scale、BF16 输出"
    if "deepseek_v41_dense_fp8" in name:
        weight = inputs[1]["shape"] if len(inputs) > 1 and kernel in ("GEMM", "BatchGemm", "gemm", "batch_gemm") else []
        if weight == [25600, 6144]:
            return "Engram", "查询行到 key/value 的 FP8 wkv 投影，实际操作数见合同"
        projection = ("wq_b" if weight in ([16384, 1280], [8192, 1280]) else
                      "wo_b" if weight in ([5120, 4096], [5120, 2048]) else "Attention dense FP8")
        return "Attention", projection + (" 原生矩阵计算，实际操作数见合同" if weight else " 内部准备/转换/搬运")
    if "deepseek_v41_bf16_linear_f32" in name and kernel in ("GEMM", "BatchGemm", "gemm", "batch_gemm"):
        if "router" in name or "/moe/" in name or (len(inputs) > 1 and inputs[1]["shape"] == [384, 5120]):
            return "Router", "BF16×BF16、FP32 输出的 384 专家路由打分"
        if len(inputs) > 1 and inputs[1]["shape"] in ([64640, 5120], [32320, 5120]):
            return "输出头", "BF16×BF16 TP 词表投影，FP32 logits"
    if ("topk" in name or "bitonic" in kernel) and ("router" in name or "router" in kernel):
        return "Router", "专家 Top-k 排序；实际输入见张量合同"
    if "deepseek_v41_control_gemv" in kernel:
        return "mHC", "FP32 控制投影"
    if "deepseek_v41_rope" in kernel or "deepseek_v41_c1_indices" in kernel:
        return "CSA2", "C1 RoPE/可见索引准备"
    if "deepseek_v41_swa_decoded_write" in kernel:
        return "CSA2", "SWA 编码/scale/packed 写入及增量 BF16 KV 副本写入"
    if "deepseek_v41_fp4_decoded_write" in kernel:
        return "CSA2", "FP4 主 KV/index 编码写入及增量 BF16 KV 副本写入"
    if "deepseek_v41_swa_pack" in kernel:
        return "CSA2", "SWA 精确 FP8 编码/scale/缓存写入"
    if "deepseek_v41_fp4_cache_write" in kernel:
        return "CSA2", "主 KV/index 精确 FP4 编码/scale/缓存写入"
    if "deepseek_v41_selected_kv" in kernel:
        return "CSA2", "按候选槽解码实际读取的 packed KV，含写入完成依赖"
    if "mxfp4" in name or "mxfp4" in kernel or "expert_n256" in name or "expert_n256" in kernel:
        if not kernel:
            return "路由专家", "专家内部张量准备/广播；实际操作来源见节点与张量合同"
        shape = inputs[1]["shape"] if kernel in ("GEMM", "BatchGemm", "gemm", "batch_gemm") and len(inputs) > 1 else []
        stage = ("W13 gate/up" if shape and any(width in shape for width in (2304, 1280)) else
                 "W2 down" if shape and any(width in shape for width in (1152, 640)) else "阶段见张量合同")
        if not shape and outputs:
            if outputs[0]["shape"][-2:] in ([5120, 2304], [5120, 1280]):
                stage = "W13 gate/up"
            elif outputs[0]["shape"][-2:] in ([1152, 5120], [640, 5120]):
                stage = "W2 down"
        return "路由专家", stage + ("矩阵计算" if shape else "压缩权重直接寻址/解码")
    if "deepseek_v41_prefill_permuted_bf16" in kernel:
        shape = inputs[1]["shape"] if len(inputs) > 1 else []
        role = "W13" if shape and shape[-1] == 327680 else "W2" if shape and shape[-1] in (40960, 73728) else "W13/W2"
        return "路由专家", role + " 压缩 N256 权重寻址、解码到 BF16 与列布局转换"
    if "bf16_identity" in kernel:
        return "数值边界", "显式 BF16 舍入/编译块边界；具体所属模块见源节点"
    if "prefill_index" in name or "reindex" in name or "candidate_gather" in kernel:
        return "CSA2", "候选 KV 打分/加权 head 归约/筛选，见张量合同"
    if "prefill_mhc" in kernel:
        return "mHC", "四流 residual collapse/post update，保留 BF16 输出边界"
    if kernel in ("GEMM", "BatchGemm", "gemm", "batch_gemm"):
        weight = inputs[1]["shape"] if len(inputs) > 1 else []
        if len(weight) == 3 and weight[-2:] in ([5120, 1280], [5120, 2304]):
            return "路由专家", "W13 gate/up 分组 BMM"
        if len(weight) == 3 and weight[-2:] in ([640, 5120], [1152, 5120]):
            return "路由专家", "W2 down 分组 BMM"
        if weight == [24, 20480]:
            return "mHC", "FP32 控制投影"
        if weight == [384, 5120]:
            return "Router", "384 专家路由打分"
        if weight == [25600, 6144]:
            return "Engram", "查询行到 key/value 投影"
        unique_projections = {
            (1280, 5120): "wq_a 输入投影", (1792, 5120): "wq_a/wkv 合并输入投影",
            (16384, 1280): "wq_b Q 展开", (8192, 1280): "wq_b Q 展开",
            (512, 5120): "wkv 输入投影", (5120, 4096): "wo_b 输出投影",
            (5120, 2048): "wo_b 输出投影",
        }
        if tuple(weight) in unique_projections:
            return "Attention", unique_projections[tuple(weight)]
        if "/attention/" in name:
            if weight in ([1024, 1280], [2048, 1280]):
                return "CSA2", "index Q 展开投影：TP4 8 heads 或 TP2 16 heads，每 head 128 维"
            if weight in ([8, 5120], [16, 5120]):
                return "CSA2", "index head 权重投影；后续缩放并四路 gather"
            if ("/mm_" in name and len(inputs) == 2 and inputs[0]["shape"] == [32, 128]
                    and len(weight) == 2 and weight[-1] == 128):
                return "CSA2", "32-head index QK 候选打分的实际编译矩阵分块"
            if inputs and inputs[0]["dtype"] == "float32":
                return "Attention", "Compressor FP32 投影；wkv/wgate 绑定尚待逐节点还原"
            shapes = {
                (1280, 5120): "wq_a 输入投影",
                (1792, 5120): "wq_a/wkv 合并输入投影",
                (16384, 1280): "wq_b Q 展开",
                (8192, 1280): "wq_b Q 展开",
                (512, 5120): "wkv 输入投影",
                (5120, 4096): "wo_b 输出投影",
                (5120, 2048): "wo_b 输出投影",
                (128, 512): "index K 投影",
                (1024, 4096): "wo_a 分组输出 GEMM"
            }
            if "/bmm" in name:
                return "Attention", "wo_a 分组输出 BMM"
            return "Attention", shapes.get(tuple(weight), "Compressor/其他投影，见节点合同")
        if "/moe/" in name:
            return "共享专家", "down 投影" if weight in ([5120, 1152], [5120, 576]) else "gate/up 投影，见源节点"
        if weight and weight[0] in (64640, 32320):
            return "输出头", "TP 词表投影"
        return "矩阵计算待细分", "源节点及完整操作数已保留"
    if "sparse_attn" in kernel or "decoded_attn" in kernel:
        return "Attention", "顺序 QK/online softmax/V 累加"
    if "sinkhorn" in kernel:
        return "mHC", "4×4 Sinkhorn"
    if kernel == "DmaMemcpy" and inputs and inputs[0]["shape"] == [512, 256, 1]:
        return "CSA2", "FP4 全缓存解包中的字节重排"
    if "/attention/" in name:
        return "Attention/CSA2", "张量准备/量化/状态更新，见源节点"
    if "/moe/" in name:
        return "MoE 准备", "激活量化/激活/路由归约，见源节点"
    if kernel.startswith("fused_kernel_"):
        return "融合表达式待细分", "保留实际 GUID 与输入输出；内部子算子无独立计时"
    return "入口/尾部/其他张量", node


def bundle_key(contract):
    if not contract:
        return None
    bundle = contract.get("attributes", {}).get("Bundle_idx")
    if bundle in (None, "N/A"):
        return None
    return contract["graph"]["path"], str(bundle)


def attention_norm_owners(nodes):
    """Prove fused RMSNorm ownership by tracing its inputs to one Q/KV GEMM.

    A width heuristic alone is insufficient: all nonconstant leaves must reach
    that same named Attention projection. Unknown or mixed dependencies stay
    unresolved. The returned node set includes casts and mean reductions.
    """
    ins = [io(n, "inputTensor:") for n in nodes]
    outs = [io(n, "outputTensor:") for n in nodes]
    producers = collections.defaultdict(list)
    for i, tensors in enumerate(outs):
        for t in tensors:
            producers[t["name"]].append(i)
    owners = {}
    for final, node in enumerate(nodes):
        if not node["op"].startswith("fused_kernel_") or len(ins[final]) != 3 or len(outs[final]) != 1:
            continue
        shape = outs[final][0]["shape"]
        if shape not in ([1, 1280], [1, 512]) or outs[final][0]["dtype"] != "bf16":
            continue
        width = shape[-1]
        if sorted(t["shape"] for t in ins[final]) != sorted([[1, 1], [1, width], [width]]):
            continue
        visited, anchors = set(), set()

        def visit(i, visited=visited, width=width, anchors=anchors):
            if i in visited:
                return True
            if len(visited) >= 32:
                return False
            n = nodes[i]
            if n["op"] in ("GEMM", "BatchGemm", "gemm", "batch_gemm"):
                if "/attention/" not in n["name"] or len(ins[i]) != 2 or ins[i][1]["shape"] != [width, 5120]:
                    return False
                anchors.add(n["name"])
                return True
            if not ("/attention/" in n["name"] or n["op"] in ("Reduction", "DmaMemset")
                    or n["op"].startswith(("fused_kernel_", "cast_", "reshape"))):
                return False
            visited.add(i)
            for t in ins[i]:
                if t["shape"] == [width]:  # The final RMSNorm channel weight.
                    continue
                source = producers.get(t["name"], []) or producers.get(t.get("alias"), [])
                if not source:
                    if t["shape"] in ([1], [1, 1]) and t["name"].startswith(("f32-", "i32-")):
                        continue
                    return False
                if len(source) != 1 or not visit(source[0]):
                    return False
            return True

        if visit(final) and len(anchors) == 1:
            role = "Q" if width == 1280 else "KV"
            proof = {"rule": "RMSNorm dependency closure to one Attention input GEMM",
                     "role": role, "anchor": next(iter(anchors)), "final": node["name"]}
            for i in visited:
                owners[nodes[i]["name"]] = proof
    return owners



def expert_mme_owners(contracts):
    """Prove physical MME tile roles from their captured weight producer."""
    producers = collections.defaultdict(list)
    for contract in contracts.values():
        for output in contract.get("outputs", []):
            producers[(contract["graph"]["path"], output["name"])].append(contract)
    owners = {}
    for key, contract in contracts.items():
        if contract["symbol"]["device_type"] != 0 or len(contract.get("inputs", [])) < 2:
            continue
        visited, anchors = set(), {}

        def visit(tensor, visited=visited, contract=contract, anchors=anchors):
            for name in (tensor["name"], tensor.get("alias")):
                if not name or name in visited:
                    continue
                visited.add(name)
                for source in producers.get((contract["graph"]["path"], name), []):
                    if "deepseek_v41_prefill_permuted_bf16" in source["symbol"]["kernel"]:
                        shape = source["inputs"][1]["shape"]
                        role = "W13" if shape[-1] == 327680 else "W2" if shape[-1] in (40960, 73728) else None
                        if role:
                            anchors[source["symbol"]["node"]] = dict(role=role, packed_weight_shape=shape)
                    elif len(visited) < 80:
                        for value in source.get("inputs", []):
                            visit(value)
        visit(contract["inputs"][1])
        roles = {row["role"] for row in anchors.values()}
        if len(roles) == 1:
            owners[key] = dict(rule="same-capture PostGraph dependency from MME B to N256 weight decoder",
                              role=roles.pop(), producers=anchors)
    return owners


def invocation_samples(rows, symbol, expected):
    rows = sorted(rows)
    if symbol["device_type"] == 1:
        packets = sum(symbol.get("working_engines", []))
    elif symbol["device_type"] == 0:
        if expected is not None:
            # A compiler MME node can emit several ROIs on the same lane.
            # Static recipe launches establish the node invocation count;
            # unique lane count alone would split one invocation into many.
            return eager_invocation_samples(rows, expected)
        packets = len({row[2] for row in rows})
    else:
        return [], None, "EDMA descriptor-to-invocation boundary not reconstructed"
    if not packets or len(rows) % packets:
        return [], None, "missing packet count or incomplete lane packet group"
    groups = [rows[i:i + packets] for i in range(0, len(rows), packets)]
    if expected is not None and len(groups) != expected:
        return [], None, "packet count differs from captured recipe frequency"
    if symbol["device_type"] == 0 and any(len({row[2] for row in group}) != packets for group in groups):
        return [], None, "MME lane ownership differs within a reconstructed invocation"
    # A stage's successive executions of one context are dependency ordered.
    if any(max(r[1] for r in a) > min(r[0] for r in b) for a, b in zip(groups, groups[1:])):
        return [], None, "ambiguous overlapping groups for the same recipe context"
    return [(max(r[1] for r in g) - min(r[0] for r in g)) / 1000 for g in groups], len(groups), None


def eager_invocation_samples(rows, launches):
    """Use exact static recipe-handle enqueues, never lane count as call count.

    A final compiler node executes once per static recipe enqueue. Validate
    equal packet ownership and disjoint executions before assigning envelopes.
    Multi-ROI nodes may emit several packets per lane in one invocation.
    """
    if not launches or len(rows) % launches:
        return [], None, "no exact enqueue count or unequal packets per static invocation"
    ordered = sorted(rows)
    width = len(ordered) // launches
    groups = [ordered[i:i + width] for i in range(0, len(ordered), width)]
    lanes = [collections.Counter(row[2] for row in group) for group in groups]
    if any(value != lanes[0] for value in lanes):
        return [], None, "eager static invocation lane ownership changed"
    if any(max(row[1] for row in a) > min(row[0] for row in b) for a, b in zip(groups, groups[1:])):
        return [], None, "eager static invocation envelopes overlap ambiguously"
    return [(max(row[1] for row in group) - min(row[0] for row in group)) / 1000
            for group in groups], launches, None


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({
            k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
            for k, v in row.items()
        } for row in rows)


def analyze(root, rank, common):
    path = root / f"rank{rank}"
    inv = json.loads((path / "inventory.json").read_text())
    recipes = json.loads((path / "recipe-symbols.json").read_text())["recipes"]
    mapped = symbols(inv, recipes)
    contracts = {
        (str(row["recipe_id"]), row["symbol"]["device_type"], row["symbol"]["full_context_id"]): row
        for row in json.loads((path / "node-contracts.json").read_text())
    }
    expert_owners = expert_mme_owners(contracts)
    activity_groups = collections.defaultdict(list)
    norm_owners = {path: attention_norm_owners(graph_nodes(Path(path)))
                   for path in {c["graph"]["path"] for c in contracts.values() if c.get("matched")}
                   if Path(path).suffix == ".pbtxt"}
    bundle_categories = collections.defaultdict(set)
    for contract in contracts.values():
        symbol = contract["symbol"]
        category, _ = classify(symbol["node"], symbol["kernel"], contract.get("inputs", []),
                               contract.get("outputs", []))
        key = bundle_key(contract)
        if key and category in ("路由专家", "Attention", "共享专家", "Engram", "mHC", "Router"):
            bundle_categories[key].add(category)
    graph_files = collections.defaultdict(list)
    for record in json.loads((root / "graph-manifest.json").read_text()):
        if f"/rank{rank}/" in record["path"]:
            graph_files[Path(record["path"]).name].append(record)
    own = json.loads((path / "device-windows.json").read_text())
    mhc_recipes = set()
    plan_entries, plan_sources = [], []
    plan_root = root.parent / f"plans/rank{rank}"
    for group in range(5):
        files = list(plan_root.glob(f"group{group}-*-prepare1.json"))
        if len(files) != 1:
            plan_entries = []
            break
        plan_sources.append(str(files[0]))
        plan_entries.extend(row for row in json.loads(files[0].read_text()) if row.get("kind") == "compute")
    provenance = []
    if any(entry.get("name", "").endswith("_mhc_submod_0") for entry in plan_entries):
        if len(plan_entries) != len(own["capture_order"]):
            raise RuntimeError("Prepared compute order differs from captured native segments")
        by_id = {str(recipe["recipe_id"]): recipe for recipe in recipes}
        for entry, enqueue in zip(plan_entries, own["capture_order"], strict=True):
            if not entry["name"].endswith("_mhc_submod_0"):
                continue
            # Bridge's program ID and Synapse's serialized recipe ID are
            # different namespaces. The exact captured segment order joins them.
            rid = enqueue[2].split(":")[0]
            if rid not in by_id or not any("control_gemv" in n["kernel"] for n in by_id[rid]["nodes"]):
                raise RuntimeError("mHC capture order differs from actual compiler symbols")
            mhc_recipes.add(rid)
            provenance.append(dict(program=entry, synapse_recipe=rid, enqueue=enqueue))
    (path / "mhc-partition-provenance.json"
     ).write_text(json.dumps(dict(plans=plan_sources, bindings=provenance), indent=2) + "\n")
    frequency = collections.Counter(row[2].split(":")[0] for row in own["capture_order"])
    windows, tokens = common["windows_us"], common["tokens"]
    ends = [w[1] for w in windows]
    launch_counts = collections.Counter()
    is_tp4 = own.get("topology", {}).get("tensor_parallel_size") == 4
    if is_tp4:
        for start, length, recipe, _ in inv.get("device_recipe_starts", inv["host_enqueues"]):
            win = bisect.bisect_right(ends, start)
            if win < len(windows) and windows[win][0] <= start and start + length <= windows[win][1]:
                launch_counts[(recipe.split(":")[0], win)] += 1
    period = sum(b - a for a, b in windows)
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    raw, engines = collections.defaultdict(list), collections.defaultdict(list)
    packet_counts = collections.Counter()
    clipped_calls = set()
    first_index = {}
    with gzip.open(path / "hardware.jsonl.gz", "rt") as stream:
        for line in stream:
            record = json.loads(line)
            start, duration, lane, index = record[:4]
            api = record[5] if len(record) > 5 else ""
            win = bisect.bisect_right(ends, start)
            if win >= len(windows):
                continue
            node, symbol = inv["nodes"][index], mapped.get(index)
            key = ((node["recipe"].split(":")[0], symbol["device_type"], symbol["full_context_id"]) if symbol else
                   (node["recipe"], -1, index))
            kernel = symbol["kernel"] if symbol else node["kernel"]
            counted = False
            while win < len(windows) and windows[win][0] < start + duration:
                lo, hi = windows[win]
                stop, begin = min(start + duration, hi), max(start, lo)
                if begin < stop:
                    first_index[key] = index
                    groups[key][win].append((begin, stop, lane, api))
                    if begin != start or stop != start + duration:
                        clipped_calls.add((key, win))
                    raw[(node["engine"], kernel)].append((begin, stop))
                    engines[node["engine"]].append((begin, stop))
                    counted = True
                win += 1
            packet_counts[key] += int(counted)
    details, aggregated, recovered = [], collections.defaultdict(list), collections.defaultdict(list)
    for key, by_window in groups.items():
        index = first_index[key]
        node, symbol = inv["nodes"][index], mapped.get(index)
        contract = contracts.get(key)
        if contract is None or not contract.get("matched"):
            # Ordinary eager graphs have short-lived recipe handles and may not
            # be serialized. Join their exact enqueue identity to saved graphs.
            launches = [r for r in inv["host_enqueues"] if r[2] == node["recipe"]]
            options = []
            for launch in launches:
                for suffix in ("-eager_final_graph-symbol.pbtxt", "-eager-finalgraph-symbol.pbtxt",
                               "-PostGraph-symbol.pbtxt"):
                    basename = Path(launch[3]).name + suffix
                    records = graph_files.get(basename, [])
                    for record in records:
                        graph = Path(record["path"])
                        options.extend((record, n) for n in graph_nodes(graph)
                                       if n["name"] == node["node"] and
                                       n["op"].lower().replace("_", "") == node["kernel"].lower().replace("_", ""))
            unique = {record["path"]: (record, match) for record, match in options}
            if len(unique) == 1:
                record, match = next(iter(unique.values()))
                contract = {"matched": True, "inputs": io(match, "inputTensor:"),
                            "outputs": io(match, "outputTensor:"), "graph": record, "attributes": match["attrs"],
                            "provenance": "exact recipeHandle enqueue name -> rank-owned final compiler graph"}
        kernel = symbol["kernel"] if symbol else node["kernel"]
        source = symbol["node"] if symbol else node["node"]
        inputs, outputs = (contract.get(part, []) if contract else [] for part in ("inputs", "outputs"))
        category, purpose = classify(source, kernel, inputs, outputs)
        if kernel == "null":
            category, purpose = "设备调度", "TPC null 描述符；无模型张量计算，保留硬件活动区间"
        elif node["engine"] == "NIC":
            category, purpose = "通信", "NIC 硬件活动；不能将 lane 事件当作 HCCL 逻辑调用"
        elif node["engine"] == "DMA" and not contract:
            category, purpose = "搬运", "DMA 描述符活动；源/目标张量未与编译图关联"
        origin = expert_owners.get(key)
        if origin:
            category = "路由专家"
            purpose = ("W13 gate/up" if origin["role"] == "W13" else "W2 down") + " 预填充物理 MME 分块"
        norm = norm_owners.get((contract.get("graph") or {}).get("path"), {}).get(source) if contract else None
        if norm:
            category, purpose = "Attention", norm["role"] + " RMSNorm：投影后的转换、统计、归一化及权重乘法"
            origin = norm
        if category == "融合表达式待细分" and bundle_categories.get(bundle_key(contract)) == {"路由专家"}:
            category = "路由专家"
            purpose = "同一编译 bundle 的专家激活/结果处理；内部表达式未独立计时"
            origin = {"rule": "unambiguous expert bundle", "bundle": bundle_key(contract)}
        if key[0] in mhc_recipes:
            category, purpose = "mHC", "已由原生计划和编译分区核验的独立控制/统计/混合分支"
        spans = [row[:2] for rows in by_window.values() for row in rows]
        if common.get("export_activity_intervals"):
            for rows in by_window.values():
                for begin, end, lane, _api in rows:
                    group = category
                    if node["engine"] == "DMA":
                        group = ("通信相关DMA" if "HCL dedicated" in lane else
                                 "命令搬运" if "pdma_tx_commands" in kernel else "其他DMA")
                    activity_groups[(node["engine"], group)].append((begin, end))
        samples, count, missing, methods = [], 0, [], set()
        for win, rows in by_window.items():
            if (key, win) in clipped_calls:
                values, n, reason = [], None, "invocation crosses the accounting window; activity retained"
            elif symbol:
                expected = frequency.get(key[0])
                if is_tp4:
                    expected = launch_counts[(key[0], win)]
                    methods.add("exact device recipe-start markers within this TP4 window"
                                if "device_recipe_starts" in inv else "exact static recipe enqueue count")
                values, n, reason = invocation_samples(rows, symbol, expected)
                methods.add("serialized working-engine/ROI contract and ordered lane packets")
            elif contract and contract.get("matched") and node["engine"] in ("TPC", "MME"):
                low, high = windows[win]
                launches = sum(low <= launch[0] and launch[0] + launch[1] <= high
                               for launch in inv["host_enqueues"] if launch[2] == node["recipe"])
                values, n, reason = eager_invocation_samples(rows, launches)
                methods.add("exact recipe-handle enqueues and final static node; validated packet ownership")
            else:
                values, n, reason = [], None, "unserialized eager node has no working-engine contract"
            samples.extend(values)
            if n is not None:
                count += n
            if reason:
                missing.append(reason)
        duration = union(spans)
        row = {
            "rank": rank,
            "recipe_id": key[0],
            "context_id": key[2],
            "engine": node["engine"],
            "category": category,
            "purpose": purpose,
            "kernel": kernel,
            "source_node": source,
            "mean_invocation_ms": statistics.mean(samples) if samples else None,
            "complete_invocation_samples": len(samples),
            "invocation_samples_ms": samples,
            "invocation_count_methods": sorted(methods),
            "observed_calls": count if not missing else None,
            "calls_per_" + common.get("unit", "token"): count / len(tokens) if not missing else None,
            "observed_lane_packets": packet_counts[key],
            "count_limitations": sorted(set(missing)),
            "activity_ms_per_" + common.get("unit", "token"): duration / len(tokens) / 1000,
            "period_pct": duration / period * 100,
            "inputs": inputs,
            "outputs": outputs,
            "compiler_contract": contract
        }
        row["classification_provenance"] = origin
        details.append(row)
        dtype_shape = json.dumps([[(v["dtype"], v["shape"]) for v in part] for part in (inputs, outputs)])
        group_key = (category, purpose, node["engine"], kernel, dtype_shape)
        aggregated[group_key].append((row, spans, samples))
        recovered[(node["engine"], kernel)].extend(spans)
    for key in raw:
        assert abs(union(raw[key]) - union(recovered[key])) < 1e-6, key
    summary = []
    for key, items in aggregated.items():
        spans = [s for _, group, _ in items for s in group]
        samples = [v for _, _, group in items for v in group]
        unknown = any(row["observed_calls"] is None for row, _, _ in items)
        count = None if unknown else sum(row["observed_calls"] for row, _, _ in items)
        duration = union(spans)
        summary.append({
            "rank": rank,
            "category": key[0],
            "purpose": key[1],
            "engine": key[2],
            "kernel": key[3],
            "dtype_shapes": json.loads(key[4]),
            "mean_invocation_ms": statistics.mean(samples) if samples else None,
            "complete_invocation_samples": len(samples),
            "observed_calls": count,
            "calls_per_" + common.get("unit", "token"): count / len(tokens) if count is not None else None,
            "activity_ms_per_" + common.get("unit", "token"): duration / len(tokens) / 1000,
            "period_pct": duration / period * 100,
            "node_keys": [[r["recipe_id"], r["engine"], r["context_id"]] for r, _, _ in items]
        })
    scale = len(tokens) * 1000
    tpc, mme = union(engines["TPC"]), union(engines["MME"])
    compute = union(engines["TPC"] + engines["MME"])
    all_device = union([span for spans in engines.values() for span in spans])
    partition = {
        "TPC_only_ms": (compute - mme) / scale,
        "MME_only_ms": (compute - tpc) / scale,
        "TPC_MME_overlap_ms": (tpc + mme - compute) / scale,
        "other_recorded_device_only_ms": (all_device - compute) / scale,
        "unattributed_or_other_stage_ms": (period - all_device) / scale
    }
    assert abs(sum(partition.values()) - period / scale) < 1e-8
    details.sort(key=lambda r: -r["activity_ms_per_" + common.get("unit", "token")])
    summary.sort(key=lambda r: -r["activity_ms_per_" + common.get("unit", "token")])
    write_csv(path / "kernel-breakdown.csv", summary)
    (path / "node-breakdown.json").write_text(json.dumps(details, ensure_ascii=False, indent=2) + "\n")
    result = {
        "rank": rank,
        "unit": common.get("unit", "token"),
        "phase": common.get("phase", "decode"),
        "tokens": tokens,
        "period_ms": period / scale,
        "partition": partition,
        "device_engine_presence_partition_ms": device_partition(engines, windows, scale),
        "host_observed_activity": host_accounting(path, windows, engines["TPC"] + engines["MME"], scale,
                                                   export_intervals=common.get("export_activity_intervals", False)),
        "trace_sha256": inv["trace_sha256"],
        "device_observability": {engine: ("positive intervals captured" if engines[engine] else
                                           "no intervals exposed; zero hardware time is not established")
                                 for engine in ("TPC", "MME", "DMA", "NIC")},
        "measured_nodes": len(details),
        "unmatched_nodes": sum(not (r["compiler_contract"] or {}).get("matched") for r in details),
        "nodes_with_unknown_calls": sum(r["observed_calls"] is None for r in details),
        "status": "all activity retained; unknown invocation boundaries and fused origins remain explicit",
        "kernel_rows": summary
    }
    (path / "kernel-breakdown.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if common.get("export_activity_intervals"):
        activity = dict(rank=rank, base_time_nanoseconds=inv["base_time_nanoseconds"],
                        windows_us=windows, tokens=tokens, trace_sha256=inv["trace_sha256"],
                        groups=[dict(engine=engine, category=category, intervals_us=merged(spans))
                                for (engine, category), spans in activity_groups.items()],
                        host_markers=inv["cpu_markers"])
        with gzip.open(path / "activity-intervals.json.gz", "wt", compresslevel=1) as stream:
            json.dump(activity, stream)
    print(json.dumps({k: v for k, v in result.items() if k != "kernel_rows"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--ranks", nargs="+", type=int, choices=range(4), default=list(range(4)))
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--activity-intervals", action="store_true",
                        help="Export merged module intervals for aligned four-rank latency accounting")
    parser.add_argument("--tokens", nargs="+", type=int,
                        help="Explicit actual cycles for a labeled diagnostic example; never rescale their time")
    args = parser.parse_args()
    # The same global trace clock and PP1-final-MoE boundary is used for all
    # ranks. Keep token IDs present in every rank's complete stage sequence.
    data = [json.loads((args.analysis / f"rank{r}/device-windows.json").read_text()) for r in range(4)]
    tokens = sorted(set.intersection(*(set(d["tokens"]) for d in data)))
    if args.tokens is not None:
        if not set(args.tokens) <= set(tokens):
            raise ValueError("Requested example lacks a complete common four-rank cycle")
        tokens = sorted(set(args.tokens))
    is_tp4 = data[0].get("topology", {}).get("tensor_parallel_size") == 4
    anchor_rank = 0 if is_tp4 else 2
    anchor = dict(zip(data[anchor_rank]["tokens"], data[anchor_rank]["windows_us"]))
    common = {
        "tokens": tokens,
        "windows_us": [anchor[t] for t in tokens],
        "unit": data[0].get("unit", "token"),
        "phase": data[0].get("phase", "decode"),
        "anchor": data[0].get("boundary", "rank2 full decoder replay cadence"),
        "anchor_rank": anchor_rank,
        "export_activity_intervals": args.activity_intervals,
    }
    if not tokens:
        raise ValueError("No complete common four-rank cycle")
    (args.analysis / "common-windows.json").write_text(json.dumps(common, indent=2) + "\n")
    jobs = []
    for rank in args.ranks:
        own = dict(common)
        if is_tp4:
            bases = [item["base_time_nanoseconds"] for item in data]
            if any(base is None for base in bases):
                raise ValueError("TP4 rank alignment needs each Kineto clock base")
            shift = (bases[anchor_rank] - bases[rank]) / 1000
            own["windows_us"] = [(begin + shift, end + shift) for begin, end in common["windows_us"]]
        jobs.append((args.analysis, rank, own))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(analyze, *job) for job in jobs]
        for future in futures:
            future.result()
