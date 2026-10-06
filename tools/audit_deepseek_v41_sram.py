# SPDX-License-Identifier: Apache-2.0
"""Inspect compiled Bridge symbol graphs; standalone decoder outputs are excluded."""

import argparse
import hashlib
import json
from pathlib import Path
import re


def tensor_info(text):
    shape = re.search(r"Sizes = (\[[^\]]+\])", text)
    size = re.search(r"sizeInBytes = (\d+)", text)
    location = re.search(r"location = in (\w+)", text)
    alias = re.search(r"isAliased = ([^,]+), type = alias, offset: 0(?: |$)", text)
    return {
        "alias": alias[1] if alias else None,
        "name": text.split("  |", 1)[0],
        "description": text,
        "shape": json.loads(shape[1]) if shape else None,
        "bytes": int(size[1]) if size else None,
        "location": location[1] if location else None
    }


def audit(path):
    raw = path.read_text()
    prefixes = ("custom_deepseek_v41_mxfp4_prepared_dequant", "custom_deepseek_v41_mxfp4_k128_dequant",
                "custom_deepseek_v41_mxfp4_n512_dequant", "custom_deepseek_v41_expert_n256_fp8",
                "custom_deepseek_v41_expert_n256_bf16", "custom_deepseek_v41_expert_n256_normal_bf16",
                "custom_deepseek_v41_expert_n256_slots_fp8_gaudi2", "custom_deepseek_v41_expert_n256_reuse_fp8_gaudi2",
                "custom_deepseek_v41_expert_n256_horizontal_fp8_gaudi2",
                "custom_deepseek_v41_expert_token_wide_sat_fp8_gaudi2",
                "custom_deepseek_v41_expert_n256_sat_fp8_gaudi2",
                "custom_deepseek_v41_expert_token_wide6_sat_fp8_gaudi2",
                "custom_deepseek_v41_expert_token_wide3_sat_fp8_gaudi2",
                "custom_deepseek_v41_expert_token_wide3_active_k_sat_fp8_gaudi2",
                "custom_deepseek_v41_expert_token_wide3_unroll_sat_fp8_gaudi2",
                "custom_deepseek_v41_expert_token_wide6_unroll_sat_fp8_gaudi2",
                "custom_deepseek_v41_expert_token_wide6_aligned_sat_fp8_gaudi2",
                "custom_deepseek_v41_expert_n256_slots6_sat_fp8_gaudi2")
    if not any(prefix in raw for prefix in prefixes):
        return None
    decode, matrix, tensors = [], [], {}
    for node in raw.split("\nnode {"):
        name, op = re.search(r'  name: "([^"]+)"', node), re.search(r'  op: "([^"]+)"', node)
        if not name or not op:
            continue
        attrs = dict(re.findall(r'key: "([^"]+)"\s+value \{\s+s: "([^"]*)"', node))
        for key, value in attrs.items():
            if key.startswith(("inputTensor:", "outputTensor:")):
                info = tensor_info(value)
                tensors[info["name"]] = info
        if op[1].startswith(prefixes):
            decode.append({"node": name[1], "output": tensor_info(attrs["outputTensor:0"])})
        if "gemm" in op[1].lower():
            matrix.append({
                "node": name[1],
                "op": op[1],
                "activation": tensor_info(attrs.get("inputTensor:0", "")),
                "weight": tensor_info(attrs.get("inputTensor:1", "")),
                "output": tensor_info(attrs.get("outputTensor:0", ""))
            })
    def canonical(info):
        seen = set()
        while info.get('alias') and info['name'] not in seen:
            seen.add(info['name'])
            target = tensors.get(info['alias'])
            if not target or target['bytes'] != info['bytes'] or target['location'] != info['location']:
                break
            info = target
        return info['name']

    decoded_names = {canonical(node["output"]) for node in decode}
    expert_matrices = []
    for node in matrix:
        if canonical(node["weight"]) in decoded_names:
            node["decoded_operand_index"] = 1
            expert_matrices.append(node)
        elif canonical(node["activation"]) in decoded_names:
            # W^T x^T reverses operand roles without moving either tensor.
            # Follow the actual decoded producer, not a fixed operand index.
            node["activation"], node["weight"] = node["weight"], node["activation"]
            node["decoded_operand_index"] = 0
            expert_matrices.append(node)
    if not matrix:
        return None  # A standalone decoder output is not an SRAM consumption proof.
    # A weight DMA may hide direct producer names. Keep the failing allocation
    # evidence rather than silently dropping a graph with decoded DRAM weights.
    consumers = {
        name: [node["node"] for node in expert_matrices if canonical(node["weight"]) == name]
        for name in decoded_names
    }
    return {
        "graph":
        str(path),
        "sha256":
        hashlib.sha256(path.read_bytes()).hexdigest(),
        "decode_node_count":
        len(decode),
        "mme_node_count":
        len(matrix),
        "expert_mme_node_count":
        len(expert_matrices),
        "decoded_bytes":
        sum(node["output"]["bytes"] or 0 for node in decode),
        "all_decoded_weights_in_sram":
        bool(decode) and all(node["output"]["location"] == "SRAM" for node in decode),
        "all_mme_weights_in_sram":
        bool(expert_matrices) and all(node["weight"]["location"] == "SRAM" for node in expert_matrices),
        "all_decoded_weights_consumed_once":
        bool(consumers) and all(len(nodes) == 1 for nodes in consumers.values()),
        "decode":
        decode,
        "matrix":
        matrix,
        "decode_consumers":
        consumers
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graphs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = [entry for path in sorted(args.graphs.rglob("*PostGraph*")) if (entry := audit(path))]
    if not result:
        raise RuntimeError("No compiled V4.1 MoE graphs found")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for entry in result:
        print(
            json.dumps({
                key: value
                for key, value in entry.items() if key not in ("decode", "matrix", "decode_consumers")
            }))
    if not all(entry["all_decoded_weights_in_sram"] and entry["all_mme_weights_in_sram"]
               and entry["all_decoded_weights_consumed_once"] for entry in result):
        raise SystemExit("Candidate did not satisfy SRAM placement")


if __name__ == "__main__":
    main()
