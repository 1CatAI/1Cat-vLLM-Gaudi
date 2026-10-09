# SPDX-License-Identifier: Apache-2.0
"""Actual same-prefix compressor audit; diagnostic copies never enter timers."""
import types


def install(attention, rows):
    import torch
    for name, shape in (("kv", (rows, 512)), ("score", (rows, 512)),
                        ("kv_history", (8, 512)), ("score_history", (8, 512))):
        attention.register_buffer("compressor_oracle_" + name,
                                  torch.empty(shape, dtype=torch.float32, device="hpu"), False)
    original = attention._project_compressor_input

    def project(self, value):
        self.compressor_oracle_kv_history.copy_(self.kv_history)
        self.compressor_oracle_score_history.copy_(self.score_history)
        kv, score = original(value)
        self.compressor_oracle_kv.copy_(kv)
        self.compressor_oracle_score.copy_(score)
        return kv, score

    attention._project_compressor_input = types.MethodType(project, attention)


def audit(parent, candidate, positions, path):
    import torch
    names = ("kv", "score", "kv_history", "score_history")
    a = {name: getattr(parent, "compressor_oracle_" + name).cpu() for name in names}
    b = {name: getattr(candidate, "compressor_oracle_" + name).cpu() for name in names}
    same = {name: torch.equal(a[name], b[name]) for name in names}
    position = positions.to(torch.int32).contiguous()
    h, sh, kv, score = (b[name].to("hpu") for name in ("kv_history", "score_history", "kv", "score"))
    output = torch.ops.custom_op.custom_deepseek_v41_compressor_sequence_bf16_gaudi2(h, sh, kv, score, position).cpu()
    actual_history = h.cpu(), sh.cpu()
    ring = (positions.cpu().long() & 7)
    expected_history = b["kv_history"].clone(), b["score_history"].clone()
    expected_history[0].index_copy_(0, ring, b["kv"])
    expected_history[1].index_copy_(0, ring, b["score"])
    h, sh = (value.to("hpu") for value in expected_history)
    accepted = []
    for i in range(position.numel()):
        accepted.append(torch.ops.custom_op.custom_deepseek_v41_compressor_pair_bf16_gaudi2(
            h, sh, kv[i:i+1].contiguous(), score[i:i+1].contiguous(), position[i:i+1].contiguous()).cpu())
    reference = torch.cat(accepted)
    delta = output.float() - reference.float()
    relative = float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference.float()).clamp_min(1e-30))
    close = bool(torch.allclose(output.float(), reference.float(), atol=.5, rtol=.008))
    state_exact = all(torch.equal(x, y) for x, y in zip(actual_history, expected_history, strict=True))
    stats = dict(max_abs=float(delta.abs().max()), rms=float(delta.square().mean().sqrt()), relative_l2=relative,
                 finite=bool(torch.isfinite(output).all()), bitwise_equal=torch.equal(output, reference),
                 passed=close and relative <= .002 and state_exact and all(same.values()))
    torch.save(dict(parent_inputs=a, candidate_inputs=b, positions=positions.cpu(), candidate_output=output,
                    c1_reference=reference, actual_history=actual_history), path)
    return dict(reference="Accepted C1 pair kernel, same actual C6 inputs and fully updated transaction history",
                input_exact=same, history_exact=state_exact, projections={"compressor": stats},
                passed=stats["passed"], full_service_quality_passed=False)
