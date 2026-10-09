# SPDX-License-Identifier: Apache-2.0
"""Native C5/Markov same-prefix probability audit; no serving or speed claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument("--candidate", choices=("mtp_fp8", "mtp_sat", "mtp_k128", "draft_mhc", "vocab_head_fp8",
                                               "draft_packed_mla", "draft_kv_decode", "draft_shared_fp8",
                                               "draft_query_fp8", "draft_dense_fp8", "draft_tail"),
                        default="mtp_fp8")
    args = parser.parse_args()
    precision_attribute = args.candidate
    attention_candidate = args.candidate in (
        'draft_packed_mla', 'draft_kv_decode', 'draft_query_fp8', 'draft_dense_fp8')
    head_candidate = args.candidate == 'vocab_head_fp8'
    draft_tail = args.candidate == 'draft_tail'
    if args.candidate == 'draft_shared_fp8' and os.environ.get('VLLM_HPU_DSV41_DSPARK_DRAFT_SHARED_FP8') != '1':
        parser.error('Prepare draft shared FP8 banks before capturing either arm')
    if args.candidate == 'draft_query_fp8' and os.environ.get('VLLM_HPU_DSV41_DSPARK_DRAFT_QUERY_FP8') != '1':
        parser.error('Prepare draft Q FP8 banks before capturing either arm')
    if args.candidate == 'draft_dense_fp8' and os.environ.get('VLLM_HPU_DSV41_DSPARK_DRAFT_DENSE_FP8') != '1':
        parser.error('Prepare draft input/output FP8 banks before capturing either arm')
    if head_candidate and os.environ.get('VLLM_HPU_DSV41_DSPARK_VOCAB_HEAD_FP8') != '1':
        parser.error('Prepare the optional vocabulary bank before capturing either arm')
    rank, tp = int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    os.environ['HLS_MODULE_ID'] = os.environ['HABANA_VISIBLE_MODULES'].split(',')[rank]
    os.environ['PT_HPU_RECIPE_CACHE_CONFIG'] = (
        os.environ.get('PT_HPU_RECIPE_CACHE_CONFIG', '').replace('{rank}', str(rank)))
    from vllm_gaudi import envs as gaudi_envs
    if draft_tail and not (gaudi_envs.VLLM_HPU_DSV41_DSPARK_DRAFT_KV_DECODE
                           and gaudi_envs.VLLM_HPU_DSV41_DSPARK_MTP_K128):
        parser.error('Combined draft gate requires both qualified implementations before capture')
    from vllm_gaudi.entrypoints.deepseek_v41 import load_native_operators, prepare_environment
    prepare_environment(args.prepared, tensor_parallel_size=tp, pipeline_parallel_size=1)
    import habana_frameworks.torch.core  # noqa:F401
    import torch
    from vllm.config import VllmConfig, ParallelConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime
    from vllm_gaudi.models.deepseek_v41_program import PreparedDraft, _weight_tree, load_weight_tree
    from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2SharedState
    from vllm_gaudi.ops.deepseek_v41_replay import _Snapshot, stage_collectives
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut
    from vllm_gaudi.ops.tp2_model_adapter import DecoderTopology
    from vllm_gaudi.ops.tp2_prepared_plan import (
        _modules, _native_entries, collect_prepared_group_replays, record_native_decoder_outputs,
        replay_native_decoder, shutdown_prepared_group_plans)

    root = Path(os.environ['DSV41_RUN_EVIDENCE'])
    report = dict(rank=rank, candidate=args.candidate, status='loading', cases=[], performance_measured=False)

    def save():
        (root / f'mtp-teacher-rank{rank}.json').write_text(json.dumps(report,indent=2)+'\n')

    class TeacherPlan(torch.nn.Module):
        def __init__(self, draft):
            super().__init__()
            self.draft, self.states = draft, tuple(layer.attention.swa for layer in draft.layers)
            self.adapter = DecoderTopology('deepseek_v41_dspark_control', (1,), 0, False, 12)
            self.metadata = SimpleNamespace(native_completion=None)
            self.compiled = torch.compile(self.body, backend='hpu_backend', fullgraph=True, dynamic=False)

        def body(self, ids, positions, target_hidden=None):
            hidden, base = self.draft.forward_local(ids[:1], positions[:5])
            last = self.draft.weights.get_submodule('2')
            scores = []
            for i in range(5):
                previous = self.draft._embed(ids[i:i+1], last.markov_head.embed)
                bias = torch.nn.functional.linear(previous.float(), last.markov_head.head.weight)
                scores.append(base[i:i+1]+bias)
            target = self.draft._head_projection(target_hidden) if target_hidden is not None else None
            return torch.cat(scores), target, hidden

        def forward(self, ids, positions, target_hidden=None):
            roots = dict(hidden_states=None, pre_mix=None, input_ids=ids, positions=positions,
                         metadata=self.metadata, state_generation=1, state_tensors=self.states,
                         attention_inputs=() if target_hidden is None else (target_hidden,))
            values = replay_native_decoder(self, **roots)
            if values is not None:
                return values[0], values[2], values[1]
            with collect_prepared_group_replays(owner=self, adapter=self.adapter,
                                               snapshot=lambda:_Snapshot(self.states), **roots) as context:
                context['group_index'] = 0
                logits, target, hidden = self.compiled(ids, positions, target_hidden)
                record_native_decoder_outputs(logits, target, hidden)
            return logits, hidden, target

    save()
    try:
        with set_current_vllm_config(VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=tp))), \
                torch.inference_mode():
            bind_worker_cpu(rank)
            torch.hpu.set_device(rank)
            operation = ('custom_deepseek_v41_mtp_sat_moe_fp8_gaudi2' if args.candidate == 'mtp_sat' else
                         'custom_deepseek_v41_mtp_moe_k128_bf16_gaudi2'
                         if args.candidate in ('mtp_k128', 'draft_tail') else
                         'custom_deepseek_v41_control_mme_f32_gaudi2' if args.candidate == 'draft_mhc' else
                         'custom_deepseek_v41_dense_quant_gaudi2'
                         if head_candidate or args.candidate == 'draft_shared_fp8' else
                         'custom_deepseek_v41_q_projection_rope_gaudi2' if args.candidate == 'draft_query_fp8' else
                         'custom_deepseek_v41_dense_fp8_gaudi2' if args.candidate == 'draft_dense_fp8' else
                         'custom_deepseek_v41_selected_kv_bf16_gaudi2' if args.candidate == 'draft_kv_decode' else
                         'custom_deepseek_v41_paged_mla_mme_gaudi2' if attention_candidate else
                         'custom_deepseek_v41_mtp_moe_fp8_gaudi2')
            load_native_operators(required=(operation,))
            if draft_tail:
                load_native_operators(required=('custom_deepseek_v41_mtp_moe_k128_bf16_gaudi2',
                                                'custom_deepseek_v41_selected_kv_bf16_gaudi2'))
            init_distributed_environment(world_size=tp, rank=rank, local_rank=rank,
                                         distributed_init_method='env://', backend='hccl')
            initialize_model_parallel(tensor_model_parallel_size=tp, pipeline_model_parallel_size=1)
            initialize_tp2_fused_ar_norm_runtime()
            bind_worker_helpers(rank)
            reduce, gather = stage_collectives(rank, True, tp)
            shard = PreparedV41Shard(args.prepared, 0, rank)
            text = json.loads((args.prepared/'config.json').read_text())['text_config']
            specs = {name: spec for name,spec in shard.specs.items()
                     if name.startswith('mtp.') or name=='head.weight'}
            weights = _weight_tree(specs)
            load_weight_tree(shard, weights, 'hpu', specs)
            weights.head.weight = weights.head.weight.bfloat16()
            stage = SimpleNamespace(weights=weights, shard=shard, config={'text_config':text}, tp_rank=rank,
                                    tensor_parallel_size=tp, reduce=reduce, all_gather=gather, bf16_head=True,
                                    shared=PagedCSA2SharedState(text,0,40,'hpu',1048576,tensor_parallel_size=tp))
            draft = PreparedDraft(stage, mxfp4_bf16_lut(torch.device('hpu')), 'hpu')
            cases = [torch.load(path,weights_only=True,map_location='cpu')
                     for path in sorted((args.fixtures/f'rank{rank}').glob('c6-*.pt'))]
            if not 3<=len(cases)<=5:
                raise ValueError('Three to five actual prefixes required')
            plans = []
            for arm in (0,1):
                if head_candidate:
                    os.environ['VLLM_HPU_DSV41_DSPARK_VOCAB_HEAD_FP8'] = str(arm)
                else:
                    for layer in draft.layers:
                        if draft_tail:
                            layer.attention.draft_kv_decode = bool(arm)
                            layer.moe.mtp_k128 = bool(arm)
                        else:
                            owner = (layer if args.candidate == 'draft_mhc'
                                     else layer.attention if attention_candidate else layer.moe)
                            setattr(owner, precision_attribute, bool(arm))
                plan = TeacherPlan(draft)
                for layer, cache in zip(draft.layers,cases[0]['draft_swa'],strict=True):
                    layer.attention.swa.copy_(cache.to('hpu'))
                ids, pos = cases[0]['ids'].to('hpu'), cases[0]['positions'].to('hpu')
                target_hidden = cases[0]['hidden'].to('hpu') if head_candidate else None
                for _ in range(3):
                    plan(ids,pos,target_hidden)
                    torch.hpu.synchronize()
                if plan not in _native_entries:
                    raise RuntimeError('Teacher C5/Markov did not capture native replay')
                if draft_tail:
                    calls = {'custom_deepseek_v41_mtp_moe_k128_bf16_gaudi2': 0,
                             'custom_deepseek_v41_selected_kv_bf16_gaudi2': 0}
                    for module in tuple(_modules):
                        if not any(owner is not None and owner[0] == id(plan) for owner in module.plan_owners):
                            continue
                        for node in module.original.graph.nodes:
                            if node.op != 'call_module':
                                continue
                            recipe = module.original.get_submodule(node.target)
                            for item in recipe.fx_module.graph.nodes:
                                if item.op == 'call_function':
                                    for name in calls:
                                        calls[name] += name in str(item.target)
                    report.setdefault('captured_draft_tail_arms', []).append(dict(arm=arm, calls=calls))
                    save()
                    if any(count != 3 * arm for count in calls.values()):
                        raise AssertionError('Both qualified draft paths must execute once in all three MTP layers')
                if args.candidate in ('draft_query_fp8', 'draft_dense_fp8'):
                    selected, dense_selected, direct_selected, signatures = 0, 0, 0, []
                    for module in tuple(_modules):
                        if not any(owner is not None and owner[0] == id(plan) for owner in module.plan_owners):
                            continue
                        for node in module.original.graph.nodes:
                            if node.op != 'call_module':
                                continue
                            recipe = module.original.get_submodule(node.target)
                            for item in recipe.fx_module.graph.nodes:
                                if item.op != 'call_function':
                                    continue
                                name = str(item.target)
                                selected += 'custom_deepseek_v41_q_projection_rope_gaudi2' in name
                                dense_selected += 'custom_deepseek_v41_dense_fp8_gaudi2' in name
                                direct_selected += 'fp8_gemm_v2' in name
                                if any(key in name for key in ('linear', 'mm.', 'bmm.', 'q_projection_rope', 'fp8')):
                                    operands = []
                                    for arg in item.args[:2]:
                                        metadata = getattr(arg, 'meta', {})
                                        value = metadata.get('val', metadata.get('example_value'))
                                        operands.append(None if not isinstance(value, torch.Tensor) else
                                                        dict(shape=list(value.shape), dtype=str(value.dtype)))
                                    signatures.append(dict(name=item.name, operator=name, operands=operands,
                                                           source=str(item.meta.get('source_fn_stack', ''))))
                    report.setdefault('captured_projection_arms', []).append(
                        dict(arm=arm, q_fp8_calls=selected, dense_fp8_calls=dense_selected,
                             direct_fp8_calls=direct_selected, linear_signatures=signatures))
                    save()
                    if args.candidate == 'draft_query_fp8' and selected != 3 * arm:
                        raise AssertionError('Teacher Q candidate was bypassed or incomplete')
                    if (args.candidate == 'draft_dense_fp8'
                            and (dense_selected != 3 * arm or direct_selected != 3 * arm)):
                        raise AssertionError('Teacher draft input/output FP8 candidate was bypassed or incomplete')
                plans.append(plan)
            for index,case in enumerate(cases):
                values=[]
                for arm, plan in enumerate(plans):
                    if head_candidate:
                        os.environ['VLLM_HPU_DSV41_DSPARK_VOCAB_HEAD_FP8'] = str(arm)
                    for layer,cache in zip(draft.layers,case['draft_swa'],strict=True):
                        layer.attention.swa.copy_(cache.to('hpu'))
                    target_hidden = case['hidden'].to('hpu') if head_candidate else None
                    logits,hidden,target=plan(case['ids'].to('hpu'),case['positions'].to('hpu'),target_hidden)
                    torch.hpu.synchronize()
                    values.append((logits.cpu().clone(),hidden.cpu().clone(),
                                   target.cpu().clone() if target is not None else None))
                identity=dict(request_id=case['request_id'],context_prefix_tokens=case['context_prefix_tokens'],
                              ids=case['ids'].tolist(),positions=case['positions'].tolist(),
                              history=case['cursor_history'].tolist())
                prefix=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
                delta=values[1][1].float()-values[0][1].float()
                report['cases'].append(dict(case=index,prefix_sha256=prefix,
                                           hidden_max_abs=float(delta.abs().max()),
                                           hidden_relative_l2=float(delta.norm()/values[0][1].float().norm()),
                                           finite=all(bool(torch.isfinite(x).all()) for pair in values for x in pair
                                                      if x is not None)))
                exported = dict(identity=identity,prefix_sha256=prefix,
                                reference_draft_logits=values[0][0],candidate_draft_logits=values[1][0])
                if head_candidate:
                    exported.update(reference_target_logits=values[0][2][:5],
                                    candidate_target_logits=values[1][2][:5],
                                    target_producer='Same frozen actual-request normalized Target hidden',
                                    preparation=draft.output_head.dspark_vocab_fp8_preparation)
                torch.save(exported,
                           root/f'mtp-prefix-case{index}-rank{rank}.pt')
                save()
            report['status']='completed_conditional_teacher'
    except Exception as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        save()
        shutdown_prepared_group_plans()


if __name__=='__main__':
    main()
