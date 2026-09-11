# SPDX-License-Identifier: Apache-2.0
"""Explicit configuration contract for the bounded four-device V4.1 runner."""

from vllm_gaudi import envs


def is_v41(config):
    model = getattr(config, "model_config", None) or getattr(config, "target_model_config", None)
    return getattr(getattr(model, "hf_config", None), "model_type", None) == "deepseek_v41"


def supports_dspark(config):
    return is_v41(config) and envs.VLLM_HPU_DSV41_PREPARED_SHARDS and envs.VLLM_HPU_DSV41_DSPARK


def configure(config):
    if not is_v41(config):
        return
    if not envs.VLLM_HPU_DSV41_PREPARED_SHARDS:
        raise ValueError("V4.1 on HPU requires VLLM_HPU_DSV41_PREPARED_SHARDS=1")
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_native_libraries
    prepare_native_libraries()
    parallel, cache = config.parallel_config, config.cache_config
    if (parallel.tensor_parallel_size, parallel.pipeline_parallel_size, parallel.data_parallel_size) != (2, 2, 1):
        raise ValueError("The V4.1 prepared profile requires TP2 x PP2, DP1")
    if config.model_config.max_model_len > 512 or config.scheduler_config.max_num_seqs != 1:
        raise ValueError("The V4.1 prepared profile supports one request and context <=512")
    if cache.enable_prefix_caching:
        raise ValueError("V4.1 opaque request state requires --no-enable-prefix-caching")
    if cache.user_specified_block_size and cache.block_size != 512:
        raise ValueError("V4.1 CSA2 state uses one complete 512-token request block")
    cache.block_size = 512
    if cache.num_gpu_blocks_override is None:
        cache.num_gpu_blocks_override = 2
    if cache.num_gpu_blocks_override != 2:
        raise ValueError("The V4.1 single-request profile reserves null block 0 and request block 1")
    if config.load_config.load_format != "dsv41_prepared":
        raise ValueError("V4.1 prepared weights require --load-format dsv41_prepared")
    spec = config.speculative_config
    if envs.VLLM_HPU_DSV41_DSPARK:
        if spec is None or spec.method != "dspark" or spec.num_speculative_tokens != 5:
            raise ValueError("V4.1 DSpark requires method=dspark and num_speculative_tokens=5")
        if spec.enable_adaptive_verification:
            raise ValueError("Adaptive DSpark verification is outside the bounded V4.1 profile")
    elif spec is not None:
        raise ValueError("Enable VLLM_HPU_DSV41_DSPARK for the integrated draft")
    if config.scheduler_config.async_scheduling:
        raise ValueError("V4.1 PP verify commits currently require --no-async-scheduling")
    if config.lora_config is not None or config.kv_transfer_config is not None:
        raise ValueError("V4.1 prepared state does not support LoRA or external KV transfer")
    if (envs.VLLM_HPU_DSV41_GRAPH_REPLAY
            and not (envs.VLLM_HPU_TP2_STATIC_GROUP_PLAN and envs.VLLM_HPU_TP2_PREPARED_COMM)):
        raise ValueError("V4.1 graph replay requires the static group plan and prepared communication")
    from vllm_gaudi.ops.deepseek_v41_state import register_state_spec
    register_state_spec(config)
