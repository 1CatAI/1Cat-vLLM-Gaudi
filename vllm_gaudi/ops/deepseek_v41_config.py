# SPDX-License-Identifier: Apache-2.0
"""Explicit configuration contract for the bounded four-device V4.1 runner."""

from vllm_gaudi import envs


def is_v41(config):
    model = getattr(config, "model_config", None) or getattr(config, "target_model_config", None)
    return getattr(getattr(model, "hf_config", None), "model_type", None) == "deepseek_v41"


def supports_dspark(config):
    return is_v41(config) and envs.VLLM_HPU_DSV41_PREPARED_SHARDS and envs.VLLM_HPU_DSV41_DSPARK


def uses_v2(config):
    return is_v41(config) and envs.VLLM_HPU_DSV41_V2


def validate_v2(config):
    if not uses_v2(config):
        return
    if not config.use_v2_model_runner or not config.scheduler_config.async_scheduling:
        raise ValueError("V4.1 V2 requires VLLM_USE_V2_MODEL_RUNNER=1 and async scheduling")
    if envs.VLLM_HPU_DSV41_DSPARK or config.speculative_config is not None:
        raise ValueError("V4.1 V2 supports ordinary C1 without DSpark")
    if not envs.VLLM_HPU_DSV41_GRAPH_REPLAY or not envs.VLLM_HPU_DSV41_DIRECT_TOKEN_IDS:
        raise ValueError("V4.1 V2 requires native graph replay and direct device token inputs")
    if (envs.VLLM_HPU_DSV41_V2_SEGMENTED_PREFIX
            and not (envs.VLLM_HPU_DSV41_NATIVE_INPUT_GRAPH
                     and envs.VLLM_HPU_DSV41_FIXED_POSITIONS
                     and envs.VLLM_HPU_DSV41_V2_EARLY_INPUT_COMMIT
                     and envs.VLLM_HPU_TP2_NATIVE_JOINT_PLAN
                     and envs.VLLM_HPU_DSV41_TP_MHC_OVERLAP)):
        raise ValueError("V4.1 segmented prefix requires native fixed inputs, early commit, and TP dependencies")
    if envs.VLLM_HPU_DSV41_V2_SEGMENTED_PREFIX and envs.VLLM_HPU_DSV41_FUSED_STAGE_IO:
        raise ValueError("V4.1 segmented prefix requires the native input graph instead of fused stage I/O")
    if (envs.VLLM_HPU_DSV41_V2_DEVICE_ENGRAM
            and not (envs.VLLM_HPU_DSV41_V2_SEGMENTED_PREFIX
                     and envs.VLLM_HPU_DSV41_ENGRAM_NATIVE_C1
                     and envs.VLLM_HPU_DSV41_ENGRAM_C1_PACKET
                     and envs.VLLM_HPU_DSV41_ENGRAM_DIRECT_INPUT)):
        raise ValueError("Device Engram requires segmented PP0 replay, native C1 packets, and direct inputs")
    # Device Engram is a C1 producer over the current token and a three-token
    # rolling history.  It is independent of the paged CSA2 capacity; prompt
    # chunks continue to use the host gather path and only decode C1 enters
    # the segmented device producer.


def validate_sampling(params):
    from vllm.exceptions import VLLMValidationError
    if params is None:
        raise VLLMValidationError("V4.1 runner requires sampling parameters")
    if (params.temperature != 0 or params.logprobs is not None or params.prompt_logprobs is not None
            or params.presence_penalty != 0 or params.frequency_penalty != 0 or params.repetition_penalty != 1
            or params.allowed_token_ids is not None or params.bad_words or params.logit_bias
            or params.structured_outputs is not None or params.min_tokens):
        raise VLLMValidationError("V4.1 DSpark currently supports unmodified greedy sampling (temperature=0)")


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
    if not 1 <= config.model_config.max_model_len <= 1048576:
        raise ValueError("V4.1 supports context lengths up to the checkpoint's 1M limit")
    if cache.enable_prefix_caching:
        raise ValueError("V4.1 opaque request state requires --no-enable-prefix-caching")
    paged = config.model_config.max_model_len > 512
    block_size = 128 if paged else 512
    if cache.user_specified_block_size and cache.block_size != block_size:
        raise ValueError(f"V4.1 requires block_size={block_size} for the selected state layout")
    cache.block_size = block_size
    if not paged:
        if config.scheduler_config.max_num_seqs != 1:
            raise ValueError("The archived bounded V4.1 layout only supports one request")
        cache.num_gpu_blocks_override = 2
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
    validate_v2(config)
    if config.scheduler_config.async_scheduling and not uses_v2(config):
        raise ValueError("V4.1 PP verify commits currently require --no-async-scheduling")
    if config.use_v2_model_runner and not uses_v2(config):
        raise ValueError("V4.1 V2 needs the explicit VLLM_HPU_DSV41_V2 backend adapter")
    if config.lora_config is not None or config.kv_transfer_config is not None:
        raise ValueError("V4.1 prepared state does not support LoRA or external KV transfer")
    if (envs.VLLM_HPU_DSV41_GRAPH_REPLAY
            and not (envs.VLLM_HPU_TP2_STATIC_GROUP_PLAN and envs.VLLM_HPU_TP2_PREPARED_COMM)):
        raise ValueError("V4.1 graph replay requires the static group plan and prepared communication")
    from vllm_gaudi.ops.deepseek_v41_state import register_state_spec
    register_state_spec(config)
