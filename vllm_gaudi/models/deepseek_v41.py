# SPDX-License-Identifier: Apache-2.0
"""HPU V4.1 model registration with prepared weights and explicit stage inputs."""

from pathlib import Path

import torch

from vllm_gaudi.ops.deepseek_v41_diagnostics import trace_phase
from torch import nn

from vllm.distributed import get_pp_group, get_tensor_model_parallel_rank
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.models.deepseek_v4.common.vision import DeepseekV4Aligner, DeepseekV4ViT
from vllm.models.deepseek_v4_1.common.mm_preprocess import (
    IMAGE, IMAGE_END, IMAGE_NEW_LINE, IMAGE_PLACEHOLDER, IMAGE_START,
    DeepseekV4VLDummyInputsBuilder, DeepseekV4VLMultiModalProcessor, DeepseekV4VLProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from vllm_gaudi import envs
from vllm_gaudi.models.deepseek_v41_program import CompiledStage, PreparedInput, PreparedStage
from vllm_gaudi.ops.deepseek_v41_host import EngramHost
from vllm_gaudi.ops.deepseek_v41_replay import StageReplay, stage_collectives


@MULTIMODAL_REGISTRY.register_processor(
    DeepseekV4VLMultiModalProcessor, info=DeepseekV4VLProcessingInfo, dummy_inputs=DeepseekV4VLDummyInputsBuilder)
class HpuDeepseekV41ForCausalLM(nn.Module, SupportsMultiModal, SupportsPP):
    requires_raw_input_tokens = True
    supports_encoder_tp_data = True

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        del prefix
        if not envs.VLLM_HPU_DSV41_PREPARED_SHARDS:
            raise RuntimeError("V4.1 HPU execution requires VLLM_HPU_DSV41_PREPARED_SHARDS=1")
        parallel = vllm_config.parallel_config
        if (parallel.tensor_parallel_size != 2 or parallel.pipeline_parallel_size != 2
                or vllm_config.scheduler_config.max_num_seqs != 1
                or vllm_config.model_config.max_model_len > 512):
            raise ValueError("The prepared V4.1 profile requires TP2 x PP2, one request and context <=512")
        if vllm_config.load_config.load_format != "dsv41_prepared":
            raise ValueError("V4.1 rank files require --load-format dsv41_prepared")
        self.config = vllm_config.model_config.hf_config
        self.multimodal_config = vllm_config.model_config.multimodal_config
        self.directory = Path(vllm_config.model_config.model)
        self.device = vllm_config.device_config.device
        self.pp_rank, self.tp_rank = get_pp_group().rank_in_group, get_tensor_model_parallel_rank()
        self.native = envs.VLLM_HPU_DSV41_GRAPH_REPLAY
        if not hasattr(torch.ops.custom_op, "custom_deepseek_v41_mxfp4_prepared_moe_bf16_gaudi2"):
            torch.ops.load_library(envs.VLLM_HPU_DSV4_TPC_OP_LIBRARY)
        if self.native:
            from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime
            initialize_tp2_fused_ar_norm_runtime()
        reduce, gather = stage_collectives(self.tp_rank, self.native)
        self.program = PreparedStage(self.directory, self.pp_rank, self.tp_rank, reduce, gather, self.device,
                                     dspark=envs.VLLM_HPU_DSV41_DSPARK)
        self.program.replay_owner = StageReplay(self.program) if self.native else None
        self.ordinary = CompiledStage(self.program)
        self.input_program = None
        if self.pp_rank == 0 and self.native:
            self.input_program = torch.compile(PreparedInput(self.program.weights.embed, self.tp_rank, reduce),
                                               backend="hpu_backend", fullgraph=True, dynamic=False)
        self.engram_host, self.step_ticket, self.last_aux = None, None, None
        self.step_is_decode = False
        self.extra = dict(vllm_config.load_config.model_loader_extra_config or {})
        self.vision = self.aligner = None
        if self.pp_rank == 0 and envs.VLLM_HPU_DSV41_VISION:
            if self.multimodal_config.mm_encoder_tp_mode != "data":
                raise ValueError("Prepared V4.1 vision weights require --mm-encoder-tp-mode data")
            # The prepared file already owns the tensors. Allocate no second
            # tower here; bind its parameters to the loaded storage below.
            with torch.device("meta"):
                self.vision, self.aligner = DeepseekV4ViT(self.config), DeepseekV4Aligner(self.config)

    def load_prepared_weights(self):
        self.program.load_prepared(self.device)
        if self.pp_rank == 0:
            if not envs.VLLM_HPU_DSV41_ENGRAM_HOST_TABLE:
                raise RuntimeError("V4.1 Engram must use the native host table; no HBM/eager fallback is available")
            self.engram_host = EngramHost(self.directory, self.tp_rank, self.device,
                checkpoint_audit=self.extra.get("checkpoint_audit"),
                force_lock=self.extra.get("engram_force_lock", False))
            self._bind_vision()

    def _bind_vision(self):
        if self.vision is None:
            return
        for name, module in (("vision", self.vision), ("aligner", self.aligner)):
            tree = self.program.weights.get_submodule(name)
            for target_name, parameter in list(module.named_parameters()):
                parent_name, _, attribute = target_name.rpartition(".")
                value = getattr(tree.get_submodule(parent_name), attribute)
                if value.shape != parameter.shape:
                    raise ValueError(f"Prepared vision geometry differs from upstream module: {name}.{target_name}")
                setattr(module.get_submodule(parent_name), attribute, nn.Parameter(value, requires_grad=False))

    def load_weights(self, weights):
        del weights
        raise RuntimeError("Use the registered dsv41_prepared loader; original HF expert loading is disabled")

    def get_language_model(self):
        return self

    @classmethod
    def get_placeholder_str(cls, modality, i):
        del i
        if modality != "image":
            raise ValueError(f"Unsupported V4.1 modality: {modality}")
        return IMAGE_PLACEHOLDER

    def embed_multimodal(self, **kwargs):
        patches = kwargs.get("patches")
        if patches is None:
            return []
        if self.vision is None:
            raise RuntimeError("Vision encoding is owned by PP0 with VLLM_HPU_DSV41_VISION=1")
        vit_grid, llm_grid = kwargs["vit_grid"].tolist(), kwargs["llm_grid"].tolist()
        types, vit_offset, span_offset, result = kwargs["types"], 0, 0, []
        for (height, width), (out_height, out_width) in zip(vit_grid, llm_grid, strict=True):
            count = height * width
            image = self.aligner(self.vision(patches[vit_offset:vit_offset + count].to(torch.bfloat16),
                                            height, width), height, width)
            span_length = out_height * (out_width + 1) + 2
            roles = types[span_offset:span_offset + span_length].to(image.device)
            span = image.new_empty(span_length, image.shape[-1])
            span[roles == IMAGE_START] = self.program.weights.image_start
            span[roles == IMAGE_NEW_LINE] = self.program.weights.image_newline
            span[roles == IMAGE_END] = self.program.weights.image_end
            span[roles == IMAGE] = image
            result.append(span)
            vit_offset, span_offset = vit_offset + count, span_offset + span_length
        return tuple(result)

    def embed_input_ids(self, input_ids, multimodal_embeddings=None, *, is_multimodal=None):
        if self.pp_rank != 0:
            raise RuntimeError("Target embedding is owned by PP0")
        values = self.program.embed(input_ids.masked_fill(input_ids == 129265, 129264))
        if multimodal_embeddings is not None and len(multimodal_embeddings):
            from vllm.model_executor.models.utils import _merge_multimodal_embeddings
            if is_multimodal is None:
                raise ValueError("V4.1 image embeddings require the processor's span mask")
            values = _merge_multimodal_embeddings(inputs_embeds=values, multimodal_embeddings=multimodal_embeddings,
                                                  is_multimodal=is_multimodal)
        return values

    def make_empty_intermediate_tensors(self, batch_size, dtype, device):
        return IntermediateTensors({
            "hidden_states": torch.empty(batch_size, 4, 5120, dtype=dtype, device=device),
            "pre_mix": torch.empty(batch_size, 4, dtype=torch.float32, device=device),
        })

    @trace_phase
    def prepare_step(self, request_id, token_ids, *, is_decode, reset=False):
        self.step_is_decode = is_decode
        if self.pp_rank == 0:
            if self.step_ticket is not None:
                raise RuntimeError("Previous V4.1 verify has not committed its accepted input prefix")
            if reset:
                self.engram_host.reset(request_id)
            image_mask = [token in (129264, 129265) for token in token_ids]
            self.step_ticket = self.engram_host.prepare(request_id, token_ids, image_mask)

    @trace_phase
    def complete_step(self, committed_inputs):
        if self.pp_rank == 0:
            if self.step_ticket is None:
                raise RuntimeError("V4.1 completion has no input transaction")
            self.engram_host.complete(self.step_ticket, committed_inputs)
            self.step_ticket = None

    @trace_phase
    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        del kwargs
        if not self.program.loaded:
            raise RuntimeError("Prepared V4.1 weights have not been loaded")
        input_ids, positions = input_ids.reshape(-1), positions.reshape(-1).to(torch.int32)
        if self.pp_rank == 0:
            if self.step_ticket is None:
                raise RuntimeError("V4.1 input metadata was not prepared by its worker")
            if (inputs_embeds is None and self.input_program is not None
                    and self.step_is_decode and input_ids.numel() == 1):
                residual, pre = self.input_program(input_ids)
            else:
                values = self.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds.reshape(-1, 5120)
                residual = values.unsqueeze(1).expand(-1, 4, -1).contiguous()
                pre = torch.zeros(input_ids.numel(), 4, device=values.device, dtype=torch.float32)
                pre[:, 0] = 1
            engram = self.step_ticket.buffers
        else:
            if intermediate_tensors is None:
                raise RuntimeError("PP1 has no matching PP0 hidden/pre-mix generation")
            residual, pre = intermediate_tensors["hidden_states"], intermediate_tensors["pre_mix"]
            engram = ()
        execute = self.program.replay_owner if self.native and self.step_is_decode else self.ordinary
        output, pre, aux = execute(residual, pre, positions, input_ids, engram)
        self.last_aux = aux
        if self.pp_rank == 0:
            return IntermediateTensors({"hidden_states": output, "pre_mix": pre})
        return output

    def compute_logits(self, hidden_states):
        return self.program.logits(hidden_states)

    def get_mtp_target_hidden_states(self):
        return self.last_aux

    def close(self):
        self.program.invalidate()
        if self.engram_host is not None:
            self.engram_host.close()

    def _apply(self, fn, recurse=True):
        if hasattr(self, "program"):
            self.program.invalidate()
        result = super()._apply(fn, recurse)
        self._bind_vision()
        return result
