# SPDX-License-Identifier: Apache-2.0
"""Static request-batch layer groups with explicit CSA2 producer dependencies."""
from torch import nn

from vllm_gaudi.models.deepseek_v41_program import PreparedInput, _compile_group
from vllm_gaudi.ops.deepseek_v41_math import rms_norm


class PreparedBatchLayerGroup(nn.Module):

    def __init__(self, stage, start, stop, *, text_input=False):
        super().__init__()
        if stage.dspark:
            raise ValueError("Request batches use ordinary decode, not draft positions")
        self.layers = nn.ModuleList(list(stage.layers[start:stop]))
        self.input = PreparedInput(stage.weights.embed, stage.tp_rank, stage.reduce) if text_input else None
        self.final = stage.pp_rank == 1 and stop == len(stage.layers)
        self.norm = stage.weights.norm if self.final else None
        config = stage.config["text_config"]
        self.eps = config["rms_norm_eps"]
        self.index_sources = tuple(int(k) for k in stage.shared.topk)
        self.kv_sources = tuple(int(k) for k in stage.shared.sources)
        self.mapping = tuple((self.index_sources.index(max(s for s in self.index_sources if s <= layer.layer)),
                              self.kv_sources.index(max(s for s in self.kv_sources
                                                        if s <= layer.layer))) if layer.attention.ratio else (-1, -1)
                             for layer in self.layers)

    def forward(self, residual, pre_mix, positions, input_ids, engram_rows, slots, pages, selections, candidates,
                main_ready, index_ready):
        if self.input is not None:
            residual, pre_mix = self.input(input_ids)
        image = (input_ids == 129264) | (input_ids == 129265)
        selections, main_ready, index_ready = list(selections), list(main_ready), list(index_ready)
        for layer, (index_source, kv_source) in zip(self.layers, self.mapping):
            rows = engram_rows[0 if layer.layer == 1 else 1] if layer.layer in (1, 14) else None
            # Ratio-zero layers consume neither compressed indices nor their
            # completion tokens; retain fixed empty operands for the ABI.
            selection = selections[index_source] if index_source >= 0 else selections[0]
            main_done = main_ready[kv_source] if kv_source >= 0 else main_ready[0]
            index_done = index_ready[kv_source] if kv_source >= 0 else index_ready[0]
            result = layer.forward_batch(residual, pre_mix, positions, image, rows, slots, pages, selection, candidates,
                                         main_done, index_done)
            residual, pre_mix, selection, candidates, main_done, index_done = result
            if index_source >= 0:
                selections[index_source] = selection
                main_ready[kv_source], index_ready[kv_source] = main_done, index_done
        pre_mix = pre_mix.contiguous()
        if self.final:
            value = (residual.float() * pre_mix.unsqueeze(-1)).sum(1).to(residual.dtype)
            residual = rms_norm(value, self.norm.weight, self.eps, request_batch=True)
            residual = residual.masked_fill((slots < 0)[:, None], 0)
        return residual, pre_mix, tuple(selections), candidates, tuple(main_ready), tuple(index_ready)


class CompiledBatchStage:
    """Finite batch shapes; request lengths and page mappings remain inputs.

    The outer native replay owns capture/submission. This class only provides
    the five four-layer recipes and their explicit inter-group tensors.
    """

    def __init__(self, stage, *, text_input=False, group_size=4, backend="hpu_backend"):
        if group_size < 1 or len(stage.layers) % group_size:
            raise ValueError("Invalid request-batch layer grouping")
        from vllm_gaudi import envs
        if backend == "hpu_backend" and envs.VLLM_HPU_DSV41_TP_MHC_OVERLAP:
            from vllm_gaudi.compilation.deepseek_v41_overlap import make_backend
            backend = make_backend()
        self.groups = tuple(
            PreparedBatchLayerGroup(stage, start, start + group_size, text_input=text_input and start == 0)
            for start in range(0, len(stage.layers), group_size))
        self.chunks = tuple(_compile_group(group, native=False, backend=backend) for group in self.groups)

    def __call__(self, hidden, pre, positions, ids, engram, slots, pages, selections, candidates, main_ready,
                 index_ready):
        for chunk in self.chunks:
            hidden, pre, selections, candidates, main_ready, index_ready = chunk(hidden, pre, positions, ids, engram,
                                                                                 slots, pages, selections, candidates,
                                                                                 main_ready, index_ready)
        return hidden, pre, None
