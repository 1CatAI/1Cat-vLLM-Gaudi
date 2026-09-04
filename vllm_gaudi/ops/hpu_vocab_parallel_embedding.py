# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    get_masked_input_and_mask,
)


@VocabParallelEmbedding.register_oot
class HPUVocabParallelEmbedding(VocabParallelEmbedding):

    def forward(self, input_):
        if self.tp_size > 1:
            masked_input, input_mask = get_masked_input_and_mask(
                input_,
                self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                self.shard_indices.num_org_vocab_padding,
                self.shard_indices.added_vocab_start_index,
                self.shard_indices.added_vocab_end_index,
            )
        else:
            masked_input = input_

        output_parallel = self.quant_method.embedding(self, masked_input.long())
        if self.tp_size > 1:
            output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
            if getattr(self, "_hpu_defer_tp2_reduce", False):
                return output_parallel
            return tensor_model_parallel_all_reduce(output_parallel)
        return output_parallel
