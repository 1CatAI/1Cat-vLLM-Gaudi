# SPDX-License-Identifier: Apache-2.0
"""Token-page lookup; the engine also requires an auxiliary state checkpoint."""

import itertools

from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager


class V41PrefixManager(FullAttentionManager):
    supports_fine_grained_hash_lookup = False

    @classmethod
    def find_longest_cache_hit(cls,
                               block_hashes,
                               max_length,
                               kv_cache_group_ids,
                               block_pool,
                               kv_cache_spec,
                               drop_eagle_block,
                               alignment_tokens,
                               dcp_world_size=1,
                               pcp_world_size=1):
        if (not kv_cache_spec.requires_auxiliary_prefix_state or not kv_cache_spec.prefix_cacheable
                or kv_cache_spec.block_size != 128 or block_pool.hash_block_size != 128 or alignment_tokens != 128
                or drop_eagle_block or dcp_world_size != 1 or pcp_world_size != 1):
            raise ValueError("V4.1 prefix lookup requires complete 128-token pages and auxiliary checkpoints")
        blocks = tuple([] for _ in kv_cache_group_ids)
        for digest in itertools.islice(block_hashes, max_length // 128):
            cached = block_pool.get_cached_block(digest, kv_cache_group_ids)
            if not cached:
                break
            for group, block in zip(blocks, cached, strict=True):
                group.append(block)
        return blocks, len(blocks[0]) * 128
