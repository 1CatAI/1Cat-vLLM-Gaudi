# SPDX-License-Identifier: Apache-2.0
"""Normal vLLM load-format registration for frozen rank-local V4.1 files."""

from pathlib import Path

from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader


@register_model_loader("dsv41_prepared")
class PreparedV41ModelLoader(BaseModelLoader):
    def download_model(self, model_config):
        if not (Path(model_config.model) / "manifest.json").is_file():
            raise FileNotFoundError("V4.1 prepared manifest is missing; finish offline preparation before loading")

    def load_weights(self, model, model_config):
        self.download_model(model_config)
        from vllm_gaudi.models.deepseek_v41 import HpuDeepseekV41ForCausalLM
        if not isinstance(model, HpuDeepseekV41ForCausalLM):
            raise TypeError("dsv41_prepared cannot load a different model architecture")
        model.load_prepared_weights()
