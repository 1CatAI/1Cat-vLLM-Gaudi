# SPDX-License-Identifier: Apache-2.0
"""Save the first changed rounding element, without performance measurement."""
import json
import os
from pathlib import Path

os.environ['HLS_MODULE_ID'] = os.environ['HABANA_VISIBLE_MODULES'].split(',')[0]
from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment  # noqa: E402
prepare_environment()
import habana_frameworks.torch.core  # noqa: E402,F401
import torch  # noqa: E402
from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers  # noqa: E402
from check_deepseek_v41_mhc_post_collapse import check_mhc_post_collapse_contract  # noqa: E402

bind_worker_cpu(0)
torch.hpu.set_device(0)
bind_worker_helpers(0)
torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])
root = Path(os.environ['DSV41_RUN_EVIDENCE'])
with torch.inference_mode():
    try:
        check_mhc_post_collapse_contract(root / 'first-difference.pt')
    except AssertionError as error:
        assert (root / 'first-difference.pt').exists()
        (root / 'result.json').write_text(json.dumps(dict(status='rounding_difference_reproduced',
                                                         error=str(error), performance_measured=False), indent=2)+'\n')
        print('saved first rounding difference; no timing', flush=True)
    else:
        raise RuntimeError('The archived numerical failure was not reproduced')
