# SPDX-License-Identifier: Apache-2.0
"""Immutable bridge artifacts survive output-directory changes before card use."""
import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[3]

@pytest.mark.parametrize('missing',[False,True])
def test_bridge_is_preserved_or_rejected_before_device_use(tmp_path,monkeypatch,missing):
    monkeypatch.syspath_prepend(str(ROOT/'tools'))
    spec=importlib.util.spec_from_file_location('component_launcher',ROOT/'tools/launch_deepseek_v41_component.py')
    launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
    parent=tmp_path/'old_case';parent.mkdir();bridge=parent/'bridge.so'
    if not missing:bridge.write_bytes(b'pinned-native-bridge')
    native=tmp_path/'native';native.mkdir()
    for name in ('hpu_dsv4_sparse_attn_pt2.cpython-312-x86_64-linux-gnu.so','libdeepseek_v4_gaudi2_kernels.so'):
        (native/name).write_bytes(b'library')
    sidecar=tmp_path/'sidecar';sidecar.mkdir();(sidecar/'manifest.json').write_text('{}')
    profile=tmp_path/'profile.json';profile.write_text(json.dumps({'command':[
        'python','--module','tools.check_deepseek_v41_index_gain_replica','--output',str(parent),
        '--sidecar',str(sidecar),'--master_port=29512'],
        'environment':{'VLLM_HPU_TP2_FUSED_AR_NORM_BRIDGE':str(bridge),'TMPDIR':str(tmp_path/'scratch')}}))
    case=tmp_path/'new_case'
    monkeypatch.setattr(sys,'argv',['component','--template',str(profile),'--native-dir',str(native),
        '--sidecar',str(sidecar),'--output',str(case),'--module','tools.check_deepseek_v41_index_gain_replica',
        '--port','29988'])
    monkeypatch.setattr(launcher.shutil,'copytree',lambda source,destination,**kwargs:Path(destination).mkdir())
    monkeypatch.setattr(launcher,'wait_for_host_memory',lambda *a,**kw:None)
    touched=[]
    @contextmanager
    def lease(*args,**kwargs):
        touched.append('leased');yield ((2,3,6,7),'')
    monkeypatch.setattr(launcher,'lease_free_modules',lease)
    def spawn(command,**kwargs):
        assert kwargs['env']['VLLM_HPU_TP2_FUSED_AR_NORM_BRIDGE']==str(bridge)
        assert kwargs['env']['HABANA_LOGS']==str(case/'habana_logs')
        touched.append('spawned');return SimpleNamespace(pid=991991)
    monkeypatch.setattr(launcher.subprocess,'Popen',spawn)
    monkeypatch.setattr(launcher,'wait_for_owned_process',lambda *a,**kw:0)
    monkeypatch.setattr(launcher,'retire_process_group',lambda *a,**kw:None)
    if missing:
        with pytest.raises(RuntimeError,match='before device lease'):launcher.main()
        assert touched==[]
    else:
        with pytest.raises(SystemExit) as exit_info:launcher.main()
        assert exit_info.value.code==0 and touched==['leased','spawned']
        launch=json.loads((case/'launch.json').read_text())
        assert launch['environment']['VLLM_HPU_TP2_FUSED_AR_NORM_BRIDGE']==str(bridge)
