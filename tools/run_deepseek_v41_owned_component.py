# SPDX-License-Identifier: Apache-2.0
"""Use the fixed leased launcher and retire exactly the completed component."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--script',type=Path,required=True)
    parser.add_argument('--modules',default='2,6,7,3')
    parser.add_argument('--status-prefix',required=True)
    parser.add_argument('--success-status',required=True)
    parser.add_argument('worker_arguments',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    workspace=Path(__file__).resolve().parents[1]
    run=args.run.resolve()
    if run.exists():
        raise FileExistsError('Use a new component run path')
    root=workspace.parent
    pool=set(range(10,20))|set(range(38,48))
    os.sched_setaffinity(0,pool)
    invocation=[sys.executable,str(workspace/'tools/run_deepseek_v41.py'),
                '--devices','4','--modules',args.modules,'--preferred-cpus','10-19,38-47',
                '--cpu-conflict-policy','relocate-or-measure',
                '--lock-dir',str(root/'locks'),
                '--secondary-lock-dir','/opt/ssd960/ymzx-dsv41-exact-full-stack-v1/locks',
                '--runtime-profile',str(args.profile.resolve()),
                '--engine-source',str(root/'builds/dsv41-tp4-dspark-main-v1/engine'),
                '--recipe-cache-dir','/opt/optane/dsv41-builds/dsv41-tp4-dspark-serving/owned-component-cache',
                str(run),'--',sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node','4',
                str(args.script),*([a for a in args.worker_arguments if a!='--'])]
    run.parent.mkdir(parents=True,exist_ok=True)
    worker=subprocess.Popen(invocation,cwd=workspace)
    retired=False
    failure=None
    try:
        while worker.poll() is None:
            paths=[run/f'{args.status_prefix}-rank{rank}.json' for rank in range(4)]
            try:
                reports=[json.loads(p.read_text()) for p in paths if p.exists()]
                failed=any(row.get('status')=='failed' for row in reports)
                done=len(reports)==4 and all(row.get('status')==args.success_status for row in reports)
            except json.JSONDecodeError:
                failed=done=False
            if (failed or done) and not retired and (run/'process.json').exists():
                metadata=json.loads((run/'process.json').read_text())
                pid,pgid=metadata['pid'],metadata['pgid']
                try:
                    command=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
                    environment=dict(item.split(b'=',1)
                                     for item in Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
                                     if b'=' in item)
                    if (os.getpgid(pid)!=pgid or pid!=pgid or os.fsencode(args.script) not in command
                            or environment.get(b'DSV41_RUN_EVIDENCE')!=os.fsencode(run)):
                        raise RuntimeError('Completed component ownership does not match its launch manifest')
                    os.killpg(pgid,signal.SIGTERM)
                    retired=True
                    failure='Component correctness failed' if failed else None
                except ProcessLookupError:
                    retired=True
            time.sleep(1)
        if failure or (worker.returncode and not retired):
            raise RuntimeError(failure or f'Component launcher exited {worker.returncode}')
    finally:
        if worker.poll() is None:
            worker.terminate()
            worker.wait(timeout=60)
    print(json.dumps(dict(run=str(run),retired=retired,status='completed'),indent=2))


if __name__=='__main__':
    main()
