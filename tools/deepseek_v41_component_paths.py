# SPDX-License-Identifier: Apache-2.0
"""Short, distinct scratch paths for component IPC sockets."""
import hashlib
import os
from pathlib import Path


def component_scratch(case: Path, template_tmpdir: str | None, root: Path | None = None) -> Path:
    case = case.resolve()
    if root is None:
        root = Path(template_tmpdir).parent if template_tmpdir else case.parent / "micro-tmp"
    scratch = root.resolve() / hashlib.sha256(os.fsencode(case)).hexdigest()[:8]
    # vLLM appends a UUID to TMPDIR for its Unix-domain message queue socket.
    if len(os.fsencode(scratch / ("0" * 36))) > 107:
        raise ValueError("IPC scratch path exceeds sockaddr_un: select a short SSD path with --ipc-tmp-root")
    return scratch
