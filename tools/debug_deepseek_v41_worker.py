# SPDX-License-Identifier: Apache-2.0
"""Run an owned torchrun diagnostic worker under GDB and archive native stacks."""

import os
from pathlib import Path
import sys


def main():
    rank = int(os.environ["LOCAL_RANK"])
    directory = Path(os.environ["DSV41_RUN_EVIDENCE"])
    commands = directory / f"gdb-rank{rank}.commands"
    commands.write_text("\n".join((
        "set pagination off",
        "set confirm off",
        "set print thread-events off",
        "handle SIGPIPE nostop noprint pass",
        "python",
        "import gdb",
        "def record_exit(event):",
        "    gdb.set_convenience_variable('dsv41_exit', getattr(event, 'exit_code', 1))",
        "gdb.events.exited.connect(record_exit)",
        "end",
        "run",
        "python",
        "if not gdb.selected_inferior().pid:",
        "    gdb.execute('quit ' + str(int(gdb.parse_and_eval('$dsv41_exit'))))",
        "end",
        "bt 40",
        "info registers",
        "x/24i $pc-32",
        "bt full 12",
        "thread apply all bt 6",
        "quit 128",
    )) + "\n")
    log = (directory / f"gdb-rank{rank}.log").open("w")
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    os.execvp("gdb", ["gdb", "--batch", "-x", str(commands), "--args", sys.executable, *sys.argv[1:]])


if __name__ == "__main__":
    main()
