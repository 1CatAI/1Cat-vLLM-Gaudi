# SPDX-License-Identifier: Apache-2.0
"""Per-worker final symbol graphs without shared JSON or DUMP environment keys."""
import os
from pathlib import Path


def prepare_physical_audit(args, rank):
    """Resolve CLI paths before changing cwd; Bridge uses its relative default.

    GRAPH_VISUALIZATION emits independent symbol graphs. Do not set a graph
    dump prefix: its default .graph_dumps lives under this worker's SSD cwd.
    This avoids rank collisions and the shared post-graph JSON truncation path.
    """
    if os.environ.get('GRAPH_VISUALIZATION') != '1':
        return None
    inherited = [key for key in os.environ if 'DUMP' in key]
    if inherited:
        raise RuntimeError(f'Physical audit forbids DUMP environment settings: {inherited}')
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    directory = args.output / 'graphs' / f'rank{rank}'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / '.graph_dumps').mkdir(exist_ok=True)
    # Cold physical evidence must not race another rank's shared recipe cache:
    # a cache hit has no post-graph artifact in this rank's working directory.
    cache = args.output / 'physical-recipes' / f'rank{rank}'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ['PT_HPU_RECIPE_CACHE_CONFIG'] = f'{cache},false,8192'
    os.chdir(directory)
    return directory
