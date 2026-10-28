#!/usr/local/bin/python
"""Smoke test of a bombe container (measurements-gpu/bombe_container.sh); exits non-zero on the first failure.

Checks that the one GPU is visible to torch and ultralytics, polyis and every compiled Cython module under polyis/
import, the mounts are where the pipeline expects them, and Gurobi optimizes a tiny model through gurobipy and through
p022's C solver (both read the WLS license at /opt/gurobi/gurobi.lic).
"""
import importlib
import os
from pathlib import Path

import numpy as np


def main():
    # torch must see exactly the one GPU the launcher passed.
    import torch
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1, f'GPUs visible: {torch.cuda.device_count()}'
    print('torch', torch.__version__, torch.cuda.get_device_name(0), 'physical GPU', os.environ.get('BOMBE_GPU'))
    # A matmul on the GPU proves kernels run, not only that the device enumerates.
    x = torch.ones(256, 256, device='cuda')
    assert float((x @ x).sum()) == 256 ** 3
    # ultralytics resolves its default device from torch; run one YOLO forward pass on the GPU with untrained weights.
    import ultralytics
    from ultralytics import YOLO
    model = YOLO('yolo11n.yaml')
    model.predict(np.zeros((64, 64, 3), dtype=np.uint8), device=0, verbose=False)
    print('ultralytics', ultralytics.__version__, 'predicted on', next(model.model.parameters()).device)
    # polyis and its config resolve (CONFIG is the scratch copy; CACHE_DIR must be the scratch cache mount).
    from polyis.utilities import get_config
    config = get_config()
    assert config['DATA']['CACHE_DIR'].rstrip('/') == '/polyis-cache', config['DATA']['CACHE_DIR']
    print('polyis config', os.environ.get('CONFIG'), 'datasets', config['EXEC']['DATASETS'])
    # Import every compiled extension under polyis/ by its module path.
    modules = sorted('.'.join(p.with_suffix('').with_suffix('').parts) for p in Path('polyis').rglob('*.so'))
    for name in modules:
        importlib.import_module(name)
    print('imported', len(modules), 'Cython modules')
    # duckdb comes from the mounted pylib (ace's container has it pip-installed; the image does not).
    import duckdb
    print('duckdb', duckdb.__version__)
    # The read-only inputs are mounted and non-empty; the cache mount is writable.
    for mount in ('/polyis-data/datasets', '/polyis-data/sources', '/otif-dataset/dataset', '/sota-results'):
        assert any(Path(mount).iterdir()), f'{mount} is empty'
    probe = Path('/polyis-cache/.bombe_check')
    probe.write_text('ok')
    probe.unlink()
    print('mounts ok')
    # gurobipy: a two-variable MIP must solve to its known optimum (x=1, y=1, objective 3).
    import gurobipy as gp
    with gp.Env(params={'OutputFlag': 0}) as env, gp.Model(env=env) as m:
        xv, yv = m.addVar(vtype='B'), m.addVar(vtype='B')
        m.setObjective(xv + 2 * yv, gp.GRB.MAXIMIZE)
        m.addConstr(xv + yv <= 2)
        m.optimize()
        assert m.Status == gp.GRB.OPTIMAL and round(m.ObjVal) == 3, (m.Status, m.ObjVal)
    print('gurobipy', gp.gurobi.version(), 'optimal')
    # p022's C solver on one tile covered in three frames: the first and last frames are always kept.
    from polyis.sample.ilp.c.gurobi import solve_ilp
    result = solve_ilp(np.zeros((3, 1, 1), dtype=np.int16), [[1], [1], [1]], np.ones((1, 1)), 1, 1, 0.1)
    assert {(0, 0), (2, 0)} <= set(map(tuple, result.selected)), result
    print('C solve_ilp selected', sorted(map(tuple, result.selected)), f'build {result.build_ms:.2f} ms')
    print('bombe check passed')


if __name__ == '__main__':
    main()
