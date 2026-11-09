import argparse
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ.get('RWB_ROOT',Path(__file__).resolve().parent))
inputs = root / 'input'
parser = argparse.ArgumentParser()
parser.add_argument('--files-only',action='store_true')
args = parser.parse_args()
identity = ['c9075a4e3ad5ce73e0de4f84bc82fa50836df388d8a65c54e9a6b0db72dc601b','five_starts','100.0']
for name in ('train_solo_track.parquet','test_solo_track.parquet'):
    hits = list(inputs.rglob(name))
    if len(hits)!=1:
        raise ValueError(f'Ожидается один {name}, найдено {hits}')
    with hits[0].open('rb') as stream:
        identity.append(hashlib.file_digest(stream,'sha256').hexdigest())
run = 'stack_'+hashlib.sha256('|'.join(identity).encode()).hexdigest()[:24]
assembly = inputs/'stack-assemble'/run/'meta_tuned'
winner = json.loads((assembly/'winner.json').read_text())
if winner['route_variant']!='fixed' or winner['kind']!='gate' or winner['epochs']!=50:
    raise ValueError(f'Иной победитель: {winner}')
for name in ('oof.parquet','calibration_history.parquet'):
    if not (assembly/name).is_file():
        raise FileNotFoundError(assembly/name)
for part,hs in enumerate(((1,8),(2,7),(3,6),(4,5)),1):
    path = inputs/'stack_oof'/f'part{part}'/run
    for h in hs:
        for week in (2,3,4):
            for col in winner['columns']:
                regime,name = col.split('__') if '__' in col else ('shared',col)
                data = path/f'h{h}'/f'oof{week}'/regime/(name+'.parquet')
                if not data.is_file() or not Path(str(data)+'.json').is_file():
                    raise FileNotFoundError(data)
for h in (1,8):
    for col in winner['columns']:
        regime,name = col.split('__') if '__' in col else ('shared',col)
        path = inputs/'evaluate-part1'/run/f'h{h}'/'test'/regime/(name+'.parquet')
        if not path.is_file() or not Path(str(path)+'.json').is_file():
            raise FileNotFoundError(path)
print(f'Входы совместимы: {run}. Часть 1 готова.',flush=True)
if not args.files_only:
    import cupy as cp
    import numpy as np
    import pandas
    import pyarrow
    import sklearn
    import IPython
    import lightgbm
    from catboost import CatBoostRegressor
    from cuml.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    cp.linalg.solve(cp.eye(2),cp.ones(2))
    x=np.arange(256,dtype='float32').reshape(64,4)
    y=x[:,0].copy()
    RandomForestRegressor(n_estimators=2,max_depth=2,n_streams=1).fit(x,y)
    CatBoostRegressor(iterations=2,task_type='GPU',allow_writing_files=False,verbose=False).fit(x,y)
    model=XGBRegressor(n_estimators=2,tree_method='hist',device='cuda').fit(x,y)
    if not json.loads(model.get_booster().save_config())['learner']['generic_param']['device'].startswith('cuda'):
        raise RuntimeError('XGBoost не использует CUDA')
    lightgbm.LGBMRegressor(n_estimators=2,verbosity=-1).fit(x,y)
    print('GPU-проверка пройдена:',cp.cuda.runtime.getDeviceProperties(0)['name'])
