import ast
import hashlib
import json
import tempfile
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

BASE = Path(__file__).resolve().parents[1]
FOLDER = BASE/'stacking/tuned_ridge'
for path in [BASE/'stacking/ridge_tuned_local.ipynb',*FOLDER.glob('*.ipynb')]:
    nb = nbformat.read(path,as_version=4)
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type=='code':
            compile(cell.source,str(path),'exec')
            tree = ast.parse(cell.source)
            for node in tree.body:
                if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='ENGINE.append':
                    compile(ast.literal_eval(node.value.args[0]),str(path),'exec')
nb = nbformat.read(BASE/'stacking/ridge_tuned_local.ipynb',as_version=4)
ns = {}
exec(nb.cells[1].source,ns)
ns.update(DEVICE='cpu',HORIZONS=(1,),BATCH_SIZE=2)
exec(nb.cells[2].source,ns)
exec(nb.cells[3].source,ns)
exec(nb.cells[8].source,ns)
ns.update(xp=np,BACKEND='cpu',cp=None,cupyx=None)
rng = np.random.default_rng(25)
items = []
for width in (5,7):
    x = rng.normal(size=(45,width))
    q = rng.normal(size=(7,width))
    mean = x.mean(axis=0)
    y = 10+x@rng.normal(size=width)
    actual = np.zeros(7) if width==7 else rng.uniform(1,20,7)
    items.append(dict(x=x-mean,query_x=q-mean,y=y,y_mean=y.mean(),query_y=actual,
                      original_train=np.ones(len(y),dtype=bool),phase='tuning'))
scores,zero = ns['tune_items'](items)
for r,item in enumerate(items):
    expected = []
    for alpha in ns['ALPHAS']:
        pred = np.maximum(0.,Ridge(alpha=alpha,solver='cholesky').fit(item['x'],item['y']).predict(item['query_x']))
        err = pred-item['query_y']
        volume = item['query_y'].sum()
        expected.append((np.abs(err).sum()+abs(err.sum()))/volume if volume>0 else np.abs(err).mean())
    np.testing.assert_allclose(scores[r],expected,rtol=1e-7,atol=1e-8)
if zero.tolist()!=[False,True] or len(ns['ALPHAS'])!=30 or 100 not in ns['ALPHAS']:
    raise ValueError('Wrong tuning grid or zero-volume fallback')
original_tune = ns['tune_items']
def limited(items):
    if len(items)>1:
        raise MemoryError('Synthetic memory limit')
    return original_tune(items)
ns['tune_items'] = limited
np.testing.assert_allclose(ns['retry_compute'](items)[0],scores)
ns['tune_items'] = original_tune
pred,metrics = ns['fit_items'](items,np.array([1.,100.]),True)
if pred.shape!=(2,7):
    raise ValueError('Wrong batch prediction shape')

with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    input_dir = root/'data'
    input_dir.mkdir()
    times = pd.date_range('2025-07-28','2025-11-01 10:30',freq='30min')
    route_ids = np.arange(4)
    idx = np.arange(len(times)*len(route_ids))
    frame = pd.DataFrame(dict(route_id=np.tile(route_ids,len(times)),timestamp=times.repeat(len(route_ids))))
    frame['target_1h'] = 100+20*np.sin(idx/43)+np.tile(route_ids,len(times))*10
    for i in range(1,7):
        frame[f'status_{i}'] = 50+i+5*np.sin(idx/(17+i))
    frame.to_parquet(input_dir/'train_solo_track.parquet',index=False)
    test = pd.DataFrame([(r,times[-1]+pd.Timedelta(minutes=30*h)) for r in route_ids for h in range(1,9)],columns=['route_id','timestamp'])
    test['id'] = np.arange(len(test))
    test.to_parquet(input_dir/'test_solo_track.parquet',index=False)
    ns.update(INPUT_DIR=str(input_dir),OUTPUT_DIR=str(root/'output'),CACHE_DIR=str(root/'cache'))
    ns['local_initialize']()
    for i in (5,6,7,9):
        exec(nb.cells[i].source,ns)
    ns['display'] = lambda *args: None
    ns['timed_log'] = lambda *args,**kwargs: None
    from types import SimpleNamespace
    ns['psutil'] = SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=4*2**30))
    ns['run_local']()
    local_root = ns['LOCAL_ROOT']
    alpha = pd.read_parquet(local_root/'h1/alphas.parquet')
    if len(alpha)!=4:
        raise ValueError('Missing selected alphas')
    for stage in ('oof2','oof3','oof4'):
        path = local_root/f'predictions/h1/{stage}.parquet'
        p = pd.read_parquet(path)
        info = json.loads(path.with_suffix('.json').read_text())
        if len(p)!=28 or not pd.Timestamp(info['fit_through'])<pd.Timestamp(info['boundary']):
            raise ValueError('Invalid local OOF')
    original_horizon = ns['HorizonData']
    ns['HorizonData'] = lambda *args: (_ for _ in ()).throw(RuntimeError('Completed horizon repeated'))
    ns['run_local']()
    ns['HorizonData'] = original_horizon
    for mode in ('evaluate','submit'):
        ns['MODE'] = mode
        ns['run_local']()
    if len(pd.read_parquet(local_root/'predictions/h1/full.parquet'))!=4:
        raise ValueError('Missing local final forecasts')
    if any((root/'cache').iterdir()):
        raise ValueError('Temporary feature cache not cleaned')
    for part in range(1,5):
        bridge_nb = nbformat.read(FOLDER/f'stacking_tuned_part_{part}.ipynb',as_version=4)
        original_nb = nbformat.read(BASE/f'stacking/stacking_part_{part}.ipynb',as_version=4)
        original_settings,bridge_settings = {},{}
        exec(original_nb.cells[1].source,original_settings)
        exec(bridge_nb.cells[1].source,bridge_settings)
        if bridge_settings['CODE_ID']!=original_settings['CODE_ID'] or bridge_settings['PARAMS']!=original_settings['PARAMS']:
            raise ValueError('Bridge invalidates existing base predictions')
    bridge_nb = nbformat.read(FOLDER/'stacking_tuned_part_1.ipynb',as_version=4)
    bridge_ns = {}
    exec(bridge_nb.cells[1].source,bridge_ns)
    for cell in bridge_nb.cells[2:-1]:
        exec(cell.source,bridge_ns)
    for source in bridge_ns['ENGINE']:
        stripped = source.lstrip()
        if source.strip()=='initialize()' or stripped.startswith('history =') or stripped.startswith('target =') or stripped.startswith("if MODE=='oof':"):
            continue
        exec(source,bridge_ns)
    if len(list(bridge_ns['specifications']()))!=28 or len(bridge_ns['predictors'](True))!=13:
        raise ValueError('Bridge does not compare all route variants')
    for stage in ('oof2','oof3','oof4','test','extra','full'):
        boundary,days = ns['stage_spec'](stage)
        query = ns['query_indices'](boundary,days)
        for h in range(2,9):
            out = ns['result_frame'](h,query).assign(prediction=1.)
            path = local_root/f'predictions/h{h}/{stage}.parquet'
            path.parent.mkdir(parents=True,exist_ok=True)
            out.to_parquet(path,index=False)
            path.with_suffix('.json').write_text(json.dumps(dict(run_id=ns['MANIFEST']['run_id'],base_code_id=ns['BASE_CODE_ID'],boundary=str(boundary),fit_through=str(boundary-pd.Timedelta(minutes=30)),tuning_end='2025-09-20')))
    bridge_ns.update(INPUT=root,TRAIN_PATH=ns['TRAIN_PATH'],TEST_PATH=ns['TEST_PATH'],
                     times=ns['times'],routes=ns['routes'],target=ns['target'])
    def collect_original(stage,cols):
        query = ns['query_indices'](*ns['stage_spec'](stage))
        frames = [ns['result_frame'](h,query) for h in range(1,9)]
        return pd.concat(frames,ignore_index=True).assign(**{c:1. for c in cols})
    bridge_ns['_original_collect_stage'] = collect_original
    joined = bridge_ns['collect_stage']('oof2',bridge_ns['predictors'](True))
    if len(joined)!=224 or joined[ns['TUNED_COLUMN']].isna().any():
        raise ValueError('Bridge join failed')
    chosen = local_root/'predictions/h1/oof2.parquet'
    old = pd.read_parquet(chosen)
    old.iloc[:-1].to_parquet(chosen,index=False)
    try:
        bridge_ns['collect_stage']('oof2',bridge_ns['predictors'](True))
    except ValueError:
        pass
    else:
        raise ValueError('Bridge accepted a missing prediction')
    old.assign(y=old.y+1).to_parquet(chosen,index=False)
    try:
        bridge_ns['collect_stage']('oof2',bridge_ns['predictors'](True))
    except ValueError:
        pass
    else:
        raise ValueError('Bridge accepted mismatched targets')
    if bridge_ns['profile_path']('meta/winner.json')!='meta_tuned/winner.json':
        raise ValueError('Original meta outputs overwritten')
    captured = []
    bridge_ns['_original_jobs'] = lambda h,s,cols: captured.extend(cols) or False
    bridge_ns['jobs'](1,'test',bridge_ns['predictors'](True))
    if ns['TUNED_COLUMN'] in captured:
        raise ValueError('Kaggle worker tries to refit external Ridge')
print('PASS: all notebook syntax/schema; spectral alpha grid vs sklearn; zero-volume fallback; memory retry; real feature pipeline on synthetic panel; local OOF/test/extra/full; restart; cache cleanup; unchanged base CODE_ID; bridge joins and target validation.')
print('CPU synthetic execution only; no NVIDIA GPU available in this validation environment.')
