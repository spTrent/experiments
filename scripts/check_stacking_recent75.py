import ast
import json
import subprocess
import tempfile
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
config = json.loads((BASE/'scripts/stacking/model_params.json').read_text())
seen = []
for path in sorted((BASE/'stacking').glob('stacking_part_*.ipynb')):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    setup = {}
    exec(notebook.cells[1].source, setup)
    if setup['PROFILE'] != 'five_starts' or setup['OOF_WEEKS'] != (2,3,4):
        raise ValueError('Wrong default protocol')
    seen.extend(setup['HORIZONS'])
    old = json.loads(subprocess.check_output(['git','show',f'HEAD:stacking/{path.name}'], cwd=BASE))
    old_setup = {}
    exec(''.join(old['cells'][1]['source']), old_setup)
    if setup['PARAMS'] != old_setup['PARAMS']:
        raise ValueError('Parameters changed')
    engine = []
    for cell in notebook.cells[2:]:
        if cell.cell_type != 'code':
            continue
        compile(cell.source, str(path), 'exec')
        tree = ast.parse(cell.source)
        if len(tree.body)==1 and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Call):
            call = tree.body[0].value
            if ast.unparse(call.func)=='ENGINE.append':
                source = ast.literal_eval(call.args[0])
                compile(source, str(path), 'exec')
                engine.append(source)
    if not engine:
        raise ValueError('Missing embedded code')
if sorted(seen) != list(range(1,9)):
    raise ValueError('Wrong horizon allocation')
nbformat.validate(nbformat.read(BASE/'stacking/rf_recent75_benchmark.ipynb', as_version=4))
ns = dict(PROFILE='five_starts')
for name in ('runtime.py','worker.py','calibration.py','meta.py'):
    exec((BASE/'scripts/stacking'/name).read_text(), ns)
with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    ns['ROOT'] = root/'current'
    checkpoint = root/'saved'
    checkpoint.mkdir()
    ns['CHECKPOINTS'] = [checkpoint]
    (checkpoint/'finished.parquet').write_bytes(b'finished prediction artifact')
    (checkpoint/'finished.parquet.json').write_text('{}')
    (checkpoint/'unfinished.parquet').write_bytes(b'no completion manifest')
    if not ns['complete']('finished.parquet') or ns['complete']('unfinished.parquet'):
        raise ValueError('Checkpoint completion detection failed')
    if not (ns['ROOT']/'finished.parquet').is_file():
        raise ValueError('Checkpoint was not restored')
ns.update(times=pd.date_range('2025-07-28','2025-11-01 10:30',freq='30min'), routes=np.arange(1000))
ns['target'] = np.ones((len(ns['times']),1000), dtype=np.float32)
ns['values'] = {name:ns['target'] for name in [ns['TARGET'],*ns['STATUSES']]}
ns['display'] = lambda *args: None
for week, expected in [(2,'2025-09-27'),(3,'2025-10-04'),(4,'2025-10-11')]:
    boundary, days = ns['stage_spec'](f'oof{week}')
    if boundary != pd.Timestamp(expected) or days != 7:
        raise ValueError('Wrong OOF boundary')
    for h in range(1,9):
        times = ns['times']
        possible = np.flatnonzero(times[:-h]+h*ns['STEP'] < boundary)
        dates = times[possible].normalize().unique()
        expected_indices = possible[times[possible].normalize().isin(dates[len(dates)//4:])]
        actual = ns['train_indices'](h,boundary,'recent_all')
        np.testing.assert_array_equal(actual,expected_indices)
        five = ns['train_indices'](h,boundary,'five_starts')
        expected_five = possible[np.isin(times[possible].hour*60+times[possible].minute,ns['TRAIN_MINUTES'])]
        np.testing.assert_array_equal(five,expected_five)
        if not (times[actual]+h*ns['STEP'] < boundary).all():
            raise ValueError('Label leakage')
if len(ns['predictors'](True)) != 12 or len(ns['predictors'](False)) != 11:
    raise ValueError('Wrong predictor count')
specs = list(ns['specifications']())
if len(specs)!=14 or {spec['kind'] for spec in specs}!={'weighted','catboost','gate'}:
    raise ValueError('Full meta-model search must remain enabled')
columns = ns['predictors'](True)
rng = np.random.default_rng(42)
small = pd.DataFrame([(day+pd.Timedelta(hours=10,minutes=30),route,h)
                     for day in pd.date_range('2025-10-04',periods=7)
                     for route in range(12) for h in range(1,9)],columns=['cutoff','route_id','horizon'])
small['timestamp'] = small.cutoff+pd.to_timedelta(small.horizon*30,unit='min')
small['raw_prediction'] = rng.uniform(10,1000,len(small))
small['y'] = 1.2*small.raw_prediction+20
for col in columns:
    small[col] = small.raw_prediction
query = small.copy()
query['cutoff'] += pd.Timedelta(days=7)
query['timestamp'] += pd.Timedelta(days=7)
for spec in ns['post_calibration_specs']():
    identity = ns['fit_post_calibration'](spec,small.iloc[:0],columns)
    np.testing.assert_array_equal(ns['apply_post_calibration'](identity,query,query.raw_prediction),query.raw_prediction)
    state = ns['fit_post_calibration'](spec,small,columns)
    prediction = ns['apply_post_calibration'](state,query,query.raw_prediction)
    np.testing.assert_array_equal(prediction,ns['apply_post_calibration'](state,query.assign(y=1e12),query.raw_prediction))
    if not np.isfinite(prediction).all() or (prediction<0).any():
        raise ValueError('Invalid calibration output')
    unseen = query.assign(route_id=999,horizon=9)
    if not np.isfinite(ns['apply_post_calibration'](state,unseen,unseen.raw_prediction)).all():
        raise ValueError('Unseen calibration group has no fallback')
    if spec['kind']=='affine' and np.mean(np.abs(prediction-query.y))>=np.mean(np.abs(query.raw_prediction-query.y)):
        raise ValueError('Affine calibration did not correct synthetic affine bias')
    if spec['kind']=='nonlinear':
        order = np.argsort(query.raw_prediction.to_numpy())
        if (np.diff(prediction[order]) < -1e-9).any():
            raise ValueError('Nonlinear calibration is not monotone')
    import joblib
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)/'calibration.joblib'
        joblib.dump(state,path)
        np.testing.assert_allclose(ns['apply_post_calibration'](joblib.load(path),query,query.raw_prediction),prediction)
feature_ns = dict(ns)
exec((BASE/'scripts/stacking/bases.py').read_text(), feature_ns)
feature_ns['horizon_dataset'] = lambda h,idx: ns['result_frame'](h,idx)[['cutoff','timestamp','route_id']].assign(signal=1.,profile28=1.)
with tempfile.TemporaryDirectory() as folder:
    data = feature_ns['HorizonData'](folder,1,96)
    if len(data.origins)!=10 or data.frame.shape!=(10,1000,1):
        raise ValueError('Five-start feature cache includes unwanted origins')
    if (data.slots[ns['query_indices'](pd.Timestamp('2025-07-28'),2)]<0).any():
        raise ValueError('Validation origins are missing from sparse feature cache')
    data.corr = np.zeros((10,1000,1),dtype=np.float32)
    data.corr_names = ['correlation']
    if data.matrix(data.origins).shape!=(10000,2):
        raise ValueError('Sparse feature matrix indexing failed')
    del data
store = {}
ns['save'] = lambda path, value: store.__setitem__(path,value)
ns['load_json'] = lambda path: store[path]
ns['require'] = lambda path: path
ns['existing'] = lambda path: None
ns['meta_predict'] = lambda spec,tr,va: (tr[spec['columns']].mean(axis=1).to_numpy(),va[spec['columns']].mean(axis=1).to_numpy())
def stage(stage, columns):
    boundary,days = ns['stage_spec'](stage)
    origins = ns['query_indices'](boundary,days)
    frame = pd.concat([ns['result_frame'](h,origins) for h in range(1,9)],ignore_index=True)
    return frame.assign(**{col:1. for col in columns},stage=stage,fit_before=boundary)
ns['collect_stage'] = stage
original_read = pd.read_parquet
try:
    pd.read_parquet = lambda path: store[path]
    with tempfile.TemporaryDirectory() as folder:
        ns['ROOT'] = Path(folder)
        ns['TEST_PATH'] = 'input_test'
        query = stage('full',[])
        store['input_test'] = query[['route_id','timestamp']].assign(id=np.arange(8000)).sample(frac=1,random_state=42)
        ns['assemble']()
        if len(store['meta/leaderboard.json'])!=32:
            raise ValueError('Calibration variants were removed')
        if len(store['meta/oof.parquet']) != 168000:
            raise ValueError('Wrong OOF rows')
        weeks = set(store['meta/validation.parquet'].week)
        if weeks != {3,4}:
            raise ValueError('Wrong validation weeks')
        for key,value in store.items():
            if key.startswith('meta/cv/post_') and key.endswith('.json') and 'identity' in value:
                week = int(key.rsplit('_',1)[-1].split('.')[0])
                if week==3 and not value['identity']:
                    raise ValueError('First validation week used future calibration data')
                if value['fit_through'] and pd.Timestamp(value['fit_through'])>=ns['stage_spec'](f'oof{week}')[0]:
                    raise ValueError('Calibration leaks future labels')
        ns['evaluate']()
        ns['submit']()
        result = pd.read_csv(Path(folder)/'submission.csv')
        np.testing.assert_array_equal(result.id,store['input_test'].id)
        if len(result)!=8000 or not np.isfinite(result.y_pred).all():
            raise ValueError('Invalid submission')
        original_winner = dict(store['meta/winner.json'])
        for kind in ('groups','affine','nonlinear','contextual'):
            store['meta/winner.json'] = dict(original_winner,post_calibration=dict(kind=kind,strength=0.5))
            ns['evaluate']()
            ns['submit']()
            if store['meta/test_post_calibration.json']['rows']!=112000 or store['meta/full_post_calibration.json']['rows']!=224000:
                raise ValueError('Final calibration history is incomplete or duplicates rows')
            result = pd.read_csv(Path(folder)/'submission.csv')
            np.testing.assert_array_equal(result.id,store['input_test'].id)
finally:
    pd.read_parquet = original_read
print('PASS: notebook schemas, embedded syntax, unchanged model parameters, recent75 masks, OOF dates, two temporal checks, assembly, evaluation and submission ordering')
print('Meta-model predictions and base artifacts were synthetic; no GPU training performed.')
