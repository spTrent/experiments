from pathlib import Path
import ast
import contextlib
import gc
import io
import json
import tempfile
import time
import tokenize
import zipfile
import joblib
import nbformat
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import optuna
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]


def check(condition, message):
    if not condition:
        raise ValueError(message)


class PredictionStub:
    def predict(self, matrix):
        return np.full(len(matrix),0.001,dtype=np.float32)


notebook = nbformat.read(ROOT / 'notebooks/features/random_forest_feature_experiments.ipynb',as_version=4)
nbformat.validate(notebook)
for cell in notebook.cells:
    if cell.cell_type!='code':
        continue
    tree = ast.parse(cell.source)
    compile(tree,'<cell>','exec')
    check(not any(isinstance(n,ast.Assert) for n in ast.walk(tree)), 'assert')
    check(not any(t.type==tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(cell.source).readline)), 'comment')
    check(not any(ast.get_docstring(n) for n in ast.walk(tree) if isinstance(n,(ast.Module,ast.FunctionDef,ast.ClassDef))), 'docstring')
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    times = pd.date_range('2025-07-28','2025-11-01 10:30',freq='30min')
    frame = pd.DataFrame(dict(timestamp=times.repeat(3),route_id=np.tile([1,2,3],len(times))))
    rng = np.random.default_rng(42)
    for name in ['target_1h',*[f'status_{i}' for i in range(1,7)]]:
        frame[name] = rng.uniform(10,100,len(frame)).astype(np.float32)
    path = root/'train.parquet'
    frame.to_parquet(path,index=False)
    scope = dict(np=np,pd=pd,pa=pa,pq=pq,Path=Path,json=json,gc=gc,joblib=joblib,optuna=optuna,time=time,
                 zipfile=zipfile,SimpleImputer=SimpleImputer,display=lambda value:None,TARGET='target_1h',
                 STATUSES=[f'status_{i}' for i in range(1,7)],STEP=pd.Timedelta(minutes=30),TARGET_SCALE=100000,
                 TRAIN_MINUTES=(570,600,630,660,690),WINDOW_OFFSETS=tuple(range(-4,5)),LAG_MAX_LOOKBACK_HOURS=24,
                 LAG_MIN_PAIRS=336,LAG_MIN_CORRELATION=0.05,FEATURE_PIPELINE_VERSION='test',TRAIN_PATH=path,
                 ALGORITHM='random_forest',RESIDUAL=False,SEED=42,N_TRIALS=1,N_STREAMS=4,RF_TREES=[150,300],
                 RF_DEPTHS=[6,8,10,12,16],PREDICTION_BATCH=128,DEVICE='cuda',RUN_CONTRACT={},HORIZONS=[1],
                 VARIANTS=['direct','residual'],BASE_PARAMS=dict(n_estimators=150,max_depth=10,min_samples_leaf=32,max_features=0.6,max_samples=0.8))
    for name in ('RUN_DIR','OUT','PREPARED','DOCS_DIR','PARAMS_DIR','POSTPROCESS_DIR','MODEL_DIR'):
        scope[name] = root/name
        scope[name].mkdir()
    with contextlib.redirect_stdout(io.StringIO()):
        exec(notebook.cells[3].source,scope)
        scope['values'] = {name: values.copy() for name,values in scope['values'].items()}
        for index in (5,7):
            exec(notebook.cells[index].source,scope)
        feature_cell = notebook.cells[9].source
        prefix = feature_cell[:feature_cell.index('for h in range(1, 9):\n    write_horizon(h)')]
        exec(prefix,scope)
        origin = np.array([times.get_loc(pd.Timestamp('2025-09-15 10:30'))])
        before = scope['horizon_dataset'](8,origin)
        for values in scope['values'].values():
            values[origin[0]+1:] += 10000
        after = scope['horizon_dataset'](8,origin)
        check(np.allclose(before.select_dtypes('number'),after.select_dtypes('number'),equal_nan=True),'Future leakage')
        for values in scope['values'].values():
            values[origin[0]+1:] -= 10000
        scope['write_horizon'](1)
        exec(notebook.cells[11].source,scope)
    model_tree = ast.parse(notebook.cells[13].source)
    exec(compile(ast.Module(body=[n for n in model_tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'<model functions>','exec'),scope)
    scope['BASE_PARAMS_BY_HORIZON'] = {h: dict(scope['BASE_PARAMS']) for h in range(1,9)}
    scope['base_params_for_horizon'] = lambda h: dict(scope['BASE_PARAMS_BY_HORIZON'][h])
    fits=[]
    def fake_fit(frame,matrix,params):
        check(np.isfinite(matrix).all(),'Nonfinite matrix')
        fits.append(frame.timestamp.max())
        return PredictionStub()
    scope['fit_forest']=fake_fit
    experiment_source=notebook.cells[15].source
    exec(experiment_source[:experiment_source.index('unknown = set(')],scope)
    scope['EXPERIMENT_NAMES']=['baseline','thin_offsets','compact_with_additions']
    numeric_names=list(before.columns)
    baseline=scope['experiment_columns'](numeric_names,{})
    check(not any(n.startswith('x_') for n in baseline),'Baseline changed')
    for group in ('time','profile','level','anomaly','trend','volatility','dispersion'):
        check(any(n.startswith(f'x_{group}_') for n in before.columns),'Missing feature group')
    with contextlib.redirect_stdout(io.StringIO()):
        exec(experiment_source[experiment_source.index('unknown = set('):],scope)
    check(len(scope['folds'])==6,'Missing additional tuning windows')
    check(scope['folds'][-1]['tuning'][-1]==pd.Timestamp('2025-10-10 10:30'),'Wrong tuning end')
    expected_fits = len(scope['VARIANTS'])*len(scope['HORIZONS'])*len(scope['folds'])*(len(scope['EXPERIMENT_NAMES'])+scope['N_TRIALS'])
    check(len(fits)==expected_fits,'Wrong fit count for comparison + tuning')
    for chosen in scope['all_selected'].values():
        check(chosen[1]['params']==scope['BASE_PARAMS'],'First trial is not BASE_PARAMS')
    check((scope['RUN_DIR']/'all_parameters_frozen.json').exists(),'Parameters not frozen')
    with contextlib.redirect_stdout(io.StringIO()):
        exec(notebook.cells[17].source,scope)
        exec(notebook.cells[17].source,scope)
    check(len(fits)==expected_fits+2,'Evaluation resume failed')
    check(all(t<pd.Timestamp('2025-10-18') for t in fits),'Test labels in training')
    summary=pd.read_csv(scope['RUN_DIR']/'feature_experiments_summary.csv')
    check(len(summary)==6,'Missing experiments')
    for variant in scope['VARIANTS']:
        scope['select_context'](variant,scope['winners'][variant])
        bundle=joblib.load(scope['MODEL_DIR']/'evaluation/h1/model.joblib')
        check(len(bundle['metadata']['selected_indices'])==len(bundle['metadata']['selected_features']),'Invalid saved feature selection')
        check((scope['DOCS_DIR']/'independent_metrics.csv').exists(),'Missing test metrics')
    with contextlib.redirect_stdout(io.StringIO()):
        exec(experiment_source[experiment_source.index('unknown = set('):],scope)
    check(len(fits)==expected_fits+2,'Comparison/search resume failed')
    for variant in scope['VARIANTS']:
        folder=scope['RUN_DIR']/variant/'docs'
        check((folder/'base_vs_optuna_summary.csv').exists(),'Missing parameter comparison')
        details=pd.read_csv(folder/'base_vs_optuna_by_fold.csv')
        check(details.fold.nunique()==6,'Incomplete parameter comparison')
print('OK: 3 candidates x 2 variants, 6 temporal windows, BASE_PARAMS first trial, parameter diagnostics, causal features, evaluation and resume; prediction stub only, no RF training')
