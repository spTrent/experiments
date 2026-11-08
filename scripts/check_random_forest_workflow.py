from pathlib import Path
import ast
import contextlib
import io
import json
import tempfile
import time
import zipfile
import gc
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import optuna
import nbformat
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]


def check(condition, message):
    if not condition:
        raise ValueError(message)


class PredictionStub:
    def predict(self, matrix):
        return np.full(len(matrix), 0.001, dtype=np.float32)


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    timestamps = pd.date_range('2025-07-28', '2025-11-01 10:30', freq='30min')
    route_ids = np.array([1, 2, 3])
    frame = pd.DataFrame({'timestamp': timestamps.repeat(3), 'route_id': np.tile(route_ids, len(timestamps))})
    rng = np.random.default_rng(42)
    for name in ['target_1h', *[f'status_{i}' for i in range(1, 7)]]:
        frame[name] = rng.uniform(10, 100, len(frame)).astype(np.float32)
    train_path = root / 'train.parquet'
    frame.to_parquet(train_path, index=False)
    direct = nbformat.read(ROOT / 'random_forest_direct.ipynb', as_version=4)
    residual = nbformat.read(ROOT / 'random_forest_residual.ipynb', as_version=4)
    for notebook in (direct, residual):
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                compile(cell.source, '<notebook>', 'exec')
    for index in range(2, len(direct.cells)):
        check(direct.cells[index].source == residual.cells[index].source, 'Ветки расходятся вне настроек')
    scope = dict(np=np, pd=pd, pa=pa, pq=pq, Path=Path, json=json, gc=gc, joblib=joblib, optuna=optuna,
                 time=time, zipfile=zipfile, SimpleImputer=SimpleImputer, display=lambda value: None,
                 TARGET='target_1h', STATUSES=[f'status_{i}' for i in range(1, 7)], STEP=pd.Timedelta(minutes=30),
                 TARGET_SCALE=100000, TRAIN_MINUTES=(570,600,630,660,690), WINDOW_OFFSETS=tuple(range(-4,5)),
                 LAG_MAX_LOOKBACK_HOURS=24, LAG_MIN_PAIRS=336, LAG_MIN_CORRELATION=0.05,
                 FEATURE_PIPELINE_VERSION='test', TRAIN_PATH=train_path, ALGORITHM='random_forest',
                 RESIDUAL=False, SEED=42, N_TRIALS=1, N_STREAMS=4, RF_TREES=[150,300], RF_DEPTHS=[10,16],
                 PREDICTION_BATCH=128, DEVICE='cuda', RUN_CONTRACT={})
    for name in ('OUT','PREPARED','RUN_DIR','DOCS_DIR','PARAMS_DIR','MODEL_DIR','POSTPROCESS_DIR'):
        scope[name] = (scope['RUN_DIR'] if name in ('DOCS_DIR','PARAMS_DIR','MODEL_DIR','POSTPROCESS_DIR') else root) / name
        scope[name].mkdir()
    with contextlib.redirect_stdout(io.StringIO()):
        for index in (3,5,7,9,11):
            exec(direct.cells[index].source, scope)
    check(scope['holdout_start'] == pd.Timestamp('2025-10-18 10:30'), 'Неверная граница test')
    mtimes = [p.stat().st_mtime_ns for p in sorted(scope['OUT'].glob('history_h*.parquet'))]
    with contextlib.redirect_stdout(io.StringIO()):
        for h in range(1,9):
            scope['write_horizon'](h)
    check(mtimes == [p.stat().st_mtime_ns for p in sorted(scope['OUT'].glob('history_h*.parquet'))], 'Кэш признаков переписан')
    source = (ROOT / 'scripts/random_forest_models.py').read_text()
    functions = ast.Module(body=[n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)], type_ignores=[])
    exec(compile(functions, '<functions>', 'exec'), scope)
    fits = []
    def fake_fit(frame, matrix, params):
        check(np.isfinite(matrix).all(), 'Не заполнены пропуски')
        fits.append((len(frame), frame.timestamp.max()))
        return PredictionStub()
    scope['fit_forest'] = fake_fit
    with contextlib.redirect_stdout(io.StringIO()):
        chosen = scope['select_horizon'](1)
        repeated = scope['select_horizon'](1)
    check(chosen == repeated and len(fits)==3, 'Поиск не возобновляется')
    fold = scope['folds'][0]
    folder = scope['prepare_stage'](1, 'fold1', fold['train'], fold['tuning'], fold['val'][0].normalize())
    ready = json.loads((folder / 'ready.json').read_text())
    training = pd.read_parquet(scope['OUT'] / 'history_h1.parquet')
    training = training.loc[np.isfinite(training.profile28) & training.cutoff.dt.normalize().isin(fold['train'].normalize()) & (training.timestamp < fold['val'][0].normalize())]
    encoded = scope['transformed_frame'](scope['add_correlation_features'](training,1,scope['cached_lags'](fold['val'][0].normalize())), ready['columns'])
    reference = SimpleImputer(strategy='median', keep_empty_features=True).fit(encoded)
    cached = joblib.load(folder / 'preprocess.joblib')
    check(np.allclose(reference.statistics_, cached.statistics_), 'Медианы рассчитаны не на train')
    scope['RESIDUAL'] = True
    data = scope['load_stage'](folder)
    restored = scope['model_target'](data['train'][0]) * 100000 + data['train'][0].profile28.to_numpy()
    check(np.allclose(restored,data['train'][0].y.to_numpy(),rtol=1e-5,atol=1e-3), 'Неверная residual-цель')
    check(scope['prepare_stage'](1,'fold1',fold['train'],fold['tuning'],fold['val'][0].normalize()) == folder, 'Нет общего кэша')
    scope['RESIDUAL'] = False
    scope['selected'] = {h: chosen for h in range(1,9)}
    scope['manifest'] = {'test_note': 'test only'}
    evaluation = source[source.index('evaluation_rows = []'):]
    with contextlib.redirect_stdout(io.StringIO()):
        exec(evaluation, scope)
        exec(evaluation, scope)
    check(len(fits)==11, 'Оценка повторно обучается после продолжения')
    check(all(t < pd.Timestamp('2025-10-18') for _,t in fits), 'Метки test попали в обучение')
    check((scope['DOCS_DIR']/'independent_metrics.csv').exists(), 'Нет метрик')
    check(len(list(scope['MODEL_DIR'].glob('evaluation/h*/model.joblib')))==8, 'Не сохранены 8 моделей')
    check(not list(root.rglob('submission.csv')), 'Создан submission')
    check(not list(root.rglob('*importance*')), 'Вычислена важность')
    study = optuna.load_study(study_name='h1', storage=f'sqlite:///{(scope["POSTPROCESS_DIR"] / "search.sqlite3").resolve().as_posix()}')
    running = study.ask()
    scope['suggest_params'](running)
    scope['N_TRIALS'] = 2
    with contextlib.redirect_stdout(io.StringIO()):
        scope['select_horizon'](1)
    check(len(fits)==14, 'Прерванный trial не восстановлен')
    check(len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)))==2, 'Неверный бюджет продолжения')
print('OK: notebook syntax, shared feature cache, train-only preprocessing, residual target, search/evaluation resume, interrupted trial, temporal boundaries, artifacts; no model training')
