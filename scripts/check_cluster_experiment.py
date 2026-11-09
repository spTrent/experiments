import ast
import contextlib
import io
import json
import tempfile
import tokenize
from pathlib import Path
import nbformat
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
path = BASE / 'notebooks/models/cluster_catboost_ridge.ipynb'
nb = nbformat.read(path, as_version=4)
nbformat.validate(nb)
for cell in nb.cells:
    if cell.cell_type != 'code':
        continue
    tree = ast.parse(cell.source)
    compile(tree, str(path), 'exec')
    if any(isinstance(n, ast.Assert) for n in ast.walk(tree)):
        raise ValueError('Обнаружен assert')
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and ast.get_docstring(node):
            raise ValueError('Обнаружен docstring')
    if any(t.type == tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(cell.source).readline)):
        raise ValueError('Обнаружен комментарий')


class DummyRegressor:
    calls = []

    def __init__(self, **settings):
        self.settings = settings

    def fit(self, x, y, sample_weight=None):
        self.value = float(np.average(y, weights=sample_weight))
        self.calls.append((len(y), self.settings))
        return self

    def predict(self, x):
        return np.full(len(x), self.value)


ns = {}
exec(nb.cells[1].source, ns)
exec(nb.cells[3].source, ns)
ns.update(CatBoostRegressor=DummyRegressor, Ridge=DummyRegressor, display=lambda *args, **kwargs: None,
          CLUSTERS=(1, 2, 3), HORIZONS=(1, 2), MIN_CLUSTER_ROUTES=3, THREADS=1,
          CAT_CONFIGS=[dict(depth=4, iterations=2, learning_rate=0.05)], RIDGE_LAMBDAS=(0.01,),
          CREATE_ARCHIVE=False, STEP=pd.Timedelta(minutes=30), PREDICT_BATCH=1000)
with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    times = pd.date_range('2025-07-28', '2025-11-01 10:30', freq='30min')
    routes = np.array([11, 25, 40, 75, 100, 140])
    phase = np.arange(len(times))[:, None]
    target = (10 + routes[None, :] / 5) * (1.2 + np.sin(phase / 48 * 2 * np.pi + np.arange(len(routes))))
    history = pd.DataFrame(dict(timestamp=np.repeat(times, len(routes)), route_id=np.tile(routes, len(times)), target_1h=target.ravel()))
    for i in range(1, 7):
        history[f'status_{i}'] = history.target_1h * (i + 1)
    future = pd.DataFrame([dict(id=i * 2 + h, route_id=r, timestamp=times[-1] + h * pd.Timedelta(minutes=30))
                           for i, r in enumerate(routes) for h in (1, 2)]).sample(frac=1, random_state=4).reset_index(drop=True)
    history.to_parquet(root / 'train.parquet', index=False)
    future.to_parquet(root / 'test.parquet', index=False)
    ns.update(TRAIN_PATH=str(root / 'train.parquet'), TEST_PATH=str(root / 'test.parquet'), OUTPUT_ROOT=str(root))
    with contextlib.redirect_stdout(io.StringIO()), ns['threadpool_limits'](limits=1):
        ns['load_inputs']()
        seg = ns['segment_routes'](ns['VAL_START'], 'check')
        for h in ns['HORIZONS']:
            train_origins = ns['origin_indices'](h, 'train')
            if not (ns['TIMES'][train_origins + h] < ns['VAL_START']).all():
                raise ValueError('Train пересекает val')
            val_origins = ns['origin_indices'](h, 'val')
            if len(val_origins) != 14 or not (ns['TIMES'][val_origins + h] < ns['TEST_START']).all():
                raise ValueError('Некорректный val')
            if len(ns['origin_indices'](h, 'full')) != len(times) - h:
                raise ValueError('Полное обучение теряет доступные метки')
        origin = np.array([2000])
        before, names = ns['features'](2, origin, seg)
        saved_values = {key: value.copy() for key, value in ns['VALUES'].items()}
        saved_roll = {key: value.copy() for key, value in ns['ROLL'].items()}
        ns['VALUES'] = {key: value.copy() for key, value in saved_values.items()}
        ns['ROLL'] = {key: value.copy() for key, value in saved_roll.items()}
        for value in [*ns['VALUES'].values(), *ns['ROLL'].values()]:
            value[2001:] += 99999
        after, _ = ns['features'](2, origin, seg)
        if not np.array_equal(before, after):
            raise ValueError('Признаки используют будущее')
        ns['VALUES'], ns['ROLL'] = saved_values, saved_roll
        before_profiles = ns['segment_routes'](ns['VAL_START'], 'before')
        ns['VALUES']['target_1h'][ns['TIMES'] >= ns['VAL_START']] += 10000
        after_profiles = ns['segment_routes'](ns['VAL_START'], 'after')
        if not np.array_equal(before_profiles['scale'], after_profiles['scale']) or any(
            not np.array_equal(before_profiles['labels'][k], after_profiles['labels'][k]) for k in ns['CLUSTERS']):
            raise ValueError('Сегментация использует val/test')
        ns['VALUES']['target_1h'] = target.astype(np.float32)
        original_dataset = ns['dataset']
        def search_dataset(h, period, segmentation):
            if period == 'test':
                raise ValueError('Подбор открыл test')
            return original_dataset(h, period, segmentation)
        ns['dataset'] = search_dataset
        frozen = ns['run_search']()
        ns['dataset'] = original_dataset
        fit_count = len(DummyRegressor.calls)
        resumed = ns['run_search']()
        if resumed != frozen or len(DummyRegressor.calls) != fit_count:
            raise ValueError('Подбор не возобновляется из checkpoint')
        ns['run_evaluation'](frozen)
        fit_count = len(DummyRegressor.calls)
        ns['run_evaluation'](frozen)
        if len(DummyRegressor.calls) != fit_count:
            raise ValueError('Оценка не возобновляется из checkpoint')
        ns['run_full'](frozen)
        submitted = pd.read_csv(ns['RUN'] / 'submission.csv')
        if submitted.id.tolist() != future.id.tolist() or not np.isfinite(submitted.y_pred).all():
            raise ValueError('Некорректный submission')
        for h in ns['HORIZONS']:
            bundle = ns['joblib'].load(ns['RUN'] / f'final_model_h{h}.joblib')
            if bundle['feature_names'] != names:
                raise ValueError('Потеряны имена признаков')
        metric_rows = [ns['metrics'](np.array([2., 3.]), np.array([4., 1.])), ns['metrics'](np.array([5.]), np.array([4.]))]
        aggregated = ns['aggregate']([dict(group='a', **r) for r in metric_rows], ['group'])
        direct = ns['metrics'](np.array([2., 3., 5.]), np.array([4., 1., 4.]))
        if not np.isclose(aggregated.iloc[0]['WAPE + |Relative Bias|'], direct['score']):
            raise ValueError('Ошибочная агрегация метрики')
        train = original_dataset(1, 'train', seg)
        fallback_seg = dict(seg, labels={1: np.full(len(routes), -1)})
        fallback_bundle = ns['fit_model'](train, fallback_seg, 'cat', 1, ns['CAT_CONFIGS'][0])
        if list(fallback_bundle['models']) != [-1] or DummyRegressor.calls[-1][0] != len(train['y']):
            raise ValueError('Fallback обучается не на всех маршрутах')
        fallback_pred = ns['predict_model'](fallback_bundle, train, fallback_seg)
        if not np.isfinite(fallback_pred).all():
            raise ValueError('Ошибка прогноза fallback')
print('OK: notebook schema, syntax, code restrictions, causal features and clusters, temporal splits, frozen selection, resume, fallback, metrics, full-history coverage and submission. CatBoost/Ridge replaced with test doubles; no real-data training.')
