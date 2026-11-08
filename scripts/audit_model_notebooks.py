from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import ast
import contextlib
import io
import json
import tempfile
import gc
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.base import BaseEstimator, RegressorMixin

ROOT = Path(__file__).resolve().parents[1]
features = (ROOT / 'scripts/five_starts_features.py').read_text()
models = (ROOT / 'scripts/five_starts_models.py').read_text()
prefix = features.split('for h in range(1, 9):\n    write_horizon(h)')[0]
model_nodes = ast.parse(models).body
definitions = ast.Module(body=[n for n in model_nodes if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))], type_ignores=[])
grid = pd.date_range('2025-07-28', '2025-11-01 10:30', freq='30min')
rng = np.random.default_rng(17)
raw = pd.DataFrame({'timestamp': grid.repeat(3), 'route_id': np.tile([2, 7, 29], len(grid))})
for name in ['target_1h', *[f'status_{i}' for i in range(1, 7)]]:
    raw[name] = rng.integers(1, 1000, len(raw)).astype(float)
requests = pd.MultiIndex.from_product([[29, 2, 7], pd.date_range(grid[-1] + pd.Timedelta(minutes=30), periods=8, freq='30min')], names=['route_id', 'timestamp']).to_frame(index=False)
requests.insert(0, 'id', np.arange(len(requests)))

with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    requests.to_parquet(root / 'test.parquet', index=False)
    def prepare(data, name):
        folder = root / name
        folder.mkdir()
        data.to_parquet(folder / 'train.parquet', index=False)
        env = dict(np=np, pd=pd, pa=pa, pq=pq, Path=Path, json=json, gc=gc, display=lambda x: None,
                   TRAIN_PATH=folder / 'train.parquet', TEST_PATH=root / 'test.parquet', DOCS_DIR=folder,
                   TARGET='target_1h', STATUSES=[f'status_{i}' for i in range(1, 7)], STEP=pd.Timedelta(minutes=30),
                   TRAIN_MINUTES=(570, 600, 630, 660, 690), WINDOW_OFFSETS=tuple(range(-4, 5)),
                   LAG_MAX_LOOKBACK_HOURS=24, LAG_MIN_PAIRS=336, LAG_MIN_CORRELATION=0.05,
                   TARGET_SCALE=100000, RESIDUAL=False, ALGORITHM='ridge', DEVICE='cpu',
                   SEED=42, N_JOBS=1, MAX_TREES=20, EARLY_STOPPING_ROUNDS=3, PREDICTION_BATCH=4)
        exec(prefix, env)
        return env
    base = prepare(raw, 'baseline')
    for i, minute in enumerate(base['TRAIN_MINUTES']):
        cutoff = pd.Timestamp('2025-10-10') + pd.Timedelta(minutes=minute)
        changed = raw.copy()
        changed.loc[changed.timestamp > cutoff, changed.columns.difference(['route_id', 'timestamp'])] += 100000
        perturbed = prepare(changed, f'future_{i}')
        origin = np.array([grid.get_loc(cutoff)])
        for h in range(1, 9):
            before = base['horizon_dataset'](h, origin)
            after = perturbed['horizon_dataset'](h, origin)
            pd.testing.assert_frame_equal(before, after)
    for i, fold in enumerate(base['folds']):
        boundary = fold['val'][0].normalize()
        changed = raw.copy()
        changed.loc[changed.timestamp >= boundary, changed.columns.difference(['route_id', 'timestamp'])] += 100000
        perturbed = prepare(changed, f'fold_{i}')
        original_lags = base['fit_lags'](boundary)
        changed_lags = perturbed['fit_lags'](boundary)
        pd.testing.assert_frame_equal(original_lags, changed_lags)
        origins = np.array([grid.get_loc(fold['train'][-1])])
        for h in range(1, 9):
            before = base['add_correlation_features'](base['horizon_dataset'](h, origins), h, original_lags)
            after = perturbed['add_correlation_features'](perturbed['horizon_dataset'](h, origins), h, changed_lags)
            pd.testing.assert_frame_equal(before, after)
    changed = raw.copy()
    changed.loc[changed.timestamp >= pd.Timestamp('2025-09-20'), 'target_1h'] += 1000000
    perturbed = prepare(changed, 'holdout')
    for fold in base['folds']:
        origins = grid.get_indexer(fold['tuning'])
        for h in range(1, 9):
            pd.testing.assert_frame_equal(base['horizon_dataset'](h, origins), perturbed['horizon_dataset'](h, origins))
    origin = grid.get_loc(pd.Timestamp('2025-10-10 10:30'))
    oracle = base['horizon_dataset'](1, np.array([origin]))
    target = base['target']
    expected = {
        'target_back_0': target[origin],
        'target_back_8': target[origin - 8],
        'target_1h_mean_12': target[origin - 11:origin + 1].mean(axis=0),
        'slot_days_back_1': target[origin + 1 - 48],
        'network_target_1h_latest': np.repeat(target[origin].sum(), 3),
    }
    for name, values in expected.items():
        if not np.allclose(oracle[name], values, rtol=1e-6):
            raise RuntimeError(f'Неверная формула {name}')
    profile_values = []
    for d in range(1, 29):
        current = target[origin - 47:origin + 1].mean(axis=0)
        previous = target[origin - d * 48 - 47:origin - d * 48 + 1].mean(axis=0)
        profile_values.append(target[origin + 1 - d * 48] * current / np.maximum(previous, 1))
    if not np.allclose(oracle.profile28, np.mean(profile_values, axis=0), rtol=1e-6):
        raise RuntimeError('Неверная формула profile28')
    signal = rng.normal(size=(len(grid), 3)).astype(np.float32)
    delayed = np.full_like(signal, np.nan)
    delayed[6:] = signal[:-6]
    table = base['calculate_dynamic_shoulder'](delayed, signal, grid, base['routes'], fit_before=grid[-1])
    if not np.all(table.selected_status6_lag_steps == 6):
        raise RuntimeError('Неверный поиск корреляционного лага')
    frame = pd.DataFrame({'route_id': base['routes'], 'cutoff': grid[origin]})
    correlated = base['correlation_block'](frame, 8, table, signal, '')
    if correlated.lagged_status6_signal.notna().any() or correlated.lagged_status6_signal_available.any():
        raise RuntimeError('В корреляционный признак попало будущее')
    if not np.allclose(correlated.lagged_status6_signal_offset_m4, signal[origin - 2]):
        raise RuntimeError('Некорректный доступный сосед корреляционного лага')
    metric_nodes = [n for n in ast.parse(features).body if isinstance(n, ast.FunctionDef) and n.name in ('score_row', 'mean_daily_score')]
    exec(compile(ast.Module(body=metric_nodes, type_ignores=[]), '<metric>', 'exec'), base)
    truth = pd.DataFrame({'cutoff': pd.to_datetime(['2025-01-01', '2025-01-01']), 'y': [10., 30.]})
    if not np.isclose(base['score_row'](truth, [20., 40.], 'test')['метрика конкурса'], 1.):
        raise RuntimeError('Неверная метрика соревнования')
    exec(compile(definitions, '<models>', 'exec'), base)
    original_prep = base['preprocessing'](['route_id', 'value'], True)
    prep_train = pd.DataFrame({'route_id': [2, 7, 29], 'value': [1., np.nan, 3.]})
    original_prep.fit(base['transformed_frame'](prep_train, prep_train.columns.tolist()))
    transformed = original_prep.transform(base['transformed_frame'](pd.DataFrame({'route_id': [2, 7], 'value': [np.nan, 1000000.]}), prep_train.columns.tolist()))
    if not np.isclose(transformed[0, 0], 0):
        raise RuntimeError('Imputer/scaler использовал статистику inference')
    print('OK: future perturbations for 5 starts x 8 horizons; 3 lag-fit boundaries; holdout isolation; independent feature/metric formulas; inference-only preprocessing.')

    fit_log = []
    class MockEstimator(RegressorMixin, BaseEstimator):
        def __init__(self, **kwargs):
            self.options = kwargs
        def fit(self, X, y, **kwargs):
            fit_log.append((self.options, X.copy(), np.asarray(y).copy(), kwargs))
            self.n_features_in_ = X.shape[1]
            self.coef_ = np.zeros(X.shape[1])
            self.best_iteration = 3
            self.best_iteration_ = 4
            self.n_estimators_ = self.options.get('n_estimators', 4)
            self.tree_count_ = 4
            self.booster_ = self
            return self
        def predict(self, X):
            return np.full(len(X), 0.25)
        def get_booster(self):
            return self
        def save_config(self):
            return json.dumps({'learner': {'generic_param': {'device': 'cpu'}}})
        def save_model(self, filename):
            Path(filename).write_text('mock model; no estimator trained')
    class MockProphet:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.extra_regressors = {}
        def add_regressor(self, name, **kwargs):
            self.extra_regressors[name] = kwargs
        def fit(self, data, **kwargs):
            fit_log.append((self.options, data.copy(), data.y.to_numpy(), kwargs))
            return self
        def predict(self, data):
            ordered = data.sort_values('ds', kind='mergesort')
            return pd.DataFrame({'ds': ordered.ds.to_numpy(), 'yhat': ordered.x0.to_numpy()})
    modules = {}
    for name, attr in [('xgboost', 'XGBRegressor'), ('catboost', 'CatBoostRegressor'), ('lightgbm', 'LGBMRegressor'), ('prophet', 'Prophet')]:
        module = ModuleType(name)
        setattr(module, attr, MockProphet if name == 'prophet' else MockEstimator)
        modules[name] = module
    modules['lightgbm'].early_stopping = lambda rounds, **kwargs: ('early_stopping', rounds, kwargs)
    modules['lightgbm'].log_evaluation = lambda period: ('log', period)
    serializer = ModuleType('prophet.serialize')
    serializer.model_to_json = lambda model: json.dumps(model.options)
    modules['prophet.serialize'] = serializer
    base.update(Ridge=MockEstimator, RandomForestRegressor=MockEstimator)
    train = pd.DataFrame({'route_id': [2, 7, 29, 2, 7, 29], 'value': [1., 2., np.nan, 4., 5., 6.],
                          'cutoff': pd.to_datetime(['2025-01-01 10:30'] * 3 + ['2025-01-02 10:30'] * 3),
                          'timestamp': pd.to_datetime(['2025-01-01 11:00'] * 3 + ['2025-01-02 11:00'] * 3),
                          'profile28': [100.] * 6, 'y': [200., 300., 400., 500., 600., 700.]})
    early = train.copy()
    early['cutoff'] += pd.Timedelta(days=5)
    early['timestamp'] += pd.Timedelta(days=5)
    columns = ['route_id', 'value']
    class Trial:
        def suggest_int(self, name, low, high, **kwargs):
            return low
        def suggest_float(self, name, low, high, **kwargs):
            return low
        def suggest_categorical(self, name, choices):
            return choices[0]
    with patch.dict('sys.modules', modules):
        for algorithm in ('xgb', 'catboost', 'lightgbm', 'ridge', 'random_forest', 'prophet'):
            for residual in (False, True):
                base['ALGORITHM'], base['RESIDUAL'] = algorithm, residual
                params = base['suggest_params'](Trial())
                with contextlib.redirect_stdout(io.StringIO()):
                    bundle = base['fit_model'](train, columns, params, early if algorithm in ('xgb', 'catboost', 'lightgbm') else None)
                options, received, actual_y, kwargs = fit_log[-1]
                expected_y = (train.y.to_numpy() - (100 if residual else 0)) / 100000
                if not np.allclose(actual_y, expected_y):
                    raise RuntimeError(f'Некорректная цель {algorithm}/{residual}')
                if algorithm in ('xgb', 'lightgbm') and not isinstance(received.route_id.dtype, pd.CategoricalDtype):
                    raise RuntimeError('Потеря категориального маршрута')
                if algorithm in ('xgb', 'catboost', 'lightgbm') and 'eval_set' not in kwargs:
                    raise RuntimeError('Нет отдельной early-stopping выборки')
                shuffled = train.iloc[[4, 0, 5, 1, 3, 2]].copy()
                prediction = base['predict'](bundle, shuffled)
                if algorithm == 'prophet':
                    design = base['prophet_matrix'](shuffled, columns, bundle['prep'], bundle['names'])
                    expected = np.maximum(0, (100 if residual else 0) + 100000 * design.x0.to_numpy())
                else:
                    expected = np.repeat(25000 + (100 if residual else 0), len(shuffled))
                if not np.allclose(prediction, expected):
                    raise RuntimeError(f'Потерян порядок или масштаб {algorithm}/{residual}')
                if algorithm in ('xgb', 'catboost', 'lightgbm') and bundle['trees'] != 4:
                    raise RuntimeError('Неверное число деревьев')
                folder = root / f'{algorithm}_{residual}'
                base['save_bundle'](bundle, folder)
                metadata = json.loads((folder / 'metadata.json').read_text())
                if metadata['algorithm'] != algorithm or metadata['residual'] != residual:
                    raise RuntimeError('Некорректные metadata модели')
    base['ALGORITHM'], base['RESIDUAL'] = 'ridge', False
    class BrokenPrediction:
        def predict(self, X):
            return np.full(len(X), -np.inf)
    try:
        base['predict'](dict(model=BrokenPrediction(), columns=columns), train)
    except ValueError:
        pass
    else:
        raise RuntimeError('Неконечные прогнозы молча прошли clipping')
    print('OK: all 12 fit/predict/save adapter contracts with test doubles; categorical routes; targets; early stopping; Prophet reordered rows; rejection of nonfinite predictions. No estimator training was run.')
