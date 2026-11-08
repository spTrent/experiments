from pathlib import Path
import ast
import contextlib
import io
import json
import tempfile
from types import SimpleNamespace
import tokenize
import gc
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import nbformat

ROOT = Path(__file__).resolve().parents[1]
for algorithm in ('xgb', 'lightgbm', 'random_forest', 'ridge', 'prophet', 'catboost'):
    for variant in ('direct', 'residual'):
        notebook = nbformat.read(ROOT / f'{algorithm}_{variant}.ipynb', as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type != 'code':
                continue
            tree = ast.parse(cell.source)
            if any(isinstance(node, ast.Assert) for node in ast.walk(tree)):
                raise RuntimeError('Найден assert')
            if any(token.type == tokenize.COMMENT for token in tokenize.generate_tokens(io.StringIO(cell.source).readline)):
                raise RuntimeError('Найден комментарий')
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and ast.get_docstring(node):
                    raise RuntimeError('Найден docstring')

feature_source = (ROOT / 'scripts/five_starts_features.py').read_text()
model_source = (ROOT / 'scripts/five_starts_models.py').read_text()
feature_prefix = feature_source.split('for h in range(1, 9):\n    write_horizon(h)')[0]
grid = pd.date_range('2025-01-01', '2025-03-15 10:30', freq='30min')
rng = np.random.default_rng(42)
data = pd.DataFrame({'timestamp': grid.repeat(2), 'route_id': np.tile([10, 20], len(grid))})
for name in ['target_1h', *[f'status_{i}' for i in range(1, 7)]]:
    data[name] = rng.integers(1, 1000, len(data)).astype(float)
requests = pd.MultiIndex.from_product([[10, 20], pd.date_range(grid[-1] + pd.Timedelta(minutes=30), periods=8, freq='30min')], names=['route_id', 'timestamp']).to_frame(index=False)
requests.insert(0, 'id', np.arange(len(requests)))

with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    data.to_parquet(root / 'train.parquet', index=False)
    requests.to_parquet(root / 'test.parquet', index=False)
    def environment(folder):
        out = root / folder / 'features'
        out.mkdir(parents=True)
        model_dir = out.parent / 'models'
        model_dir.mkdir()
        folders = {name: out.parent / directory for name, directory in [('DOCS_DIR', 'docs'), ('IMPORTANCE_DIR', 'importance'), ('POSTPROCESS_DIR', 'postprocessing'), ('PARAMS_DIR', 'model_params')]}
        for folder in folders.values():
            folder.mkdir()
        return dict(Path=Path, json=json, gc=gc, np=np, pd=pd, pa=pa, pq=pq, display=lambda value: None,
                    TRAIN_PATH=root / 'train.parquet', TEST_PATH=root / 'test.parquet', OUT=out,
                    MODEL_DIR=model_dir, RUN_DIR=out.parent, **folders, TARGET='target_1h', STATUSES=[f'status_{i}' for i in range(1, 7)],
                    STEP=pd.Timedelta(minutes=30), TRAIN_MINUTES=(570, 600, 630, 660, 690),
                    WINDOW_OFFSETS=tuple(range(-4, 5)), LAG_MAX_LOOKBACK_HOURS=24,
                    LAG_MIN_PAIRS=336, LAG_MIN_CORRELATION=0.05, TARGET_SCALE=100000,
                    FEATURE_PIPELINE_VERSION='test', RESIDUAL=False, ALGORITHM='ridge', DEVICE='cpu',
                    SEED=42, N_JOBS=1, N_TRIALS=2, MAX_TREES=2, EARLY_STOPPING_ROUNDS=1, PREDICTION_BATCH=5)
    first = environment('first')
    exec(feature_prefix, first)
    cutoff = pd.Timestamp('2025-03-10 10:30')
    origin = np.array([grid.get_loc(cutoff)])
    reference = {h: first['horizon_dataset'](h, origin) for h in range(1, 9)}
    altered = data.copy()
    altered.loc[altered.timestamp > cutoff, altered.columns.difference(['route_id', 'timestamp'])] *= 100
    altered.to_parquet(root / 'train.parquet', index=False)
    second = environment('second')
    exec(feature_prefix, second)
    for h in range(1, 9):
        pd.testing.assert_frame_equal(reference[h], second['horizon_dataset'](h, origin))
    data.to_parquet(root / 'train.parquet', index=False)
    with contextlib.redirect_stdout(io.StringIO()):
        exec(feature_source, first)
    fit_calls = []
    def fake_fit(frame, columns, params, early=None, trees=None):
        fit_calls.append((frame.timestamp.min(), frame.timestamp.max(), len(frame)))
        model = SimpleNamespace(named_steps={'preprocess': SimpleNamespace(get_feature_names_out=lambda: columns), 'model': SimpleNamespace(coef_=np.zeros(len(columns)))})
        return dict(model=model, columns=columns, trees=None)
    def fake_predict(bundle, frame):
        return frame.profile28.to_numpy()
    source_tree = ast.parse(model_source)
    definitions = [node for node in source_tree.body if isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom))]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), '<definitions>', 'exec'), first)
    encoding = first.copy()
    exec(compile(ast.Module(body=definitions, type_ignores=[]), '<encoding>', 'exec'), encoding)
    encoding['routes'] = np.arange(1000) * 3 + 10
    sample = pd.DataFrame({'route_id': encoding['routes'], 'value': np.arange(1000, dtype=np.float32)})
    columns = sample.columns.tolist()
    matrix = encoding['transformed_frame'](sample, columns)
    bits = encoding['route_bit_columns']()
    if len(bits) != 10 or matrix.shape != (1000, 11):
        raise RuntimeError('Неверная ширина двоичного кодирования')
    if matrix[bits].drop_duplicates().shape[0] != 1000:
        raise RuntimeError('Коллизия кодов маршрутов')
    unknown = encoding['transformed_frame'](pd.DataFrame({'route_id': [-99], 'value': [0.]}), columns)
    if unknown[bits].to_numpy().any():
        raise RuntimeError('Неизвестный маршрут должен иметь нулевой код')
    prep = encoding['preprocessing'](columns, True)
    transformed = prep.fit_transform(matrix)
    encoding['joblib'].dump(prep, root / 'preprocess.joblib')
    loaded = encoding['joblib'].load(root / 'preprocess.joblib')
    if not np.allclose(transformed, loaded.transform(matrix)) or transformed.shape != (1000, 11):
        raise RuntimeError('Преобразования не сохраняются корректно')
    prophet_data = encoding['prophet_matrix'](sample.assign(timestamp=pd.Timestamp('2025-01-01')), columns, prep, [f'x{i}' for i in range(11)])
    if prophet_data.shape != (1000, 12):
        raise RuntimeError('Неверная ширина матрицы Prophet')
    for column in first['CORRELATION_COLUMNS']:
        if pd.read_parquet(first['OUT'] / 'history_h1.parquet', columns=[column])[column].notna().any():
            raise RuntimeError('Базовый parquet содержит обученные корреляционные признаки')
    first.update(fit_model=fake_fit, predict=fake_predict, save_bundle=lambda *args: None)
    first['optuna'].logging.set_verbosity(first['optuna'].logging.ERROR)
    actions = [node for node in source_tree.body if not isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom))]
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(ast.Module(body=actions, type_ignores=[]), '<workflow>', 'exec'), first)
    search_count = 8 * 2 * 3
    if len(fit_calls) != search_count + 16:
        raise RuntimeError(f'Неверное число fit: {len(fit_calls)}')
    boundary = first['holdout_start'].normalize()
    if any(last >= boundary for _, last, _ in fit_calls[:search_count + 8]):
        raise RuntimeError('Test попал в обучение до refit')
    for h, (_, last, count) in enumerate(fit_calls[-8:], 1):
        eligible = pd.read_parquet(first['OUT'] / f'history_h{h}.parquet')
        eligible = eligible.loc[np.isfinite(eligible.profile28)]
        if last != eligible.timestamp.max() or count != len(eligible):
            raise RuntimeError('Refit не использует все доступные метки')
    result = pd.read_csv(first['DOCS_DIR'] / 'submission.csv')
    if result.columns.tolist() != ['id', 'y_pred'] or result.id.tolist() != requests.id.tolist():
        raise RuntimeError('Неверный формат submission')
    import zipfile
    with zipfile.ZipFile(first['OUT'].parent / 'results.zip') as archive:
        if any(name.startswith('features/') for name in archive.namelist()):
            raise RuntimeError('FE попал в архив результатов')
        if 'docs/independent_metrics.csv' not in archive.namelist():
            raise RuntimeError('Нет независимых метрик в архиве')
    for filename in ('feature_importance.csv', 'feature_pipeline.json', 'metrics.csv', 'profile_control.csv', 'status6_lags_validation.csv', 'status6_lags_evaluation.csv', 'status6_lags_full.csv', 'submission.csv'):
        if not (first['DOCS_DIR'] / filename).is_file():
            raise RuntimeError(f'Нет документа {filename}')
    for h in range(1, 9):
        best = json.loads((first['PARAMS_DIR'] / f'best_h{h}.json').read_text())
        if not best['params'] or 'tuning_score' not in best:
            raise RuntimeError('Нет лучших параметров Optuna')
    with zipfile.ZipFile(first['RUN_DIR'] / 'reports.zip') as archive:
        if 'model_params/best_h8.json' not in archive.namelist() or 'docs/feature_importance.csv' not in archive.namelist():
            raise RuntimeError('Неполный архив отчётов')
    if {p.name for p in first['RUN_DIR'].iterdir() if p.is_dir()} != {'features', 'importance', 'models', 'postprocessing', 'docs', 'model_params'}:
        raise RuntimeError('Неверная структура папок')
    tuning_artifact = pd.read_parquet(first['POSTPROCESS_DIR'] / 'tuning_predictions_h1.parquet')
    if tuning_artifact.is_unbiased_oof.any() or (tuning_artifact.model_info_through < tuning_artifact.timestamp.max()).any():
        raise RuntimeError('Tuning ошибочно помечен как независимый OOF')
    restored = first.copy()
    exec(compile(ast.Module(body=[node for node in definitions if isinstance(node, ast.FunctionDef) and node.name in ('predict', 'offset', 'transformed_frame')], type_ignores=[]), '<prediction>', 'exec'), restored)
    class ConstantModel:
        def predict(self, frame):
            return np.full(len(frame), -0.001)
    sample = reference[1].copy()
    sample['profile28'] = [50., 200.]
    restored['RESIDUAL'] = True
    output = restored['predict'](dict(model=ConstantModel(), columns=['route_id']), sample)
    if not np.allclose(output, [0., 100.]):
        raise RuntimeError('Неверное восстановление residual')
print('OK: 12 notebooks; causal features on 8 horizons; mocked search/evaluation/refit; submission; archives; residual restoration; binary route encoding and preprocessing serialization. No estimator training was run.')
