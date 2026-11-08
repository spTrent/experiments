import argparse
import ast
import io
import json
import tempfile
import tokenize
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import nbformat
import numpy as np
import pandas as pd


def require(condition, message):
    if not condition:
        raise ValueError(message)


class DisplayHandle:
    def update(self, value):
        pass


def fixture(root):
    rng = np.random.default_rng(42)
    grid = pd.date_range('2025-09-17', '2025-11-01 10:30', freq='30min')
    routes = np.array([11, 31])
    index = np.arange(len(grid))
    target = np.maximum(0, 100 + 50 * np.sin(index[:, None] / 48 * 2 * np.pi) + rng.normal(0, 12, (len(grid), 2)))
    target[(grid >= '2025-10-04') & (grid < '2025-10-18'), 0] = 0
    panel = dict(route_id=np.tile(routes, len(grid)), timestamp=grid.repeat(2), target_1h=target.ravel())
    for status in range(1, 7):
        panel[f'status_{status}'] = np.maximum(0, target * status / 6 + rng.normal(0, 5, target.shape)).ravel()
    pd.DataFrame(panel).to_parquet(root / 'train_solo_track.parquet', index=False)
    future = pd.DataFrame([dict(id=f'{route}_{h}', route_id=route, timestamp=grid[-1] + h * pd.Timedelta(minutes=30)) for route in routes for h in range(1, 9)])
    future = future.sample(frac=1, random_state=2).reset_index(drop=True)
    future.to_parquet(root / 'test_solo_track.parquet', index=False)
    return future


def load_notebook(root, device):
    path = Path(__file__).resolve().parents[1] / 'ridge_by_route_horizon.ipynb'
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    namespace = {}
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != 'code':
            continue
        text = cell.source
        tree = ast.parse(text)
        require(not any(isinstance(node, ast.Assert) for node in ast.walk(tree)), 'Forbidden assert')
        require(not any(token.type == tokenize.COMMENT for token in tokenize.generate_tokens(io.StringIO(text).readline)), 'Forbidden comment')
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
                require(ast.get_docstring(node) is None, 'Forbidden docstring')
        if index == 1:
            text = text.replace("TRAIN_PATH = Path('data/train_solo_track.parquet')", f"TRAIN_PATH = Path({str(root / 'train_solo_track.parquet')!r})")
            text = text.replace("TEST_PATH = Path('data/test_solo_track.parquet')", f"TEST_PATH = Path({str(root / 'test_solo_track.parquet')!r})")
            text = text.replace("OUTPUT_ROOT = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()", f'OUTPUT_ROOT = Path({str(root)!r})')
            text = text.replace("DEVICE = 'cuda'", f'DEVICE = {device!r}')
            text = text.replace('N_TRIALS = 30', 'N_TRIALS = 12').replace('EXPECTED_ROUTES = 1000', 'EXPECTED_ROUTES = 2')
            text = text.replace('MAX_BATCH_SIZE = 16', 'MAX_BATCH_SIZE = 3').replace('CHECKPOINT_EVERY = 5', 'CHECKPOINT_EVERY = 1')
        if index in [11, 13, 15, 17]:
            tree = ast.parse(text)
            tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
            compiled = compile(tree, f'cell_{index}', 'exec')
        else:
            compiled = compile(text, f'cell_{index}', 'exec')
        exec(compiled, namespace)
        if index == 1:
            namespace['display'] = lambda *args, **kwargs: DisplayHandle()
    return namespace


def check_padded_matrices(ns):
    generator = np.random.default_rng(17)
    items = []
    for index, (rows, width) in enumerate([(3, 1), (11, 5), (7, 3)]):
        frame = pd.DataFrame(generator.normal(size=(rows, width)), columns=[f'x{i}' for i in range(width)])
        if index == 0:
            frame[:] = 0
        frame['missing'] = np.nan
        frame['y'] = np.maximum(0, generator.normal(20, 10, rows))
        columns = [name for name in frame if name != 'y']
        preprocessing, matrix = ns['prepare_matrix'](frame, columns)
        x_mean = matrix.mean(axis=0)
        matrix -= x_mean
        items.append(dict(phase='search', x=matrix, query_x=matrix[:2], y=frame.y.to_numpy(), query_y=frame.y.to_numpy()[:2],
                          y_mean=float(frame.y.mean()), original_train=np.ones(rows, dtype=bool), preprocessing=preprocessing,
                          x_mean=x_mean, alpha=[1e-4, 1., 1e6][index]))
    batch = ns['DeviceBatch'](items, spectral=True)
    for spectral in [True, False]:
        coefficients = batch.coefficients([item['alpha'] for item in items], spectral=spectral)
        metrics, predictions = batch.metrics(coefficients)
        for index, item in enumerate(items):
            model = ns['Ridge'](alpha=item['alpha']).fit(item['x'], item['y'])
            expected = np.maximum(model.predict(item['query_x']), 0)
            require(np.allclose(predictions[index, :2], expected, rtol=1e-6, atol=1e-7), 'Padded solver mismatch')
            expected_metrics = ns['metric_values'](item['y'], np.maximum(model.predict(item['x']), 0))
            for name in ['n', 'actual_sum', 'prediction_sum', 'absolute_error', 'signed_error', 'score']:
                require(np.isclose(metrics['train'][index][name], expected_metrics[name], atol=1e-7), 'Padded metric mismatch')


def check_causality(ns):
    all_values = {name: value.copy() for name, value in ns['all_values'].items()}
    network = {name: value.copy() for name, value in ns['network_signals'].items()}
    origins = np.array([700, 701, 702])
    before = ns['RouteContext'](1).namespace['horizon_dataset'](8, origins)
    for values in ns['all_values'].values():
        values[703:] += 10000
    for values in ns['network_signals'].values():
        values[703:] += 20000
    after = ns['RouteContext'](1).namespace['horizon_dataset'](8, origins)
    require(before.equals(after), 'Feature uses future data')
    ns['all_values'] = all_values
    ns['network_signals'] = network
    first = ns['RouteContext'](1).lag_table('search')
    boundary = ns['times'].searchsorted(ns['val_start'])
    backup = {name: value.copy() for name, value in ns['all_values'].items()}
    for values in ns['all_values'].values():
        values[boundary:] *= 10
    second = ns['RouteContext'](1).lag_table('search')
    require(first.equals(second), 'Lag selection uses validation/test')
    ns['all_values'] = backup
    ns['feature_local'] = ns['threading'].local()


def check_threads(ns):
    tasks = [(0, 1), (1, 8), (0, 4), (1, 1)]
    sequential = [ns['prepare_pair'](task, 'search') for task in tasks]
    with ThreadPoolExecutor(max_workers=2) as executor:
        concurrent = list(executor.map(lambda task: ns['prepare_pair'](task, 'search'), tasks))
    for expected, actual in zip(sequential, concurrent):
        require(expected['route_id'] == actual['route_id'], 'Route state crossed between workers')
        require(np.array_equal(expected['x'], actual['x']), 'Concurrent features differ')
        require(expected['lag_table'].equals(actual['lag_table']), 'Concurrent lag tables differ')


def check_interrupted_search(ns):
    original = ns['atomic_joblib']
    interrupted = False

    def save_and_interrupt(path, value):
        original(path, value)
        if Path(path).parent.name == 'studies':
            raise InterruptedError('Synthetic interruption after checkpoint')

    ns['atomic_joblib'] = save_and_interrupt
    try:
        ns['run_search']()
    except InterruptedError:
        interrupted = True
    finally:
        ns['atomic_joblib'] = original
    require(interrupted, 'Interruption was not triggered')
    checkpoints = list((ns['RUN_ROOT'] / 'studies').glob('*.joblib'))
    require(len(checkpoints) == 1, 'Unexpected checkpoint count')
    require(ns['completed_trials'](joblib.load(checkpoints[0])) == 1, 'Checkpoint is not a completed trial')
    ns['run_search']()


def check_study_sequence(ns, root):
    saved_root = ns['RUN_ROOT']
    baseline = root / 'sequence_check'
    for name in ['studies', 'reports', 'params', 'trials']:
        (baseline / name).mkdir(parents=True, exist_ok=True)
    try:
        ns['RUN_ROOT'] = baseline
        for task in [(0, 1), (1, 8)]:
            item = ns['prepare_pair'](task, 'search')
            ns['process_device_batch']([item], 'search')
            together = joblib.load(saved_root / 'studies' / f"{item['key']}.joblib")
            separate = joblib.load(baseline / 'studies' / f"{item['key']}.joblib")
            for left, right in zip(together.trials, separate.trials):
                require(np.isclose(left.params['alpha'], right.params['alpha'], rtol=1e-8), 'Batching or resume changed alpha proposals')
                require(np.isclose(left.value, right.value, rtol=1e-7, atol=1e-7), 'Batching changed trial score')
    finally:
        ns['RUN_ROOT'] = saved_root


def run_checks(root, device):
    future = fixture(root)
    ns = load_notebook(root, device)
    with ns['threadpool_limits'](limits=1):
        check_padded_matrices(ns)
        check_causality(ns)
        check_threads(ns)
        check_interrupted_search(ns)
        check_study_sequence(ns, root)
        ns['run_evaluation']()
        ns['evaluation_report']()
        ns['run_full']()
        ns['finish_run']()
        saved = {path: path.stat().st_mtime_ns for path in ns['MODEL_ROOT'].rglob('*.joblib')}
        require(len(saved) == 16, 'Wrong model count')
        ns['MAX_BATCH_SIZE'] = 2
        ns['run_search']()
        ns['run_evaluation']()
        ns['evaluation_report']()
        ns['run_full']()
        require(all(path.stat().st_mtime_ns == stamp for path, stamp in saved.items()), 'Completed models were overwritten')
        record = json.loads((ns['RUN_ROOT'] / 'params' / 'route_0000_h1.json').read_text())
        require(record['objective_name'] == 'MAE_zero_val_volume', 'Zero-volume fallback failed')
        submission = pd.read_csv(root / 'submission.csv')
        require(submission.id.equals(future.id) and len(submission) == 16, 'Submission order changed')
        item = ns['prepare_pair']((1, 8), 'full')
        artifact = joblib.load(ns['MODEL_ROOT'] / 'h8' / 'route_0001.joblib')
        context = ns['RouteContext'](1)
        frame = context.namespace['horizon_dataset'](8, np.array([len(ns['times']) - 1]))
        frame = context.namespace['add_correlation_features'](frame, 8, artifact['lag_table'])
        replay = ns['predict_model'](artifact, frame)[0]
        expected = submission.set_index('id').loc['31_8', 'y_pred']
        require(np.isclose(replay, expected, rtol=1e-10), 'Portable model with preprocessing failed')
        print('All synthetic checks passed; backend:', ns['BACKEND'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='route_ridge_gpu_check_') as directory:
        run_checks(Path(directory), args.device)


if __name__ == '__main__':
    main()
