import ast
import contextlib
import io
import json
import tempfile
import tokenize
import types
from pathlib import Path
import nbformat
import numpy as np
import pandas as pd

notebook = nbformat.read('notebooks/clustering/xgboost_rich_gpu_shrinkage.ipynb', as_version=4)
nbformat.validate(notebook)
ns = {}
for cell in notebook.cells:
    if cell.cell_type != 'code':
        continue
    tree = ast.parse(cell.source)
    if any(isinstance(node, ast.Assert) for node in ast.walk(tree)) or any(t.type == tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(cell.source).readline)):
        raise ValueError('Комментарии/assert запрещены')
    if any(isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and ast.get_docstring(node) for node in ast.walk(tree)):
        raise ValueError('Docstring запрещён')
    if cell.id != notebook.cells[-1].id:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(cell.source, ns)
if ns['VAL_DAYS'] != 28 or ns['CAL_FIT_DAYS'] != 21 or len(ns['CALIBRATION_STRATEGIES']) != 15:
    raise ValueError('Неверный протокол')
rows = [dict(route_id=r, calibration_group=0, h=h, cutoff=day, timestamp=day + pd.Timedelta(minutes=30 * h),
             y=100. + day.day, y_pred=100.) for day in pd.date_range('2025-09-20', periods=21) for r in (0, 1) for h in (1, 2)]
frame = pd.DataFrame(rows)
ns.update(HORIZONS=(1, 2))
long_ratio = frame.y.sum() / frame.y_pred.sum()
for half_life in (3, 7, 14):
    age = (frame.cutoff.max() - frame.cutoff).dt.total_seconds().to_numpy() / 86400
    weight = np.exp2(-age / half_life)
    fresh = float(np.sum(frame.y * weight) / np.sum(frame.y_pred * weight))
    for alpha in (.25, .5, .75):
        factors = ns['fit_factors'](frame, f'shrink_recent_{half_life}_{alpha}')
        if not np.isclose(factors['1'], np.clip((1 - alpha) * long_ratio + alpha * fresh, .8, 1.25)):
            raise ValueError('Неверная усадка коэффициента')

calls = []
with tempfile.TemporaryDirectory() as directory:
    ns.update(RUN=Path(directory))

    def phase(stage, boundary, start, end, frozen):
        calls.append((boundary, start, end))
        for h in ns['HORIZONS']:
            for name, k in ns['calibration_variants']():
                frame[frame.h.eq(h)].iloc[:2].to_parquet(ns['RUN'] / f'{stage}_h{h}_{name}.parquet', index=False)
        return pd.DataFrame([dict(stage=stage)])

    ns['original_shrinkage_phase'] = phase
    start, end = pd.Timestamp('2025-09-20'), pd.Timestamp('2025-10-11')
    ns['phase_forecasts']('calibration_fit', start, start, end)
    if calls != [(start + pd.Timedelta(days=7 * i), start + pd.Timedelta(days=7 * i), start + pd.Timedelta(days=7 * (i + 1))) for i in range(3)]:
        raise ValueError('Неверные недельные OOF-границы')
    if len(pd.read_parquet(Path(directory) / 'calibration_fit_h1_xgb_k8.parquet')) != 6:
        raise ValueError('OOF-блоки не собраны')

ns['cp'] = types.SimpleNamespace(asarray=np.asarray, float32=np.float32, concatenate=np.concatenate,
                                 ndarray=np.ndarray, asnumpy=lambda a: a,
                                 get_default_memory_pool=lambda: types.SimpleNamespace(free_all_blocks=lambda: None))
ns.update(GPU_BATCH_ROWS=2, ACTIVE_H=1, ACTIVE=ns['EXPERIMENTS'][0])
names = ['rich_route_index', 'signal']
train = dict(x=np.array([[0., 1.], [1., 2.], [0., 3.], [1., 4.]], dtype=np.float32), names=names,
             route_index=np.array([0, 1, 0, 1]), y=np.array([10., 40., 30., 80.]))
seg = dict(scale=np.array([10., 20.]), labels={8: np.array([0, 1])})
matrix_calls, train_calls = [], []


def quantile(iterator, **kwargs):
    batches = []
    for repeat in range(2):
        iterator.reset()
        current = []
        while iterator.next(lambda **batch: current.append(batch)):
            pass
        if not current or current[0]['feature_types'][-1] != 'c':
            raise ValueError('Нет CUDA-батчей или категории маршрута')
        batches.append(np.concatenate([b['data'] for b in current]))
    if not np.array_equal(batches[0], batches[1]):
        raise ValueError('DataIter.reset меняет данные')
    matrix_calls.append(batches[0])
    return current


class ProbeBooster:
    def save_config(self):
        return json.dumps(dict(learner=dict(generic_param=dict(device='cuda:0'))))

    def set_param(self, params):
        if params['device'] != 'cuda:0':
            raise ValueError('CPU inference')

    def inplace_predict(self, x, **kwargs):
        return np.ones(len(x), dtype=np.float32)


def fit(params, matrix, num_boost_round):
    if params['device'] != 'cuda:0' or params['objective'] != 'reg:absoluteerror':
        raise ValueError('Неверный GPU objective')
    train_calls.append(num_boost_round)
    return ProbeBooster()


ns['xgb'] = types.SimpleNamespace(QuantileDMatrix=quantile, train=fit)
with contextlib.redirect_stdout(io.StringIO()):
    model = ns['fit_model'](train, seg, 'cat', 8, {})
prediction = ns['predict_model'](model, train, seg)
if not np.array_equal(prediction, seg['scale'][train['route_index']]) or len(matrix_calls) != 2:
    raise ValueError('Неверная GPU-предикция или группа')
if train_calls != [ns['PARAMETERS'][ns['ACTIVE']['parameter_set']]['1']['trees']] * 2:
    raise ValueError('Изменены выбранные числа деревьев')
print('Notebook: формат, параметры, 15 правил, формула усадки, 3 causal OOF-блока, DataIter/reset, GPU fit/predict и категории: OK. CUDA заменена тестовыми адаптерами, настоящее GPU-обучение не запускалось.')
