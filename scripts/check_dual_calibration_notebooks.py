import ast
import contextlib
import io
import json
import tempfile
import tokenize
import types
import zipfile
from pathlib import Path
import nbformat
import numpy as np
import pandas as pd


class ProbeMeta:
    def set_param(self, params):
        if params['device'] != 'cuda:0':
            raise ValueError('CPU inference')

    def save_config(self):
        return json.dumps(dict(learner=dict(generic_param=dict(device='cuda:0'))))

    def inplace_predict(self, x, **kwargs):
        return x[:, 3] / np.maximum(x[:, 0], .001)


for filename in ('xgboost_oof_smoothed_dual.ipynb', 'xgboost_oof_meta_dual.ipynb'):
    notebook = nbformat.read(Path('clustering') / filename, as_version=4)
    nbformat.validate(notebook)
    ns = {}
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        tree = ast.parse(cell.source)
        if any(isinstance(n, ast.Assert) for n in ast.walk(tree)) or any(t.type == tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(cell.source).readline)):
            raise ValueError('Комментарии/assert запрещены')
        if any(isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and ast.get_docstring(n) for n in ast.walk(tree)):
            raise ValueError('Docstring запрещён')
        if cell.id != notebook.cells[-1].id:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(cell.source, ns)
    times = pd.date_range('2025-07-01', '2025-11-01 10:30', freq='30min')
    routes = np.arange(4)
    values = (100 + np.arange(len(times))[:, None] * .01 + routes[None, :] * 10).astype(np.float32)
    requests = pd.DataFrame([dict(id=r * 2 + h, route_id=r, timestamp=times[-1] + pd.Timedelta(minutes=30 * h)) for r in routes for h in (1, 2)]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_start = pd.Timestamp('2025-10-18')
    val_start = test_start - pd.Timedelta(days=28)
    fits = []
    meta_targets = []
    ns['cp'] = types.SimpleNamespace(asarray=np.asarray, float32=np.float32, float64=np.float64, ndarray=np.ndarray,
        asnumpy=lambda a: a, sum=np.sum, mean=np.mean, exp2=np.exp2, clip=np.clip,
        get_default_memory_pool=lambda: types.SimpleNamespace(free_all_blocks=lambda: None))

    def segment(boundary, stage):
        ns['FEATURE_NS'] = dict(boundary=str(boundary))
        return dict(boundary=str(boundary), scale=np.full(4, 100.), labels={1: np.zeros(4, dtype=int), 8: np.array([0, 0, 1, 1])})

    def features(h, origins, segmentation):
        return np.column_stack([values[origins].ravel() / 100., np.tile(routes, len(origins))]).astype(np.float32), ['signal', 'rich_route_index']

    def calibration_data(h, origins, segmentation):
        ns['ACTIVE_H'] = h
        x, names = features(h, origins, segmentation)
        return dict(x=x, names=names, origins=origins, route_index=np.tile(routes, len(origins)), y=values[origins + h].ravel().astype(np.float64))

    def fit_model(data, segmentation, family, k, settings):
        if times[data['origins'] + ns['ACTIVE_H']].max() >= pd.Timestamp(segmentation['boundary']):
            raise ValueError('Обучение использует будущие метки')
        fits.append((ns['ACTIVE_H'], segmentation['boundary'], k))
        return dict(feature_names=data['names'])

    def predict_model(bundle, data, segmentation):
        return data['x'][:, 0].astype(np.float64) * 100 / (1 + .1 * data['route_index'])

    def matrix(data, **kwargs):
        if not isinstance(data, np.ndarray) or not isinstance(kwargs['label'], np.ndarray) or not isinstance(kwargs['weight'], np.ndarray):
            raise ValueError('Не переданы GPU-данные метамодели')
        if kwargs['feature_types'][-2:] != ['c', 'c']:
            raise ValueError('Категории метамодели не зафиксированы')
        meta_targets.append(kwargs['label'].copy())
        return data

    def train_meta(params, matrix, num_boost_round):
        if params['device'] != 'cuda:0' or params['objective'] != 'reg:squarederror':
            raise ValueError('Неверная конфигурация метамодели')
        return ProbeMeta()

    ns.update(TIMES=times, ROUTES=routes, VALUES={'target_1h': values, **{f'status_{i}': values * i for i in (1, 2, 3)}}, REQUESTS=requests,
        ROLL={w: pd.DataFrame(values).rolling(w, min_periods=1).mean().to_numpy(dtype=np.float32) for w in (4, 48)},
        VAL_START=val_start, TEST_START=test_start, TEST_END=pd.Timestamp('2025-11-01'),
        HORIZONS=(1, 2), MODEL_CLUSTERS=(1, 8), STEP=pd.Timedelta(minutes=30),
        segment_routes=segment, features=features, calibration_data=calibration_data,
        fit_model=fit_model, predict_model=predict_model, display=lambda *args: None,
        xgb=types.SimpleNamespace(QuantileDMatrix=matrix, train=train_meta))
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / 'run'
        root.mkdir()

        def initialize():
            ns.update(RUN=root, CAL_SPLIT=val_start + pd.Timedelta(days=21), CAL_GROUP_LABELS=np.array([0, 0, 1, 1]))

        ns['initialize_calibration'] = initialize
        with contextlib.redirect_stdout(io.StringIO()):
            ns['run_dual_submissions']()
        frozen = json.loads((root / 'calibration_frozen.json').read_text())
        if frozen['test_used'] or pd.Timestamp(frozen['selection_end_exclusive']) != test_start:
            raise ValueError('Test использован для выбора')
        eval_file = root / 'evaluation_complete.json'
        if not json.loads(eval_file.read_text())['test_evaluated_before_full_refit']:
            raise ValueError('Нет оценки до полного refit')
        for scope in ('train_val', 'full'):
            frame = pd.read_csv(root / f'submission_{scope}.csv')
            if not frame.id.equals(requests.id) or not np.isfinite(frame.y_pred).all() or (frame.y_pred < 0).any():
                raise ValueError('Некорректный сабмит ' + scope)
        train_val_files = list(root.glob('base_train_val_h*.joblib'))
        for path in train_val_files:
            if ns['joblib'].load(path)['fit_boundary'] != str(test_start):
                raise ValueError('Train+val checkpoint обучен на test')
        full_files = list(root.glob('base_full_h*.joblib'))
        if len(full_files) != 2 or not fits:
            raise ValueError('Нет полного refit')
        with zipfile.ZipFile(root.with_suffix('.zip')) as archive:
            names = {Path(p).name for p in archive.namelist()}
            if not {'submission_train_val.csv', 'submission_full.csv', 'independent_metrics.csv', 'protocol.json', 'evaluation_note.txt'} <= names:
                raise ValueError('Финальный ZIP неполный')
        if ns['CORRECTION_MODE'] == 'meta':
            if frozen['selected']['kind'] != 'meta':
                raise ValueError('Синтетический сценарий не покрыл выбор метамодели')
            frame = ns['read_oof']('train_val', frozen['selected']['variant'])
            ns['FEATURE_NS'] = dict(boundary=str(val_start))
            before, columns = ns['meta_features'](frame)
            if len(columns) != 12 or any('route' in c and c != 'meta_log_scale' for c in columns):
                raise ValueError('Метамодель не упрощена до12 признаков без ID маршрута')
            if any(c['max_depth'] > 2 or c['trees'] > 150 for c in ns['META_CONFIGS']):
                raise ValueError('Метамодель слишком сложная')
            first = frame[frame.cutoff.eq(frame.cutoff.min())].copy()
            causal_before, _ = ns['meta_features'](first)
            position = times.get_loc(first.cutoff.min())
            saved = {key: array[position + 1:].copy() for key, array in ns['VALUES'].items()}
            for key in ns['VALUES']:
                ns['VALUES'][key][position + 1:] += 100000
            causal_after, _ = ns['meta_features'](first)
            for key, array in saved.items():
                ns['VALUES'][key][position + 1:] = array
            if not np.array_equal(causal_before, causal_after):
                raise ValueError('Будущие значения влияют на метапризнаки')
            changed = frame.copy()
            changed.y *= 1000
            after, _ = ns['meta_features'](changed)
            if not np.array_equal(before, after) or 'y' in columns or not meta_targets:
                raise ValueError('Метапризнаки читают таргет')
            raw_prediction = ns['predict_correction'](ns['joblib'].load(root / 'correction_train_val.joblib'), frame)[0]
            altered_prediction = ns['predict_correction'](ns['joblib'].load(root / 'correction_train_val.joblib'), changed)[0]
            if not np.array_equal(raw_prediction, altered_prediction):
                raise ValueError('Предсказание калибратора читает ответы test')
    print(filename, ': полный синтетический сценарий, OOF, выбор до test, 2 refit-режима, CUDA-контракт метамодели, 4 сабмита и ZIP: OK')
print('Настоящее GPU-обучение не выполнялось; GPU адаптеры и базовые модели заменены тестовыми.')
