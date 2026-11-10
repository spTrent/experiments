import ast
import contextlib
import io
import tempfile
import tokenize
from pathlib import Path
import nbformat
import numpy as np
import pandas as pd

path = Path('notebooks/clustering/catboost_rich_train_val_submission.ipynb')
notebook = nbformat.read(path, as_version=4)
nbformat.validate(notebook)
ns = {}
for cell in notebook.cells:
    if cell.cell_type != 'code':
        continue
    tree = ast.parse(cell.source)
    if any(isinstance(n, ast.Assert) for n in ast.walk(tree)) or any(t.type == tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(cell.source).readline)):
        raise ValueError('Запрещённый комментарий или assert')
    if any(isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and ast.get_docstring(n) for n in ast.walk(tree)):
        raise ValueError('Запрещённый docstring')
    if cell.id != notebook.cells[-1].id:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(cell.source, ns)

times = pd.date_range('2025-10-01', '2025-11-01 10:30', freq='30min')
routes = np.arange(4)
boundary = pd.Timestamp('2025-10-18')
requests = pd.DataFrame([dict(id=i * 2 + h, route_id=r, timestamp=times[-1] + pd.Timedelta(minutes=h * 30)) for i, r in enumerate(routes) for h in (1, 2)]).sample(frac=1, random_state=42).reset_index(drop=True)
calls, feature_cutoffs = [], []


def segmentation(before, stage):
    if before != boundary:
        raise ValueError('Статистики используют local test')
    return dict(boundary=str(before), scale=np.ones(4), labels={8: np.zeros(4, dtype=int)})


def data(h, origins, seg):
    ns['ACTIVE_H'] = h
    return dict(origins=origins, route_index=np.tile(np.arange(4), len(origins)), y=np.repeat(100 + origins * .01 + h, 4))


def fit(train, seg, family, k, settings):
    h = ns['ACTIVE_H']
    if k != 8 or times[train['origins'] + h].max() >= boundary:
        raise ValueError('Неверное обучение')
    calls.append(h)
    return dict(feature_names=['probe'])


def predict(bundle, frame, seg):
    return np.full(len(frame['route_index']), 100.)


def features(h, origins, seg):
    feature_cutoffs.extend(times[origins])
    return np.ones((4, 1)), ['probe']


with tempfile.TemporaryDirectory() as directory:
    ns.update(load_inputs=lambda: None, segment_routes=segmentation, calibration_data=data, fit_model=fit,
              predict_model=predict, features=features, TIMES=times, ROUTES=routes, REQUESTS=requests,
              TEST_START=boundary, TEST_END=pd.Timestamp('2025-11-01'), RUN=Path(directory),
              STEP=pd.Timedelta(minutes=30), HORIZONS=(1, 2), CREATE_ARCHIVE=False, display=lambda *args: None)
    with contextlib.redirect_stdout(io.StringIO()):
        ns['run_train_val_submission']()
    calibrated = pd.read_csv(Path(directory) / 'submission.csv')
    raw = pd.read_csv(Path(directory) / 'submission_raw.csv')
    audit = pd.read_csv(Path(directory) / 'training_boundary_audit.csv')
    report = pd.read_csv(Path(directory) / 'train_val_metrics.csv')
    if not calibrated.id.equals(requests.id) or not np.allclose(calibrated.y_pred, raw.y_pred * 1.05):
        raise ValueError('Нарушено сопоставление id или множителя')
    if set(report.period) != {'train', 'test'} or not (pd.to_datetime(audit.last_training_target) < boundary).all():
        raise ValueError('Неверный аудит/отчёт')
    if feature_cutoffs != [times[-1], times[-1]]:
        raise ValueError('Для сабмита не использована свежая история')
    with contextlib.redirect_stdout(io.StringIO()):
        ns['run_train_val_submission']()
    if calls != [1, 2]:
        raise ValueError('Не работает возобновление checkpoint')
print('Train+val: границы fit, свежие признаки, train/test, raw/×1.05, порядок id и checkpoint: OK. Настоящее GPU-обучение заменено тестовой моделью.')
