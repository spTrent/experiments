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
path = BASE / 'notebooks/clustering/cluster_catboost_calibration.ipynb'
nb = nbformat.read(path, as_version=4)
nbformat.validate(nb)
for cell in nb.cells:
    if cell.cell_type != 'code':
        continue
    tree = ast.parse(cell.source)
    compile(tree, str(path), 'exec')
    if any(isinstance(node, ast.Assert) for node in ast.walk(tree)):
        raise ValueError('Обнаружен assert')
    if any(isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and ast.get_docstring(node) for node in ast.walk(tree)):
        raise ValueError('Обнаружен docstring')
    if any(token.type == tokenize.COMMENT for token in tokenize.generate_tokens(io.StringIO(cell.source).readline)):
        raise ValueError('Обнаружен комментарий')


class DummyCatBoost:
    calls = 0

    def __init__(self, **settings):
        if settings['task_type'] != 'GPU':
            raise ValueError('CatBoost должен использовать GPU')

    def fit(self, x, y, sample_weight=None):
        self.value = 0.92 * float(np.average(y, weights=sample_weight))
        DummyCatBoost.calls += 1
        return self

    def predict(self, x):
        return np.full(len(x), self.value)


ns = {}
for index in (1, 3, 4):
    exec(nb.cells[index].source, ns)
ns.update(HORIZONS=(1, 2), CLUSTERS=(1, 2), MIN_CLUSTER_ROUTES=2, THREADS=1, CREATE_ARCHIVE=False,
          require_gpu=lambda: None, CatBoostRegressor=DummyCatBoost, display=lambda *args, **kwargs: None,
          STEP=pd.Timedelta(minutes=30))
example = pd.DataFrame(dict(h=[1, 1, 2, 2], y=[2., 5., 3., 8.], y_pred=[1., 4., 3., 6.]))
factors = ns['fit_factors'](example, 'global')
if not all(np.isclose(factor, 1.25) for factor in factors.values()):
    raise ValueError('Не работает ограничение множителя')
example['y'] *= 0.9
factors = ns['fit_factors'](example, 'global')
score = ns['metrics'](example.y.to_numpy(), ns['corrected'](example, factors))
if abs(score['relative_bias']) > 1e-12:
    raise ValueError('Общий коэффициент не балансирует обучающий объём')
for candidate in np.linspace(0.8, 1.25, 101):
    trial = ns['metrics'](example.y.to_numpy(), example.y_pred.to_numpy() * candidate)
    if trial['score'] < score['score'] - 1e-12:
        raise ValueError('Отношение сумм не минимизирует калибровочную метрику')
if not np.array_equal(ns['corrected'](example, ns['fit_factors'](example, 'raw')), example.y_pred.to_numpy()):
    raise ValueError('Raw изменяет прогнозы')
local = ns['fit_factors'](example, 'horizon')
half = ns['fit_factors'](example, 'horizon_shrink50')
for h in ns['HORIZONS']:
    expected = np.clip((example.y.sum() / example.y_pred.sum() + ns['scalar_ratio'](example[example.h == h])) / 2, *ns['SCALE_BOUNDS'])
    if not np.isclose(half[str(h)], expected):
        raise ValueError('Неверная усадка по горизонтам')

group_example = pd.DataFrame([
    dict(route_id=r, calibration_group=g, h=h, y=value, y_pred=10.)
    for g, routes, value in [(0, [1, 2], 12.), (1, [3, 4], 8.), (-1, [5], 11.)]
    for r in routes for h in (1, 2)
])
group_factors = ns['fit_factors'](group_example, 'group')
if not np.isclose(group_factors['groups']['0']['1'], 1.2) or not np.isclose(group_factors['groups']['1']['1'], 0.8) or '-1' in group_factors['groups']:
    raise ValueError('Неверные групповые поправки или fallback')
group_half = ns['fit_factors'](group_example, 'group_shrink50')
if not np.isclose(group_half['groups']['0']['1'], (1.2 + group_half['1']) / 2):
    raise ValueError('Неверная усадка групп')
group_example.loc[(group_example.calibration_group == 0) & (group_example.h == 2), 'y'] = 10.
gh = ns['fit_factors'](group_example, 'group_horizon')
if not np.isclose(gh['groups']['0']['2'], 1.) or not np.isclose(gh['groups']['0']['1'], 1.2):
    raise ValueError('Горизонтные поправки групп смешаны')
ns['ROUTES'] = np.array([1, 2, 3, 4, 5])
ns['CAL_GROUP_LABELS'] = np.array([0, 0, 1, 1, -1])
frame_h1 = group_example[group_example.h == 1]
fast_pred = ns['apply_factors'](frame_h1.y_pred.to_numpy(), 1, np.arange(5), gh)
if not np.allclose(fast_pred, ns['corrected'](frame_h1, gh)):
    raise ValueError('Пакетный прогноз не применяет групповую карту')
zero_group = group_example.copy()
zero_group.loc[zero_group.calibration_group == 0, 'y_pred'] = 0
if '0' in ns['fit_factors'](zero_group, 'group')['groups']:
    raise ValueError('Нулевая сумма группы не использует fallback')

with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    times = pd.date_range('2025-07-28', '2025-11-01 10:30', freq='30min')
    routes = np.array([7, 11, 50, 77, 90, 130])
    ticks = np.arange(len(times))[:, None]
    target = (routes[None, :] + 20) * (1.2 + 0.3 * np.sin(ticks / 48 * 2 * np.pi + np.arange(len(routes))))
    history = pd.DataFrame(dict(timestamp=np.repeat(times, len(routes)), route_id=np.tile(routes, len(times)), target_1h=target.ravel()))
    for i in range(1, 7):
        history[f'status_{i}'] = history.target_1h * i
    requests = pd.DataFrame([dict(id=i * 2 + h, route_id=route, timestamp=times[-1] + h * ns['STEP'])
                             for i, route in enumerate(routes) for h in ns['HORIZONS']]).sample(frac=1, random_state=5).reset_index(drop=True)
    history.to_parquet(root / 'train.parquet', index=False)
    requests.to_parquet(root / 'test.parquet', index=False)
    ns.update(TRAIN_PATH=str(root / 'train.parquet'), TEST_PATH=str(root / 'test.parquet'), OUTPUT_ROOT=str(root))
    with contextlib.redirect_stdout(io.StringIO()), ns['threadpool_limits'](limits=1):
        ns['initialize_calibration']()
        stable_groups = ns['CAL_GROUP_LABELS'].copy()
        try:
            ns['phase_forecasts']('test', ns['TEST_START'], ns['TEST_START'], ns['TEST_END'])
        except ValueError:
            pass
        else:
            raise ValueError('Test открыт до фиксации')
        ns['phase_forecasts']('calibration_fit', ns['VAL_START'], ns['VAL_START'], ns['CAL_SPLIT'])
        ns['phase_forecasts']('calibration_select', ns['CAL_SPLIT'], ns['CAL_SPLIT'], ns['TEST_START'])
        calls = DummyCatBoost.calls
        ns['phase_forecasts']('calibration_fit', ns['VAL_START'], ns['VAL_START'], ns['CAL_SPLIT'])
        if DummyCatBoost.calls != calls:
            raise ValueError('Не работает продолжение OOF')
        real_read_phase = ns['read_phase']
        def read_before_test(stage, name):
            if stage == 'test':
                raise ValueError('Выбор калибровки читает test')
            return real_read_phase(stage, name)
        ns['read_phase'] = read_before_test
        frozen = ns['select_calibration']()
        ns['read_phase'] = real_read_phase
        before = json.loads((ns['RUN'] / 'calibration_frozen.json').read_text())
        ns['evaluate_calibration'](frozen)
        if before != json.loads((ns['RUN'] / 'calibration_frozen.json').read_text()):
            raise ValueError('Оценка test меняет выбранную процедуру')
        calls = DummyCatBoost.calls
        ns['evaluate_calibration'](frozen)
        if DummyCatBoost.calls != calls:
            raise ValueError('Не работает продолжение оценки')
        for stage, boundary in [('calibration_fit', ns['VAL_START']), ('calibration_select', ns['CAL_SPLIT']), ('test', ns['TEST_START'])]:
            for h in ns['HORIZONS']:
                rows = json.loads((ns['RUN'] / f'{stage}_h{h}_cat_k1_metrics.json').read_text())
                expected = int((ns['TIMES'][h:] < boundary).sum()) * len(routes)
                observed = next(row['n'] for row in rows if row['period'] == 'train' and row['strategy'] == 'raw')
                if observed != expected:
                    raise ValueError('Обучение пересекает проверочный период или теряет старты')
        ns['finalize_calibration'](frozen)
        if not np.array_equal(stable_groups, ns['CAL_GROUP_LABELS']):
            raise ValueError('Группы калибровки изменились при переобучении базовых моделей')
        selected_strategies = set(pd.read_csv(ns['RUN'] / 'calibration_selection.csv').strategy)
        if not {'group', 'group_shrink50', 'group_horizon', 'group_horizon_shrink50'}.issubset(selected_strategies):
            raise ValueError('Групповые способы не участвуют в выборе')
        for name in ('submission.csv', 'submission_raw.csv'):
            submission = pd.read_csv(ns['RUN'] / name)
            if submission.id.tolist() != requests.id.tolist() or not np.isfinite(submission.y_pred).all() or (submission.y_pred < 0).any():
                raise ValueError('Некорректный submission')
        final = json.loads((ns['RUN'] / 'final_calibration.json').read_text())
        if final['strategy'] != frozen['variants'][frozen['winner']]['strategy']:
            raise ValueError('После test изменён способ калибровки')
        if before != json.loads((ns['RUN'] / 'calibration_frozen.json').read_text()):
            raise ValueError('Полный refit перезаписал оценочные коэффициенты')
        summary = pd.read_csv(ns['RUN'] / 'independent_calibration_metrics.csv')
        if set(summary.period) != {'train', 'test'} or not {'raw', 'selected'}.issubset(set(summary.strategy)):
            raise ValueError('Нет сравнения train/test до и после калибровки')
print('OK: notebook/code restrictions, bounded scalar optimum, shrinkage, pre-test selection, rolling temporal boundaries, frozen test coefficients, resume, final refit and two submissions. No real-data training; CatBoost replaced with a test double.')
