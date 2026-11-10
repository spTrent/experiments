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
for name in ('catboost', 'xgboost'):
    path = BASE / f'notebooks/clustering/{name}_rich_calibration.ipynb'
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    ns = {}
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        tree = ast.parse(cell.source)
        if any(isinstance(node, ast.Assert) for node in ast.walk(tree)):
            raise ValueError('assert запрещён')
        if any(isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and ast.get_docstring(node) for node in ast.walk(tree)):
            raise ValueError('docstring запрещён')
        if any(t.type == tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(cell.source).readline)):
            raise ValueError('Комментарий запрещён')
        if cell.id != notebook.cells[-1].id:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(cell.source, ns)
    parameters = ns['PARAMETERS'][ns['EXPERIMENTS'][0]['parameter_set']]
    for h in range(1, 9):
        if name == 'xgboost':
            expected = json.loads((BASE / f'notebooks/xgb_tuning/best_params/best_h{h}.json').read_text())['refit_params']
            if dict(parameters[str(h)]['params'], n_estimators=parameters[str(h)]['trees']) != expected:
                raise ValueError('Изменены лучшие параметры XGBoost')
        elif parameters[str(h)]['trees'] > 900 or parameters[str(h)]['params']['depth'] > 6:
            raise ValueError('Изменена выигравшая конфигурация CatBoost rich')
    ns.update(HORIZONS=(1, 2), CLUSTERS=(1, 2), MIN_CLUSTER_ROUTES=2, ROUTES=np.arange(5),
              CAL_GROUP_LABELS=np.array([0, 0, 1, 1, -1]), display=lambda *args: None)
    rows = []
    for day in pd.date_range('2025-10-04', periods=7):
        for h in (1, 2):
            for route in range(5):
                prediction = 20. + route * 5 + day.day + h
                rows.append(dict(route_id=route, calibration_group=int(ns['CAL_GROUP_LABELS'][route]), h=h,
                                 cutoff=day + pd.Timedelta(hours=10, minutes=30),
                                 timestamp=day + pd.Timedelta(hours=10, minutes=30 * (h + 1)),
                                 y=1.2 * prediction + 7., y_pred=prediction))
    frame = pd.DataFrame(rows)
    for strategy in ns['CALIBRATION_STRATEGIES']:
        factors = ns['fit_factors'](frame, strategy)
        result = ns['corrected'](frame, factors)
        if not np.isfinite(result).all() or (result < 0).any():
            raise ValueError('Некорректная калибровка ' + strategy)
        for h in (1, 2):
            mask = frame.h.eq(h)
            actual = ns['apply_factors'](frame.loc[mask, 'y_pred'].to_numpy(), h, frame.loc[mask, 'route_id'].to_numpy(), factors)
            if not np.allclose(result[mask], actual):
                raise ValueError('Разные поправки при оценке и финальном inference')
        if strategy == 'affine' and not np.allclose(result, frame.y, atol=.001):
            raise ValueError('Affine не восстановила известные наклон и сдвиг')
        json.dumps(factors)
    select = frame.copy()
    select[['cutoff', 'timestamp']] += pd.Timedelta(days=7)
    with tempfile.TemporaryDirectory() as directory:
        calls = []

        def read_phase(stage, variant):
            calls.append(stage)
            if stage not in ('calibration_fit', 'calibration_select'):
                raise ValueError('Test прочитан при выборе калибровки')
            return frame.copy() if stage == 'calibration_fit' else select.copy()

        ns.update(RUN=Path(directory), read_phase=read_phase, VAL_START=pd.Timestamp('2025-10-04'),
                  CAL_SPLIT=pd.Timestamp('2025-10-11'), TEST_START=pd.Timestamp('2025-10-18'))
        with contextlib.redirect_stdout(io.StringIO()):
            frozen = ns['select_calibration']()
        if set(calls) != {'calibration_fit', 'calibration_select'} or frozen['test_used']:
            raise ValueError('Выбор использовал test')
        if set(frozen['comparison_factors']['cat_k1']) != set(ns['CALIBRATION_STRATEGIES']):
            raise ValueError('Не все методы заморожены до test')
        times = pd.date_range('2025-09-01', periods=4000, freq='30min')

        def calibration_data(h, origins, segmentation):
            route_index = np.tile(np.arange(5), len(origins))
            y = np.repeat(100 + origins * .01, 5) + route_index
            return dict(origins=origins, route_index=route_index, y=y)

        ns.update(TIMES=times, STEP=pd.Timedelta(minutes=30),
                  segment_routes=lambda boundary, stage: {}, calibration_data=calibration_data,
                  fit_model=lambda *args: {}, predict_model=lambda bundle, data, segmentation: np.full(len(data['y']), 95.))
        with contextlib.redirect_stdout(io.StringIO()):
            summary = ns['phase_forecasts']('test', ns['TEST_START'], ns['TEST_START'], pd.Timestamp('2025-10-19'), frozen)
        if set(summary.strategy) != set(ns['CALIBRATION_STRATEGIES']) | {'selected'} or set(summary.period) != {'train', 'test'}:
            raise ValueError('В сводной таблице отсутствуют методы или train/test')
        if summary[['WAPE', 'WAPE + |Relative Bias|']].isna().any().any():
            raise ValueError('Пустые метрики сравнения калибровок')
    print(name, ': параметры, формат notebook, 18 калибровок, affine, fallback, inference, выбор без test: OK')
