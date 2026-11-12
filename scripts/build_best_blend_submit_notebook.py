import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = json.loads((root / 'stacking/blending/best_models_blend_kaggle.ipynb').read_text())
original = {c['id']: ''.join(c['source']) for c in source['cells']}
runtime = (root / 'scripts/cluster_experiment/best_blend_submit.py').read_text()
cells = []


def add(kind, text):
    if kind == 'code':
        text = ast.unparse(ast.parse(text))
        compile(text, '<submit>', 'exec')
    cell = dict(cell_type=kind, id=f'blend-submit-{len(cells)}', metadata={}, source=(text.strip() + '\n').splitlines(True))
    if kind == 'code':
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)


def definitions(text, names):
    return '\n\n'.join(ast.unparse(n) for n in ast.parse(text).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names)


def inner(name):
    tree = ast.parse(original[name])
    return ast.literal_eval(tree.body[1].value.args[0].args[0])


def replace(text, before, after):
    if before not in text:
        before = ast.unparse(ast.parse(before))
    if before not in text:
        raise ValueError(f'Не найден шаблон: {before}')
    return text.replace(before, after)


def component(name, text):
    add('code', f"{name} = dict(globals())\nexec(compile({text!r}, {name!r}, 'exec'), {name})")


add('markdown', '''# Два сабмита блендинга

Победитель до локального test: `cat_d4__ridge`, исторический score 0.317416. Этот ноутбук не повторяет OOF-обучение и перебор метамоделей.

Подключите через Add Input:
1. Датасет `session6_blend_input` из прежнего запуска.
2. Датасет `xgb_rich_k8_blend_input` из прежнего запуска.
3. **Output завершённого blend-best-2** через Notebook Output либо распакованный датасет из `best_blend_bf2ef3a96ffc.zip`. Нужна вся папка, включая `session6/models/early_anchor/`.

Если текущая интерактивная сессия ещё жива, output может лежать в `/kaggle/working`; он также найдётся автоматически. Одного скачанного ipynb недостаточно: веса раннего Session6 и parquet-прогнозы находятся в output.

Включите GPU, как для предыдущего ноутбука. Обучение запускается пользователем в Kaggle. Готовые файлы: `/kaggle/working/submission_train_val.csv` и `/kaggle/working/submission_full_history.csv`. Автоматической отправки на соревнование нет.

Запуск последовательно собирает два независимых комплекта моделей:
- **train_val**: все обучаемые статистики, базы, модели остатка Session6, веса бленда, метамодель и калибровка используют цели строго до 18 октября 2025. Для метамодели берутся только четыре прежние OOF-недели. Это дополнительный диагностический сабмит по явному запросу пользователя.
- **full_history**: базы обучаются на всей размеченной истории; метамодель и калибровка используют все доступные честные исторические прогнозы, включая прежний local test.

У обоих вариантов признаки будущего прогноза используют всю наблюдаемую историю до 1 ноября 10:30. В train_val факты после 18 октября доступны как входная история признаков, но не как обучающие цели или данные подбора параметров. EWMA-поправка Session6 в train_val использует только остатки до границы обучения. Каталоги моделей и кэшей раздельные; перед первым CSV заново оценивается local test 18–31 октября теми же моделями и калибраторами. Затем сохраняется первый CSV, и только после этого разрешён запуск полного переобучения. Метки test используются только для метрик, а не для обновления моделей train_val.

В режиме full_history все базовые обучаемые компоненты переобучаются до последнего доступного факта. Для метамодели используются готовые прогнозы до 18 октября и прогнозы локального test: каждый из них получен до обучения соответствующей базы на его целевых значениях. Настройки CatBoost глубины 4 и Ridge остаются зафиксированными. После включения test в финальный fit новой независимой оценки нет. Независимая от полного переобучения оценка test получена до переобучения; сам период ранее изучался.

Session6 для будущего восстанавливается по рецепту: переобучение CatBoost и LightGBM, две модели остатка и EWMA7. Базовый CatBoost Session6 сохраняет исходный отбор: первые `len(eligible) // 4` допустимых стартов отбрасываются, обучение идёт на оставшихся последних примерно 75%. Это правило применяется отдельно внутри train_val и full_history. LightGBM сохраняет свои пять утренних стартов. Остатки обучаются на прогнозах сохранённого раннего CatBoost. Это перенос рецепта на full history, а не гарантия сохранения локального score. Для моделей второго уровня полная доступная обучающая история — только честные исторические прогнозы, не предсказания баз на собственном train.

Прогнозы сохраняются по горизонтам. Повторный запуск с теми же входами в той же рабочей папке переиспользует готовые горизонты. В новой сессии для продолжения скопируйте папку `blend_submit_*` в `/kaggle/working`.''')
add('code', original['data-contract'])
add('code', runtime)
add('code', '''import cupy as cp
import lightgbm as lgb
import torch
import joblib
import xgboost as xgb
from catboost import CatBoostRegressor
from catboost.utils import get_gpu_device_count
from threadpoolctl import threadpool_limits

EVALUATION_DIR = None
SESSION_DATA_DIR = None
XGB_DATA_DIR = None
if cp.cuda.runtime.getDeviceCount() < 1 or get_gpu_device_count() < 1:
    raise RuntimeError('Нужен GPU Kaggle с CuPy, cuML, CatBoost, XGBoost и LightGBM')
initialize_submission(EVALUATION_DIR, SESSION_DATA_DIR, XGB_DATA_DIR)
fit_report = refit_report
''')

s = inner('session_runtime')
s = replace(s, "original_target[eval_origins + h].ravel()", "observed_query(h).ravel()")
s = replace(s, "target[eval_origins + h].ravel()", "np.full(len(eval_origins) * len(routes), np.nan)")
s = replace(s, "fit = frame.cutoff < fit_before", "fit = frame.fit_y.notna()")
s += '\n' + definitions(runtime, {'session_full', 'session_refit_report'})
s = replace(s, 'known = frame.timestamp < cutoff - pd.Timedelta(hours=4)', 'known = (frame.timestamp < cutoff - pd.Timedelta(hours=4)) & fit_mask')
s += '\nSESSION_TRAIN_REPORTS = {}\nreport = session_refit_report\nshow_fit = lambda: None\n'
component('session_runtime', s)
add('code', "session_future = session_runtime['session_full']()\nfor name in ['means', 'level24', 'extra_signals', 'hub_signal', 'long_week', 'deviations']:\n    session_runtime.pop(name, None)\ngc.collect()")

s = inner('cat_runtime')
s = replace(s, "query_origins = origins[(TIMES[origins] >= start) & (TIMES[origins] < stop) &\n                                    (TIMES[origins].hour == 10) & (TIMES[origins].minute == 30)]", "query_origins = QUERY_ORIGINS")
s = replace(s, 'actual = VALUES[TARGET][query_origins + h]', 'actual = observed_query(h)')
s = replace(s, "del model, train_x, query_x, train_pred", "model.save_model(str(folder / f'h{h}_group{group}.cbm'))\n                joblib.dump(segmentation, folder / 'segmentation.joblib')\n                del model, train_x, query_x, train_pred")
component('cat_runtime', s)
add('code', "cat_future = cat_runtime['fit_rich_cat'](FULL_BOUNDARY, FULL_BOUNDARY + pd.Timedelta(days=1), 'future')")

gpu = (root / 'scripts/cluster_experiment/xgb_gpu_shrinkage.py').read_text()
s = "import time\nimport xgboost as xgb\nimport cupy as cp\nGPU_BATCH_ROWS = 65536\nGPU_MAX_BIN = 256\nGPU_QUANTILE_BATCHES = 8\n"
s += definitions(gpu, {'GPUTrainingIterator', 'gpu_features', 'fit_model', 'predict_model'})
s += '\n' + definitions(runtime, {'fit_rich_xgb_full'})
add('code', f"xgb_rich_runtime = dict(cat_runtime)\nexec(compile({s!r}, 'rich_xgb_full', 'exec'), xgb_rich_runtime)\nxgb_future = xgb_rich_runtime['fit_rich_xgb_full']()")

s = inner('rf_runtime')
s = replace(s, "frame['y'] = target[batch + h].ravel()", "frame['y'] = past(target, batch + h).ravel()")
s = replace(s, 'np.isfinite(frame.profile28) & np.isfinite(frame.y)', 'np.isfinite(frame.profile28)')
s = replace(s, 'query_origins = np.flatnonzero((minute == 630) & (times >= start) & (times + h * STEP < stop))', 'query_origins = QUERY_ORIGINS')
s = replace(s, 'del model, train_matrix, query_matrix, train, query, train_pred, query_pred', "joblib.dump(dict(model=model, imputer=imputer, median=median if bool(spec.route_median) else None, lag_table=lag_table, spec=spec), folder / f'h{h}_model.joblib')\n            del model, train_matrix, query_matrix, train, query, train_pred, query_pred")
component('rf_runtime', s)
add('code', "rf_future = rf_runtime['fit_rf'](FULL_BOUNDARY, FULL_BOUNDARY + pd.Timedelta(days=1), 'future')")

s = inner('clean_xgb_runtime')
s = replace(s, "query = origins[(TIMES[origins] >= start) & (TIMES[origins + h] < stop) &\n                            (TIMES[origins].hour == 10) & (TIMES[origins].minute == 30)]", 'query = QUERY_ORIGINS')
s = replace(s, 'actual = VALUES[TARGET][query + h].ravel()', 'actual = observed_query(h).ravel()')
s = replace(s, 'del model, query_matrix, train_prediction', "model.save_model(str(folder / f'h{h}_model.ubj'))\n            del model, query_matrix, train_prediction")
component('clean_xgb_runtime', s)
add('code', "clean_future = clean_xgb_runtime['fit_clean_xgb'](FULL_BOUNDARY, FULL_BOUNDARY + pd.Timedelta(days=1), 'future')")
add('code', original['meta-runtime'])
add('code', "finish_submission(dict(session6=session_future, rf=rf_future, cat_rich=cat_future, xgb_rich=xgb_future, xgb_clean=clean_future))")
workflow = cells[4:]
cells = cells[:4]
for scope, label in [('train_val', 'Обучение на train + val'), ('full_history', 'Обучение на всей истории')]:
    add('markdown', f'## {label}\n\nОтдельный комплект моделей и файл `submission_{scope}.csv`. В train_val после обучения выполняется новая оценка test, затем сохраняется сабмит теми же моделями. Full_history запускается только после наличия оценки и первого CSV. Справочные метрики прошлого запуска сохранены отдельно.')
    add('code', f"enter_submission_scope({scope!r})")
    for cell in workflow:
        add(cell['cell_type'], ''.join(cell['source']))
    add('code', "for name in ['session_runtime', 'cat_runtime', 'rf_runtime', 'clean_xgb_runtime', 'xgb_rich_runtime']:\n    globals().pop(name, None)\ngc.collect()\ncp.get_default_memory_pool().free_all_blocks()")
digest = hashlib.sha256(json.dumps(cells, sort_keys=True).encode()).hexdigest()
cells[3]['source'].insert(0, f'SUBMIT_CODE_HASH = {digest!r}\n')
output = root / 'stacking/blending/best_blend_submission_kaggle.ipynb'
output.write_text(json.dumps(dict(nbformat=4, nbformat_minor=5, metadata=source['metadata'], cells=cells), ensure_ascii=False, indent=1) + '\n')
print(output)
