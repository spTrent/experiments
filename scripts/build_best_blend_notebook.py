import ast
import hashlib
import json
from pathlib import Path
from textwrap import dedent


root = Path(__file__).resolve().parents[1]
folder = root / 'scripts/cluster_experiment'
notebooks = root / 'notebooks'
cells = []


def source(path, index):
    return ''.join(json.loads(path.read_text())['cells'][index]['source'])


def code(text):
    return ast.unparse(ast.parse(text))


def definitions(text, names):
    return '\n\n'.join(ast.unparse(node) for node in ast.parse(text).body
                      if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names)


def assignments(text, names):
    return '\n'.join(ast.unparse(node) for node in ast.parse(text).body
                     if isinstance(node, ast.Assign) and len(node.targets) == 1 and
                     isinstance(node.targets[0], ast.Name) and node.targets[0].id in names)


def add(kind, key, text):
    text = dedent(text).strip() + '\n'
    cell = dict(cell_type=kind, id=key, metadata={}, source=text.splitlines(True))
    if kind == 'code':
        compile(text, key, 'exec')
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)


def component(name, text):
    compile(text, name, 'exec')
    if any(isinstance(node, ast.Assert) for node in ast.walk(ast.parse(text))):
        raise ValueError(name)
    add('code', name, f"""{name} = dict(globals())
exec(compile(r'''
{text}
''', '{name}', 'exec'), {name})
""")


add('markdown', 'guide', """
# Блендинг Session6, Random Forest, CatBoost rich и двух XGBoost

Создайте два Kaggle Dataset из комплектов рядом с ноутбуком и подключите их через Add Input. Файлы в датасетах должны быть доступны в распакованном виде. Внутри каждого датасета нужные файлы лежат в одной папке, которая может быть вложенной.

1. **Session6.** Комплект `session6_blend_input.zip` содержит `comparison_meta.json`, `train_solo_track.parquet`, `test_solo_track.parquet` и `validation_predictions.parquet`.
2. **XGBoost rich k8.** Комплект `xgb_rich_k8_blend_input.zip` содержит `manifest.json`, `protocol.json`, `evaluation_complete.json` и 40 parquet-файлов. В них лежат raw-прогнозы на четыре OOF-недели и локальный test для восьми горизонтов.

Ноутбук найдёт оба набора внутри `/kaggle/input`. Если подключено несколько подходящих версий, укажите папки с файлами в `SESSION_DATA_DIR` и `XGB_DATA_DIR` в ячейке настроек.

Выберите GPU и образ Kaggle с рабочими CUDA, RAPIDS/cuML, CuPy, PyTorch, CatBoost, LightGBM, XGBoost и PyArrow. Запустите Save & Run All. RF, базовые CatBoost, XGBoost на очищенной истории и Gating Network используют GPU. LightGBM Session6, Ridge и небольшие метамодели CatBoost/XGBoost работают на CPU. Длительность зависит от GPU и числа маршрутов в кластерах. Прогнозы RF, CatBoost rich и XGBoost на очищенной истории сохраняются после каждого горизонта. Повторный запуск в той же рабочей директории с теми же входами и кодом переиспользует законченные горизонты. Для возобновления в новой сессии перенесите каталог результата в `/kaggle/working` до запуска.

В локальном тесте оцениваются прогнозы из 10:30 на h1–h8 за 18–31 октября 2025. Все настройки ансамбля выбираются на неделях до этого периода. Архивная оценка уже изучалась в проекте. Эксперимент сохраняет локальные прогнозы и таблицы. Новый конкурсный CSV требует отдельного полного переобучения всех выбранных компонентов.
""")

shared = (folder / 'best_blend_data.py').read_text()
bases = (folder / 'best_blend_bases.py').read_text()
meta = (folder / 'best_blend_meta.py').read_text()
add('code', 'data-contract', shared)
add('code', 'read-inputs', """
import cupy as cp
from cuml.ensemble import RandomForestRegressor
from catboost.utils import get_gpu_device_count
import lightgbm as lgb
import torch

if cp.cuda.runtime.getDeviceCount() < 1 or get_gpu_device_count() < 1 or not torch.cuda.is_available():
    raise RuntimeError('Включите GPU в Kaggle и проверьте CUDA для cuML, CatBoost и PyTorch.')
SESSION_DATA_DIR = None
XGB_DATA_DIR = None
initialize(SESSION_DATA_DIR, XGB_DATA_DIR)
""")

add('markdown', 'anchor-guide', """
## Session6 и объём горизонта

На четырёх OOF-неделях используем рецепт Session6. Базовые CatBoost и LightGBM обучаются один раз до 13 сентября. Прогнозы 13–19 сентября дают первые остатки. Затем две модели остатка CatBoost обучаются заново к каждой следующей неделе на доступных более ранних прогнозах. Причинный EWMA7 учитывает прошлые факты с запасом четыре часа. Для локального test используем архивный прогноз Session6, соответствующий исходному решению 0.326. Это учитывает нужный контроль и создаёт различие между восстановленным OOF и архивным test. Его нужно иметь в виду при интерпретации переноса весов.

У всех вариантов формы в каждом старте и горизонте одна сумма. Она равна сумме Session6 по маршрутам. Отдельные варианты потом проверяют ограниченную поправку к этой сумме.
""")

legacy = root / 'best/raw-vs-clean.ipynb'
anchor = []
for index in (5, 6, 7, 8, 9, 10, 11, 13):
    text = source(legacy, index)
    if index == 11:
        text = text.replace("h,'test',", "h,'oof',")
    if index == 13:
        text = text.replace("'all','test',", "'all','oof',")
    anchor.append(code(text))
component('session_runtime', dedent("""
    from catboost import CatBoostRegressor

    times, routes, values = TIMES, ROUTES, VALUES
    target = original_target = values[TARGET]
    names = [TARGET, *STATUSES]
    keys = KEYS
    meta = SESSION_META
    weight = float(meta['weight'])
    positions = np.arange(len(times))
    minutes = times.hour * 60 + times.minute
""") + '\n\n'.join(anchor) + '\n\n' + definitions(bases, {'session6_oof'}))

add('markdown', 'cat-guide', """
## CatBoost rich с восемью кластерами

Перед каждой OOF-неделей заново считаются статистики маршрутов, KMeans и выбранные динамические лаги. Модели CatBoost MAE обучаются по кластерам на всех получасовых стартах и всех целях до границы окна. Для слишком малых кластеров используется общая модель. Признаки, параметры и число деревьев взяты из rich-ноутбука. Прогнозы сохраняются без калибровки.
""")
cat = notebooks / 'clustering/catboost_rich_calibration.ipynb'
cat_config = source(cat, 1)
rich_source = next(ast.literal_eval(node.value) for node in ast.parse(cat_config).body
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and
                   node.targets[0].id == 'RICH_SOURCE')
cat_source = '\n\n'.join([
    dedent("""
        import joblib
        from catboost import CatBoostRegressor
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from threadpoolctl import threadpool_limits

        CLUSTERS = (1, 8)
        MIN_CLUSTER_ROUTES = 10
        RUN = ARTIFACTS / 'cat_rich'
        RUN.mkdir(exist_ok=True)
        ROLL = {w: pd.DataFrame(VALUES[TARGET]).rolling(w, min_periods=1).mean().to_numpy(dtype=np.float32)
                for w in (4, 12, 48, 336)}
    """),
    assignments(cat_config, {'PARAMETERS'}),
    'RICH_SOURCE = ' + repr(code(rich_source)),
    definitions(source(cat, 2), {'segment_routes', 'features'}),
    'original_features_rich = features\noriginal_segment_rich = segment_routes',
    definitions(source(cat, 6), {'segment_routes', 'features', 'model_frame'}),
    definitions(bases, {'fit_rich_cat', 'rich_matrix'}),
])
component('cat_runtime', cat_source)

add('markdown', 'rf-guide', """
## Random Forest с индивидуальными признаками по горизонтам

Воспроизводим сохранённый рецепт леса. Для h1–h8 остаются свои параметры и наборы признаков. Медиана по маршруту применяется только там, где она выбрана в исходной конфигурации. Лаги и заполнение пропусков оцениваются внутри обучающего периода каждой недели. Обучение использует пять утренних стартов, проверка использует 10:30.
""")
rf = notebooks / 'features/rf_best_full_history_submit_kaggle.ipynb'
rf_source = '\n\n'.join([
    dedent("""
        import re
        import cupy as cp
        from cuml.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer

        times, routes, values = TIMES, ROUTES, VALUES
        RUN = ARTIFACTS / 'rf'
        RUN.mkdir(exist_ok=True)
    """),
    assignments(source(rf, 2), {'BASE_PARAMS_BY_HORIZON', 'N_STREAMS', 'PREDICTION_BATCH', 'TRAIN_MINUTES',
                                'WINDOW_OFFSETS', 'LAG_MAX_LOOKBACK_HOURS', 'LAG_MIN_PAIRS', 'LAG_MIN_CORRELATION', 'RESIDUAL'}),
    assignments(source(rf, 3), {'best_params', 'selected'}),
    code(source(rf, 6)),
    code(source(rf, 7)),
    definitions(source(rf, 8), {'stable_ratio', 'available_statistics', 'add_experimental_features', 'horizon_dataset'}),
    assignments(source(rf, 9), {'TARGET_SCALE'}),
    definitions(source(rf, 9), {'model_columns'}),
    assignments(source(rf, 10), {'EXPERIMENTS'}),
    definitions(source(rf, 10), {'model_target', 'transformed_frame', 'fit_forest', 'predict_matrix', 'experiment_columns'}),
    definitions(bases, {'rf_data', 'fit_rf'}),
])
component('rf_runtime', rf_source)

add('markdown', 'clean-xgb-guide', """
## Обычный XGBoost на очищенной истории

Добавляем отдельный прогноз XGBoost MAE без кластеризации. Используем прежние признаки, параметры и число деревьев по горизонтам. Причинный MAD-4 фильтр очищает цель и шесть статусов по истории того же получасового слота. Лаги переоцениваются до каждой границы обучения. Обучаем на всех получасовых стартах, метрики считаем по исходным фактам из 10:30.

Для общего бленда проверяем варианты с этой моделью и без неё, а также пару Session6 + XGBoost clean. Прогноз входит в смесь без отдельного калибратора. Для сравнения при фиксированном объёме все варианты используют объём Session6. Новых входных датасетов не требуется.
""")
clean = notebooks / 'xgb_tuning/xgb_clean_vs_raw.ipynb'
clean_feature_code = next(ast.literal_eval(node.value) for node in ast.parse(source(clean, 6)).body
                          if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and
                          node.targets[0].id == 'FEATURE_CODE')
clean_xgb_source = '\n\n'.join([
    dedent("""
        import xgboost as xgb
        import cupy as cp

        RUN = ARTIFACTS / 'xgb_clean'
        RUN.mkdir(exist_ok=True)
        CLEAN_XGB_CONTEXT = None
    """),
    assignments(source(clean, 1), {'BEST_PARAMS', 'CLEANING', 'TARGET_SCALE', 'OBJECTIVE',
                                   'LAG_MAX_LOOKBACK_HOURS', 'LAG_MIN_PAIRS', 'LAG_MIN_CORRELATION'}),
    definitions(source(clean, 5), {'clean_raw_array'}),
    'FEATURE_CODE = ' + repr(code(clean_feature_code)),
    definitions(bases, {'CleanXgbBatches', 'clean_xgb_matrix', 'predict_clean_xgb',
                        'clean_xgb_features', 'fit_clean_xgb'}),
])
component('clean_xgb_runtime', clean_xgb_source)

add('markdown', 'oof-guide', """
## Сборка OOF

Для XGBoost rich берём четыре недели готовых raw-прогнозов. Читаем файлы прямо из Kaggle Dataset. Сверяем хеши данных, границы, ключи, факты и время каждой цели. Метапризнаки из истории используют только наблюдения, доступные на момент прогноза. `profile28` строится из предыдущих 28 дней с поправкой на текущий масштаб маршрута. Базовые метрики на OOF и test будут собраны в итоговой таблице.
""")
add('code', 'oof-build', """
session_oof = session_runtime['session6_oof']()
for name in ['means', 'level24', 'extra_signals', 'hub_signal', 'long_week', 'deviations', 'validation_lag_table']:
    session_runtime.pop(name, None)
oof_parts = []
for index, (start, stop) in enumerate(OOF_WINDOWS):
    stage = f'oof_week_{index}' if index < 3 else 'calibration_select'
    xgb = read_xgb(stage)
    cat = cat_runtime['fit_rich_cat'](start, stop, 'oof')
    rf = rf_runtime['fit_rf'](start, stop, 'oof')
    xgb_clean = clean_xgb_runtime['fit_clean_xgb'](start, stop, 'oof')
    oof_parts.append(assemble(start, stop, session_oof, rf, cat, xgb, xgb_clean))
    gc.collect()
oof = pd.concat(oof_parts, ignore_index=True)
save_parquet(ARTIFACTS / 'base_oof.parquet', oof)
display(pd.DataFrame([dict(model=name, **metric(oof.y, oof[name])) for name in BASES]))
""")
add('markdown', 'meta-guide', """
## Кандидаты метамодели

Проверяем Session6, равные веса, парные бленды, общий бленд с XGBoost clean и без него, а также усаженные веса по горизонтам. Ridge, CatBoost MAE и XGB MAE учатся предсказывать остаток общего бленда. Gating Network выдаёт для каждой строки веса моделей, которые складываются в единицу. Начальная точка сети берётся из общего бленда, а штраф ограничивает отход от него. Все поправки формы возвращаются к сумме Session6.

Входы включают прогнозы в масштабе маршрута, их отклонения от Session6, разброс, горизонт, день недели, текущий уровень, изменения за 1 и 2 часа, средний уровень маршрута за 24 часа и текущий объём сети. Сезонный `profile28` участвует как прогноз и признак. Отдельная поправка объёма сравнивает фиксированную сумму, усаженные общий и свежий коэффициенты и Ridge по агрегатам. Множители объёма ограничены ±5%.

На 4–10 октября метамодели проверяются после fit на первых двух OOF-неделях. На 11–17 октября используются первые три. Победитель определяется по среднему WAPE + |Relative Bias| двух недель. После фиксации схемы метамодели обучаются на четырёх OOF-неделях. На test публикуются результаты всех заранее заданных вариантов, включая победителя val.
""")
add('code', 'meta-runtime', meta)
add('code', 'freeze-selection', """
frozen = fit_selection(oof)
display(pd.DataFrame([frozen]))
""")
add('markdown', 'test-guide', """
## Локальный test и отчёт

RF, CatBoost rich и XGBoost на очищенной истории переобучаются на целях строго до 18 октября. XGBoost rich берётся из сохранённого запуска с той же границей. Для каждой новой метамодели печатаются train OOF и локальный test. У train OOF базовые прогнозы вне обучающих окон, а метамодель оценивается на своём обучении. В отчёт также входят срезы по дням, неделям и горизонтам, выбор по val, построчные прогнозы и параметры метамоделей.

Независимая от полного переобучения оценка test получена до переобучения.
""")
add('code', 'evaluate-test', """
if json.loads((ARTIFACTS / 'frozen_selection.json').read_text()) != frozen:
    raise ValueError('Сначала зафиксируйте выбор по val')
xgb_test = read_xgb('train_val')
session_test = read_session6_test()
cat_test = cat_runtime['fit_rich_cat'](LOCAL_TEST_START, LOCAL_TEST_END, 'test')
rf_test = rf_runtime['fit_rf'](LOCAL_TEST_START, LOCAL_TEST_END, 'test')
clean_test = clean_xgb_runtime['fit_clean_xgb'](LOCAL_TEST_START, LOCAL_TEST_END, 'test')
test = assemble(LOCAL_TEST_START, LOCAL_TEST_END, session_test, rf_test, cat_test, xgb_test, clean_test)
evaluate(oof, test, frozen)
archive_report()
""")

digest = hashlib.sha256(''.join(''.join(item['source']) for item in cells).encode()).hexdigest()
setup = next(item for item in cells if item['id'] == 'read-inputs')
setup['source'].insert(0, f'NOTEBOOK_HASH = {digest!r}\n')
metadata = dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                language_info=dict(name='python'))
output = root / 'stacking/blending/best_models_blend_kaggle.ipynb'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(dict(nbformat=4, nbformat_minor=5, metadata=metadata, cells=cells),
                             ensure_ascii=False, indent=1) + '\n')
print(output)
