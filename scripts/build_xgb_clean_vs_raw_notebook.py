import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / 'notebooks/xgb_tuning/xgb_cleaning.ipynb'
source = json.loads(SOURCE_PATH.read_text())
params_path = ROOT / 'notebooks/xgb_tuning/best_params/best_params.json'
parameters = json.loads(params_path.read_text())
fixed = {h: record['refit_params'] for h, record in parameters['horizons'].items()}


def source_code(index):
    return ''.join(source['cells'][index]['source'])


def definitions(index, names):
    tree = ast.parse(source_code(index))
    return '\n\n'.join(ast.unparse(node) for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names)


feature_code = next(ast.literal_eval(node.value) for node in ast.parse(source_code(6)).body if isinstance(node, ast.Assign) and any(isinstance(item, ast.Name) and item.id == 'FEATURE_CODE' for item in node.targets))
feature_tree = ast.parse(feature_code)
lag_code = '\n\n'.join(ast.unparse(node) for node in feature_tree.body if isinstance(node, ast.FunctionDef) and node.name in {'calculate_dynamic_shoulder', 'add_correlation_features'})
correlation_columns = next(ast.literal_eval(node.value) for node in feature_tree.body if isinstance(node, ast.Assign) and any(isinstance(item, ast.Name) and item.id == 'CORRELATION_COLUMNS' for item in node.targets))
cleaner_code = definitions(5, {'clean_raw_array'})
metric_code = definitions(9, {'metric_row', 'require_cuda'})
calibration_code = definitions(11, {'calibration_keys', 'fit_calibrator', 'apply_calibrator'})
cells = []


def md(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)})


def code(text):
    tree = ast.parse(text)
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': (ast.unparse(tree) + '\n').splitlines(keepends=True)})


md('''# XGBoost: исходная история против очищенной, train → калибровка → train + val → test

Контрольный опыт с двумя вариантами на всех получасовых стартах: `raw` использует исходные цель и статусы; `clean` очищает цель и все шесть статусов до построения признаков. В каждом варианте восемь direct-моделей h1–h8 с одинаковыми признаками, гиперпараметрами, seed и временными границами. Параметры, включая число деревьев, встроены из `notebooks/xgb_tuning/best_params/best_params.json`, секция `refit_params`. Optuna и early stopping отсутствуют.

1. Обучение на train; прогнозы val в 10:30 для калибровки. Test на этом этапе не прогнозируется.
2. Четыре калибровки: none, global, grouped, nonlinear. Ранняя половина val служит для fit, поздняя — для выбора типа по WAPE + |Relative Bias|. После выбора коэффициенты переоцениваются по всем OOS-прогнозам val и исходным фактам val.
3. Обе модели заново обучаются на train + val, на всех получасовых стартах; калибровка зафиксирована. Корреляционный словарь переоценивается только до test.
4. Сравнение исходного и очищенного вариантов на одних и тех же исходных ответах test, до и после выбранной калибровки.

Последние семь полных прогнозных дат — test, предыдущие дни делятся 80/20 на train/val. Для исходной истории: train до 7 октября, val 7–24 октября, test 25–31 октября 2025 года. Октябрьский test уже изучался в проекте; это не новый слепой период. Независимая от выбора на val оценка test получается после обучения на train + val, без включения test в fit. По последнему указанию пользователя полного переобучения на всей размеченной истории и сабмитов нет. Обучение запускает пользователь в Kaggle на GPU.
''')
config = '''from pathlib import Path
import ast
import gc
import hashlib
import json
import math
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb
try:
    import cupy as cp
except ImportError as error:
    raise RuntimeError('Для GPU-матриц нужен CuPy в окружении Kaggle') from error
from sklearn.isotonic import IsotonicRegression
from IPython.display import display
DATA_DIR = None
CACHE_INPUTS = {'raw': None, 'clean': None}
OUTPUT_DIR = Path('/kaggle/working/xgb_clean_vs_raw')
DEVICE = 'cuda'
N_JOBS = 4
SEED = 42
TEST_DAYS = 7
TRAIN_FRACTION = 0.8
TARGET_SCALE = 100000.0
OBJECTIVE = 'reg:absoluteerror'
TARGET = 'target_1h'
STATUSES = [f'status_{i}' for i in range(1, 7)]
STEP = pd.Timedelta(minutes=30)
VARIANTS = ['raw', 'clean']
CALIBRATIONS = ['none', 'global', 'grouped', 'nonlinear']
CLEANING = {'window_days': 28, 'min_days': 14, 'z_threshold': 4.0, 'log_scale_floor': 0.25, 'mad_normalization': 1.4826}
GROUP_SHRINKAGE_ROWS = 2000.0
ISOTONIC_BINS = 128
LAG_MAX_LOOKBACK_HOURS = 24
LAG_MIN_PAIRS = 336
LAG_MIN_CORRELATION = 0.05
PIPELINE_VERSION = 'xgb_clean_vs_raw_mad4_all_halfhours_v1'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_DIR = OUTPUT_DIR / 'runs' / pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
RUN_DIR.mkdir(parents=True, exist_ok=False)
'''
code(config + '\nBEST_PARAMS = ' + repr(fixed) + '\nPARAMETER_SOURCE_SHA256 = ' + repr(hashlib.sha256(params_path.read_bytes()).hexdigest()) + '\nCLEANING_IMPLEMENTATION_SHA256 = ' + repr(hashlib.sha256(cleaner_code.encode()).hexdigest()))
code(source_code(2) + '\ncp.cuda.Device(0).use()\ncp.cuda.get_current_stream().synchronize()\nprint("Модель и матрицы: cuda:0; входы float32 CuPy")')
code(definitions(3, {'file_hash', 'save_json'}) + '''

if DATA_DIR is None:
    candidates = [p.parent for p in Path('/kaggle/input').rglob('train_solo_track.parquet')]
    if not candidates and Path('data/train_solo_track.parquet').is_file():
        candidates = [Path('data')]
    if len(candidates) != 1:
        raise ValueError(f'Укажите DATA_DIR: найдено {len(candidates)} train parquet')
    DATA_DIR = candidates[0]
TRAIN_PATH = Path(DATA_DIR) / 'train_solo_track.parquet'
history = pd.read_parquet(TRAIN_PATH, columns=['route_id', 'timestamp', TARGET, *STATUSES]).sort_values(['timestamp', 'route_id'])
times = pd.DatetimeIndex(history.timestamp.unique())
routes = np.sort(history.route_id.unique())
if history.isna().any().any() or history.duplicated(['timestamp', 'route_id']).any():
    raise ValueError('Пропуски или дубликаты')
if not times.equals(pd.date_range(times[0], times[-1], freq=STEP)) or len(history) != len(times) * len(routes):
    raise ValueError('Нарушена полная получасовая панель')
raw_values = {name: history[name].to_numpy(dtype=np.float32).reshape(len(times), len(routes)).copy() for name in [TARGET, *STATUSES]}
if any(not np.isfinite(array).all() or (array < 0).any() for array in raw_values.values()):
    raise ValueError('Нужны конечные неотрицательные исходные значения')
del history
end = times[-1]
dates = times[(times.hour == 10) & (times.minute == 30) & (times + 8 * STEP <= end)]
if len(dates) < TEST_DAYS + 20:
    raise ValueError('Недостаточно полных дат')
before_test = dates[:-TEST_DAYS]
split_index = int(math.floor(len(before_test) * TRAIN_FRACTION))
val_start = before_test[split_index].normalize()
test_start = dates[-TEST_DAYS].normalize()
val_dates = dates[(dates >= val_start) & (dates < test_start)]
test_dates = dates[dates >= test_start]
calibration_select_start = val_dates[len(val_dates) // 2].normalize()
route_levels = np.median(raw_values[TARGET][times < val_start], axis=0)
route_groups = pd.DataFrame({'route_id': routes, 'volume_group': pd.qcut(pd.Series(route_levels).rank(method='first'), min(5, len(routes)), labels=False).to_numpy()})
route_groups.to_csv(RUN_DIR / 'route_volume_groups_train.csv', index=False)
pd.DataFrame({'route_id': routes, 'gpu_category_code': np.arange(len(routes))}).to_csv(RUN_DIR / 'route_encoding.csv', index=False)
route_dtype = pd.CategoricalDtype(categories=routes)
display(pd.DataFrame([{'split': 'train', 'from': str(times[0]), 'before': str(val_start)}, {'split': 'calibration_fit', 'from': str(val_start), 'before': str(calibration_select_start)}, {'split': 'calibration_select', 'from': str(calibration_select_start), 'before': str(test_start)}, {'split': 'train_plus_val_refit', 'from': str(times[0]), 'before': str(test_start)}, {'split': 'test', 'from': str(test_dates[0]), 'through': str(test_dates[-1])}]))
''')
md('''## Причинная очистка только в варианте clean

Тот же фильтр, что в завершённом опыте MAD-4: log1p, предыдущие 28 дней того же получасового слота, минимум 14 наблюдений, медиана и точный MAD одного окна. Масштаб `max(1.4826 × MAD, 0.25)`, порог 4. Выброс заменяется `expm1` медианы. Проверяются верхние/нижние выбросы, включая нули; строки не удаляются. Raw-ветка не проходит фильтр.

Фильтр использует только предыдущие исходные наблюдения. На val/test очищается только уже известная к cutoff входная история; исходные ответы test остаются неизменными. Факты для метрик и калибровки всегда исходные. В начале окна без достаточной истории очистки нет.
''')
code(cleaner_code)
code('FEATURE_CODE = ' + repr(feature_code) + '\nCORRELATION_COLUMNS = ' + repr(correlation_columns) + '\n\n' + lag_code)
md('''## Два отдельных кэша признаков

Raw и clean используют разные исходные массивы, лаги и словари корреляционных лагов, но один набор колонок и одинаковые строки. В кэш входят все получасы с размеченной целью до test и только старты 10:30 внутри test. `y` — исходная цель, `y_fit` — цель обучения выбранной ветки. Обе исключены из входов.

Кэши различаются отпечатком исходного parquet, кода признаков, очистки и границ. Совместимые кэши ищутся в Kaggle Input; явно задать их можно через CACHE_INPUTS. Оценочные корреляционные словари используют только train. Для train + val они будут переоценены до test, остальные признаки причинны и не требуют пересборки.
''')
code('''def cache_complete(folder, expected_fingerprint):
    folder = Path(folder)
    try:
        metadata = json.loads((folder / 'manifest.json').read_text())
        if metadata['fingerprint'] != expected_fingerprint:
            return False
        for h in range(1, 9):
            path = folder / f'history_h{h}.parquet'
            record = json.loads(path.with_suffix('.json').read_text())
            if record['fingerprint'] != expected_fingerprint or record['sha256'] != file_hash(path):
                return False
        return True
    except (OSError, ValueError, KeyError):
        return False

cache_dirs = {}
phase_signals = {}
cache_fingerprints = {}
cleaning_rows = []
input_manifests = [p.parent for p in Path('/kaggle/input').rglob('manifest.json')]
source_hash = file_hash(TRAIN_PATH)
for variant in VARIANTS:
    if variant == 'raw':
        values = raw_values
    else:
        values = {}
        for name in [TARGET, *STATUSES]:
            cleaned, flagged = clean_raw_array(raw_values[name], times, CLEANING)
            values[name] = cleaned
            for split, mask in [('train', times < val_start), ('val', (times >= val_start) & (times < test_start)), ('test_observed_history', times >= test_start)]:
                marks = flagged[mask]
                cleaning_rows.append({'column': name, 'split': split, 'rows': int(marks.size), 'replaced': int(marks.sum()), 'replaced_pct': float(100 * marks.mean()), 'zeros_replaced': int((marks & (raw_values[name][mask] == 0)).sum())})
            del cleaned, flagged
    context = {'np': np, 'pd': pd, 'STEP': STEP, 'TARGET': TARGET, 'STATUSES': STATUSES, 'times': times, 'routes': routes, 'values': values}
    exec(FEATURE_CODE, context)
    lag_table = context['calculate_dynamic_shoulder'](context['target'], context['hub_signal'], times, routes, fit_before=val_start, max_lookback_hours=LAG_MAX_LOOKBACK_HOURS, min_pairs=LAG_MIN_PAIRS, min_correlation=LAG_MIN_CORRELATION)
    context['validation_lag_table'] = lag_table
    lag_table.to_csv(RUN_DIR / f'{variant}_status6_lags_train.csv', index=False)
    phase_signals[variant] = {'target': context['target'], 'signal': context['hub_signal']}
    spec = {'version': PIPELINE_VERSION, 'variant': variant, 'source_sha256': source_hash, 'feature_code_sha256': hashlib.sha256(FEATURE_CODE.encode()).hexdigest(), 'cleaning': CLEANING if variant == 'clean' else None, 'cleaning_code_sha256': CLEANING_IMPLEMENTATION_SHA256 if variant == 'clean' else None, 'val_start': str(val_start), 'test_start': str(test_start), 'last_test_origin': str(test_dates[-1]), 'lag_settings': [LAG_MAX_LOOKBACK_HOURS, LAG_MIN_PAIRS, LAG_MIN_CORRELATION]}
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    cache_fingerprints[variant] = fingerprint
    local_cache = OUTPUT_DIR / 'feature_cache' / fingerprint
    explicit = CACHE_INPUTS.get(variant)
    if explicit is not None:
        if not cache_complete(explicit, fingerprint):
            raise ValueError(f'Несовместимый кэш {variant}')
        folder = Path(explicit)
    else:
        folder = next((p for p in [local_cache, *input_manifests] if cache_complete(p, fingerprint)), local_cache)
    if folder == local_cache:
        folder.mkdir(parents=True, exist_ok=True)
        save_json(folder / 'manifest.json', {'fingerprint': fingerprint, 'spec': spec, 'routes': routes.tolist()})
    cache_dirs[variant] = folder
    print(variant, 'кэш', folder, flush=True)
    for h in range(1, 9):
        path = folder / f'history_h{h}.parquet'
        try:
            record = json.loads(path.with_suffix('.json').read_text())
            ready = record['fingerprint'] == fingerprint and record['sha256'] == file_hash(path)
        except (OSError, ValueError, KeyError):
            ready = False
        if ready:
            print(variant, f'h{h}: готовый кэш', flush=True)
            continue
        origins = np.arange(len(times) - h)
        eligible = (times[origins + h] < test_start) | times[origins].isin(test_dates)
        origins = origins[eligible]
        temporary = path.with_suffix('.partial')
        writer = None
        try:
            for start in range(0, len(origins), 24):
                block = origins[start:start + 24]
                frame = context['horizon_dataset'](h, block)
                frame['y'] = raw_values[TARGET][block + h].ravel()
                frame['y_fit'] = context['target'][block + h].ravel()
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression='zstd')
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
        temporary.replace(path)
        save_json(path.with_suffix('.json'), {'fingerprint': fingerprint, 'rows': len(origins) * len(routes), 'sha256': file_hash(path)})
        print(variant, f'h{h}: {len(origins) * len(routes):,} строк', flush=True)
    del context, values
    gc.collect()
cleaning_report = pd.DataFrame(cleaning_rows)
cleaning_report.to_csv(RUN_DIR / 'cleaning_report.csv', index=False)
display(cleaning_report)
for name in ['raw_values', 'frame', 'table']:
    globals().pop(name, None)
gc.collect()
save_json(RUN_DIR / 'fixed_settings.json', {'params': BEST_PARAMS, 'parameter_source_sha256': PARAMETER_SOURCE_SHA256, 'cleaning': CLEANING, 'cache_fingerprints': cache_fingerprints, 'val_start': str(val_start), 'test_start': str(test_start), 'calibration_select_start': str(calibration_select_start), 'train_plus_val_refit': True, 'full_history_refit': False, 'test_used_for_fit_or_selection': False, 'input_device': 'cuda:0', 'input_dtype': 'float32', 'route_encoding': 'fixed_routes_index_categorical'})
''')
md('''## Этап 1: train → прогнозы val

Все получасовые обучающие метки строго раньше val. Здесь test не прогнозируется; его строка в промежуточной таблице отмечена как ожидающая train + val. Число деревьев фиксировано из best_params. После каждого fit показываются train и val по исходной цели, включая одинаковый train в 10:30. Модели и построчные OOS-прогнозы val сохраняются.
''')
md('''## Матрицы и модель на GPU

Обучающие матрицы float32 и цели передаются в XGBoost как CuPy-массивы на cuda:0. Обучающая матрица остаётся на GPU до расчёта train-прогноза, затем освобождается. Val/test также прогнозируются из CuPy через inplace_predict; на CPU возвращаются только предсказания для метрик и калибровки. route_id кодируется по общему фиксированному справочнику routes и явно помечается feature_types=c; остальные колонки — q. Порядок колонок сохраняется в feature_columns.json, названия и типы записываются в Booster.

Очистка, подготовка признаков, parquet, метрики и четыре калибровки остаются на CPU. На GPU последовательно загружается один горизонт одной ветки; все горизонты одновременно в VRAM не хранятся. Перед fit выводится объём входных GPU-массивов и свободной памяти. Используется одна GPU cuda:0, даже если доступно две T4. Требуется установленный CuPy.
''')
code(metric_code + '''

def model_columns(frame):
    return frame.columns.drop(['cutoff', 'timestamp', 'y', 'y_fit', 'overlap']).tolist()

def make_model(h, columns):
    types = ['c' if name == 'route_id' else 'q' for name in columns]
    return xgb.XGBRegressor(**BEST_PARAMS[str(h)], objective=OBJECTIVE, eval_metric='mae', tree_method='hist', device=DEVICE, enable_categorical=True, feature_types=types, max_cat_to_onehot=1, random_state=SEED, n_jobs=N_JOBS)

def release_gpu_pool():
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

def gpu_matrix(frame, columns):
    host = np.empty((len(frame), len(columns)), dtype=np.float32)
    route_position = columns.index('route_id')
    codes = pd.Index(routes).get_indexer(frame.route_id.astype('int64'))
    if (codes < 0).any():
        raise ValueError('Маршрут отсутствует в фиксированном справочнике')
    host[:, route_position] = codes
    numeric = [name for name in columns if name != 'route_id']
    positions = [index for index, name in enumerate(columns) if name != 'route_id']
    host[:, positions] = frame[numeric].to_numpy(dtype=np.float32)
    matrix = cp.asarray(host, dtype=cp.float32, order='C')
    if matrix.device.id != 0:
        raise RuntimeError('Матрица должна находиться на cuda:0')
    return matrix

def fit_gpu_model(h, data, mask, columns):
    release_gpu_pool()
    matrix = gpu_matrix(data.loc[mask, columns], columns)
    labels = cp.asarray(data.loc[mask, 'y_fit'].to_numpy(dtype=np.float32) / TARGET_SCALE, dtype=cp.float32)
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    print('GPU inputs GiB', round((matrix.nbytes + labels.nbytes) / 2**30, 3), 'free GiB before fit', round(free_bytes / 2**30, 3), flush=True)
    model = make_model(h, columns)
    model.fit(matrix, labels, verbose=False)
    require_cuda(model)
    expected = ['c' if name == 'route_id' else 'q' for name in columns]
    if model.get_booster().feature_types != expected:
        raise RuntimeError('Не сохранена категориальность route_id')
    model.get_booster().feature_names = columns
    return model, matrix

def predict_gpu_matrix(model, matrix):
    prediction = model.get_booster().inplace_predict(matrix)
    result = cp.maximum(prediction.astype(cp.float64) * TARGET_SCALE, 0)
    return cp.asnumpy(result)

def predict_gpu_frame(model, frame, columns):
    matrix = gpu_matrix(frame, columns)
    result = predict_gpu_matrix(model, matrix)
    del matrix
    release_gpu_pool()
    return result

feature_columns = {}
calibration_predictions = {}
stage1_rows = []
for variant in VARIANTS:
    prediction_parts = []
    for h in range(1, 9):
        data = pd.read_parquet(cache_dirs[variant] / f'history_h{h}.parquet')
        data['route_id'] = data.route_id.astype(route_dtype)
        columns = model_columns(data)
        if h in feature_columns and feature_columns[h] != columns:
            raise ValueError('Разные входные колонки raw и clean')
        feature_columns[h] = columns
        train = data.timestamp.lt(val_start)
        train1030 = train & data.cutoff.isin(dates[dates < val_start])
        val = data.cutoff.isin(val_dates)
        print('TRAIN', variant, f'h{h}', 'rows', int(train.sum()), 'trees', BEST_PARAMS[str(h)]['n_estimators'], flush=True)
        clock = time.monotonic()
        model, train_gpu = fit_gpu_model(h, data, train, columns)
        model.save_model(RUN_DIR / f'{variant}_h{h}_train_only.ubj')
        train_pred = predict_gpu_matrix(model, train_gpu)
        del train_gpu
        release_gpu_pool()
        rows = [metric_row(data.loc[train], train_pred, 'train_all_halfhours')]
        rows.append(metric_row(data.loc[train1030], train_pred[data.loc[train].cutoff.isin(dates[dates < val_start]).to_numpy()], 'train_1030'))
        part = data.loc[val, ['cutoff', 'timestamp', 'route_id', 'y']].copy()
        part['prediction'] = predict_gpu_frame(model, data.loc[val, columns], columns)
        part['horizon'] = h
        rows.append(metric_row(part, part.prediction, 'val_oos'))
        rows.append({'split': 'test', 'status': 'не оценивается до train_plus_val'})
        table = pd.DataFrame(rows)
        table.insert(0, 'horizon', h)
        table.insert(0, 'variant', variant)
        stage1_rows.append(table)
        display(table)
        prediction_parts.append(part)
        print('DONE TRAIN', variant, f'h{h}', 'seconds', round(time.monotonic() - clock, 1), flush=True)
        del data, model, train_pred
        release_gpu_pool()
    calibration_predictions[variant] = pd.concat(prediction_parts, ignore_index=True)
    calibration_predictions[variant].to_parquet(RUN_DIR / f'{variant}_val_oos.parquet', index=False)
pd.concat(stage1_rows, ignore_index=True).to_csv(RUN_DIR / 'train_only_metrics.csv', index=False)
save_json(RUN_DIR / 'feature_columns.json', feature_columns)
''')
md('''## Выбор и фиксация калибровки по val

Прежние четыре метода: none; общий множитель; множитель по горизонту × train-квинтилю объёма со стягиванием к общему (2 000 псевдострок); изотоническая калибровка на 128 квантильных корзинах с пропорциональной экстраполяцией. Параметры методов не перебираются.

Каждый вариант выбирает тип только по поздней половине val с коэффициентами, обученными на ранней половине. Затем коэффициенты выбранного типа переоцениваются по всему val на прогнозах train-only моделей. Test не используется. При переходе к train + val коэффициенты замораживаются; смена обучающего периода может изменить смещение модели, что и проверяется на test.
''')
code(calibration_code + '''

selection_rows = []
selected_calibrations = {}
calibrators = {}
for variant in VARIANTS:
    panel = calibration_predictions[variant]
    fit_part = panel.loc[panel.cutoff < calibration_select_start]
    select_part = panel.loc[panel.cutoff >= calibration_select_start]
    rows = []
    for method in CALIBRATIONS:
        fitted = fit_calibrator(fit_part, method)
        pred = apply_calibrator(select_part, fitted)
        rows.append({'variant': variant, 'calibration': method, **metric_row(select_part, pred, 'calibration_select')})
    selected_method = min(rows, key=lambda row: row['score'])['calibration']
    selected_calibrations[variant] = selected_method
    calibrators[variant] = fit_calibrator(panel, selected_method)
    selection_rows.extend(rows)
selection = pd.DataFrame(selection_rows)
selection['selected_on_val'] = selection.apply(lambda row: selected_calibrations[row.variant] == row.calibration, axis=1)
selection.to_csv(RUN_DIR / 'calibration_selection.csv', index=False)
display(selection)
save_json(RUN_DIR / 'frozen_calibrators.json', {'selected': selected_calibrations, 'calibrators': calibrators, 'fit_on': 'train_only_oos_val_predictions_and_raw_val_targets', 'selection_start': str(calibration_select_start), 'test_used': False})
print('Зафиксировано до train + val:', selected_calibrations, flush=True)
''')
md('''## Этап 2: заново обучить на train + val и оценить test

Во всех fit метка timestamp строго раньше test_start. Размеченный test остаётся вне обучения. Статистический лаг переоценивается до test_start отдельно для raw и clean; train-квинтили калибровки остаются зафиксированными. На test используются уже доступные к cutoff наблюдения, включая историю прошлых test-дней; будущие относительно cutoff статусы и цели не входят в признаки.

После каждого fit показывается train + val и test по исходному таргету до и после выбранной на val калибровки. Проверка выполняется на одном наборе пар cutoff/route/horizon. Сравнение с общей фиксированной калибровкой none отделяет влияние очистки от изменения выбранного калибратора. Полного refit с включением test нет.
''')
code('''test_predictions = {}
trainval_metric_rows = []
for variant in VARIANTS:
    signal = phase_signals[variant]
    hub_signal = signal['signal']
    refit_lag_table = calculate_dynamic_shoulder(signal['target'], hub_signal, times, routes, fit_before=test_start, max_lookback_hours=LAG_MAX_LOOKBACK_HOURS, min_pairs=LAG_MIN_PAIRS, min_correlation=LAG_MIN_CORRELATION)
    refit_lag_table.to_csv(RUN_DIR / f'{variant}_status6_lags_trainval.csv', index=False)
    prediction_parts = []
    for h in range(1, 9):
        data = add_correlation_features(pd.read_parquet(cache_dirs[variant] / f'history_h{h}.parquet'), h, refit_lag_table)
        data['route_id'] = data.route_id.astype(route_dtype)
        columns = feature_columns[h]
        trainval = data.timestamp.lt(test_start)
        trainval1030 = trainval & data.cutoff.isin(dates[dates < test_start])
        evaluate = data.cutoff.isin(test_dates)
        if not data.loc[trainval, 'timestamp'].lt(test_start).all() or not data.loc[evaluate, 'cutoff'].ge(test_start).all():
            raise ValueError('Нарушены границы test')
        print('TRAIN+VAL', variant, f'h{h}', 'rows', int(trainval.sum()), 'max_label', str(data.loc[trainval, 'timestamp'].max()), 'calibration', selected_calibrations[variant], flush=True)
        clock = time.monotonic()
        model, train_gpu = fit_gpu_model(h, data, trainval, columns)
        model.save_model(RUN_DIR / f'{variant}_h{h}_trainval.ubj')
        train_pred = predict_gpu_matrix(model, train_gpu)
        del train_gpu
        release_gpu_pool()
        rows = [{'calibration': 'none', **metric_row(data.loc[trainval], train_pred, 'trainval_all_halfhours')}]
        train_panel = data.loc[trainval1030, ['cutoff', 'timestamp', 'route_id', 'y']].copy()
        train_panel['prediction'] = train_pred[data.loc[trainval].cutoff.isin(dates[dates < test_start]).to_numpy()]
        train_panel['horizon'] = h
        test_panel = data.loc[evaluate, ['cutoff', 'timestamp', 'route_id', 'y']].copy()
        test_panel['prediction'] = predict_gpu_frame(model, data.loc[evaluate, columns], columns)
        test_panel['horizon'] = h
        test_panel['prediction_calibrated'] = apply_calibrator(test_panel, calibrators[variant])
        for split, panel in [('trainval_1030', train_panel), ('test', test_panel)]:
            rows.append({'calibration': 'none', **metric_row(panel, panel.prediction, split)})
            rows.append({'calibration': 'selected_on_val:' + selected_calibrations[variant], **metric_row(panel, apply_calibrator(panel, calibrators[variant]), split)})
        table = pd.DataFrame(rows)
        table.insert(0, 'horizon', h)
        table.insert(0, 'variant', variant)
        trainval_metric_rows.append(table)
        display(table)
        train_panel['prediction_calibrated'] = apply_calibrator(train_panel, calibrators[variant])
        train_panel.to_parquet(RUN_DIR / f'{variant}_h{h}_trainval_1030_predictions.parquet', index=False)
        test_panel.to_parquet(RUN_DIR / f'{variant}_h{h}_test_predictions.parquet', index=False)
        prediction_parts.append(test_panel)
        print('DONE TRAIN+VAL', variant, f'h{h}', 'seconds', round(time.monotonic() - clock, 1), flush=True)
        del data, model, train_pred, train_panel
        release_gpu_pool()
    test_predictions[variant] = pd.concat(prediction_parts, ignore_index=True)
    test_predictions[variant].to_parquet(RUN_DIR / f'{variant}_test_predictions.parquet', index=False)
trainval_metrics = pd.concat(trainval_metric_rows, ignore_index=True)
trainval_metrics.to_csv(RUN_DIR / 'trainval_metrics_by_horizon.csv', index=False)
''')
code('''comparison_rows = []
daily_rows = []
for variant in VARIANTS:
    train_panel = pd.concat([pd.read_parquet(RUN_DIR / f'{variant}_h{h}_trainval_1030_predictions.parquet') for h in range(1, 9)], ignore_index=True)
    for split, panel in [('trainval_1030', train_panel), ('test', test_predictions[variant])]:
        for kind, column in [('none', 'prediction'), ('selected_on_val:' + selected_calibrations[variant], 'prediction_calibrated')]:
            comparison_rows.append({'variant': variant, 'calibration': kind, **metric_row(panel, panel[column], split)})
    panel = test_predictions[variant]
    for cutoff, block in panel.groupby('cutoff'):
        daily_rows.append({'variant': variant, 'cutoff': str(cutoff), **metric_row(block, block.prediction_calibrated, 'test_selected_calibration')})
comparison = pd.DataFrame(comparison_rows)
comparison.to_csv(RUN_DIR / 'comparison_trainval_test.csv', index=False)
display(comparison)
daily = pd.DataFrame(daily_rows)
daily.to_csv(RUN_DIR / 'test_metrics_by_day.csv', index=False)
display(daily)
keys = ['cutoff', 'timestamp', 'route_id', 'horizon']
paired = test_predictions['raw'].merge(test_predictions['clean'], on=keys, how='outer', suffixes=('_raw', '_clean'), indicator=True, validate='one_to_one')
if not paired['_merge'].eq('both').all() or not np.array_equal(paired.y_raw.to_numpy(), paired.y_clean.to_numpy()):
    raise ValueError('Raw и clean оценены на разных исходных ответах')
paired['absolute_error_raw'] = np.abs(paired.prediction_calibrated_raw - paired.y_raw)
paired['absolute_error_clean'] = np.abs(paired.prediction_calibrated_clean - paired.y_raw)
paired.to_parquet(RUN_DIR / 'paired_test_predictions.parquet', index=False)
selected_test = comparison.loc[comparison.split.eq('test') & comparison.calibration.str.startswith('selected_on_val:')].set_index('variant')
delta = float(selected_test.loc['clean', 'score'] - selected_test.loc['raw', 'score'])
print('Test clean − raw, выбранные калибровки:', delta, '(меньше нуля — лучше clean)')
none_test = comparison.loc[comparison.split.eq('test') & comparison.calibration.eq('none')].set_index('variant')
print('Test clean − raw, без калибровки:', float(none_test.loc['clean', 'score'] - none_test.loc['raw', 'score']))
save_json(RUN_DIR / 'final_protocol.json', {'fit': 'train_plus_val_only', 'fit_label_before': str(test_start), 'calibration_frozen_before_trainval': True, 'selected_calibrations': selected_calibrations, 'test_used_for_fit_or_selection': False, 'full_history_refit': False, 'submissions_created': False, 'clean_minus_raw_test_score': delta})
print('Независимая от выбора на val оценка test получена после обучения на train + val, без включения test в fit. Полное переобучение на всей истории не выполнялось.')
print('Эти даты уже изучались в проекте. Отчёты и построчные прогнозы:', RUN_DIR)
''')
for index, cell in enumerate(cells):
    cell['id'] = f'clean-vs-raw-{index:02d}'
notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.11'}, 'accelerator': 'GPU'}, 'nbformat': 4, 'nbformat_minor': 5}
output = ROOT / 'notebooks/xgb_tuning/xgb_clean_vs_raw.ipynb'
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
print(output)
