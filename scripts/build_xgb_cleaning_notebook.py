import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / 'notebooks/xgb_tuning/xgb_tuning.ipynb'
source = json.loads(SOURCE_PATH.read_text())
parameter_path = ROOT / 'notebooks/xgb_tuning/best_params/best_params.json'
parameters = json.loads(parameter_path.read_text())
fixed = {h: record['refit_params'] for h, record in parameters['horizons'].items()}
feature_tree = ast.parse(''.join(source['cells'][5]['source']))
feature_code = ast.literal_eval(feature_tree.body[0].value)
metric_tree = ast.parse(''.join(source['cells'][8]['source']))
metric_code = ast.unparse(next(node for node in metric_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'metric_row'))
cells = []


class StripDocs(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body.pop(0)
        return node


def code(text):
    tree = StripDocs().visit(ast.parse(text))
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': (ast.unparse(tree) + '\n').splitlines(keepends=True)})


def md(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)})


md('''# XGBoost: очистка исходных рядов → признаки → два варианта → калибровка → сабмиты

Версия прикреплённого `xgb-tuning (1).ipynb`. Признаки сохранены из него. До их построения очищаются `target_1h` и все `status_1`–`status_6`. Обучение выполняет пользователь в Kaggle на GPU.

Два набора по восемь direct-моделей: все 48 получасовых стартов и девять стартов 08:30–12:30. Гиперпараметры, включая число деревьев, встроены из `notebooks/xgb_tuning/best_params/best_params.json`, секция `refit_params`. Подбора XGBoost, Optuna и early stopping нет.

Последние семь полных дней — test. Предшествующие даты разделены 80/20 на train/val, как в источнике. Val/test воспроизводят старт 10:30. Внутри val первая половина служит для fit четырёх калибраторов, вторая — для сравнения. После выбора калибраторы переоцениваются на всём val. Test оценивается до финального переобучения и не используется для выбора калибровки. Эти даты уже изучались в проекте и не являются новым слепым периодом.

Метрики всегда используют исходный таргет. Обучающая цель очищена; исходная цель сохраняется отдельно. Все восемь моделей каждого варианта затем переобучаются на всей доступной размеченной истории. Создаются четыре сабмита на вариант и копия выбранной на val калибровки.
''')
code('''from pathlib import Path
import gc
import hashlib
import json
import math
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from IPython.display import display
DATA_DIR = None
CACHE_INPUT = None
OUTPUT_DIR = Path('/kaggle/working/xgb_cleaning')
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
START_MINUTES = list(range(510, 751, 30))
VARIANTS = ['all_halfhours', 'nine_starts']
CALIBRATIONS = ['none', 'global', 'grouped', 'nonlinear']
CLEANING = {'window_days': 28, 'min_days': 14, 'z_threshold': 4.0, 'log_scale_floor': 0.25, 'mad_normalization': 1.4826}
GROUP_SHRINKAGE_ROWS = 2000.0
ISOTONIC_BINS = 128
LAG_MAX_LOOKBACK_HOURS = 24
LAG_MIN_PAIRS = 336
LAG_MIN_CORRELATION = 0.05
PIPELINE_VERSION = 'xgb_raw_cleaning_log_slot_mad_v2'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_DIR = OUTPUT_DIR / 'runs' / pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
RUN_DIR.mkdir(parents=True, exist_ok=False)
''' + '\nBEST_PARAMS = ' + repr(fixed) + '\nPARAMETER_SOURCE_SHA256 = ' + repr(hashlib.sha256(parameter_path.read_bytes()).hexdigest()) + '\nSOURCE_NOTEBOOK_SHA256 = ' + repr(hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest()))
code(''.join(source['cells'][2]['source']))
code('''def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def save_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)

if DATA_DIR is None:
    candidates = [p.parent for p in Path('/kaggle/input').rglob('train_solo_track.parquet') if (p.parent / 'test_solo_track.parquet').is_file()]
    if not candidates and Path('data/train_solo_track.parquet').is_file():
        candidates = [Path('data')]
    if len(candidates) != 1:
        raise ValueError(f'Укажите DATA_DIR: найдено {len(candidates)} папок с train/test')
    DATA_DIR = candidates[0]
DATA_DIR = Path(DATA_DIR)
TRAIN_PATH = DATA_DIR / 'train_solo_track.parquet'
TEST_PATH = DATA_DIR / 'test_solo_track.parquet'
history = pd.read_parquet(TRAIN_PATH, columns=['route_id', 'timestamp', TARGET, *STATUSES]).sort_values(['timestamp', 'route_id'])
test = pd.read_parquet(TEST_PATH, columns=['id', 'timestamp', 'route_id'])
times = pd.DatetimeIndex(history.timestamp.unique())
routes = np.sort(history.route_id.unique())
if history.isna().any().any() or history.duplicated(['timestamp', 'route_id']).any():
    raise ValueError('Пропуски или дубликаты в исходной истории')
if not times.equals(pd.date_range(times[0], times[-1], freq=STEP)) or len(history) != len(times) * len(routes):
    raise ValueError('Нарушена полная получасовая панель')
end = times[-1]
if (end.hour, end.minute) != (10, 30):
    raise ValueError('История должна заканчиваться в 10:30')
expected = pd.MultiIndex.from_product([routes, [end + h * STEP for h in range(1, 9)]], names=['route_id', 'timestamp'])
actual = pd.MultiIndex.from_frame(test[['route_id', 'timestamp']])
if test.isna().any().any() or not test.id.is_unique or not actual.is_unique or len(actual) != len(expected) or len(expected.difference(actual)):
    raise ValueError('Некорректные конкурсные запросы')
dates = times[(times.hour == 10) & (times.minute == 30) & (times + 8 * STEP <= end)]
if len(dates) < TEST_DAYS + 20:
    raise ValueError('Недостаточно дат')
before_test = dates[:-TEST_DAYS]
split_index = int(math.floor(len(before_test) * TRAIN_FRACTION))
val_start = before_test[split_index].normalize()
test_start = dates[-TEST_DAYS].normalize()
val_dates = dates[(dates >= val_start) & (dates < test_start)]
calibration_select_start = val_dates[len(val_dates) // 2].normalize()
raw_target = history[TARGET].to_numpy(dtype=np.float32).reshape(len(times), len(routes)).copy()
route_levels = np.median(raw_target[times < val_start], axis=0)
route_groups = pd.DataFrame({'route_id': routes, 'volume_group': pd.qcut(pd.Series(route_levels).rank(method='first'), 5, labels=False).to_numpy()})
route_groups.to_csv(RUN_DIR / 'route_volume_groups_train.csv', index=False)
display(pd.DataFrame([{'split': 'train', 'before': str(val_start)}, {'split': 'calibration_fit', 'from': str(val_start), 'before': str(calibration_select_start)}, {'split': 'calibration_select', 'from': str(calibration_select_start), 'before': str(test_start)}, {'split': 'test', 'from': str(test_start), 'through': str(dates[-1])}]))
''')
md('''## Очистка до признаков

Фиксированный робастный rolling z-score применяется в `log1p`-масштабе отдельно для маршрута, поля и получасового слота суток. Медиана и MAD берутся из предыдущих 28 дней этого же слота, минимум 14 наблюдений; текущая точка исключена. MAD — медиана абсолютных отклонений наблюдений окна от медианы этого же окна. Робастный масштаб равен `max(1.4826 × MAD, 0.25)`, порог — 4. Отмеченные верхние и нижние всплески заменяются `expm1` исторической медианы. Первые 14 дней каждого слота остаются исходными. Параметры не перебираются.

Статистики окна используют предыдущие исходные наблюдения. Нет центрированных окон, интерполяции из будущего или удаления временных строк. На val/test очистка также использует только уже наблюдённую историю; факты test для оценки сохраняются исходными. Нулевые значения тоже проверяются правилом нижнего выброса; число замен нулей выводится отдельно. Такое правило может сглаживать реальные редкие события — эффект оценивается по исходному таргету.
''')
code('''def clean_raw_array(array, timestamps, settings):
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError('Для log1p нужны конечные неотрицательные исходные значения')
    cleaned = array.copy()
    flagged = np.zeros(array.shape, dtype=bool)
    slots = timestamps.hour * 2 + timestamps.minute // 30
    for slot in range(48):
        positions = np.flatnonzero(slots == slot)
        logged = np.log1p(array[positions].astype(np.float64))
        window_days = settings['window_days']
        min_days = settings['min_days']
        if not 1 <= min_days <= window_days:
            raise ValueError('Нужно 1 <= min_days <= window_days')
        if len(positions) <= min_days:
            continue
        padded = np.pad(logged, ((window_days, 0), (0, 0)), constant_values=np.nan)
        windows = np.lib.stride_tricks.sliding_window_view(padded, window_days, axis=0)[:-1]
        eligible = np.arange(len(positions)) >= min_days
        past = windows[eligible]
        median = np.nanmedian(past, axis=-1)
        mad = np.nanmedian(np.abs(past - median[..., None]), axis=-1)
        scale = np.maximum(settings['mad_normalization'] * mad, settings['log_scale_floor'])
        mask = np.abs(logged[eligible] - median) > settings['z_threshold'] * scale
        destination = positions[eligible]
        cleaned[destination] = np.where(mask, np.expm1(median), array[destination]).astype(np.float32)
        flagged[destination] = mask
    return cleaned, flagged

values = {}
cleaning_rows = []
for name in [TARGET, *STATUSES]:
    raw_array = history[name].to_numpy(dtype=np.float32).reshape(len(times), len(routes))
    cleaned, flagged = clean_raw_array(raw_array, times, CLEANING)
    values[name] = cleaned
    for split, mask in [('train', times < val_start), ('val', (times >= val_start) & (times < test_start)), ('test_and_final_history', times >= test_start)]:
        marks = flagged[mask]
        cleaning_rows.append({'column': name, 'split': split, 'rows': int(marks.size), 'replaced': int(marks.sum()), 'share': float(marks.mean()), 'zeros_replaced': int((marks & (raw_array[mask] == 0)).sum())})
    positions, columns = np.nonzero(flagged)
    sample = pd.DataFrame({'timestamp': times[positions[:1000]], 'route_id': routes[columns[:1000]], 'raw': raw_array[positions[:1000], columns[:1000]], 'clean': cleaned[positions[:1000], columns[:1000]]})
    sample.to_csv(RUN_DIR / f'cleaning_sample_{name}.csv', index=False)
    del cleaned, flagged, raw_array
cleaning_report = pd.DataFrame(cleaning_rows)
cleaning_report.to_csv(RUN_DIR / 'cleaning_report.csv', index=False)
display(cleaning_report)
del history
''')
code('FEATURE_CODE = ' + repr(feature_code) + '\nexec(FEATURE_CODE, globals())')
md('''## Кэш очищенных признаков

Оба варианта используют одни и те же признаки из очищенных рядов. В кэше сохраняются все доступные получасовые старты; выбор девяти стартов выполняется маской при fit. `y` — исходный факт для метрик, `y_clean` — очищенная цель обучения. Оба поля исключены из модельных входов. Словарь корреляционных лагов для независимой оценки оценивается только до val; перед полным refit он переоценивается на всей истории. Старый неочищенный кэш несовместим. Совместимый кэш ищется в подключённых Kaggle Dataset; при необходимости задайте CACHE_INPUT. Незавершённый локальный кэш достраивается по горизонтам.
''')
code('''validation_lag_table = calculate_dynamic_shoulder(target, hub_signal, times, routes, fit_before=val_start, max_lookback_hours=LAG_MAX_LOOKBACK_HOURS, min_pairs=LAG_MIN_PAIRS, min_correlation=LAG_MIN_CORRELATION)
validation_lag_table.to_csv(RUN_DIR / 'status6_lags_train.csv', index=False)
cache_spec = {'version': PIPELINE_VERSION, 'source_sha256': file_hash(TRAIN_PATH), 'requests_sha256': file_hash(TEST_PATH), 'feature_code_sha256': hashlib.sha256(FEATURE_CODE.encode()).hexdigest(), 'cleaning': CLEANING, 'cleaning_code_sha256': CLEANING_IMPLEMENTATION_SHA256, 'val_start': str(val_start), 'test_start': str(test_start), 'lag_settings': [LAG_MAX_LOOKBACK_HOURS, LAG_MIN_PAIRS, LAG_MIN_CORRELATION]}
fingerprint = hashlib.sha256(json.dumps(cache_spec, sort_keys=True).encode()).hexdigest()
LOCAL_CACHE = OUTPUT_DIR / 'feature_cache' / fingerprint

def cache_complete(folder):
    folder = Path(folder)
    try:
        metadata = json.loads((folder / 'manifest.json').read_text())
        if metadata['fingerprint'] != fingerprint:
            return False
        for h in range(1, 9):
            record = json.loads((folder / f'h{h}.json').read_text())
            if record['fingerprint'] != fingerprint or record['history_sha256'] != file_hash(folder / f'history_h{h}.parquet') or record['future_sha256'] != file_hash(folder / f'future_h{h}.parquet'):
                return False
        return True
    except (OSError, ValueError, KeyError):
        return False

if CACHE_INPUT is not None:
    if not cache_complete(CACHE_INPUT):
        raise ValueError('CACHE_INPUT несовместим или неполон')
    CACHE_DIR = Path(CACHE_INPUT)
else:
    candidates = [LOCAL_CACHE] + [p.parent for p in Path('/kaggle/input').rglob('manifest.json')]
    CACHE_DIR = next((folder for folder in candidates if cache_complete(folder)), LOCAL_CACHE)
if CACHE_DIR == LOCAL_CACHE:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    save_json(CACHE_DIR / 'manifest.json', {'fingerprint': fingerprint, 'spec': cache_spec, 'source_end': str(end), 'routes': routes.tolist()})
print('Кэш:', CACHE_DIR)
for h in range(1, 9):
    path = CACHE_DIR / f'history_h{h}.parquet'
    future_path = CACHE_DIR / f'future_h{h}.parquet'
    record_path = CACHE_DIR / f'h{h}.json'
    try:
        record = json.loads(record_path.read_text())
        ready = record['fingerprint'] == fingerprint and record['history_sha256'] == file_hash(path) and record['future_sha256'] == file_hash(future_path)
    except (OSError, ValueError, KeyError):
        ready = False
    if ready:
        print(f'h{h}: совместимый кэш найден')
        continue
    origins = np.arange(len(times) - h)
    temporary = path.with_suffix('.partial')
    writer = None
    try:
        for start in range(0, len(origins), 24):
            block = origins[start:start + 24]
            frame = horizon_dataset(h, block)
            frame['y'] = raw_target[block + h].ravel()
            frame['y_clean'] = target[block + h].ravel()
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression='zstd')
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(path)
    future = horizon_dataset(h, np.array([len(times) - 1]))
    requests = test.loc[test.timestamp == end + h * STEP]
    future = requests.merge(future, on=['route_id', 'timestamp'], how='left', validate='one_to_one')
    if not future.cutoff.eq(end).all():
        raise ValueError('Ошибка будущих признаков')
    future.to_parquet(future_path, index=False)
    save_json(record_path, {'fingerprint': fingerprint, 'history_sha256': file_hash(path), 'future_sha256': file_hash(future_path), 'rows': len(origins) * len(routes)})
    print(f'h{h}: {len(origins) * len(routes):,} строк')
for name in ['values', 'means', 'level24', 'extra_signals', 'network_values', 'frame', 'table', 'future', 'raw_target']:
    globals().pop(name, None)
gc.collect()
save_json(RUN_DIR / 'fixed_settings.json', {'source_notebook_sha256': SOURCE_NOTEBOOK_SHA256, 'parameter_source_sha256': PARAMETER_SOURCE_SHA256, 'params': BEST_PARAMS, 'cleaning': CLEANING, 'feature_fingerprint': fingerprint, 'val_start': str(val_start), 'calibration_select_start': str(calibration_select_start), 'test_start': str(test_start), 'test_used_for_selection': False})
''')
code(metric_code + '''

def model_columns(frame):
    return frame.columns.drop(['cutoff', 'timestamp', 'y', 'y_clean', 'overlap']).tolist()

def start_mask(frame, variant):
    if variant == 'all_halfhours':
        return np.ones(len(frame), dtype=bool)
    minutes = frame.cutoff.dt.hour * 60 + frame.cutoff.dt.minute
    return minutes.isin(START_MINUTES).to_numpy()

def make_model(h):
    return xgb.XGBRegressor(**BEST_PARAMS[str(h)], objective=OBJECTIVE, eval_metric='mae', tree_method='hist', device=DEVICE, enable_categorical=True, max_cat_to_onehot=1, random_state=SEED, n_jobs=N_JOBS)

def require_cuda(model):
    actual = json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
    if not actual.startswith('cuda'):
        raise RuntimeError(f'Обучение перешло на {actual}')

route_dtype = pd.CategoricalDtype(categories=routes)
evaluation_parts = {variant: [] for variant in VARIANTS}
model_metric_parts = []
feature_columns = {}
for h in range(1, 9):
    data = pd.read_parquet(CACHE_DIR / f'history_h{h}.parquet')
    data['route_id'] = data.route_id.astype(route_dtype)
    columns = model_columns(data)
    feature_columns[h] = columns
    train_available = data.timestamp.lt(val_start).to_numpy()
    eval_mask = data.cutoff.isin(dates[dates >= val_start]).to_numpy()
    train1030 = train_available & data.cutoff.isin(dates[dates < val_start]).to_numpy()
    for variant in VARIANTS:
        mask = train_available & start_mask(data, variant)
        model = make_model(h)
        model.fit(data.loc[mask, columns], data.loc[mask, 'y_clean'] / TARGET_SCALE, verbose=False)
        require_cuda(model)
        model.save_model(RUN_DIR / f'{variant}_h{h}_before_refit.ubj')
        train_pred = np.maximum(model.predict(data.loc[mask, columns]).astype(np.float64) * TARGET_SCALE, 0)
        train_table = metric_row(data.loc[mask], train_pred, 'train_fit_starts_raw_target')
        part = data.loc[train1030 | eval_mask, ['cutoff', 'timestamp', 'route_id', 'y']].copy()
        part['prediction'] = np.maximum(model.predict(data.loc[train1030 | eval_mask, columns]).astype(np.float64) * TARGET_SCALE, 0)
        part['horizon'] = h
        part['split'] = np.where(part.cutoff < val_start, 'train_1030', np.where(part.cutoff < test_start, 'val', 'test'))
        rows = [train_table]
        for split in ['train_1030', 'val', 'test']:
            block = part.loc[part.split.eq(split)]
            rows.append(metric_row(block, block.prediction, split))
        table = pd.DataFrame(rows)
        table.insert(0, 'horizon', h)
        table.insert(0, 'variant', variant)
        model_metric_parts.append(table)
        display(table)
        evaluation_parts[variant].append(part)
        part.to_parquet(RUN_DIR / f'{variant}_h{h}_evaluation.parquet', index=False)
        del model, train_pred
        gc.collect()
    del data
    gc.collect()
model_metrics = pd.concat(model_metric_parts, ignore_index=True)
model_metrics.to_csv(RUN_DIR / 'model_metrics_before_refit.csv', index=False)
save_json(RUN_DIR / 'feature_columns.json', feature_columns)
evaluation = {variant: pd.concat(parts, ignore_index=True) for variant, parts in evaluation_parts.items()}
del evaluation_parts
''')
md('''## Четыре калибровки

1. `none`: исходный прогноз.
2. `global`: сумма исходных целей / сумма прогнозов.
3. `grouped`: отдельный множитель для горизонта × квинтиля объёма маршрута. Квинтили определены по исходной train-истории. Групповой множитель стягивается к общему с фиксированными 2 000 псевдостроками.
4. `nonlinear`: общая монотонная кусочно-линейная функция прогноза. Изотоническая регрессия обучается на не более чем 128 квантильных корзинах; веса — число строк. За границами обученного диапазона сохраняется отношение крайнего откалиброванного значения к исходному, чтобы не обрезать большие прогнозы до константы.

Сравнение выполняется по общей WAPE + |Relative Bias| на поздней половине val, отдельно для двух вариантов. Калибраторы на этом шаге видят только раннюю половину val. Затем все четыре переоцениваются на полном val, а тип победителя остаётся зафиксированным. Ни XGBoost, ни параметры очистки/калибровки не перебираются. Таблицы test для всех калибровок диагностические и не меняют выбор.
''')
code('''def calibration_keys(frame):
    mapping = route_groups.set_index('route_id').volume_group
    group = frame.route_id.astype('int64').map(mapping)
    if group.isna().any():
        raise ValueError('Маршрут без группы объёма')
    return frame.horizon.astype(str) + ':' + group.astype(int).astype(str)

def fit_calibrator(frame, method):
    actual = frame.y.to_numpy(dtype=np.float64)
    pred = frame.prediction.to_numpy(dtype=np.float64)
    if len(frame) == 0 or pred.sum() <= 0:
        raise ValueError('Недостаточно прогнозов для калибровки')
    global_factor = float(actual.sum() / pred.sum())
    result = {'method': method, 'global_factor': global_factor}
    if method == 'grouped':
        groups = pd.DataFrame({'key': calibration_keys(frame).to_numpy(), 'y': actual, 'pred': pred}).groupby('key').agg(y=('y', 'sum'), pred=('pred', 'sum'), n=('y', 'size'))
        local = groups.y / groups.pred.replace(0, np.nan)
        weight = groups.n / (groups.n + GROUP_SHRINKAGE_ROWS)
        factors = (weight * local + (1 - weight) * global_factor).fillna(global_factor)
        result['factors'] = {str(k): float(v) for k, v in factors.items()}
    elif method == 'nonlinear':
        edges = np.unique(np.quantile(pred, np.linspace(0, 1, ISOTONIC_BINS + 1)))
        bins = np.searchsorted(edges[1:-1], pred, side='right')
        binned = pd.DataFrame({'bin': bins, 'x': pred / TARGET_SCALE, 'y': actual / TARGET_SCALE}).groupby('bin').agg(x=('x', 'mean'), y=('y', 'mean'), weight=('y', 'size')).sort_values('x')
        if len(binned) < 2:
            result['fallback_global'] = True
        else:
            iso = IsotonicRegression(y_min=0, out_of_bounds='clip')
            iso.fit(binned.x, binned.y, sample_weight=binned.weight)
            result['x'] = (iso.X_thresholds_ * TARGET_SCALE).tolist()
            result['y'] = (iso.y_thresholds_ * TARGET_SCALE).tolist()
    return result

def apply_calibrator(frame, calibrator):
    pred = frame.prediction.to_numpy(dtype=np.float64)
    method = calibrator['method']
    if method == 'none':
        result = pred.copy()
    elif method == 'global' or calibrator.get('fallback_global', False):
        result = pred * calibrator['global_factor']
    elif method == 'grouped':
        factor = calibration_keys(frame).map(calibrator['factors']).fillna(calibrator['global_factor']).to_numpy(dtype=np.float64)
        result = pred * factor
    elif method == 'nonlinear':
        x = np.asarray(calibrator['x'])
        y = np.asarray(calibrator['y'])
        result = np.interp(pred, x, y)
        lower = pred < x[0]
        upper = pred > x[-1]
        result[lower] = pred[lower] * (y[0] / x[0] if x[0] > 0 else calibrator['global_factor'])
        result[upper] = pred[upper] * (y[-1] / x[-1] if x[-1] > 0 else calibrator['global_factor'])
    else:
        raise ValueError(method)
    if not np.isfinite(result).all():
        raise ValueError('Неконечный результат калибровки')
    return np.maximum(result, 0)

calibrators = {}
selected_calibrations = {}
selection_rows = []
for variant, part in evaluation.items():
    fit_part = part.loc[part.split.eq('val') & part.cutoff.lt(calibration_select_start)]
    select_part = part.loc[part.split.eq('val') & part.cutoff.ge(calibration_select_start)]
    variant_rows = []
    for method in CALIBRATIONS:
        fitted = fit_calibrator(fit_part, method)
        pred = apply_calibrator(select_part, fitted)
        row = {'variant': variant, 'calibration': method, **metric_row(select_part, pred, 'calibration_select')}
        variant_rows.append(row)
    selected_calibrations[variant] = min(variant_rows, key=lambda row: row['score'])['calibration']
    selection_rows.extend(variant_rows)
    val_part = part.loc[part.split.eq('val')]
    calibrators[variant] = {method: fit_calibrator(val_part, method) for method in CALIBRATIONS}
selection = pd.DataFrame(selection_rows)
display(selection)
selection.to_csv(RUN_DIR / 'calibration_selection.csv', index=False)
save_json(RUN_DIR / 'calibrators.json', {'fit_period': [str(val_start), str(test_start)], 'selection_split': str(calibration_select_start), 'selected': selected_calibrations, 'calibrators': calibrators, 'group_shrinkage_rows': GROUP_SHRINKAGE_ROWS, 'isotonic_bins': ISOTONIC_BINS})
calibration_metric_rows = []
for variant, part in evaluation.items():
    for method, fitted in calibrators[variant].items():
        for split in ['train_1030', 'val', 'test']:
            block = part.loc[part.split.eq(split)]
            pred = apply_calibrator(block, fitted)
            calibration_metric_rows.append({'variant': variant, 'calibration': method, 'selected_on_val': method == selected_calibrations[variant], **metric_row(block, pred, split)})
calibration_metrics = pd.DataFrame(calibration_metric_rows)
calibration_metrics.to_csv(RUN_DIR / 'independent_calibrated_metrics_before_refit.csv', index=False)
display(calibration_metrics)
print('Выбор только на val:', selected_calibrations)
print('Независимая от выбора по val оценка test получена до переобучения. Даты уже изучались в проекте.')
''')
md('''## Полное переобучение

Все 16 моделей заново обучаются на всей доступной очищенной размеченной истории: восемь для всех получасов и восемь для девяти стартов. Число деревьев фиксировано из `refit_params`, early stopping отсутствует. Статистический лаг переоценивается на полной истории. Причинный rolling-фильтр не имеет отдельного fit на будущем и остаётся тем же. Калибраторы, обученные на исходных фактах val и прогнозах моделей до refit, фиксируются; по in-sample прогнозам полной модели они не переобучаются. Test после refit не оценивается.
''')
code('''full_lag_table = calculate_dynamic_shoulder(target, hub_signal, times, routes, fit_before=end + STEP, max_lookback_hours=LAG_MAX_LOOKBACK_HOURS, min_pairs=LAG_MIN_PAIRS, min_correlation=LAG_MIN_CORRELATION)
full_lag_table.to_csv(RUN_DIR / 'status6_lags_full.csv', index=False)
future_parts = {variant: [] for variant in VARIANTS}
full_rows = []
for h in range(1, 9):
    data = add_correlation_features(pd.read_parquet(CACHE_DIR / f'history_h{h}.parquet'), h, full_lag_table)
    future = add_correlation_features(pd.read_parquet(CACHE_DIR / f'future_h{h}.parquet'), h, full_lag_table)
    data['route_id'] = data.route_id.astype(route_dtype)
    future['route_id'] = future.route_id.astype(route_dtype)
    columns = feature_columns[h]
    for variant in VARIANTS:
        mask = start_mask(data, variant)
        model = make_model(h)
        model.fit(data.loc[mask, columns], data.loc[mask, 'y_clean'] / TARGET_SCALE, verbose=False)
        require_cuda(model)
        model.save_model(RUN_DIR / f'{variant}_h{h}_full.ubj')
        result = future[['id', 'route_id', 'timestamp', 'cutoff']].copy()
        result['horizon'] = h
        result['prediction'] = np.maximum(model.predict(future[columns]).astype(np.float64) * TARGET_SCALE, 0)
        future_parts[variant].append(result)
        pred = np.maximum(model.predict(data.loc[mask, columns]).astype(np.float64) * TARGET_SCALE, 0)
        row = {'variant': variant, 'horizon': h, **metric_row(data.loc[mask], pred, 'full_history_after_refit_raw_target')}
        full_rows.append(row)
        old = model_metrics.loc[model_metrics.variant.eq(variant) & model_metrics.horizon.eq(h)]
        display(pd.concat([old, pd.DataFrame([row])], ignore_index=True))
        del model, pred
        gc.collect()
    del data, future
    gc.collect()
pd.DataFrame(full_rows).to_csv(RUN_DIR / 'full_fit_metrics.csv', index=False)
''')
code('''submission_rows = []
for variant in VARIANTS:
    frame = pd.concat(future_parts[variant], ignore_index=True)
    if len(frame) != len(test) or not frame.id.is_unique or set(frame.horizon.unique()) != set(range(1, 9)):
        raise ValueError('Неполный набор прогнозов')
    frame.to_parquet(RUN_DIR / f'{variant}_future_raw.parquet', index=False)
    for method in CALIBRATIONS:
        frame['y_pred'] = apply_calibrator(frame, calibrators[variant][method])
        submission = test[['id']].merge(frame[['id', 'y_pred']], on='id', how='left', validate='one_to_one')
        if len(submission) != len(test) or not submission.id.equals(test.id.reset_index(drop=True)) or not np.isfinite(submission.y_pred).all() or (submission.y_pred < 0).any():
            raise ValueError('Некорректный сабмит')
        path = RUN_DIR / f'submission_{variant}_{method}.csv'
        submission.to_csv(path, index=False)
        if method == selected_calibrations[variant]:
            submission.to_csv(OUTPUT_DIR / f'submission_{variant}_selected.csv', index=False)
        submission_rows.append({'variant': variant, 'calibration': method, 'selected_on_val': method == selected_calibrations[variant], 'file': str(path), 'rows': len(submission)})
submissions = pd.DataFrame(submission_rows)
submissions.to_csv(RUN_DIR / 'submission_files.csv', index=False)
display(submissions)
print('Выбранные сабмиты:', OUTPUT_DIR / 'submission_all_halfhours_selected.csv', OUTPUT_DIR / 'submission_nine_starts_selected.csv')
print('Независимая оценка test получена до полного переобучения; таблица independent_calibrated_metrics_before_refit.csv.')
''')
cleaning_cell = next(cell for cell in cells if cell['cell_type'] == 'code' and 'def clean_raw_array(' in ''.join(cell['source']))
cleaning_tree = ast.parse(''.join(cleaning_cell['source']))
cleaning_function = next(node for node in cleaning_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'clean_raw_array')
cleaning_hash = hashlib.sha256(ast.unparse(cleaning_function).encode()).hexdigest()
cleaning_cell['source'].insert(0, 'CLEANING_IMPLEMENTATION_SHA256 = ' + repr(cleaning_hash) + '\n')
notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.11'}, 'accelerator': 'GPU'}, 'nbformat': 4, 'nbformat_minor': 5}
output = ROOT / 'notebooks/xgb_tuning/xgb_cleaning.ipynb'
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
print(output)
