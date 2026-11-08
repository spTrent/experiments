history = pd.read_parquet(TRAIN_PATH, columns=['route_id', 'timestamp', TARGET, *STATUSES]).sort_values(['timestamp', 'route_id'])
test = pd.read_parquet(TEST_PATH, columns=['id', 'route_id', 'timestamp'])
times = pd.DatetimeIndex(history.timestamp.unique())
routes = np.sort(history.route_id.unique())
if not (history.notna().all().all() and test.notna().all().all()):
    raise ValueError('В данных есть пропуски')
if history.duplicated(['route_id', 'timestamp']).any():
    raise ValueError('Повторяется ключ маршрута и времени')
if not times.equals(pd.date_range(times[0], times[-1], freq=STEP)):
    raise ValueError('Нарушена получасовая сетка')
if not len(history) == len(times) * len(routes):
    raise ValueError('Неполная панель маршрутов')
if not (test.id.is_unique and (not test.duplicated(['route_id', 'timestamp']).any())):
    raise ValueError('Повторы идентификаторов или ключей теста')
end = times[-1]
if not (end.hour, end.minute) == (10, 30):
    raise ValueError('Последняя точка истории должна быть в 10:30')
expected = pd.MultiIndex.from_product([routes, [end + h * STEP for h in range(1, 9)]], names=['route_id', 'timestamp'])
if not expected.difference(pd.MultiIndex.from_frame(test[['route_id', 'timestamp']])).empty:
    raise ValueError('Тест не покрывает все ожидаемые маршруты и горизонты')
if not len(test) == len(expected):
    raise ValueError('Число строк теста отличается от ожидаемого')
values = {name: history[name].to_numpy(dtype=np.float32).reshape(len(times), len(routes)) for name in [TARGET, *STATUSES]}
del history
if not all((np.isfinite(array).all() for array in values.values())):
    raise ValueError('В числовых рядах есть бесконечные значения')
dates = times[(times.hour == 10) & (times.minute == 30) & (times + 8 * STEP <= end)]
selection_end = pd.Timestamp('2025-09-20')
if len(dates) < 14 or dates[-14].normalize() < selection_end or times[-1] < selection_end:
    raise ValueError('Недостаточно истории для отдельного test после подбора')
holdout_start = dates[-14]
selection_dates = dates[dates + 8 * STEP < selection_end]
if len(selection_dates) <= 28:
    raise ValueError('До 20 сентября недостаточно истории для train и трёх недель подбора')
folds = []
for start in range(len(selection_dates) - 21, len(selection_dates), 7):
    folds.append(dict(train=selection_dates[:start-7], val=selection_dates[start-7:start],
                      tuning=selection_dates[start:start+7], test=dates[-14:]))
split_dates = folds[-1]
tune_start = split_dates['tuning'][0]
LAG_VALIDATION_FIT_BEFORE = split_dates['val'][0].normalize()
display(pd.DataFrame([dict(fold=i+1, period=k, first=v[0], last=v[-1], days=len(v)) for i,f in enumerate(folds) for k,v in f.items()]))


import pyarrow as pa
import pyarrow.parquet as pq
target = values[TARGET]

def past(array, positions):
    positions = np.asarray(positions, dtype=np.int64)
    valid = (positions >= 0) & (positions < len(array))
    result = array[np.clip(positions, 0, len(array) - 1)].astype(np.float32)
    result[~valid] = np.nan
    return result

def average_available(matrices):
    total = np.zeros_like(matrices[0], dtype=np.float32)
    count = np.zeros_like(total)
    for matrix in matrices:
        valid = np.isfinite(matrix)
        total += np.where(valid, matrix, 0)
        count += valid
    return np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)

def rolling(array, steps):
    return pd.DataFrame(array).rolling(steps, min_periods=1).mean().to_numpy(dtype=np.float32)
means = {(name, steps): rolling(array, steps) for name, array in values.items() for steps in (4, 12)}
level24 = rolling(target, 48)
positions = np.arange(len(times))
extra_signals = {}
for name in STATUSES[:3]:
    extra_signals[name + '_hourly'] = values[name] + past(values[name], positions - 1)
extra_signals['warehouse_total_load'] = sum((extra_signals[name + '_hourly'] for name in STATUSES[:3]))
extra_signals['warehouse_processing_rate'] = extra_signals['status_3_hourly'] / (extra_signals['status_1_hourly'] + np.float32(1e-05))
for name in STATUSES[3:]:
    extra_signals[name + '_diff_hourly'] = values[name] - past(values[name], positions - 2)
extra_signals['hub_buffer'] = values['status_5'] - values['status_4']
extra_signals['hub_gate_bottleneck'] = extra_signals['status_4_diff_hourly'] - extra_signals['status_6_diff_hourly']
extra_signals['target_acceleration'] = past(target, positions - 1) - past(target, positions - 2)
hub_signal = extra_signals['status_6_diff_hourly']
extra_signals['target_vs_mean_6h'] = target - means[TARGET, 12]
recent = means[TARGET, 4]
usual = average_available([past(recent, positions - day * 48) for day in range(1, 8)])
extra_signals['morning_level_ratio'] = np.divide(recent, usual, out=np.full_like(recent, np.nan), where=usual > 0)

def offset_name(base, offset):
    if offset == 0:
        return base
    return f"{base}_offset_{('m' if offset < 0 else 'p')}{abs(offset)}"

def add_window(common, base, array, anchor, offsets=WINDOW_OFFSETS):
    for offset in offsets:
        common[offset_name(base, offset)] = past(array, anchor + offset)

def common_features(origins):
    common = {}
    local_target_lags = sorted({k - offset for k in (0, 1, 2, 4) for offset in WINDOW_OFFSETS if k - offset >= 0})
    for lag in local_target_lags:
        common[f'target_back_{lag}'] = past(target, origins - lag)
    causal_offsets = tuple((offset for offset in WINDOW_OFFSETS if offset <= 0))
    for name, array in values.items():
        for steps in (4, 12):
            add_window(common, f'{name}_mean_{steps}', means[name, steps], origins, causal_offsets)
        if name in STATUSES:
            common[f'{name}_latest'] = array[origins]
            for lag in range(1, 5):
                common[f'{name}_back_{lag}'] = past(array, origins - lag)
            if name in STATUSES[:3]:
                change = array - past(array, positions - 2)
                add_window(common, f'{name}_change_1h', change, origins, causal_offsets)
    for name, array in extra_signals.items():
        add_window(common, name, array, origins, causal_offsets)
    for lag in (48, 96):
        add_window(common, f'target_1h_lag_{lag}', target, origins - lag)
    add_window(common, 'warehouse_total_load_lag_48', extra_signals['warehouse_total_load'], origins - 48)
    add_window(common, 'status_6_diff_hourly_lag_48', hub_signal, origins - 48)
    return common
CORRELATION_COLUMNS = ['selected_status6_lag_steps', 'selected_status6_correlation', 'selected_status6_pair_count', 'status6_lag_valid', 'lagged_status6_signal', 'lagged_status6_signal_available']

for name, series in values.items():
    network = series.sum(axis=1, dtype=np.float64).astype(np.float32)[:, None]
    for suffix, signal in [('latest', network), ('change_1h', network - past(network, positions-2))]:
        extra_signals[f'network_{name}_{suffix}'] = np.broadcast_to(signal, target.shape)


def calculate_dynamic_shoulder(target_values, status6_diff, timestamps, route_ids, *, fit_before, max_lookback_hours=24, min_pairs=336, min_correlation=0.05, route_block_size=128):
    grid = pd.DatetimeIndex(timestamps)
    route_ids = np.asarray(route_ids)
    target_values, status6_diff = (np.asarray(target_values), np.asarray(status6_diff))
    if target_values.shape != status6_diff.shape or target_values.shape != (len(grid), len(route_ids)):
        raise ValueError('Ожидаются два массива [время, маршрут] одной формы')
    if len(grid) == 0 or not grid.equals(pd.date_range(grid[0], periods=len(grid), freq=STEP)):
        raise ValueError('Для выбора лага нужна упорядоченная получасовая сетка без дубликатов')
    if len(np.unique(route_ids)) != len(route_ids):
        raise ValueError('Идентификаторы маршрутов повторяются')
    steps = float(max_lookback_hours) * 2
    if not np.isfinite(steps) or steps < 2 or (not steps.is_integer()):
        raise ValueError('Граница поиска должна задавать целое число получасов, не меньше двух')
    if not isinstance(min_pairs, (int, np.integer)) or min_pairs < 3:
        raise ValueError('min_pairs должен быть целым числом >=3')
    if not 0 <= min_correlation <= 1 or route_block_size < 1:
        raise ValueError('Некорректные параметры выбора корреляции')
    max_lag = int(steps)
    stop = int(grid.searchsorted(pd.Timestamp(fit_before), side='left'))
    anchors = np.arange(max_lag + 2, stop)
    best_corr = np.full(len(route_ids), np.nan)
    best_lag = np.full(len(route_ids), np.nan)
    best_pairs = np.zeros(len(route_ids), dtype=np.int32)
    for start in range(0, len(route_ids), route_block_size):
        finish = min(start + route_block_size, len(route_ids))
        x = target_values[anchors, start:finish].astype(np.float64)
        for lag in range(2, max_lag + 1):
            z = status6_diff[anchors - lag, start:finish].astype(np.float64)
            valid = np.isfinite(x) & np.isfinite(z)
            count = valid.sum(axis=0)
            xx, zz = (np.where(valid, x, 0), np.where(valid, z, 0))
            divisor = np.maximum(count, 1)
            sx, sz = (xx.sum(axis=0), zz.sum(axis=0))
            covariance = (xx * zz).sum(axis=0) - sx * sz / divisor
            vx = np.maximum((xx * xx).sum(axis=0) - sx * sx / divisor, 0)
            vz = np.maximum((zz * zz).sum(axis=0) - sz * sz / divisor, 0)
            denominator = np.sqrt(vx * vz)
            corr = np.divide(covariance, denominator, out=np.full(finish - start, np.nan), where=(count >= min_pairs) & (denominator > 0))
            corr = np.clip(corr, -1, 1)
            previous = best_corr[start:finish]
            better = np.isfinite(corr) & (~np.isfinite(previous) | (corr > previous))
            best_corr[start:finish][better] = corr[better]
            best_lag[start:finish][better] = lag
            best_pairs[start:finish][better] = count[better]
    accepted = np.isfinite(best_corr) & (best_corr > min_correlation)
    return pd.DataFrame({'route_id': route_ids, 'selected_status6_lag_steps': np.where(accepted, best_lag, np.nan), 'selected_status6_correlation': best_corr, 'selected_status6_pair_count': best_pairs, 'status6_lag_valid': accepted.astype(np.int8)})

def correlation_block(frame, h, lag_table, signal_array, prefix, chunk_rows=200000):
    if h not in range(1, 9) or chunk_rows < 1:
        raise ValueError('Некорректный горизонт или размер блока')
    extra_columns = [offset_name(base, offset) for offset in WINDOW_OFFSETS if offset != 0 for base in ('lagged_status6_signal', 'lagged_status6_signal_available')]
    names = CORRELATION_COLUMNS + extra_columns
    result = np.full((len(frame), len(names)), np.nan, dtype=np.float32)
    if lag_table is None:
        names = [prefix + name for name in names]
        return pd.concat([frame.drop(columns=names, errors='ignore'), pd.DataFrame(result, columns=names, index=frame.index)], axis=1)
    by_route = lag_table.set_index('route_id').reindex(routes)
    route_index = pd.Index(routes)
    lag = by_route.selected_status6_lag_steps.to_numpy(dtype=np.float32)
    for start in range(0, len(frame), chunk_rows):
        finish = min(start + chunk_rows, len(frame))
        block = frame.iloc[start:finish]
        origin = times.get_indexer(pd.DatetimeIndex(block.cutoff))
        route = route_index.get_indexer(block.route_id)
        if (origin < 0).any() or (route < 0).any():
            raise ValueError('Строка не сопоставлена с историей')
        selected = lag[route]
        for j, name in enumerate(CORRELATION_COLUMNS[:4]):
            result[start:finish, j] = by_route[name].to_numpy(dtype=np.float32)[route]
        for offset in WINDOW_OFFSETS:
            source = origin + h - np.nan_to_num(selected, nan=0).astype(np.int64) + offset
            available = np.isfinite(selected) & (source >= 0) & (source <= origin) & (source < len(times))
            signal = signal_array[np.clip(source, 0, len(times) - 1), route].astype(np.float32)
            available &= np.isfinite(signal)
            result[start:finish, names.index(offset_name('lagged_status6_signal', offset))] = np.where(available, signal, np.nan)
            result[start:finish, names.index(offset_name('lagged_status6_signal_available', offset))] = available.astype(np.float32)
    names = [prefix + name for name in names]
    frame = frame.drop(columns=names, errors='ignore')
    return pd.concat([frame, pd.DataFrame(result, columns=names, index=frame.index)], axis=1)
def remove_seasonality(series):
    typical = average_available([past(series, positions - 48*d) for d in range(1,8)])
    return series - typical

residual_target = remove_seasonality(target)
residual_signal = remove_seasonality(hub_signal)

def fit_lags(boundary):
    tables = []
    for y, signal, prefix in ((target,hub_signal,''),(residual_target,residual_signal,'residual_')):
        table = calculate_dynamic_shoulder(y,signal,times,routes,fit_before=boundary,max_lookback_hours=LAG_MAX_LOOKBACK_HOURS,min_pairs=LAG_MIN_PAIRS,min_correlation=LAG_MIN_CORRELATION)
        tables.append(table.rename(columns={c:prefix+c for c in table.columns if c != 'route_id'}))
    return tables[0].merge(tables[1],on='route_id',validate='one_to_one')

def add_correlation_features(frame,h,lag_table):
    frame = correlation_block(frame,h,lag_table,hub_signal,'')
    if lag_table is None:
        return correlation_block(frame,h,None,residual_signal,'residual_')
    residual_table = lag_table[['route_id',*['residual_'+c for c in CORRELATION_COLUMNS[:4]]]].copy()
    residual_table.columns = ['route_id',*CORRELATION_COLUMNS[:4]]
    return correlation_block(frame,h,residual_table,residual_signal,'residual_')



def horizon_dataset(h, origins):
    slot = origins + h
    dates = times[origins] + h * STEP
    common = common_features(origins)
    frame = pd.DataFrame({'cutoff': times[origins].repeat(len(routes)), 'timestamp': dates.repeat(len(routes)), 'route_id': np.tile(routes, len(origins)), **{name: matrix.ravel() for name, matrix in common.items()}})
    calendar = dict(dayofweek=dates.dayofweek, is_weekend=(dates.dayofweek >= 5).astype(np.int8))
    calendar['weekday_sin'] = np.sin(2 * np.pi * dates.dayofweek / 7)
    calendar['weekday_cos'] = np.cos(2 * np.pi * dates.dayofweek / 7)
    for name, array in calendar.items():
        frame[name] = np.repeat(array, len(routes))
    frame = add_correlation_features(frame, h, None)
    seasonal_columns = {}
    for offset in WINDOW_OFFSETS:
        for day in (1, 2, 7):
            seasonal_columns[offset_name(f'slot_days_back_{day}', offset)] = past(target, slot - day * 48 + offset).ravel()
        for name, days in [('slot_daily7', range(1, 8)), ('slot_weekly4', (7, 14, 21, 28))]:
            seasonal_columns[offset_name(name, offset)] = average_available([past(target, slot - day * 48 + offset) for day in days]).ravel()
    for base in ('slot_daily7','slot_weekly4'):
        window = np.column_stack([seasonal_columns[offset_name(base,k)] for k in WINDOW_OFFSETS])
        valid = np.isfinite(window)
        maximum = np.where(valid, window, -np.inf).max(axis=1)
        minimum = np.where(valid, window, np.inf).min(axis=1)
        peak = valid & (window == maximum[:,None])
        count = peak.sum(axis=1)
        shift = np.divide(peak @ np.asarray(WINDOW_OFFSETS, dtype=float), count, out=np.full(len(window),np.nan), where=count>0)
        seasonal_columns[base+'_peak_minutes'] = shift * 30
        seasonal_columns[base+'_range'] = np.where(count>0, maximum-minimum, np.nan)
        seasonal_columns[base+'_right_minus_left'] = window[:,5:].mean(axis=1)-window[:,:4].mean(axis=1)
    frame = pd.concat([frame, pd.DataFrame(seasonal_columns, index=frame.index)], axis=1)
    profile = np.zeros((len(origins), len(routes)))
    counts = np.zeros_like(profile)
    for day in range(1, 29):
        level = past(level24, origins - day * 48)
        adjusted = past(target, slot - day * 48) * level24[origins] / np.maximum(level, 1)
        profile += np.nan_to_num(adjusted)
        counts += np.isfinite(adjusted)
    frame['profile28'] = np.divide(profile, counts, out=np.full_like(profile, np.nan), where=counts > 0).ravel()
    if h == 1:
        lower = np.zeros_like(target[origins])
        upper = past(target, origins - 2)
        for position in (origins - 1, origins):
            total = past(target, position)
            lower, upper = (np.maximum(0, total - upper), np.minimum(total, total - lower))
        seasonal = frame[['slot_daily7', 'slot_weekly4']].mean(axis=1).to_numpy()
        frame['overlap'] = np.where((lower <= upper).ravel(), (lower + upper).ravel() / 2 + seasonal / 2, seasonal)
    else:
        frame['overlap'] = np.nan
    raw_target_sources = []
    for lag in range(9):
        raw_target_sources.append((f'target_back_{lag}', -lag, lag in (0, 1, 2, 4)))
    for lag in (48, 96):
        for offset in WINDOW_OFFSETS:
            raw_target_sources.append((offset_name(f'target_1h_lag_{lag}', offset), -lag + offset, offset == 0))
    for day in (1, 2, 7):
        for offset in WINDOW_OFFSETS:
            raw_target_sources.append((offset_name(f'slot_days_back_{day}', offset), h - day * 48 + offset, offset == 0))
    seen, redundant = (set(), [])
    for name, source, central in sorted(raw_target_sources, key=lambda item: not item[2]):
        if source in seen:
            redundant.append(name)
        else:
            seen.add(source)
    frame = frame.drop(columns=redundant)
    numeric = frame.columns.difference(['cutoff', 'timestamp', 'route_id'])
    return frame.astype({name: 'float32' for name in numeric}).copy()

def write_horizon(h):
    minutes = times.hour * 60 + times.minute
    origins = np.flatnonzero(np.isin(minutes, TRAIN_MINUTES) & (np.arange(len(times)) + h < len(times)))
    path = OUT / f'history_h{h}.parquet'
    temporary = path.with_suffix('.partial')
    writer = None
    try:
        for start in range(0, len(origins), 35):
            batch = origins[start:start+35]
            frame = horizon_dataset(h, batch)
            frame['y'] = target[batch+h].ravel()
            table = pa.Table.from_pandas(frame,preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary,table.schema,compression='zstd')
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(path)
    features = horizon_dataset(h, np.array([len(times) - 1]))
    requests = test.loc[test.timestamp.eq(end + h * STEP)]
    future = requests.merge(features, on=['route_id', 'timestamp'], how='left', validate='one_to_one')
    if not future.cutoff.eq(end).all() or not future.id.equals(requests.id.reset_index(drop=True)):
        raise ValueError('Ошибка сопоставления тестовых строк')
    future.to_parquet(OUT / f'submission_h{h}.parquet', index=False)
    print(f'h={h}: история {len(origins)*len(routes):,}, submission {len(future):,}, модельных признаков {len(frame.columns) - 5}')
for h in range(1, 9):
    write_horizon(h)
feature_names = list(dict.fromkeys((name for h in range(1, 9) for name in pq.read_schema(OUT / f'history_h{h}.parquet').names)))
FEATURE_GROUPS = {'calendar': ['dayofweek', 'is_weekend', 'weekday_sin', 'weekday_cos'], 'local_statuses': [name for name in feature_names if any((name.startswith(prefix) for prefix in ('status_1_hourly', 'status_2_hourly', 'status_3_hourly', 'warehouse_total_load', 'warehouse_processing_rate'))) and 'lag_48' not in name], 'hub_combinations': [name for name in feature_names if any((name.startswith(prefix) for prefix in ('status_4_diff_hourly', 'status_5_diff_hourly', 'status_6_diff_hourly', 'hub_buffer', 'hub_gate_bottleneck'))) and 'lag_48' not in name], 'historical_lags': [name for name in feature_names if 'lag_48' in name or 'lag_96' in name], 'target_change': [name for name in feature_names if name.startswith('target_acceleration')], 'correlation_lag': [name for name in feature_names if name in CORRELATION_COLUMNS or name.startswith('lagged_status6_signal')], 'seasonal_windows': [name for name in feature_names if name.startswith('slot_')], 'recent_target': [name for name in feature_names if name.startswith('target_back_')], 'recent_statuses': [name for name in feature_names if any((name.startswith(f'status_{i}_back_') or name == f'status_{i}_latest' for i in range(1, 7)))]}

del values, means, level24, extra_signals, positions, recent, usual
FEATURE_GROUPS['network'] = [name for name in feature_names if name.startswith('network_')]
FEATURE_GROUPS['residual_correlation_lag'] = [name for name in feature_names if name.startswith('residual_')]
(DOCS_DIR / 'feature_pipeline.json').write_text(json.dumps({'version': FEATURE_PIPELINE_VERSION, 'window_offsets_steps': list(WINDOW_OFFSETS), 'training_origins_minutes': TRAIN_MINUTES, 'evaluation_origin': '10:30', 'correlation_storage': 'NaN placeholders; fit on stage train before model use', 'validation_lag_fit_before': str(LAG_VALIDATION_FIT_BEFORE), 'selection_end_exclusive': str(selection_end), 'all_folds': [{key: [str(period.min()), str(period.max())] for key, period in fold.items()} for fold in folds], 'splits': {key: [str(period.min()), str(period.max())] for key, period in split_dates.items()}, 'feature_groups': FEATURE_GROUPS}, ensure_ascii=False, indent=2))


def mean_daily_score(frame, prediction):
    actual = frame.y.to_numpy(dtype=np.float64)
    errors = np.asarray(prediction, dtype=np.float64) - actual
    totals = pd.DataFrame({'cutoff': frame.cutoff.dt.normalize().to_numpy(), 'volume': actual, 'error': errors, 'absolute': np.abs(errors)}).groupby('cutoff').sum()
    denominator = totals.volume.where(totals.volume > 0)
    return ((totals.absolute + totals.error.abs()) / denominator).mean()

def score_row(frame, predictions, split):
    actual = frame.y.to_numpy(dtype=np.float64)
    errors = np.asarray(predictions, dtype=np.float64) - actual
    volume = actual.sum()
    wape = np.abs(errors).sum() / volume if volume > 0 else np.nan
    signed_bias = errors.sum() / volume if volume > 0 else np.nan
    return {'выборка': split, 'даты': frame.cutoff.dt.normalize().nunique(), 'строки': len(frame), 'WAPE': wape, 'Relative Bias': signed_bias, '|Relative Bias|': abs(signed_bias), 'метрика конкурса': wape + abs(signed_bias), 'среднее по дням': mean_daily_score(frame, predictions)}
TARGET_SCALE = 100000

def model_columns(data):
    return data.columns.drop(['timestamp', 'cutoff', 'y', 'profile28', 'overlap']).tolist()
