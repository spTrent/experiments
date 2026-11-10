def calculate_dynamic_shoulder(target_values, status6_diff, timestamps, route_ids, *,
                               fit_before, max_lookback_hours=24,
                               min_pairs=336, min_correlation=0.05, route_block_size=128):
    """Обучить статистический лаг маршрута на timestamp < fit_before; без print и мутации входов."""
    grid = pd.DatetimeIndex(timestamps)
    route_ids = np.asarray(route_ids)
    target_values, status6_diff = np.asarray(target_values), np.asarray(status6_diff)
    if target_values.shape != status6_diff.shape or target_values.shape != (len(grid), len(route_ids)):
        raise ValueError("Ожидаются два массива [время, маршрут] одной формы")
    if len(grid) == 0 or not grid.equals(pd.date_range(grid[0], periods=len(grid), freq=STEP)):
        raise ValueError("Для выбора лага нужна упорядоченная получасовая сетка без дубликатов")
    if len(np.unique(route_ids)) != len(route_ids):
        raise ValueError("Идентификаторы маршрутов повторяются")
    steps = float(max_lookback_hours) * 2
    if not np.isfinite(steps) or steps < 2 or not steps.is_integer():
        raise ValueError("Граница поиска должна задавать целое число получасов, не меньше двух")
    if not isinstance(min_pairs, (int, np.integer)) or min_pairs < 3:
        raise ValueError("min_pairs должен быть целым числом >=3")
    if not 0 <= min_correlation <= 1 or route_block_size < 1:
        raise ValueError("Некорректные параметры выбора корреляции")
    max_lag = int(steps)
    stop = int(grid.searchsorted(pd.Timestamp(fit_before), side="left"))
    # Первые две точки diff(2) неизвестны; общие якоря одинаковы для всех L.
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
            xx, zz = np.where(valid, x, 0), np.where(valid, z, 0)
            divisor = np.maximum(count, 1)
            sx, sz = xx.sum(axis=0), zz.sum(axis=0)
            covariance = (xx * zz).sum(axis=0) - sx * sz / divisor
            vx = np.maximum((xx * xx).sum(axis=0) - sx * sx / divisor, 0)
            vz = np.maximum((zz * zz).sum(axis=0) - sz * sz / divisor, 0)
            denominator = np.sqrt(vx * vz)
            corr = np.divide(covariance, denominator, out=np.full(finish-start, np.nan),
                             where=(count >= min_pairs) & (denominator > 0))
            corr = np.clip(corr, -1, 1)
            previous = best_corr[start:finish]
            better = np.isfinite(corr) & (~np.isfinite(previous) | (corr > previous))
            best_corr[start:finish][better] = corr[better]
            best_lag[start:finish][better] = lag
            best_pairs[start:finish][better] = count[better]
    accepted = np.isfinite(best_corr) & (best_corr > min_correlation)
    return pd.DataFrame({"route_id": route_ids,
                         "selected_status6_lag_steps": np.where(accepted, best_lag, np.nan),
                         "selected_status6_correlation": best_corr,
                         "selected_status6_pair_count": best_pairs,
                         "status6_lag_valid": accepted.astype(np.int8)})

def add_correlation_features(frame, h, lag_table, chunk_rows=200_000):
    """Применить готовый словарь; источник каждого значения <= собственного origin."""
    if h not in range(1, 9) or chunk_rows < 1:
        raise ValueError("Некорректный горизонт или размер блока")
    by_route = lag_table.set_index("route_id").reindex(routes)
    route_index = pd.Index(routes)
    lag = by_route.selected_status6_lag_steps.to_numpy(dtype=np.float32)
    quality = {name: by_route[name].to_numpy(dtype=np.float32) for name in CORRELATION_COLUMNS[:4]}
    for name in CORRELATION_COLUMNS:
        frame[name] = np.full(len(frame), np.nan, dtype=np.float32)
    column_positions = {name: frame.columns.get_loc(name) for name in CORRELATION_COLUMNS}
    for start in range(0, len(frame), chunk_rows):
        finish = min(start + chunk_rows, len(frame))
        block = frame.iloc[start:finish]
        origin = times.get_indexer(pd.DatetimeIndex(block.cutoff))
        route = route_index.get_indexer(block.route_id)
        if (origin < 0).any() or (route < 0).any():
            raise ValueError("Строка признаков не сопоставлена с историей")
        selected = lag[route]
        source = origin + h - np.nan_to_num(selected, nan=0).astype(np.int64)
        available = (np.isfinite(selected) & (selected >= h) & (source >= 0)
                     & (source <= origin) & (source < len(times)))
        signal = hub_signal[np.clip(source, 0, len(times)-1), route].astype(np.float32)
        available &= np.isfinite(signal)
        for name, array in quality.items():
            frame.iloc[start:finish, column_positions[name]] = array[route]
        frame.iloc[start:finish, column_positions["lagged_status6_signal"]] = np.where(available, signal, np.nan)
        frame.iloc[start:finish, column_positions["lagged_status6_signal_available"]] = available.astype(np.float32)
    return frame