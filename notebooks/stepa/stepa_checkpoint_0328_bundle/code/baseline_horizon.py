def horizon_dataset(h, origins):
    slot = origins + h
    dates = times[origins] + h * STEP
    frame = pd.DataFrame({'cutoff': times[origins].repeat(len(routes)), 'timestamp': dates.repeat(len(routes)), 'route_id': np.tile(routes, len(origins))})
    for name, matrix in common_features(origins).items():
        frame[name] = matrix.ravel()
    calendar = {'hour': dates.hour, 'minute': dates.minute, 'hour_float': dates.hour + dates.minute / 60.0, 'dayofweek': dates.dayofweek, 'is_weekend': (dates.dayofweek >= 5).astype(np.int8)}
    calendar['hour_sin'] = np.sin(2 * np.pi * calendar['hour_float'] / 24)
    calendar['hour_cos'] = np.cos(2 * np.pi * calendar['hour_float'] / 24)
    calendar['weekday_sin'] = np.sin(2 * np.pi * dates.dayofweek / 7)
    calendar['weekday_cos'] = np.cos(2 * np.pi * dates.dayofweek / 7)
    for name, array in calendar.items():
        frame[name] = np.repeat(array, len(routes))
    frame = add_correlation_features(frame, h, validation_lag_table)
    for days in (1, 2, 7):
        frame[f'slot_days_back_{days}'] = past(target, slot - days * 48).ravel()
    for name, days in [('slot_daily7', range(1, 8)), ('slot_weekly4', (7, 14, 21, 28))]:
        frame[name] = average_available([past(target, slot - d * 48) for d in days]).ravel()
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
    numeric = frame.columns.difference(['cutoff', 'timestamp', 'route_id'])
    return frame.astype({name: 'float32' for name in numeric})