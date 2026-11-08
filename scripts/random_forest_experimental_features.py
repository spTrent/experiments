def stable_ratio(numerator, denominator):
    return numerator / np.maximum(np.abs(denominator), 1.0)


def available_statistics(window):
    valid = np.isfinite(window)
    count = valid.sum(axis=0)
    mean = np.divide(np.where(valid, window, 0).sum(axis=0, dtype=np.float64), count,
                     out=np.full(window.shape[1:], np.nan), where=count > 0)
    variance = np.divide(np.where(valid, (window - mean) ** 2, 0).sum(axis=0), count,
                         out=np.full_like(mean, np.nan), where=count > 0)
    spread = np.where(count > 0, np.where(valid, window, -np.inf).max(axis=0) - np.where(valid, window, np.inf).min(axis=0), np.nan)
    return mean, np.sqrt(variance), spread, count


def add_experimental_features(frame, h, origins):
    additions = {}
    additions['x_time_cutoff_minutes'] = np.repeat(times[origins].hour * 60 + times[origins].minute, len(routes))
    additions['x_profile_profile28'] = frame.profile28.to_numpy()
    additions['x_level_level24'] = level24[origins].ravel()
    typical = average_available([past(target, origins - day * 48) for day in range(1, 8)])
    current = target[origins]
    additions['x_anomaly_target_difference'] = (current - typical).ravel()
    additions['x_anomaly_target_ratio'] = stable_ratio(current, typical).ravel()
    for name in (TARGET, *STATUSES[:3]):
        short, long = means[name, 4][origins], means[name, 12][origins]
        additions[f'x_trend_{name}_difference'] = (short - long).ravel()
        additions[f'x_trend_{name}_ratio'] = stable_ratio(short, long).ravel()
        for steps in (4, 12):
            window = np.stack([past(values[name], origins - lag) for lag in range(steps)])
            mean, std, spread, count = available_statistics(window)
            change = available_statistics(np.abs(np.diff(window, axis=0)))[0]
            additions[f'x_volatility_{name}_std_{steps}'] = std.ravel()
            additions[f'x_volatility_{name}_range_{steps}'] = spread.ravel()
            additions[f'x_volatility_{name}_change_{steps}'] = change.ravel()
    for label, days in [('daily7', range(1,8)), ('weekly4', (7,14,21,28))]:
        window = np.stack([past(target, origins + h - day * 48) for day in days])
        mean, std, spread, count = available_statistics(window)
        additions[f'x_dispersion_{label}_std'] = std.ravel()
        additions[f'x_dispersion_{label}_relative_std'] = stable_ratio(std, mean).ravel()
        additions[f'x_dispersion_{label}_count'] = count.ravel()
    return pd.concat([frame, pd.DataFrame(additions, index=frame.index)], axis=1)
