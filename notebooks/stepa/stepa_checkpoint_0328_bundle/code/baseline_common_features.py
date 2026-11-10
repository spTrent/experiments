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

means = {(name, steps): rolling(array, steps)
         for name, array in values.items() for steps in (4, 12)}
level24 = rolling(target, 48)
positions = np.arange(len(times))
extra_signals = {}
for name in STATUSES[:3]:
    extra_signals[name + "_hourly"] = values[name] + past(values[name], positions - 1)
extra_signals["warehouse_total_load"] = sum(extra_signals[name + "_hourly"] for name in STATUSES[:3])
extra_signals["warehouse_processing_rate"] = (
    extra_signals["status_3_hourly"] / (extra_signals["status_1_hourly"] + np.float32(1e-5)))
for name in STATUSES[3:]:
    extra_signals[name + "_diff_hourly"] = values[name] - past(values[name], positions - 2)
extra_signals["hub_buffer"] = values["status_5"] - values["status_4"]
extra_signals["hub_gate_bottleneck"] = (
    extra_signals["status_4_diff_hourly"] - extra_signals["status_6_diff_hourly"])
extra_signals["target_acceleration"] = past(target, positions - 1) - past(target, positions - 2)
hub_signal = extra_signals["status_6_diff_hourly"]

def common_features(origins):
    common = {}
    for k in (0, 1, 2, 4):
        common[f"target_back_{k}"] = past(target, origins - k)
    for name, array in values.items():
        for steps in (4, 12):
            common[f"{name}_mean_{steps}"] = means[name, steps][origins]
        if name in STATUSES:
            common[f"{name}_latest"] = array[origins]
            if name in STATUSES[:3]:
                common[f"{name}_change_1h"] = array[origins] - past(array, origins - 2)
    for name, array in extra_signals.items():
        common[name] = array[origins]
    for lag in (48, 96):
        common[f"target_1h_lag_{lag}"] = past(target, origins - lag)
    common["warehouse_total_load_lag_48"] = past(extra_signals["warehouse_total_load"], origins - 48)
    common["status_6_diff_hourly_lag_48"] = past(hub_signal, origins - 48)
    common["target_vs_mean_6h"] = target[origins] - means[TARGET, 12][origins]
    recent = means[TARGET, 4][origins]
    usual = average_available([past(means[TARGET, 4], origins - day*48) for day in range(1, 8)])
    common["morning_level_ratio"] = np.divide(
        recent, usual, out=np.full_like(recent, np.nan), where=usual > 0)
    return common

CORRELATION_COLUMNS = ["selected_status6_lag_steps", "selected_status6_correlation",
                       "selected_status6_pair_count", "status6_lag_valid",
                       "lagged_status6_signal", "lagged_status6_signal_available"]
FEATURE_GROUPS = {
    "calendar": ["hour", "minute", "hour_float", "hour_sin", "hour_cos", "dayofweek",
                 "is_weekend", "weekday_sin", "weekday_cos"],
    "local_statuses": ["status_1_hourly", "status_2_hourly", "status_3_hourly",
                       "warehouse_total_load", "warehouse_processing_rate"],
    "hub_combinations": ["status_4_diff_hourly", "status_5_diff_hourly", "status_6_diff_hourly",
                         "hub_buffer", "hub_gate_bottleneck"],
    "historical_lags": ["target_1h_lag_48", "target_1h_lag_96",
                        "warehouse_total_load_lag_48", "status_6_diff_hourly_lag_48"],
    "target_change": ["target_acceleration"],
    "correlation_lag": CORRELATION_COLUMNS,
    "pilot_start_time": ["start_sin", "start_cos"],
    "pilot_peaks": ["peak7_mean_max", "peak7_mean_level",
                    "peak7_share_-1", "peak7_share_0", "peak7_share_1"],
}