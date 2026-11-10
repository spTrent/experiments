import ast
import gc
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/xgb_cleaning_audit/threshold_sweep'
OUT.mkdir(parents=True, exist_ok=True)
TEMP = OUT / 'arrays'
TEMP.mkdir(exist_ok=True)
THRESHOLDS = [3, 4, 5, 6]
FIELDS = ['target_1h'] + [f'status_{i}' for i in range(1, 7)]
BLOCK_ROUTES = 50
started = time.monotonic()
notebook = json.loads((ROOT / 'notebooks/xgb_tuning/xgb_cleaning.ipynb').read_text())
for cell in notebook['cells']:
    if cell['cell_type'] != 'code':
        continue
    tree = ast.parse(''.join(cell['source']))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [item.id for item in node.targets if isinstance(item, ast.Name)]
        if 'FEATURE_CODE' in names:
            FEATURE_CODE = ast.literal_eval(node.value)
        if 'CLEANING' in names:
            CLEANING = ast.literal_eval(node.value)
history = pd.read_parquet(ROOT / 'data/train_solo_track.parquet').sort_values(['timestamp', 'route_id'])
times = pd.DatetimeIndex(history.timestamp.unique())
routes = np.sort(history.route_id.unique())
shape = (len(times), len(routes))
raw = {}
cleaned = {threshold: {} for threshold in THRESHOLDS}
raw_rows = []
slots = times.hour * 2 + times.minute // 30
for field in FIELDS:
    raw[field] = np.lib.format.open_memmap(TEMP / f'raw_{field}.npy', mode='w+', dtype='float32', shape=shape)
    raw[field][:] = history[field].to_numpy(dtype=np.float32).reshape(shape)
    for threshold in THRESHOLDS:
        cleaned[threshold][field] = np.lib.format.open_memmap(TEMP / f'z{threshold}_{field}.npy', mode='w+', dtype='float32', shape=shape)
        cleaned[threshold][field][:] = raw[field]
    replaced = {threshold: 0 for threshold in THRESHOLDS}
    zero_replaced = {threshold: 0 for threshold in THRESHOLDS}
    for slot in range(48):
        positions = np.flatnonzero(slots == slot)
        logged = np.log1p(raw[field][positions].astype(np.float64))
        padded = np.pad(logged, ((CLEANING['window_days'], 0), (0, 0)), constant_values=np.nan)
        windows = np.lib.stride_tricks.sliding_window_view(padded, CLEANING['window_days'], axis=0)[:-1]
        eligible = np.arange(len(positions)) >= CLEANING['min_days']
        past = windows[eligible]
        median = np.nanmedian(past, axis=-1)
        mad = np.nanmedian(np.abs(past - median[..., None]), axis=-1)
        scale = np.maximum(CLEANING['mad_normalization'] * mad, CLEANING['log_scale_floor'])
        robust_z = np.abs(logged[eligible] - median) / scale
        destination = positions[eligible]
        replacement = np.expm1(median).astype(np.float32)
        original = raw[field][destination]
        for threshold in THRESHOLDS:
            flags = robust_z > threshold
            cleaned[threshold][field][destination] = np.where(flags, replacement, original)
            replaced[threshold] += int(flags.sum())
            zero_replaced[threshold] += int((flags & (original == 0)).sum())
    for threshold in THRESHOLDS:
        cleaned[threshold][field].flush()
        raw_rows.append({'threshold': threshold, 'field': field, 'raw_values': int(np.prod(shape)), 'replaced': replaced[threshold], 'raw_replaced_pct': 100 * replaced[threshold] / np.prod(shape), 'zero_replaced': zero_replaced[threshold]})
    raw[field].flush()
    print('raw', field, replaced, 'elapsed', round(time.monotonic() - started), flush=True)
del history, past, windows, logged, padded, original, replacement, robust_z
raw_report = pd.DataFrame(raw_rows)
raw_report.to_csv(OUT / 'raw_replacements.csv', index=False)
reference = pd.read_csv(ROOT / 'artifacts/xgb_cleaning_audit/raw_cleaning_mad_summary.csv').set_index('column')
for row in raw_report.loc[raw_report.threshold.eq(4)].itertuples():
    if row.replaced != reference.loc[row.field, 'replaced']:
        raise ValueError(f'Notebook cleaner mismatch: {row.field}')
base_ns = {'np': np, 'pd': pd, 'STEP': pd.Timedelta(minutes=30)}
parsed = ast.parse(FEATURE_CODE)
lag_definition = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == 'calculate_dynamic_shoulder')
exec(compile(ast.Module(body=[lag_definition], type_ignores=[]), '<lag>', 'exec'), base_ns)
dates = times[(times.hour == 10) & (times.minute == 30) & (times + pd.Timedelta(hours=4) <= times[-1])]
val_start = dates[int(np.floor((len(dates) - 7) * .8))].normalize()
states = {0: raw, **cleaned}
lag_tables = {}
network = {}
for threshold, values in states.items():
    signal = values['status_6'].copy()
    signal[2:] -= values['status_6'][:-2]
    signal[:2] = np.nan
    lag_tables[threshold] = base_ns['calculate_dynamic_shoulder'](values['target_1h'], signal, times, routes, fit_before=val_start)
    network[threshold] = {name: values[name].sum(axis=1, dtype=np.float64).astype(np.float32)[:, None] for name in ['target_1h', 'status_1', 'status_2', 'status_3']}
    print('lag fit', threshold, 'elapsed', round(time.monotonic() - started), flush=True)
del signal


def feature_family(name):
    if name.startswith('selected_status6_') or name.startswith('lagged_status6_') or name == 'status6_lag_valid':
        return 'correlation_target_status6'
    for field in FIELDS:
        if name.startswith(field + '_') or name.startswith('network_' + field + '_'):
            return field
    if name.startswith(('target_', 'slot_')) or name in ['profile28', 'morning_level_ratio']:
        return 'target_1h'
    if name.startswith(('warehouse_', 'hub_')):
        return 'mixed_statuses'
    return 'calendar'


counts = defaultdict(lambda: [0, 0, 0, 0])
feature_counts = defaultdict(lambda: [0, 0])
family_columns = None
for route_start in range(0, len(routes), BLOCK_ROUTES):
    route_end = min(route_start + BLOCK_ROUTES, len(routes))
    block_routes = routes[route_start:route_end]
    contexts = {}
    for threshold, values in states.items():
        context = {'np': np, 'pd': pd, 'STEP': pd.Timedelta(minutes=30), 'TARGET': 'target_1h', 'STATUSES': FIELDS[1:], 'times': times, 'routes': block_routes, 'values': {name: values[name][:, route_start:route_end] for name in FIELDS}}
        exec(FEATURE_CODE, context)
        context['validation_lag_table'] = lag_tables[threshold].iloc[route_start:route_end]
        context['network_values'] = network[threshold]
        contexts[threshold] = context
    for h in range(1, 9):
        origins = np.arange(len(times) - h)
        baseline = contexts[0]['horizon_dataset'](h, origins)
        columns = baseline.columns.drop(['cutoff', 'timestamp', 'route_id', 'overlap']).tolist()
        if family_columns is None:
            family_columns = defaultdict(list)
            for name in columns:
                family_columns[feature_family(name)].append(name)
            (OUT / 'feature_families.json').write_text(json.dumps(dict(family_columns), ensure_ascii=False, indent=2) + '\n')
        baseline_values = baseline[columns].to_numpy(dtype=np.float32)
        minutes = times[origins].hour * 60 + times[origins].minute
        nine_mask = np.repeat(np.isin(minutes, list(range(510, 751, 30))), len(block_routes))
        column_indices = {name: index for index, name in enumerate(columns)}
        groups = {family: [column_indices[name] for name in names] for family, names in family_columns.items()}
        groups['all_model_features'] = list(range(len(columns)))
        for threshold in THRESHOLDS:
            frame = contexts[threshold]['horizon_dataset'](h, origins)
            candidate = frame[columns].to_numpy(dtype=np.float32)
            both_nan = np.isnan(baseline_values) & np.isnan(candidate)
            available = ~both_nan
            changed = (baseline_values != candidate) & available
            for variant, mask in [('all_halfhours', None), ('nine_starts', nine_mask)]:
                for family, indices in groups.items():
                    difference = changed[:, indices]
                    valid = available[:, indices]
                    if mask is not None:
                        difference = difference[mask]
                        valid = valid[mask]
                    counter = counts[(threshold, variant, family)]
                    counter[0] += int(difference.sum())
                    counter[1] += int(valid.sum())
                    counter[2] += int(difference.any(axis=1).sum())
                    counter[3] += int(len(difference))
                feature_diff = changed if mask is None else changed[mask]
                feature_valid = available if mask is None else available[mask]
                sums = feature_diff.sum(axis=0)
                totals = feature_valid.sum(axis=0)
                for index, name in enumerate(columns):
                    counter = feature_counts[(threshold, variant, name)]
                    counter[0] += int(sums[index])
                    counter[1] += int(totals[index])
            del frame, candidate, both_nan, available, changed
        del baseline, baseline_values
    contexts.clear()
    gc.collect()
    print('routes processed', route_end, 'of', len(routes), 'elapsed', round(time.monotonic() - started), flush=True)
    rows = [{'threshold': threshold, 'variant': variant, 'family': family, 'changed_cells': values[0], 'available_cells': values[1], 'changed_rows': values[2], 'rows': values[3], 'feature_cells_changed_pct': 100 * values[0] / values[1] if values[1] else 0, 'rows_with_change_pct': 100 * values[2] / values[3] if values[3] else 0} for (threshold, variant, family), values in counts.items()]
    pd.DataFrame(rows).to_csv(OUT / 'feature_changes.csv', index=False)
feature_rows = [{'threshold': threshold, 'variant': variant, 'feature': feature, 'family': feature_family(feature), 'changed_cells': values[0], 'available_cells': values[1], 'changed_pct': 100 * values[0] / values[1] if values[1] else 0} for (threshold, variant, feature), values in feature_counts.items()]
pd.DataFrame(feature_rows).to_csv(OUT / 'individual_feature_changes.csv', index=False)
features = pd.DataFrame(rows)
summary = raw_report[['threshold', 'field', 'raw_replaced_pct']].copy()
for variant in ['all_halfhours', 'nine_starts']:
    subset = features.loc[features.variant.eq(variant) & features.family.isin(FIELDS), ['threshold', 'family', 'feature_cells_changed_pct', 'rows_with_change_pct']]
    subset = subset.rename(columns={'family': 'field', 'feature_cells_changed_pct': f'{variant}_changed_cells_pct', 'rows_with_change_pct': f'{variant}_changed_rows_pct'})
    summary = summary.merge(subset, on=['threshold', 'field'], validate='one_to_one')
summary.to_csv(OUT / 'comparison.csv', index=False)
metadata = {'thresholds': THRESHOLDS, 'settings_except_threshold': CLEANING, 'source_rows': int(np.prod(shape)), 'route_count': len(routes), 'timestamp_count': len(times), 'horizons': list(range(1, 9)), 'correlation_fit_before': str(val_start), 'comparison': 'exact float32 inequality; two NaNs equal; denominator excludes two NaNs', 'all_halfhours': 'all available labeled origin/horizon pairs', 'nine_starts': 'same history, origins 08:30 through 12:30 inclusive', 'elapsed_seconds': time.monotonic() - started}
(OUT / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
print(summary.to_string(index=False), flush=True)
print('DONE', round(time.monotonic() - started), flush=True)

for path in TEMP.glob('*.npy'):
    path.unlink()
TEMP.rmdir()
