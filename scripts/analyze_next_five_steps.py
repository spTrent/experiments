import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/next_five_steps'
OUT.mkdir(parents=True, exist_ok=True)
frame = pd.read_parquet(ROOT / 'best/validation_predictions.parquet').rename(columns={'session_prediction': 'prediction'})
frame['y'] = frame.y.astype(np.float64)
frame['prediction'] = frame.prediction.astype(np.float64)
frame['error'] = frame.prediction - frame.y
frame['absolute'] = frame.error.abs()
horizons = frame.groupby('horizon').agg(actual=('y', 'sum'), predicted=('prediction', 'sum'), absolute=('absolute', 'sum'), signed=('error', 'sum'))
horizons['bias_pct'] = 100 * horizons.signed / horizons.actual
horizons['WAPE'] = horizons.absolute / horizons.actual
horizons['global_WAPE_contribution'] = horizons.absolute / frame.y.sum()
horizons.to_csv(OUT / 'baseline_horizon_diagnostics.csv')
groups = frame.groupby(['cutoff', 'horizon']).agg(actual=('y', 'sum'), predicted=('prediction', 'sum'))
groups['error'] = groups.predicted - groups.actual
floor = float(groups.error.abs().sum() / frame.y.sum())
daily = frame.groupby('cutoff').agg(actual=('y', 'sum'), predicted=('prediction', 'sum'), absolute=('absolute', 'sum'))
daily['bias_pct'] = 100 * (daily.predicted - daily.actual) / daily.actual
daily['WAPE'] = daily.absolute / daily.actual
daily['score'] = daily.WAPE + daily.bias_pct.abs() / 100
daily.to_csv(OUT / 'baseline_daily_diagnostics.csv')
route = frame.groupby('route_id').agg(actual=('y', 'sum'), signed=('error', 'sum'), absolute=('absolute', 'sum'))
route['bias'] = route.signed / route.actual
route.to_csv(OUT / 'baseline_route_diagnostics.csv')
weeks = frame.assign(week=np.where(frame.cutoff < '2025-10-25', 'week1', 'week2')).groupby(['route_id', 'week']).agg(actual=('y', 'sum'), signed=('error', 'sum')).reset_index()
weeks['bias'] = weeks.signed / weeks.actual
pivot = weeks.pivot(index='route_id', columns='week', values='bias')
summary = {'source': 'best/validation_predictions.parquet', 'scope': 'descriptive already-seen holdout 18-31 Oct; no predictive training or candidate calibration fit', 'WAPE': float(frame.absolute.sum() / frame.y.sum()), 'pooled_score': float((frame.absolute.sum() + abs(frame.error.sum())) / frame.y.sum()), 'pooled_bias_pct': float(frame.error.sum() / frame.y.sum() * 100), 'mean_absolute_daily_bias_pct': float(daily.bias_pct.abs().mean()), 'max_absolute_daily_bias_pct': float(daily.bias_pct.abs().max()), 'WAPE_floor_given_each_cutoff_horizon_predicted_total': floor, 'observed_WAPE_minus_fixed_total_floor': float(frame.absolute.sum() / frame.y.sum() - floor), 'floor_note': 'Triangle inequality lower bound using actual test targets; diagnostic only, unavailable at inference; not a claimed achievable score', 'week1_week2_route_bias_correlation': float(pivot.week1.corr(pivot.week2)), 'route_bias_same_sign_weeks_pct': float(((pivot.week1 > 0) == (pivot.week2 > 0)).mean() * 100), 'route_bias_abs_median_pct': float(route.bias.abs().median() * 100)}
reference = json.loads((ROOT / 'best/bundle/result.json').read_text())
if not np.isclose(summary['pooled_score'], reference['score'], rtol=0, atol=1e-12):
    raise ValueError('Baseline metrics do not reproduce')
(OUT / 'evidence.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
notebook = json.loads((ROOT / 'best/raw-vs-clean.ipynb').read_text())
namespace = {'np': np, 'pd': pd}
definitions = [node for node in ast.parse(''.join(notebook['cells'][3]['source'])).body if isinstance(node, ast.FunctionDef)]
exec(compile(ast.Module(body=definitions, type_ignores=[]), '<cleaners>', 'exec'), namespace)
history = pd.read_parquet(ROOT / 'best/data/train_solo_track.parquet', columns=['route_id', 'timestamp', 'target_1h']).sort_values(['timestamp', 'route_id'])
times = pd.DatetimeIndex(history.timestamp.unique())
y = history.target_1h.to_numpy(dtype=np.float32).reshape(len(times), -1)
rows = []
for branch in ['median6', 'seasonal_mad']:
    clean = namespace['clean_series'](y, branch)
    for label, start, end in [('correction_fit', '2025-10-04', '2025-10-18'), ('test_diagnostic', '2025-10-18', '2025-11-01')]:
        origins = np.flatnonzero((times >= start) & (times < end) & (times.hour == 10) & (times.minute == 30))
        indices = (origins[:, None] + np.arange(1, 9)[None, :]).ravel()
        raw_labels = y[indices].astype(np.float64)
        clean_labels = clean[indices].astype(np.float64)
        rows.append({'branch': branch, 'period': label, 'rows': int(raw_labels.size), 'changed': int(np.count_nonzero(raw_labels != clean_labels)), 'raw_total': float(raw_labels.sum()), 'clean_total': float(clean_labels.sum()), 'target_volume_removed_pct': float(100 * (raw_labels - clean_labels).sum() / raw_labels.sum())})
pd.DataFrame(rows).to_csv(OUT / 'calibration_label_volume_shift.csv', index=False)
print(json.dumps(summary, ensure_ascii=False, indent=2))
print(pd.DataFrame(rows).to_string(index=False))
