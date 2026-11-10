import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'best/full_pipeline_cleaning_0326'
OUT = ROOT / 'artifacts/best_cleaning_review'
OUT.mkdir(parents=True, exist_ok=True)
KEYS = ['cutoff', 'horizon', 'route_id']
raw = pd.read_parquet(ROOT / 'best/validation_predictions.parquet').rename(columns={'session_prediction': 'prediction'})
raw = raw[KEYS + ['y', 'prediction']].copy()
raw['branch'] = 'raw'
raw['split'] = 'test'
frames = [raw]
for branch in ['median6', 'seasonal_mad']:
    frame = pd.read_parquet(SOURCE / f'predictions_{branch}.parquet')
    frames.append(frame.loc[frame.split.eq('test'), KEYS + ['y', 'prediction', 'branch', 'split']].copy())
panel = pd.concat(frames, ignore_index=True)


def metric(frame):
    y = frame.y.to_numpy(dtype=np.float64)
    p = frame.prediction.to_numpy(dtype=np.float64)
    error = p - y
    return {'rows': len(frame), 'actual_total': float(y.sum()), 'prediction_total': float(p.sum()), 'WAPE': float(np.abs(error).sum() / y.sum()), 'relative_bias': float(error.sum() / y.sum()), 'abs_bias': float(abs(error.sum()) / y.sum()), 'score': float((np.abs(error).sum() + abs(error.sum())) / y.sum()), 'MAE': float(np.abs(error).mean())}


summary_rows = []
by_day = []
by_horizon = []
paired_stats = []
subgroups = []
source_labels = pd.read_parquet(ROOT / 'best/data/train_solo_track.parquet', columns=['route_id', 'timestamp', 'target_1h']).set_index(['timestamp', 'route_id']).target_1h
for branch, part in panel.groupby('branch', sort=False):
    if part.duplicated(KEYS).any() or len(part) != 112000:
        raise ValueError('Bad prediction keys')
    timestamps = part.cutoff + pd.to_timedelta(part.horizon * 30, unit='min')
    truth = source_labels.reindex(pd.MultiIndex.from_arrays([timestamps, part.route_id])).to_numpy()
    if not np.array_equal(truth, part.y.to_numpy()):
        raise ValueError('Wrong or cleaned evaluation labels')
    days = []
    for cutoff, block in part.groupby('cutoff'):
        item = {'branch': branch, 'cutoff': str(cutoff), **metric(block)}
        by_day.append(item)
        days.append(item['score'])
    summary_rows.append({'branch': branch, **metric(part), 'mean_daily_score': float(np.mean(days))})
    for horizon, block in part.groupby('horizon'):
        by_horizon.append({'branch': branch, 'horizon': int(horizon), **metric(block)})
    if branch == 'raw':
        continue
    paired = raw.merge(part, on=KEYS, how='outer', suffixes=('_raw', '_clean'), indicator=True, validate='one_to_one')
    if not paired['_merge'].eq('both').all() or not np.array_equal(paired.y_raw, paired.y_clean):
        raise ValueError('Incomparable evaluation panels')
    difference = paired.prediction_clean - paired.prediction_raw
    raw_error = np.abs(paired.prediction_raw - paired.y_raw)
    clean_error = np.abs(paired.prediction_clean - paired.y_raw)
    paired_stats.append({'branch': branch, 'mean_prediction_change': float(difference.mean()), 'mean_absolute_prediction_change': float(difference.abs().mean()), 'median_absolute_prediction_change': float(difference.abs().median()), 'p95_absolute_prediction_change': float(difference.abs().quantile(.95)), 'route_rows_with_smaller_absolute_error_pct': float((clean_error < raw_error).mean() * 100), 'prediction_change_pct_of_mean_target': float(difference.abs().mean() / paired.y_raw.mean() * 100)})
    q95 = paired.y_raw.quantile(.95)
    for name, mask in [('zero_target', paired.y_raw.eq(0)), ('positive_below_p95', paired.y_raw.gt(0) & paired.y_raw.lt(q95)), ('target_top5pct', paired.y_raw.ge(q95))]:
        subgroups.append({'branch': branch, 'group': name, 'rows': int(mask.sum()), 'delta_WAPE_contribution': float((clean_error[mask].sum() - raw_error[mask].sum()) / paired.y_raw.sum()), 'mean_prediction_change': float(difference[mask].mean()), 'mean_absolute_error_change': float(clean_error[mask].mean() - raw_error[mask].mean())})
    paired.drop(columns='_merge').to_parquet(OUT / f'{branch}_paired_test.parquet', index=False)
summary = pd.DataFrame(summary_rows).set_index('branch')
recorded = pd.read_csv(SOURCE / 'pipeline_scores.csv').query("split == 'test'").set_index('branch')
for branch in summary.index:
    for column in ['WAPE', 'relative_bias', 'score', 'mean_daily_score']:
        if not np.isclose(summary.loc[branch, column], recorded.loc[branch, column], rtol=0, atol=1e-12):
            raise ValueError('Saved metric mismatch')
for column in ['WAPE', 'abs_bias', 'score', 'mean_daily_score']:
    summary['delta_' + column + '_vs_raw'] = summary[column] - summary.loc['raw', column]
summary.to_csv(OUT / 'verified_test_summary.csv')
daily = pd.DataFrame(by_day)
horizons = pd.DataFrame(by_horizon)
counts = pd.read_csv(SOURCE / 'cleaning_counts.csv')
counts['changed_pct'] = counts.changed / counts.rows * 100
counts.to_csv(OUT / 'cleaning_percentages.csv', index=False)
raw_day = daily.loc[daily.branch.eq('raw')].set_index('cutoff')
raw_horizon = horizons.loc[horizons.branch.eq('raw')].set_index('horizon')
daily['delta_score_vs_raw'] = daily.score - daily.cutoff.map(raw_day.score)
daily['delta_WAPE_vs_raw'] = daily.WAPE - daily.cutoff.map(raw_day.WAPE)
horizons['delta_score_vs_raw'] = horizons.score - horizons.horizon.map(raw_horizon.score)
horizons['delta_WAPE_vs_raw'] = horizons.WAPE - horizons.horizon.map(raw_horizon.WAPE)
daily.to_csv(OUT / 'daily_comparison.csv', index=False)
horizons.to_csv(OUT / 'horizon_comparison.csv', index=False)
pd.DataFrame(paired_stats).to_csv(OUT / 'paired_prediction_changes.csv', index=False)
pd.DataFrame(subgroups).to_csv(OUT / 'target_group_comparison.csv', index=False)
by_week = []
for branch in ['median6', 'seasonal_mad']:
    frame = daily.loc[daily.branch.eq(branch)]
    row = {'branch': branch, 'days_improved': int(frame.delta_score_vs_raw.lt(0).sum()), 'days_worsened': int(frame.delta_score_vs_raw.gt(0).sum()), 'first_week_mean_daily_delta': float(frame.loc[frame.cutoff < '2025-10-25', 'delta_score_vs_raw'].mean()), 'second_week_mean_daily_delta': float(frame.loc[frame.cutoff >= '2025-10-25', 'delta_score_vs_raw'].mean())}
    by_week.append(row)
pd.DataFrame(by_week).to_csv(OUT / 'day_win_summary.csv', index=False)
notebook = json.loads((ROOT / 'best/raw-vs-clean.ipynb').read_text())
errors = [output.get('evalue') for cell in notebook['cells'] for output in cell.get('outputs', []) if output.get('output_type') == 'error']
if errors:
    raise ValueError(errors)
metadata = {'source_notebook': 'best/raw-vs-clean.ipynb', 'models': 'CatBoost plus LightGBM plus residual CatBoost; not XGBoost', 'run_minutes': notebook['metadata']['papermill']['duration'] / 60, 'test_start': str(panel.cutoff.min()), 'test_end': str(panel.cutoff.max()), 'branches': ['raw', 'median6', 'seasonal_mad'], 'verified_same_keys_and_raw_targets': True, 'recomputed_metrics_match_csv_atol': 1e-12, 'raw_control_retrained_in_this_run': False, 'cleaning': 'upper clipping only; zeros retained', 'selected_on_tuning': json.loads((SOURCE / 'manifest.json').read_text())['selected_on_tuning'], 'models_trained_locally': False}
(OUT / 'verification.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
print('VERIFIED TEST SUMMARY')
print(summary.to_string())
print('DAY WINS')
print(pd.DataFrame(by_week).to_string(index=False))
print('PAIRED CHANGES')
print(pd.DataFrame(paired_stats).to_string(index=False))
print('TARGET GROUPS')
print(pd.DataFrame(subgroups).to_string(index=False))
print('HORIZON DELTAS')
print(horizons.pivot(index='horizon', columns='branch', values='delta_score_vs_raw').to_string())
