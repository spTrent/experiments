import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'Report/transfer_audit'
OUT.mkdir(parents=True, exist_ok=True)
ARCHIVE = Path('/Users/sptrent/Downloads/best_blend_bf2ef3a96ffc')


def stats(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if not np.isfinite(y).all() or not np.isfinite(p).all() or y.sum() <= 0:
        raise ValueError('Invalid predictions or targets')
    wape = np.abs(p - y).sum() / y.sum()
    bias = (p - y).sum() / y.sum()
    return dict(WAPE=wape, signed_bias=bias, score=wape + abs(bias))


def audit_predictions():
    keys = ['cutoff', 'route_id', 'horizon']
    frame = pd.read_parquet(ARCHIVE / 'test_predictions.parquet', columns=keys + [
        'y', 'cat_d4__fixed', 'cat_d4__ridge', 'session6'])
    cat = pd.concat([pd.read_parquet(ROOT / f'artifacts/clustering/catboost_c3345bacb1fc3146/test_h{h}_cat_k8.parquet') for h in range(1, 9)])
    cat = cat.rename(columns={'h': 'horizon', 'y': 'cat_y'})
    frame = frame.merge(cat[keys + ['cat_y', 'y_calibrated']], on=keys, validate='one_to_one')
    if len(frame) != 112000 or not np.array_equal(frame.y, frame.cat_y):
        raise ValueError('Archived predictions do not align')
    groups = frame.groupby(['cutoff', 'horizon'])
    frame['cat_volume_same_shape'] = frame.cat_d4__fixed / groups.cat_d4__fixed.transform('sum') * groups.y_calibrated.transform('sum')
    variants = ['session6', 'cat_d4__fixed', 'cat_d4__ridge', 'cat_volume_same_shape']
    daily = pd.DataFrame([dict(date=str(day.date()), weekday=day.day_name(), variant=v, actual=float(f.y.to_numpy(dtype=np.float64).sum()), **stats(f.y, f[v])) for day, f in frame.groupby('cutoff') for v in variants])
    daily.to_csv(OUT / 'daily_metrics.csv', index=False)
    horizon_rows = []
    for horizon, part in frame.groupby('horizon'):
        for v in variants:
            horizon_rows.append(dict(horizon=int(horizon), variant=v, **stats(part.y, part[v])))
    pd.DataFrame(horizon_rows).to_csv(OUT / 'horizon_metrics.csv', index=False)
    rows = []
    for v in variants:
        d = daily.loc[daily.variant.eq(v)]
        weighted_abs_bias = np.average(np.abs(d.signed_bias), weights=d.actual)
        s = stats(frame.y, frame[v])
        rows.append(dict(variant=v, **s, daily_mean=d.score.mean(), daily_median=d.score.median(), daily_p90=d.score.quantile(.9), daily_worst=d.score.max(), weighted_daily_score=np.average(d.score, weights=d.actual), cancellation_gap=weighted_abs_bias-abs(s['signed_bias'])))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / 'period_vs_daily.csv', index=False)
    pivot = daily.pivot(index='date', columns='variant', values='score')
    delta = (pivot.cat_volume_same_shape - pivot.cat_d4__fixed).to_numpy()
    rng = np.random.default_rng(20261112)
    draws = rng.integers(0, len(delta), size=(10000, len(delta)))
    iid = np.mean(delta[draws], axis=1)
    starts = rng.integers(0, len(delta), size=(10000, 7))
    blocks = np.stack([starts, (starts+1) % len(delta)], axis=-1).reshape(10000, -1)
    boot = np.mean(delta[blocks], axis=1)
    result = dict(cat_wins_days=int((delta < 0).sum()), days=len(delta), mean_daily_delta=float(delta.mean()), exploratory_iid_interval_95=np.quantile(iid, [.025, .975]).tolist(), exploratory_circular_2day_interval_95=np.quantile(boot, [.025, .975]).tolist())
    (OUT / 'daily_comparison.json').write_text(json.dumps(result, indent=2))
    print(summary.to_string(index=False))
    print(json.dumps(result))
    print(daily[daily.weekday.eq('Saturday')].to_string(index=False))


def audit_morning():
    path = ROOT / 'best/full_pipeline_cleaning_0326/inputs'
    request = pd.read_parquet(path / 'test_solo_track.parquet')
    history = pd.read_parquet(path / 'train_solo_track.parquet')
    columns = ['target_1h'] + [f'status_{i}' for i in range(1, 7)]
    network = history.groupby('timestamp')[columns].sum().astype(np.float64)
    network.to_csv(OUT / 'network_history.csv')
    cutoff = network.index.max()
    morning = network.loc[(network.index.hour >= 6) & ((network.index.hour < 10) | ((network.index.hour == 10) & (network.index.minute <= 30)))].copy()
    morning['day'] = morning.index.normalize()
    morning['slot'] = morning.index.hour*2 + morning.index.minute//30
    profiles = morning.pivot(index='day', columns='slot', values=columns)
    log_profiles = np.log1p(profiles)
    reference = log_profiles.loc[(log_profiles.index < cutoff.normalize()) & (log_profiles.index >= cutoff.normalize()-pd.Timedelta(days=28))]
    scale = reference.std().clip(lower=.05)
    distances = (((reference-log_profiles.loc[cutoff.normalize()])/scale)**2).mean(axis=1)**.5
    similarity = pd.DataFrame({'date': distances.index, 'distance': distances.values})
    similarity['weekday'] = similarity.date.dt.day_name()
    similarity.sort_values('distance').to_csv(OUT / 'morning_neighbours.csv', index=False)
    ratios = []
    now = network.loc[cutoff]
    recent = network.loc[(network.index.hour == cutoff.hour) & (network.index.minute == cutoff.minute) & (network.index < cutoff) & (network.index >= cutoff-pd.Timedelta(days=28))]
    for c in columns:
        ratios.append(dict(feature=c, current=float(now[c]), past28_median=float(recent[c].median()), ratio=float(now[c]/recent[c].median()), percentile=float((recent[c]<=now[c]).mean())))
    pd.DataFrame(ratios).to_csv(OUT / 'current_level_shift.csv', index=False)
    morning_totals = morning.groupby('day')[columns].sum()
    earlier_mornings = morning_totals.loc[(morning_totals.index < cutoff.normalize()) & (morning_totals.index >= cutoff.normalize()-pd.Timedelta(days=28))]
    morning_ratios = pd.DataFrame({'feature': columns, 'ratio_to_past28_median': (morning_totals.loc[cutoff.normalize()]/earlier_mornings.median()).to_numpy()})
    morning_ratios.to_csv(OUT / 'morning_level_shift.csv', index=False)
    target = history.pivot(index='timestamp', columns='route_id', values='target_1h').sort_index().to_numpy(dtype=np.float64)[-48:]
    low = np.zeros(target.shape[1])
    high = target[0].copy()
    feasible = np.ones(target.shape[1], dtype=bool)
    for values in target[1:]:
        next_low = np.maximum(0, values-high)
        next_high = np.minimum(values, values-low)
        feasible &= next_low <= next_high + 1e-6
        low, high = next_low, next_high
    feasibility = dict(routes=int(target.shape[1]), feasible_last48=int(feasible.sum()), inconsistent_last48=int((~feasible).sum()), interpretation='Existence of nonnegative half-hour flows whose adjacent sums equal observed hourly targets; diagnostic only')
    (OUT / 'rolling_sum_feasibility.json').write_text(json.dumps(feasibility, indent=2))
    metadata = dict(history_rows=len(history), history_start=str(history.timestamp.min()), cutoff=str(cutoff), request_rows=len(request), request_dates=[str(x) for x in sorted(request.timestamp.dt.date.unique())], request_times=[str(x) for x in sorted(request.timestamp.unique())], request_routes=int(request.route_id.nunique()))
    (OUT / 'data_scope.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata))
    print(similarity.sort_values('distance').head(8).to_string(index=False))
    print(pd.DataFrame(ratios).to_string(index=False))


if __name__ == '__main__':
    audit_predictions()
    audit_morning()
