import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('directory', nargs='?', default='artifacts/clustering/calibration_367876306')
args = parser.parse_args()
root = Path(args.directory)


def read_phase(stage, variant):
    paths = sorted(root.glob(f'{stage}_h*_{variant}.parquet'))
    if not paths:
        raise FileNotFoundError(stage + ' ' + variant)
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def fit_factors(frame, half_life, shrink_days):
    age = (frame.cutoff.max() - frame.cutoff).dt.total_seconds().to_numpy() / 86400
    weight = np.ones(len(frame)) if half_life is None else np.exp2(-age / half_life)
    weighted = frame.assign(actual=frame.y * weight, predicted=frame.y_pred * weight, weight=weight)
    global_factor = float(weighted.actual.sum() / weighted.predicted.sum())
    grouped = weighted.groupby('route_id')[['actual', 'predicted', 'weight']].sum()
    local = grouped.actual / grouped.predicted.clip(lower=1.)
    effective_days = grouped.weight / frame.h.nunique()
    blend = effective_days / (effective_days + shrink_days)
    factors = (global_factor + blend * (local - global_factor)).clip(.8, 1.25)
    denominator = float((weighted.predicted * weighted.route_id.map(factors)).sum())
    factors = (factors * weighted.actual.sum() / denominator).clip(.8, 1.25)
    return factors, float(np.clip(global_factor, .8, 1.25))


def metric(frame, factors, fallback):
    pred = frame.y_pred * frame.route_id.map(factors).fillna(fallback)
    error = pred.to_numpy() - frame.y.to_numpy()
    volume = float(frame.y.sum())
    return dict(WAPE=float(np.abs(error).sum() / volume), relative_bias=float(error.sum() / volume), score=float((np.abs(error).sum() + abs(error.sum())) / volume))


rows, choices = [], {}
for variant in ('cat_k1', 'cat_k8'):
    fit, select = [read_phase(s, variant) for s in ('calibration_fit', 'calibration_select')]
    if fit.timestamp.max() >= select.cutoff.min():
        raise ValueError('Периоды пересекаются')
    candidates = []
    for half_life in (None, 3, 7):
        for shrink_days in (1, 3, 7, 14, 1000000):
            factors, fallback = fit_factors(fit, half_life, shrink_days)
            row = dict(variant=variant, half_life=half_life, shrink_days=shrink_days, **metric(select, factors, fallback))
            candidates.append(row)
            rows.append(row)
    choices[variant] = min(candidates, key=lambda r: r['score'])
winner = min(choices, key=lambda v: choices[v]['score'])
(root / 'route_calibration_frozen.json').write_text(json.dumps(dict(winner=winner, choices=choices, note='Выбор по val до test. Test ранее изучался.'), ensure_ascii=False, indent=2))
pd.DataFrame(rows).to_csv(root / 'route_calibration_selection.csv', index=False)
reports = []
for variant, choice in choices.items():
    train = pd.concat([read_phase(s, variant) for s in ('calibration_fit', 'calibration_select')], ignore_index=True)
    factors, fallback = fit_factors(train, choice['half_life'], choice['shrink_days'])
    factors.rename('factor').to_csv(root / f'route_calibration_factors_{variant}.csv')
    for period, frame in [('train_oof', train), ('test', read_phase('test', variant))]:
        reports.append(dict(variant=variant, selected_on_val=variant == winner, period=period, **metric(frame, factors, fallback)))
pd.DataFrame(reports).to_csv(root / 'route_calibration_metrics.csv', index=False)
print(pd.DataFrame(choices).T.to_string())
print(pd.DataFrame(reports).to_string(index=False))
