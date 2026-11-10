import json
from pathlib import Path
import numpy as np
import pandas as pd

root = Path('artifacts/clustering/learned_residual_367906407')
source = Path('artifacts/clustering/calibration_367876306')
choice = json.loads((root / 'learned_residual_frozen.json').read_text())
k, config = choice['k'], choice['config']
fit, select = [pd.read_parquet(root / f'meta_oof_{stage}_k{k}_config{config}.parquet') for stage in ('calibration_fit', 'calibration_select')]


def mixed(frame, strength):
    return (1 - strength) * frame.y_pred.to_numpy() + strength * frame.meta_unshifted.to_numpy()


def calibrate(history, query, strength, mode):
    pred = mixed(query, strength)
    known = pd.concat([history, query], ignore_index=True)
    known_pred = mixed(known, strength)
    if mode == 'recent_ratio3':
        weight = np.exp2(-(history.cutoff.max() - history.cutoff).dt.total_seconds().to_numpy() / 86400 / 3)
        factor = np.clip(np.sum(history.y * weight) / np.sum(mixed(history, strength) * weight), .8, 1.25)
        return pred * factor, [dict(factor=float(factor))]
    result = np.empty(len(query))
    audit = []
    for cutoff in sorted(query.cutoff.unique()):
        cutoff = pd.Timestamp(cutoff)
        available = (known.timestamp < cutoff - pd.Timedelta(hours=4)).to_numpy()
        weight = np.exp2(-(cutoff - known.loc[available, 'cutoff']).dt.total_seconds().to_numpy() / 86400 / 7)
        actual, forecast = known.loc[available, 'y'].to_numpy(), known_pred[available]
        selected = query.cutoff.eq(cutoff).to_numpy()
        if mode == 'rolling_ratio7':
            value = float(np.clip(np.sum(actual * weight) / np.sum(forecast * weight), .8, 1.25))
            result[selected] = pred[selected] * value
        else:
            value = float(np.average(actual - forecast, weights=weight))
            result[selected] = np.maximum(pred[selected] + value, 0)
        audit.append(dict(cutoff=str(cutoff), fit_through=str(known.loc[available, 'timestamp'].max()), value=value))
    return result, audit


def metric(frame, pred):
    error = pred - frame.y.to_numpy()
    volume = float(frame.y.sum())
    return dict(WAPE=float(np.abs(error).sum() / volume), relative_bias=float(error.sum() / volume), score=float((np.abs(error).sum() + abs(error.sum())) / volume))


rows = []
for strength in (0., .25, .5, .75, 1.):
    for mode in ('recent_ratio3', 'rolling_offset7', 'rolling_ratio7'):
        prediction, audit = calibrate(fit, select, strength, mode)
        rows.append(dict(strength=strength, mode=mode, **metric(select, prediction)))
selected = min(rows, key=lambda r: r['score'])
(root / 'strength_frozen.json').write_text(json.dumps(selected, indent=2))
pd.DataFrame(rows).to_csv(root / 'strength_selection.csv', index=False)
test = pd.read_parquet(root / 'learned_residual_test_predictions.parquet')
test['timestamp'] = test.cutoff + pd.to_timedelta(test.h * 30, unit='m')
all_val = pd.concat([fit, select], ignore_index=True)
prediction, audit = calibrate(all_val, test, selected['strength'], selected['mode'])
report = dict(period='test', protocol='frozen' if selected['mode'] == 'recent_ratio3' else 'prequential') | selected | metric(test, prediction)
(root / 'strength_test.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
pd.DataFrame(audit).to_csv(root / 'strength_test_audit.csv', index=False)
test.assign(strength_prediction=prediction).to_parquet(root / 'strength_test_predictions.parquet', index=False)
print('Val selection:', selected)
print('Test before final refit:', report)
