import json
from pathlib import Path
import numpy as np
import pandas as pd

roots = dict(cat=Path('artifacts/clustering/calibration_367876306'), xgb=Path('artifacts/clustering/xgb_367884401'))
legacy = Path('notebooks/stepa/stepa_checkpoint_0328_bundle/inputs')
profiles = pd.read_parquet(legacy / 'selected_evaluation.parquet')[['route_id', 'cutoff', 'horizon', 'profile28', 'y']].rename(columns={'horizon': 'h', 'y': 'profile_y'})
out = Path('artifacts/clustering/profile_blend')
out.mkdir(exist_ok=True)


def read_phase(algorithm, stage):
    frame = pd.concat([pd.read_parquet(p) for p in sorted(roots[algorithm].glob(f'{stage}_h*_cat_k8.parquet'))], ignore_index=True)
    result = frame.merge(profiles, on=['route_id', 'cutoff', 'h'], validate='one_to_one')
    if len(result) != len(frame) or not np.array_equal(result.y, result.profile_y):
        raise ValueError('Несовпадающий причинный профиль')
    return result


def pred(frame, weight):
    return weight * frame.y_pred.to_numpy() + (1 - weight) * frame.profile28.to_numpy()


def factor(frame, weight, half_life):
    age = (frame.cutoff.max() - frame.cutoff).dt.total_seconds().to_numpy() / 86400
    recency = np.ones(len(frame)) if half_life is None else np.exp2(-age / half_life)
    return float(np.clip(np.sum(frame.y * recency) / np.sum(pred(frame, weight) * recency), .8, 1.25))


def metric(frame, values):
    error = values - frame.y.to_numpy()
    volume = float(frame.y.sum())
    return dict(WAPE=float(np.abs(error).sum() / volume), relative_bias=float(error.sum() / volume), score=float((np.abs(error).sum() + abs(error.sum())) / volume))


rows = []
for algorithm in roots:
    fit, select = [read_phase(algorithm, stage) for stage in ('calibration_fit', 'calibration_select')]
    for weight in (0., .25, .5, .75, 1.):
        for half_life in (None, 2, 3, 7):
            coefficient = factor(fit, weight, half_life)
            rows.append(dict(algorithm=algorithm, model_weight=weight, half_life=half_life, **metric(select, pred(select, weight) * coefficient)))
choice = min(rows, key=lambda row: row['score'])
(out / 'frozen.json').write_text(json.dumps(choice, indent=2))
pd.DataFrame(rows).to_csv(out / 'selection.csv', index=False)
algorithm, weight, half_life = choice['algorithm'], choice['model_weight'], choice['half_life']
all_val = pd.concat([read_phase(algorithm, stage) for stage in ('calibration_fit', 'calibration_select')], ignore_index=True)
coefficient = factor(all_val, weight, half_life)
test = read_phase(algorithm, 'test')
reports = [dict(period=period, factor=coefficient, **metric(frame, pred(frame, weight) * coefficient)) for period, frame in [('train_oof', all_val), ('test', test)]]
pd.DataFrame(reports).to_csv(out / 'independent_metrics.csv', index=False)
full_factor = factor(pd.concat([all_val, test], ignore_index=True), weight, half_life)
submission = pd.read_csv(roots[algorithm] / 'submission_raw.csv').merge(pd.read_parquet(legacy / 'final_predictions.parquet')[['id', 'profile28']], on='id', validate='one_to_one', sort=False)
submission.y_pred = (weight * submission.y_pred + (1 - weight) * submission.profile28) * full_factor
if len(submission) != 8000 or not np.isfinite(submission.y_pred).all() or (submission.y_pred < 0).any():
    raise ValueError('Некорректный сабмит')
submission[['id', 'y_pred']].to_csv(out / 'submission.csv', index=False)
(out / 'final_metadata.json').write_text(json.dumps(dict(choice=choice, final_factor=full_factor, note='Выбор на val до test. Обученные базы полностью переобучены в исходных GPU-запусках. Профиль причинный из прежнего пакета. Коэффициент для сабмита обновлён по всем доступным OOF. Test уже изучался.'), ensure_ascii=False, indent=2))
print(choice)
print(pd.DataFrame(reports).to_string(index=False))
