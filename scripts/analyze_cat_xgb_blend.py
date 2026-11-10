import json
from pathlib import Path
import numpy as np
import pandas as pd

cat_root = Path('artifacts/clustering/calibration_367876306')
xgb_root = Path('artifacts/clustering/xgb_367884401')
out = Path('artifacts/clustering/cat_xgb_blend')
out.mkdir(exist_ok=True)


def read_phase(stage):
    frames = []
    for root in (cat_root, xgb_root):
        frame = pd.concat([pd.read_parquet(p) for p in sorted(root.glob(f'{stage}_h*_cat_k8.parquet'))], ignore_index=True)
        frames.append(frame)
    merged = frames[0].merge(frames[1][['route_id', 'cutoff', 'h', 'y', 'y_pred']], on=['route_id', 'cutoff', 'h'], suffixes=('_cat', '_xgb'), validate='one_to_one')
    if len(merged) != len(frames[0]) or not np.array_equal(merged.y_cat, merged.y_xgb):
        raise ValueError('Несовпадающие OOF')
    return merged.rename(columns={'y_cat': 'y'})


def prediction(frame, weight):
    return weight * frame.y_pred_cat.to_numpy() + (1 - weight) * frame.y_pred_xgb.to_numpy()


def factor(frame, weight, half_life):
    age = (frame.cutoff.max() - frame.cutoff).dt.total_seconds().to_numpy() / 86400
    recency = np.ones(len(frame)) if half_life is None else np.exp2(-age / half_life)
    return float(np.clip(np.sum(frame.y * recency) / np.sum(prediction(frame, weight) * recency), .8, 1.25))


def metric(frame, pred):
    error = pred - frame.y.to_numpy()
    volume = float(frame.y.sum())
    return dict(WAPE=float(np.abs(error).sum() / volume), relative_bias=float(error.sum() / volume), score=float((np.abs(error).sum() + abs(error.sum())) / volume))


fit, select = [read_phase(stage) for stage in ('calibration_fit', 'calibration_select')]
rows = []
for weight in (0., .25, .5, .75, 1.):
    for half_life in (None, 2, 3, 7):
        coefficient = factor(fit, weight, half_life)
        rows.append(dict(cat_weight=weight, half_life=half_life, factor=coefficient, **metric(select, prediction(select, weight) * coefficient)))
choice = min(rows, key=lambda row: row['score'])
(out / 'frozen.json').write_text(json.dumps(choice, indent=2))
pd.DataFrame(rows).to_csv(out / 'selection.csv', index=False)
all_val = pd.concat([fit, select], ignore_index=True)
test = read_phase('test')
coefficient = factor(all_val, choice['cat_weight'], choice['half_life'])
reports = [dict(period=period, factor=coefficient, **metric(frame, prediction(frame, choice['cat_weight']) * coefficient)) for period, frame in [('train_oof', all_val), ('test', test)]]
pd.DataFrame(reports).to_csv(out / 'independent_metrics.csv', index=False)
full_factor = factor(pd.concat([all_val, test], ignore_index=True), choice['cat_weight'], choice['half_life'])
submission = pd.read_csv(cat_root / 'submission_raw.csv').merge(pd.read_csv(xgb_root / 'submission_raw.csv'), on='id', suffixes=('_cat', '_xgb'), validate='one_to_one', sort=False)
submission['y_pred'] = (choice['cat_weight'] * submission.y_pred_cat + (1 - choice['cat_weight']) * submission.y_pred_xgb) * full_factor
if not np.isfinite(submission.y_pred).all() or (submission.y_pred < 0).any() or len(submission) != 8000:
    raise ValueError('Некорректный сабмит')
submission[['id', 'y_pred']].to_csv(out / 'submission.csv', index=False)
(out / 'final_metadata.json').write_text(json.dumps(dict(choice=choice, final_factor=full_factor, note='Базы CatBoost/XGBoost полностью переобучены в исходных GPU-запусках. Выбор смеси и half-life по val; test до полного refit. Итоговый коэффициент обновлён по доступным OOF val/test. Test уже изучался.'), ensure_ascii=False, indent=2))
print(choice)
print(pd.DataFrame(reports).to_string(index=False))
