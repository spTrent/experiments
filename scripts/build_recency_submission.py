import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('directory')
args = parser.parse_args()
root = Path(args.directory)
selection = json.loads((root / 'recency_frozen.json').read_text())
source = json.loads((root / 'calibration_frozen.json').read_text())
variant = selection['winner']
if variant != source['winner']:
    raise ValueError('Полностью переобученная база относится к другому варианту')
choice = selection['choices'][variant]
frames = []
for stage in ('calibration_fit', 'calibration_select', 'test'):
    frames.extend(pd.read_parquet(path) for path in sorted(root.glob(f'{stage}_h*_{variant}.parquet')))
oof = pd.concat(frames, ignore_index=True)
if oof.duplicated(['route_id', 'cutoff', 'h']).any():
    raise ValueError('Повторные OOF-прогнозы')
half_life = choice['half_life']
age = (oof.cutoff.max() - oof.cutoff).dt.total_seconds().to_numpy() / 86400
weights = np.ones(len(oof)) if half_life is None else np.exp2(-age / half_life)
factor = float(np.clip((oof.y * weights).sum() / (oof.y_pred * weights).sum(), .8, 1.25))
submission = pd.read_csv(root / 'submission_raw.csv')
submission.y_pred *= factor
if submission.id.duplicated().any() or not np.isfinite(submission.y_pred).all() or (submission.y_pred < 0).any():
    raise ValueError('Некорректные итоговые прогнозы')
submission.to_csv(root / 'submission_recency.csv', index=False)
(root / 'submission_recency_metadata.json').write_text(json.dumps(dict(variant=variant, half_life=half_life, final_factor=factor, n_oof=len(oof), n_predictions=len(submission), note='Базы полностью переобучены в исходном Kaggle-запуске. Test оценён до полного refit и обновления коэффициента. Half-life выбран до test; итоговый коэффициент обновлён на всех доступных OOF val/test.'), ensure_ascii=False, indent=2))
print(dict(variant=variant, half_life=half_life, final_factor=factor, n_predictions=len(submission)))
