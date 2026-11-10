from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

parser=argparse.ArgumentParser()
parser.add_argument('directory',nargs='?',default='artifacts/clustering/calibration_367876306')
args=parser.parse_args()
root=Path(args.directory)


def read_phase(stage,variant):
    paths=sorted(root.glob(f'{stage}_h*_{variant}.parquet'))
    if not paths:
        raise FileNotFoundError(stage+' '+variant)
    frame=pd.concat([pd.read_parquet(path) for path in paths],ignore_index=True)
    if frame.duplicated(['route_id','cutoff','h']).any():
        raise ValueError('Повторяющиеся прогнозы')
    return frame


def fit_factor(frame,half_life):
    age=(frame.cutoff.max()-frame.cutoff).dt.total_seconds().to_numpy()/86400
    weights=np.ones(len(frame)) if half_life is None else np.exp2(-age/half_life)
    value=float((frame.y*weights).sum()/(frame.y_pred*weights).sum())
    return float(np.clip(value,0.8,1.25))


def metric(frame,factor):
    errors=frame.y_pred.to_numpy()*factor-frame.y.to_numpy()
    volume=float(frame.y.sum())
    return dict(WAPE=float(np.abs(errors).sum()/volume),relative_bias=float(errors.sum()/volume),
                score=float((np.abs(errors).sum()+abs(errors.sum()))/volume))


rows,choices=[],{}
for variant in ('cat_k1','cat_k8'):
    fit,select=(read_phase(stage,variant) for stage in ('calibration_fit','calibration_select'))
    if fit.timestamp.max() >= select.cutoff.min():
        raise ValueError('Периоды калибровки пересекаются')
    local=[]
    for half_life in (None,1,2,3,7):
        factor=fit_factor(fit,half_life)
        row=dict(variant=variant,half_life=half_life,factor=factor,**metric(select,factor))
        rows.append(row)
        local.append(row)
    best=min(local,key=lambda row:row['score'])
    choices[variant]=dict(half_life=best['half_life'],selection_score=best['score'],
                          factor=fit_factor(pd.concat([fit,select],ignore_index=True),best['half_life']))
winner=min(choices,key=lambda name:choices[name]['selection_score'])
frozen=dict(winner=winner,choices=choices,note='Выбор по второй неделе val. Test ранее изучался; исследовательская проверка, не новый слепой holdout.')
(root/'recency_frozen.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2))
pd.DataFrame(rows).to_csv(root/'recency_calibration_selection.csv',index=False)
reports=[]
for variant,choice in choices.items():
    test=read_phase('test',variant)
    reports.append(dict(variant=variant,selected_on_val=variant==winner,half_life=choice['half_life'],factor=choice['factor'],**metric(test,choice['factor'])))
pd.DataFrame(reports).to_csv(root/'recency_calibration_test.csv',index=False)
print(pd.DataFrame(reports).to_string(index=False))
