from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

parser=argparse.ArgumentParser()
parser.add_argument('directory',nargs='?',default='artifacts/clustering/calibration_367876306')
root=Path(parser.parse_args().directory)


def read(stage,variant):
    return pd.concat([pd.read_parquet(p) for p in sorted(root.glob(f'{stage}_h*_{variant}.parquet'))],ignore_index=True)


def predict(history,query,half_life):
    known=pd.concat([history,query],ignore_index=True)
    predicted=np.empty(len(query),dtype=float)
    audit=[]
    for cutoff in sorted(query.cutoff.unique()):
        cutoff=pd.Timestamp(cutoff)
        past=known[known.timestamp<cutoff-pd.Timedelta(hours=4)]
        if past.empty:
            raise ValueError('Нет доступной истории')
        age=(cutoff-past.cutoff).dt.total_seconds().to_numpy()/86400
        weights=np.ones(len(past)) if half_life is None else np.exp2(-age/half_life)
        factor=float(np.clip((past.y*weights).sum()/(past.y_pred*weights).sum(),.8,1.25))
        selected=query.cutoff.eq(cutoff).to_numpy()
        predicted[selected]=query.loc[selected,'y_pred']*factor
        audit.append(dict(cutoff=str(cutoff),fit_through=str(past.timestamp.max()),factor=factor))
    return predicted,audit


def metric(frame,pred):
    errors=pred-frame.y.to_numpy()
    volume=float(frame.y.sum())
    return dict(WAPE=float(np.abs(errors).sum()/volume),relative_bias=float(errors.sum()/volume),score=float((np.abs(errors).sum()+abs(errors.sum()))/volume))


rows,choices=[],{}
for variant in ('cat_k1','cat_k8'):
    fit,select=read('calibration_fit',variant),read('calibration_select',variant)
    local=[]
    for half_life in (None,1,2,3,7):
        prediction,audit=predict(fit,select,half_life)
        row=dict(variant=variant,half_life=half_life,**metric(select,prediction))
        rows.append(row);local.append(row)
    choices[variant]=min(local,key=lambda row:row['score'])
winner=min(choices,key=lambda name:choices[name]['score'])
(root/'rolling_frozen.json').write_text(json.dumps(dict(winner=winner,choices=choices,protocol='prequential; only labels timestamp < cutoff - 4 hours'),indent=2))
pd.DataFrame(rows).to_csv(root/'rolling_selection.csv',index=False)
reports=[]
for variant,choice in choices.items():
    fit,select,test=[read(stage,variant) for stage in ('calibration_fit','calibration_select','test')]
    pred,audit=predict(pd.concat([fit,select],ignore_index=True),test,choice['half_life'])
    reports.append(dict(variant=variant,selected_on_val=variant==winner,half_life=choice['half_life'],**metric(test,pred)))
    pd.DataFrame(audit).to_csv(root/f'rolling_audit_{variant}.csv',index=False)
    altered=test.copy()
    last=altered.cutoff.max()
    altered.loc[altered.cutoff==last,'y']*=10
    check,_=predict(pd.concat([fit,select],ignore_index=True),altered,choice['half_life'])
    if not np.array_equal(check,pred):
        raise ValueError('Изменение недоступных будущих меток влияет на прогноз')
pd.DataFrame(reports).to_csv(root/'rolling_test.csv',index=False)
print(pd.DataFrame(reports).to_string(index=False))
