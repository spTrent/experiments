import json
import tempfile
import contextlib
import io
from pathlib import Path
import numpy as np
import pandas as pd

n=json.loads(Path('notebooks/clustering/learned_residual_gpu.ipynb').read_text())
ns={}
for i in (1,2,3):
    exec(''.join(n['cells'][i]['source']),ns)


class DummyCat:
    def __init__(self,**params):
        if params['task_type']!='GPU':
            raise ValueError('CPU training')
    def fit(self,x,y,sample_weight=None):
        self.value=float(np.average(y,weights=sample_weight))
        return self
    def predict(self,x):
        return np.full(len(x),self.value)


with tempfile.TemporaryDirectory() as folder:
    root=Path(folder)
    routes=np.array([10,20,30,40])
    times=pd.date_range('2025-07-28','2025-11-01 10:30',freq='30min')
    signal=(100+np.arange(len(times))[:,None]/100+np.arange(4)[None,:]).astype(np.float32)
    values={'target_1h':signal}
    for i in range(1,7):
        values[f'status_{i}']=signal*i
    requests=pd.DataFrame([dict(id=i*2+h,route_id=r,timestamp=times[-1]+h*pd.Timedelta(minutes=30)) for i,r in enumerate(routes) for h in (1,2)]).sample(frac=1,random_state=42).reset_index(drop=True)
    raw_submission=requests[['id']].copy();raw_submission['y_pred']=135.;raw_submission.to_csv(root/'submission_raw.csv',index=False)
    for stage,days in [('calibration_fit',['2025-10-04 10:30','2025-10-05 10:30']),('calibration_select',['2025-10-11 10:30','2025-10-12 10:30']),('test',['2025-10-18 10:30','2025-10-19 10:30'])]:
        for h in (1,2):
            rows=[]
            for day in pd.to_datetime(days):
                pos=times.get_loc(day)
                for i,r in enumerate(routes):
                    y=float(signal[pos+h,i]);rows.append(dict(route_id=r,cutoff=day,timestamp=day+h*pd.Timedelta(minutes=30),h=h,y=y,y_pred=.92*y))
            pd.DataFrame(rows).to_parquet(root/f'{stage}_h{h}_cat_k8.parquet',index=False)
    ns.update(SOURCE_ROOT=root,RUN=root,TIMES=times,ROUTES=routes,VALUES=values,HORIZONS=(1,2),CLUSTERS=(1,2),STEP=pd.Timedelta(minutes=30),
              META_SEGMENTATION=dict(scale=np.full(4,100.)),META_LABELS=np.array([0,0,1,1]),
              ROLL={w:pd.DataFrame(signal).rolling(w,min_periods=1).mean().to_numpy(dtype=np.float32) for w in (4,12,48,336)},
              REQUESTS=requests,CatBoostRegressor=DummyCat,display=lambda *args:None,initialize_meta=lambda:None)
    frame=ns['meta_frame'](ns['source_frame']('calibration_fit'))
    model=ns['meta_fit'](frame,2,ns['META_CONFIGS'][0])
    before=ns['meta_predict'](model,frame)
    query=ns['meta_frame'](ns['source_frame']('calibration_select'))
    online,audit=ns['online_meta_offset'](model,frame,query)
    future=query.copy()
    future.loc[future.cutoff==future.cutoff.max(),'y']*=100
    unchanged,_=ns['online_meta_offset'](model,frame,future)
    if not np.array_equal(online,unchanged) or any(pd.Timestamp(a['fit_through'])>=pd.Timestamp(a['cutoff'])-pd.Timedelta(hours=4) for a in audit):
        raise ValueError('Rolling correction uses unavailable labels')
    frame['y']+=10000
    after=ns['meta_predict'](model,frame)
    if not np.array_equal(before,after):
        raise ValueError('Prediction reads target')
    with contextlib.redirect_stdout(io.StringIO()):
        ns['run_meta']()
    report=pd.read_csv(root/'learned_residual_independent_metrics.csv')
    if set(report.period)!={'train','test'}:
        raise ValueError('Missing independent metrics')
    submission=pd.read_csv(root/'submission.csv')
    if submission.id.tolist()!=requests.id.tolist() or not np.isfinite(submission.y_pred).all():
        raise ValueError('Invalid submission')
print('Learned residual: feature matching, target-independent prediction, temporal selection, train/test report and final submission: OK. GPU model replaced with a test double.')
