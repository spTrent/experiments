import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

n=json.loads(Path('notebooks/clustering/nightly_rich_gpu.ipynb').read_text())
ns={}
for i in (1,2,3,4):
    exec(''.join(n['cells'][i]['source']),ns)
length,width=3000,4
times=pd.date_range('2025-07-28',periods=length,freq='30min')
ticks=np.arange(length)[:,None]
values={'target_1h':(100+10*np.sin(ticks/10)+np.arange(width)[None,:]).astype(np.float32)}
for i in range(1,7):
    values[f'status_{i}']=(values['target_1h']*(i+1)).astype(np.float32)
ns.update(TIMES=times,ROUTES=np.arange(width),VALUES=values,STEP=pd.Timedelta(minutes=30),
          ROLL={w:pd.DataFrame(values['target_1h']).rolling(w,min_periods=1).mean().to_numpy(dtype=np.float32) for w in (4,12,48,336)})
ns['segment_routes']=lambda boundary,stage:dict(scale=np.full(width,100.),labels={1:np.zeros(width,dtype=int),8:np.zeros(width,dtype=int)})
with tempfile.TemporaryDirectory() as folder:
    ns['RUN']=Path(folder)
    exec(''.join(n['cells'][-2]['source']),ns)
    segmentation=ns['segment_routes'](times[2000],'check')
    x,names=ns['features'](4,np.array([1500]),segmentation)
    if not np.isfinite(x).all() or len(names)!=len(set(names)):
        raise ValueError('Invalid rich features')
    for value in values.values():
        value[1501:]+=100000
    after,_=ns['features'](4,np.array([1500]),segmentation)
    if not np.array_equal(x,after):
        raise ValueError('Future leakage in rich features')
    frame=pd.DataFrame(dict(cutoff=pd.to_datetime(['2025-10-04','2025-10-10']),h=[1,2],y=[10.,30.],y_pred=[10.,20.]))
    factors=ns['fit_factors'](frame,'recent_global_3')
    expected=(10*0.25+30)/(10*0.25+20)
    if not np.isclose(factors['1'],min(1.25,expected)):
        raise ValueError('Incorrect recency coefficient')
    print('Rich features:',x.shape[1],'; finite, unique names, causal, recency coefficients: OK')

class ProbeRegressor:
    def __init__(self,**params):
        self.params=params
    def fit(self,frame,y,sample_weight=None):
        if 'route_code' not in frame:
            raise ValueError('Missing categorical route')
        if self.params.get('device')=='cuda:0' and not isinstance(frame.route_code.dtype,pd.CategoricalDtype):
            raise ValueError('XGBoost route must be categorical')
        if self.params.get('task_type')=='GPU' and self.params.get('cat_features')!=['route_code']:
            raise ValueError('CatBoost route declaration missing')
        self.value=float(np.average(y,weights=sample_weight))
        return self
    def predict(self,frame):
        if 'route_code' not in frame:
            raise ValueError('Missing route at inference')
        return np.full(len(frame),self.value)
    def get_booster(self):
        return self
    def save_config(self):
        return json.dumps(dict(learner=dict(generic_param=dict(device='cuda:0'))))

import types
ns.update(CatBoostRegressor=ProbeRegressor,xgb=types.SimpleNamespace(XGBRegressor=ProbeRegressor),ACTIVE_H=1,check_deadline=lambda:None)
train=dict(x=x,names=names,route_index=np.arange(width),y=np.arange(width,dtype=float)+100.)
segmentation['labels']={1:np.zeros(width,dtype=int)}
for algorithm,parameter_set in [('cat','catboost_direct'),('xgb','xgb_direct')]:
    ns['ACTIVE']=dict(algorithm=algorithm,parameter_set=parameter_set,residual=False,name='probe',loss='MAE' if algorithm=='cat' else 'reg:absoluteerror',compact=True)
    model=ns['fit_model'](train,segmentation,'cat',1,{})
    pred=ns['predict_model'](model,train,segmentation)
    if not np.isfinite(pred).all():
        raise ValueError('Invalid categorical model prediction')
print('Categorical route passed to CatBoost/XGBoost fit and inference: OK; estimators replaced with probes.')
