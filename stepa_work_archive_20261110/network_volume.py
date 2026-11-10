"""Прогноз суммы восьми перекрывающихся часовых target по всей сети.

Это сумма меток для конкурсной метрики, не уникальный физический объём за 4 часа.
Обучение до 4 октября 00:00, прогнозы в 10:30 на tuning и holdout.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


def build(root):
    cache = root/'inputs/network_training.parquet'
    if cache.exists():
        return pd.read_parquet(cache)
    path = Path('/root/rwb-stepa-submission/data/train_solo_track.parquet')
    cols = ['timestamp','target_1h']+[f'status_{j}' for j in range(1,7)]
    d = pd.read_parquet(path,columns=cols).groupby('timestamp',sort=True).sum()
    assert d.index.equals(pd.date_range(d.index[0],d.index[-1],freq='30min'))
    x = pd.DataFrame(index=d.index)
    target = d.target_1h
    x['current_target'] = target
    for col in d.columns:
        for lag in (0,1,2,4,8,24,48,96,336):
            x[f'{col}_lag{lag}'] = d[col].shift(lag)/target
        for w in (4,12,48,336):
            x[f'{col}_mean{w}'] = d[col].rolling(w,min_periods=w).mean()/target
        for lag in (1,2,4,48):
            x[f'{col}_change{lag}'] = (d[col]-d[col].shift(lag))/target
    hour = d.index.hour+d.index.minute/60
    x['hour_sin'] = np.sin(hour*2*np.pi/24)
    x['hour_cos'] = np.cos(hour*2*np.pi/24)
    x['dayofweek'] = d.index.dayofweek
    x['day_trend'] = (d.index-d.index[0]).total_seconds()/86400
    future = sum(target.shift(-h) for h in range(1,9))
    x['future_sum'] = future
    x['label_ratio'] = future/(target*8)
    x['timestamp'] = x.index
    x['label_end'] = x.timestamp+pd.Timedelta(hours=4)
    # Последняя точка без labels нужна для submission; пропуски будущей цели не заполняем.
    valid = x.drop(columns=['future_sum','label_ratio']).notna().all(axis=1)
    x = x.loc[valid].reset_index(drop=True)
    x.to_parquet(cache,index=False)
    return x


def fit_predict(root, variant='depth4'):
    data = build(root)
    cols = data.columns.drop(['timestamp','label_end','future_sum','label_ratio']).tolist()
    # Календарный тренд может плохо экстраполировать; исключаем его из первого варианта.
    cols.remove('day_trend')
    train = data.label_end < pd.Timestamp('2025-10-04')
    tune = data.timestamp.between('2025-10-04','2025-10-17 23:59') & data.timestamp.dt.hour.eq(10) & data.timestamp.dt.minute.eq(30)
    query = data.timestamp.dt.hour.eq(10) & data.timestamp.dt.minute.eq(30) & data.timestamp.ge('2025-10-04')
    assert data.loc[train,'label_end'].max()+pd.Timedelta(hours=4) < data.loc[tune,'timestamp'].min()
    model = CatBoostRegressor(iterations=1500,depth=4,learning_rate=.03,loss_function='RMSE',
        l2_leaf_reg=20,random_seed=42,thread_count=4,allow_writing_files=False,verbose=300)
    model.fit(data.loc[train,cols],data.loc[train,'label_ratio'],
        eval_set=(data.loc[tune,cols],data.loc[tune,'label_ratio']),early_stopping_rounds=150)
    output = data.loc[query,['timestamp','future_sum','current_target']].copy()
    output['network_prediction'] = np.maximum(model.predict(data.loc[query,cols])*output.current_target.to_numpy()*8,0)
    print('NETWORK',model.tree_count_,'TREES', 'TRAIN_ROWS',int(train.sum()),flush=True)
    print(output.to_string(index=False),flush=True)
    return {'model':model,'columns':cols,'trees':model.tree_count_,'data':data,'predictions':output,
            'fit_through':str(data.loc[train,'label_end'].max())}

