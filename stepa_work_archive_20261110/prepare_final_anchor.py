"""Проверить формулы anchor на OOS-строках, затем добавить только последний cutoff."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
path=Path('/root/rwb-stepa-submission/data/train_solo_track.parquet')
columns=['timestamp','route_id','target_1h']+[f'status_{i}' for i in range(1,7)]
history=pd.read_parquet(path,columns=columns).sort_values(['timestamp','route_id'])
times=pd.DatetimeIndex(history.timestamp.unique());routes=np.sort(history.route_id.unique())
assert len(history)==len(times)*len(routes)
assert times.equals(pd.date_range(times[0],times[-1],freq='30min'))
assert not history.duplicated(['timestamp','route_id']).any()
original=pd.read_parquet(ROOT/'inputs/anchor_features.parquet')
original=original.loc[original.cutoff<=pd.Timestamp('2025-10-31 10:30')]
origins=times.get_indexer(pd.DatetimeIndex(sorted(set(original.cutoff)|{times[-1]})))
assert (origins>=336).all()
values={c:history[c].to_numpy(dtype=np.float32).reshape(len(times),len(routes)) for c in columns[2:]}
del history
means={}
for c,array in values.items():
 for w in (4,12):
  means[c,w]=pd.DataFrame(array).rolling(w,min_periods=1).mean().to_numpy(dtype=np.float32)
result={'cutoff':times[origins].repeat(len(routes)), 'route_id':np.tile(routes,len(origins))}
target=values['target_1h']
for lag in (0,1,2,4):result[f'target_back_{lag}']=target[origins-lag].ravel()
for c in values:
 for w in (4,12):result[f'{c}_mean_{w}']=means[c,w][origins].ravel()
 if c.startswith('status'):result[c+'_latest']=values[c][origins].ravel()
for lag in (48,96):result[f'target_1h_lag_{lag}']=target[origins-lag].ravel()
result['level24_feature']=pd.DataFrame(target).rolling(48,min_periods=1).mean().to_numpy(dtype=np.float32)[origins].ravel()
result['target_vs_mean_6h']=(target[origins]-means['target_1h',12][origins]).ravel()
usual=np.zeros_like(means['target_1h',4][origins])
for day in range(1,8):usual+=means['target_1h',4][origins-day*48]
usual/=np.float32(7)
result['morning_level_ratio']=(means['target_1h',4][origins]/usual).ravel()
s1=values['status_1'][origins]+values['status_1'][origins-1]
s3=values['status_3'][origins]+values['status_3'][origins-1]
result['warehouse_processing_rate']=(s3/(s1+np.float32(1e-5))).ravel()
result['target_acceleration']=(target[origins-1]-target[origins-2]).ravel()
new=pd.DataFrame(result)[original.columns]
compare=original.merge(new,on=['cutoff','route_id'],validate='one_to_one',suffixes=('_old','_new'))
assert len(compare)==28000
for col in original.columns.drop(['cutoff','route_id']):
 np.testing.assert_allclose(compare[col+'_old'],compare[col+'_new'],rtol=1e-6,atol=1e-5,err_msg=col)
assert len(new)==29000 and np.isfinite(new.select_dtypes(include='number')).all().all()
new.to_parquet(ROOT/'inputs/anchor_features.parquet',index=False)
print('VERIFIED_ANCHOR_FORMULAS',len(compare),'FINAL_ORIGIN',times[-1],'OUTPUT_ROWS',len(new))

