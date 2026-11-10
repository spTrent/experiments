"""Сравнение признаков stepa330: h4/h5/h8, recent75, шесть CatBoost fit."""
from pathlib import Path
import gc
import hashlib
import importlib.metadata
import json
import os
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from IPython.display import display

TARGET = 'target_1h'
STATUSES = [f'status_{i}' for i in range(1,7)]
STEP = pd.Timedelta(minutes=30)
LAG_MAX_LOOKBACK_HOURS = 24
LAG_MIN_PAIRS = 336
LAG_MIN_CORRELATION = 0.05
TARGET_SCALE = 100000.
VARIANTS = ('control','seasonal_features')
SEASONAL_FEATURES = ['profile28','level24_feature','target_level24_logratio',
                     'mean2h_level24_logratio','profile28_minus_daily7']
STATUS_FEATURES = []
PILOT_H6_FEATURES = ['start_sin','start_cos','peak7_mean_max','peak7_mean_level',
                     'peak7_share_-1','peak7_share_0','peak7_share_1']


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.partial')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False));temp.replace(path)


def write_parquet(path,frame):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.partial');frame.to_parquet(temp,index=False);temp.replace(path)


def initialize():
    global ROOT,TRAIN_PATH,TEST_PATH
    TRAIN_PATH=Path(DATA_DIR)/'train_solo_track.parquet'
    TEST_PATH=Path(DATA_DIR)/'test_solo_track.parquet'
    for path in (TRAIN_PATH,TEST_PATH):
        if not path.is_file():raise FileNotFoundError(path)
    if not 0<RECENT_FRACTION<=1 or not set(HORIZONS).issubset({4,5,8}):
        raise ValueError('Проверьте RECENT_FRACTION и HORIZONS (пилот только h4/h5/h8)')
    identity=dict(code=CODE_ID,train=digest(TRAIN_PATH),test=digest(TEST_PATH),
                  recent_fraction=RECENT_FRACTION,trees=MAX_TREES,patience=EARLY_STOPPING,
                  depth=7,learning_rate=.05,loss='MAE',seed=42,horizons=list(HORIZONS))
    run_id=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    ROOT=Path(OUTPUT_DIR)/f'pilot_{run_id}';ROOT.mkdir(parents=True,exist_ok=True)
    write_json(ROOT/'manifest.json',identity)
    print('Результаты:',ROOT,flush=True)


def load_panel():
    global history,times,routes,values,target,test,end,dates,tune_start,holdout_start,LAG_VALIDATION_FIT_BEFORE
    history=pd.read_parquet(TRAIN_PATH,columns=['route_id','timestamp',TARGET,*STATUSES]).sort_values(['timestamp','route_id'])
    if history.isna().any().any() or history.duplicated(['route_id','timestamp']).any():
        raise ValueError('Пропуски или дубликаты в train')
    times=pd.DatetimeIndex(history.timestamp.unique());routes=np.sort(history.route_id.unique())
    if not times.equals(pd.date_range(times[0],times[-1],freq=STEP)) or len(history)!=len(times)*len(routes):
        raise ValueError('Неполная получасовая панель')
    values={c:history[c].to_numpy(dtype=np.float32).reshape(len(times),len(routes)) for c in [TARGET,*STATUSES]}
    del history
    if not all(np.isfinite(a).all() and (a>=0).all() for a in values.values()):
        raise ValueError('Нечисловые или отрицательные значения в данных')
    target=values[TARGET];end=times[-1]
    test=pd.read_parquet(TEST_PATH,columns=['id','route_id','timestamp'])
    expected=pd.MultiIndex.from_product([routes,[end+h*STEP for h in range(1,9)]],names=['route_id','timestamp'])
    actual=pd.MultiIndex.from_frame(test[['route_id','timestamp']])
    if test.isna().any().any() or not test.id.is_unique or not actual.is_unique or len(expected)!=len(actual) or len(expected.difference(actual)):
        raise ValueError('Неверные конкурсные запросы')
    dates=times[(times.hour==10)&(times.minute==30)&(times+8*STEP<=end)]
    if len(dates)<=42:raise ValueError('Недостаточно истории для train, tuning и holdout')
    tune_start,holdout_start=dates[-28],dates[-14]
    LAG_VALIDATION_FIT_BEFORE=tune_start.normalize()
    if tune_start-LAG_VALIDATION_FIT_BEFORE<pd.Timedelta(hours=4):
        raise ValueError('Недостаточный embargo')
    report=dict(train_start=str(times[0]),train_end=str(end),routes=len(routes),
                fit_labels_before=str(LAG_VALIDATION_FIT_BEFORE),
                tuning_start=str(tune_start),tuning_end=str(dates[-15]),
                holdout_start=str(holdout_start),holdout_end=str(dates[-1]))
    write_json(ROOT/'periods.json',report);display(pd.DataFrame([report]).T)


def fit_correlation_dictionary():
    global validation_lag_table
    validation_lag_table=calculate_dynamic_shoulder(target,hub_signal,times,routes,
        fit_before=LAG_VALIDATION_FIT_BEFORE,max_lookback_hours=LAG_MAX_LOOKBACK_HOURS,
        min_pairs=LAG_MIN_PAIRS,min_correlation=LAG_MIN_CORRELATION)
    validation_lag_table.to_csv(ROOT/'status6_lags_train.csv',index=False)


def add_pilot_features(frame,h,origins):
    """Сезонный пакет из предыдущего пилота; источники доступны на собственном cutoff."""
    current=target[origins];level=level24[origins]
    frame['level24_feature']=level.ravel()
    frame['target_level24_logratio']=(np.log1p(current)-np.log1p(level)).ravel()
    frame['mean2h_level24_logratio']=(np.log1p(means[TARGET,4][origins])-np.log1p(level)).ravel()
    frame['profile28_minus_daily7']=(frame.profile28-frame.slot_daily7).astype(np.float32)
    return frame.astype({c:'float32' for c in [*SEASONAL_FEATURES,*STATUS_FEATURES]})


def origin_sets(h):
    eligible=np.flatnonzero(times[:-h]+h*STEP<LAG_VALIDATION_FIT_BEFORE)
    start=int(np.floor((1-RECENT_FRACTION)*len(eligible)))
    train=eligible[start:]
    evaluate=times.get_indexer(dates[-28:])
    if not len(train) or (evaluate<0).any():raise ValueError('Пустая/неполная выборка')
    if not times[train+h].max()<times[evaluate].min()-pd.Timedelta(hours=4):
        raise ValueError('Утечка labels или недостаточный embargo')
    return train,evaluate


def build_features(h):
    path=ROOT/'features'/f'h{h}.parquet';path.parent.mkdir(exist_ok=True)
    train,evaluate=origin_sets(h)
    info=dict(horizon=h,train_rows=len(train)*len(routes),
              first_train_cutoff=str(times[train[0]]),last_train_label=str(times[train[-1]+h]),
              original_eligible_origins=int(np.sum(times[:-h]+h*STEP<LAG_VALIDATION_FIT_BEFORE)),
              retained_origins=len(train),evaluation_origins=len(evaluate),history_for_lags='all past')
    write_json(path.with_suffix('.json'),info);display(pd.DataFrame([info]))
    if path.exists():return path
    origins=np.sort(np.unique(np.concatenate([train,evaluate])))
    temp=path.with_suffix('.partial');writer=None;started=time.monotonic()
    try:
        for start in range(0,len(origins),FEATURE_ORIGIN_BATCH):
            idx=origins[start:start+FEATURE_ORIGIN_BATCH]
            frame=add_pilot_features(horizon_dataset(h,idx),h,idx)
            frame['y']=target[idx+h].ravel()
            frame['split']=np.repeat(np.where(np.isin(idx,train),'train',np.where(times[idx]<holdout_start,'tuning','holdout')),len(routes))
            table=pa.Table.from_pandas(frame,preserve_index=False)
            if writer is None:writer=pq.ParquetWriter(temp,table.schema,compression='zstd')
            writer.write_table(table)
    finally:
        if writer is not None:writer.close()
    temp.replace(path)
    print(f'h{h}: признаки {(time.monotonic()-started)/60:.1f} мин',flush=True)
    return path


def columns_for(frame,variant):
    excluded={'cutoff','timestamp','y','profile28','overlap','split',*PILOT_H6_FEATURES,
              *SEASONAL_FEATURES,*STATUS_FEATURES}
    control=[c for c in frame.columns if c not in excluded]
    columns=control+({'control':[],'seasonal_features':SEASONAL_FEATURES}[variant])
    if len(columns)!=len(set(columns)) or not set(columns).issubset(frame):
        raise ValueError('Неверные признаки')
    return columns


def metric(frame,p):
    y=frame.y.to_numpy(dtype=np.float64);p=np.asarray(p,dtype=np.float64)
    if y.shape!=p.shape or not np.isfinite(p).all() or (p<0).any() or len(y)==0:
        raise ValueError('Некорректные прогнозы/факты')
    e=p-y;volume=float(y.sum())
    daily=pd.DataFrame(dict(day=frame.cutoff.dt.normalize(),volume=y,absolute=np.abs(e),signed=e)).groupby('day').sum()
    nonzero=daily.volume>0
    return dict(rows=len(frame),days=len(daily),WAPE=float(np.abs(e).sum()/volume) if volume>0 else None,
                relative_bias=float(e.sum()/volume) if volume>0 else None,
                score=float((np.abs(e).sum()+abs(e.sum()))/volume) if volume>0 else None,
                mean_daily_score=float(((daily.absolute+daily.signed.abs())/daily.volume)[nonzero].mean()) if nonzero.any() else None,
                zero_volume_days=int((~nonzero).sum()))



RECIPE = {1:'control',2:'control',3:'control',4:'control',5:'seasonal_features',6:'seasonal_features',7:'control',8:'seasonal_features'}


def initialize_submission():
    global ROOT,TRAIN_PATH,TEST_PATH
    TRAIN_PATH=Path(DATA_DIR)/'train_solo_track.parquet';TEST_PATH=Path(DATA_DIR)/'test_solo_track.parquet'
    for p in [TRAIN_PATH,TEST_PATH,Path(SEED_DIR)/'manifest.json']:
        if not p.is_file():raise FileNotFoundError(p)
    seed=json.loads((Path(SEED_DIR)/'manifest.json').read_text())
    actual=dict(train=digest(TRAIN_PATH),test=digest(TEST_PATH))
    if actual!=seed['data']:raise ValueError('Исходные данные отличаются от данных пилотов')
    for item in seed['files']:
        if digest(Path(SEED_DIR)/item['path'])!=item['sha256']:raise ValueError('Повреждён кеш пилота')
    identity=dict(code=CODE_ID,data=actual,seeds=seed,recipe=RECIPE,recent_fraction=RECENT_FRACTION,
                  max_trees=MAX_TREES,early_stopping=EARLY_STOPPING,
                  blend=dict(h1_catboost=1.,later_catboost=.7,later_profile28=.3),calibration='additive')
    ident=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    ROOT=Path(OUTPUT_DIR)/f'submit_{ident}';ROOT.mkdir(parents=True,exist_ok=True)
    write_json(ROOT/'manifest.json',identity)
    print('Результаты:',ROOT,flush=True)


def validate_frame(frame,h,info):
    origins=times.get_indexer(dates[-28:])
    expected=pd.DataFrame(dict(cutoff=times[origins].repeat(len(routes)),route_id=np.tile(routes,len(origins))))
    expected['horizon']=h;expected['timestamp']=expected.cutoff+h*STEP
    expected['y']=target[origins+h].ravel()
    expected['split']=np.where(expected.cutoff<holdout_start,'tuning','holdout')
    keys=['cutoff','route_id','horizon','timestamp','y','split']
    if frame.duplicated(['cutoff','route_id','horizon']).any() or not all(np.array_equal(frame[k].to_numpy(),expected[k].to_numpy()) for k in keys):
        raise ValueError(f'Неполные/несовпадающие прогнозы h{h}')
    if pd.Timestamp(info['fit_labels_before'])!=LAG_VALIDATION_FIT_BEFORE:
        raise ValueError('Неверная граница обучения кешированной модели')
    p=frame.prediction.to_numpy()
    if not np.isfinite(p).all() or (p<0).any():raise ValueError('Некорректный кешированный прогноз')
    return frame


def make_catboost(trees):
    from catboost import CatBoostRegressor
    from catboost.utils import get_gpu_device_count
    if get_gpu_device_count()<1:raise RuntimeError('Нет NVIDIA GPU для CatBoost')
    return CatBoostRegressor(iterations=trees,depth=7,learning_rate=.05,loss_function='MAE',eval_metric='MAE',
            task_type='GPU',devices='0',cat_features=['route_id'],random_seed=42,
            allow_writing_files=False,verbose=100,thread_count=min(4,os.cpu_count() or 1))


def save_model_atomic(model,path):
    tmp=path.with_suffix('.partial');model.save_model(str(tmp),format='cbm');tmp.replace(path)


def train_missing_evaluation():
    for h in (1,3,7):
        directory=ROOT/'evaluation'/f'h{h}';directory.mkdir(parents=True,exist_ok=True)
        if (directory/'complete.json').exists():continue
        data=pd.read_parquet(build_features(h));cols=columns_for(data,'control')
        train=data.split.eq('train');tune=data.split.eq('tuning');evaluate=~train
        model=make_catboost(MAX_TREES);started=time.monotonic()
        model.fit(data.loc[train,cols],data.loc[train,'y']/TARGET_SCALE,
                  eval_set=(data.loc[tune,cols],data.loc[tune,'y']/TARGET_SCALE),
                  early_stopping_rounds=EARLY_STOPPING,use_best_model=True)
        frame=data.loc[evaluate,['cutoff','route_id','timestamp','y','split','profile28','slot_daily7','slot_weekly4']].copy()
        frame['horizon']=h;frame['prediction']=np.maximum(model.predict(data.loc[evaluate,cols])*TARGET_SCALE,0)
        save_model_atomic(model,directory/'model.cbm');write_parquet(directory/'predictions.parquet',frame)
        write_json(directory/'complete.json',dict(horizon=h,variant='control',trees=model.tree_count_,features=cols,
                   seconds=time.monotonic()-started,fit_labels_before=str(LAG_VALIDATION_FIT_BEFORE)))
        print(f'evaluation h{h}: {(time.monotonic()-started)/60:.1f} мин',flush=True)
        del model,data,frame;gc.collect()


def evaluation_item(h,variant):
    directory=ROOT/'evaluation'/f'h{h}' if h in (1,3,7) else Path(SEED_DIR)/f'h{h}'/variant
    info=json.loads((directory/'complete.json').read_text())
    frame=validate_frame(pd.read_parquet(directory/'predictions.parquet'),h,info)
    return frame,info


def raw_blend(frame):
    p=frame.prediction.to_numpy(dtype=np.float64)
    base=np.where(frame.horizon.to_numpy()==1,p,.7*p+.3*frame.profile28.to_numpy(dtype=np.float64))
    if not np.isfinite(base).all() or (base<0).any():raise ValueError('Некорректная смесь')
    return base


def fit_shift(frame):
    return float(np.mean(frame.y.to_numpy(dtype=np.float64)-raw_blend(frame)))


def score_pipeline():
    controls=[];selected=[];tree_counts={};rows=[]
    for h in range(1,9):
        a,_=evaluation_item(h,'control');b,info=evaluation_item(h,RECIPE[h])
        controls.append(a);selected.append(b);tree_counts[str(h)]=int(info['trees'])
    for label,pieces in [('control',controls),('selected',selected)]:
        frame=pd.concat(pieces,ignore_index=True)
        tune=frame.split.eq('tuning');shift=fit_shift(frame.loc[tune])
        frame['model_prediction']=frame.prediction
        frame['raw_blend']=raw_blend(frame)
        frame['prediction']=np.maximum(frame.raw_blend.to_numpy()+shift,0)
        for split,part in frame.groupby('split'):
            rows.append(dict(recipe=label,split=split,horizon=0,shift=shift,**metric(part,part.prediction)))
            for h,p in part.groupby('horizon'):rows.append(dict(recipe=label,split=split,horizon=h,shift=shift,**metric(p,p.prediction)))
        write_parquet(ROOT/f'{label}_evaluation.parquet',frame)
        if label=='selected':
            # Переоценка только после получения и сохранения независимой от refit оценки.
            selected_raw=pd.concat(selected,ignore_index=True)
            final_shift=fit_shift(selected_raw)
    report=pd.DataFrame(rows);report.to_csv(ROOT/'pipeline_metrics.csv',index=False)
    write_json(ROOT/'evaluation_report.json',dict(recipe={str(h):v for h,v in RECIPE.items()},
              metrics=rows,note='Holdout 18–31 октября ранее изучался; оценка сохранена до final refit.'))
    settings=dict(recipe={str(h):v for h,v in RECIPE.items()},tree_counts=tree_counts,
                  final_shift=final_shift,calibration='additive',calibration_rows=len(selected_raw),
                  calibration_start=str(selected_raw.cutoff.min()),calibration_end=str(selected_raw.cutoff.max()),
                  h1_catboost_weight=1.,later_catboost_weight=.7,later_profile28_weight=.3)
    write_json(ROOT/'submission_settings.json',settings)
    display(report[report.horizon.eq(0)])
    print('Состав зафиксирован по пилотам; holdout не переключает рецепт автоматически.',flush=True)
    return settings


def final_origin_set(h):
    eligible=np.arange(len(times)-h)
    train=eligible[int(np.floor((1-RECENT_FRACTION)*len(eligible))):]
    if not len(train) or times[train+h].max()>end:raise ValueError('Неверные labels full train')
    return train


def build_final_features(h):
    directory=ROOT/'final'/f'h{h}';directory.mkdir(parents=True,exist_ok=True)
    path=directory/'train.parquet';future=directory/'future.parquet'
    origins=final_origin_set(h)
    if not future.exists():
        frame=add_pilot_features(horizon_dataset(h,np.array([len(times)-1])),h,np.array([len(times)-1]))
        requests=test[test.timestamp.eq(end+h*STEP)]
        frame=requests.merge(frame,on=['route_id','timestamp'],validate='one_to_one',sort=False)
        if not frame.cutoff.eq(end).all():raise ValueError('Не совпали конкурсные ключи')
        write_parquet(future,frame)
    if not path.exists():
        temp=path.with_suffix('.partial');writer=None
        try:
            for start in range(0,len(origins),FEATURE_ORIGIN_BATCH):
                idx=origins[start:start+FEATURE_ORIGIN_BATCH]
                frame=add_pilot_features(horizon_dataset(h,idx),h,idx);frame['y']=target[idx+h].ravel()
                table=pa.Table.from_pandas(frame,preserve_index=False)
                if writer is None:writer=pq.ParquetWriter(temp,table.schema,compression='zstd')
                writer.write_table(table)
        finally:
            if writer is not None:writer.close()
        temp.replace(path)
    write_json(directory/'train_range.json',dict(first_cutoff=str(times[origins[0]]),last_cutoff=str(times[origins[-1]]),
               last_label=str(times[origins[-1]+h]),rows=len(origins)*len(routes),recent_fraction=RECENT_FRACTION,
               correlation_fit_before=str(end+STEP)))
    return path,future


def train_final(settings):
    global validation_lag_table
    if not (ROOT/'evaluation_report.json').exists():raise ValueError('Сначала сохраните оценку полного pipeline')
    validation_lag_table=calculate_dynamic_shoulder(target,hub_signal,times,routes,fit_before=end+STEP,
        max_lookback_hours=LAG_MAX_LOOKBACK_HOURS,min_pairs=LAG_MIN_PAIRS,min_correlation=LAG_MIN_CORRELATION)
    validation_lag_table.to_csv(ROOT/'status6_lags_final.csv',index=False)
    for h in range(1,9):
        directory=ROOT/'final'/f'h{h}'
        if (directory/'complete.json').exists():continue
        started=time.monotonic();path,future_path=build_final_features(h)
        data=pd.read_parquet(path);future=pd.read_parquet(future_path);cols=columns_for(data,RECIPE[h])
        model=make_catboost(settings['tree_counts'][str(h)])
        print(f'full h{h}: {len(data):,} строк, {len(cols)} признаков, {settings["tree_counts"][str(h)]} деревьев',flush=True)
        model.fit(data[cols],data.y/TARGET_SCALE)
        f=future[['id','cutoff','route_id','timestamp','profile28']].copy();f['horizon']=h
        f['prediction']=np.maximum(model.predict(future[cols])*TARGET_SCALE,0)
        if not np.isfinite(f.prediction).all():raise ValueError('Некорректный full-прогноз')
        save_model_atomic(model,directory/'model.cbm');write_parquet(directory/'predictions.parquet',f)
        write_json(directory/'complete.json',dict(horizon=h,variant=RECIPE[h],trees=model.tree_count_,
                  features=cols,seconds=time.monotonic()-started,rows=len(data)))
        del model,data,future,f;gc.collect()
        if not KEEP_FINAL_FEATURES:
            path.unlink();future_path.unlink()  # Только производные признаки завершённого fit.


def make_submission(settings):
    frame=pd.concat([pd.read_parquet(ROOT/'final'/f'h{h}'/'predictions.parquet') for h in range(1,9)],ignore_index=True)
    if len(frame)!=len(test) or not frame.id.is_unique or set(frame.horizon)!=set(range(1,9)):
        raise ValueError('Неполные full-прогнозы')
    expected=test.merge(frame[['id','route_id','timestamp']],on='id',suffixes=('_test','_pred'),validate='one_to_one',how='left')
    if not (expected.route_id_test.eq(expected.route_id_pred).all() and expected.timestamp_test.eq(expected.timestamp_pred).all()):
        raise ValueError('Конкурсные ключи не совпали')
    frame['y_pred']=np.maximum(raw_blend(frame)+settings['final_shift'],0)
    out=test[['id']].merge(frame[['id','y_pred']],on='id',how='left',sort=False,validate='one_to_one')
    if not out.id.equals(test.id) or not np.isfinite(out.y_pred).all() or (out.y_pred<0).any():
        raise ValueError('Неверный порядок/значения submission')
    temp=ROOT/'submission.partial';out.to_csv(temp,index=False);temp.replace(ROOT/'submission.csv')
    write_json(ROOT/'submission_check.json',dict(rows=len(out),unique_ids=bool(out.id.is_unique),
               finite=True,nonnegative=True,original_order=True,sha256=digest(ROOT/'submission.csv')))
    print('ГОТОВО:',ROOT/'submission.csv',flush=True)
    return out
