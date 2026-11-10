META_CONFIGS = [dict(depth=3,iterations=300,learning_rate=0.03,l2_leaf_reg=50,loss_function='MAE'),
                dict(depth=4,iterations=450,learning_rate=0.03,l2_leaf_reg=30,loss_function='MAE'),
                dict(depth=5,iterations=900,learning_rate=0.03,l2_leaf_reg=50,loss_function='MAE'),
                dict(depth=3,iterations=300,learning_rate=0.03,l2_leaf_reg=50,loss_function='RMSE')]


def source_frame(stage):
    return pd.concat([pd.read_parquet(SOURCE_ROOT/f'{stage}_h{h}_cat_k8.parquet') for h in HORIZONS],ignore_index=True)


def meta_frame(raw):
    blocks=[]
    for h in HORIZONS:
        part=raw[raw.h==h]
        cutoffs=pd.DatetimeIndex(part.cutoff.unique()).sort_values()
        origins=TIMES.get_indexer(cutoffs)
        if (origins<0).any():
            raise ValueError('Неизвестные cutoff')
        x,names=features(h,origins,META_SEGMENTATION)
        data=pd.DataFrame(x,columns=names)
        data['cutoff']=np.repeat(cutoffs,len(ROUTES))
        data['route_id']=np.tile(ROUTES,len(cutoffs))
        data['h']=h
        data['timestamp']=data.cutoff+h*STEP
        data=data.merge(part[['route_id','cutoff','h','y','y_pred']],on=['route_id','cutoff','h'],how='left',validate='one_to_one',sort=False)
        indices=pd.Index(ROUTES).get_indexer(data.route_id)
        if (indices<0).any() or data[['y','y_pred']].isna().any().any():
            raise ValueError('Нарушено сопоставление метапризнаков')
        data['scale']=META_SEGMENTATION['scale'][indices]
        data['group']=META_LABELS[indices]
        data['raw_normalized']=data.y_pred/data.scale
        data['route_code']=data.route_id.astype(str)
        data['horizon_code']=str(h)
        blocks.append(data)
    return pd.concat(blocks,ignore_index=True)


def meta_columns(frame):
    return [c for c in frame if c not in ('route_id','cutoff','timestamp','h','y','y_pred','y_corrected','y_calibrated','scale','group')]


def meta_fit(frame,k,params):
    models={}
    labels=np.zeros(len(frame),dtype=int) if k==1 else frame.group.to_numpy()
    columns=meta_columns(frame)
    target=((frame.y-frame.y_pred)/frame.scale).to_numpy()
    weights=(frame.scale/frame.scale.mean()).to_numpy()
    for group in np.unique(labels):
        selected=np.ones(len(frame),dtype=bool) if group==-1 else labels==group
        model=CatBoostRegressor(**params,task_type='GPU',devices='0',thread_count=THREADS,random_seed=SEED,
                                cat_features=['route_code','horizon_code'],verbose=False,allow_const_label=True,allow_writing_files=False)
        model.fit(frame.loc[selected,columns],target[selected],sample_weight=weights[selected])
        models[int(group)]=model
    return dict(k=k,params=params,columns=columns,models=models)


def meta_predict(model,frame):
    if 'ensemble' in model:
        return .5*meta_predict(model['ensemble'][0],frame)+.5*meta_predict(model['ensemble'][1],frame)
    labels=np.zeros(len(frame),dtype=int) if model['k']==1 else frame.group.to_numpy()
    correction=np.empty(len(frame),dtype=np.float64)
    for group,estimator in model['models'].items():
        selected=labels==group
        correction[selected]=estimator.predict(frame.loc[selected,model['columns']])
    return np.maximum(frame.y_pred.to_numpy()+correction*frame.scale.to_numpy(),0)


def balanced_ratio(frame,pred):
    return float(np.clip(frame.y.sum()/max(float(pred.sum()),1.),*SCALE_BOUNDS))


def fit_selected_meta(frame,k,index):
    if index==-1:
        return dict(ensemble=[meta_fit(frame,k,META_CONFIGS[i]) for i in (1,2)])
    return meta_fit(frame,k,META_CONFIGS[index])


def online_meta_offset(model,history,query):
    known=pd.concat([history,query],ignore_index=True)
    unshifted=meta_predict(model,known)
    errors=known.y.to_numpy()-unshifted
    raw=meta_predict(model,query)
    output=np.empty(len(query),dtype=np.float64)
    audit=[]
    for cutoff in sorted(query.cutoff.unique()):
        cutoff=pd.Timestamp(cutoff)
        available=(known.timestamp<cutoff-pd.Timedelta(hours=4)).to_numpy()
        if not available.any():
            raise ValueError('Нет доступной истории для rolling-поправки')
        age=(cutoff-known.loc[available,'cutoff']).dt.total_seconds().to_numpy()/86400
        offset=float(np.average(errors[available],weights=np.exp2(-age/7.)))
        selected=query.cutoff.eq(cutoff).to_numpy()
        output[selected]=np.maximum(raw[selected]+offset,0)
        audit.append(dict(cutoff=str(cutoff),fit_through=str(known.loc[available,'timestamp'].max()),offset=offset))
    return output,audit


def assess_meta(model,fit,select,k,index):
    fit_pred=meta_predict(model,fit)
    query_pred=meta_predict(model,select)
    ratio=balanced_ratio(fit,fit_pred)
    rolling,audit=online_meta_offset(model,fit,select)
    pd.DataFrame(audit).to_csv(RUN/f'meta_rolling_validation_k{k}_config{index}.csv',index=False)
    for stage,frame,pred in [('calibration_fit',fit,fit_pred),('calibration_select',select,query_pred)]:
        frame[['route_id','cutoff','timestamp','h','y','y_pred','scale','group']].assign(meta_unshifted=pred).to_parquet(RUN/f'meta_oof_{stage}_k{k}_config{index}.parquet',index=False)
    fit_offset=float(np.mean(fit.y.to_numpy()-fit_pred))
    modes=[('raw',fit_pred,query_pred,1.),('ratio',fit_pred*ratio,query_pred*ratio,ratio),
           ('rolling7',np.maximum(fit_pred+fit_offset,0),rolling,1.)]
    rows=[]
    for mode,fit_out,select_out,factor in modes:
        for period,frame,pred in [('train',fit,fit_out),('calibration_select',select,select_out)]:
            rows.append(dict(k=k,config=index,mode=mode,balance=mode=='ratio',period=period,factor=factor,**metrics(frame.y.to_numpy(),pred)))
    display(pd.DataFrame(rows))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    return rows


def initialize_meta():
    global SOURCE_ROOT,META_SEGMENTATION,META_LABELS
    load_inputs()
    found=[p.parent for p in Path('/kaggle/input').rglob('calibration_frozen.json') if p.parent.name=='cluster_experiment_3387cb53bcd07507']
    if len(found)!=1:
        raise ValueError('Подключите Output ноутбука clustering-calibration, версия 367876306')
    SOURCE_ROOT=found[0]
    source_config=json.loads((SOURCE_ROOT/'manifest.json').read_text())
    config=json.loads((RUN/'manifest.json').read_text())
    if source_config['train_hash']!=config['train_hash'] or source_config['test_hash']!=config['test_hash']:
        raise ValueError('История отличается от базового запуска')
    META_SEGMENTATION=segment_routes(VAL_START,'meta_groups')
    group_table=pd.read_csv(SOURCE_ROOT/'calibration_groups.csv').set_index('route_id').reindex(ROUTES)
    if group_table.calibration_group.isna().any():
        raise ValueError('Нет фиксированных групп')
    META_LABELS=group_table.calibration_group.to_numpy(dtype=int)
    print('Источник OOF и полностью переобученных баз:',SOURCE_ROOT,flush=True)


def run_meta():
    initialize_meta()
    fit=meta_frame(source_frame('calibration_fit'))
    select=meta_frame(source_frame('calibration_select'))
    if fit.cutoff.max()>=select.cutoff.min():
        raise ValueError('Пересечение fit/select')
    rows=[]
    for k in (1,max(CLUSTERS)):
        components={}
        for index,params in enumerate(META_CONFIGS):
            model=meta_fit(fit,k,params)
            rows.extend(assess_meta(model,fit,select,k,index))
            if index in (1,2):
                components[index]=model
            del model
            gc.collect()
        rows.extend(assess_meta(dict(ensemble=[components[1],components[2]]),fit,select,k,-1))
        del components
        gc.collect()
    pd.DataFrame(rows).to_csv(RUN/'learned_residual_selection.csv',index=False)
    selected=min([r for r in rows if r['period']=='calibration_select'],key=lambda r:r['score'])
    write_json(RUN/'learned_residual_frozen.json',selected)
    all_val=pd.concat([fit,select],ignore_index=True)
    model=fit_selected_meta(all_val,selected['k'],selected['config'])
    ratio=balanced_ratio(all_val,meta_predict(model,all_val)) if selected['mode']=='ratio' else 1.
    write_json(RUN/'learned_residual_test_factor.json',dict(factor=ratio))
    test=meta_frame(source_frame('test'))
    if selected['mode']=='rolling7':
        test_prediction,audit=online_meta_offset(model,all_val,test)
        pd.DataFrame(audit).to_csv(RUN/'learned_residual_test_rolling_audit.csv',index=False)
    else:
        test_prediction=meta_predict(model,test)*ratio
    report=[]
    for period,frame in [('train',all_val),('test',test)]:
        for method,pred in [('raw',frame.y_pred.to_numpy()),('learned_residual',test_prediction if period=='test' else meta_predict(model,frame)*ratio)]:
            report.append(dict(period=period,method=method,protocol='prequential' if method=='learned_residual' and selected['mode']=='rolling7' and period=='test' else 'frozen_or_in_sample',**metrics(frame.y.to_numpy(),pred)))
    pd.DataFrame(report).to_csv(RUN/'learned_residual_independent_metrics.csv',index=False)
    display(pd.DataFrame(report))
    print(pd.DataFrame(report).to_string(index=False),flush=True)
    print('Оценка test получена до полного переобучения. Этот test уже изучался в проекте. Для rolling7 это последовательная оценка: используются только метки timestamp < cutoff − 4 часа; правило зафиксировано до test.',flush=True)
    test['y_corrected']=test_prediction
    test[['route_id','cutoff','h','y','y_pred','y_corrected']].assign(meta_unshifted=meta_predict(model,test)).to_parquet(RUN/'learned_residual_test_predictions.parquet',index=False)
    all_oof=pd.concat([all_val,test],ignore_index=True)
    final_model=fit_selected_meta(all_oof,selected['k'],selected['config'])
    final_ratio=balanced_ratio(all_oof,meta_predict(final_model,all_oof)) if selected['mode']=='ratio' else 1.
    final_report=pd.DataFrame(report+[dict(period='full_in_sample_meta',method='learned_residual',**metrics(all_oof.y.to_numpy(),meta_predict(final_model,all_oof)*final_ratio))])
    final_report.to_csv(RUN/'learned_residual_final_report.csv',index=False)
    display(final_report)
    print(final_report.to_string(index=False),flush=True)
    print('Train/test выше относятся к прежней оценочной модели, до полного refit метамоделей.',flush=True)
    joblib.dump(dict(model=final_model,factor=final_ratio,mode=selected['mode'],groups=META_LABELS,segmentation=META_SEGMENTATION),RUN/'learned_residual_final.joblib',compress=3)
    raw_submission=pd.read_csv(SOURCE_ROOT/'submission_raw.csv')
    query=REQUESTS.merge(raw_submission,on='id',how='left',validate='one_to_one')
    query['h']=((query.timestamp-TIMES[-1])/STEP).astype(int)
    query['cutoff']=TIMES[-1]
    query['y']=0.
    final_frame=meta_frame(query)
    if selected['mode']=='rolling7':
        final_frame['y_corrected'],audit=online_meta_offset(final_model,all_oof,final_frame)
        pd.DataFrame(audit).to_csv(RUN/'learned_residual_final_rolling_audit.csv',index=False)
    else:
        final_frame['y_corrected']=meta_predict(final_model,final_frame)*final_ratio
    submission=REQUESTS.merge(final_frame[['route_id','cutoff','h','y_corrected']].assign(timestamp=lambda d:d.cutoff+d.h*STEP)[['route_id','timestamp','y_corrected']],on=['route_id','timestamp'],how='left',validate='one_to_one',sort=False)
    if not submission.id.equals(REQUESTS.id) or not np.isfinite(submission.y_corrected).all() or (submission.y_corrected<0).any():
        raise ValueError('Некорректный submission')
    submission[['id','y_corrected']].rename(columns={'y_corrected':'y_pred'}).to_csv(RUN/'submission.csv',index=False)
    print('Базовые модели взяты из полного refit исходного запуска на той же всей истории; метамодели переобучены на всех доступных OOF val/test.',flush=True)
    print('Сабмит:',RUN/'submission.csv',flush=True)
