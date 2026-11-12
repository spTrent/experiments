def restored_session6():
    from catboost import CatBoostRegressor
    base = EXACT_ROOT
    bundle = base / 'stepa_checkpoint_0328_bundle'
    folder = ARTIFACTS / 'session6_restored'
    folder.mkdir(exist_ok=True)
    cached = folder / 'predictions.parquet'
    if cached.is_file():
        return pd.read_parquet(cached)
    scope = {'__name__': '__restore__'}
    exec(compile((bundle / 'code/baseline_definitions.py').read_text(), 'baseline', 'exec'), scope)
    scope.update(ROOT=folder, TRAIN_PATH=SESSION_DATA / 'train_solo_track.parquet', TEST_PATH=SESSION_DATA / 'test_solo_track.parquet')
    scope['load_panel']()
    for name in ['baseline_common_features.py', 'baseline_correlations.py', 'baseline_horizon.py']:
        exec(compile((bundle / 'code' / name).read_text(), name, 'exec'), scope)
    boundary = pd.Timestamp('2025-10-04') if SUBMISSION_SCOPE == 'train_val' else FULL_BOUNDARY
    lag_name = 'status6_lags_train.csv' if SUBMISSION_SCOPE == 'train_val' else 'status6_lags_final.csv'
    scope['validation_lag_table'] = pd.read_csv(base / 'server_output/original_catboost_run' / lag_name)
    def features(h, origins):
        return scope['add_pilot_features'](scope['horizon_dataset'](h, origins), h, origins)
    hist = pd.read_parquet(bundle / 'inputs/selected_evaluation.parquet').sort_values(KEYS).reset_index(drop=True)
    origins = TIMES.get_indexer(pd.DatetimeIndex(hist.cutoff.unique()))
    query_origins = np.concatenate([origins, FUTURE_ORIGIN])
    outputs = []
    for h in HORIZONS:
        path = folder / f'h{h}.parquet'
        if path.is_file():
            outputs.append(pd.read_parquet(path))
            continue
        metadata = json.loads((bundle / 'models/baseline' / f'h{h}/complete.json').read_text())
        columns = metadata['features']
        saved = base / 'server_output/original_catboost_run/evaluation' / f'h{h}/model.cbm'
        if h == 4:
            saved = RECOVERED_ROOT / 'evaluation/h4/model.cbm'
        model = CatBoostRegressor()
        if SUBMISSION_SCOPE == 'train_val' and saved.is_file():
            model.load_model(str(saved))
        else:
            eligible = np.flatnonzero(TIMES[:len(TIMES)-h] + h * STEP < boundary)
            eligible = eligible[len(eligible)//4:]
            train = pd.concat([features(h, eligible[i:i+32])[columns] for i in range(0, len(eligible), 32)], ignore_index=True)
            labels = VALUES[TARGET][eligible+h].ravel()
            model = CatBoostRegressor(iterations=metadata['trees'], depth=7, learning_rate=.05,
                loss_function='MAE', task_type='GPU', devices='0', cat_features=['route_id'],
                random_seed=42, verbose=250, thread_count=4, allow_writing_files=False)
            model.fit(train, labels/100000.)
            train_pred = np.maximum(model.predict(train)*100000., 0)
            check_origins = origins[TIMES[origins]>=LOCAL_TEST_START]
            check = features(h, check_origins)
            check_pred = np.maximum(model.predict(check[columns])*100000., 0)
            refit_report('session6_catboost', boundary, h, 'all', 'test', labels, train_pred,
                VALUES[TARGET][check_origins+h].ravel() if SUBMISSION_SCOPE=='train_val' else [],
                check_pred if SUBMISSION_SCOPE=='train_val' else [])
            del train, labels, train_pred, check
        if model.feature_names_ != columns:
            raise ValueError(f'Порядок признаков Session6 h{h} отличается')
        model.save_model(str(folder/f'h{h}.cbm'))
        frame = features(h, query_origins)
        frame['horizon'] = h
        frame['model_prediction'] = np.maximum(model.predict(frame[columns])*100000., 0)
        frame['raw_blend'] = frame.model_prediction if h==1 else .7*frame.model_prediction+.3*frame.profile28
        frame['y'] = np.nan
        known = query_origins+h < len(TIMES)
        frame.loc[np.repeat(known,len(ROUTES)), 'y'] = VALUES[TARGET][query_origins[known]+h].ravel()
        frame['split'] = np.where(frame.cutoff < LOCAL_TEST_START, 'tuning', np.where(frame.cutoff < LOCAL_TEST_END, 'holdout', 'submission'))
        save_parquet(path, frame)
        outputs.append(frame)
        del frame, model
        gc.collect()
    all_predictions = pd.concat(outputs,ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    reference = hist[KEYS+['model_prediction']].merge(all_predictions[KEYS+['model_prediction']],on=KEYS,validate='one_to_one',suffixes=('_reference','_restored'))
    reference['absolute_difference'] = abs(reference.model_prediction_reference-reference.model_prediction_restored)
    reference.groupby('horizon').absolute_difference.agg(['mean','max']).to_csv(folder/'base_recovery_comparison.csv')
    display(reference.groupby('horizon').absolute_difference.agg(['mean','max']))
    if SUBMISSION_SCOPE == 'full_history':
        previous = pd.concat([pd.read_parquet(SUBMISSION_ROOT/'train_val/session6_restored'/f'h{h}.parquet') for h in HORIZONS],ignore_index=True)
        all_predictions = pd.concat([previous.loc[previous.cutoff.lt(LOCAL_TEST_END)], all_predictions.loc[all_predictions.cutoff.eq(FUTURE_CUTOFF)]],ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    anchors = all_predictions[['cutoff','route_id',*verification_scope['strategy'].shallow.ANCHOR_COLUMNS]].drop_duplicates(['cutoff','route_id'])
    (folder/'inputs').mkdir(exist_ok=True)
    save_parquet(folder/'inputs/anchor_features.parquet',anchors)
    strategy = verification_scope['strategy']
    residual_train = all_predictions.loc[all_predictions.y.notna() & all_predictions.timestamp.lt(FULL_BOUNDARY)].copy()
    calibrator = strategy.fit(residual_train, folder)
    joblib.dump(calibrator,folder/'calibrator.joblib')
    volume = np.maximum(strategy.predict(calibrator, all_predictions, folder),0)
    if SUBMISSION_SCOPE == 'train_val':
        cutoff = pd.Timestamp('2025-10-25 10:30')
        prefix = all_predictions.loc[all_predictions.cutoff.le(cutoff)].copy()
        current = prefix.cutoff.eq(cutoff)
        prefix.loc[current, 'y'] = np.nan
        prefix_volume = strategy.predict(calibrator, prefix, folder)[current.to_numpy()]
        if not np.allclose(prefix_volume, volume[all_predictions.cutoff.eq(cutoff)], rtol=1e-12, atol=1e-6):
            raise ValueError('Session6 использует текущие или будущие цели в EWMA')
    light = verification_scope['session6_verified_light'][KEYS+['core_reproduced']].copy()
    from predict_round100 import Predictor
    predictor = Predictor(base/'session6_bundle',bundle/'data',base/'session6_bundle/reference_0328.csv',WORK_ROOT/'lightgbm_future_cache')
    future_shape = predictor.keys[['timestamp','route_id']].copy()
    future_shape['core_reproduced'] = predictor.core(86)
    future_shape['cutoff'] = FUTURE_CUTOFF
    future_shape['horizon'] = ((future_shape.timestamp-FUTURE_CUTOFF)/STEP).astype(int)
    light = pd.concat([light, future_shape[KEYS+['core_reproduced']]],ignore_index=True)
    result = all_predictions[KEYS+['y']].assign(volume=volume).merge(light,on=KEYS,validate='one_to_one')
    totals = result.groupby(GROUP_KEYS)[['volume','core_reproduced']].transform('sum')
    weight = json.loads((base/'session6_bundle/manifest.json').read_text())['shape_weight']
    result['session6'] = (1-weight)*result.volume + weight*result.core_reproduced*totals.volume/totals.core_reproduced
    if SUBMISSION_SCOPE=='train_val':
        reference = pd.read_parquet(base/'validation_predictions.parquet')
        comparison = reference.merge(result[KEYS+['session6']],on=KEYS,validate='one_to_one')
        comparison['difference'] = comparison.session6-comparison.session_prediction
        save_parquet(folder/'session6_reference_comparison.parquet',comparison)
        display(pd.DataFrame([dict(variant=name,**metric(comparison.y,comparison[col])) for name,col in [('original','session_prediction'),('restored','session6')]]))
    result = result.loc[result.cutoff.isin(TIMES[QUERY_ORIGINS])]
    save_parquet(cached,result)
    return result
