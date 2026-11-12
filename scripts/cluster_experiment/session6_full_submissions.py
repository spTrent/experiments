def make_volume_inputs(frame, h, model_prediction):
    result = frame[['cutoff', 'route_id', 'timestamp', 'profile28', *anchor_columns]].copy()
    result['horizon'] = h
    result['model_prediction'] = model_prediction
    result['raw_blend'] = model_prediction if h == 1 else .7 * model_prediction + .3 * result.profile28
    return result


def append_refit_metrics(rows, component, horizon, group, y, pred):
    row = dict(component=component, horizon=horizon, group=group, period='full_refit_in_sample', **metric(y, pred))
    rows.append(row)
    display(pd.DataFrame([row]))
    pd.DataFrame(rows).to_csv(RUN / 'full_refit_metrics.csv', index=False)


def train_full_bases(segmentation):
    route_groups = pd.Series(segmentation['labels'][8], index=routes)
    positions = np.arange(len(times))
    minutes = times.hour * 60 + times.minute
    future_origin = np.array([len(times) - 1])
    model_dir = RUN / 'models'
    model_dir.mkdir(exist_ok=True)
    outputs, rows = [], []
    for h in range(1, 9):
        eligible = positions[positions + h < len(times)]
        cat_columns_h = cat_columns + (SEASONAL_FEATURES if h in (5, 6, 8) else [])
        cat = CatBoostRegressor(iterations=cat_trees[h - 1], depth=7, learning_rate=.05, loss_function='MAE',
                                task_type='GPU', devices='0', cat_features=['route_id'], random_seed=42,
                                allow_writing_files=False, verbose=250, thread_count=4)
        train = pd.concat([make_cat_frame(eligible[start:start + 32], h)[cat_columns_h]
                           for start in range(0, len(eligible), 32)], ignore_index=True)
        cat.fit(train, target[eligible + h].ravel() / 100000)
        cat.save_model(str(model_dir / f'catboost_full_h{h}.cbm'))
        del train
        gc.collect()
        eval_origins = eligible[minutes[eligible] == 630]
        check = make_cat_frame(eval_origins, h)
        pred = np.maximum(cat.predict(check[cat_columns_h]) * 100000, 0)
        append_refit_metrics(rows, 'CatBoost', h, 'all', target[eval_origins + h].ravel(), pred)
        query = make_cat_frame(future_origin, h)
        result = make_volume_inputs(query, h, np.maximum(cat.predict(query[cat_columns_h]) * 100000, 0))
        del cat, check, query, pred
        gc.collect()
        lgb_origins = eligible[(eligible >= 336) & np.isin(minutes[eligible], [570, 600, 630, 660, 690])]
        train = session_shape_features(lgb_origins, h, target, level24)
        query = session_shape_features(future_origin, h, target, level24)
        if not result[keys].reset_index(drop=True).equals(query[keys].astype(result[keys].dtypes).reset_index(drop=True)):
            raise ValueError('Ключи CatBoost и LightGBM отличаются')
        config = meta[str(h)]
        columns = config['columns']
        if set(columns) & {'y', 'timestamp', 'cutoff'}:
            raise ValueError('Метка входит в признаки LightGBM')
        train_scale = train.scale.to_numpy(dtype=float)
        train_groups = train.route_id.map(route_groups).to_numpy()
        query_groups = query.route_id.map(route_groups).to_numpy()
        global_shape = np.empty(len(query))
        cluster_shape = np.empty(len(query))
        for group in ['global', *sorted(set(train_groups) - {-1})]:
            all_routes = group == 'global'
            fit = np.ones(len(train), dtype=bool) if all_routes else train_groups == group
            forecast = np.ones(len(query), dtype=bool) if all_routes else query_groups == group
            model = lgb.LGBMRegressor(**dict(config['params'], n_estimators=config['trees'], n_jobs=4))
            model.fit(train.loc[fit, columns], train.y.to_numpy()[fit] / train_scale[fit],
                      sample_weight=train_scale[fit] / train_scale.mean())
            model.booster_.save_model(str(model_dir / f'lightgbm_full_h{h}_{group}.txt'))
            check = fit & train.cutoff.dt.hour.eq(10).to_numpy() & train.cutoff.dt.minute.eq(30).to_numpy()
            pred = np.maximum(model.predict(train.loc[check, columns]) * train_scale[check], 0)
            append_refit_metrics(rows, 'LightGBM', h, group, train.y.to_numpy()[check], pred)
            shape = np.maximum(model.predict(query.loc[forecast, columns]) * query.scale.to_numpy()[forecast], 0)
            if all_routes:
                global_shape[:] = shape
                cluster_shape[:] = shape
            else:
                cluster_shape[forecast] = shape
            del model, pred, shape
        outputs.append(result.assign(global_shape=global_shape, clustered_shape=cluster_shape))
        del train, query
        gc.collect()
    return pd.concat(outputs, ignore_index=True), rows


def historical_residual_inputs():
    history = pd.read_parquet(RESIDUAL_INPUTS['selected_evaluation.parquet'])
    anchors = pd.read_parquet(RESIDUAL_INPUTS['anchor_features.parquet'], columns=['cutoff', 'route_id', *anchor_columns])
    history = history.merge(anchors, on=['cutoff', 'route_id'], how='left', validate='many_to_one')
    source = times.get_indexer(pd.DatetimeIndex(history.timestamp))
    indices = pd.Index(routes).get_indexer(history.route_id)
    if history.duplicated(keys).any() or history[anchor_columns].isna().any().any() or (source < 0).any() or (indices < 0).any():
        raise ValueError('Некорректные исторические OOF-прогнозы')
    if not history.timestamp.eq(history.cutoff + pd.to_timedelta(history.horizon * 30, unit='m')).all() or not np.array_equal(history.y.to_numpy(), target[source, indices]):
        raise ValueError('OOF-факты или время не совпадают с историей')
    if len(history) != 28 * 8 * len(routes) or history.timestamp.max() >= times[-1] - pd.Timedelta(hours=4):
        raise ValueError('Неожиданный период OOF для моделей поправки')
    return history


def refit_volume(history, future, rows):
    fit = residual_features(history)
    query = residual_features(future)
    known = history.timestamp < times[-1] - pd.Timedelta(hours=4)
    age = (times[-1] - history.loc[known, 'cutoff']).dt.total_seconds().to_numpy() / 86400
    target_residual = (history.y.to_numpy() - history.raw_blend.to_numpy()) / 100000
    forecasts, audit = [], []
    for depth, trees, l2 in [(4, 450, 30), (5, 900, 50)]:
        model = CatBoostRegressor(iterations=trees, depth=depth, learning_rate=.03, l2_leaf_reg=l2,
                                  loss_function='MAE', cat_features=['route_id', 'horizon'], random_seed=42,
                                  task_type='GPU', devices='0', thread_count=4, allow_writing_files=False, verbose=False)
        model.fit(fit, target_residual)
        model.save_model(str(RUN / 'models' / f'residual_full_depth{depth}.cbm'))
        in_sample = history.raw_blend.to_numpy() + model.predict(fit) * 100000
        errors = history.y.to_numpy() - in_sample
        offset = float(np.average(errors[known], weights=np.exp2(-age / 7)))
        append_refit_metrics(rows, f'Residual depth {depth}', 'all', 'all', history.y.to_numpy(), np.maximum(in_sample, 0))
        forecasts.append(np.maximum(future.raw_blend.to_numpy() + model.predict(query) * 100000 + offset, 0))
        audit.append(dict(depth=depth, trees=trees, ewma_offset=offset, fit_through=str(history.timestamp.max()),
                          bias_fit_through=str(history.loc[known, 'timestamp'].max())))
    pd.DataFrame(audit).to_csv(RUN / 'volume_refit_audit.csv', index=False)
    return np.mean(forecasts, axis=0)


def write_submission(requests, forecast, prediction, filename):
    output = requests.merge(forecast[['route_id', 'timestamp']].assign(y_pred=prediction), on=['route_id', 'timestamp'],
                            how='left', sort=False, validate='one_to_one')
    if len(output) != 8000 or not output.id.is_unique or not output.id.equals(requests.id) or not np.isfinite(output.y_pred).all() or output.y_pred.lt(0).any():
        raise ValueError('Некорректный submission')
    output[['id', 'y_pred']].to_csv(RUN / filename, index=False)


def run_full_submissions():
    global start_tune
    start_tune = times[-1] + STEP
    historical_scores = pd.DataFrame(REFERENCE_SCORES)
    historical_scores.to_csv(RUN / 'test_before_full_refit.csv', index=False)
    print('Архивная независимая от полного refit оценка test 18–31 октября. Этот период ранее изучался.')
    display(historical_scores)
    if get_gpu_device_count() < 1:
        raise RuntimeError('Включите GPU для CatBoost')
    requests = pd.read_parquet(TEST_INPUT, columns=['id', 'route_id', 'timestamp'])
    requests['timestamp'] = pd.to_datetime(requests.timestamp)
    history = historical_residual_inputs()
    with threadpool_limits(limits=4):
        prepare_history(values)
        segmentation = segment_routes(start_tune, 'full')
        future, rows = train_full_bases(segmentation)
        future['volume_model'] = refit_volume(history, future, rows)
    weight = float(meta['weight'])
    for shape, filename in [('global_shape', 'submission.csv'), ('clustered_shape', 'submission_clusters.csv')]:
        adjusted = normalize_shape(future, shape, 'volume_model')
        pred = (1 - weight) * future.volume_model.to_numpy() + weight * adjusted
        sums = future.assign(pred=pred).groupby(group_keys)[['pred', 'volume_model']].sum()
        if not np.allclose(sums.pred, sums.volume_model, rtol=1e-10, atol=1e-5):
            raise ValueError('Нарушено сохранение объёма горизонта')
        future[shape + '_calibrated'] = pred
        write_submission(requests, future, pred, filename)
    future.to_parquet(RUN / 'future_predictions.parquet', index=False)
    (RUN / 'protocol.json').write_text(json.dumps(dict(fit_end_exclusive=str(start_tune),
        shape_weight=weight, selected_shape='global', local_test_evaluation_before_full_refit=True,
        volume_catboost_starts='all_eligible', shape_lightgbm_starts=[570, 600, 630, 660, 690],
        residual_fit_scope='archived_raw_oof_oct04_oct31', new_local_test_score=False), ensure_ascii=False, indent=2))
    with zipfile.ZipFile(RUN.with_suffix('.zip'), 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=2) as archive:
        for path in sorted(RUN.rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(RUN.parent))
    print('Готовые файлы', RUN / 'submission.csv', RUN / 'submission_clusters.csv')
    print('ZIP', RUN.with_suffix('.zip'))
