def session6_oof():
    global start_tune, start_test, eval_origins, out
    out = ARTIFACTS / 'session6'
    (out / 'models' / 'early_anchor').mkdir(parents=True, exist_ok=True)
    path = out / 'oof.parquet'
    if path.is_file():
        return pd.read_parquet(path)
    start_tune = OOF_START - pd.Timedelta(days=7)
    eval_origins = positions[(times >= start_tune) & (times < LOCAL_TEST_START) & (minutes == 630)]
    start_test = start_tune
    prepare_history(values)
    base = pd.concat([base_predictions('early_anchor', h) for h in HORIZONS], ignore_index=True)
    base = base.sort_values(keys).reset_index(drop=True)
    blocks = []
    for start, stop in OOF_WINDOWS:
        start_test = start
        block = base.loc[base.cutoff.lt(stop)].reset_index(drop=True)
        prediction = finish_pipeline(block, 'early_anchor', start)
        result = prediction.loc[prediction.cutoff.ge(start), keys + ['y', 'prediction', 'volume_model', 'shape']]
        blocks.append(result.rename(columns={'prediction': 'session6'}))
    result = pd.concat(blocks, ignore_index=True)
    save_parquet(path, result)
    return result


def fit_rich_cat(start, stop, split):
    if start >= LOCAL_TEST_START and not (ARTIFACTS / 'frozen_selection.json').is_file():
        raise ValueError('Сначала зафиксируйте выбор по val')
    folder = RUN / start.strftime('%Y%m%d')
    folder.mkdir(parents=True, exist_ok=True)
    outputs = [folder / f'h{h}.parquet' for h in HORIZONS]
    if not all(path.is_file() for path in outputs):
        with threadpool_limits(limits=THREADS):
            segmentation = segment_routes(start, start.strftime('%Y%m%d'))
        route_groups = segmentation['labels'][8]
        route_scale = segmentation['scale']
        origins = np.arange(len(TIMES) - 8)
        for h, path in zip(HORIZONS, outputs):
            if path.is_file():
                continue
            train_origins = np.flatnonzero(TIMES[:len(TIMES) - h] + h * STEP < start)
            query_origins = origins[(TIMES[origins] >= start) & (TIMES[origins] < stop) &
                                    (TIMES[origins].hour == 10) & (TIMES[origins].minute == 30)]
            x, columns = rich_matrix(h, train_origins, segmentation)
            query, query_columns = rich_matrix(h, query_origins, segmentation)
            if columns != query_columns:
                raise ValueError('Расходятся колонки CatBoost rich')
            y = VALUES[TARGET][train_origins + h]
            actual = VALUES[TARGET][query_origins + h]
            predictions = np.empty_like(actual, dtype=np.float64)
            spec = PARAMETERS['selected_cat'][str(h)]
            for group in np.unique(route_groups):
                query_routes = np.flatnonzero(route_groups == group)
                train_routes = np.arange(len(ROUTES)) if group == -1 else query_routes
                train_x = model_frame(x[:, train_routes].reshape(-1, len(columns)), columns, 'cat')
                train_y = y[:, train_routes].reshape(-1)
                train_scale = np.tile(route_scale[train_routes], len(train_origins))
                query_x = model_frame(query[:, query_routes].reshape(-1, len(columns)), columns, 'cat')
                model = CatBoostRegressor(**spec['params'], iterations=spec['trees'], loss_function='MAE',
                                          task_type='GPU', devices='0', thread_count=THREADS,
                                          random_seed=SEED, verbose=False, allow_writing_files=False,
                                          allow_const_label=True, bootstrap_type='Bayesian', cat_features=['route_code'])
                model.fit(train_x, train_y / train_scale, sample_weight=train_scale / route_scale.mean())
                train_pred = np.maximum(model.predict(train_x) * train_scale, 0)
                query_pred = np.maximum(model.predict(query_x).reshape(len(query_origins), -1) * route_scale[query_routes], 0)
                predictions[:, query_routes] = query_pred
                fit_report('cat_rich', start, h, int(group), split, train_y, train_pred, actual[:, query_routes], query_pred)
                del model, train_x, query_x, train_pred
                gc.collect()
            frame = pd.DataFrame(dict(cutoff=TIMES[query_origins].repeat(len(ROUTES)), route_id=np.tile(ROUTES, len(query_origins)),
                                      horizon=h, y=actual.ravel(), cat_rich=predictions.ravel()))
            save_parquet(path, frame)
            del x, query
            gc.collect()
    return pd.concat([pd.read_parquet(path) for path in outputs], ignore_index=True)


def rich_matrix(h, origins, segmentation):
    if not len(origins):
        raise ValueError('Пустой диапазон CatBoost rich')
    routes_count = len(ROUTES)
    matrix = None
    for start in range(0, len(origins), 32):
        batch = origins[start:start + 32]
        raw, names = features(h, batch, segmentation)
        if matrix is None:
            columns = names
            matrix = np.empty((len(origins), routes_count, len(columns)), dtype=np.float32)
        if names != columns:
            raise ValueError('Расходятся колонки батчей CatBoost rich')
        matrix[start:start + len(batch)] = raw.reshape(len(batch), routes_count, len(columns))
    return matrix, columns


def rf_data(h, origins, lag_table, spec):
    batches, parts = [], []
    for start in range(0, len(origins), 24):
        batch = origins[start:start + 24]
        frame = horizon_dataset(h, batch)
        frame['y'] = target[batch + h].ravel()
        frame = frame.loc[np.isfinite(frame.profile28) & np.isfinite(frame.y)].reset_index(drop=True)
        frame = add_correlation_features(frame, h, lag_table)
        matrix = transformed_frame(frame, model_columns(frame))
        columns = experiment_columns(list(matrix.columns), EXPERIMENTS[spec.experiment])
        batches.append(matrix[columns].replace([np.inf, -np.inf], np.nan))
        parts.append(frame[['cutoff', 'timestamp', 'route_id', 'y', 'profile28']].copy())
    return pd.concat(parts, ignore_index=True), pd.concat(batches, ignore_index=True)


def fit_rf(start, stop, split):
    if start >= LOCAL_TEST_START and not (ARTIFACTS / 'frozen_selection.json').is_file():
        raise ValueError('Сначала зафиксируйте выбор по val')
    folder = RUN / start.strftime('%Y%m%d')
    folder.mkdir(parents=True, exist_ok=True)
    outputs = [folder / f'h{h}.parquet' for h in HORIZONS]
    if not all(path.is_file() for path in outputs):
        lag_table = fit_lags(start)
        minute = times.hour * 60 + times.minute
        for h, path in zip(HORIZONS, outputs):
            if path.is_file():
                continue
            spec = selected.loc[h]
            if bool(spec.trimmed):
                raise ValueError('Для этого рецепта RF ожидается полная обучающая история')
            train_origins = np.flatnonzero(np.isin(minute, TRAIN_MINUTES) & (times + h * STEP < start))
            query_origins = np.flatnonzero((minute == 630) & (times >= start) & (times + h * STEP < stop))
            train, train_x = rf_data(h, train_origins, lag_table, spec)
            query, query_x = rf_data(h, query_origins, lag_table, spec)
            if bool(spec.route_median):
                median = train_x.groupby(train.route_id).median()
                train_x = train_x.fillna(median.reindex(train.route_id).set_axis(train_x.index))
                query_x = query_x.fillna(median.reindex(query.route_id).set_axis(query_x.index))
            imputer = SimpleImputer(strategy='median', keep_empty_features=True)
            train_matrix = imputer.fit_transform(train_x).astype(np.float32, copy=False)
            query_matrix = imputer.transform(query_x).astype(np.float32, copy=False)
            del train_x, query_x
            model = fit_forest(train, train_matrix, best_params[h])
            train_pred = predict_matrix(model, train, train_matrix)
            query_pred = predict_matrix(model, query, query_matrix)
            fit_report('rf', start, h, 'all', split, train.y, train_pred, query.y, query_pred)
            save_parquet(path, query[['cutoff', 'route_id', 'y']].assign(horizon=h, rf=query_pred))
            del model, train_matrix, query_matrix, train, query, train_pred, query_pred
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
    return pd.concat([pd.read_parquet(path) for path in outputs], ignore_index=True)


class CleanXgbBatches(xgb.DataIter):
    def __init__(self, matrix, labels, columns):
        self.matrix = matrix
        self.labels = labels
        self.columns = columns
        self.types = ['c' if column == 'route_id' else 'q' for column in columns]
        self.position = 0
        super().__init__(release_data=True)

    def reset(self):
        self.position = 0

    def next(self, input_data):
        if self.position >= len(self.labels):
            return False
        start = self.position
        stop = min(start + 65536, len(self.labels))
        self.position = stop
        input_data(data=cp.asarray(self.matrix[start:stop]), label=cp.asarray(self.labels[start:stop]),
                   feature_names=self.columns, feature_types=self.types)
        return True


def clean_xgb_matrix(h, origins, features):
    route_count = len(ROUTES)
    route_codes = np.arange(route_count, dtype=np.float32)
    matrix = None
    for start in range(0, len(origins), 24):
        block = origins[start:start + 24]
        frame = features(h, block)
        columns = frame.columns.drop(['cutoff', 'timestamp', 'overlap']).tolist()
        frame['route_id'] = np.tile(route_codes, len(block))
        if matrix is None:
            names = columns
            matrix = np.empty((len(origins) * route_count, len(names)), dtype=np.float32)
        if columns != names:
            raise ValueError('Расходятся колонки обычного XGBoost')
        matrix[start * route_count:(start + len(block)) * route_count] = frame[columns].to_numpy(dtype=np.float32)
    if matrix is None:
        raise ValueError('Пустая матрица обычного XGBoost')
    return matrix, names


def predict_clean_xgb(model, matrix):
    result = np.empty(len(matrix), dtype=np.float64)
    for start in range(0, len(matrix), 65536):
        stop = min(start + 65536, len(matrix))
        result[start:stop] = cp.asnumpy(model.inplace_predict(cp.asarray(matrix[start:stop]))) * TARGET_SCALE
    return np.maximum(result, 0)


def clean_xgb_features():
    global CLEAN_XGB_CONTEXT
    if CLEAN_XGB_CONTEXT is None:
        values = {}
        rows = []
        for name in [TARGET, *STATUSES]:
            values[name], flagged = clean_raw_array(VALUES[name], TIMES, CLEANING)
            rows.append(dict(column=name, replaced=int(flagged.sum()), rows=flagged.size))
        pd.DataFrame(rows).to_csv(RUN / 'cleaning_counts.csv', index=False)
        CLEAN_XGB_CONTEXT = dict(np=np, pd=pd, TARGET=TARGET, STATUSES=STATUSES, STEP=STEP,
                                 values=values, times=TIMES, routes=ROUTES)
        exec(FEATURE_CODE, CLEAN_XGB_CONTEXT)
    return CLEAN_XGB_CONTEXT


def fit_clean_xgb(start, stop, split):
    if start >= LOCAL_TEST_START and not (ARTIFACTS / 'frozen_selection.json').is_file():
        raise ValueError('Сначала зафиксируйте выбор по val')
    folder = RUN / start.strftime('%Y%m%d')
    folder.mkdir(parents=True, exist_ok=True)
    outputs = [folder / f'h{h}.parquet' for h in HORIZONS]
    if not all(path.is_file() for path in outputs):
        features = clean_xgb_features()
        target = features['target']
        lag_table = features['calculate_dynamic_shoulder'](
            target, features['hub_signal'], TIMES, ROUTES, fit_before=start,
            max_lookback_hours=LAG_MAX_LOOKBACK_HOURS, min_pairs=LAG_MIN_PAIRS,
            min_correlation=LAG_MIN_CORRELATION)
        features['validation_lag_table'] = lag_table
        lag_table.to_csv(folder / 'status6_lags.csv', index=False)
        for h, path in zip(HORIZONS, outputs):
            if path.is_file():
                continue
            origins = np.arange(len(TIMES) - h)
            train = origins[TIMES[origins + h] < start]
            query = origins[(TIMES[origins] >= start) & (TIMES[origins + h] < stop) &
                            (TIMES[origins].hour == 10) & (TIMES[origins].minute == 30)]
            print('XGBoost clean', split, start.date(), f'h{h}', flush=True)
            matrix, columns = clean_xgb_matrix(h, train, features['horizon_dataset'])
            labels = (target[train + h].ravel() / TARGET_SCALE).astype(np.float32)
            batches = CleanXgbBatches(matrix, labels, columns)
            quantile = xgb.QuantileDMatrix(batches, max_bin=256, max_quantile_batches=8,
                                           enable_categorical=True)
            params = BEST_PARAMS[str(h)].copy()
            trees = params.pop('n_estimators')
            params.update(objective=OBJECTIVE, eval_metric='mae', tree_method='hist', device='cuda:0',
                          max_bin=256, max_cat_to_onehot=1, seed=SEED, nthread=THREADS)
            model = xgb.train(params, quantile, num_boost_round=trees)
            device = json.loads(model.save_config())['learner']['generic_param']['device']
            if not device.startswith('cuda') or model.feature_types != batches.types:
                raise RuntimeError('XGBoost clean потерял CUDA или категориальный route_id')
            del quantile, batches
            cp.get_default_memory_pool().free_all_blocks()
            train_prediction = predict_clean_xgb(model, matrix)
            del matrix
            query_matrix, query_columns = clean_xgb_matrix(h, query, features['horizon_dataset'])
            if columns != query_columns:
                raise ValueError('Расходятся колонки train и проверки обычного XGBoost')
            prediction = predict_clean_xgb(model, query_matrix)
            actual = VALUES[TARGET][query + h].ravel()
            fit_report('xgb_clean', start, h, 'all', split, VALUES[TARGET][train + h].ravel(),
                       train_prediction, actual, prediction)
            frame = pd.DataFrame(dict(cutoff=TIMES[query].repeat(len(ROUTES)), route_id=np.tile(ROUTES, len(query)),
                                      horizon=h, y=actual, xgb_clean=prediction))
            save_parquet(path, frame)
            del model, query_matrix, train_prediction
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
    return pd.concat([pd.read_parquet(path) for path in outputs], ignore_index=True)
