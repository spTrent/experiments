import importlib.metadata
import platform
import zipfile
import joblib
import optuna
import psutil
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge


def offset(frame):
    return frame.profile28.to_numpy(dtype=float) if RESIDUAL else np.zeros(len(frame))


def model_target(frame):
    return (frame.y.to_numpy(dtype=float) - offset(frame)) / TARGET_SCALE


def route_bit_columns():
    return [f'route_bit_{i}' for i in range(max(1, int(len(routes)).bit_length()))]


def transformed_frame(frame, columns):
    result = frame[columns].copy()
    if ALGORITHM in ('xgb', 'lightgbm'):
        result['route_id'] = pd.Categorical(result.route_id, categories=routes)
    elif ALGORITHM == 'catboost':
        result['route_id'] = result.route_id.astype(str)
    else:
        codes = pd.Index(routes).get_indexer(result.route_id).astype(np.int64) + 1
        result = result.drop(columns='route_id')
        for i, name in enumerate(route_bit_columns()):
            result[name] = ((codes >> i) & 1).astype(np.float32)
    if ALGORITHM == 'random_forest':
        result = result.astype(np.float32, copy=False)
    return result


def preprocessing(columns, scale):
    numeric = [c for c in columns if c != 'route_id']
    steps = [('impute', SimpleImputer(strategy='median', keep_empty_features=True))]
    if scale:
        steps.append(('scale', StandardScaler()))
    return ColumnTransformer([
        ('numeric', Pipeline(steps), numeric),
        ('route', 'passthrough', route_bit_columns())
    ], sparse_threshold=0.0)


def suggest_params(trial):
    if ALGORITHM == 'xgb':
        return dict(max_depth=trial.suggest_int('max_depth', 3, 9),
                    learning_rate=trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
                    min_child_weight=trial.suggest_float('min_child_weight', 1, 100, log=True),
                    subsample=trial.suggest_float('subsample', 0.65, 1),
                    colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1),
                    reg_alpha=trial.suggest_float('reg_alpha', 1e-5, 10, log=True),
                    reg_lambda=trial.suggest_float('reg_lambda', 0.1, 100, log=True))
    if ALGORITHM == 'lightgbm':
        return dict(num_leaves=trial.suggest_int('num_leaves', 15, 127, log=True),
                    learning_rate=trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
                    min_child_samples=trial.suggest_int('min_child_samples', 20, 200, log=True),
                    subsample=trial.suggest_float('subsample', 0.65, 1),
                    colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1),
                    reg_alpha=trial.suggest_float('reg_alpha', 1e-5, 10, log=True),
                    reg_lambda=trial.suggest_float('reg_lambda', 0.1, 100, log=True))
    if ALGORITHM == 'catboost':
        return dict(depth=trial.suggest_int('depth', 4, 8),
                    learning_rate=trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
                    l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1, 100, log=True),
                    random_strength=trial.suggest_float('random_strength', 0, 3),
                    bagging_temperature=trial.suggest_float('bagging_temperature', 0, 2))
    if ALGORITHM == 'random_forest':
        return dict(n_estimators=trial.suggest_categorical('n_estimators', [150, 300, 500]),
                    max_depth=trial.suggest_categorical('max_depth', [10, 16, 24]),
                    min_samples_leaf=trial.suggest_int('min_samples_leaf', 2, 64, log=True),
                    max_features=trial.suggest_float('max_features', 0.25, 0.9),
                    max_samples=trial.suggest_float('max_samples', 0.5, 1))
    if ALGORITHM == 'ridge':
        return dict(alpha=trial.suggest_float('alpha', 1e-3, 1e6, log=True))
    return dict(changepoint_prior_scale=trial.suggest_float('changepoint_prior_scale', 0.001, 0.2, log=True),
                seasonality_prior_scale=trial.suggest_float('seasonality_prior_scale', 0.01, 10, log=True),
                regressor_prior_scale=trial.suggest_float('regressor_prior_scale', 0.001, 10, log=True),
                n_changepoints=trial.suggest_categorical('n_changepoints', [5, 10, 20]))


def prophet_matrix(frame, columns, prep, names):
    encoded = prep.transform(transformed_frame(frame, columns))
    if hasattr(encoded, 'toarray'):
        encoded = encoded.toarray()
    result = pd.DataFrame(encoded, columns=names)
    result.insert(0, 'ds', frame.timestamp.to_numpy())
    return result


def fit_model(frame, columns, params, early=None, trees=None):
    params = dict(params)
    if len(frame) == 0 or not np.isfinite(frame.y.to_numpy()).all():
        raise ValueError('Обучающая выборка пуста или содержит некорректные метки')
    if early is not None and (len(early) == 0 or frame.timestamp.max() >= early.cutoff.min().normalize()):
        raise ValueError('Нарушена временная граница train/val')
    y = model_target(frame)
    if not np.isfinite(y).all():
        raise ValueError('Некорректная цель после преобразования')
    if ALGORITHM == 'xgb':
        from xgboost import XGBRegressor
        options = dict(n_estimators=trees or MAX_TREES, device=DEVICE, tree_method='hist',
                       enable_categorical=True, max_cat_to_onehot=1, objective='reg:squarederror', eval_metric='mae',
                       random_state=SEED, n_jobs=N_JOBS, **params)
        kwargs = {}
        if early is not None:
            options['early_stopping_rounds'] = EARLY_STOPPING_ROUNDS
            kwargs['eval_set'] = [(transformed_frame(early, columns), model_target(early))]
        model = XGBRegressor(**options)
        model.fit(transformed_frame(frame, columns), y, verbose=False, **kwargs)
        actual_device = json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
        if DEVICE == 'cuda' and not actual_device.startswith('cuda'):
            raise RuntimeError('Запрошен GPU, но XGBoost работает на CPU. Проверьте CUDA или явно выберите DEVICE=cpu.')
        count = model.best_iteration + 1 if early is not None else options['n_estimators']
        return dict(model=model, columns=columns, trees=int(count))
    if ALGORITHM == 'lightgbm':
        from lightgbm import LGBMRegressor, early_stopping, log_evaluation
        model = LGBMRegressor(n_estimators=trees or MAX_TREES, objective='regression', metric='l1',
                              device_type=DEVICE, subsample_freq=1, importance_type='gain',
                              max_cat_to_onehot=1, verbosity=-1, random_state=SEED, n_jobs=N_JOBS, **params)
        kwargs = dict(categorical_feature=['route_id'], callbacks=[log_evaluation(0)])
        if early is not None:
            kwargs['eval_set'] = [(transformed_frame(early, columns), model_target(early))]
            kwargs['callbacks'].append(early_stopping(EARLY_STOPPING_ROUNDS, first_metric_only=True, verbose=False))
        model.fit(transformed_frame(frame, columns), y, **kwargs)
        count = model.best_iteration_ if early is not None else model.n_estimators_
        return dict(model=model, columns=columns, trees=int(count))
    if ALGORITHM == 'catboost':
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(iterations=trees or MAX_TREES, loss_function='RMSE', eval_metric='MAE',
                                  task_type='GPU' if DEVICE == 'cuda' else 'CPU',
                                  thread_count=N_JOBS, random_seed=SEED, bootstrap_type='Bayesian',
                                  allow_writing_files=False, one_hot_max_size=1, verbose=False, **params)
        kwargs = {}
        if early is not None:
            kwargs.update(eval_set=(transformed_frame(early, columns), model_target(early)),
                          early_stopping_rounds=EARLY_STOPPING_ROUNDS, use_best_model=True)
        model.fit(transformed_frame(frame, columns), y, cat_features=['route_id'], **kwargs)
        return dict(model=model, columns=columns, trees=int(model.tree_count_))
    if ALGORITHM in ('ridge', 'random_forest'):
        if ALGORITHM == 'random_forest':
            from cuml.ensemble import RandomForestRegressor
            estimator = RandomForestRegressor(random_state=SEED, n_streams=1, output_type='numpy', **params)
            y = y.astype(np.float32)
        else:
            estimator = Ridge(solver='lsqr', tol=1e-5, **params)
        model = Pipeline([('preprocess', preprocessing(columns, ALGORITHM == 'ridge')), ('model', estimator)])
        model.fit(transformed_frame(frame, columns), y)
        return dict(model=model, columns=columns, trees=None)
    from prophet import Prophet
    width = len(columns) - 1 + len(route_bit_columns())
    estimated_gb = len(frame) * width * 8 * 6 / 1024 ** 3
    available_gb = psutil.virtual_memory().available / 1024 ** 3
    print(f'Prophet: оценка рабочей памяти {estimated_gb:.1f} ГБ; доступно {available_gb:.1f} ГБ')
    if estimated_gb > available_gb * 0.8:
        raise MemoryError('Для полного Prophet с категориальными маршрутами нужна машина с большей RAM. Строки и признаки автоматически не сокращаются.')
    prep = preprocessing(columns, True)
    prep.fit(transformed_frame(frame, columns))
    names = [f'x{i}' for i in range(len(prep.get_feature_names_out()))]
    data = prophet_matrix(frame, columns, prep, names)
    active = [name for name in names if data[name].nunique() > 1]
    prior = params.pop('regressor_prior_scale')
    model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True,
                    uncertainty_samples=0, **params)
    for name in active:
        model.add_regressor(name, prior_scale=prior, standardize=False)
    data = data[['ds', *active]].copy()
    data['y'] = y
    model.fit(data, seed=SEED)
    return dict(model=model, columns=columns, trees=None, prep=prep, names=names, active=active)


def predict(bundle, frame):
    chunks = []
    for start in range(0, len(frame), PREDICTION_BATCH):
        part = frame.iloc[start:start + PREDICTION_BATCH]
        if ALGORITHM == 'prophet':
            data = prophet_matrix(part, bundle['columns'], bundle['prep'], bundle['names'])
            order = np.argsort(data.ds.to_numpy(), kind='stable')
            forecast = bundle['model'].predict(data.iloc[order][['ds', *bundle['active']]])['yhat'].to_numpy()
            raw = np.empty(len(part), dtype=np.float64)
            raw[order] = forecast
        else:
            raw = bundle['model'].predict(transformed_frame(part, bundle['columns']))
        raw = np.asarray(raw, dtype=np.float64)
        restored = offset(part) + TARGET_SCALE * raw
        if raw.shape != (len(part),) or not np.isfinite(restored).all():
            raise ValueError('Модель вернула некорректные прогнозы')
        chunks.append(np.maximum(0, restored))
    return np.concatenate(chunks)


def save_bundle(bundle, folder):
    folder.mkdir(parents=True, exist_ok=True)
    metadata = dict(algorithm=ALGORITHM, residual=RESIDUAL, target_scale=TARGET_SCALE,
                    columns=bundle['columns'], trees=bundle['trees'], route_categories=routes.tolist(),
                    route_encoding='native_categorical' if ALGORITHM in ('xgb', 'catboost', 'lightgbm') else 'binary_index_plus_one',
                    route_bit_columns=route_bit_columns() if ALGORITHM not in ('xgb', 'catboost', 'lightgbm') else [])
    if ALGORITHM == 'xgb':
        bundle['model'].save_model(folder / 'model.ubj')
    elif ALGORITHM == 'lightgbm':
        bundle['model'].booster_.save_model(str(folder / 'model.txt'))
    elif ALGORITHM == 'catboost':
        bundle['model'].save_model(str(folder / 'model.cbm'))
    elif ALGORITHM == 'prophet':
        from prophet.serialize import model_to_json
        (folder / 'model.json').write_text(model_to_json(bundle['model']))
        joblib.dump({k: bundle[k] for k in ('prep', 'names', 'active')}, folder / 'preprocess.joblib', compress=3)
    else:
        joblib.dump(bundle['model'], folder / 'model.joblib', compress=3)
    (folder / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))


def add_report(rows, phase, h, trial=None, fold=None):
    records = [dict(phase=phase, horizon=h, trial=trial, fold=fold, **row) for row in rows]
    pd.DataFrame(records).to_csv(DOCS_DIR / 'metrics.csv', mode='a', header=not (DOCS_DIR / 'metrics.csv').exists(), index=False)
    display(pd.DataFrame(records))


def prediction_part(frame, prediction, h, period):
    part = frame[['cutoff', 'timestamp', 'route_id', 'y', 'profile28']].copy()
    part['prediction'] = prediction
    part[ALGORITHM] = prediction
    part['horizon'] = h
    part['period'] = period
    return part


def export_results():
    (DOCS_DIR / 'evaluation_note.txt').write_text(manifest['test_note'])
    with zipfile.ZipFile(RUN_DIR / 'reports.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for folder in (DOCS_DIR, PARAMS_DIR, IMPORTANCE_DIR):
            for path in folder.rglob('*'):
                if path.is_file():
                    archive.write(path, path.relative_to(RUN_DIR))
    with zipfile.ZipFile(RUN_DIR / 'results.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in RUN_DIR.rglob('*'):
            if path.is_file() and path.suffix != '.zip' and OUT not in path.parents:
                archive.write(path, path.relative_to(RUN_DIR))


versions = {}
for name in ('numpy', 'pandas', 'pyarrow', 'scikit-learn', 'xgboost', 'catboost', 'lightgbm', 'prophet', 'optuna', 'joblib'):
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        pass
if ALGORITHM == 'random_forest':
    import cuml
    versions['cuml'] = cuml.__version__
manifest = dict(algorithm=ALGORITHM, residual=RESIDUAL, seed=SEED, device=DEVICE, n_trials=N_TRIALS,
                python=platform.python_version(), platform=platform.platform(), versions=versions,
                feature_pipeline=FEATURE_PIPELINE_VERSION, starts=TRAIN_MINUTES,
                holdout_start=str(holdout_start), selection_end_exclusive=str(selection_end), search_objective='mean of three tuning WAPE + absolute relative bias scores',
                test_note='Независимая оценка test получена до переобучения на всей истории. Период test ранее использовался в проекте; это не новый слепой holdout.')
(DOCS_DIR / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
lag_tables = [fit_lags(f['val'][0].normalize()) for f in folds]
lag_tables[-1].to_csv(DOCS_DIR / 'status6_lags_validation.csv', index=False)
for i, table in enumerate(lag_tables):
    table.to_csv(DOCS_DIR / f'status6_lags_fold_{i+1}.csv', index=False)


def select_horizon(h):
    history = pd.read_parquet(OUT / f'history_h{h}.parquet', filters=[('timestamp', '<', selection_end)])
    history = history.loc[np.isfinite(history.profile28)].copy()
    columns = model_columns(history)
    def objective(trial):
        params = suggest_params(trial)
        metrics, tree_counts, tuning_parts = [], [], []
        for i, (fold, lag_table) in enumerate(zip(folds, lag_tables)):
            train = history.loc[history.cutoff.dt.normalize().isin(fold['train'].normalize()) & (history.timestamp < fold['val'][0].normalize())].copy()
            early = history.loc[history.cutoff.isin(fold['val'])].copy()
            tuning = history.loc[history.cutoff.isin(fold['tuning'])].copy()
            if len(train) == 0 or len(early) == 0 or len(tuning) == 0:
                raise ValueError('Пустая временная выборка')
            if train.timestamp.max() >= early.cutoff.min().normalize() or early.timestamp.max() >= tuning.cutoff.min().normalize() or tuning.timestamp.max() >= selection_end:
                raise ValueError('Пересекаются временные периоды train/val/tuning/test')
            train = add_correlation_features(train, h, lag_table)
            early = add_correlation_features(early, h, lag_table)
            tuning = add_correlation_features(tuning, h, lag_table)
            bundle = fit_model(train, columns, params, early if ALGORITHM in ('xgb', 'catboost', 'lightgbm') else None)
            tuning_prediction = predict(bundle, tuning)
            rows = [score_row(train, predict(bundle, train), 'train'), score_row(tuning, tuning_prediction, 'tuning')]
            part = prediction_part(tuning, tuning_prediction, h, 'tuning')
            part['fold'] = i + 1
            part['fit_labels_before'] = fold['val'][0].normalize()
            part['selection_labels_through'] = folds[-1]['tuning'][-1] + h * STEP
            part['model_info_through'] = part['selection_labels_through']
            part['is_unbiased_oof'] = False
            tuning_parts.append(part)
            add_report(rows, 'search', h, trial.number, i + 1)
            metrics.append(rows[-1]['метрика конкурса'])
            if bundle['trees'] is not None:
                tree_counts.append(bundle['trees'])
            del bundle, train, early, tuning
            gc.collect()
        trial.set_user_attr('tree_counts', tree_counts)
        trial.set_user_attr('fold_scores', metrics)
        value = float(np.mean(metrics))
        if not np.isfinite(value):
            raise ValueError('Неконечная метрика подбора')
        completed = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
        if not completed or value < min(item.value for item in completed):
            pd.concat(tuning_parts, ignore_index=True).to_parquet(POSTPROCESS_DIR / f'tuning_predictions_h{h}.parquet', index=False)
        return value
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED),
                                storage=f'sqlite:///{(POSTPROCESS_DIR / "search.sqlite3").resolve().as_posix()}',
                                study_name=f'h{h}', load_if_exists=False)
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1, gc_after_trial=True)
    study.trials_dataframe().to_csv(DOCS_DIR / f'trials_h{h}.csv', index=False)
    counts = study.best_trial.user_attrs['tree_counts']
    selected = dict(algorithm=ALGORITHM, residual=RESIDUAL, horizon=h, params=study.best_params, trees=int(np.median(counts)) if counts else None,
                    selection_end_exclusive=str(selection_end), selection_labels_through=str(folds[-1]['tuning'][-1] + h * STEP),
                    trial=study.best_trial.number, tuning_score=study.best_value,
                    fold_scores=study.best_trial.user_attrs['fold_scores'])
    (PARAMS_DIR / f'best_h{h}.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2))
    return selected


if ALGORITHM == 'prophet':
    full_rows = pq.ParquetFile(OUT / 'history_h1.parquet').metadata.num_rows
    width = len(pq.read_schema(OUT / 'history_h1.parquet').names) - 6 + len(route_bit_columns())
    estimated_gb = full_rows * width * 8 * 6 / 1024 ** 3
    available_gb = psutil.virtual_memory().available / 1024 ** 3
    print(f'Prophet full refit: оценка {estimated_gb:.1f} ГБ, доступно {available_gb:.1f} ГБ')
    if estimated_gb > available_gb * 0.8:
        raise MemoryError('Недостаточно RAM для полного Prophet. Нужна среда с большей памятью; автоматического сокращения данных нет.')

selected = {}
for h in range(1, 9):
    selected[h] = select_horizon(h)
    gc.collect()
(PARAMS_DIR / 'selection_frozen.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2))
evaluation_lags = fit_lags(holdout_start.normalize())
evaluation_lags.to_csv(DOCS_DIR / 'status6_lags_evaluation.csv', index=False)
evaluation_rows = []
for h in range(1, 9):
    history = pd.read_parquet(OUT / f'history_h{h}.parquet')
    history = history.loc[np.isfinite(history.profile28)].copy()
    columns = model_columns(history)
    train = add_correlation_features(history.loc[history.timestamp < holdout_start.normalize()].copy(), h, evaluation_lags)
    local_test = add_correlation_features(history.loc[history.cutoff.isin(folds[-1]['test'])].copy(), h, evaluation_lags)
    if len(train) == 0 or len(local_test) == 0 or train.timestamp.max() >= local_test.cutoff.min().normalize():
        raise ValueError('Нарушена граница независимого test')
    bundle = fit_model(train, columns, selected[h]['params'], trees=selected[h]['trees'])
    rows = []
    for split, frame in [('train', train), ('test', local_test)]:
        pred = predict(bundle, frame)
        rows.append(score_row(frame, pred, split))
        part = prediction_part(frame, pred, h, split)
        part['model_info_through'] = holdout_start.normalize() - STEP
        part['is_unbiased_oof'] = split == 'test'
        part.to_parquet(POSTPROCESS_DIR / f'{split}_predictions_h{h}.parquet', index=False)
    add_report(rows, 'independent_evaluation', h)
    evaluation_rows.extend([dict(horizon=h, **row) for row in rows])
    save_bundle(bundle, MODEL_DIR / 'evaluation' / f'h{h}')
    del bundle, history, train, local_test, part, frame
    gc.collect()
pd.DataFrame(evaluation_rows).to_csv(DOCS_DIR / 'independent_metrics_by_horizon.csv', index=False)
aggregate_rows = []
for split in ('train', 'test'):
    part = pd.concat([pd.read_parquet(POSTPROCESS_DIR / f'{split}_predictions_h{h}.parquet') for h in range(1, 9)], ignore_index=True)
    aggregate_rows.append(score_row(part, part.prediction, split))
    aggregate_rows.append(score_row(part, part.profile28, split + ' profile28'))
    if split == 'test':
        part.to_parquet(POSTPROCESS_DIR / 'test_predictions.parquet', index=False)
    del part
independent_metrics = pd.DataFrame(aggregate_rows)
display(independent_metrics)
independent_metrics.to_csv(DOCS_DIR / 'independent_metrics.csv', index=False)
profile_control = independent_metrics.copy()
profile_control['model'] = np.where(profile_control['выборка'].str.endswith(' profile28'), 'profile28', ALGORITHM)
profile_control['выборка'] = profile_control['выборка'].str.replace(' profile28', '', regex=False)
profile_control.to_csv(DOCS_DIR / 'profile_control.csv', index=False)
print(manifest['test_note'])
export_results()

full_lags = fit_lags(end + STEP)
full_lags.to_csv(DOCS_DIR / 'status6_lags_full.csv', index=False)
submission_parts, importance_parts = [], []
for h in range(1, 9):
    history = pd.read_parquet(OUT / f'history_h{h}.parquet')
    history = history.loc[np.isfinite(history.profile28)].copy()
    history = add_correlation_features(history, h, full_lags)
    columns = model_columns(history)
    future = add_correlation_features(pd.read_parquet(OUT / f'submission_h{h}.parquet'), h, full_lags)
    bundle = fit_model(history, columns, selected[h]['params'], trees=selected[h]['trees'])
    save_bundle(bundle, MODEL_DIR / 'full' / f'h{h}')
    rows = [score_row(history, predict(bundle, history), 'вся история после refit')]
    add_report(rows, 'full_refit', h)
    display(pd.DataFrame([row for row in evaluation_rows if row['horizon'] == h]))
    print(manifest['test_note'])
    part = future[['id']].copy()
    part['y_pred'] = predict(bundle, future)
    submission_parts.append(part)
    if ALGORITHM in ('xgb', 'catboost', 'lightgbm', 'random_forest'):
        model = bundle['model']
        names = columns
        if ALGORITHM == 'random_forest':
            names = model.named_steps['preprocess'].get_feature_names_out()
            model = model.named_steps['model']
        if ALGORITHM == 'random_forest' and not hasattr(model, 'feature_importances_'):
            sample = history.sample(n=min(PREDICTION_BATCH, len(history)), random_state=SEED)
            matrix = bundle['model'].named_steps['preprocess'].transform(transformed_frame(sample, columns))
            target = model_target(sample)
            base_mae = np.abs(target - model.predict(matrix)).mean()
            order = np.random.default_rng(SEED).permutation(len(sample))
            values = []
            for j in range(matrix.shape[1]):
                original = matrix[:, j].copy()
                matrix[:, j] = original[order]
                values.append(np.abs(target - model.predict(matrix)).mean() - base_mae)
                matrix[:, j] = original
            importance_kind = 'permutation_increase_in_scaled_mae'
        else:
            values = model.feature_importances_
            importance_kind = 'native_feature_importance'
        importance_values = pd.Series(values, index=names, name=f'h{h}').sort_values(ascending=False)
        importance_values.to_csv(IMPORTANCE_DIR / f'importance_h{h}.csv')
        importance_parts.append(importance_values)
    if ALGORITHM == 'ridge':
        model = bundle['model']
        coefficients = pd.Series(model.named_steps['model'].coef_, index=model.named_steps['preprocess'].get_feature_names_out(), name=f'h{h}')
        coefficients.to_csv(IMPORTANCE_DIR / f'coefficients_h{h}.csv')
        importance_parts.append(coefficients.abs())
    if ALGORITHM == 'prophet':
        from prophet.utilities import regressor_coefficients
        coefficients = regressor_coefficients(bundle['model'])
        mapping = dict(zip(bundle['names'], bundle['prep'].get_feature_names_out()))
        coefficients['feature'] = coefficients.regressor.map(mapping)
        coefficients.to_csv(IMPORTANCE_DIR / f'coefficients_h{h}.csv', index=False)
        importance_parts.append(pd.Series(coefficients.coef.to_numpy(), index=coefficients.feature, name=f'h{h}').abs())
    if ALGORITHM in ('xgb', 'catboost', 'lightgbm', 'random_forest', 'ridge'):
        del model
    del bundle, history, future
    gc.collect()
predictions = pd.concat(submission_parts, ignore_index=True)
submission = test[['id']].merge(predictions, on='id', how='left', validate='one_to_one')
if len(predictions) != len(test) or not np.isfinite(submission.y_pred).all() or (submission.y_pred < 0).any():
    raise ValueError('Некорректный submission')
submission.to_csv(DOCS_DIR / 'submission.csv', index=False)
importance_table = pd.concat(importance_parts, axis=1).rename_axis('feature')
importance_table['mean_all_8'] = importance_table.fillna(0).sum(axis=1) / 8
importance_table['importance_kind'] = 'absolute_coefficient' if ALGORITHM in ('ridge', 'prophet') else importance_kind
importance_table = importance_table.sort_values('mean_all_8', ascending=False)
importance_table.to_csv(DOCS_DIR / 'feature_importance.csv')
export_results()
print(f'Результаты: {RUN_DIR.resolve()}')
print(f'Лучшие параметры: {PARAMS_DIR.resolve()}')
print(f'Для Git: {RUN_DIR / "reports.zip"}; модели и прогнозы: {RUN_DIR / "results.zip"}')
