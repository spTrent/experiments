import time
import joblib
import optuna
import cuml
import sklearn
import zipfile
from sklearn.impute import SimpleImputer


def atomic_json(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def model_target(frame):
    baseline = frame.profile28.to_numpy() if RESIDUAL else 0
    return ((frame.y.to_numpy() - baseline) / TARGET_SCALE).astype(np.float32)


def transformed_frame(frame, columns):
    result = frame[columns].drop(columns='route_id').astype(np.float32)
    codes = pd.Index(routes).get_indexer(frame.route_id).astype(np.int64) + 1
    bits = {f'route_bit_{i}': ((codes >> i) & 1).astype(np.float32)
            for i in range(max(1, int(len(routes)).bit_length()))}
    return pd.concat([result, pd.DataFrame(bits, index=frame.index)], axis=1).copy()


def cached_lags(boundary):
    path = OUT / f'lags_{pd.Timestamp(boundary).strftime("%Y%m%dT%H%M%S")}.parquet'
    if path.exists():
        return pd.read_parquet(path)
    table = fit_lags(boundary)
    temporary = path.with_suffix('.partial')
    table.to_parquet(temporary, index=False)
    temporary.replace(path)
    return table


def prepare_stage(h, stage, train_dates=None, validation_dates=None, boundary=None):
    folder = PREPARED / f'h{h}_{stage}'
    marker = folder / 'ready.json'
    if marker.exists():
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    filters = [('timestamp', '<', selection_end)] if stage.startswith('fold') else None
    history = pd.read_parquet(OUT / f'history_h{h}.parquet', filters=filters)
    history = history.loc[np.isfinite(history.profile28)].copy()
    mask = history.timestamp < boundary
    if train_dates is not None:
        mask &= history.cutoff.dt.normalize().isin(train_dates.normalize())
    train = history.loc[mask].copy()
    valid = history.loc[history.cutoff.isin(validation_dates)].copy()
    del history
    if train.empty or valid.empty or train.timestamp.max() >= valid.cutoff.min().normalize():
        raise ValueError('Нарушено временное разделение')
    if stage.startswith('fold') and valid.timestamp.max() >= selection_end:
        raise ValueError('Tuning выходит за границу подбора')
    table = cached_lags(boundary)
    train = add_correlation_features(train, h, table)
    valid = add_correlation_features(valid, h, table)
    columns = model_columns(train)
    prep = SimpleImputer(strategy='median', keep_empty_features=True)
    for name, frame in [('train', train), ('valid', valid)]:
        encoded = transformed_frame(frame, columns)
        matrix = prep.fit_transform(encoded) if name == 'train' else prep.transform(encoded)
        np.save(folder / f'{name}.npy', np.asarray(matrix, dtype=np.float32))
        frame[['cutoff', 'timestamp', 'route_id', 'y', 'profile28']].to_parquet(folder / f'{name}.parquet', index=False)
        del matrix, encoded
    joblib.dump(prep, folder / 'preprocess.joblib')
    atomic_json(marker, dict(columns=columns, boundary=str(boundary), rows_train=len(train), rows_valid=len(valid)))
    return folder


def load_stage(folder):
    return {name: (pd.read_parquet(folder / f'{name}.parquet'), np.load(folder / f'{name}.npy', mmap_mode='r')) for name in ('train', 'valid')}


def fit_forest(frame, matrix, params):
    model = RandomForestRegressor(random_state=SEED, n_streams=N_STREAMS, output_type='numpy', **params)
    model.fit(matrix, model_target(frame))
    cp.cuda.get_current_stream().synchronize()
    return model


def predict_matrix(model, frame, matrix):
    result = np.empty(len(frame), dtype=np.float64)
    for start in range(0, len(frame), PREDICTION_BATCH):
        stop = min(start + PREDICTION_BATCH, len(frame))
        raw = np.asarray(model.predict(matrix[start:stop]), dtype=np.float64).reshape(-1)
        baseline = frame.profile28.iloc[start:stop].to_numpy() if RESIDUAL else 0
        result[start:stop] = np.maximum(0, raw * TARGET_SCALE + baseline)
    if not np.isfinite(result).all():
        raise ValueError('Некорректный прогноз')
    return result


def prediction_part(frame, prediction, h, period):
    part = frame.copy()
    part['prediction'] = prediction
    part[ALGORITHM] = prediction
    part['horizon'] = h
    part['period'] = period
    return part


def report(rows, phase, h, trial=None, fold=None, seconds=None):
    records = [dict(phase=phase, horizon=h, trial=trial, fold=fold, seconds=seconds, **row) for row in rows]
    path = DOCS_DIR / f'{phase}_h{h}_trial{trial}_fold{fold}.json'
    atomic_json(path, records)
    display(pd.DataFrame(records))


def suggest_params(trial):
    return dict(n_estimators=trial.suggest_categorical('n_estimators', RF_TREES),
                max_depth=trial.suggest_categorical('max_depth', RF_DEPTHS),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 2, 64, log=True),
                max_features=trial.suggest_float('max_features', 0.25, 0.9),
                max_samples=trial.suggest_float('max_samples', 0.5, 1))


def select_horizon(h):
    sampler_path = POSTPROCESS_DIR / f'sampler_h{h}.joblib'
    sampler = joblib.load(sampler_path) if sampler_path.exists() else optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction='minimize', sampler=sampler,
                               storage=f'sqlite:///{(POSTPROCESS_DIR / "search.sqlite3").resolve().as_posix()}',
                               study_name=f'h{h}', load_if_exists=True)
    for trial in study.get_trials(states=(optuna.trial.TrialState.RUNNING,)):
        study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
        if trial.params:
            study.enqueue_trial(trial.params, skip_if_exists=False)
    completed = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
    remaining = max(0, N_TRIALS - len(completed))
    folders = []
    if remaining:
        for i, fold in enumerate(folds):
            folders.append(prepare_stage(h, f'fold{i+1}', fold['train'], fold['tuning'], fold['val'][0].normalize()))
    def objective(trial):
        started = time.perf_counter()
        params = suggest_params(trial)
        scores, parts = [], []
        for i, folder in enumerate(folders):
            fold_started = time.perf_counter()
            data = load_stage(folder)
            train, x_train = data['train']
            tuning, x_tuning = data['valid']
            model = fit_forest(train, x_train, params)
            prediction = predict_matrix(model, tuning, x_tuning)
            rows = [score_row(train, predict_matrix(model, train, x_train), 'train'), score_row(tuning, prediction, 'tuning')]
            scores.append(rows[-1]['метрика конкурса'])
            part = prediction_part(tuning, prediction, h, 'tuning')
            part['fold'] = i + 1
            part['fit_labels_before'] = folds[i]['val'][0].normalize()
            part['selection_labels_through'] = folds[-1]['tuning'][-1] + h * STEP
            part['model_info_through'] = part['selection_labels_through']
            part['is_unbiased_oof'] = False
            parts.append(part)
            report(rows, 'search', h, trial.number, i + 1, time.perf_counter() - fold_started)
            del model, data, train, tuning, x_train, x_tuning
            gc.collect()
        value = float(np.mean(scores))
        if not np.isfinite(value):
            raise ValueError('Неконечная метрика подбора')
        path = POSTPROCESS_DIR / f'tuning_h{h}_trial{trial.number}.parquet'
        temporary = path.with_suffix('.partial')
        pd.concat(parts, ignore_index=True).to_parquet(temporary, index=False)
        temporary.replace(path)
        trial.set_user_attr('fold_scores', scores)
        trial.set_user_attr('elapsed_seconds', time.perf_counter() - started)
        return value
    def save_sampler(study, trial):
        temporary = sampler_path.with_suffix('.partial')
        joblib.dump(study.sampler, temporary)
        temporary.replace(sampler_path)
    if remaining:
        study.optimize(objective, n_trials=remaining, n_jobs=1, gc_after_trial=True, callbacks=[save_sampler])
    study.trials_dataframe().to_csv(DOCS_DIR / f'trials_h{h}.csv', index=False)
    best = study.best_trial
    best_part = pd.read_parquet(POSTPROCESS_DIR / f'tuning_h{h}_trial{best.number}.parquet')
    best_part.to_parquet(POSTPROCESS_DIR / f'tuning_predictions_h{h}.parquet', index=False)
    chosen = dict(algorithm=ALGORITHM, residual=RESIDUAL, horizon=h, params=best.params, trial=best.number,
                  tuning_score=best.value, fold_scores=best.user_attrs['fold_scores'], trees=best.params['n_estimators'],
                  selection_end_exclusive=str(selection_end), selection_labels_through=str(folds[-1]['tuning'][-1] + h * STEP))
    atomic_json(PARAMS_DIR / f'best_h{h}.json', chosen)
    durations = [t.user_attrs['elapsed_seconds'] for t in study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))]
    print(f'h={h}: завершено {len(durations)} trials, среднее {np.mean(durations)/60:.1f} мин/trial')
    return chosen


manifest = dict(algorithm=ALGORITHM, residual=RESIDUAL, seed=SEED, device=DEVICE, n_trials=N_TRIALS,
                n_streams=N_STREAMS, prediction_batch=PREDICTION_BATCH, search_trees=RF_TREES, search_depths=RF_DEPTHS,
                feature_cache=str(OUT), prepared_cache=str(PREPARED), holdout_start=str(holdout_start),
                selection_end_exclusive=str(selection_end), contract=RUN_CONTRACT,
                test_note='Независимая оценка test получена на моделях, обученных только на истории до test, включая train и val. Переобучение на всей истории и submission не выполняются. Период test ранее использовался в проекте; это не новый слепой holdout.')
atomic_json(DOCS_DIR / 'manifest.json', manifest)

selected = {}
for h in range(1, 9):
    selected[h] = select_horizon(h)
    gc.collect()
atomic_json(PARAMS_DIR / 'selection_frozen.json', selected)

evaluation_rows = []
for h in range(1, 9):
    checkpoint = MODEL_DIR / 'evaluation' / f'h{h}'
    done = checkpoint / 'complete.json'
    if done.exists():
        rows = json.loads(done.read_text())['rows']
        display(pd.DataFrame(rows))
    else:
        started = time.perf_counter()
        folder = prepare_stage(h, 'evaluation', validation_dates=folds[-1]['test'], boundary=holdout_start.normalize())
        data = load_stage(folder)
        train, x_train = data['train']
        model = fit_forest(train, x_train, selected[h]['params'])
        rows = []
        for name, split in [('train', 'train'), ('valid', 'test')]:
            frame, matrix = data[name]
            prediction = predict_matrix(model, frame, matrix)
            rows.append(score_row(frame, prediction, split))
            part = prediction_part(frame, prediction, h, split)
            part['model_info_through'] = holdout_start.normalize() - STEP
            part['is_unbiased_oof'] = split == 'test'
            path = POSTPROCESS_DIR / f'{split}_predictions_h{h}.parquet'
            temporary = path.with_suffix('.partial')
            part.to_parquet(temporary, index=False)
            temporary.replace(path)
        checkpoint.mkdir(parents=True, exist_ok=True)
        metadata = json.loads((folder / 'ready.json').read_text())
        metadata.update(algorithm=ALGORITHM, residual=RESIDUAL, target_scale=TARGET_SCALE, route_categories=routes.tolist(), route_encoding='binary_index_plus_one', params=selected[h]['params'])
        bundle = dict(model=model, preprocess=joblib.load(folder / 'preprocess.joblib'), metadata=metadata)
        temporary = checkpoint / 'model.partial'
        joblib.dump(bundle, temporary, compress=0)
        temporary.replace(checkpoint / 'model.joblib')
        atomic_json(checkpoint / 'metadata.json', metadata)
        report(rows, 'independent_evaluation', h, seconds=time.perf_counter() - started)
        atomic_json(done, dict(rows=rows))
        del bundle, model, data, train, x_train, frame, matrix, part
        gc.collect()
    evaluation_rows.extend([dict(horizon=h, **row) for row in rows])
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
(DOCS_DIR / 'evaluation_note.txt').write_text(manifest['test_note'])
records = []
for path in sorted(DOCS_DIR.glob('*_h*_trial*_fold*.json')):
    records.extend(json.loads(path.read_text()))
pd.DataFrame(records).to_csv(DOCS_DIR / 'metrics.csv', index=False)
with zipfile.ZipFile(RUN_DIR / 'reports.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for folder in (DOCS_DIR, PARAMS_DIR):
        for path in folder.rglob('*'):
            if path.is_file():
                archive.write(path, path.relative_to(RUN_DIR))
print(manifest['test_note'])
print(f'Результаты: {RUN_DIR.resolve()}')
