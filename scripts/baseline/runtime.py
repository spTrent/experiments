def metrics(frame, prediction):
    y = frame.y.to_numpy(dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    if p.shape != y.shape or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Некорректные факты или прогнозы')
    e = p - y
    daily = pd.DataFrame(dict(day=frame.cutoff.dt.normalize().to_numpy(), y=y,
                              absolute=np.abs(e), signed=e)).groupby('day').sum()
    if (daily.y <= 0).any():
        raise ValueError('Неопределённая метрика: нулевой объём дня')
    return dict(rows=len(frame), days=len(daily), WAPE=float(daily.absolute.sum()/daily.y.sum()),
                signed_bias=float(daily.signed.sum()/daily.y.sum()),
                score=float((daily.absolute.sum()+abs(daily.signed.sum()))/daily.y.sum()),
                daily_score=float(((daily.absolute+daily.signed.abs())/daily.y).mean()))

def atomic_parquet(frame, path):
    temporary = path.with_suffix('.partial.parquet')
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)

def archive(name, include_models=False):
    destination = RUN_ROOT.parent / name
    temporary = destination.with_suffix('.partial.zip')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for path in sorted(RUN_ROOT.rglob('*')):
            if not path.is_file() or '.partial' in path.name:
                continue
            if path.suffix == '.cbm' and not include_models:
                continue
            z.write(path, str(Path(RUN_ROOT.name) / path.relative_to(RUN_ROOT)))
    temporary.replace(destination)
    print('Saved:', destination.name, f'{destination.stat().st_size / 2**20:.1f} MiB')

def completed(stage, h):
    marker = RUN_ROOT / 'checkpoints' / f'{stage}_h{h}.json'
    if not marker.exists():
        return None
    info = json.loads(marker.read_text())
    if info['run_id'] != RUN_ID:
        raise ValueError('Checkpoint принадлежит другому запуску')
    for rel, digest in info['files'].items():
        path = RUN_ROOT / rel
        if not path.is_file() or file_hash(path) != digest:
            raise ValueError(f'Повреждён checkpoint: {path}. Восстановите архив или используйте новую RUN_ROOT.')
    return info

def model_columns(frame):
    return [c for c in frame if c not in ['timestamp', 'cutoff', 'y', 'profile28', 'overlap']]

def recent_mask(frame, boundary):
    eligible = frame.timestamp < boundary
    cutoffs = pd.DatetimeIndex(frame.loc[eligible, 'cutoff'].unique()).sort_values()
    if len(cutoffs) == 0:
        raise ValueError('Пустой train')
    start = cutoffs[int(np.floor((1-RECENT_TRAIN_FRACTION)*len(cutoffs)))]
    return eligible & frame.cutoff.ge(start), start

def make_model(trees):
    return CatBoostRegressor(iterations=trees, depth=7, learning_rate=0.05,
                             loss_function='MAE', eval_metric='MAE', task_type='GPU',
                             devices=GPU_DEVICES, thread_count=THREAD_COUNT,
                             cat_features=['route_id'], random_seed=42,
                             allow_writing_files=False, verbose=300)

weekly_days = dates[-7*WEEKLY_BLOCKS:]
if len(weekly_days) != 42 or not weekly_days.equals(pd.date_range(weekly_days[0], periods=42, freq='D')):
    raise ValueError('Нужны 42 последовательные даты')
STAGES = [('legacy', tune_start.normalize(), dates[-28:])]
for number in range(WEEKLY_BLOCKS):
    block = weekly_days[number*7:(number+1)*7]
    STAGES.append((f'week{number+1}', block[0].normalize(), block))
if RUN_FULL_REFIT:
    STAGES.append(('full', end + STEP, pd.DatetimeIndex([end])))
stage_table = pd.DataFrame([dict(stage=s, fit_before=b, first=d.min(), last=d.max(), days=len(d))
                            for s,b,d in STAGES])
stage_table.to_csv(RUN_ROOT / 'reports' / 'stages.csv', index=False)
display(stage_table)
print('Всего fit:', len(STAGES)*8, '; weekly: фиксированные 1500 деревьев, без early stopping на будущей неделе')

lag_tables = {}
for stage, boundary, days in STAGES:
    lag_tables[stage] = calculate_dynamic_shoulder(
        target, hub_signal, times, routes, fit_before=boundary,
        max_lookback_hours=LAG_MAX_LOOKBACK_HOURS,
        min_pairs=LAG_MIN_PAIRS, min_correlation=LAG_MIN_CORRELATION)
    lag_tables[stage].to_csv(RUN_ROOT / 'lags' / f'{stage}.csv', index=False)

def build_history(h):
    path = RUN_ROOT / f'features_h{h}.partial.parquet'
    writer = None
    try:
        for start in range(0, len(times)-h, 48):
            origins = np.arange(start, min(start+48, len(times)-h))
            frame = horizon_dataset(h, origins)
            frame['y'] = target[origins+h].ravel()
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression='zstd')
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    frame = pd.read_parquet(path)
    path.unlink()
    return frame

def run_task(stage, boundary, days, h, data):
    old = completed(stage, h)
    if old is not None:
        print('Resume: skip', stage, h)
        return
    tic = time.perf_counter()
    data = add_correlation_features(data, h, lag_tables[stage])
    columns = model_columns(data)
    mask, start = recent_mask(data, boundary)
    if not (data.loc[mask, 'timestamp'] < boundary).all():
        raise ValueError('Train пересекает границу fit')
    if stage == 'full':
        query = horizon_dataset(h, np.array([len(times)-1]))
        query = add_correlation_features(query, h, lag_tables[stage])
        trees = completed('legacy', h)['trees']
    else:
        query = data.loc[data.cutoff.isin(days)].copy()
        trees = ITERATIONS
    if len(query) != len(days)*len(routes):
        raise ValueError('Неполная панель query')
    model = make_model(trees)
    x = data.loc[mask, columns]
    y = data.loc[mask, 'y'] / TARGET_SCALE
    if stage == 'legacy':
        tune = query.loc[query.cutoff < holdout_start]
        model.fit(x, y, eval_set=(tune[columns], tune.y/TARGET_SCALE),
                  early_stopping_rounds=100, use_best_model=True)
        info_through = tune.timestamp.max()
    else:
        model.fit(x, y)
        info_through = data.loc[mask, 'timestamp'].max()
    train_metrics = metrics(data.loc[mask], np.maximum(model.predict(x)*TARGET_SCALE, 0))
    del x, y
    prediction = np.maximum(model.predict(query[columns])*TARGET_SCALE, 0)
    keep = ['cutoff', 'timestamp', 'route_id', 'profile28', 'overlap']
    if stage != 'full':
        keep.append('y')
    result = query[keep].copy()
    result['horizon'] = h
    result['catboost'] = prediction
    result['seasonal'] = query[['slot_daily7', 'slot_weekly4']].mean(axis=1)
    result['raw_prediction'] = np.where(h == 1, prediction, .7*prediction+.3*query.profile28)
    result['stage'] = stage
    result['fit_before'] = boundary
    result['model_info_through'] = info_through
    result['run_id'] = RUN_ID
    result['selection_caveat'] = 'parameters_previously_selected_in_project'
    if stage == 'full':
        result = test.loc[test.timestamp.eq(end+h*STEP)].merge(
            result, on=['route_id', 'timestamp'], validate='one_to_one')
    pred_path = RUN_ROOT / 'predictions' / f'{stage}_h{h}.parquet'
    atomic_parquet(result, pred_path)
    model_path = RUN_ROOT / 'models' / f'{stage}_h{h}.cbm'
    temporary = model_path.with_suffix('.partial.cbm')
    model.save_model(str(temporary))
    temporary.replace(model_path)
    report = dict(stage=stage, horizon=h, run_id=RUN_ID, fit_before=str(boundary),
                  train_start=str(start), train_label_max=str(data.loc[mask, 'timestamp'].max()),
                  model_info_through=str(info_through), trees=model.tree_count_,
                  train=train_metrics, seconds=time.perf_counter()-tic, features=columns)
    if stage != 'full':
        report['query'] = metrics(result, prediction)
    importance_path = RUN_ROOT / 'reports' / f'{stage}_h{h}_importance.csv'
    pd.DataFrame(dict(feature=columns, importance=model.get_feature_importance())).to_csv(importance_path, index=False)
    report['files'] = {str(p.relative_to(RUN_ROOT)): file_hash(p) for p in [pred_path, model_path, importance_path]}
    save_json(RUN_ROOT / 'checkpoints' / f'{stage}_h{h}.json', report)
    print(stage, 'h=', h, 'trees=', model.tree_count_, 'seconds=', round(report['seconds']))
    display(pd.DataFrame([dict(split=k, **report[k]) for k in ['train', 'query'] if k in report]))
    del model, query, result
    gc.collect()
    archive('results.zip')

def run_all():
    for h in range(1, 9):
        if all(completed(s, h) is not None for s, _, _ in STAGES):
            print('Горизонт уже завершён:', h)
            continue
        print('Подготовка признаков h=', h)
        data = build_history(h)
        try:
            for stage, boundary, days in STAGES:
                run_task(stage, boundary, days, h, data)
        finally:
            del data
            gc.collect()
            archive('checkpoint.zip', include_models=True)
