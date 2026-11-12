def read_xgb_archive(stage, boundary, end):
    with zipfile.ZipFile(XGB_ARCHIVE) as archive:
        prefix = XGB_ARCHIVE.stem + '/'
        manifest = json.loads(archive.read(prefix + 'manifest.json'))
        current = json.loads((RUN / 'manifest.json').read_text())
        previous = manifest['cat_configs'][0]
        if manifest['train_hash'] != current['train_hash'] or manifest['test_hash'] != current['test_hash'] or previous['parameters'] != PARAMETERS:
            raise ValueError('Архив XGBoost использует другие входы или параметры')
        protocol = json.loads(archive.read(prefix + 'protocol.json'))
        evaluation = json.loads(archive.read(prefix + 'evaluation_complete.json'))
        allowed = {'oof_week_2': pd.Timestamp(protocol['val_start']) + pd.Timedelta(days=14),
                   'calibration_select': pd.Timestamp(protocol['calibration_split']),
                   'train_val': pd.Timestamp(evaluation['train_val_model_target_end_exclusive'])}
        if allowed[stage] != boundary or protocol['model_clusters'] != [8] or manifest['clusters'] != list(CLUSTERS) or manifest['seed'] != SEED:
            raise ValueError('Границы или настройки кластеров архива отличаются')
        parts = [pd.read_parquet(BytesIO(archive.read(prefix + f'{stage}_h{h}_xgb_k8.parquet'))) for h in HORIZONS]
    frame = pd.concat(parts, ignore_index=True).rename(columns={'h': 'horizon', 'y_pred': 'xgb_all'})
    check_panel(frame, boundary, end)
    return frame


def check_panel(frame, start, end):
    expected = pd.MultiIndex.from_product([
        pd.date_range(start + pd.Timedelta(hours=10, minutes=30), end, freq='D', inclusive='left'),
        HORIZONS, ROUTES], names=keys)
    actual = pd.MultiIndex.from_frame(frame[keys])
    if actual.has_duplicates or len(actual) != len(expected) or len(expected.difference(actual)):
        raise ValueError('Неполная панель: ' + str(start))
    if not frame.timestamp.eq(frame.cutoff + pd.to_timedelta(frame.horizon * 30, unit='m')).all():
        raise ValueError('Некорректные timestamps целей')


def fit_five_starts(stage, boundary, end):
    segmentation = segment_routes(boundary, stage)
    reports, parts = [], []
    for h in HORIZONS:
        train_origins, query_origins = scope_origins(h, boundary, boundary, end)
        train_origins = train_origins[np.isin(minutes[train_origins], MORNING_STARTS)]
        if not len(train_origins):
            raise ValueError('Нет обучающих утренних стартов')
        train = calibration_data(h, train_origins, segmentation)
        query = calibration_data(h, query_origins, segmentation)
        path = RUN / f'{stage}_five_h{h}.joblib'
        if path.exists():
            bundle = joblib.load(path)
            if bundle['fit_boundary'] != str(boundary) or bundle['fit_code'] != CODE_HASH:
                raise ValueError('Несовместимый checkpoint')
        else:
            bundle = fit_model(train, segmentation, 'xgb', 8, {})
            bundle.update(fit_boundary=str(boundary), fit_code=CODE_HASH)
            temporary_path = path.with_suffix('.tmp')
            joblib.dump(bundle, temporary_path, compress=3)
            temporary_path.replace(path)
        train_prediction = predict_model(bundle, train, segmentation)
        query_prediction = predict_model(bundle, query, segmentation)
        parts.append(prediction_frame(h, query, query_prediction).rename(columns={'h': 'horizon', 'y_pred': 'xgb_five'}))
        rows = [dict(horizon=h, period=period, **metrics(y, pred)) for period, y, pred in
                [('train', train['y'], train_prediction), ('test' if stage == 'test' else 'val', query['y'], query_prediction)]]
        reports.extend(rows)
        display(pd.DataFrame(rows))
        pd.DataFrame(reports).to_csv(RUN / f'{stage}_five_fit_metrics.csv', index=False)
        del train, query, bundle, train_prediction, query_prediction
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    frame = pd.concat(parts, ignore_index=True)
    check_panel(frame, boundary, end)
    return frame


def merge_predictions(left, right, column):
    merged = left.merge(right[keys + ['timestamp', 'y', column]], on=keys, how='outer',
                        validate='one_to_one', suffixes=('', '_incoming'))
    if len(merged) != len(left) or len(merged) != len(right) or not np.array_equal(merged.y, merged.y_incoming) or not merged.timestamp.eq(merged.timestamp_incoming).all():
        raise ValueError('Не совпали ключи, факты или timestamp: ' + column)
    if not np.isfinite(merged[column]).all() or merged[column].lt(0).any():
        raise ValueError('Некорректный прогноз: ' + column)
    return merged.drop(columns=['y_incoming', 'timestamp_incoming'])


def fixed_volume_mix(frame):
    result = frame.copy()
    anchor = result.volume_model.to_numpy(dtype=float)
    group = ['cutoff', 'horizon']
    volume = result.groupby(group).volume_model.transform('sum').to_numpy(dtype=float)
    for name in ('all', 'five'):
        raw = result[f'xgb_{name}'].to_numpy(dtype=float)
        total = result.groupby(group)[f'xgb_{name}'].transform('sum').to_numpy(dtype=float)
        normalized = np.divide(raw * volume, total, out=anchor.copy(), where=total > 0)
        replacement = (1 - weight) * anchor + weight * normalized
        result[f'mix_{name}'] = .75 * result.session_prediction + .25 * replacement
    totals = result.groupby(group)[['volume_model', 'session_prediction', 'mix_all', 'mix_five']].sum()
    if not np.allclose(totals[['session_prediction', 'mix_all', 'mix_five']], totals.volume_model.to_numpy()[:, None], atol=1e-5, rtol=1e-10):
        raise ValueError('Изменился объём горизонта')
    return result


def volume_factor(frame, column, alpha):
    total_pred = float(frame[column].sum())
    if total_pred <= 0:
        raise ValueError('Нет объёма для калибровки')
    ratio = float(np.clip(frame.y.sum() / total_pred, .95, 1.05))
    return 1 + alpha * (ratio - 1)


def rows_for(frame, period, columns):
    return [dict(period=period, variant=column, **metrics(frame.y.to_numpy(), frame[column].to_numpy())) for column in columns]


def choose_on_val(val):
    shape_rows = []
    weeks = [('val_week1', val[val.cutoff < VAL_SECOND]), ('val_week2', val[val.cutoff >= VAL_SECOND])]
    columns = ('session_prediction', 'xgb_all', 'xgb_five', 'mix_all', 'mix_five')
    for period, part in [*weeks, ('val_all', val)]:
        shape_rows.extend(rows_for(part, period, columns))
    table = show_metrics(shape_rows, 'val_shape_metrics.csv')
    candidates = table[table.variant.isin(['mix_all', 'mix_five']) & table.period.isin(['val_week1', 'val_week2'])]
    selected = candidates.groupby('variant').WAPE.mean().idxmin()
    first, second = (part for _, part in weeks)
    calibration = []
    for alpha in (0., .25, .5, 1.):
        factor = volume_factor(first, selected, alpha)
        calibration.append(dict(alpha=alpha, factor_fit_on_week1=factor,
                                **metrics(second.y.to_numpy(), second[selected].to_numpy() * factor)))
    show_metrics(calibration, 'val_volume_calibration.csv')
    alpha = min(calibration, key=lambda row: row['score'])['alpha']
    selection = dict(shape=selected, alpha=alpha, factor=volume_factor(val, selected, alpha),
                     fit_through=str(val.timestamp.max()), test_used_for_selection=False)
    write_json(RUN / 'selection_frozen.json', selection)
    return selection


def make_val_anchor():
    folder = RUN / 'anchor_oof.parquet'
    if folder.exists():
        return pd.read_parquet(folder)
    prepare_history(VALUES)
    parts = [base_predictions('session6_oof', h) for h in HORIZONS]
    base = pd.concat(parts, ignore_index=True).sort_values(keys).reset_index(drop=True)
    corrected = []
    for start, end in ((VAL_FIRST, VAL_SECOND), (VAL_SECOND, TEST_START)):
        block = base[base.cutoff < end].reset_index(drop=True)
        pred = finish_pipeline(block, 'session6_oof', start)
        query = pred[pred.cutoff >= start].rename(columns={'prediction': 'session_prediction'})
        query = query.merge(block[keys + ['timestamp']], on=keys, validate='one_to_one')
        check_panel(query, start, end)
        corrected.append(query)
    val = pd.concat(corrected, ignore_index=True)
    val.to_parquet(folder, index=False)
    return val


def run_starts_comparison():
    global RUN, out, CAL_GROUP_LABELS, times, routes, positions, minutes, eval_origins
    global start_tune, original_target, target, values, names, meta, weight
    load_inputs()
    if TEST_START != pd.Timestamp('2025-10-18') or TEST_END != pd.Timestamp('2025-11-01'):
        raise ValueError('Архивы относятся к локальному test 18–31 октября')
    out = RUN
    times, routes, values = TIMES, ROUTES, VALUES
    original_target = target = VALUES[TARGET]
    names = [TARGET, *STATUSES]
    positions = np.arange(len(times))
    minutes = times.hour * 60 + times.minute
    eval_origins = positions[(times >= ANCHOR_SEED) & (times < TEST_START) & (minutes == 630)]
    start_tune = ANCHOR_SEED
    CAL_GROUP_LABELS = np.zeros(len(routes), dtype=int)
    with zipfile.ZipFile(SESSION_ARCHIVE) as archive:
        meta = json.loads(archive.read('comparison_meta.json'))
    weight = float(meta['weight'])
    (RUN / 'models' / 'session6_oof').mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=THREADS):
        val_anchor = make_val_anchor()
        val_parts = []
        for stage, boundary, end in [('oof_week_2', VAL_FIRST, VAL_SECOND), ('calibration_select', VAL_SECOND, TEST_START)]:
            anchor = val_anchor[val_anchor.cutoff.ge(boundary) & val_anchor.cutoff.lt(end)].copy()
            joined = merge_predictions(anchor, read_xgb_archive(stage, boundary, end), 'xgb_all')
            val_parts.append(merge_predictions(joined, fit_five_starts(stage, boundary, end), 'xgb_five'))
        val = fixed_volume_mix(pd.concat(val_parts, ignore_index=True))
        val.to_parquet(RUN / 'val_predictions.parquet', index=False)
        selection = choose_on_val(val)
        with zipfile.ZipFile(SESSION_ARCHIVE) as archive:
            anchor = pd.read_parquet(BytesIO(archive.read('validation_predictions.parquet')))
            volume = pd.read_parquet(BytesIO(archive.read('evaluation_prediction_values.parquet')), columns=keys + ['calibrated_prediction'])
        check_panel(anchor, TEST_START, TEST_END)
        anchor = anchor.merge(volume.rename(columns={'calibrated_prediction': 'volume_model'}), on=keys, how='left', validate='one_to_one')
        joined = merge_predictions(anchor, read_xgb_archive('train_val', TEST_START, TEST_END), 'xgb_all')
        test = fixed_volume_mix(merge_predictions(joined, fit_five_starts('test', TEST_START, TEST_END), 'xgb_five'))
    test['selected_calibrated'] = test[selection['shape']] * selection['factor']
    columns = ('session_prediction', 'xgb_all', 'xgb_five', 'mix_all', 'mix_five', 'selected_calibrated')
    show_metrics(rows_for(test, 'test', columns), 'test_metrics.csv')
    rows = []
    for label, parts in [('horizon', test.groupby('horizon')), ('day', test.groupby('cutoff')), ('week', test.groupby(test.cutoff.dt.day.ge(25)))]:
        for value, part in parts:
            rows.extend(dict(slice=label, value=str(value), **row) for row in rows_for(part, 'test', columns))
    pd.DataFrame(rows).to_csv(RUN / 'test_slices.csv', index=False)
    test.to_parquet(RUN / 'test_predictions.parquet', index=False)
    (RUN / 'evaluation_note.txt').write_text('Локальный test оценён после выбора по val и до полного переобучения. Выборка ранее изучалась. На val используется восстановленный причинный Session6, на test архивный объём и прогноз Session6. Новый сабмит в этом опыте не собирается.')
    with zipfile.ZipFile(RUN.with_suffix('.zip'), 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=2) as archive:
        for path in sorted(RUN.rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(RUN.parent))
    print((RUN / 'evaluation_note.txt').read_text())
    print(RUN.with_suffix('.zip'))
