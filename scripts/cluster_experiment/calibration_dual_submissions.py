def show_metrics(rows, filename):
    table = pd.DataFrame(rows)
    table.to_csv(RUN / filename, index=False)
    display(table)
    print(table.to_string(index=False), flush=True)
    return table


def scope_origins(h, boundary, start, end):
    positions = np.arange(len(TIMES) - h)
    train = positions[TIMES[positions + h] < boundary]
    query = positions[(TIMES[positions] >= start) & (TIMES[positions + h] < end)
                      & (TIMES[positions].hour == 10) & (TIMES[positions].minute == 30)]
    if not len(train) or not len(query) or TIMES[train + h].max() >= boundary:
        raise ValueError('Нарушены временные границы')
    return train, query


def base_forecast(stage, boundary, start, end, variants, future=False):
    segmentation = segment_routes(boundary, stage)
    rows, futures = [], {}
    for h in HORIZONS:
        train_origins, query_origins = scope_origins(h, boundary, start, end)
        train = calibration_data(h, train_origins, segmentation)
        for name, k in variants:
            model_path = RUN / f'base_{stage}_h{h}_{name}.joblib'
            if model_path.exists():
                bundle = joblib.load(model_path)
                if bundle.get('fit_boundary') != str(boundary) or bundle.get('fit_code') != CODE_HASH:
                    raise ValueError('Несовместимый checkpoint')
            else:
                bundle = fit_model(train, segmentation, 'xgb', k, {})
                bundle.update(fit_boundary=str(boundary), fit_code=CODE_HASH)
                temporary = model_path.with_suffix('.tmp')
                joblib.dump(bundle, temporary, compress=3)
                temporary.replace(model_path)
            raw_train = predict_model(bundle, train, segmentation)
            query = calibration_data(h, query_origins, segmentation)
            raw_query = predict_model(bundle, query, segmentation)
            for period, y, pred in [('train', train['y'], raw_train), ('test' if stage == 'train_val' else stage, query['y'], raw_query)]:
                rows.append(dict(stage=stage, variant=name, h=h, period=period, method='raw', **metrics(y, pred)))
            prediction_frame(h, query, raw_query).to_parquet(RUN / f'{stage}_h{h}_{name}.parquet', index=False)
            if future:
                x, names = features(h, np.array([len(TIMES) - 1]), segmentation)
                if names != bundle['feature_names']:
                    raise ValueError('Нарушен порядок признаков')
                raw = predict_model(bundle, dict(x=x, route_index=np.arange(len(ROUTES))), segmentation)
                futures.setdefault(name, []).append(pd.DataFrame(dict(route_id=ROUTES, cutoff=TIMES[-1],
                    timestamp=TIMES[-1] + h * STEP, h=h, calibration_group=CAL_GROUP_LABELS, y_pred=raw)))
                del x, raw
            show_metrics(rows[-2:], f'fit_report_{stage}_h{h}_{name}.csv')
            del bundle, query, raw_train, raw_query
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
        del train
        gc.collect()
    show_metrics(rows, f'base_{stage}_metrics_by_horizon.csv')
    return {name: pd.concat(parts, ignore_index=True) for name, parts in futures.items()}


def read_oof(stage, name):
    parts = [pd.read_parquet(RUN / f'{stage}_h{h}_{name}.parquet') for h in HORIZONS]
    frame = pd.concat(parts, ignore_index=True)
    if frame.duplicated(['route_id', 'cutoff', 'h']).any():
        raise ValueError('Повторные OOF-прогнозы')
    return frame


def gpu_ratio(frame, half_life=None, alpha=1.):
    y = cp.asarray(frame.y.to_numpy(), dtype=cp.float64)
    pred = cp.asarray(frame.y_pred.to_numpy(), dtype=cp.float64)
    if float(cp.sum(pred)) <= 0 or float(cp.sum(y)) <= 0:
        raise ValueError('Нет положительного объёма для калибровки')
    long_ratio = cp.sum(y) / cp.sum(pred)
    if half_life is None:
        ratio = long_ratio
    else:
        age = cp.asarray((frame.cutoff.max() - frame.cutoff).dt.total_seconds().to_numpy() / 86400)
        weight = cp.exp2(-age / half_life)
        recent_ratio = cp.sum(y * weight) / cp.sum(pred * weight)
        ratio = (1 - alpha) * long_ratio + alpha * recent_ratio
    return float(cp.clip(ratio, *SCALE_BOUNDS))


def smoothing_factor(frame, strategy):
    if strategy == 'raw':
        return 1.
    if strategy == 'fixed_1.05':
        return 1.05
    if strategy == 'global':
        return gpu_ratio(frame)
    if strategy.startswith('recent_global_'):
        return gpu_ratio(frame, float(strategy.rsplit('_', 1)[-1]))
    _, _, half_life, alpha = strategy.split('_')
    return gpu_ratio(frame, float(half_life), float(alpha))


def save_submission(frame, prediction, filename):
    output = REQUESTS.merge(frame[['route_id', 'timestamp']].assign(y_pred=prediction), on=['route_id', 'timestamp'],
                            how='left', sort=False, validate='one_to_one')
    if not output.id.equals(REQUESTS.id) or not np.isfinite(output.y_pred).all() or (output.y_pred < 0).any():
        raise ValueError('Некорректный submission')
    output[['id', 'y_pred']].to_csv(RUN / filename, index=False)


def meta_features(frame):
    origins = TIMES.get_indexer(pd.DatetimeIndex(frame.cutoff))
    route_index = pd.Index(ROUTES).get_indexer(frame.route_id)
    if (origins < 0).any() or (route_index < 0).any():
        raise ValueError('Неизвестный cutoff или маршрут метамодели')
    scale = META_SEGMENTATION['scale'][route_index]
    raw = frame.y_pred.to_numpy()
    current = VALUES[TARGET][origins, route_index]
    columns = ['meta_raw_normalized', 'meta_log_raw', 'meta_log_scale', 'meta_target_current',
               'meta_target_mean_4', 'meta_target_mean_48', 'meta_target_delta_1h',
               'meta_status_1', 'meta_status_2', 'meta_status_3', 'meta_horizon', 'meta_group']
    matrix = np.column_stack([raw / scale, np.log1p(raw), np.log1p(scale), current / scale,
        ROLL[4][origins, route_index] / scale, ROLL[48][origins, route_index] / scale,
        (current - VALUES[TARGET][np.maximum(origins - 2, 0), route_index]) / scale,
        *[VALUES[f'status_{i}'][origins, route_index] / scale for i in (1, 2, 3)],
        frame.h.to_numpy() - 1, META_LABELS[route_index] + 1]).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError('Неконечные метапризнаки')
    return matrix, columns


def fit_meta(frame, config):
    x, columns = meta_features(frame)
    scales = META_SEGMENTATION['scale'][pd.Index(ROUTES).get_indexer(frame.route_id)]
    valid = frame.y_pred.to_numpy() > np.maximum(scales * META_MIN_PRED_SCALE, 1.)
    if not valid.any():
        raise ValueError('Нет положительных прогнозов для обучения коэффициента')
    raw = cp.asarray(frame.y_pred.to_numpy()[valid], dtype=cp.float32)
    target = cp.asarray(frame.y.to_numpy()[valid], dtype=cp.float32) / raw
    weights = raw / cp.mean(raw)
    device_x = cp.asarray(x[valid], dtype=cp.float32)
    types = ['float'] * len(columns)
    for c in ('meta_horizon', 'meta_group'):
        types[columns.index(c)] = 'c'
    matrix = xgb.QuantileDMatrix(device_x, label=target, weight=weights, feature_names=columns,
                                feature_types=types, enable_categorical=True, max_bin=GPU_MAX_BIN)
    settings = dict(config)
    trees = settings.pop('trees')
    model = xgb.train(dict(settings, device='cuda:0', tree_method='hist', objective='reg:squarederror',
                     max_bin=GPU_MAX_BIN, max_cat_to_onehot=1, nthread=THREADS, seed=SEED), matrix, num_boost_round=trees)
    if not json.loads(model.save_config())['learner']['generic_param']['device'].startswith('cuda'):
        raise RuntimeError('Метамодель переключилась на CPU')
    result = dict(model=model, config=config, feature_names=columns, fallback=gpu_ratio(frame))
    del matrix, device_x, raw, target, weights
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    return result


def meta_coefficients(model, frame):
    x, columns = meta_features(frame)
    if columns != model['feature_names']:
        raise ValueError('Нарушен порядок метапризнаков')
    factors = np.empty(len(frame), dtype=np.float64)
    model['model'].set_param(dict(device='cuda:0'))
    for start in range(0, len(frame), GPU_BATCH_ROWS):
        batch = cp.asarray(x[start:start + GPU_BATCH_ROWS], dtype=cp.float32)
        values = model['model'].inplace_predict(batch, validate_features=False)
        if not isinstance(values, cp.ndarray):
            raise RuntimeError('Метапредсказания не на CUDA')
        factors[start:start + len(batch)] = cp.asnumpy(cp.clip(values, *SCALE_BOUNDS))
    scale = META_SEGMENTATION['scale'][pd.Index(ROUTES).get_indexer(frame.route_id)]
    small = frame.y_pred.to_numpy() <= np.maximum(scale * META_MIN_PRED_SCALE, 1.)
    factors[small] = model['fallback']
    return factors


def apply_meta(model, frame, strength):
    learned = meta_coefficients(model, frame)
    return np.clip((1 - strength) * model['fallback'] + strength * learned, *SCALE_BOUNDS)


def fit_correction(frame, choice):
    if choice['kind'] == 'scalar':
        return dict(kind='scalar', factor=smoothing_factor(frame, choice['strategy']))
    return dict(kind='meta', model=fit_meta(frame, META_CONFIGS[choice['config']]), strength=choice['strength'])


def predict_correction(correction, frame):
    factor = np.full(len(frame), correction['factor']) if correction['kind'] == 'scalar' else apply_meta(correction['model'], frame, correction['strength'])
    return frame.y_pred.to_numpy() * factor, factor


def choose_correction():
    rows = []
    for name, k in calibration_variants():
        fit, select = read_oof('calibration_fit', name), read_oof('calibration_select', name)
        if fit.timestamp.max() >= CAL_SPLIT or select.timestamp.max() >= TEST_START:
            raise ValueError('Выбор калибровки пересекает будущий период')
        for strategy in CALIBRATION_STRATEGIES:
            factor = smoothing_factor(fit, strategy)
            for period, frame in [('calibration_fit', fit), ('calibration_select', select)]:
                rows.append(dict(variant=name, k=k, kind='scalar', strategy=strategy, config=-1, strength=0.,
                                  period=period, **metrics(frame.y.to_numpy(), frame.y_pred.to_numpy() * factor)))
        if CORRECTION_MODE == 'meta':
            for index, config in enumerate(META_CONFIGS):
                model = fit_meta(fit, config)
                factors = {period: meta_coefficients(model, frame) for period, frame in [('calibration_fit', fit), ('calibration_select', select)]}
                report = []
                for strength in META_STRENGTHS:
                    for period, frame in [('calibration_fit', fit), ('calibration_select', select)]:
                        factor = np.clip((1 - strength) * model['fallback'] + strength * factors[period], *SCALE_BOUNDS)
                        row = dict(variant=name, k=k, kind='meta', strategy='learned_coefficient', config=index,
                                   strength=strength, period=period, **metrics(frame.y.to_numpy(), frame.y_pred.to_numpy() * factor))
                        rows.append(row)
                        report.append(row)
                show_metrics(report, f'meta_fit_report_{name}_{index}.csv')
                joblib.dump(model, RUN / f'meta_selection_{name}_{index}.joblib', compress=3)
                del model
                gc.collect()
    show_metrics(rows, 'calibration_selection.csv')
    selected = min([r for r in rows if r['period'] == 'calibration_select'], key=lambda r: r['score'])
    frozen = dict(selected=selected, selection_end_exclusive=str(TEST_START), test_used=False,
                  protocol='three_weekly_oof_fit_then_one_week_selection', mode=CORRECTION_MODE)
    write_json(RUN / 'calibration_frozen.json', frozen)
    print('Выбор ДО test:', selected, flush=True)
    return selected


def full_refit_future(choice):
    segmentation = segment_routes(TIMES[-1] + STEP, 'full')
    parts, rows = [], []
    for h in HORIZONS:
        train = calibration_data(h, np.arange(len(TIMES) - h), segmentation)
        model_path = RUN / f'base_full_h{h}_{choice["variant"]}.joblib'
        if model_path.exists():
            bundle = joblib.load(model_path)
            if bundle.get('fit_code') != CODE_HASH or bundle.get('fit_boundary') != str(TIMES[-1] + STEP):
                raise ValueError('Неверный full checkpoint')
        else:
            bundle = fit_model(train, segmentation, 'xgb', choice['k'], {})
            bundle.update(fit_code=CODE_HASH, fit_boundary=str(TIMES[-1] + STEP))
            temporary = model_path.with_suffix('.tmp')
            joblib.dump(bundle, temporary, compress=3)
            temporary.replace(model_path)
        raw = predict_model(bundle, train, segmentation)
        rows.append(dict(h=h, period='full_in_sample', method='raw', **metrics(train['y'], raw)))
        del train, raw
        gc.collect()
        x, names = features(h, np.array([len(TIMES) - 1]), segmentation)
        prediction = predict_model(bundle, dict(x=x, route_index=np.arange(len(ROUTES))), segmentation)
        parts.append(pd.DataFrame(dict(route_id=ROUTES, cutoff=TIMES[-1], timestamp=TIMES[-1] + h * STEP,
                                      h=h, calibration_group=CAL_GROUP_LABELS, y_pred=prediction)))
        show_metrics(rows[-1:], f'full_fit_report_h{h}.csv')
        historical = pd.read_csv(RUN / 'independent_metrics.csv')
        print('Оценка train/test до full refit:', flush=True)
        print(historical.to_string(index=False), flush=True)
        del bundle, x
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    show_metrics(rows, 'full_in_sample_metrics.csv')
    return pd.concat(parts, ignore_index=True)


def run_dual_submissions():
    global ACTIVE, PIPELINE_VERSION, CAT_CONFIGS, CAL_SPLIT, CAL_GROUP_LABELS, META_SEGMENTATION, META_LABELS, FEATURE_NS
    ACTIVE = EXPERIMENTS[0]
    PIPELINE_VERSION = 'dual_gpu_calibration_v1_' + CORRECTION_MODE
    CAT_CONFIGS = [dict(experiment=ACTIVE, parameters=PARAMETERS, mode=CORRECTION_MODE)]
    with threadpool_limits(limits=THREADS):
        initialize_calibration()
        META_LABELS = CAL_GROUP_LABELS.copy()
        write_json(RUN / 'protocol.json', dict(mode=CORRECTION_MODE, parameter_sources=PARAMETER_SOURCES,
            gpu_data='CuPy batches -> CUDA QuantileDMatrix', test_start=str(TEST_START), val_start=str(VAL_START),
            calibration_split=str(CAL_SPLIT), model_clusters=list(MODEL_CLUSTERS),
            meta_configs=META_CONFIGS, strengths=list(META_STRENGTHS), scale_bounds=list(SCALE_BOUNDS),
            test_used_for_selection=False, train_val_submission='no_local_test_in_fit', full_submission='all_labelled_history'))
        variants = calibration_variants()
        for index in range(3):
            start = VAL_START + pd.Timedelta(days=index * 7)
            stage = f'oof_week_{index}'
            base_forecast(stage, start, start, start + pd.Timedelta(days=7), variants)
        for h in HORIZONS:
            for name, k in variants:
                parts = [pd.read_parquet(RUN / f'oof_week_{i}_h{h}_{name}.parquet') for i in range(3)]
                pd.concat(parts, ignore_index=True).to_parquet(RUN / f'calibration_fit_h{h}_{name}.parquet', index=False)
        base_forecast('calibration_select', CAL_SPLIT, CAL_SPLIT, TEST_START, variants)
        META_SEGMENTATION = segment_routes(VAL_START, 'meta_fixed')
        meta_feature_ns = FEATURE_NS
        choice = choose_correction()
        all_val = pd.concat([read_oof(s, choice['variant']) for s in ('calibration_fit', 'calibration_select')], ignore_index=True)
        correction = fit_correction(all_val, choice)
        joblib.dump(correction, RUN / 'correction_train_val.joblib', compress=3)
        write_json(RUN / 'test_correction_frozen.json', dict(choice=choice, fit_through=str(all_val.timestamp.max()), test_used=False))
        futures = base_forecast('train_val', TEST_START, TEST_START, TEST_END, [(choice['variant'], choice['k'])], future=True)
        FEATURE_NS = meta_feature_ns
        test = read_oof('train_val', choice['variant'])
        reports = []
        for period, frame in [('meta_train_oof_in_sample', all_val), ('test', test)]:
            calibrated, factors = predict_correction(correction, frame)
            for method, prediction in [('raw', frame.y_pred.to_numpy()), ('calibrated', calibrated)]:
                reports.append(dict(period=period, method=method, **metrics(frame.y.to_numpy(), prediction)))
            frame.assign(calibrated=calibrated, factor=factors).to_parquet(RUN / f'{period}_calibration_predictions.parquet', index=False)
        show_metrics(reports, 'independent_metrics.csv')
        future = futures[choice['variant']]
        pred, factors = predict_correction(correction, future)
        save_submission(future, pred, 'submission_train_val.csv')
        save_submission(future, future.y_pred.to_numpy(), 'submission_train_val_raw.csv')
        future.assign(calibrated=pred, factor=factors).to_parquet(RUN / 'future_train_val_predictions.parquet', index=False)
        write_json(RUN / 'evaluation_complete.json', dict(choice=choice, test_evaluated_before_full_refit=True,
            train_val_model_target_end_exclusive=str(TEST_START), test_used_for_selection=False))
        all_oof = pd.concat([all_val, test], ignore_index=True)
        final_correction = fit_correction(all_oof, choice)
        joblib.dump(final_correction, RUN / 'correction_full.joblib', compress=3)
        final_oof_prediction, final_oof_factor = predict_correction(final_correction, all_oof)
        final_meta_report = dict(period='full_oof_calibrator_in_sample', method='calibrated',
                                **metrics(all_oof.y.to_numpy(), final_oof_prediction))
        show_metrics(reports + [final_meta_report], 'final_calibrator_fit_report.csv')
        all_oof.assign(calibrated=final_oof_prediction, factor=final_oof_factor).to_parquet(RUN / 'full_oof_calibrator_in_sample_predictions.parquet', index=False)
        final_future = full_refit_future(choice)
        FEATURE_NS = meta_feature_ns
        pred, factors = predict_correction(final_correction, final_future)
        save_submission(final_future, pred, 'submission_full.csv')
        save_submission(final_future, final_future.y_pred.to_numpy(), 'submission_full_raw.csv')
        final_future.assign(calibrated=pred, factor=factors).to_parquet(RUN / 'future_full_predictions.parquet', index=False)
        note = 'Test оценён до полного refit, выбор модели/калибровки только по val. Этот test уже изучался в проекте. submission_train_val использует базы target timestamp < TEST_START и калибратор только на OOF val. submission_full использует базы со всей истории и калибратор на всех доступных OOF val+бывший test без повторного выбора. Мета-признаки используют фиксированные статистики до val для одинакового масштаба метамодели. Все fit и model inference получают CUDA-матрицы; parquet, признаки и отчёты готовятся на CPU.'
        (RUN / 'evaluation_note.txt').write_text(note)
        show_metrics(reports, 'final_report_historical_metrics.csv')
        print(note, flush=True)
    archive_path = RUN.with_suffix('.zip')
    with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(RUN.rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(RUN.parent))
    print('Основные сабмиты:', RUN / 'submission_train_val.csv', RUN / 'submission_full.csv', flush=True)
    print('Итоговый архив:', archive_path, flush=True)
