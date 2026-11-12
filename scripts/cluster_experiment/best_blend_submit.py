def locate_evaluation(path=None):
    required = ['inputs.json', 'base_oof.parquet', 'test_predictions.parquet',
                'train_test_metrics.csv', 'frozen_selection.json']
    if path is None:
        candidates = set()
        for root in [Path('/kaggle/input'), Path('/kaggle/working')]:
            candidates.update(p.parent for p in root.rglob('frozen_selection.json')
                              if all((p.parent / name).is_file() for name in required))
        if len(candidates) != 1:
            raise ValueError(f'Укажите EVALUATION_DIR: найдено {sorted(map(str, candidates))}')
        path = candidates.pop()
    path = Path(path)
    if not all((path / name).is_file() for name in required):
        raise FileNotFoundError(f'Неполный output блендинга: {path}')
    return path


def initialize_submission(evaluation_path=None, session_path=None, xgb_path=None):
    global EVALUATION, ARTIFACTS, FROZEN, HISTORICAL_METRICS, FULL_BOUNDARY, REQUESTS, META_HISTORY
    global FUTURE_ORIGIN, FUTURE_CUTOFF, SUBMISSION_ROOT, ALL_META_HISTORY
    EVALUATION = locate_evaluation(evaluation_path)
    contract = json.loads((EVALUATION / 'inputs.json').read_text())
    globals()['NOTEBOOK_HASH'] = contract['code']
    initialize(session_path, xgb_path)
    for key in ['session_sha256', 'xgb_sha256']:
        actual = json.loads((ARTIFACTS / 'inputs.json').read_text())[key]
        if actual != contract[key]:
            raise ValueError(f'Входы отличаются от завершённого запуска: {key}')
    FROZEN = json.loads((EVALUATION / 'frozen_selection.json').read_text())
    if FROZEN['variant'] != 'cat_d4__ridge' or FROZEN['test_used_for_selection']:
        raise ValueError('Ожидался зафиксированный до test cat_d4__ridge')
    digest = hashlib.sha256((SUBMIT_CODE_HASH + json.dumps(contract, sort_keys=True) +
                            digest_file(EVALUATION / 'base_oof.parquet') +
                            digest_file(EVALUATION / 'test_predictions.parquet')).encode()).hexdigest()[:12]
    ARTIFACTS = Path('/kaggle/working') / f'blend_submit_{digest}'
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    write_json(ARTIFACTS / 'frozen_selection.json', FROZEN)
    SUBMISSION_ROOT = ARTIFACTS
    FULL_BOUNDARY = TIMES[-1] + STEP
    FUTURE_ORIGIN = np.array([len(TIMES) - 1])
    FUTURE_CUTOFF = TIMES[-1]
    REQUESTS = pd.read_parquet(SESSION_DATA / 'test_solo_track.parquet', columns=['id', 'route_id', 'timestamp'])
    REQUESTS['timestamp'] = pd.to_datetime(REQUESTS.timestamp)
    expected = pd.MultiIndex.from_product([ROUTES, [FUTURE_CUTOFF + h * STEP for h in HORIZONS]])
    actual = pd.MultiIndex.from_frame(REQUESTS[['route_id', 'timestamp']])
    if not REQUESTS.id.is_unique or actual.has_duplicates or len(actual) != len(expected) or len(expected.difference(actual)):
        raise ValueError('Конкурсный test не соответствует восьми будущим горизонтам')
    columns = KEYS + ['y', *BASES, 'level24', 'current', 'delta1h', 'delta2h', 'network_current']
    META_HISTORY = pd.concat([pd.read_parquet(EVALUATION / name)[columns]
                             for name in ['base_oof.parquet', 'test_predictions.parquet']], ignore_index=True)
    origins = TIMES.get_indexer(META_HISTORY.cutoff)
    routes = pd.Index(ROUTES).get_indexer(META_HISTORY.route_id)
    targets = origins + META_HISTORY.horizon.to_numpy()
    if (META_HISTORY.duplicated(KEYS).any() or (origins < 0).any() or (routes < 0).any()
            or (targets >= len(TIMES)).any() or not np.array_equal(META_HISTORY.y, VALUES[TARGET][targets, routes])):
        raise ValueError('Некорректные исторические прогнозы для финальной метамодели')
    if not np.isfinite(META_HISTORY[columns[3:]].to_numpy(dtype=float)).all():
        raise ValueError('Пропуски в исторических прогнозах')
    ALL_META_HISTORY = META_HISTORY.copy()
    HISTORICAL_METRICS = pd.read_csv(EVALUATION / 'train_test_metrics.csv')
    HISTORICAL_METRICS.to_csv(ARTIFACTS / 'metrics_before_full_refit.csv', index=False)
    display(HISTORICAL_METRICS.loc[HISTORICAL_METRICS.variant.eq(FROZEN['variant'])])
    print('Это оценка до полного переобучения. После включения local test в fit новой независимой оценки нет.')


def enter_submission_scope(scope):
    global ARTIFACTS, FULL_BOUNDARY, META_HISTORY, SUBMISSION_SCOPE, QUERY_ORIGINS
    if scope not in ('train_val', 'full_history'):
        raise ValueError(f'Неизвестный режим: {scope}')
    if scope == 'full_history':
        marker = SUBMISSION_ROOT / 'train_val' / 'evaluation_complete.json'
        if not marker.is_file() or not (SUBMISSION_ROOT / 'train_val' / 'submission.csv').is_file():
            raise RuntimeError('Сначала завершите новую оценку train_val на test и первый сабмит')
        audit = json.loads(marker.read_text())
        if audit['code'] != SUBMIT_CODE_HASH or audit['boundary'] != str(LOCAL_TEST_START):
            raise ValueError('Оценка test относится к другому коду или границе')
    SUBMISSION_SCOPE = scope
    FULL_BOUNDARY = LOCAL_TEST_START if scope == 'train_val' else TIMES[-1] + STEP
    test_origins = np.flatnonzero((TIMES >= LOCAL_TEST_START) & (TIMES < LOCAL_TEST_END) & (TIMES.hour == 10) & (TIMES.minute == 30))
    QUERY_ORIGINS = np.concatenate([test_origins, FUTURE_ORIGIN]) if scope == 'train_val' else FUTURE_ORIGIN.copy()
    ARTIFACTS = SUBMISSION_ROOT / scope
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    source_history = ALL_META_HISTORY
    if scope == 'full_history':
        recent = pd.read_parquet(SUBMISSION_ROOT / 'train_val' / 'test_predictions.parquet')
        early = ALL_META_HISTORY.loc[(ALL_META_HISTORY.cutoff + ALL_META_HISTORY.horizon * STEP).lt(LOCAL_TEST_START)]
        source_history = pd.concat([early, recent[early.columns]], ignore_index=True)
    target_times = source_history.cutoff + source_history.horizon * STEP
    META_HISTORY = source_history.loc[target_times.lt(FULL_BOUNDARY)].reset_index(drop=True)
    if META_HISTORY.empty or not (META_HISTORY.cutoff + META_HISTORY.horizon * STEP).lt(FULL_BOUNDARY).all():
        raise ValueError('Некорректная граница обучения метамодели')
    write_json(ARTIFACTS / 'frozen_selection.json', FROZEN)
    HISTORICAL_METRICS.to_csv(ARTIFACTS / 'metrics_before_full_refit.csv', index=False)
    write_json(ARTIFACTS / 'submission_protocol.json', dict(
        code=SUBMIT_CODE_HASH, evaluation=str(EVALUATION), frozen=FROZEN, scope=scope,
        training_target_end_exclusive=str(FULL_BOUNDARY), feature_cutoff=str(FUTURE_CUTOFF),
        local_test_used_for_fit=scope == 'full_history',
        meta_rows=len(META_HISTORY), meta_last_target=str((META_HISTORY.cutoff + META_HISTORY.horizon * STEP).max()),
        historical_observations_used_for_future_features=True,
        test_evaluated_before_full_refit=True, new_independent_test_score=scope == 'train_val',
        session6_refit='bases and residuals restricted to training boundary; EWMA uses only pre-boundary residuals',
        session6_cat_training_starts='eligible[len(eligible) // 4:] as in original recipe',
        train_val_authorization='explicit user request for comparison submission'))
    print('Режим:', scope, 'обучение на целях строго до', FULL_BOUNDARY,
          'признаки на момент', FUTURE_CUTOFF, 'OOF-строк', len(META_HISTORY), flush=True)


def refit_report(model, boundary, horizon, group, split, train_y, train_pred, query_y, query_pred):
    row = dict(model=model, horizon=horizon, group=group, period=f'{SUBMISSION_SCOPE}_train', **metric(train_y, train_pred))
    rows = [row]
    actual = np.asarray(query_y).ravel()
    predicted = np.asarray(query_pred).ravel()
    known = np.isfinite(actual)
    if SUBMISSION_SCOPE == 'train_val' and known.any():
        rows.append(dict(model=model, horizon=horizon, group=group, period='test',
                         **metric(actual[known], predicted[known])))
    table = pd.DataFrame(rows)
    display(table)
    table.to_csv(ARTIFACTS / f'refit_{model}_h{horizon}_{group}.csv', index=False)


def future_context():
    y = VALUES[TARGET]
    pos = len(TIMES) - 1
    windows = np.arange(-47, 1)
    level = y[pos + windows].mean(axis=0, dtype=np.float64)
    seasonal = pos - 48 * np.arange(1, 29)
    historical_level = np.stack([y[p + windows].mean(axis=0, dtype=np.float64) for p in seasonal])
    ratio = level / np.maximum(historical_level, 1)
    return pd.concat([pd.DataFrame(dict(cutoff=FUTURE_CUTOFF, horizon=h, route_id=ROUTES,
        level24=level, current=y[pos], delta1h=y[pos] - y[pos - 2], delta2h=y[pos] - y[pos - 4],
        network_current=y[pos].sum(dtype=np.float64), profile28=(y[seasonal + h] * ratio).mean(axis=0)))
        for h in HORIZONS], ignore_index=True).sort_values(KEYS).reset_index(drop=True)


def observed_query(h):
    result = np.full((len(QUERY_ORIGINS), len(ROUTES)), np.nan)
    known = QUERY_ORIGINS + h < len(TIMES)
    result[known] = VALUES[TARGET][QUERY_ORIGINS[known] + h]
    return result


def join_predictions(frame, parts):
    for name, incoming in parts.items():
        selected = incoming.loc[incoming.cutoff.isin(frame.cutoff.unique()), KEYS + [name]]
        if selected.duplicated(KEYS).any() or len(selected) != len(frame):
            raise ValueError(f'Неверные ключи: {name}')
        frame = frame.merge(selected, on=KEYS, how='left', validate='one_to_one')
    if not np.isfinite(frame[BASES]).all().all() or frame[BASES].lt(0).any().any():
        raise ValueError('Некорректные базовые прогнозы')
    return frame


def finish_submission(parts):
    matrix = shape_matrix(META_HISTORY)
    joint = fit_weights(matrix, META_HISTORY.y)
    shape = fit_shape(FROZEN['shape'], META_HISTORY, matrix, joint)
    volumes = fit_volumes(META_HISTORY)
    def predict(frame):
        values = predict_shape(FROZEN['shape'], shape, frame, shape_matrix(frame), joint)
        return values * volume_multipliers(frame, volumes)[FROZEN['volume']]
    fit_prediction = predict(META_HISTORY)
    if SUBMISSION_SCOPE == 'train_val':
        test = join_predictions(context(LOCAL_TEST_START, LOCAL_TEST_END), parts)
        test_prediction = predict(test)
        refit_report(FROZEN['variant'], FULL_BOUNDARY, 'all', 'all', 'test',
                     META_HISTORY.y, fit_prediction, test.y, test_prediction)
        rows = [dict(model=FROZEN['variant'], period=period, **metric(y, pred))
                for period, y, pred in [('train_oof', META_HISTORY.y, fit_prediction), ('test', test.y, test_prediction)]]
        rows.extend(dict(model=name, period='test', **metric(test.y, test[name])) for name in BASES)
        table = pd.DataFrame(rows)
        table.to_csv(ARTIFACTS / 'train_test_metrics.csv', index=False)
        display(table)
        test['blend_prediction'] = test_prediction
        save_parquet(ARTIFACTS / 'test_predictions.parquet', test)
        daily = [dict(day=str(day), **metric(frame.y, frame.blend_prediction))
                 for day, frame in test.groupby(test.cutoff.dt.normalize())]
        pd.DataFrame(daily).to_csv(ARTIFACTS / 'test_daily_metrics.csv', index=False)
        write_json(ARTIFACTS / 'evaluation_complete.json', dict(
            code=SUBMIT_CODE_HASH, boundary=str(FULL_BOUNDARY),
            test_evaluated_before_full_refit=True, same_models_for_test_and_submission=True,
            test_used_for_selection=False, **metric(test.y, test_prediction)))
        print('Новая оценка test завершена до полного переобучения. Эти же модели делают первый сабмит.')
    else:
        refit_report(FROZEN['variant'], FULL_BOUNDARY, 'all', 'all', 'future', META_HISTORY.y, fit_prediction, [], [])
    joblib.dump(dict(frozen=FROZEN, bases=BASES, joint=joint, shape=shape, volumes=volumes), ARTIFACTS / 'submission_meta.joblib')
    frame = join_predictions(future_context(), parts)
    frame['timestamp'] = frame.cutoff + frame.horizon * STEP
    frame['y_pred'] = predict(frame)
    result = REQUESTS.merge(frame[['route_id', 'timestamp', 'y_pred']], on=['route_id', 'timestamp'],
                            how='left', sort=False, validate='one_to_one')
    if not result.id.equals(REQUESTS.id) or not np.isfinite(result.y_pred).all() or result.y_pred.lt(0).any():
        raise ValueError('Неверный submission')
    result[['id', 'y_pred']].to_csv(ARTIFACTS / 'submission.csv', index=False)
    destination = SUBMISSION_ROOT.parent / f'submission_{SUBMISSION_SCOPE}.csv'
    result[['id', 'y_pred']].to_csv(destination, index=False)
    save_parquet(ARTIFACTS / 'future_predictions.parquet', frame)
    pd.DataFrame([dict(model=base, weight=float(w)) for base, w in zip(BASES, joint)]).to_csv(ARTIFACTS / 'full_meta_weights.csv', index=False)
    archive_report()
    print('Готово:', destination, result.shape)
    print('Независимая от полного переобучения оценка test получена до переобучения; период test ранее изучался.')


def session_full():
    global start_tune, start_test, eval_origins, out
    out = ARTIFACTS / 'session6_full'
    out.mkdir(exist_ok=True)
    cached = out / 'future.parquet'
    if cached.is_file():
        return pd.read_parquet(cached)
    start_tune = OOF_START - pd.Timedelta(days=7)
    prepare_history(values)
    origins = positions[(times >= start_tune) & (times.hour == 10) & (times.minute == 30) &
                        (times + max(HORIZONS) * STEP < FULL_BOUNDARY)]
    historical = []
    for h in HORIZONS:
        path = EVALUATION / 'session6' / 'models' / 'early_anchor' / f'catboost_h{h}.cbm'
        if not path.is_file():
            raise FileNotFoundError(f'Подключите полный output, включая {path.name}')
        model = CatBoostRegressor()
        model.load_model(str(path))
        f = make_cat_frame(origins, h)
        columns = cat_columns + (SEASONAL_FEATURES if h in (5, 6, 8) else [])
        part = f[['cutoff', 'route_id', 'timestamp', 'profile28', *anchor_columns]].copy()
        part['horizon'] = h
        part['y'] = part['fit_y'] = target[origins + h].ravel()
        part['model_prediction'] = np.maximum(model.predict(f[columns]) * 100000, 0)
        part['raw_blend'] = part.model_prediction if h == 1 else .7 * part.model_prediction + .3 * part.profile28
        part['shape'] = 1.
        historical.append(part)
        del model, f
    history = pd.concat(historical, ignore_index=True)
    save_parquet(out / 'residual_training.parquet', history)
    start_tune = FULL_BOUNDARY
    start_test = FULL_BOUNDARY if SUBMISSION_SCOPE == 'train_val' else FUTURE_CUTOFF
    eval_origins = QUERY_ORIGINS
    (out / 'models' / 'full').mkdir(parents=True, exist_ok=True)
    prepare_history(values)
    future = []
    for h in HORIZONS:
        path = out / f'base_h{h}.parquet'
        if not path.is_file():
            save_parquet(path, base_predictions('full', h))
        future.append(pd.read_parquet(path))
    combined = pd.concat([history, *future], ignore_index=True)
    result = finish_pipeline(combined, 'full', FULL_BOUNDARY)
    result = result.loc[result.cutoff.isin(TIMES[QUERY_ORIGINS]), keys + ['prediction']].rename(columns={'prediction': 'session6'})
    save_parquet(cached, result)
    return result


def session_refit_report(branch, stage, h, split, y, p):
    key = (stage, h)
    if split == 'train':
        SESSION_TRAIN_REPORTS[key] = (np.asarray(y), np.asarray(p))
    elif key in SESSION_TRAIN_REPORTS:
        train_y, train_pred = SESSION_TRAIN_REPORTS.pop(key)
        refit_report('session6', FULL_BOUNDARY, h, stage, split, train_y, train_pred, y, p)


def fit_rich_xgb_full():
    folder = ARTIFACTS / 'xgb_rich_full'
    folder.mkdir(exist_ok=True)
    outputs = [folder / f'h{h}.parquet' for h in HORIZONS]
    if all(p.is_file() for p in outputs):
        return pd.concat([pd.read_parquet(p) for p in outputs], ignore_index=True)
    with threadpool_limits(limits=THREADS):
        segmentation = segment_routes(FULL_BOUNDARY, 'full_xgb')
    manifest = json.loads((XGB_DATA / 'manifest.json').read_text())
    parameters = manifest['cat_configs'][0]['parameters']
    for h, path in zip(HORIZONS, outputs):
        if path.is_file():
            continue
        print('Full XGB rich', h, flush=True)
        origins = np.flatnonzero(TIMES[:len(TIMES) - h] + h * STEP < FULL_BOUNDARY)
        matrix, names = rich_matrix(h, origins, segmentation)
        query, query_names = rich_matrix(h, QUERY_ORIGINS, segmentation)
        if names != query_names:
            raise ValueError('Разные признаки XGB rich')
        train = dict(x=matrix.reshape(-1, len(names)), names=names,
                     route_index=np.tile(np.arange(len(ROUTES)), len(origins)), y=VALUES[TARGET][origins + h].ravel())
        globals()['PARAMETERS'] = parameters
        globals()['ACTIVE'] = dict(parameter_set='selected_xgb', residual=False)
        globals()['ACTIVE_H'] = h
        bundle = fit_model(train, segmentation, 'xgb', 8, {})
        pred = predict_model(bundle, train, segmentation)
        future = predict_model(bundle, dict(x=query.reshape(-1, len(names)), route_index=np.tile(np.arange(len(ROUTES)), len(QUERY_ORIGINS))), segmentation)
        refit_report('xgb_rich', FULL_BOUNDARY, h, 'all', 'future', train['y'], pred, observed_query(h).ravel(), future)
        for group, item in bundle['models'].items():
            item['model'].save_model(str(folder / f'h{h}_group{group}.ubj'))
        joblib.dump(segmentation, folder / 'segmentation.joblib')
        save_parquet(path, pd.DataFrame(dict(cutoff=TIMES[QUERY_ORIGINS].repeat(len(ROUTES)), horizon=h, route_id=np.tile(ROUTES, len(QUERY_ORIGINS)), xgb_rich=future)))
        del matrix, query, train, pred, bundle
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    return pd.concat([pd.read_parquet(p) for p in outputs], ignore_index=True)
