def keep_recent_origins(origins, horizon, model):
    first = len(origins) // 4
    recent = origins[first:]
    if not len(recent):
        raise ValueError(f'Нет обучающих стартов для {model} h{horizon}')
    print(model, f'h{horizon}', f'стартов {len(recent)}/{len(origins)}', flush=True)
    return recent


def fit_recent_lightgbm(predictor, base, boundary):
    source = base / 'server_output/lightgbm86'
    prepared = base / 'prepared_features'
    folder = ARTIFACTS / 'lightgbm86_recent75'
    folder.mkdir(exist_ok=True)
    path = folder / 'predictions.parquet'
    if path.is_file():
        return pd.read_parquet(path)
    cfg = json.loads((source / 'config.json').read_text())
    data = pd.read_parquet(prepared / 'features_cb101246dc27_starts5.parquet')
    extra = pd.read_parquet(prepared / 'extra_long_5_0_f3390bfbe3_6a081832b4f4.parquet')
    if len(data) != len(extra) or data.duplicated(KEYS).any():
        raise ValueError('Подготовленные признаки LightGBM не совпадают')
    from core_robust import extend_robust
    data = extend_robust(pd.concat([data.reset_index(drop=True), extra.reset_index(drop=True)], axis=1), cfg)
    del extra
    future = predictor.data(cfg)
    outputs = []
    for h in HORIZONS:
        saved = joblib.load(source / f'h{h}/model_bundle.joblib')
        columns = saved['columns']
        part = data.loc[data.horizon.eq(h)]
        train = part.loc[part.timestamp.lt(boundary) & part.y.notna()]
        starts = pd.DatetimeIndex(train.cutoff.unique()).sort_values()
        recent = starts[keep_recent_origins(np.arange(len(starts)), h, 'LightGBM86')]
        train = train.loc[train.cutoff.isin(recent)]
        model = lgb.LGBMRegressor(**dict(saved['model'].get_params(), n_estimators=saved['trees']))
        model.fit(train[columns], train.y / train.scale, sample_weight=train.scale / train.scale.mean())
        train_prediction = np.maximum(model.predict(train[columns]) * train.scale.to_numpy(), 0)
        local = part.loc[part.cutoff.ge(LOCAL_TEST_START) & part.cutoff.lt(LOCAL_TEST_END) & part.y.notna() & part.cutoff.dt.hour.eq(10) & part.cutoff.dt.minute.eq(30)]
        if SUBMISSION_SCOPE == 'train_val':
            local_prediction = np.maximum(model.predict(local[columns]) * local.scale.to_numpy(), 0)
            refit_report('lightgbm86_recent75', boundary, h, 'all', 'test', train.y, train_prediction, local.y, local_prediction)
        else:
            refit_report('lightgbm86_recent75', boundary, h, 'all', 'future', train.y, train_prediction, [], [])
        check = part.loc[part.cutoff.ge('2025-10-04') & part.cutoff.lt(LOCAL_TEST_END) & part.cutoff.dt.hour.eq(10) & part.cutoff.dt.minute.eq(30)]
        query = pd.concat([check, future.loc[future.horizon.eq(h)]], ignore_index=True)
        prediction = np.maximum(model.predict(query[columns]) * query.scale.to_numpy(), 0)
        outputs.append(query[KEYS].assign(core_reproduced=prediction))
        model.booster_.save_model(str(folder / f'h{h}.txt'))
    result = pd.concat(outputs, ignore_index=True)
    save_parquet(path, result)
    return result


def save_exact_volume_variant():
    frame = pd.read_parquet(ARTIFACTS / 'future_predictions.parquet')
    canonical = pd.read_csv(EXACT_ROOT / 'submission_session6.csv')
    keys = REQUESTS.merge(canonical.rename(columns={'y_pred': 'exact_session6'}), on='id', validate='one_to_one')
    frame = keys.merge(frame[['route_id', 'timestamp', 'y_pred']], on=['route_id', 'timestamp'], validate='one_to_one', sort=False)
    totals = frame.groupby('timestamp')[['y_pred', 'exact_session6']].transform('sum')
    if len(frame) != len(REQUESTS) or not frame.id.equals(REQUESTS.id) or totals.le(0).any().any():
        raise ValueError('Не совпали конкурсные ключи или объёмы')
    prediction = frame.y_pred / totals.y_pred * totals.exact_session6
    if not np.isfinite(prediction).all() or prediction.lt(0).any():
        raise ValueError('Некорректный прогноз с архивным объёмом')
    path = WORK_ROOT / f'submission_{SUBMISSION_SCOPE}_exact_session6_volume.csv'
    pending = path.with_suffix('.partial')
    frame[['id']].assign(y_pred=prediction).to_csv(pending, index=False)
    pending.replace(path)
    print(path, flush=True)
    if SUBMISSION_SCOPE == 'train_val':
        local = pd.read_parquet(ARTIFACTS / 'test_predictions.parquet')
        anchor = pd.read_parquet(EXACT_ROOT / 'validation_predictions.parquet', columns=KEYS + ['session_prediction'])
        local = local.merge(anchor, on=KEYS, validate='one_to_one')
        if len(local) != len(anchor):
            raise ValueError('Не совпали ключи архивной локальной оценки')
        totals = local.groupby(GROUP_KEYS)[['blend_prediction', 'session_prediction']].transform('sum')
        adjusted = local.blend_prediction / totals.blend_prediction * totals.session_prediction
        table = pd.DataFrame([dict(variant=name, **metric(local.y, values)) for name, values in [
            ('recent75_ridge', local.blend_prediction), ('recent75_exact_volume', adjusted), ('exact_session6', local.session_prediction)]])
        table.to_csv(ARTIFACTS / 'exact_volume_test_comparison.csv', index=False)
        display(table)


import os

RF_START_COUNTS = (5, 9) if os.environ.get('BLEND_RF_SKIP_48') == '1' else (5, 9, 48)


def rf_start_minutes(count):
    if count not in RF_START_COUNTS:
        raise ValueError(f'Неизвестное число стартов RF: {count}')
    if count == 48:
        return np.arange(0, 24 * 60, 30)
    return 630 + np.arange(-(count // 2), count // 2 + 1) * 30


def select_rf_window(runtime):
    folder = SUBMISSION_ROOT / 'rf_window_selection'
    folder.mkdir(exist_ok=True)
    marker = folder / 'selection.json'
    if marker.is_file():
        saved = json.loads(marker.read_text())
        if (saved['code'] != SUBMIT_CODE_HASH or saved['starts'] not in RF_START_COUNTS
                or tuple(saved.get('candidates', (5, 9, 48))) != RF_START_COUNTS):
            raise ValueError('Несовместимый выбор окна RF')
        print('Окно RF из сохранённой validation:', saved['starts'], flush=True)
        return saved['starts']
    history = META_HISTORY
    target_times = history.cutoff + history.horizon * STEP
    rows = []
    for start, stop in OOF_WINDOWS[2:]:
        train = history.loc[target_times.lt(start)].reset_index(drop=True)
        valid = history.loc[history.cutoff.ge(start) & history.cutoff.lt(stop)].reset_index(drop=True)
        if train.empty or valid.empty or not valid.cutoff.lt(LOCAL_TEST_START).all():
            raise ValueError('Некорректные границы validation для RF')
        matrix = shape_matrix(train)
        joint = fit_weights(matrix, train.y)
        shape = fit_shape(FROZEN['shape'], train, matrix, joint)
        train_prediction = predict_shape(FROZEN['shape'], shape, train, matrix, joint)
        query_origins = TIMES.get_indexer(pd.DatetimeIndex(valid.cutoff.unique()))
        if (query_origins < 0).any():
            raise ValueError('Не найдены старты validation')
        for count in RF_START_COUNTS:
            candidate = folder / f'rf_{count}'
            candidate.mkdir(exist_ok=True)
            runtime['RF_START_COUNT'] = count
            runtime['QUERY_ORIGINS'] = query_origins
            runtime['RUN'] = candidate
            def report(model, boundary, horizon, group, split, train_y, train_pred, query_y, query_pred):
                table = pd.DataFrame([
                    dict(model=model, starts=count, horizon=horizon, period='train', **metric(train_y, train_pred)),
                    dict(model=model, starts=count, horizon=horizon, period='val', **metric(query_y, query_pred)),
                ])
                display(table)
                table.to_csv(candidate / f'{boundary:%Y%m%d}_h{horizon}_metrics.csv', index=False)
            runtime['fit_report'] = report
            prediction = runtime['fit_rf'](start, stop, 'val')
            variant = join_predictions(valid.drop(columns='rf'), {'rf': prediction})
            shape_prediction = predict_shape(FROZEN['shape'], shape, variant, shape_matrix(variant), joint)
            row = dict(week=str(start.date()), starts=count, period='val', **metric(variant.y, shape_prediction))
            rows.append(row)
            display(pd.DataFrame([dict(week=str(start.date()), starts=count, period='train_oof_reference',
                                       **metric(train.y, train_prediction)), row]))
    scores = pd.DataFrame(rows)
    scores.to_csv(folder / 'validation_metrics.csv', index=False)
    ranking = scores.groupby('starts', as_index=False).agg(mean_score=('score', 'mean'),
                worst_score=('score', 'max'), weeks=('score', 'count'))
    if len(ranking) != len(RF_START_COUNTS) or not ranking.weeks.eq(2).all() or not np.isfinite(ranking.mean_score).all():
        raise ValueError('Неполное сравнение окон RF')
    ranking = ranking.sort_values(['mean_score', 'worst_score', 'starts'])
    ranking.to_csv(folder / 'validation_ranking.csv', index=False)
    display(ranking)
    winner = int(ranking.iloc[0].starts)
    write_json(marker, dict(code=SUBMIT_CODE_HASH, starts=winner, candidates=list(RF_START_COUNTS), test_used=False, anchor='session6'))
    return winner


def fit_selected_rf(runtime):
    if SUBMISSION_SCOPE == 'train_val':
        winner = select_rf_window(runtime)
    else:
        saved = json.loads((SUBMISSION_ROOT / 'rf_window_selection/selection.json').read_text())
        if (saved['code'] != SUBMIT_CODE_HASH or saved['starts'] not in RF_START_COUNTS
                or tuple(saved.get('candidates', (5, 9, 48))) != RF_START_COUNTS):
            raise ValueError('Несовместимый выбор окна RF')
        winner = saved['starts']
    runtime.update(RF_START_COUNT=winner, QUERY_ORIGINS=QUERY_ORIGINS, fit_report=refit_report,
                   RUN=ARTIFACTS / f'rf_{winner}')
    runtime['RUN'].mkdir(exist_ok=True)
    write_json(ARTIFACTS / 'rf_window.json', dict(starts=winner, minutes=rf_start_minutes(winner).tolist()))
    print('Выбранный RF:', winner, 'стартов', flush=True)
    return runtime['fit_rf'](FULL_BOUNDARY, FULL_BOUNDARY + pd.Timedelta(days=1), 'future')
