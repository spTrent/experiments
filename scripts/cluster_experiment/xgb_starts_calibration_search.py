from sklearn.isotonic import IsotonicRegression


FACTOR_GRID = tuple(round(value / 100, 2) for value in range(50, 151, 5))
MODEL_STARTS = {'five_starts': (570, 600, 630, 660, 690), 'all_half_hours': None}
FACTOR_METHODS = ('global', 'horizon', 'horizon_shrink50', 'group', 'group_shrink50',
                  'group_horizon', 'group_horizon_shrink50', 'offset', 'affine', 'affine_horizon',
                  'affine_group', 'affine_group_shrink50', 'recent_global_2', 'recent_global_3',
                  'recent_global_7', 'recent_global_14') + tuple(
    f'shrink_recent_{days}_{alpha}' for days in (3, 7, 14) for alpha in (.25, .5, .75))


def calibration_specs():
    yield dict(kind='raw')
    yield dict(kind='fixed', multiplier=1.025)
    for multiplier in FACTOR_GRID:
        yield dict(kind='fixed', multiplier=multiplier)
    for method in FACTOR_METHODS:
        for multiplier in FACTOR_GRID:
            yield dict(kind='factors', method=method, multiplier=multiplier)
    for slope in FACTOR_GRID:
        yield dict(kind='affine_grid', slope=slope)
    for kind, options in [('nonlinear', (32, 128)), ('route_horizon', (28, 56)), ('meta', range(len(META_CONFIGS)))]:
        for option in options:
            for strength in (.5, 1.):
                for multiplier in FACTOR_GRID:
                    yield dict(kind=kind, option=option, strength=strength, multiplier=multiplier)


def fit_candidate(history, spec):
    kind = spec['kind']
    state = {'kind': kind}
    if kind in ('raw', 'fixed'):
        return state
    if kind == 'factors':
        state['factors'] = fit_factors(history, spec['method'])
    elif kind == 'affine_grid':
        scale = max(float(history.y.mean()), 1.)
        x, y = history.y_pred.to_numpy(dtype=float) / scale, history.y.to_numpy(dtype=float) / scale
        fitted = spec['slope'] * x
        state['offset'] = float(brentq(lambda b: np.maximum(fitted + b, 0).sum() - y.sum(),
                                       -max(float(fitted.max()), 1.), max(float(y.mean()), 1.))) * scale
    elif kind == 'nonlinear':
        p = history.y_pred.to_numpy(dtype=float)
        edges = np.unique(np.quantile(p, np.linspace(0, 1, spec['option'] + 1)))
        bins = pd.DataFrame({'bin': np.searchsorted(edges[1:-1], p), 'p': p, 'y': history.y.to_numpy()})
        stats = bins.groupby('bin').agg(p=('p', 'mean'), y=('y', 'mean'), n=('y', 'size'))
        state['fallback'] = scalar_ratio(history)
        if len(stats) >= 2:
            model = IsotonicRegression(y_min=0., out_of_bounds='clip').fit(stats.p, stats.y, sample_weight=stats.n)
            state.update(x=model.X_thresholds_, y=model.y_thresholds_)
    elif kind == 'route_horizon':
        parent = fit_factors(history, 'horizon_shrink50')
        stats = history.groupby(['route_id', 'h']).agg(p=('y_pred', 'sum'), y=('y', 'sum'), n=('y', 'size'))
        baseline = np.array([parent[str(h)] for h in stats.index.get_level_values('h')])
        local = np.divide(stats.y.to_numpy(), stats.p.to_numpy(), out=baseline.copy(), where=stats.p.to_numpy() > 0)
        strength = stats.n.to_numpy() / (stats.n.to_numpy() + spec['option'])
        state.update(parent=parent, local=pd.Series(np.clip(baseline + strength * (local - baseline), *SCALE_BOUNDS), index=stats.index))
    elif kind == 'meta':
        state['model'] = fit_meta(history, META_CONFIGS[spec['option']])
    else:
        raise ValueError(kind)
    return state


def candidate_base_prediction(frame, spec, state):
    p = frame.y_pred.to_numpy(dtype=float)
    kind = state['kind']
    if kind in ('raw', 'fixed'):
        return p
    if kind == 'factors':
        return corrected(frame, state['factors'])
    if kind == 'affine_grid':
        return np.maximum(spec['slope'] * p + state['offset'], 0)
    if kind == 'meta':
        return p * apply_meta(state['model'], frame, spec['strength'])
    if kind == 'route_horizon':
        index = pd.MultiIndex.from_frame(frame[['route_id', 'h']])
        factor = state['local'].reindex(index).to_numpy()
        parent = frame.h.map(lambda h: state['parent'][str(h)]).to_numpy()
        adjusted = p * np.where(np.isfinite(factor), factor, parent)
    elif 'x' in state:
        x, y = state['x'], state['y']
        adjusted = np.interp(p, x, y)
        for mask, k in ((p < x[0], 0), (p > x[-1], -1)):
            adjusted[mask] = p[mask] * (y[k] / x[k] if x[k] > 0 else state['fallback'])
    else:
        adjusted = p * state['fallback']
    return p + spec['strength'] * (adjusted - p)


def select_branch_calibration(name, fit, select):
    if fit.timestamp.max() >= CAL_SPLIT or select.cutoff.min() < CAL_SPLIT or select.timestamp.max() >= TEST_START:
        raise ValueError('Нарушены границы подбора')
    rows = []
    last_fit_key = last_prediction_key = None
    for spec in calibration_specs():
        fit_key = tuple((key, value) for key, value in spec.items() if key not in ('multiplier', 'strength'))
        prediction_key = (fit_key, spec.get('strength'))
        if fit_key != last_fit_key:
            state = fit_candidate(fit, spec)
            last_fit_key = fit_key
        if prediction_key != last_prediction_key:
            base = candidate_base_prediction(select, spec, state)
            last_prediction_key = prediction_key
        predicted = base * spec.get('multiplier', 1.)
        rows.append(dict(model=name, spec=json.dumps(spec, sort_keys=True), **metrics(select.y.to_numpy(), predicted)))
    table = pd.DataFrame(rows).sort_values('score', kind='stable').reset_index(drop=True)
    table.to_csv(RUN / f'{name}_calibration_search.csv', index=False)
    winner = json.loads(table.iloc[0].spec)
    print(name, 'кандидатов', len(table), 'выбор по val', winner, 'score', table.iloc[0].score, flush=True)
    all_val = pd.concat([fit, select], ignore_index=True)
    return winner, fit_candidate(all_val, winner)


def evaluate_raw(raw, h, data, selected):
    raw_metrics = metrics(data['y'], raw)
    if selected is None:
        return raw_metrics, raw
    spec, state = selected
    frame = prediction_frame(h, data, raw)
    calibrated = candidate_base_prediction(frame, spec, state) * spec.get('multiplier', 1.)
    return metrics(data['y'], calibrated), calibrated


def run_stage(stage, boundary, end, choices=None):
    segmentation = segment_routes(boundary, stage)
    rows, forecasts = [], {name: [] for name in MODEL_STARTS}
    for h in HORIZONS:
        all_origins, query_origins = scope_origins(h, boundary, boundary, end)
        minutes = TIMES[all_origins].hour * 60 + TIMES[all_origins].minute
        query = calibration_data(h, query_origins, segmentation)
        for name, starts in MODEL_STARTS.items():
            train_origins = all_origins if starts is None else all_origins[np.isin(minutes, starts)]
            train = calibration_data(h, train_origins, segmentation)
            path = RUN / f'{stage}_{name}_h{h}.joblib'
            if path.exists():
                bundle = joblib.load(path)
                if bundle.get('fit_code') != CODE_HASH or bundle.get('fit_boundary') != str(boundary):
                    raise ValueError('Несовместимый checkpoint')
            else:
                bundle = fit_model(train, segmentation, 'xgb', 8, {})
                bundle.update(fit_code=CODE_HASH, fit_boundary=str(boundary))
                temporary_path = path.with_suffix('.tmp')
                joblib.dump(bundle, temporary_path, compress=3)
                temporary_path.replace(path)
            morning = (TIMES[train_origins].hour == 10) & (TIMES[train_origins].minute == 30)
            keep = np.repeat(morning, len(ROUTES))
            train_eval = dict(x=train['x'][keep], y=train['y'][keep], route_index=train['route_index'][keep], origins=train_origins[morning])
            del train
            spec = None if choices is None else choices[name]
            train_metrics, _ = evaluate_raw(predict_model(bundle, train_eval, segmentation), h, train_eval, spec)
            del train_eval
            raw_query = predict_model(bundle, query, segmentation)
            query_metrics, calibrated = evaluate_raw(raw_query, h, query, spec)
            forecasts[name].append(prediction_frame(h, query, raw_query).assign(calibrated=calibrated))
            block = [dict(model=name, h=h, period=period, **values) for period, values in
                     [('train', train_metrics), ('test' if choices is not None else 'val', query_metrics)]]
            rows.extend(block)
            display(pd.DataFrame(block)[['model', 'h', 'period', 'WAPE', 'relative_bias', 'score']])
            del bundle, raw_query, calibrated
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
        del query
    table = pd.DataFrame(rows)
    table.to_csv(RUN / f'{stage}_fit_metrics.csv', index=False)
    result = {name: pd.concat(parts, ignore_index=True) for name, parts in forecasts.items()}
    for name, frame in result.items():
        frame.to_parquet(RUN / f'{stage}_{name}_predictions.parquet', index=False)
    return result, table


def run_independent_calibrations():
    global META_SEGMENTATION, META_LABELS
    with threadpool_limits(limits=THREADS):
        initialize_calibration()
        META_SEGMENTATION = segment_routes(VAL_START, 'meta_fixed')
        META_LABELS = CAL_GROUP_LABELS.copy()
        history = {name: [] for name in MODEL_STARTS}
        for index in range(3):
            boundary = VAL_START + pd.Timedelta(days=index * 7)
            block, _ = run_stage(f'oof_week_{index}', boundary, boundary + pd.Timedelta(days=7))
            for name in MODEL_STARTS:
                history[name].append(block[name])
        select, _ = run_stage('calibration_select', CAL_SPLIT, TEST_START)
        choices = {name: select_branch_calibration(name, pd.concat(history[name], ignore_index=True), select[name])
                   for name in MODEL_STARTS}
        for name, (spec, state) in choices.items():
            joblib.dump(state, RUN / f'{name}_calibrator.joblib', compress=3)
        write_json(RUN / 'calibration_frozen.json', dict(test_used_for_selection=False, train_val_target_end_exclusive=str(TEST_START),
                  models={name: spec for name, (spec, state) in choices.items()}, grid=list(FACTOR_GRID)))
        _, final = run_stage('train_val', TEST_START, TEST_END, choices)
    totals = aggregate(final.to_dict('records'), ['model', 'period'])
    totals['relative_bias'] = totals.signed / totals.volume
    summary = totals.pivot(index='model', columns='period', values=['WAPE', 'relative_bias', 'WAPE + |Relative Bias|'])
    summary.columns = [f'{period} {metric}' for metric, period in summary.columns]
    columns = [f'{period} {metric}' for period in ('train', 'test') for metric in ('WAPE', 'relative_bias', 'WAPE + |Relative Bias|')]
    summary = summary[columns].reset_index().set_index('model').loc[list(MODEL_STARTS)].reset_index()
    summary.to_csv(RUN / 'final_train_test.csv', index=False)
    note = 'Train и test оцениваются из 10:30. Обе базы переобучены на train + val, калибровки обучены на OOF val. Локальный test получен до полного переобучения и не использован в fit или выборе. Период ранее изучался.'
    (RUN / 'evaluation_note.txt').write_text(note)
    with zipfile.ZipFile(RUN.with_suffix('.zip'), 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=2) as archive:
        for path in sorted(RUN.rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(RUN.parent))
    print(note)
    print(RUN.with_suffix('.zip'))
    display(summary)
