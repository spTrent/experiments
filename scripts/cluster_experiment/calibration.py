def calibration_variants():
    return [(f'cat_k{k}', k) for k in CLUSTERS]


def initialize_calibration():
    global CAL_SPLIT, CAL_GROUP_LABELS
    if not 0 < CAL_FIT_DAYS < VAL_DAYS or not 0 < SCALE_BOUNDS[0] <= 1 <= SCALE_BOUNDS[1]:
        raise ValueError('Некорректные периоды или границы калибровки')
    load_inputs()
    CAL_SPLIT = VAL_START + pd.Timedelta(days=CAL_FIT_DAYS)
    group_segmentation = segment_routes(VAL_START, 'calibration_groups')
    CAL_GROUP_LABELS = group_segmentation['labels'][max(CLUSTERS)].copy()
    pd.DataFrame(dict(route_id=ROUTES, calibration_group=CAL_GROUP_LABELS)).to_csv(RUN / 'calibration_groups.csv', index=False)
    display(pd.DataFrame([
        dict(period='calibration_fit', start=str(VAL_START), end_exclusive=str(CAL_SPLIT)),
        dict(period='calibration_select', start=str(CAL_SPLIT), end_exclusive=str(TEST_START)),
        dict(period='test', start=str(TEST_START), end_exclusive=str(TEST_END)),
    ]))


def calibration_data(h, origins, segmentation):
    x, names = features(h, origins, segmentation)
    return dict(x=x, names=names, origins=origins,
                y=VALUES[TARGET][origins + h].reshape(-1).astype(np.float64),
                route_index=np.tile(np.arange(len(ROUTES)), len(origins)))


def prediction_frame(h, data, pred):
    return pd.DataFrame(dict(route_id=ROUTES[data['route_index']],
                             cutoff=np.repeat(TIMES[data['origins']], len(ROUTES)),
                             timestamp=np.repeat(TIMES[data['origins']] + h * STEP, len(ROUTES)),
                             h=h, calibration_group=CAL_GROUP_LABELS[data['route_index']], y=data['y'], y_pred=pred))


def scalar_ratio(frame):
    actual, predicted = float(frame.y.sum()), float(frame.y_pred.sum())
    if actual <= 0 or predicted <= 0 or not np.isfinite(actual + predicted):
        raise ValueError('Для калибровки нужны положительные конечные суммы факта и прогноза')
    return actual / predicted


def fit_factors(frame, strategy):
    if set(frame.h.unique()) != set(HORIZONS):
        raise ValueError('Калибровочная выборка должна содержать все горизонты')
    if strategy == 'raw':
        return {str(h): 1.0 for h in HORIZONS}
    if strategy in ('group', 'group_shrink50', 'group_horizon', 'group_horizon_shrink50'):
        if 'calibration_group' not in frame:
            raise ValueError('Нет фиксированных групп калибровки')
        by_horizon = strategy.startswith('group_horizon')
        baseline = fit_factors(frame, 'horizon' if by_horizon else 'global')
        factors = dict(baseline, groups={})
        shrink = 0.5 if strategy.endswith('shrink50') else 1.0
        for group, part in frame.groupby('calibration_group'):
            if group < 0 or part.route_id.nunique() < MIN_CLUSTER_ROUTES:
                continue
            values = {}
            for h in HORIZONS:
                local = part[part.h == h] if by_horizon else part
                if local.empty or local.y.sum() <= 0 or local.y_pred.sum() <= 0:
                    continue
                ratio = scalar_ratio(local)
                values[str(h)] = float(np.clip((1 - shrink) * baseline[str(h)] + shrink * ratio, *SCALE_BOUNDS))
            if values:
                factors['groups'][str(int(group))] = values
        return factors
    global_ratio = scalar_ratio(frame)
    if strategy not in ('global', 'horizon_shrink50', 'horizon'):
        raise ValueError('Неизвестный способ калибровки')
    shrink = {'global': 0.0, 'horizon_shrink50': 0.5, 'horizon': 1.0}[strategy]
    factors = {}
    for h in HORIZONS:
        local_ratio = scalar_ratio(frame[frame.h == h])
        value = (1 - shrink) * global_ratio + shrink * local_ratio
        factors[str(h)] = float(np.clip(value, *SCALE_BOUNDS))
    return factors


def corrected(frame, factors):
    multiplier = frame.h.map(lambda h: factors[str(int(h))]).to_numpy(dtype=np.float64, copy=True)
    for group, values in factors.get('groups', {}).items():
        for h, value in values.items():
            selected = (frame.calibration_group.to_numpy() == int(group)) & (frame.h.to_numpy() == int(h))
            multiplier[selected] = value
    pred = frame.y_pred.to_numpy(dtype=np.float64) * multiplier
    if not np.isfinite(pred).all() or (pred < 0).any():
        raise ValueError('Некорректный скорректированный прогноз')
    return pred


def apply_factors(raw, h, route_index, factors):
    per_route = np.full(len(ROUTES), factors[str(h)], dtype=np.float64)
    for group, values in factors.get('groups', {}).items():
        per_route[CAL_GROUP_LABELS == int(group)] = values.get(str(h), factors[str(h)])
    return raw * per_route[route_index]


def factor_records(factors):
    rows = [dict(scope='fallback', calibration_group=-1, h=h, factor=factors[str(h)]) for h in HORIZONS]
    for group, values in factors.get('groups', {}).items():
        rows.extend(dict(scope='group', calibration_group=int(group), h=int(h), factor=value) for h, value in values.items())
    return rows


def phase_forecasts(stage, boundary, query_start, query_end, frozen=None):
    if stage == 'test' and (frozen is None or not (RUN / 'calibration_frozen.json').exists()):
        raise ValueError('Test закрыт до фиксации калибровки')
    segmentation = segment_routes(boundary, stage)
    all_rows = []
    for h in HORIZONS:
        positions = np.arange(len(TIMES) - h)
        train_origins = positions[TIMES[positions + h] < boundary]
        query_mask = ((TIMES[positions] >= query_start) & (TIMES[positions] < query_end)
                      & (TIMES[positions].hour == 10) & (TIMES[positions].minute == 30)
                      & (TIMES[positions + h] < query_end))
        query_origins = positions[query_mask]
        if not len(train_origins) or not len(query_origins):
            raise ValueError('Пустой обучающий или проверочный период')
        train = calibration_data(h, train_origins, segmentation)
        query = calibration_data(h, query_origins, segmentation)
        for name, k in calibration_variants():
            path = RUN / f'{stage}_h{h}_{name}.parquet'
            metric_path = RUN / f'{stage}_h{h}_{name}_metrics.json'
            if path.exists() and metric_path.exists():
                all_rows.extend(json.loads(metric_path.read_text()))
                continue
            bundle = fit_model(train, segmentation, 'cat', k, CAT_CONFIGS[0])
            report = []
            for period, data in [('train', train), (stage, query)]:
                raw = predict_model(bundle, data, segmentation)
                report.append(dict(variant=name, h=h, strategy='raw', period=period, **metrics(data['y'], raw)))
                if frozen is not None:
                    choice = frozen['variants'][name]
                    pred = apply_factors(raw, h, data['route_index'], choice['factors'])
                    report.append(dict(variant=name, h=h, strategy='selected', period=period, **metrics(data['y'], pred)))
                if period == stage:
                    frame = prediction_frame(h, data, raw)
                    if frozen is not None:
                        frame['y_calibrated'] = pred
                    frame.to_parquet(path, index=False)
            write_json(metric_path, report)
            all_rows.extend(report)
            display(pd.DataFrame(report)[['variant', 'h', 'strategy', 'period', 'WAPE', 'relative_bias', 'score']])
            if frozen is None:
                print('Test закрыт. Калибровка и выбор используют только периоды до test.', flush=True)
            del bundle
            gc.collect()
        del train, query
        gc.collect()
    summary = aggregate(all_rows, ['variant', 'strategy', 'period'])
    summary['relative_bias'] = summary.signed / summary.volume
    pd.DataFrame(all_rows).to_csv(RUN / f'{stage}_metrics_by_horizon.csv', index=False)
    summary.to_csv(RUN / f'{stage}_metrics.csv', index=False)
    display(summary)
    return summary


def read_phase(stage, name):
    frame = pd.concat([pd.read_parquet(RUN / f'{stage}_h{h}_{name}.parquet') for h in HORIZONS], ignore_index=True)
    if frame.duplicated(['route_id', 'cutoff', 'h']).any() or not np.isfinite(frame[['y', 'y_pred']]).all().all():
        raise ValueError('Некорректные сохранённые прогнозы')
    return frame


def select_calibration():
    rows, choices, factor_rows = [], {}, []
    strategies = ('raw', 'global', 'horizon_shrink50', 'horizon',
                  'group_shrink50', 'group', 'group_horizon_shrink50', 'group_horizon')
    for name, k in calibration_variants():
        fit = read_phase('calibration_fit', name)
        select = read_phase('calibration_select', name)
        if not ((fit.cutoff >= VAL_START) & (fit.timestamp < CAL_SPLIT)).all():
            raise ValueError('Пересечение периода оценки коэффициентов с выбором')
        if not ((select.cutoff >= CAL_SPLIT) & (select.timestamp < TEST_START)).all():
            raise ValueError('Период выбора пересекает test')
        candidates = []
        for strategy in strategies:
            factors = fit_factors(fit, strategy)
            factor_rows.extend(dict(variant=name, strategy=strategy, fitted_on='calibration_fit', **row) for row in factor_records(factors))
            for period, frame in [('calibration_fit', fit), ('calibration_select', select)]:
                record = dict(variant=name, strategy=strategy, period=period,
                              **metrics(frame.y.to_numpy(), corrected(frame, factors)))
                rows.append(record)
                if period == 'calibration_select':
                    candidates.append(record)
        best = min(candidates, key=lambda row: row['score'])
        chosen_strategy = best['strategy']
        all_val = pd.concat([fit, select], ignore_index=True)
        factors = fit_factors(all_val, chosen_strategy)
        choices[name] = dict(k=k, strategy=chosen_strategy, selection_score=best['score'], factors=factors)
    winner = min(choices, key=lambda name: choices[name]['selection_score'])
    frozen = dict(winner=winner, variants=choices, base_settings=CAT_CONFIGS[0],
                  coefficient_fit_for_selection='first_val_block', selection_period='second_val_block',
                  evaluation_coefficient_fit='all_val_rolling_oof', test_used=False,
                  val_start=str(VAL_START), calibration_split=str(CAL_SPLIT), test_start=str(TEST_START),
                  calibration_groups=dict(k=max(CLUSTERS), fitted_before=str(VAL_START),
                                          routes=[str(r) for r in ROUTES], labels=CAL_GROUP_LABELS.tolist()))
    write_json(RUN / 'calibration_frozen.json', frozen)
    table = pd.DataFrame(rows)
    table.to_csv(RUN / 'calibration_selection.csv', index=False)
    pd.DataFrame(factor_rows).to_csv(RUN / 'calibration_candidates_factors.csv', index=False)
    display(table[['variant', 'strategy', 'period', 'WAPE', 'relative_bias', 'score']])
    display(pd.DataFrame([dict(variant=name, strategy=choice['strategy'], **row)
                          for name, choice in choices.items() for row in factor_records(choice['factors'])]))
    print('Победитель выбран до test:', winner, choices[winner]['strategy'], flush=True)
    return frozen


def evaluate_calibration(frozen):
    if frozen != json.loads((RUN / 'calibration_frozen.json').read_text()):
        raise ValueError('Настройки отличаются от зафиксированных до test')
    if frozen['calibration_groups']['labels'] != CAL_GROUP_LABELS.tolist() or frozen['calibration_groups']['routes'] != [str(r) for r in ROUTES]:
        raise ValueError('Изменились фиксированные группы калибровки')
    summary = phase_forecasts('test', TEST_START, TEST_START, TEST_END, frozen)
    summary['selected_on_val'] = (summary.variant == frozen['winner']) & (summary.strategy == 'selected')
    summary.to_csv(RUN / 'independent_calibration_metrics.csv', index=False)
    display(summary)
    note = 'Независимая оценка test получена до переобучения на всей доступной размеченной истории. Этот test ранее изучался в проекте и не является новым слепым holdout. Коэффициенты, способ калибровки и вариант модели зафиксированы до test. Train включает бывшую val.'
    (RUN / 'evaluation_note.txt').write_text(note)
    write_json(RUN / 'calibration_evaluation_complete.json', dict(frozen=frozen))
    print(note)
    return summary


def finalize_calibration(frozen):
    marker = RUN / 'calibration_evaluation_complete.json'
    if not marker.exists() or json.loads(marker.read_text())['frozen'] != frozen:
        raise ValueError('Сначала завершите оценку неизменённой процедуры')
    name = frozen['winner']
    choice = frozen['variants'][name]
    oof = pd.concat([read_phase(stage, name) for stage in ('calibration_fit', 'calibration_select', 'test')], ignore_index=True)
    final_factors = fit_factors(oof, choice['strategy'])
    write_json(RUN / 'final_calibration.json', dict(variant=name, strategy=choice['strategy'], factors=final_factors,
                                                   fit_period='available_rolling_oof_val_and_former_test',
                                                   independent_test_factors=choice['factors']))
    display(pd.DataFrame([dict(stage=stage, **row) for stage, factors in [('independent_test', choice['factors']), ('final', final_factors)]
                          for row in factor_records(factors)]))
    segmentation = segment_routes(TIMES[-1] + STEP, 'full')
    predictions, metric_rows = [], []
    for h in HORIZONS:
        train = calibration_data(h, np.arange(len(TIMES) - h), segmentation)
        model_path = RUN / f'calibrated_final_model_h{h}.joblib'
        if model_path.exists():
            bundle = joblib.load(model_path)
        else:
            bundle = fit_model(train, segmentation, 'cat', choice['k'], frozen['base_settings'])
            temporary = model_path.with_suffix('.tmp')
            joblib.dump(bundle, temporary, compress=3)
            temporary.replace(model_path)
        raw = predict_model(bundle, train, segmentation)
        for strategy, pred in [('raw', raw), ('selected', apply_factors(raw, h, train['route_index'], final_factors))]:
            metric_rows.append(dict(variant=name, h=h, strategy=strategy, period='full_in_sample', **metrics(train['y'], pred)))
        historical = pd.read_csv(RUN / 'test_metrics_by_horizon.csv')
        historical = historical[(historical.variant == name) & (historical.h == h)]
        display(pd.concat([historical, pd.DataFrame(metric_rows[-2:])], ignore_index=True)[
            ['variant', 'h', 'strategy', 'period', 'WAPE', 'relative_bias', 'score']])
        print('Train/test в таблице — прежние оценочные модели и прежние коэффициенты, до полного refit.', flush=True)
        x, names = features(h, np.array([len(TIMES) - 1]), segmentation)
        if names != bundle['feature_names']:
            raise ValueError('Изменился порядок признаков')
        raw_future = predict_model(bundle, dict(x=x, route_index=np.arange(len(ROUTES))), segmentation)
        predictions.append(pd.DataFrame(dict(route_id=ROUTES, timestamp=TIMES[-1] + h * STEP,
                                              y_pred=apply_factors(raw_future, h, np.arange(len(ROUTES)), final_factors), y_raw=raw_future)))
        del train, bundle
        gc.collect()
    forecast = pd.concat(predictions, ignore_index=True)
    submission = REQUESTS.merge(forecast, on=['route_id', 'timestamp'], how='left', sort=False, validate='one_to_one')
    if not submission.id.equals(REQUESTS.id) or not np.isfinite(submission[['y_pred', 'y_raw']]).all().all() or (submission[['y_pred', 'y_raw']] < 0).any().any():
        raise ValueError('Некорректный submission')
    submission[['id', 'y_pred']].to_csv(RUN / 'submission.csv', index=False)
    submission[['id', 'y_raw']].rename(columns={'y_raw': 'y_pred'}).to_csv(RUN / 'submission_raw.csv', index=False)
    aggregate(metric_rows, ['variant', 'strategy', 'period']).to_csv(RUN / 'full_calibration_metrics.csv', index=False)
    if CREATE_ARCHIVE:
        with zipfile.ZipFile(RUN.with_suffix('.zip'), 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(RUN.rglob('*')):
                if path.is_file():
                    archive.write(path, path.relative_to(RUN.parent))
    print('Сабмит:', RUN / 'submission.csv')
    print('Независимая оценка test получена до переобучения базовых моделей и обновления коэффициентов на всей доступной OOF-истории.')
