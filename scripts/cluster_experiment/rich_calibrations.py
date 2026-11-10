from scipy.optimize import brentq, minimize_scalar

legacy_fit_factors = fit_factors
CALIBRATION_STRATEGIES = ('raw', 'fixed_1.025', 'fixed_1.05', 'global', 'horizon', 'horizon_shrink50',
                          'group', 'group_shrink50', 'group_horizon', 'group_horizon_shrink50',
                          'recent_global_2', 'recent_global_3', 'recent_global_7',
                          'offset', 'affine', 'affine_horizon', 'affine_group', 'affine_group_shrink50')


def affine_pair(frame, offset_only=False):
    scale = max(float(frame.y.mean()), 1.)
    x = frame.y_pred.to_numpy(dtype=np.float64) / scale
    y = frame.y.to_numpy(dtype=np.float64) / scale
    if not np.isfinite(x).all() or not np.isfinite(y).all() or y.sum() <= 0:
        raise ValueError('Некорректные данные affine-калибровки')

    def intercept(a):
        return float(brentq(lambda b: np.maximum(a * x + b, 0).sum() - y.sum(),
                            -max(float((a * x).max()), 1.), max(float(y.mean()), 1.), xtol=1e-10))

    def objective(a):
        b = intercept(a)
        error = np.maximum(a * x + b, 0) - y
        return float((np.abs(error).sum() + abs(error.sum())) / y.sum())

    if offset_only or np.ptp(x) < 1e-10:
        a = 1.
    else:
        search = minimize_scalar(objective, bounds=AFFINE_SLOPE_BOUNDS, method='bounded', options=dict(xatol=1e-5))
        candidates = [float(search.x), 1., *AFFINE_SLOPE_BOUNDS]
        a = min(candidates, key=objective)
    return float(a), intercept(a) * scale


def fit_factors(frame, strategy):
    if strategy.startswith('fixed_'):
        return {str(h): float(strategy.split('_', 1)[1]) for h in HORIZONS}
    if strategy not in ('offset', 'affine', 'affine_horizon', 'affine_group', 'affine_group_shrink50'):
        return legacy_fit_factors(frame, strategy)
    a, b = affine_pair(frame, strategy == 'offset')
    result = {str(h): a for h in HORIZONS}
    result['offsets'] = {str(h): b for h in HORIZONS}
    if strategy == 'affine_horizon':
        for h in HORIZONS:
            a, b = affine_pair(frame[frame.h == h])
            result[str(h)], result['offsets'][str(h)] = a, b
    if strategy.startswith('affine_group'):
        result['groups'], result['group_offsets'] = {}, {}
        strength = .5 if strategy.endswith('shrink50') else 1.
        for group, part in frame.groupby('calibration_group'):
            if group < 0 or part.route_id.nunique() < MIN_CLUSTER_ROUTES:
                continue
            a, b = affine_pair(part)
            key = str(int(group))
            result['groups'][key] = {str(h): (1 - strength) * result[str(h)] + strength * a for h in HORIZONS}
            result['group_offsets'][key] = {str(h): (1 - strength) * result['offsets'][str(h)] + strength * b for h in HORIZONS}
    return result


def calibration_coefficients(h, groups, factors):
    a = np.full(len(groups), factors[str(h)], dtype=np.float64)
    b = np.full(len(groups), factors.get('offsets', {}).get(str(h), 0.), dtype=np.float64)
    for group, values in factors.get('groups', {}).items():
        selected = groups == int(group)
        a[selected] = values.get(str(h), factors[str(h)])
        b[selected] = factors.get('group_offsets', {}).get(group, {}).get(str(h), factors.get('offsets', {}).get(str(h), 0.))
    return a, b


def corrected(frame, factors):
    output = np.empty(len(frame), dtype=np.float64)
    groups = frame.calibration_group.to_numpy()
    for h in HORIZONS:
        selected = frame.h.to_numpy() == h
        a, b = calibration_coefficients(h, groups[selected], factors)
        output[selected] = np.maximum(a * frame.y_pred.to_numpy()[selected] + b, 0)
    if not np.isfinite(output).all():
        raise ValueError('Неконечная калибровка')
    return output


def apply_factors(raw, h, route_index, factors):
    a, b = calibration_coefficients(h, CAL_GROUP_LABELS[route_index], factors)
    output = np.maximum(np.asarray(raw) * a + b, 0)
    if not np.isfinite(output).all():
        raise ValueError('Неконечная калибровка')
    return output


def factor_records(factors):
    groups = np.array([-1, *[int(g) for g in factors.get('groups', {})]])
    records = []
    for h in HORIZONS:
        a, b = calibration_coefficients(h, groups, factors)
        records.extend(dict(scope='fallback' if group == -1 else 'group', calibration_group=int(group), h=h,
                            factor=float(slope), offset=float(offset)) for group, slope, offset in zip(groups, a, b))
    return records


def select_calibration():
    rows, choices, factor_rows, comparisons = [], {}, [], {}
    for name, k in calibration_variants():
        fit, select = (read_phase(stage, name) for stage in ('calibration_fit', 'calibration_select'))
        if fit.timestamp.max() >= CAL_SPLIT or select.timestamp.max() >= TEST_START or select.cutoff.min() < CAL_SPLIT:
            raise ValueError('Пересечение периодов калибровки и test')
        candidates = []
        for strategy in CALIBRATION_STRATEGIES:
            factors = fit_factors(fit, strategy)
            factor_rows.extend(dict(variant=name, strategy=strategy, fitted_on='calibration_fit', **row) for row in factor_records(factors))
            for period, frame in [('calibration_fit', fit), ('calibration_select', select)]:
                row = dict(variant=name, strategy=strategy, period=period, **metrics(frame.y.to_numpy(), corrected(frame, factors)))
                rows.append(row)
                if period == 'calibration_select':
                    candidates.append(row)
        best = min(candidates, key=lambda row: row['score'])
        all_val = pd.concat([fit, select], ignore_index=True)
        comparisons[name] = {strategy: fit_factors(all_val, strategy) for strategy in CALIBRATION_STRATEGIES}
        choices[name] = dict(k=k, strategy=best['strategy'], selection_score=best['score'], factors=comparisons[name][best['strategy']])
    winner = min(choices, key=lambda name: choices[name]['selection_score'])
    frozen = dict(winner=winner, variants=choices, comparison_factors=comparisons, base_settings=CAT_CONFIGS[0],
                  coefficient_fit_for_selection='first_val_block', selection_period='second_val_block',
                  evaluation_coefficient_fit='all_val_rolling_oof', test_used=False,
                  val_start=str(VAL_START), calibration_split=str(CAL_SPLIT), test_start=str(TEST_START),
                  calibration_groups=dict(k=max(CLUSTERS), fitted_before=str(VAL_START), routes=[str(r) for r in ROUTES], labels=CAL_GROUP_LABELS.tolist()))
    write_json(RUN / 'calibration_frozen.json', frozen)
    pd.DataFrame(rows).to_csv(RUN / 'calibration_selection.csv', index=False)
    pd.DataFrame(factor_rows).to_csv(RUN / 'calibration_candidates_factors.csv', index=False)
    table = pd.DataFrame(rows)
    display(table[['variant', 'strategy', 'period', 'WAPE', 'relative_bias', 'score']])
    print(table[['variant', 'strategy', 'period', 'WAPE', 'relative_bias', 'score']].to_string(index=False), flush=True)
    print('Победитель зафиксирован до test:', winner, choices[winner]['strategy'], flush=True)
    return frozen


def check_deadline():
    return None


def require_gpu():
    if TASK_TYPE != 'GPU':
        raise ValueError('Обучение только на GPU')
    if ACTIVE['algorithm'] == 'cat' and get_gpu_device_count() < 1:
        raise RuntimeError('CatBoost не видит CUDA. Включите GPU в Kaggle.')
    if ACTIVE['algorithm'] == 'xgb':
        probe = xgb.XGBRegressor(n_estimators=1, max_depth=1, device='cuda:0', tree_method='hist', n_jobs=THREADS)
        probe.fit(np.arange(16, dtype=np.float32).reshape(8, 2), np.arange(8, dtype=np.float32))
        device = json.loads(probe.get_booster().save_config())['learner']['generic_param']['device']
        if not device.startswith('cuda'):
            raise RuntimeError('XGBoost не видит CUDA; CPU fallback запрещён')
        del probe
    print('GPU проверен:', ACTIVE['algorithm'], flush=True)


def run_rich_calibration():
    global ACTIVE, PIPELINE_VERSION, CAT_CONFIGS
    ACTIVE = EXPERIMENTS[0]
    PIPELINE_VERSION = 'rich_calibration_v3_' + ACTIVE['name']
    CAT_CONFIGS = [dict(experiment=ACTIVE, parameters=PARAMETERS, parameter_sources=PARAMETER_SOURCES)]
    STEP_VALUE = pd.Timedelta(minutes=30)
    if STEP != STEP_VALUE:
        raise ValueError('Неверный шаг времени')
    with threadpool_limits(limits=THREADS):
        initialize_calibration()
        write_json(RUN / 'calibration_protocol.json', dict(strategies=list(CALIBRATION_STRATEGIES), affine_slope_bounds=list(AFFINE_SLOPE_BOUNDS), test_protocol='all_coefficients_frozen_before_test', test_used_for_selection=False))
        pd.DataFrame([dict(h=int(h), trees=p['trees'], **p['params']) for h, p in PARAMETERS[ACTIVE['parameter_set']].items()]).to_csv(RUN / 'effective_model_params.csv', index=False)
        phase_forecasts('calibration_fit', VAL_START, VAL_START, CAL_SPLIT)
        phase_forecasts('calibration_select', CAL_SPLIT, CAL_SPLIT, TEST_START)
        frozen = select_calibration()
        comparison = evaluate_calibration(frozen)
        comparison.to_csv(RUN / 'all_calibrations_train_test.csv', index=False)
        print('Сравнение всех калибровок; победитель выбран до test:', flush=True)
        print(comparison[['variant', 'strategy', 'period', 'WAPE', 'relative_bias', 'WAPE + |Relative Bias|', 'selected_on_val']].to_string(index=False), flush=True)
        finalize_calibration(frozen)
    print('Независимая оценка test получена до полного переобучения. Все финальные модели обучены на всей доступной размеченной истории. Test уже изучался в проекте.', flush=True)
