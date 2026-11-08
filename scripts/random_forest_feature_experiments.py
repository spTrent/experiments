import re

EXPERIMENTS = {
    'baseline': dict(),
    'drop_constant': dict(drop_constant=True),
    'compact_availability': dict(compact_availability=True),
    'calendar_dayofweek': dict(calendar=True),
    'thin_offsets': dict(thin=True),
    'drop_network': dict(network='drop'),
    'thin_network': dict(network='thin'),
    'add_time': dict(add=['time']),
    'add_profile': dict(add=['profile']),
    'add_level24': dict(add=['level']),
    'add_anomaly': dict(add=['anomaly']),
    'add_trends': dict(add=['trend']),
    'add_volatility': dict(add=['volatility']),
    'add_seasonal_dispersion': dict(add=['dispersion']),
    'all_additions': dict(add=['time','profile','level','anomaly','trend','volatility','dispersion']),
    'compact': dict(drop_constant=True, compact_availability=True, calendar=True, thin=True, network='thin'),
    'compact_with_additions': dict(drop_constant=True, compact_availability=True, calendar=True, thin=True, network='thin',
                                   add=['time','profile','level','anomaly','trend','volatility','dispersion'])
}


def experiment_columns(names, spec):
    result = []
    for name in names:
        if name.startswith('x_'):
            if any(name.startswith(f'x_{group}_') for group in spec.get('add', [])):
                result.append(name)
            continue
        if spec.get('drop_constant') and name == 'selected_status6_pair_count':
            continue
        if spec.get('compact_availability') and 'signal_available' in name:
            continue
        if spec.get('calendar') and name in ('weekday_sin','weekday_cos','is_weekend'):
            continue
        if spec.get('network') == 'drop' and name.startswith('network_'):
            continue
        if spec.get('network') == 'thin' and name.startswith('network_') and '_offset_' in name:
            continue
        if spec.get('thin') and not name.startswith('slot_') and re.search(r'_offset_m[134]$', name):
            continue
        result.append(name)
    if spec.get('compact_availability'):
        for name in ('status6_lag_valid', 'residual_status6_lag_valid'):
            if name in names and name not in result:
                result.append(name)
    return result


base_load_stage = load_stage


def load_stage(folder):
    data = base_load_stage(folder)
    metadata = json.loads((folder / 'ready.json').read_text())
    names = list(joblib.load(folder / 'preprocess.joblib').feature_names_in_)
    selected_names = experiment_columns(names, EXPERIMENTS[ACTIVE_EXPERIMENT])
    indices = [names.index(name) for name in selected_names]
    return {name: (frame, matrix[:, indices]) for name, (frame, matrix) in data.items()}


def select_context(variant, experiment):
    global RESIDUAL, ACTIVE_EXPERIMENT, DOCS_DIR, PARAMS_DIR, POSTPROCESS_DIR, MODEL_DIR
    RESIDUAL = variant == 'residual'
    ACTIVE_EXPERIMENT = experiment
    folder = RUN_DIR / variant
    DOCS_DIR = folder / 'docs'
    PARAMS_DIR = folder / 'model_params'
    POSTPROCESS_DIR = folder / 'postprocessing'
    MODEL_DIR = folder / 'models'
    for directory in (DOCS_DIR, PARAMS_DIR, POSTPROCESS_DIR, MODEL_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def comparison_fit(variant, experiment, h, i):
    select_context(variant, experiment)
    path = RUN_DIR / 'comparisons' / variant / experiment / f'h{h}_fold{i+1}.json'
    if path.exists():
        rows = json.loads(path.read_text())
        display(pd.DataFrame(rows))
        return rows
    fold = folds[i]
    started = time.perf_counter()
    folder = prepare_stage(h, f'fold{i+1}', fold['train'], fold['tuning'], fold['val'][0].normalize())
    data = load_stage(folder)
    train, matrix = data['train']
    model = fit_forest(train, matrix, base_params_for_horizon(h))
    rows = []
    for key, split in [('train','train'),('valid','tuning')]:
        frame, x = data[key]
        prediction = predict_matrix(model, frame, x)
        row = score_row(frame, prediction, split)
        error = prediction - frame.y.to_numpy()
        row.update(variant=variant, experiment=experiment, horizon=h, fold=i+1, features=x.shape[1],
                   volume=float(frame.y.sum()), absolute_error=float(np.abs(error).sum()), signed_error=float(error.sum()),
                   seconds=time.perf_counter()-started)
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, rows)
    display(pd.DataFrame(rows))
    del model, data, train, matrix, frame, x
    gc.collect()
    return rows


unknown = set(EXPERIMENT_NAMES) - set(EXPERIMENTS)
if unknown or 'baseline' not in EXPERIMENT_NAMES:
    raise ValueError(f'Неверный список экспериментов: {unknown}; baseline обязателен')
if not VARIANTS or set(VARIANTS) - {'direct','residual'}:
    raise ValueError('VARIANTS должен содержать direct и/или residual')
comparison_fits = len(VARIANTS) * len(EXPERIMENT_NAMES) * len(HORIZONS) * len(folds)
print(f'Сравнение: {comparison_fits} обучений; Optuna: {len(VARIANTS)*len(HORIZONS)*N_TRIALS*len(folds)}; test: {len(VARIANTS)*len(HORIZONS)}')

comparison_records = []
for variant in VARIANTS:
    for h in HORIZONS:
        for i in range(len(folds)):
            for experiment in EXPERIMENT_NAMES:
                comparison_records.extend(comparison_fit(variant, experiment, h, i))
comparison = pd.DataFrame(comparison_records)
comparison.to_csv(RUN_DIR / 'feature_experiments_by_horizon.csv', index=False)
pooled = comparison.groupby(['variant','experiment','fold','выборка'], as_index=False)[['volume','absolute_error','signed_error']].sum()
pooled['WAPE'] = pooled.absolute_error / pooled.volume
pooled['Relative Bias'] = pooled.signed_error / pooled.volume
pooled['|Relative Bias|'] = pooled['Relative Bias'].abs()
pooled['метрика конкурса'] = pooled.WAPE + pooled['|Relative Bias|']
pooled.to_csv(RUN_DIR / 'feature_experiments_by_fold.csv', index=False)
summary = pooled.loc[pooled['выборка'].eq('tuning')].groupby(['variant','experiment'],as_index=False)[['WAPE','Relative Bias','|Relative Bias|','метрика конкурса']].mean()
baseline = summary.loc[summary.experiment.eq('baseline'),['variant','метрика конкурса']].rename(columns={'метрика конкурса':'baseline_score'})
summary = summary.merge(baseline,on='variant')
summary['delta_vs_baseline'] = summary['метрика конкурса'] - summary.baseline_score
summary = summary.sort_values(['variant','метрика конкурса','experiment'])
summary.to_csv(RUN_DIR / 'feature_experiments_summary.csv',index=False)
display(summary)
tuning_details = comparison.loc[comparison['выборка'].eq('tuning')].copy()
reference = tuning_details.loc[tuning_details.experiment.eq('baseline'), ['variant','horizon','fold','метрика конкурса']].rename(columns={'метрика конкурса':'baseline_score'})
deltas = tuning_details.merge(reference, on=['variant','horizon','fold'])
deltas['delta_vs_baseline'] = deltas['метрика конкурса'] - deltas.baseline_score
deltas.to_csv(RUN_DIR / 'feature_deltas_by_window.csv', index=False)
stability = deltas.groupby(['variant','experiment','horizon'], as_index=False).agg(
    mean_score=('метрика конкурса','mean'), worst_score=('метрика конкурса','max'),
    score_std=('метрика конкурса','std'), mean_delta=('delta_vs_baseline','mean'),
    wins=('delta_vs_baseline', lambda values: int((values < 0).sum())), windows=('fold','count'),
    mean_wape=('WAPE','mean'), mean_signed_bias=('Relative Bias','mean'), mean_abs_bias=('|Relative Bias|','mean'))
stability.to_csv(RUN_DIR / 'feature_stability_by_horizon.csv', index=False)
display(stability)
winners = {variant: summary.loc[summary.variant.eq(variant)].iloc[0].experiment for variant in VARIANTS}
atomic_json(RUN_DIR / 'feature_selection_frozen.json', dict(winners=winners, selection_end_exclusive=str(selection_end), base_params=BASE_PARAMS_BY_HORIZON))

all_selected = {}
for variant in VARIANTS:
    select_context(variant, winners[variant])
    selected = {}
    for h in HORIZONS:
        selected[h] = select_horizon(h)
    all_selected[variant] = selected
    atomic_json(PARAMS_DIR / 'selection_frozen.json', selected)
atomic_json(RUN_DIR / 'all_parameters_frozen.json', all_selected)

parameter_comparisons = []
for variant in VARIANTS:
    select_context(variant, winners[variant])
    for h in HORIZONS:
        study = optuna.load_study(study_name=f'h{h}', storage=f'sqlite:///{(POSTPROCESS_DIR / "search.sqlite3").resolve().as_posix()}')
        completed = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
        base_trials = [trial for trial in completed if trial.params == base_params_for_horizon(h)]
        if not base_trials:
            raise ValueError('Исходная конфигурация не проверена в Optuna')
        base_trial = min(base_trials, key=lambda trial: trial.value)
        best_trial = study.best_trial
        if best_trial.value > base_trial.value + 1e-12:
            raise ValueError('Подбор проиграл исходной конфигурации')
        for label, trial in [('base',base_trial),('optuna',best_trial)]:
            for i in range(len(folds)):
                rows = json.loads((DOCS_DIR / f'search_h{h}_trial{trial.number}_fold{i+1}.json').read_text())
                for row in rows:
                    parameter_comparisons.append(dict(variant=variant, configuration=label, experiment=winners[variant],
                                                     params=json.dumps(trial.params, sort_keys=True), **row))
    details = pd.DataFrame([row for row in parameter_comparisons if row['variant'] == variant])
    details.to_csv(DOCS_DIR / 'base_vs_optuna_by_fold.csv', index=False)
    aggregate = details.loc[details['выборка'].eq('tuning')].groupby(['horizon','configuration'],as_index=False).agg(
        WAPE=('WAPE','mean'), signed_bias=('Relative Bias','mean'), absolute_bias=('|Relative Bias|','mean'),
        score=('метрика конкурса','mean'), worst_window=('метрика конкурса','max'))
    base_score = aggregate.loc[aggregate.configuration.eq('base'),['horizon','score']].rename(columns={'score':'base_score'})
    aggregate = aggregate.merge(base_score,on='horizon')
    aggregate['delta_vs_base'] = aggregate.score - aggregate.base_score
    aggregate.to_csv(DOCS_DIR / 'base_vs_optuna_summary.csv', index=False)
    display(aggregate)
