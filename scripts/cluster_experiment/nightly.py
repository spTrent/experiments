import xgboost as xgb
from datetime import datetime, timezone


def check_deadline():
    if datetime.now(timezone.utc) >= datetime.fromisoformat(STOP_NEW_FITS_UTC):
        raise TimeoutError('Достигнуто время остановки новых обучений')


def seasonal_base(x, names):
    total = np.zeros(len(x), dtype=np.float64)
    count = np.zeros(len(x), dtype=np.float64)
    for days in (7, 14, 28):
        col = names.index(f'target_season_{days}')
        available = x[:, names.index(f'target_season_{days}_available')]
        total += x[:, col] * available
        count += available
    return np.divide(total, count, out=x[:, names.index('target_back_0')].astype(np.float64).copy(), where=count > 0)


def fit_model(train, segmentation, family, k, settings):
    check_deadline()
    scales = segmentation['scale'][train['route_index']]
    target = train['y'] / scales
    if ACTIVE['residual']:
        target = target - seasonal_base(train['x'], train['names'])
    weights = scales / scales.mean()
    labels = segmentation['labels'][k][train['route_index']]
    params = dict(PARAMETERS[ACTIVE['parameter_set']][str(ACTIVE_H)]['params'])
    trees = PARAMETERS[ACTIVE['parameter_set']][str(ACTIVE_H)]['trees']
    if ACTIVE.get('compact'):
        trees = min(trees, 900)
        key = 'depth' if ACTIVE['algorithm'] == 'cat' else 'max_depth'
        params[key] = min(params[key], 6)
    models = {}
    for group in np.unique(labels):
        check_deadline()
        select = np.ones(len(labels), dtype=bool) if group == -1 else labels == group
        started = time.monotonic()
        if ACTIVE['algorithm'] == 'cat':
            model = CatBoostRegressor(**params, iterations=trees, loss_function=ACTIVE['loss'],
                                      task_type='GPU', devices='0', thread_count=THREADS,
                                      random_seed=SEED, verbose=False, allow_writing_files=False,
                                      allow_const_label=True, bootstrap_type='Bayesian')
        else:
            model = xgb.XGBRegressor(**params, n_estimators=trees, objective=ACTIVE['loss'],
                                     tree_method='hist', device='cuda:0', n_jobs=THREADS, random_state=SEED)
        model.fit(train['x'][select], target[select], sample_weight=weights[select])
        if ACTIVE['algorithm'] == 'xgb':
            device = json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
            if not device.startswith('cuda'):
                raise RuntimeError('XGBoost переключился на CPU')
        models[int(group)] = dict(model=model, scaler=None)
        print(ACTIVE['name'], 'h', ACTIVE_H, 'k', k, 'group', int(group), 'seconds', round(time.monotonic()-started, 1), flush=True)
    return dict(models=models, k=k, residual=ACTIVE['residual'], feature_names=train['names'], experiment=ACTIVE)


def predict_model(bundle, data, segmentation):
    labels = segmentation['labels'][bundle['k']][data['route_index']]
    pred = np.empty(len(labels), dtype=np.float64)
    for group, item in bundle['models'].items():
        indices = np.flatnonzero(labels == group)
        for start in range(0, len(indices), PREDICT_BATCH):
            rows = indices[start:start + PREDICT_BATCH]
            pred[rows] = item['model'].predict(data['x'][rows])
    if bundle['residual']:
        pred += seasonal_base(data['x'], bundle['feature_names'])
    return np.maximum(pred * segmentation['scale'][data['route_index']], 0)


original_calibration_data = calibration_data


def calibration_data(h, origins, segmentation):
    global ACTIVE_H
    ACTIVE_H = h
    return original_calibration_data(h, origins, segmentation)


def run_nightly():
    global ACTIVE, PIPELINE_VERSION, CAT_CONFIGS, RUN
    completed = []
    root = Path(OUTPUT_ROOT)
    for experiment in EXPERIMENTS:
        ACTIVE = experiment
        PIPELINE_VERSION = 'nightly_gpu_v1_' + ACTIVE['name']
        CAT_CONFIGS = [dict(experiment=ACTIVE, parameters=PARAMETERS)]
        try:
            check_deadline()
            initialize_calibration()
            with threadpool_limits(limits=THREADS):
                phase_forecasts('calibration_fit', VAL_START, VAL_START, CAL_SPLIT)
                phase_forecasts('calibration_select', CAL_SPLIT, CAL_SPLIT, TEST_START)
                frozen = select_calibration()
                summary = evaluate_calibration(frozen)
            selected = frozen['variants'][frozen['winner']]
            record = dict(experiment=ACTIVE['name'], directory=str(RUN), selection_score=selected['selection_score'],
                          winner=frozen['winner'], strategy=selected['strategy'])
            completed.append(record)
            write_json(root / 'nightly_completed.json', completed)
            display(pd.DataFrame(completed).sort_values('selection_score'))
        except TimeoutError as error:
            print(str(error), flush=True)
            break
        except Exception as error:
            write_json(root / ('nightly_error_' + ACTIVE['name'] + '.json'), dict(error=repr(error), directory=str(RUN)))
            print('Ошибка эксперимента:', ACTIVE['name'], repr(error), flush=True)
            gc.collect()
    if completed:
        best = min(completed, key=lambda row: row['selection_score'])
        ACTIVE = next(e for e in EXPERIMENTS if e['name'] == best['experiment'])
        RUN = Path(best['directory'])
        frozen = json.loads((RUN / 'calibration_frozen.json').read_text())
        print('Лучший по val:', best, flush=True)
        if datetime.now(timezone.utc) < datetime.fromisoformat(STOP_NEW_FITS_UTC):
            with threadpool_limits(limits=THREADS):
                finalize_calibration(frozen)
        else:
            print('Полный refit отложен из-за дедлайна; сабмит не собран.', flush=True)
    print('Независимые test-метрики получены до полного переобучения. Test ранее изучался; победитель выбирался по val.', flush=True)
