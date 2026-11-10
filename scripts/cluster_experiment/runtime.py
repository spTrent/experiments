from pathlib import Path
import gc
import hashlib
import importlib.metadata
import json
import time
import zipfile
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from catboost.utils import get_gpu_device_count
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from IPython.display import display


def require_gpu():
    global cp
    if TASK_TYPE != 'GPU':
        raise ValueError("Обучение разрешено только на GPU: TASK_TYPE='GPU'")
    if get_gpu_device_count() < 1:
        raise RuntimeError('CatBoost не видит GPU. Включите NVIDIA GPU в настройках Kaggle.')
    try:
        import cupy as cp
        probe = cp.linalg.solve(cp.eye(2, dtype=cp.float64), cp.ones(2, dtype=cp.float64))
        if not np.allclose(cp.asnumpy(probe), 1):
            raise RuntimeError('Ошибка проверки CUDA')
        cp.cuda.get_current_stream().synchronize()
    except Exception as error:
        raise RuntimeError('Для Ridge требуется рабочий CuPy под CUDA вашей Kaggle-сессии. CPU fallback отключён.') from error
    print('GPU обязателен: CatBoost CUDA и CuPy проверены.', flush=True)


class GPURidge:
    def __init__(self, alpha):
        self.alpha = float(alpha)

    def fit(self, x, y, sample_weight=None):
        width = x.shape[1] + 1
        gram = cp.zeros((width, width), dtype=cp.float64)
        rhs = cp.zeros(width, dtype=cp.float64)
        for start in range(0, len(y), PREDICT_BATCH):
            stop = min(start + PREDICT_BATCH, len(y))
            z = cp.empty((stop - start, width), dtype=cp.float64)
            z[:, :-1] = cp.asarray(x[start:stop])
            z[:, -1] = 1
            weights = cp.ones(stop - start) if sample_weight is None else cp.asarray(sample_weight[start:stop])
            gram += z.T @ (z * weights[:, None])
            rhs += z.T @ (cp.asarray(y[start:stop]) * weights)
        penalty = cp.full(width, self.alpha, dtype=cp.float64)
        penalty[-1] = 0
        solution = cp.asnumpy(cp.linalg.solve(gram + cp.diag(penalty), rhs))
        if not np.isfinite(solution).all():
            raise ValueError('Неконечные коэффициенты GPU Ridge')
        self.coef_, self.intercept_ = solution[:-1], float(solution[-1])
        return self

    def predict(self, x):
        return cp.asnumpy(cp.asarray(x, dtype=cp.float64) @ cp.asarray(self.coef_) + self.intercept_)


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def load_inputs():
    global TIMES, ROUTES, VALUES, REQUESTS, VAL_START, TEST_START, TEST_END, RUN, ROLL
    require_gpu()
    if THREADS < 1 or PREDICT_BATCH < 1 or MIN_CLUSTER_ROUTES < 1:
        raise ValueError('Некорректные настройки исполнения')
    if not RIDGE_LAMBDAS or min(RIDGE_LAMBDAS) <= 0 or not CAT_CONFIGS:
        raise ValueError('Нужны положительные регуляризации и конфигурации CatBoost')
    train_path, test_path = Path(TRAIN_PATH), Path(TEST_PATH)
    if not (train_path.is_file() and test_path.is_file()) and Path('/kaggle/input').exists():
        matches = [list(Path('/kaggle/input').rglob(p.name)) for p in (train_path, test_path)]
        if any(len(m) != 1 for m in matches):
            raise ValueError('Укажите однозначные TRAIN_PATH и TEST_PATH')
        train_path, test_path = (m[0] for m in matches)
    history = pd.read_parquet(train_path, columns=['route_id', 'timestamp', TARGET, *STATUSES])
    REQUESTS = pd.read_parquet(test_path, columns=['id', 'route_id', 'timestamp'])
    for frame in (history, REQUESTS):
        frame['timestamp'] = pd.to_datetime(frame.timestamp)
        if frame.isna().any().any() or frame.duplicated(['route_id', 'timestamp']).any():
            raise ValueError('Пропуски или повторяющиеся пары маршрут × время')
    history = history.sort_values(['timestamp', 'route_id'])
    TIMES = pd.DatetimeIndex(history.timestamp.unique())
    ROUTES = np.sort(history.route_id.unique())
    if len(history) != len(TIMES) * len(ROUTES) or not TIMES.equals(pd.date_range(TIMES[0], TIMES[-1], freq='30min')):
        raise ValueError('Нужна полная получасовая панель')
    if (TIMES[-1].hour, TIMES[-1].minute) != (10, 30):
        raise ValueError('Ожидался последний cutoff в 10:30')
    expected = pd.MultiIndex.from_product([ROUTES, [TIMES[-1] + h * STEP for h in HORIZONS]])
    actual = pd.MultiIndex.from_frame(REQUESTS[['route_id', 'timestamp']])
    if not REQUESTS.id.is_unique or len(actual) != len(expected) or len(expected.difference(actual)):
        raise ValueError('Конкурсный test не соответствует маршрутам и горизонтам')
    VALUES = {c: history[c].to_numpy(dtype=np.float32).reshape(len(TIMES), len(ROUTES)) for c in [TARGET, *STATUSES]}
    del history
    if any(not np.isfinite(a).all() for a in VALUES.values()) or (VALUES[TARGET] < 0).any():
        raise ValueError('Некорректные значения истории')
    full_dates = TIMES[(TIMES.hour == 10) & (TIMES.minute == 30) & (TIMES + max(HORIZONS) * STEP <= TIMES[-1])]
    if len(full_dates) <= VAL_DAYS + TEST_DAYS + 28:
        raise ValueError('Недостаточно истории для train/val/test')
    TEST_START = full_dates[-TEST_DAYS].normalize()
    VAL_START = full_dates[-TEST_DAYS - VAL_DAYS].normalize()
    TEST_END = full_dates[-1].normalize() + pd.Timedelta(days=1)
    if min(CLUSTERS) != 1 or max(CLUSTERS) > len(ROUTES):
        raise ValueError('CLUSTERS должен включать 1 и не превышать число маршрутов')
    config = dict(version=PIPELINE_VERSION, code=CODE_HASH, clusters=list(CLUSTERS), horizons=list(HORIZONS),
                  min_cluster_routes=MIN_CLUSTER_ROUTES, ridge_lambdas=list(RIDGE_LAMBDAS), cat_configs=CAT_CONFIGS,
                  seed=SEED, val_days=VAL_DAYS, test_days=TEST_DAYS, task_type=TASK_TYPE,
                  versions={p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'scikit-learn', 'catboost', 'pyarrow')},
                  train_hash=digest_file(train_path), test_hash=digest_file(test_path))
    run_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    RUN = Path(OUTPUT_ROOT) / ('cluster_experiment_' + run_id)
    RUN.mkdir(parents=True, exist_ok=True)
    write_json(RUN / 'manifest.json', config)
    splits = pd.DataFrame([
        dict(period='train', first=str(TIMES[0]), end_exclusive=str(VAL_START), starts='все 48'),
        dict(period='val', first=str(VAL_START), end_exclusive=str(TEST_START), starts='10:30'),
        dict(period='test', first=str(TEST_START), end_exclusive=str(TEST_END), starts='10:30'),
    ])
    splits.to_csv(RUN / 'splits.csv', index=False)
    display(splits)
    ROLL = {w: pd.DataFrame(VALUES[TARGET]).rolling(w, min_periods=1).mean().to_numpy(dtype=np.float32) for w in (4, 12, 48, 336)}
    print('Маршрутов:', len(ROUTES), 'Выходы:', RUN, 'CatBoost:', TASK_TYPE)


def segment_routes(boundary, stage):
    mask = TIMES < boundary
    y = VALUES[TARGET][mask].astype(np.float64)
    scale = np.maximum(y.mean(axis=0), 1.0)
    normalized = y / scale
    grid = TIMES[mask]
    descriptors = {}
    for slot in range(48):
        select = grid.hour * 2 + grid.minute // 30 == slot
        descriptors[f'slot_{slot:02}'] = normalized[select].mean(axis=0)
    for day in range(7):
        descriptors[f'weekday_{day}'] = normalized[grid.dayofweek == day].mean(axis=0)
    descriptors['log_cv'] = np.log1p(normalized.std(axis=0))
    descriptors['zero_share'] = (y == 0).mean(axis=0)
    descriptors['jump_share'] = (np.abs(np.diff(normalized, axis=0)) > 1).mean(axis=0)
    descriptors['log_q95'] = np.log1p(np.quantile(normalized, 0.95, axis=0))
    frame = pd.DataFrame(descriptors, index=ROUTES)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError('Недостаточно данных для профилей маршрутов')
    scaler = StandardScaler().fit(frame)
    z = scaler.transform(frame)
    labels, estimators = {1: np.zeros(len(ROUTES), dtype=int)}, {}
    for k in CLUSTERS:
        if k == 1:
            continue
        model = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(z)
        groups = model.labels_.copy()
        counts = np.bincount(groups, minlength=k)
        groups[counts[groups] < MIN_CLUSTER_ROUTES] = -1
        labels[k], estimators[k] = groups, model
    artifact = dict(boundary=str(boundary), routes=ROUTES, scale=scale, labels=labels,
                    descriptor_scaler=scaler, clusterers=estimators, descriptor_columns=list(frame.columns))
    joblib.dump(artifact, RUN / f'segments_{stage}.joblib')
    frame.insert(0, 'route_id', ROUTES)
    frame['target_scale'] = scale
    for k, groups in labels.items():
        frame[f'cluster_{k}'] = groups
    frame.to_csv(RUN / f'route_profiles_{stage}.csv', index=False)
    display(pd.DataFrame([dict(stage=stage, k=k, cluster=int(g), routes=int((groups == g).sum()))
                          for k, groups in labels.items() for g in np.unique(groups)]))
    return artifact


def origin_indices(h, period):
    labelled = np.arange(len(TIMES) - h)
    targets = TIMES[labelled + h]
    morning = (TIMES[labelled].hour == 10) & (TIMES[labelled].minute == 30)
    masks = dict(train=targets < VAL_START, train_val=targets < TEST_START,
                 val=(TIMES[labelled] >= VAL_START) & (targets < TEST_START) & morning,
                 test=(TIMES[labelled] >= TEST_START) & (TIMES[labelled] < TEST_END) & morning,
                 full=np.ones(len(labelled), dtype=bool))
    return labelled[masks[period]]


def features(h, origins, segmentation):
    scale = segmentation['scale'].astype(np.float32)
    columns, names = [], []
    shape = (len(origins), len(ROUTES))
    def add(name, array):
        names.append(name)
        columns.append(np.broadcast_to(array, shape).reshape(-1))
    def lag(name, array, indices):
        valid = indices >= 0
        data = array[np.maximum(indices, 0)] / scale
        add(name, np.where(valid[:, None], data, 0))
        add(name + '_available', valid[:, None])
    for delay in (0, 1, 2, 4, 12):
        lag(f'target_back_{delay}', VALUES[TARGET], origins - delay)
    for status in STATUSES:
        lag(status, VALUES[status], origins)
        add(status + '_delta_1h', (VALUES[status][origins] - VALUES[status][np.maximum(origins - 2, 0)]) / scale)
    for window, array in ROLL.items():
        lag(f'target_mean_{window}', array, origins)
    for days in (1, 7, 14, 28):
        lag(f'target_season_{days}', VALUES[TARGET], origins + h - 48 * days)
    dates = TIMES[origins] + h * STEP
    for label, phase in [('weekday', dates.dayofweek.to_numpy() / 7), ('slot', (dates.hour * 2 + dates.minute / 30).to_numpy() / 48)]:
        add(label + '_sin', np.sin(2 * np.pi * phase)[:, None])
        add(label + '_cos', np.cos(2 * np.pi * phase)[:, None])
    add('log_route_scale', np.log1p(scale)[None, :])
    matrix = np.column_stack(columns).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError('Неконечные признаки')
    return matrix, names


def dataset(h, period, segmentation):
    origins = origin_indices(h, period)
    x, names = features(h, origins, segmentation)
    y = VALUES[TARGET][origins + h].reshape(-1).astype(np.float64)
    route_index = np.tile(np.arange(len(ROUTES)), len(origins))
    return dict(x=x, y=y, route_index=route_index, origins=origins, names=names)


def metrics(y, pred):
    if not np.isfinite(pred).all():
        raise ValueError('Неконечный прогноз')
    errors = pred - y
    total = float(y.sum())
    if total <= 0:
        raise ValueError('WAPE не определена при нулевом суммарном факте')
    absolute, signed = float(np.abs(errors).sum()), float(errors.sum())
    return dict(n=len(y), volume=total, absolute=absolute, signed=signed,
                WAPE=absolute / total, relative_bias=signed / total,
                score=(absolute + abs(signed)) / total)


def aggregate(rows, keys):
    table = pd.DataFrame(rows)
    totals = table.groupby(keys, as_index=False)[['n', 'volume', 'absolute', 'signed']].sum()
    totals['WAPE'] = totals.absolute / totals.volume
    totals['abs_relative_bias'] = totals.signed.abs() / totals.volume
    totals['WAPE + |Relative Bias|'] = totals.WAPE + totals.abs_relative_bias
    return totals


def fit_model(train, segmentation, family, k, settings):
    scales = segmentation['scale'][train['route_index']]
    target = train['y'] / scales
    weights = scales / scales.mean()
    labels = segmentation['labels'][k][train['route_index']]
    models = {}
    groups = np.unique(labels)
    for group in groups:
        select = np.ones(len(labels), dtype=bool) if group == -1 else labels == group
        x, y, w = train['x'][select], target[select], weights[select]
        started = time.monotonic()
        if family == 'ridge':
            scaler = StandardScaler().fit(x)
            z = scaler.transform(x)
            model = GPURidge(alpha=float(settings['lambda']) * len(x))
            model.fit(z, y, sample_weight=w)
        else:
            scaler = None
            model = CatBoostRegressor(**settings, loss_function='MAE', random_seed=SEED,
                                      task_type=TASK_TYPE, thread_count=THREADS, verbose=False,
                                      allow_writing_files=False, allow_const_label=True, l2_leaf_reg=10)
            model.fit(x, y, sample_weight=w)
        models[int(group)] = dict(model=model, scaler=scaler)
        print(f'{family} k={k} group={group}: {len(y):,} строк, {time.monotonic() - started:.1f} с', flush=True)
        del x, y, w
    return dict(models=models, family=family, k=k, settings=settings, feature_names=train['names'])


def predict_model(bundle, data, segmentation):
    labels = segmentation['labels'][bundle['k']][data['route_index']]
    pred = np.empty(len(labels), dtype=np.float64)
    for group, item in bundle['models'].items():
        indices = np.flatnonzero(labels == group)
        for start in range(0, len(indices), PREDICT_BATCH):
            rows = indices[start:start + PREDICT_BATCH]
            x = data['x'][rows]
            if item['scaler'] is not None:
                x = item['scaler'].transform(x)
            pred[rows] = item['model'].predict(x)
    pred *= segmentation['scale'][data['route_index']]
    return np.maximum(pred, 0)


def variants():
    return [('ridge_global', 'ridge', 1), *[(f'cat_k{k}', 'cat', k) for k in CLUSTERS]]


def run_search():
    segmentation = segment_routes(VAL_START, 'search')
    rows, selected = [], {}
    for h in HORIZONS:
        train, val = dataset(h, 'train', segmentation), dataset(h, 'val', segmentation)
        for name, family, k in variants():
            choices = [dict(lambda_=float(v)) for v in RIDGE_LAMBDAS] if family == 'ridge' else CAT_CONFIGS
            choices = [{'lambda': c['lambda_']} if family == 'ridge' else dict(c) for c in choices]
            path = RUN / f'search_h{h}_{name}.json'
            if path.exists():
                saved = json.loads(path.read_text())
                selected[f'{name}_h{h}'] = saved['best']
                rows.extend(saved['rows'])
                continue
            local, best = [], None
            for candidate, settings in enumerate(choices):
                bundle = fit_model(train, segmentation, family, k, settings)
                report = []
                for period, data in [('train', train), ('val', val)]:
                    record = dict(variant=name, h=h, candidate=candidate, period=period,
                                  **metrics(data['y'], predict_model(bundle, data, segmentation)))
                    local.append(record)
                    report.append(record)
                display(pd.DataFrame(report)[['variant', 'h', 'candidate', 'period', 'WAPE', 'relative_bias', 'score']])
                print('Test закрыт до фиксации настроек; score = WAPE + |Relative Bias|', flush=True)
                score = report[-1]['score']
                if best is None or score < best['score']:
                    best = dict(candidate=candidate, settings=settings, score=score)
                del bundle
                gc.collect()
            write_json(path, dict(best=best, rows=local))
            selected[f'{name}_h{h}'] = best
            rows.extend(local)
        del train, val
        gc.collect()
    chosen = [r for r in rows if r['candidate'] == selected[f"{r['variant']}_h{r['h']}"]['candidate']]
    summary = aggregate(chosen, ['variant', 'period'])
    validation = summary[summary.period == 'val'].sort_values(['WAPE + |Relative Bias|', 'variant'])
    winner = str(validation.iloc[0].variant)
    frozen = dict(selected=selected, winner=winner, selection_period='val', val_start=str(VAL_START), test_start=str(TEST_START))
    write_json(RUN / 'selection_frozen.json', frozen)
    pd.DataFrame(rows).to_csv(RUN / 'search_metrics.csv', index=False)
    summary.to_csv(RUN / 'validation_summary.csv', index=False)
    display(summary)
    print('Выбран по val:', winner, flush=True)
    return frozen


def save_predictions(h, name, period, data, pred):
    origins = data['origins']
    frame = pd.DataFrame(dict(route_id=ROUTES[data['route_index']], cutoff=np.repeat(TIMES[origins], len(ROUTES)),
                              timestamp=np.repeat(TIMES[origins] + h * STEP, len(ROUTES)), h=h, y=data['y'], y_pred=pred))
    frame.to_parquet(RUN / f'{period}_h{h}_{name}.parquet', index=False)


def run_evaluation(frozen):
    segmentation = segment_routes(TEST_START, 'evaluation')
    rows = []
    for h in HORIZONS:
        train, test = dataset(h, 'train_val', segmentation), dataset(h, 'test', segmentation)
        for name, family, k in variants():
            path = RUN / f'evaluation_h{h}_{name}.json'
            if path.exists():
                rows.extend(json.loads(path.read_text()))
                continue
            settings = frozen['selected'][f'{name}_h{h}']['settings']
            bundle = fit_model(train, segmentation, family, k, settings)
            report = []
            for period, data in [('train', train), ('test', test)]:
                pred = predict_model(bundle, data, segmentation)
                report.append(dict(variant=name, h=h, period=period, **metrics(data['y'], pred)))
                if period == 'test':
                    save_predictions(h, name, period, data, pred)
            display(pd.DataFrame(report)[['variant', 'h', 'period', 'WAPE', 'relative_bias', 'score']])
            write_json(path, report)
            rows.extend(report)
            del bundle
            gc.collect()
        del train, test
        gc.collect()
    summary = aggregate(rows, ['variant', 'period'])
    summary['selected_on_val'] = summary.variant == frozen['winner']
    pd.DataFrame(rows).to_csv(RUN / 'independent_metrics_by_horizon.csv', index=False)
    summary.to_csv(RUN / 'independent_metrics.csv', index=False)
    display(summary)
    note = 'Независимая оценка test получена до переобучения на всей доступной размеченной истории. Train здесь включает бывшую val. Период test уже использовался в других экспериментах проекта. Победитель выбран только по val.'
    (RUN / 'evaluation_note.txt').write_text(note)
    print(note)
    write_json(RUN / 'evaluation_complete.json', dict(winner=frozen['winner']))
    return summary


def run_full(frozen):
    if not (RUN / 'evaluation_complete.json').exists():
        raise ValueError('Сначала завершите независимую оценку')
    segmentation = segment_routes(TIMES[-1] + STEP, 'full')
    name, family, k = next(v for v in variants() if v[0] == frozen['winner'])
    rows, predictions = [], []
    for h in HORIZONS:
        model_path = RUN / f'final_model_h{h}.joblib'
        train = dataset(h, 'full', segmentation)
        settings = frozen['selected'][f'{name}_h{h}']['settings']
        if model_path.exists():
            bundle = joblib.load(model_path)
        else:
            bundle = fit_model(train, segmentation, family, k, settings)
            temporary = model_path.with_suffix('.tmp')
            joblib.dump(bundle, temporary, compress=3)
            temporary.replace(model_path)
        rows.append(dict(variant=name, h=h, period='full_in_sample',
                         **metrics(train['y'], predict_model(bundle, train, segmentation))))
        x, feature_names = features(h, np.array([len(TIMES) - 1]), segmentation)
        if feature_names != bundle['feature_names']:
            raise ValueError('Изменился порядок признаков')
        data = dict(x=x, route_index=np.arange(len(ROUTES)))
        pred = predict_model(bundle, data, segmentation)
        predictions.append(pd.DataFrame(dict(route_id=ROUTES, timestamp=TIMES[-1] + h * STEP, y_pred=pred)))
        independent = pd.read_csv(RUN / 'independent_metrics_by_horizon.csv')
        display(pd.concat([independent[(independent.variant == name) & (independent.h == h)], pd.DataFrame([rows[-1]])])[
            ['variant', 'h', 'period', 'WAPE', 'relative_bias', 'score']])
        print('Строки train/test относятся к модели ДО полного переобучения.', flush=True)
        del train, bundle
        gc.collect()
    forecast = pd.concat(predictions, ignore_index=True)
    submission = REQUESTS.merge(forecast, on=['route_id', 'timestamp'], how='left', validate='one_to_one', sort=False)
    if len(submission) != len(REQUESTS) or not submission.id.equals(REQUESTS.id) or not np.isfinite(submission.y_pred).all() or (submission.y_pred < 0).any():
        raise ValueError('Нарушен формат сабмита')
    submission[['id', 'y_pred']].to_csv(RUN / 'submission.csv', index=False)
    aggregate(rows, ['variant', 'period']).to_csv(RUN / 'full_in_sample_metrics.csv', index=False)
    if CREATE_ARCHIVE:
        with zipfile.ZipFile(RUN.with_suffix('.zip'), 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(RUN.rglob('*')):
                if path.is_file():
                    archive.write(path, path.relative_to(RUN.parent))
    print('Сабмит:', RUN / 'submission.csv')
    print('Независимая оценка test получена до переобучения на всей доступной размеченной истории.')
