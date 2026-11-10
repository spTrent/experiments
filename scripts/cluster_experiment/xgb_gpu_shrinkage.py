class GPUTrainingIterator(xgb.DataIter):
    def __init__(self, train, indices, segmentation):
        self.train = train
        self.indices = indices
        self.segmentation = segmentation
        self.position = 0
        self.device_batch = None
        super().__init__(release_data=True)

    def reset(self):
        self.position = 0
        self.device_batch = None

    def next(self, input_data):
        if self.position >= len(self.indices):
            return False
        rows = self.indices[self.position:self.position + GPU_BATCH_ROWS]
        self.position += len(rows)
        x = gpu_features(self.train['x'][rows], self.train['names'])
        scales = self.segmentation['scale'][self.train['route_index'][rows]]
        y = cp.asarray(self.train['y'][rows] / scales, dtype=cp.float32)
        weight = cp.asarray(scales / self.weight_scale, dtype=cp.float32)
        self.device_batch = (x, y, weight)
        input_data(data=x, label=y, weight=weight, feature_names=self.feature_names, feature_types=self.feature_types)
        return True


def gpu_features(x, names):
    values = cp.asarray(x, dtype=cp.float32)
    route = values[:, names.index('rich_route_index')].reshape(-1, 1)
    return cp.concatenate([values, route], axis=1)


def require_gpu():
    global cp
    if TASK_TYPE != 'GPU':
        raise ValueError('Обучение разрешено только на GPU')
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError('Нет CUDA GPU')
        x = cp.arange(16, dtype=cp.float32).reshape(8, 2)
        probe = xgb.QuantileDMatrix(x, label=cp.arange(8, dtype=cp.float32), max_bin=GPU_MAX_BIN)
        model = xgb.train(dict(device='cuda:0', tree_method='hist', max_bin=GPU_MAX_BIN, max_depth=1), probe, num_boost_round=1)
        if not json.loads(model.save_config())['learner']['generic_param']['device'].startswith('cuda'):
            raise RuntimeError('CPU fallback')
        del model, probe, x
    except Exception as error:
        raise RuntimeError('Нужны рабочие CuPy/CUDA и CUDA-сборка XGBoost с QuantileDMatrix(DataIter). Включите GPU в Kaggle. CPU fallback отключён.') from error
    print('CUDA data/model проверены:', xgb.__version__, cp.cuda.runtime.memGetInfo(), flush=True)


def fit_model(train, segmentation, family, k, settings):
    labels = segmentation['labels'][k][train['route_index']]
    specification = PARAMETERS[ACTIVE['parameter_set']][str(ACTIVE_H)]
    feature_names = list(train['names']) + ['route_code']
    feature_types = ['float'] * len(train['names']) + ['c']
    params = dict(specification['params'], objective='reg:absoluteerror', eval_metric='mae',
                  device='cuda:0', tree_method='hist', max_bin=GPU_MAX_BIN,
                  max_cat_to_onehot=1, nthread=THREADS, seed=SEED)
    models = {}
    for group in np.unique(labels):
        indices = np.arange(len(labels)) if group == -1 else np.flatnonzero(labels == group)
        iterator = GPUTrainingIterator(train, indices, segmentation)
        iterator.weight_scale = float(segmentation['scale'][train['route_index']].mean())
        iterator.feature_names, iterator.feature_types = feature_names, feature_types
        started = time.monotonic()
        try:
            matrix = xgb.QuantileDMatrix(iterator, max_bin=GPU_MAX_BIN, max_quantile_batches=GPU_QUANTILE_BATCHES,
                                        enable_categorical=True)
            model = xgb.train(params, matrix, num_boost_round=specification['trees'])
        except Exception as error:
            raise RuntimeError('Ошибка CUDA-обучения. При нехватке VRAM уменьшите GPU_BATCH_ROWS/GPU_MAX_BIN или оставьте MODEL_CLUSTERS=(8,). CPU fallback не используется.') from error
        if not json.loads(model.save_config())['learner']['generic_param']['device'].startswith('cuda'):
            raise RuntimeError('XGBoost переключился на CPU')
        models[int(group)] = dict(model=model, scaler=None)
        del matrix, iterator
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        print('GPU h', ACTIVE_H, 'k', k, 'group', int(group), 'rows', len(indices), 'seconds', round(time.monotonic() - started, 1), flush=True)
    return dict(models=models, k=k, residual=False, feature_names=train['names'], experiment=ACTIVE)


def predict_model(bundle, data, segmentation):
    labels = segmentation['labels'][bundle['k']][data['route_index']]
    prediction = np.empty(len(labels), dtype=np.float64)
    for group, item in bundle['models'].items():
        indices = np.flatnonzero(labels == group)
        item['model'].set_param(dict(device='cuda:0'))
        for start in range(0, len(indices), GPU_BATCH_ROWS):
            rows = indices[start:start + GPU_BATCH_ROWS]
            x = gpu_features(data['x'][rows], bundle['feature_names'])
            result = item['model'].inplace_predict(x, validate_features=False)
            if not isinstance(result, cp.ndarray):
                raise RuntimeError('Предсказание не вернуло CUDA-массив')
            prediction[rows] = cp.asnumpy(result)
            del x, result
    return np.maximum(prediction * segmentation['scale'][data['route_index']], 0)


def calibration_variants():
    return [(f'xgb_k{k}', k) for k in MODEL_CLUSTERS]


previous_shrinkage_fit = fit_factors


def fit_factors(frame, strategy):
    if not strategy.startswith('shrink_recent_'):
        return previous_shrinkage_fit(frame, strategy)
    parts = strategy.split('_')
    half_life, alpha = float(parts[2]), float(parts[3])
    long_ratio = scalar_ratio(frame)
    age = (frame.cutoff.max() - frame.cutoff).dt.total_seconds().to_numpy() / 86400
    weights = np.exp2(-age / half_life)
    recent_ratio = float(np.sum(frame.y * weights) / np.sum(frame.y_pred * weights))
    ratio = float(np.clip((1 - alpha) * long_ratio + alpha * recent_ratio, *SCALE_BOUNDS))
    return {str(h): ratio for h in HORIZONS}


CALIBRATION_STRATEGIES = ('raw', 'fixed_1.05', 'global', 'recent_global_3', 'recent_global_7', 'recent_global_14') + tuple(
    f'shrink_recent_{half_life}_{alpha}' for half_life in (3, 7, 14) for alpha in (.25, .5, .75))


def phase_forecasts(stage, boundary, query_start, query_end, frozen=None):
    if stage != 'calibration_fit':
        return original_shrinkage_phase(stage, boundary, query_start, query_end, frozen)
    boundaries = pd.date_range(query_start, query_end, freq='7D').tolist()
    if boundaries[-1] != query_end:
        boundaries.append(query_end)
    root = RUN
    block_results = []
    for index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        globals()['RUN'] = root / f'oof_week_{index}'
        RUN.mkdir(exist_ok=True)
        try:
            block_results.append(original_shrinkage_phase(stage, start, start, end, frozen))
        finally:
            globals()['RUN'] = root
    for h in HORIZONS:
        for name, k in calibration_variants():
            frames = [pd.read_parquet(root / f'oof_week_{i}' / f'{stage}_h{h}_{name}.parquet') for i in range(len(block_results))]
            pd.concat(frames, ignore_index=True).to_parquet(root / f'{stage}_h{h}_{name}.parquet', index=False)
    return pd.concat(block_results, ignore_index=True)
