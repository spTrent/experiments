from contextlib import contextmanager


@contextmanager
def active_bases(names):
    global BASES, VOLUME_INPUTS
    previous = BASES, VOLUME_INPUTS
    BASES = list(names)
    VOLUME_INPUTS = [name for name in BASES if name not in ('session6', 'xgb_clean')]
    try:
        yield
    finally:
        BASES, VOLUME_INPUTS = previous


def fit_candidate(name, frame):
    names = [base for base in BASES if name != f'without_{base}']
    with active_bases(names):
        matrix = shape_matrix(frame)
        joint = fit_weights(matrix, frame.y)
        volumes = fit_volumes(frame)
        fitted = dict(name=name, bases=names, joint=joint, volumes=volumes)
        if name.startswith('shape_'):
            features, level = meta_features(frame, matrix, matrix @ joint)
            prep = StandardScaler().fit(features)
            torch.manual_seed(SEED)
            width = features.shape[1]
            model = torch.nn.Linear(width, 1) if name == 'shape_linear' else torch.nn.Sequential(
                torch.nn.Linear(width, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
            last = model if isinstance(model, torch.nn.Linear) else model[-1]
            torch.nn.init.zeros_(last.weight)
            torch.nn.init.zeros_(last.bias)
            x = torch.as_tensor(prep.transform(features), dtype=torch.float32)
            scale = max(float(frame.y.mean()), 1.)
            target = torch.as_tensor(frame.y.to_numpy() / scale, dtype=torch.float32)
            baseline = torch.as_tensor(matrix @ joint / scale, dtype=torch.float32)
            codes, groups = pd.factorize(pd.MultiIndex.from_frame(frame[GROUP_KEYS]), sort=False)
            indices = torch.as_tensor(codes, dtype=torch.int64)
            anchor = torch.as_tensor(frame.session6.to_numpy() / scale, dtype=torch.float32)
            totals = torch.zeros(len(groups)).index_add(0, indices, anchor)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.01)
            losses = []
            for epoch in range(160):
                correction = .10 * torch.tanh(model(x).ravel())
                raw = baseline * (1 + correction)
                sums = torch.zeros(len(groups)).index_add(0, indices, raw)
                prediction = torch.where(sums[indices] > 0, raw * totals[indices] / sums[indices].clamp_min(1e-12), anchor)
                loss = (prediction - target).abs().sum() / target.sum() + .1 * correction.square().mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                losses.append(float(loss.detach()))
            fitted.update(prep=prep, model=model.eval(), loss=losses)
        else:
            fitted['shape'] = fit_shape('cat_d4', frame, matrix, joint)
        if name.startswith('horizon_'):
            global_prediction = predict_candidate(dict(fitted, name='cat_d4__ridge'), frame)
            table = frame[['horizon', 'y']].assign(prediction=global_prediction).groupby('horizon')[['y', 'prediction']].sum()
            ratio = (table.y + 1) / (table.prediction + 1)
            global_ratio = (table.y.sum() + 1) / (table.prediction.sum() + 1)
            alpha = float(name.split('_')[1])
            relative = np.clip(np.exp(alpha * (np.log(ratio) - np.log(global_ratio))), .97, 1.03)
            fitted['horizon_factors'] = relative.to_dict()
        return fitted


def predict_candidate(fitted, frame):
    with active_bases(fitted['bases']):
        matrix = shape_matrix(frame)
        if fitted['name'].startswith('shape_'):
            features, level = meta_features(frame, matrix, matrix @ fitted['joint'])
            x = torch.as_tensor(fitted['prep'].transform(features), dtype=torch.float32)
            with torch.no_grad():
                correction = .10 * torch.tanh(fitted['model'](x).ravel()).numpy()
            shape = with_volume(frame, (matrix @ fitted['joint']) * (1 + correction))
        else:
            shape = predict_shape('cat_d4', fitted['shape'], frame, matrix, fitted['joint'])
        factor = volume_multipliers(frame, fitted['volumes'])['ridge']
        if 'horizon_factors' in fitted:
            factor = np.clip(factor * frame.horizon.map(fitted['horizon_factors']).to_numpy(), .95, 1.05)
        result = shape * factor
        if not np.isfinite(result).all() or (result < 0).any():
            raise ValueError('Некорректный прогноз кандидата')
        return result
