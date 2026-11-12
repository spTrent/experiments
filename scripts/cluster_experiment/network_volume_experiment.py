from sklearn.pipeline import make_pipeline


def load_network(path):
    history = pd.read_parquet(path, columns=['timestamp', TARGET, *STATUSES])
    if history.isna().any().any():
        raise ValueError('Пропуски в истории сети')
    network = history.groupby('timestamp', sort=True)[[TARGET, *STATUSES]].sum()
    if not network.index.equals(pd.date_range(network.index[0], network.index[-1], freq=STEP)):
        raise ValueError('Неполная получасовая история')
    if not np.isfinite(network.to_numpy()).all() or network.lt(0).any().any():
        raise ValueError('Некорректные объёмы')
    return network


def network_features(network):
    times = network.index
    values = network.to_numpy(dtype=np.float64)
    positions = np.flatnonzero((times.hour == 10) & (times.minute == 30) &
                               (np.arange(len(times)) >= 29 * 48))
    means = pd.Series(values[:, 0]).rolling(48).mean().to_numpy()
    days = positions[:, None] - 48 * np.arange(1, 29)
    common = {}
    for i, name in enumerate(network.columns):
        current = values[positions, i]
        common[f'{name}_level'] = np.log1p(current)
        for lag in [2, 4, 8]:
            common[f'{name}_change_{lag}'] = np.log1p(current) - np.log1p(values[positions-lag, i])
        common[f'{name}_seasonal_deviation'] = np.log1p(current) - np.log1p(np.median(values[days, i], axis=1))
    frames = []
    for h in HORIZONS:
        seasonal = np.median(values[days+h, 0], axis=1)
        scale = means[positions] / np.maximum(np.median(means[days], axis=1), 1.)
        timestamp = times[positions] + h * STEP
        target = np.full(len(positions), np.nan)
        known = positions+h < len(times)
        target[known] = values[positions[known]+h, 0]
        frames.append(pd.DataFrame(dict(common, cutoff=times[positions], timestamp=timestamp, horizon=h,
                                        network_profile=np.maximum(seasonal*scale, 1.), actual_volume=target)))
    return pd.concat(frames, ignore_index=True)


def aggregate(frame, network):
    values = frame[BASES + ['y']].astype(np.float64)
    totals = frame[GROUP_KEYS].join(values).groupby(GROUP_KEYS, as_index=False).sum()
    table = totals.merge(network, on=GROUP_KEYS, validate='one_to_one')
    if (len(table) != len(totals) or not np.allclose(table.y, table.actual_volume, rtol=1e-7, atol=1)
            or table.session6.le(0).any()):
        raise ValueError('Объёмы архивных прогнозов и исходные факты не согласованы')
    return table


def design(table, with_anchor):
    cols = [c for c in table if c.endswith('_level') or '_change_' in c or c.endswith('_seasonal_deviation')]
    features = table[cols].copy()
    features['log_profile'] = np.log(table.network_profile)
    for h in HORIZONS[1:]:
        features[f'horizon_{h}'] = table.horizon.eq(h).astype(float)
    day = table.cutoff.dt.dayofweek
    features['weekday_sin'] = np.sin(2*np.pi*day/7)
    features['weekday_cos'] = np.cos(2*np.pi*day/7)
    features['horizon_trend'] = (table.horizon - 4.5) * features[f'{TARGET}_change_4']
    if with_anchor:
        anchor = table.session6.to_numpy()
        features['log_anchor'] = np.log(anchor)
        for name in BASES[1:]:
            features[f'{name}_vs_anchor'] = np.log((table[name]+1)/(anchor+1))
    return features


def fit_predict_volumes(train, query, network, boundary):
    if not train.timestamp.lt(boundary).all():
        raise ValueError('Обучающие цели пересекли границу')
    ratios = np.log((train.y+1)/(train.session6+1))
    model = make_pipeline(StandardScaler(), Ridge(alpha=100.))
    model.fit(design(train, True), ratios)
    daily_log_ratio = model.predict(design(query, True))
    earlier = network.loc[network.timestamp.lt(boundary) & network.actual_volume.notna()]
    if earlier.cutoff.nunique() < 14:
        raise ValueError('Мало утренней истории')
    direct = make_pipeline(StandardScaler(), Ridge(alpha=100.))
    direct.fit(design(earlier, False), np.log((earlier.actual_volume+1)/(earlier.network_profile+1)))
    independent = query.network_profile.to_numpy() * np.exp(direct.predict(design(query, False)))
    anchor = query.session6.to_numpy()
    result = {'session6': anchor}
    for strength in (.25, .5, 1.):
        result[f'coefficient_{strength:g}'] = anchor * np.clip(np.exp(strength*daily_log_ratio), .95, 1.05)
    for weight in (.25, .5, 1.):
        result[f'direct_{weight:g}'] = (1-weight)*anchor + weight*independent
    return result


def score_variants(train, query, network, boundary, period, only=None):
    matrix = shape_matrix(train)
    joint = fit_weights(matrix, train.y)
    fitted = fit_shape('cat_d4', train, matrix, joint)
    rows = []
    output = {}
    for label, frame in [('train_oof', train), (period, query)]:
        shape = predict_shape('cat_d4', fitted, frame, shape_matrix(frame), joint)
        train_volume = aggregate(train, network)
        query_volume = aggregate(frame, network)
        variants = fit_predict_volumes(train_volume, query_volume, network, boundary)
        totals = frame[GROUP_KEYS].assign(shape=shape).groupby(GROUP_KEYS).shape.transform('sum').to_numpy()
        if np.any(totals <= 0):
            raise ValueError('Нулевой объём формы')
        shares = shape / totals
        for name, volume in variants.items():
            if only is not None and name not in only:
                continue
            mapped = frame[GROUP_KEYS].merge(query_volume[GROUP_KEYS].assign(volume=volume), on=GROUP_KEYS,
                                             validate='many_to_one', how='left').volume.to_numpy()
            prediction = shares*mapped
            stats = metric(frame.y, prediction)
            volume_error = volume - query_volume.y.to_numpy()
            volume_wape = np.abs(volume_error).sum()/query_volume.y.sum()
            rows.append(dict(period=label, variant=name, volume_WAPE=volume_wape, **stats))
            if label == period:
                output[name] = prediction
    return pd.DataFrame(rows), output
