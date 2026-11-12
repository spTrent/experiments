import joblib
import torch
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


PAIRS = BASES[1:]
VOLUME_INPUTS = [name for name in PAIRS if name != 'xgb_clean']
SHAPES = ['session6', 'equal', 'joint', 'joint_without_xgb_clean', 'by_horizon',
          *[f'pair_{name}' for name in PAIRS],
          'ridge_100', 'ridge_1000', 'cat_d2', 'cat_d4', 'xgb_d2', 'xgb_d3', 'gating_8', 'gating_16']
VOLUMES = ['fixed', 'ratio', 'ewma7', 'ridge']
META_DEVICE = 'cuda'


def with_volume(frame, values, anchor=None):
    result = np.maximum(np.asarray(values, dtype=np.float64), 0)
    reference = frame.session6.to_numpy(dtype=np.float64) if anchor is None else np.asarray(anchor, dtype=np.float64)
    totals = frame[GROUP_KEYS].assign(pred=result, reference=reference).groupby(GROUP_KEYS, sort=False).transform('sum')
    reference_total = totals.reference.to_numpy()
    prediction_total = totals.pred.to_numpy()
    out = np.divide(result * reference_total, prediction_total, out=reference.copy(), where=prediction_total > 0)
    predicted = frame[GROUP_KEYS].assign(value=out).groupby(GROUP_KEYS, sort=False).value.sum()
    expected = frame[GROUP_KEYS].assign(value=reference).groupby(GROUP_KEYS, sort=False).value.sum()
    if not np.isfinite(out).all() or not np.allclose(predicted, expected, rtol=1e-10, atol=1e-5):
        raise ValueError('Не удалось сохранить заданный объём')
    return out


def shape_matrix(frame):
    return np.column_stack([with_volume(frame, frame[name]) for name in BASES])


def fit_weights(x, y):
    y = np.asarray(y, dtype=np.float64)
    denominator = y.sum()
    if denominator <= 0:
        raise ValueError('Нулевая сумма фактов для подбора весов')
    weights = np.zeros(x.shape[1])
    weights[0] = 1
    prediction = x[:, 0].copy()
    best = np.abs(prediction - y).sum() / denominator
    for step in (.2, .05):
        for _ in range(20):
            choice = None
            for source in np.flatnonzero(weights >= step - 1e-10):
                for destination in range(len(weights)):
                    if source == destination:
                        continue
                    amount = min(step, weights[source])
                    trial = prediction + amount * (x[:, destination] - x[:, source])
                    loss = np.abs(trial - y).sum() / denominator
                    if loss < best - 1e-9:
                        best = loss
                        choice = source, destination, amount, trial
            if choice is None:
                break
            source, destination, amount, prediction = choice
            weights[source] -= amount
            weights[destination] += amount
    return weights


def meta_features(frame, matrix, baseline):
    level = np.maximum(frame.level24.to_numpy(dtype=np.float64), 1)
    context = pd.DataFrame(dict(
        log_level=np.log1p(level), current=frame.current.to_numpy() / level,
        delta1h=frame.delta1h.to_numpy() / level, delta2h=frame.delta2h.to_numpy() / level,
        log_network=np.log1p(frame.network_current.to_numpy()),
        horizon=frame.horizon.to_numpy(), dayofweek=frame.cutoff.dt.day_of_week.to_numpy(),
        shape_disagreement=matrix.std(axis=1) / level, blend=baseline / level))
    for index, name in enumerate(BASES):
        context[f'{name}_at_anchor_volume'] = matrix[:, index] / level
        context[f'{name}_minus_session6'] = (matrix[:, index] - matrix[:, 0]) / level
    return context.clip(-50, 50).to_numpy(dtype=np.float32), level


class GatingNetwork(torch.nn.Module):
    def __init__(self, width, experts, hidden, prior):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Linear(width, hidden), torch.nn.Tanh(),
                                          torch.nn.Linear(hidden, experts))
        self.register_buffer('prior', torch.as_tensor(prior, dtype=torch.float32))
        torch.nn.init.zeros_(self.layers[-1].weight)
        torch.nn.init.zeros_(self.layers[-1].bias)

    def forward(self, context):
        return torch.softmax(self.prior.log() + self.layers(context), dim=1)


def fit_gating(x, experts, y, level, prior, hidden):
    torch.manual_seed(SEED)
    scaler = StandardScaler().fit(x)
    prior = (prior + .02) / (prior.sum() + .02 * len(prior))
    model = GatingNetwork(x.shape[1], experts.shape[1], hidden, prior).to(META_DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    normalized = experts / level[:, None]
    target = np.asarray(y, dtype=np.float64) / level
    weights = level / level.mean()
    tensors = [torch.as_tensor(value, dtype=torch.float32, device=META_DEVICE)
               for value in (scaler.transform(x), normalized, target, weights)]
    tx, te, ty, tw = tensors
    for _ in range(15):
        permutation = torch.randperm(len(x), device=META_DEVICE)
        for rows in permutation.split(4096):
            gate = model(tx[rows])
            pred = (gate * te[rows]).sum(dim=1)
            loss = ((pred - ty[rows]).abs() * tw[rows]).mean() + .03 * (gate - model.prior).square().sum(dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return scaler, model.cpu()


def fit_shape(name, frame, matrix, joint):
    if name == 'session6':
        return None
    if name == 'equal':
        return np.full(len(BASES), 1 / len(BASES))
    if name == 'joint':
        return joint
    if name == 'joint_without_xgb_clean':
        mask = np.array([base != 'xgb_clean' for base in BASES])
        weights = np.zeros(len(BASES))
        weights[mask] = fit_weights(matrix[:, mask], frame.y)
        return weights
    if name.startswith('pair_'):
        index = BASES.index(name.removeprefix('pair_'))
        weights = np.zeros(len(BASES))
        weights[[0, index]] = fit_weights(matrix[:, [0, index]], frame.y)
        return weights
    if name == 'by_horizon':
        weights = []
        for h in HORIZONS:
            mask = frame.horizon.eq(h).to_numpy()
            weights.append(.5 * joint + .5 * fit_weights(matrix[mask], frame.y.to_numpy()[mask]))
        return np.stack(weights)
    baseline = matrix @ joint
    features, level = meta_features(frame, matrix, baseline)
    residual = (frame.y.to_numpy(dtype=np.float64) - baseline) / level
    sample_weight = level / level.mean()
    if name.startswith('gating_'):
        return fit_gating(features, matrix, frame.y, level, joint, int(name.split('_')[-1]))
    if name.startswith('ridge_'):
        prep = StandardScaler().fit(features)
        model = Ridge(alpha=float(name.split('_')[-1]))
        model.fit(prep.transform(features), residual, sample_weight=sample_weight)
        return prep, model
    depth = int(name[-1])
    if name.startswith('cat_'):
        model = CatBoostRegressor(depth=depth, iterations=220, loss_function='MAE', learning_rate=.035,
                                  l2_leaf_reg=20, random_seed=SEED, verbose=False, allow_writing_files=False,
                                  thread_count=THREADS)
    else:
        model = XGBRegressor(max_depth=depth, n_estimators=220, objective='reg:absoluteerror',
                             tree_method='hist', learning_rate=.035, reg_lambda=20,
                             min_child_weight=30, random_state=SEED, n_jobs=THREADS)
    model.fit(features, residual, sample_weight=sample_weight)
    return None, model


def predict_shape(name, fitted, frame, matrix, joint):
    if name == 'session6':
        return matrix[:, 0]
    if isinstance(fitted, np.ndarray):
        if fitted.ndim == 1:
            return matrix @ fitted
        return np.sum(matrix * fitted[frame.horizon.to_numpy() - 1], axis=1)
    baseline = matrix @ joint
    prep, model = fitted
    features, level = meta_features(frame, matrix, baseline)
    if name.startswith('gating_'):
        weights = gate_weights(prep, model, features)
        return with_volume(frame, np.sum(matrix * weights, axis=1))
    encoded = features if prep is None else prep.transform(features)
    correction = np.clip(model.predict(encoded), -.5, .5) * level
    return with_volume(frame, baseline + correction)


def gate_weights(prep, model, features):
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(prep.transform(features), dtype=torch.float32)).numpy()


def volume_table(frame, with_y=False):
    aggregations = {name: (name, 'sum') for name in BASES}
    aggregations['network_current'] = ('network_current', 'first')
    if with_y:
        aggregations['y'] = ('y', 'sum')
    return frame.groupby(GROUP_KEYS, as_index=False, sort=True).agg(**aggregations)


def volume_features(table):
    anchor = np.maximum(table.session6.to_numpy(), 1)
    features = np.column_stack([np.log1p(anchor), np.log1p(table.network_current),
                                table.horizon, table.cutoff.dt.day_of_week,
                                *[np.log((table[name].to_numpy() + 1) / (anchor + 1)) for name in VOLUME_INPUTS]])
    return features


def fit_volumes(frame):
    table = volume_table(frame, with_y=True)
    if table.session6.sum() <= 0:
        raise ValueError('Нулевой объём Session6')
    ages = (table.cutoff.max() - table.cutoff).dt.total_seconds().to_numpy() / 86400
    w = np.exp2(-ages / 7)
    ratio = table.y.sum() / table.session6.sum()
    recent_ratio = np.dot(table.y, w) / np.dot(table.session6, w)
    x = volume_features(table)
    prep = StandardScaler().fit(x)
    model = Ridge(alpha=100).fit(prep.transform(x), np.log((table.y + 1) / (table.session6 + 1)))
    return dict(fixed=1., ratio=float(1 + .5 * (ratio - 1)),
                ewma7=float(1 + .5 * (recent_ratio - 1)), ridge=(prep, model))


def volume_multipliers(frame, fitted):
    table = volume_table(frame)
    prep, model = fitted['ridge']
    fitted = dict(fitted, ridge=1 + .5 * np.expm1(model.predict(prep.transform(volume_features(table)))))
    return {name: frame[GROUP_KEYS].merge(table[GROUP_KEYS].assign(factor=np.clip(factor, .95, 1.05)),
                                           on=GROUP_KEYS, how='left', validate='many_to_one').factor.to_numpy()
            for name, factor in fitted.items()}


def fit_meta(train, query, phase, split):
    train_matrix, query_matrix = shape_matrix(train), shape_matrix(query)
    joint = fit_weights(train_matrix, train.y)
    volume_models = fit_volumes(train)
    train_factors = volume_multipliers(train, volume_models)
    query_factors = volume_multipliers(query, volume_models)
    rows, fitted, predictions = [], {}, {}
    for name in SHAPES:
        model = fit_shape(name, train, train_matrix, joint)
        fitted[name] = model
        train_prediction = predict_shape(name, model, train, train_matrix, joint)
        query_prediction = predict_shape(name, model, query, query_matrix, joint)
        local = []
        for volume in VOLUMES:
            label = f'{name}__{volume}'
            train_values = train_prediction * train_factors[volume]
            query_values = query_prediction * query_factors[volume]
            predictions[label] = query_values
            local.extend([dict(phase=phase, variant=label, shape=name, volume=volume, period=period, **metric(y, p))
                          for period, y, p in [('train_oof', train.y, train_values), (split, query.y, query_values)]])
        display(pd.DataFrame(local))
        rows.extend(local)
    return pd.DataFrame(rows), dict(joint=joint, shapes=fitted, volumes=volume_models), predictions


def fit_selection(oof):
    blocks = []
    for fold, (start, stop) in enumerate(OOF_WINDOWS[2:], 3):
        train = oof.loc[oof.cutoff.lt(start)]
        validation = oof.loc[oof.cutoff.ge(start) & oof.cutoff.lt(stop)]
        rows, _, _ = fit_meta(train, validation, f'val_week_{fold}', 'val')
        blocks.append(rows)
    metrics = pd.concat(blocks, ignore_index=True)
    metrics.to_csv(ARTIFACTS / 'selection_metrics.csv', index=False)
    ranking = metrics.loc[metrics.period.eq('val')].groupby(['variant', 'shape', 'volume'], as_index=False).agg(
        mean_val_score=('score', 'mean'), worst_val_score=('score', 'max'), weeks=('score', 'count'))
    ranking['order'] = ranking['shape'].map({name: i for i, name in enumerate(SHAPES)})
    ranking = ranking.sort_values(['mean_val_score', 'worst_val_score', 'order', 'variant'])
    if not ranking.weeks.eq(2).all():
        raise ValueError('В подборе должны присутствовать обе недели val')
    ranking.to_csv(ARTIFACTS / 'selection_ranking.csv', index=False)
    display(ranking.drop(columns='order').head(20))
    selected = ranking.iloc[0]
    frozen = dict(variant=selected.variant, shape=selected['shape'], volume=selected.volume,
                  mean_val_score=float(selected.mean_val_score), test_used_for_selection=False,
                  meta_fit_weeks=4, local_test_before_full_refit=True)
    write_json(ARTIFACTS / 'frozen_selection.json', frozen)
    return frozen


def evaluate(oof, test, frozen):
    rows, models, predictions = fit_meta(oof, test, 'frozen_local_test', 'test')
    base_rows = [dict(phase='base', variant=column, shape=column, volume='raw', period=period,
                      **metric(frame.y, frame[column]))
                 for period, frame in [('train_oof', oof), ('test', test)] for column in BASES]
    table = pd.concat([pd.DataFrame(base_rows), rows], ignore_index=True)
    table['selected_on_val'] = table.variant.eq(frozen['variant'])
    table.to_csv(ARTIFACTS / 'train_test_metrics.csv', index=False)
    predictions = pd.concat([test.reset_index(drop=True), pd.DataFrame(predictions)], axis=1)
    save_parquet(ARTIFACTS / 'test_predictions.parquet', predictions)
    columns = [*BASES, *[f'{name}__fixed' for name in SHAPES]]
    if frozen['variant'] not in columns:
        columns.append(frozen['variant'])
    slices = []
    for kind, key in [('day', predictions.cutoff), ('horizon', predictions.horizon),
                      ('week', ((predictions.cutoff.dt.normalize() - LOCAL_TEST_START).dt.days // 7) + 1)]:
        for value, group in predictions.groupby(key):
            slices.extend(dict(slice=kind, value=str(value), variant=name, **metric(group.y, group[name])) for name in columns)
    pd.DataFrame(slices).to_csv(ARTIFACTS / 'test_slices.csv', index=False)
    weights = []
    matrix = shape_matrix(test)
    features, _ = meta_features(test, matrix, matrix @ models['joint'])
    for name, model in models['shapes'].items():
        if isinstance(model, np.ndarray):
            if model.ndim == 1:
                weights.append(dict(variant=name, horizon='all', **dict(zip(BASES, model))))
            else:
                weights.extend(dict(variant=name, horizon=h, **dict(zip(BASES, row)))
                               for h, row in zip(HORIZONS, model))
        elif name.startswith('gating_'):
            prep, network = model
            frame = pd.DataFrame(gate_weights(prep, network, features), columns=BASES)
            means = frame.groupby(test.horizon.to_numpy()).mean()
            weights.extend(dict(variant=name, horizon=h, **row.to_dict()) for h, row in means.iterrows())
            models['shapes'][name] = dict(prep=prep, state=network.state_dict(),
                                         width=features.shape[1], hidden=int(name.split('_')[-1]))
    pd.DataFrame(weights).to_csv(ARTIFACTS / 'blend_weights.csv', index=False)
    artifact = ARTIFACTS / 'evaluation_meta.joblib'
    temporary = artifact.with_suffix('.partial')
    joblib.dump(dict(frozen=frozen, bases=BASES, models=models), temporary)
    temporary.replace(artifact)
    comparison = ['joint__fixed', 'joint_without_xgb_clean__fixed', 'pair_xgb_clean__fixed']
    display(table.loc[table.selected_on_val | table.variant.isin([*BASES, *comparison])])
    note = ('Независимая от полного переобучения оценка test получена до переобучения. '
            'Локальный период 18–31 октября уже изучался. Выбор сделан на двух более ранних неделях. '
            'Новые метамодели обучены только на OOF до test. '
            'Session6 на OOF восстановлен по рецепту, а на test взят из архива решения 0.326. '
            'Полное переобучение и конкурсный сабмит не выполняются.')
    (ARTIFACTS / 'evaluation_note.txt').write_text(note)
    print(note)
