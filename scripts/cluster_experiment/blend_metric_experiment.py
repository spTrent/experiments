import os
os.environ['OMP_NUM_THREADS'] = '4'
import torch
import joblib
from sklearn.preprocessing import StandardScaler


def find_previous_run(explicit=None):
    required = ['base_oof.parquet', 'test_predictions.parquet', 'frozen_selection.json']
    candidates = [Path(explicit)] if explicit else sorted({p.parent for p in Path('/kaggle/input').rglob('base_oof.parquet')
        if all((p.parent / f).is_file() for f in required)})
    if len(candidates) != 1:
        raise ValueError(f'Укажите EVALUATION_DIR: {candidates}')
    return candidates[0]


def fit_candidate(name, frame):
    matrix = shape_matrix(frame)
    joint = fit_weights(matrix, frame.y)
    if name == 'cat_d4__ridge':
        return dict(name=name, joint=joint, shape=fit_shape('cat_d4', frame, matrix, joint), volumes=fit_volumes(frame))
    features, level = meta_features(frame, matrix, matrix @ joint)
    prep = StandardScaler().fit(features)
    torch.manual_seed(SEED)
    width = features.shape[1]
    model = torch.nn.Linear(width, 1) if name.startswith('linear') else torch.nn.Sequential(
        torch.nn.Linear(width, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
    last = model if isinstance(model, torch.nn.Linear) else model[-1]
    torch.nn.init.zeros_(last.weight)
    torch.nn.init.zeros_(last.bias)
    x = torch.as_tensor(prep.transform(features), dtype=torch.float32)
    scale = max(float(np.mean(frame.y)), 1.)
    y = torch.as_tensor(frame.y.to_numpy() / scale, dtype=torch.float32)
    anchor = torch.as_tensor((matrix @ joint) / scale, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.01)
    losses = []
    for epoch in range(160):
        correction = .05 * torch.tanh(model(x).ravel())
        prediction = anchor * (1 + correction)
        error = prediction - y
        loss = error.abs().sum() / y.sum()
        if name.endswith('metric'):
            loss = loss + error.sum().abs() / y.sum()
        loss = loss + .1 * correction.square().mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        losses.append(float(loss.detach()))
    return dict(name=name, joint=joint, prep=prep, model=model.eval(), loss=losses)


def predict_candidate(fitted, frame):
    matrix = shape_matrix(frame)
    if fitted['name'] == 'cat_d4__ridge':
        return predict_shape('cat_d4', fitted['shape'], frame, matrix, fitted['joint']) * volume_multipliers(frame, fitted['volumes'])['ridge']
    features, level = meta_features(frame, matrix, matrix @ fitted['joint'])
    x = torch.as_tensor(fitted['prep'].transform(features), dtype=torch.float32)
    with torch.no_grad():
        correction = .05 * torch.tanh(fitted['model'](x).ravel()).numpy()
    return matrix @ fitted['joint'] * (1 + correction)


def verify_frame(frame):
    needed = [*BASES, 'y', 'level24', 'current', 'delta1h', 'delta2h', 'network_current']
    if frame.duplicated(KEYS).any() or not np.isfinite(frame[needed]).all().all() or frame.y.lt(0).any():
        raise ValueError('Некорректные исторические прогнозы')
    groups = frame.groupby(GROUP_KEYS).route_id.nunique()
    if not groups.eq(frame.route_id.nunique()).all():
        raise ValueError('Неполные группы маршрутов')


def experiment_select(oof):
    rows = []
    for fold, (start, stop) in enumerate(OOF_WINDOWS[2:], 3):
        train = oof.loc[(oof.cutoff + oof.horizon * STEP).lt(start)].reset_index(drop=True)
        val = oof.loc[oof.cutoff.ge(start) & oof.cutoff.lt(stop)].reset_index(drop=True)
        for name in CANDIDATES:
            model = fit_candidate(name, train)
            local = [dict(fold=fold, candidate=name, period=period, **metric(frame.y, predict_candidate(model, frame)))
                     for period, frame in [('train_oof', train), ('val', val)]]
            display(pd.DataFrame(local))
            rows.extend(local)
    table = pd.DataFrame(rows)
    table.to_csv(ARTIFACTS / 'validation_metrics.csv', index=False)
    ranking = table.loc[table.period.eq('val')].groupby('candidate', as_index=False).agg(
        mean_score=('score', 'mean'), worst_score=('score', 'max'))
    ranking['order'] = ranking.candidate.map({n: i for i, n in enumerate(CANDIDATES)})
    ranking = ranking.sort_values(['mean_score', 'worst_score', 'order'])
    ranking.to_csv(ARTIFACTS / 'validation_ranking.csv', index=False)
    frozen = dict(candidate=ranking.iloc[0].candidate, test_used_for_selection=False,
                  epochs=160, correction_bound=.05, candidates=CANDIDATES)
    write_json(ARTIFACTS / 'frozen_selection.json', frozen)
    display(ranking)
    return frozen


def experiment_test(oof, test, frozen):
    rows, predictions = [], test[KEYS + ['y']].copy()
    models = {}
    for name in dict.fromkeys(['cat_d4__ridge', frozen['candidate']]):
        model = fit_candidate(name, oof)
        models[name] = model
        local = []
        for period, frame in [('train_oof', oof), ('test', test)]:
            prediction = predict_candidate(model, frame)
            local.append(dict(candidate=name, selected=name == frozen['candidate'], period=period, **metric(frame.y, prediction)))
            if period == 'test':
                predictions[name] = prediction
        rows.extend(local)
        display(pd.DataFrame(local))
    pd.DataFrame(rows).to_csv(ARTIFACTS / 'train_test_metrics.csv', index=False)
    save_parquet(ARTIFACTS / 'test_predictions.parquet', predictions)
    daily = []
    for day, frame in predictions.groupby(predictions.cutoff.dt.normalize()):
        for name in models:
            daily.append(dict(day=str(day.date()), candidate=name, **metric(frame.y, frame[name])))
    pd.DataFrame(daily).to_csv(ARTIFACTS / 'daily_metrics.csv', index=False)
    joblib.dump(models, ARTIFACTS / 'evaluation_models.joblib')
    note = ('Оценка test получена до полного переобучения. Test 18–31 октября уже изучался ранее. '
            'Кандидат выбран на двух предтестовых неделях. Train OOF — собственное обучение метамодели. '
            'Новые базы не обучались, конкурсный сабмит не создавался. Для сабмита нужен полный refit всех компонентов.')
    (ARTIFACTS / 'evaluation_note.txt').write_text(note)
    print(note)
    archive_report()
