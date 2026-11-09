def collect(stage_names):
    pieces = []
    for stage in stage_names:
        for h in range(1, 9):
            if completed(stage, h) is None:
                raise ValueError(f'Не завершён {stage}, h{h}')
            pieces.append(pd.read_parquet(RUN_ROOT / 'predictions' / f'{stage}_h{h}.parquet'))
    frame = pd.concat(pieces, ignore_index=True).sort_values(['cutoff', 'horizon', 'route_id']).reset_index(drop=True)
    if frame.duplicated(['cutoff', 'route_id', 'horizon']).any():
        raise ValueError('Дубли ключей прогнозов')
    if not np.isfinite(frame[['catboost', 'raw_prediction', 'profile28']].to_numpy()).all():
        raise ValueError('Неконечный прогноз')
    return frame

def finish():
    legacy = collect(['legacy'])
    tuning = legacy.cutoff < holdout_start
    offset = float((legacy.loc[tuning, 'y']-legacy.loc[tuning, 'raw_prediction']).mean())
    legacy['y_pred'] = np.maximum(legacy.raw_prediction + offset, 0)
    legacy['period'] = np.where(tuning, 'tuning', 'holdout')
    legacy['calibration_info_through'] = legacy.loc[tuning, 'timestamp'].max()
    atomic_parquet(legacy, RUN_ROOT / 'legacy_predictions.parquet')

    weekly = collect([f'week{i}' for i in range(1, WEEKLY_BLOCKS+1)])
    if not (weekly.model_info_through < weekly.cutoff).all():
        raise ValueError('Недельная модель использует будущее')
    warmup = weekly.cutoff < tune_start
    fixed_offset = float((weekly.loc[warmup, 'y']-weekly.loc[warmup, 'raw_prediction']).mean())
    weekly['y_pred'] = np.where(warmup, np.nan, np.maximum(weekly.raw_prediction+fixed_offset, 0))
    weekly['period'] = np.where(warmup, 'warmup', np.where(weekly.cutoff < holdout_start, 'tuning', 'holdout'))
    weekly['calibration_info_through'] = weekly.loc[warmup, 'timestamp'].max()
    atomic_parquet(weekly, RUN_ROOT / 'weekly_predictions.parquet')
    rows, daily_rows = [], []
    for protocol, frame in [('legacy', legacy), ('weekly', weekly.loc[~warmup])]:
        for period, part in frame.groupby('period'):
            for h in [0, *range(1, 9)]:
                block = part if h == 0 else part.loc[part.horizon.eq(h)]
                for col in ['catboost', 'raw_prediction', 'y_pred']:
                    rows.append(dict(protocol=protocol, period=period, horizon=h, prediction=col,
                                     **metrics(block, block[col])))
        for day, part in frame.groupby('cutoff'):
            for col in ['catboost', 'raw_prediction', 'y_pred']:
                daily_rows.append(dict(protocol=protocol, cutoff=day, saturday=day.dayofweek == 5,
                                       period=part.period.iloc[0], prediction=col, **metrics(part, part[col])))
    report = pd.DataFrame(rows)
    report.to_csv(RUN_ROOT / 'reports' / 'metrics.csv', index=False)
    pd.DataFrame(daily_rows).to_csv(RUN_ROOT / 'reports' / 'daily_metrics.csv', index=False)
    display(report.query('horizon == 0'))
    calibration = dict(legacy_tuning_offset=offset, weekly_warmup_offset=fixed_offset,
                       weights=dict(h1_catboost=1.0, h2_h8_catboost=0.7, h2_h8_profile28=0.3),
                       legacy_reference=dict(holdout_WAPE=0.321816, holdout_score=0.333315,
                                             holdout_daily_score=0.340499),
                       warning='Сравнение с сохранённым запуском, не битовая идентичность GPU. Параметры уже выбирались в проекте.')
    if RUN_FULL_REFIT:
        future = collect(['full'])
        final_offset = float((legacy.y-legacy.raw_prediction).mean())
        future['y_pred'] = np.maximum(future.raw_prediction+final_offset, 0)
        future['calibration_info_through'] = legacy.timestamp.max()
        atomic_parquet(future, RUN_ROOT / 'full_predictions.parquet')
        submission = test[['id']].merge(future[['id', 'y_pred']], on='id', how='left', validate='one_to_one')
        if (len(submission) != len(test) or not submission.id.equals(test.id.reset_index(drop=True))
                or not np.isfinite(submission.y_pred).all() or (submission.y_pred < 0).any()):
            raise ValueError('Некорректный submission')
        submission.to_csv(RUN_ROOT / 'submission.csv', index=False)
        calibration['submission_offset'] = final_offset
    save_json(RUN_ROOT / 'reports' / 'calibration.json', calibration)
    timings = [json.loads(path.read_text()) for path in sorted((RUN_ROOT / 'checkpoints').glob('*.json'))]
    pd.DataFrame([{k: r[k] for k in ['stage', 'horizon', 'trees', 'seconds', 'train_start', 'fit_before']}
                  for r in timings]).to_csv(RUN_ROOT / 'reports' / 'timings.csv', index=False)
    save_json(RUN_ROOT / 'status.json', dict(complete=True, run_id=RUN_ID, tasks=len(timings),
                                          full_refit=RUN_FULL_REFIT))
    archive('results.zip')
    archive('checkpoint.zip', include_models=True)
    destination = RUN_ROOT.parent / 'models.zip'
    temporary = destination.with_suffix('.partial.zip')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for folder in ['models', 'lags']:
            for path in sorted((RUN_ROOT / folder).glob('*')):
                if path.is_file() and '.partial' not in path.name:
                    z.write(path, str(Path(RUN_ROOT.name) / path.relative_to(RUN_ROOT)))
        z.write(RUN_ROOT / 'manifest.json', f'{RUN_ROOT.name}/manifest.json')
    temporary.replace(destination)
    print('Готово. Сохраните выполненный notebook, results.zip и models.zip.')
    print('Для продолжения в новой сессии нужен checkpoint.zip, а не results.zip.')

finish()
