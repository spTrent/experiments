def run_train_val_submission():
    global ACTIVE, PIPELINE_VERSION, CAT_CONFIGS
    ACTIVE = EXPERIMENTS[0]
    PIPELINE_VERSION = 'catboost_rich_train_val_submission_v1'
    CAT_CONFIGS = [dict(experiment=ACTIVE, parameters=PARAMETERS, parameter_sources=PARAMETER_SOURCES,
                        fit_scope='train_plus_val_only', prediction_multiplier=SUBMISSION_MULTIPLIER)]
    with threadpool_limits(limits=THREADS):
        load_inputs()
        segmentation = segment_routes(TEST_START, 'train_val_only')
        if pd.Timestamp(segmentation['boundary']) != TEST_START:
            raise ValueError('Неверная граница обучаемых статистик')
        write_json(RUN / 'submission_protocol.json', dict(training_target_end_exclusive=str(TEST_START),
                   training_scope='train_plus_val', local_test_used_for_fit=False,
                   learned_statistics_fit_before=str(TEST_START), future_feature_cutoff=str(TIMES[-1]),
                   multiplier=SUBMISSION_MULTIPLIER, multiplier_source='fixed_before_this_run_from_previous_val_selection',
                   full_history_refit=False, note='Диагностический сабмит по явному запросу пользователя: локальный test исключён из fit, но доступен как история признаков.'))
        write_json(RUN / 'parameter_sources.json', PARAMETER_SOURCES)
        pd.DataFrame([dict(h=int(h), trees=p['trees'], **p['params']) for h, p in PARAMETERS[ACTIVE['parameter_set']].items()]).to_csv(RUN / 'effective_model_params.csv', index=False)
        reports, forecasts, audit = [], [], []
        for h in HORIZONS:
            positions = np.arange(len(TIMES) - h)
            train_origins = positions[TIMES[positions + h] < TEST_START]
            test_origins = positions[(TIMES[positions] >= TEST_START) & (TIMES[positions] < TEST_END)
                                     & (TIMES[positions].hour == 10) & (TIMES[positions].minute == 30)
                                     & (TIMES[positions + h] < TEST_END)]
            if not len(train_origins) or not len(test_origins) or TIMES[train_origins + h].max() >= TEST_START:
                raise ValueError('Некорректное временное разбиение')
            train = calibration_data(h, train_origins, segmentation)
            model_path = RUN / f'train_val_model_h{h}.joblib'
            if model_path.exists():
                bundle = joblib.load(model_path)
                if bundle.get('training_boundary') != str(TEST_START) or bundle.get('training_code') != CODE_HASH:
                    raise ValueError('Checkpoint не относится к train + val этой версии')
            else:
                bundle = fit_model(train, segmentation, 'cat', 8, {})
                bundle.update(training_boundary=str(TEST_START), training_code=CODE_HASH)
                temporary = model_path.with_suffix('.tmp')
                joblib.dump(bundle, temporary, compress=3)
                temporary.replace(model_path)
            train_prediction = predict_model(bundle, train, segmentation)
            for strategy, multiplier in [('raw', 1.), ('fixed_1.05', SUBMISSION_MULTIPLIER)]:
                reports.append(dict(h=h, strategy=strategy, period='train', **metrics(train['y'], train_prediction * multiplier)))
            del train, train_prediction
            gc.collect()
            test = calibration_data(h, test_origins, segmentation)
            test_prediction = predict_model(bundle, test, segmentation)
            for strategy, multiplier in [('raw', 1.), ('fixed_1.05', SUBMISSION_MULTIPLIER)]:
                reports.append(dict(h=h, strategy=strategy, period='test', **metrics(test['y'], test_prediction * multiplier)))
            pd.DataFrame(dict(route_id=ROUTES[test['route_index']], cutoff=np.repeat(TIMES[test_origins], len(ROUTES)),
                              timestamp=np.repeat(TIMES[test_origins] + h * STEP, len(ROUTES)), h=h,
                              y=test['y'], y_raw=test_prediction, y_pred=test_prediction * SUBMISSION_MULTIPLIER)).to_parquet(RUN / f'train_val_test_h{h}.parquet', index=False)
            table = pd.DataFrame(reports[-4:])
            display(table[['h', 'strategy', 'period', 'WAPE', 'relative_bias', 'score']])
            print(table[['h', 'strategy', 'period', 'WAPE', 'relative_bias', 'score']].to_string(index=False), flush=True)
            del test, test_prediction
            gc.collect()
            x, names = features(h, np.array([len(TIMES) - 1]), segmentation)
            if names != bundle['feature_names']:
                raise ValueError('Изменился порядок признаков')
            future = predict_model(bundle, dict(x=x, route_index=np.arange(len(ROUTES))), segmentation)
            forecasts.append(pd.DataFrame(dict(route_id=ROUTES, timestamp=TIMES[-1] + h * STEP,
                                               y_raw=future, y_pred=future * SUBMISSION_MULTIPLIER)))
            audit.append(dict(h=h, training_rows=len(train_origins) * len(ROUTES),
                              last_training_target=str(TIMES[train_origins + h].max()),
                              training_end_exclusive=str(TEST_START), feature_cutoff=str(TIMES[-1])))
            del bundle, x, future
            gc.collect()
        summary = aggregate(reports, ['strategy', 'period'])
        summary['relative_bias'] = summary.signed / summary.volume
        pd.DataFrame(reports).to_csv(RUN / 'train_val_metrics_by_horizon.csv', index=False)
        summary.to_csv(RUN / 'train_val_metrics.csv', index=False)
        pd.DataFrame(audit).to_csv(RUN / 'training_boundary_audit.csv', index=False)
        display(summary)
        print(summary.to_string(index=False), flush=True)
        submission = REQUESTS.merge(pd.concat(forecasts, ignore_index=True), on=['route_id', 'timestamp'], how='left', sort=False, validate='one_to_one')
        if not submission.id.equals(REQUESTS.id) or not np.isfinite(submission[['y_raw', 'y_pred']]).all().all() or (submission[['y_raw', 'y_pred']] < 0).any().any():
            raise ValueError('Некорректный submission')
        submission[['id', 'y_raw']].rename(columns={'y_raw': 'y_pred'}).to_csv(RUN / 'submission_raw.csv', index=False)
        submission[['id', 'y_pred']].to_csv(RUN / 'submission.csv', index=False)
        note = 'Модели и обучаемые статистики обучены только на train + val, target timestamp < TEST_START. Полного переобучения на локальном test не было по явному запросу пользователя. Test не использован для выбора настроек или множителя в этом запуске; ранее этот период изучался. История local test используется только для причинных признаков. Это диагностический вариант, не стандартный full-history refit.'
        (RUN / 'evaluation_note.txt').write_text(note)
        if CREATE_ARCHIVE:
            with zipfile.ZipFile(RUN.with_suffix('.zip'), 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(RUN.rglob('*')):
                    if path.is_file():
                        archive.write(path, path.relative_to(RUN.parent))
        print(note, flush=True)
        print('Raw:', RUN / 'submission_raw.csv', 'С множителем:', RUN / 'submission.csv', flush=True)
