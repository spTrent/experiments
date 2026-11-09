TUNED_COLUMN = 'five_starts__route_ridge_tuned'
_original_profile_path = profile_path
_original_predictors = predictors
_original_collect_stage = collect_stage
_original_jobs = jobs

def profile_path(rel):
    rel = _original_profile_path(rel)
    return rel.replace('meta/','meta_tuned/',1) if rel.startswith('meta/') else rel

def predictors(include_routes):
    cols = _original_predictors(include_routes)
    return cols+[TUNED_COLUMN] if include_routes else cols

def jobs(h,stage,cols):
    return _original_jobs(h,stage,[c for c in cols if c!=TUNED_COLUMN])

def specifications():
    without = _original_predictors(False)
    fixed = _original_predictors(True)
    for variant,cols in [('without',without),('fixed',fixed),('tuned',without+[TUNED_COLUMN]),('both',fixed+[TUNED_COLUMN])]:
        yield dict(kind='weighted',columns=cols,route_variant=variant)
        for depth in (3,5):
            for iterations in (100,300):
                yield dict(kind='catboost',depth=depth,iterations=iterations,columns=cols,route_variant=variant)
        for epochs in (50,150):
            yield dict(kind='gate',epochs=epochs,columns=cols,route_variant=variant)

_original_assemble = assemble

def assemble():
    _original_assemble()
    board = load_json('meta/leaderboard.json')
    reference = min(entry['mean_week_score'] for entry in board
                    if entry['id'].startswith('raw_') and entry['spec']['route_variant'] in ('without','fixed'))
    table = pd.DataFrame([dict(candidate=entry['id'],route_variant=entry['spec']['route_variant'],
                        kind=entry['kind'],columns=entry['n_columns'],score=entry['mean_week_score'],
                        improvement_vs_original_raw=reference-entry['mean_week_score']) for entry in board])
    save('meta/tuned_comparison.parquet',table)
    display(table)

def tuned_export_root():
    global TUNED_ROOT,TUNED_MANIFEST
    if 'TUNED_ROOT' in globals():
        return TUNED_ROOT
    hits = list(INPUT.glob('**/ridge_tuned_*/manifest.json'))
    fingerprints = [file_digest(TRAIN_PATH),file_digest(TEST_PATH)]
    approved = []
    for path in hits:
        value = json.loads(path.read_text(encoding='utf-8'))
        if (value.get('base_code_id')==CODE_ID and value.get('code_id')==EXPECTED_TUNED_CODE_ID
                and value.get('profile')==PROFILE and value.get('fingerprints')==fingerprints
                and value.get('column')==TUNED_COLUMN and value.get('oof_weeks')==list(STACK_WEEKS)
                and value.get('tuning_end')=='2025-09-20'):
            approved.append((path,value))
    if len(approved)!=1:
        raise ValueError(f'Подключите ровно один полный экспорт tuned Ridge для текущих данных и версии; найдено {len(approved)}')
    path,TUNED_MANIFEST = approved[0]
    TUNED_ROOT = path.parent
    return TUNED_ROOT

def collect_stage(stage,columns):
    original = [c for c in columns if c!=TUNED_COLUMN]
    frame = _original_collect_stage(stage,original)
    if TUNED_COLUMN not in columns:
        return frame
    root = tuned_export_root()
    boundary,days = stage_spec(stage)
    blocks = []
    for h in range(1,9):
        path = root/f'predictions/h{h}/{stage}.parquet'
        if not path.is_file() or not path.with_suffix('.json').is_file():
            raise FileNotFoundError(f'Локальный Ridge: отсутствует h{h}/{stage}. Запустите нужный MODE на ноутбуке с RTX 4060 и обновите экспорт.')
        info = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        if (info.get('run_id')!=TUNED_MANIFEST['run_id'] or info.get('base_code_id')!=CODE_ID
                or pd.Timestamp(info['boundary'])!=boundary or not pd.Timestamp(info['fit_through'])<boundary
                or info.get('tuning_end')!='2025-09-20'):
            raise ValueError(f'Неверные границы или версия tuned Ridge: {path}')
        blocks.append(pd.read_parquet(path))
    extra = pd.concat(blocks,ignore_index=True).set_index(KEY)
    base = frame.set_index(KEY)
    if extra.index.has_duplicates or len(extra)!=len(base) or not base.index.difference(extra.index).empty:
        raise ValueError(f'Неполные или дублированные ключи tuned Ridge: {stage}')
    extra = extra.reindex(base.index)
    for name in ('timestamp','y'):
        if name in base and (name not in extra or not np.array_equal(base[name],extra[name])):
            raise ValueError(f'Несовпадение {name} tuned Ridge: {stage}')
    pred = extra.prediction.to_numpy()
    if not np.isfinite(pred).all() or (pred<0).any():
        raise ValueError('Некорректные прогнозы tuned Ridge')
    base[TUNED_COLUMN] = pred
    return base.reset_index()
