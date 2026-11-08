def predictors(include_routes):
    families = [*MODEL_NAMES,*(['route_ridge'] if include_routes else [])]
    return [f'{r}__{name}' for r in REGIMES for name in families]+['profile28']

def jobs(h,stage,cols):
    return any(not complete(task_path(h,stage,*col.split('__'))) for col in cols if col!='profile28') or not complete(task_path(h,stage,'shared','profile28'))

def run_stage(data,stage,cols):
    h = data.h
    boundary,days = stage_spec(stage)
    query = query_indices(boundary,days)
    tic = time.monotonic()
    data.prepare(boundary,int(query[-1])+1)
    timing = dict(lag_seconds=time.monotonic()-tic,route_seconds=0.,matrix_seconds=0.,global_fit_seconds=0.)
    for regime in REGIMES:
        origins = train_indices(h,boundary,regime)
        if times[origins[-1]]+h*STEP >= boundary:
            raise ValueError(f'Утечка меток {stage}')
        route_rel = task_path(h,stage,regime,'route_ridge')
        if f'{regime}__route_ridge' in cols and not complete(route_rel):
            tic = time.monotonic()
            route_stage(data,stage,regime,origins,query,boundary)
            timing['route_seconds'] += time.monotonic()-tic
        names = [n for n in MODEL_NAMES if f'{regime}__{n}' in cols and not complete(task_path(h,stage,regime,n))]
        if not names:
            continue
        tic = time.monotonic()
        x, train = global_data(data,origins,'train',True)
        q, validation = global_data(data,query,'query',False)
        timing['matrix_seconds'] += time.monotonic()-tic
        encoded = None
        previous_family = None
        for name in names:
            family = name.rsplit('_',1)[0]
            if family!=previous_family:
                encoded = None
                if family in ('ridge','rf'):
                    tic = time.monotonic()
                    encoded = numeric_data(x,q,train.route_id,validation.route_id,data.folder,family=='ridge')
                    timing['matrix_seconds'] += time.monotonic()-tic
                previous_family = family
            started = time.monotonic()
            row,pred,device = fit_global(name,PARAMS[name][str(h)],x,q,train,validation,data.cols+data.corr_names,data.folder,encoded)
            finish_fit(task_path(h,stage,regime,name),result_frame(h,query),pred,row,boundary,
                       train.timestamp.max(),time.monotonic()-started,device)
            timing['global_fit_seconds'] += time.monotonic()-started
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
        del x,train,q,validation,encoded
    rel = task_path(h,stage,'shared','profile28')
    if not complete(rel):
        origins = train_indices(h,boundary,'five_starts')
        p = data.baseline[origins].ravel()
        keep = np.isfinite(p)
        row = metric(target[origins+h].ravel()[keep],p[keep])
        finish_fit(rel,result_frame(h,query),data.baseline[query].ravel(),row,boundary,
                   times[origins[-1]]+h*STEP,0.,'seasonal')

    return timing

def run_worker():
    global ETA_CONTEXT
    if MODE=='oof':
        if not OOF_WEEKS or any(i not in range(1,5) for i in OOF_WEEKS):
            raise ValueError('OOF_WEEKS должен содержать номера недель 1–4')
        stages = [f'oof{i}' for i in sorted(set(OOF_WEEKS))]
        cols = predictors(True)
    else:
        winner = load_json('meta/winner.json')
        cols = winner['columns']
        stages = ['test'] if MODE=='evaluate' else ['extra','full']
        if MODE=='submit':
            if load_json('meta/test_report.json')['winner']!=winner:
                raise ValueError('Отчёт test относится к другому победителю')
    device_setup()
    schedule = [(h,s) for h in HORIZONS for s in stages if jobs(h,s,cols)]
    completed = {}
    for h in HORIZONS:
        for s in stages:
            rel = f'h{h}/timings/{s}.json'
            if (h,s) not in schedule and existing(rel):
                completed[h,s] = load_json(rel)['stage_seconds']
    ETA_CONTEXT = dict(pending=schedule,completed=completed)
    for h in HORIZONS:
        pending = [s for horizon,s in schedule if horizon==h]
        if not pending:
            continue
        with TemporaryDirectory() as tmp:
            start = time.monotonic()
            last_origin = max(query_indices(*stage_spec(s))[-1] for s in pending)
            data = HorizonData(tmp,h,int(last_origin)+1)
            feature_seconds = time.monotonic()-start
            print(f'h{h}: признаки {feature_seconds/60:.1f} мин')
            future = {other for other,_ in schedule if HORIZONS.index(other)>HORIZONS.index(h)}
            ETA_CONTEXT.update(future_features=len(future),features=feature_seconds)
            for stage in pending:
                start = time.monotonic()
                ETA_CONTEXT.update(current=(h,stage),start=start)
                timing = run_stage(data,stage,cols)
                duration = time.monotonic()-start
                completed[h,stage] = duration
                save(f'h{h}/timings/{stage}.json',dict(stage_seconds=duration,features_seconds=feature_seconds,**timing))
                display(pd.DataFrame([dict(horizon=h,stage=stage,seconds=duration,**timing)]))
                show_eta()
            del data
        gc.collect()
