def post_calibration_specs():
    for kind in ('horizon','groups','hierarchical','shift','affine','nonlinear','contextual'):
        for strength in (0.5,1.0):
            yield dict(kind=kind,strength=strength)

def calibration_context(frame, pred, columns):
    idx = times.get_indexer(pd.DatetimeIndex(frame.cutoff))
    rid = pd.Index(routes).get_indexer(frame.route_id)
    if (idx<0).any() or (rid<0).any():
        raise ValueError('Нет причинной истории для ситуационной калибровки')
    bases = frame[columns].to_numpy(dtype=float)
    return np.column_stack([np.log1p(np.maximum(pred,0)),
        np.log1p(np.maximum(bases.mean(axis=1),0)),np.log1p(bases.std(axis=1)),
        frame.horizon.to_numpy(),frame.cutoff.dt.dayofweek.to_numpy(),
        *[np.log1p(np.maximum(values[name][idx,rid],0)) for name in [TARGET,*STATUSES]]])

def calibration_group_keys(frame, state):
    if state['kind']=='horizon':
        return frame.horizon.astype(str).to_numpy()
    if state['kind']=='hierarchical':
        return (frame.route_id.astype(str)+'_'+frame.horizon.astype(str)).to_numpy()
    route_map = state['route_groups']
    return np.array([route_map.get(str(route),'unknown') for route in frame.route_id])

def fit_post_calibration(spec, history, columns):
    state = dict(kind=spec['kind'],strength=spec['strength'],columns=columns,identity=history.empty)
    if history.empty:
        return state
    p = history.raw_prediction.to_numpy(dtype=float)
    y = history.y.to_numpy(dtype=float)
    if not np.isfinite(p).all() or not np.isfinite(y).all() or (p<0).any() or (y<0).any():
        raise ValueError('Некорректная история калибровки')
    state.update(rows=len(history),fit_through=str(history.timestamp.max()))
    scale = max(float(y.mean()),1.)
    global_factor = float(np.clip(y.sum()/p.sum(),0.25,4.)) if p.sum()>0 else 1.
    state['global_factor'] = global_factor
    kind = state['kind']
    if kind in ('horizon','groups','hierarchical'):
        if kind=='groups':
            route_stats = history.assign(zero=history.y.eq(0)).groupby('route_id').agg(volume=('y','mean'),zero=('zero','mean'))
            edges = np.quantile(route_stats.volume,[1/3,2/3])
            zero_edge = float(route_stats.zero.median())
            state['route_groups'] = {str(route):f'{np.searchsorted(edges,row.volume)}_{int(row.zero>zero_edge)}'
                                     for route,row in route_stats.iterrows()}
            state.update(volume_edges=edges.tolist(),zero_edge=zero_edge)
        keys = calibration_group_keys(history,state)
        grouped = pd.DataFrame(dict(key=keys,y=y,p=p)).groupby('key').agg(y=('y','sum'),p=('p','sum'),n=('y','size'))
        state['horizon_factors'] = {}
        if kind=='hierarchical':
            for h, rows in history.groupby('horizon'):
                volume = float(rows.raw_prediction.sum())
                ratio = float(rows.y.sum()/volume) if volume>0 else global_factor
                weight = len(rows)/(len(rows)+200.)
                state['horizon_factors'][str(h)] = float(np.clip(weight*ratio+(1-weight)*global_factor,0.25,4.))
        state['factors'] = {}
        for key,row in grouped.iterrows():
            parent = state['horizon_factors'].get(key.rsplit('_',1)[-1],global_factor) if kind=='hierarchical' else global_factor
            ratio = row.y/row.p if row.p>0 else parent
            prior = 28. if kind=='hierarchical' else 200.
            weight = row.n/(row.n+prior)
            state['factors'][key] = float(np.clip(weight*ratio+(1-weight)*parent,0.25,4.))
    elif kind in ('shift','affine'):
        from scipy.optimize import minimize
        u,v = p/scale,y/scale
        def objective(theta):
            a,b = (1.,theta[0]) if kind=='shift' else theta
            error = np.maximum(0.,a*u+b)-v
            return (np.abs(error).sum()+abs(error.sum()))/max(v.sum(),1.)+0.01*((a-1)**2+b*b)
        initial = [0.] if kind=='shift' else [1.,0.]
        bounds = [(-1.,1.)] if kind=='shift' else [(0.5,1.5),(-1.,1.)]
        result = minimize(objective,initial,method='Powell',bounds=bounds,options=dict(maxiter=100,xtol=1e-5,ftol=1e-6))
        chosen = result.x if np.isfinite(result.fun) and result.fun<objective(initial) else initial
        a,b = (1.,chosen[0]) if kind=='shift' else chosen
        state.update(a=float(a),b=float(b*scale),optimizer_success=bool(result.success))
    elif kind=='nonlinear':
        from sklearn.isotonic import IsotonicRegression
        groups = np.searchsorted(np.unique(np.quantile(p,np.linspace(0,1,33))),p,side='right')
        bins = pd.DataFrame(dict(group=groups,p=p,y=y)).groupby('group').agg(p=('p','mean'),y=('y','mean'),n=('y','size'))
        model = IsotonicRegression(y_min=0.,out_of_bounds='clip').fit(bins.p,bins.y,sample_weight=bins.n)
        state.update(x=model.X_thresholds_.tolist(),y=model.y_thresholds_.tolist())
    elif kind=='contextual':
        from sklearn.ensemble import HistGradientBoostingRegressor
        from threadpoolctl import threadpool_limits
        model = HistGradientBoostingRegressor(max_iter=60,max_leaf_nodes=7,max_depth=3,
                    learning_rate=0.05,min_samples_leaf=200,l2_regularization=10.,early_stopping=False,random_state=42)
        with threadpool_limits(limits=THREADS):
            model.fit(calibration_context(history,p,columns),(y-p)/scale)
        state.update(model=model,scale=scale)
    else:
        raise ValueError(f'Неизвестная калибровка {kind}')
    return state

def apply_post_calibration(state, frame, pred):
    p = np.asarray(pred,dtype=float)
    if state['identity']:
        return p.copy()
    kind = state['kind']
    if kind in ('horizon','groups','hierarchical'):
        keys = calibration_group_keys(frame,state)
        parents = [state['horizon_factors'].get(str(h),state['global_factor']) for h in frame.horizon]
        mapped = np.array([state['factors'].get(key,parent) for key,parent in zip(keys,parents)])
        adjusted = p*mapped
    elif kind in ('shift','affine'):
        adjusted = state['a']*p+state['b']
    elif kind=='nonlinear':
        adjusted = np.interp(p,state['x'],state['y'])
    else:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=THREADS):
            correction = state['model'].predict(calibration_context(frame,p,state['columns']))*state['scale']
        adjusted = p+np.clip(correction,-state['scale'],state['scale'])
    result = np.maximum(0.,p+state['strength']*(np.maximum(adjusted,0.)-p))
    if not np.isfinite(result).all():
        raise ValueError('Калибровка дала нечисловой прогноз')
    return result

def raw_meta_history(oof,label):
    frames = []
    for week in VALIDATION_WEEKS:
        valid = oof.loc[oof.stage==f'oof{week}']
        train = oof.loc[oof.cutoff<valid.cutoff.min()]
        raw = pd.read_parquet(require(f'meta/cv/{label}_{week}.parquet')).prediction.to_numpy()
        if len(raw)!=len(train)+len(valid):
            raise ValueError('Размер кеша прогнозов метамодели не совпадает')
        frames.append(valid.assign(raw_prediction=raw[len(train):]))
    return pd.concat(frames,ignore_index=True)

def compare_post_calibration(oof, spec, label, report, raw_label, history):
    rows = []
    for week in VALIDATION_WEEKS:
        valid = oof.loc[oof.stage==f'oof{week}']
        train = oof.loc[oof.cutoff<valid.cutoff.min()]
        rel = f'meta/cv/{label}_{week}.json'
        if existing(rel):
            cached = load_json(rel)
        else:
            tic = time.monotonic()
            past = history.loc[history.timestamp<valid.cutoff.min().normalize()]
            state = fit_post_calibration(spec['post_calibration'],past,spec['columns'])
            raw = pd.read_parquet(require(f'meta/cv/{raw_label}_{week}.parquet')).prediction.to_numpy()
            p = apply_post_calibration(state,train,raw[:len(train)])
            q = apply_post_calibration(state,valid,raw[len(train):])
            model_rows = validation_rows(label,week,train,valid,p,q)
            cached = dict(model=model_rows,seconds=time.monotonic()-tic,calibration_rows=len(past),
                          fit_through=state.get('fit_through'),identity=state['identity'],spec=spec)
            save(rel,cached)
        rows.extend(cached['model'])
    report.extend(rows)
    valid_rows = [r for r in rows if r['split']=='validation' and r['horizon']==0]
    return dict(id=label,mean_week_score=float(np.mean([r['score'] for r in valid_rows])),
                pooled_score=combine(valid_rows)['score'],kind=spec['kind'],n_columns=len(spec['columns']),spec=spec)

def final_post_calibration(stage,spec,train,query,p,q,extra=None):
    history = pd.read_parquet(require('meta/calibration_history.parquet'))
    if extra is not None:
        history = pd.concat([history,extra],ignore_index=True)
    if history.duplicated(KEY).any() or not (history.timestamp<query.cutoff.min()).all():
        raise ValueError('Нарушена временная граница калибровки')
    started = time.monotonic()
    state = fit_post_calibration(spec['post_calibration'],history,spec['columns'])
    a,b = apply_post_calibration(state,train,p),apply_post_calibration(state,query,q)
    import joblib
    path = ROOT/f'meta/{stage}_post_calibration.joblib'
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix('.partial')
    joblib.dump(state,temp)
    temp.replace(path)
    metadata = {k:v for k,v in state.items() if k!='model'}
    save(f'meta/{stage}_post_calibration.json',dict(**metadata,seconds=time.monotonic()-started))
    return a,b
