def route_features(data, origins, r):
    raw = data.matrix(origins,r)
    names = data.cols+data.corr_names
    keep = [i for i,c in enumerate(names) if c not in ('dayofweek','is_weekend')]
    hours = times[origins] + data.h*STEP
    angle = (hours.hour*2+hours.minute/30)*2*np.pi/48
    matrix = np.column_stack([raw[:,keep],np.sin(angle),np.cos(angle)]).astype(np.float64)
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix

def route_item(data, origins, query, r, stage):
    a = route_features(data,origins,r)
    b = route_features(data,query,r)
    imputer = SimpleImputer(strategy='median',keep_empty_features=True)
    a = imputer.fit_transform(a)
    b = imputer.transform(b)
    active = np.flatnonzero(np.ptp(a,axis=0)>0)
    if not len(active):
        active = [0]
    scaler = StandardScaler()
    a = scaler.fit_transform(a[:,active])
    b = scaler.transform(b[:,active])
    mean = a.mean(axis=0)
    y = target[origins+data.h,r].astype(np.float64)
    qy = np.zeros(len(query)) if stage=='full' else target[query+data.h,r].astype(np.float64)
    return dict(x=a-mean,query_x=b-mean,y=y,y_mean=float(y.mean()),query_y=qy,
                original_train=np.ones(len(y),dtype=bool),phase=stage,r=r)

def route_result(items, reference):
    batch = DeviceBatch(items,spectral=False)
    chosen = np.full(len(items),ROUTE_ALPHA)
    coef = batch.coefficients(chosen,False)
    metrics, predictions = batch.metrics(coef)
    if reference:
        for row,item in enumerate(items[:2]):
            reference_model = Ridge(alpha=chosen[row],solver='cholesky').fit(item['x'],item['y'])
            expected = np.maximum(0,reference_model.predict(item['query_x']))
            if not np.allclose(expected,predictions[row,:len(expected)],rtol=1e-6,atol=max(1.,np.abs(item['y']).max())*1e-7):
                raise ValueError('Проверка маршрутной Ridge против sklearn не пройдена')
    return predictions,metrics

def route_with_retry(items,reference):
    try:
        return route_result(items,reference)
    except cp.cuda.memory.OutOfMemoryError:
        if len(items)==1:
            raise
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    mid = len(items)//2
    parts = [route_with_retry(items[s],reference)
             for s in (slice(0,mid),slice(mid,None))]
    return np.concatenate([v for v,_ in parts]),{k:parts[0][1][k]+parts[1][1][k] for k in parts[0][1]}

def route_prefix(h,stage,regime):
    return f'h{h}/routes/{stage}/{regime}'

def route_stage(data,stage,regime,origins,query,boundary):
    started = time.monotonic()
    prefix = route_prefix(data.h,stage,regime)
    outputs,train_rows = [],[]
    for start in range(0,len(routes),8):
        rel = f'{prefix}/{start}.json'
        if existing(rel) is None:
            batch_start = time.monotonic()
            ids = range(start,min(start+8,len(routes)))
            items = [route_item(data,origins,query,r,stage) for r in ids]
            out, metrics = route_with_retry(items,start==0)
            rows = {}
            for split,values in metrics.items():
                rows[split] = combine([totals(v['n'],v['actual_sum'],v['absolute_error'],v['signed_error']) for v in values])
            save(rel,dict(values=out.tolist(),metrics=rows,seconds=time.monotonic()-batch_start,backend='cuda',
                          alpha=ROUTE_ALPHA,run_id=RUN_ID,boundary=str(boundary),fit_through=str(times[origins[-1]]+data.h*STEP)))
            display(pd.DataFrame(rows).T.assign(batch=start,stage=stage,regime=regime))
        batch = load_json(rel)
        outputs.append(np.asarray(batch['values']))
        train_rows.append(batch['metrics']['train'])
    rel = task_path(data.h,stage,regime,'route_ridge')
    finish_fit(rel,result_frame(data.h,query),np.concatenate(outputs).T.ravel(),combine(train_rows),
                   boundary,times[origins[-1]]+data.h*STEP,time.monotonic()-started,'cuda')
