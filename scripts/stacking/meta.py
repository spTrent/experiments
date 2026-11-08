def collect_stage(stage,columns):
    boundary,days = stage_spec(stage)
    origins = query_indices(boundary,days)
    blocks = []
    for h in range(1,9):
        block = result_frame(h,origins).set_index(KEY)
        for col in columns:
            regime,name = ('shared','profile28') if col=='profile28' else col.split('__')
            rel = task_path(h,stage,regime,name)
            info = load_json(rel+'.json')
            if (info['run_id']!=RUN_ID or pd.Timestamp(info['boundary'])!=boundary
                    or not pd.Timestamp(info['fit_through'])<boundary):
                raise ValueError(f'Граница обучения {rel}')
            frame = pd.read_parquet(require(rel)).set_index(KEY)
            if not frame.index.is_unique or not block.index.difference(frame.index).empty or len(frame)!=len(block):
                raise ValueError(f'Неполные или повторяющиеся ключи {rel}')
            frame = frame.reindex(block.index)
            for key in ('timestamp','y'):
                if key in block and not np.array_equal(block[key],frame[key]):
                    raise ValueError(f'Несовпадение {key} в {rel}')
            pred = frame.prediction.to_numpy()
            if not np.isfinite(pred).all() or (pred<0).any():
                raise ValueError(f'Некорректные значения {rel}')
            block[col] = pred
        blocks.append(block.reset_index())
    return pd.concat(blocks,ignore_index=True).assign(stage=stage,fit_before=boundary)

def convex_weights(x,y):
    from scipy.optimize import linprog
    scale = y.sum()
    if scale<=0:
        raise ValueError('Нулевой объём meta-train')
    k = x.shape[1]
    weights = np.full(k,1/k)
    best,upper = weights.copy(),np.inf
    cuts = []
    for _ in range(2000):
        err = x @ weights-y
        value = (np.abs(err).sum()+abs(err.sum()))/scale
        if value<upper:
            best,upper = weights.copy(),value
        grad = (np.sign(err)+np.sign(err.sum())) @ x /scale
        cuts.append((np.r_[grad,-1.],grad @ weights-value))
        a,b = zip(*cuts)
        lp = linprog(np.r_[np.zeros(k),1.],A_ub=np.asarray(a),b_ub=b,A_eq=[np.r_[np.ones(k),0.]],
                     b_eq=[1.],bounds=[(0,1)]*k+[(0,None)],method='highs')
        if not lp.success:
            raise RuntimeError(lp.message)
        if upper-lp.fun<=1e-7:
            best = np.maximum(best,0.)
            return best/best.sum()
        weights = lp.x[:-1]
    raise RuntimeError('LP не достиг допуска 1e-7')

def meta_context(frame,columns):
    x = frame[columns].to_numpy(dtype=np.float64)
    return np.column_stack([x,x.std(axis=1),frame.horizon,frame.cutoff.dt.dayofweek])

def calibration_factors(actual, predicted, strength):
    predicted = np.asarray(predicted,dtype=float)
    ratio = np.divide(actual,predicted,out=np.ones_like(predicted),where=predicted>0)
    return (1-strength)+strength*ratio

def calibrate_bases(train,query,cols,strength):
    values = train[cols].to_numpy(dtype=float)
    calibrated = np.empty_like(values)
    weeks = (train.cutoff.dt.normalize()-START).dt.days//7
    volume = 0.
    predicted = np.zeros(len(cols))
    for week in sorted(weeks.unique()):
        mask = weeks.eq(week)
        raw = values[mask]
        calibrated[mask] = raw*calibration_factors(volume,predicted,strength)
        volume += train.loc[mask,'y'].sum()
        predicted += raw.sum(axis=0)
    factors = calibration_factors(volume,predicted,strength)
    future = query[cols].to_numpy(dtype=float)*factors
    return train.assign(**dict(zip(cols,calibrated.T))),query.assign(**dict(zip(cols,future.T)))

def meta_predict(spec,train,query):
    cols = spec['columns']
    if spec.get('base_strength',0):
        train,query = calibrate_bases(train,query,cols,spec['base_strength'])
    x = train[cols].to_numpy(dtype=float)
    q = query[cols].to_numpy(dtype=float)
    y = train.y.to_numpy(dtype=float)
    kind = spec['kind']
    if kind=='weighted':
        weights = convex_weights(x,y)
        return x@weights,q@weights
    features = meta_context(train,cols)
    valid = meta_context(query,cols)
    if kind=='catboost':
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(iterations=spec['iterations'],depth=spec['depth'],learning_rate=0.03,
                                  task_type='GPU',random_seed=42,thread_count=THREADS,allow_writing_files=False,verbose=False)
        base = x.mean(axis=1)
        model.fit(features,y-base)
        return np.maximum(base+model.predict(features),0),np.maximum(q.mean(axis=1)+model.predict(valid),0)
    import torch
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    scaler = StandardScaler().fit(features)
    def tensor(a):
        return torch.as_tensor(a,dtype=torch.float32,device='cuda')
    net = torch.nn.Sequential(torch.nn.Linear(features.shape[1],16),torch.nn.ReLU(),
                              torch.nn.Linear(16,len(cols)),torch.nn.Softmax(dim=1)).cuda()
    torch.nn.init.zeros_(net[-2].weight)
    torch.nn.init.zeros_(net[-2].bias)
    a,b = tensor(scaler.transform(features)),tensor(scaler.transform(valid))
    u,v,t = tensor(x),tensor(q),tensor(y)
    optimizer = torch.optim.Adam(net.parameters(),lr=0.001)
    for _ in range(spec['epochs']):
        optimizer.zero_grad()
        err = (net(a)*u).sum(dim=1)-t
        loss = (err.abs().sum()+err.sum().abs())/t.sum()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return (net(a)*u).sum(dim=1).cpu().numpy(),(net(b)*v).sum(dim=1).cpu().numpy()

def specifications():
    for include in (False,True):
        cols = predictors(include)
        yield dict(kind='weighted',columns=cols)
        for depth in (3,5):
            for iterations in (100,300):
                yield dict(kind='catboost',depth=depth,iterations=iterations,columns=cols)
        for epochs in (50,150):
            yield dict(kind='gate',epochs=epochs,columns=cols)

def validation_rows(label,week,train,valid,pred_train,pred_valid):
    rows = []
    for split,frame,pred in [('train',train,pred_train),('validation',valid,pred_valid)]:
        for h in [0,*range(1,9)]:
            mask = np.ones(len(frame),dtype=bool) if h==0 else frame.horizon.eq(h)
            rows.append(dict(candidate=label,week=week,split=split,horizon=h,**metric(frame.y[mask],pred[mask])))
    display(pd.DataFrame(rows).query('horizon==0'))
    return rows

def compare_candidate(oof, spec, label, report, raw_label=None):
    rows = []
    past_actual,past_prediction = 0.,0.
    strength = spec.get('post_strength',0.)
    for week in (2,3,4):
        valid = oof.loc[oof.stage==f'oof{week}']
        train = oof.loc[oof.cutoff<valid.cutoff.min()]
        rel = f'meta/cv/{label}_{week}'
        if existing(rel+'.json'):
            cached = load_json(rel+'.json')
        else:
            tic = time.monotonic()
            if strength:
                raw = pd.read_parquet(require(f'meta/cv/{raw_label}_{week}.parquet')).prediction.to_numpy()
                factor = float(calibration_factors(past_actual,past_prediction,strength))
                p,q = raw[:len(train)]*factor,raw[len(train):]*factor
                raw_valid_volume = float(raw[len(train):].sum())
            else:
                p,q = meta_predict(spec,train,valid)
                raw_valid_volume = float(q.sum())
                if not spec.get('base_strength'):
                    save(rel+'.parquet',pd.DataFrame(dict(prediction=np.concatenate([p,q]))))
            model_rows = validation_rows(label,week,train,valid,p,q)
            baselines = []
            if spec['kind']=='weighted' and not (strength or spec.get('base_strength')):
                cols = spec['columns']
                single = min(cols,key=lambda c:metric(train.y,train[c])['score'])
                for key,tr,va in [(single,train[single],valid[single]),('mean',train[cols].mean(axis=1),valid[cols].mean(axis=1))]:
                    baselines.extend(validation_rows(f'baseline_{len(cols)}_{key}',week,train,valid,tr.to_numpy(),va.to_numpy()))
            cached = dict(model=model_rows,baselines=baselines,seconds=time.monotonic()-tic,
                          raw_valid_volume=raw_valid_volume,spec=spec)
            save(rel+'.json',cached)
        rows.extend(cached['model'])
        report.extend(cached['baselines'])
        past_actual += valid.y.sum()
        past_prediction += cached['raw_valid_volume']
    if strength:
        spec = dict(spec,post_factor=float(calibration_factors(past_actual,past_prediction,strength)),
                    post_oof_actual=float(past_actual),post_oof_prediction=float(past_prediction))
    valid_rows = [r for r in rows if r['split']=='validation' and r['horizon']==0]
    report.extend(rows)
    return dict(id=label,mean_week_score=float(np.mean([r['score'] for r in valid_rows])),
                pooled_score=combine(valid_rows)['score'],kind=spec['kind'],n_columns=len(spec['columns']),spec=spec)

def report_base_calibration(oof):
    rows = []
    for week in (2,3,4):
        valid = oof.loc[oof.stage==f'oof{week}']
        past = oof.loc[oof.cutoff<valid.cutoff.min()]
        cols = predictors(True)
        for strength in (0.,0.5,1.):
            factors = calibration_factors(past.y.sum(),past[cols].sum().to_numpy(),strength)
            for col,factor in zip(cols,factors):
                rows.append(dict(week=week,model=col,strength=strength,factor=factor,
                                 **metric(valid.y,valid[col]*factor)))
    table = pd.DataFrame(rows)
    save('meta/base_calibration.parquet',table)
    display(table.pivot_table(index='model',columns='strength',values='score'))

def assemble():
    oof = pd.concat([collect_stage(f'oof{i}',predictors(True)) for i in range(1,5)],ignore_index=True)
    if len(oof)!=224000 or oof.duplicated(KEY).any():
        raise ValueError('OOF должен содержать 224000 уникальных строк')
    save('meta/oof.parquet',oof)
    report_base_calibration(oof)
    report = []
    leaderboard = [compare_candidate(oof,spec,f'raw_{i:02}',report) for i,spec in enumerate(specifications())]
    def rank(entry):
        spec = entry['spec']
        return (entry['mean_week_score'],entry['n_columns'],entry['kind']!='weighted',
                bool(spec.get('base_strength') or spec.get('post_strength')),entry['id'])
    best = min(leaderboard,key=rank)
    for mode in ('base_strength','post_strength'):
        for strength in (0.5,1.0):
            spec = dict(best['spec'],**{mode:strength})
            leaderboard.append(compare_candidate(oof,spec,f'{mode}_{strength}',report,best['id']))
    ordered = sorted(leaderboard,key=rank)
    save('meta/validation.parquet',pd.DataFrame(report))
    save('meta/leaderboard.json',ordered)
    save('meta/winner.json',ordered[0]['spec'])
    display(pd.DataFrame(ordered).drop(columns='spec'))

def save_calibration(stage,spec,train,post_factor):
    cols = spec['columns']
    factors = calibration_factors(train.y.sum(),train[cols].sum().to_numpy(),spec.get('base_strength',0.))
    save(f'meta/{stage}_calibration.parquet',pd.DataFrame(dict(model=[*cols,'ensemble'],
                                                              factor=[*factors,post_factor])))

def evaluate():
    winner = load_json('meta/winner.json')
    oof = pd.read_parquet(require('meta/oof.parquet'))
    test = collect_stage('test',winner['columns'])
    a,b = meta_predict(winner,oof,test)
    raw_test = b.copy()
    post_factor = winner.get('post_factor',1.)
    a,b = a*post_factor,b*post_factor
    save_calibration('test',winner,oof,post_factor)
    rows = {'train':metric(oof.y,a),'test':metric(test.y,b)}
    save('meta/test_predictions.parquet',test.assign(prediction=b,raw_prediction=raw_test))
    save('meta/test_report.json',dict(winner=winner,metrics=rows,note='Независимая оценка test получена до переобучения на всей истории. Этот период ранее использовался в проекте.'))
    display(pd.DataFrame(rows).T)

def submit():
    winner = load_json('meta/winner.json')
    if load_json('meta/test_report.json')['winner']!=winner:
        raise ValueError('Изменилась конфигурация после оценки')
    cols = winner['columns']
    test = pd.read_parquet(require('meta/test_predictions.parquet'))
    oof = pd.read_parquet(require('meta/oof.parquet'))
    train = pd.concat([oof,test.loc[test.cutoff<TEST_START+7*DAY],collect_stage('extra',cols)],ignore_index=True)
    if len(train)!=336000 or train.duplicated(KEY).any():
        raise ValueError('Неполные финальные OOF')
    forecast = collect_stage('full',cols)
    a,p = meta_predict(winner,train,forecast)
    post_factor = 1.
    if winner.get('post_strength'):
        post_factor = float(calibration_factors(winner['post_oof_actual']+test.y.sum(),
                                                winner['post_oof_prediction']+test.raw_prediction.sum(),
                                                winner['post_strength']))
    a,p = a*post_factor,p*post_factor
    save_calibration('full',winner,train,post_factor)
    display(pd.DataFrame([metric(train.y,a)],index=['full OOF train']))
    forecast['y_pred'] = p
    result = pd.read_parquet(TEST_PATH)[['id','route_id','timestamp']].merge(
        forecast[['route_id','timestamp','y_pred']],on=['route_id','timestamp'],how='left',validate='one_to_one',sort=False)
    if len(result)!=8000 or not result.id.is_unique or not np.isfinite(result.y_pred).all() or (result.y_pred<0).any():
        raise ValueError('Нарушен контракт submission')
    path = ROOT/('submission_fast.csv' if PROFILE=='fast' else 'submission.csv')
    temp = path.with_suffix('.partial')
    result[['id','y_pred']].to_csv(temp,index=False)
    temp.replace(path)
    print(path)
