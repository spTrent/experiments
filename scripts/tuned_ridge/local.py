from contextlib import nullcontext
import platform
import zipfile
import psutil
from importlib.metadata import version

TUNING_STARTS = tuple(pd.Timestamp(s) for s in ('2025-08-30','2025-09-06','2025-09-13'))
ALPHAS = np.sort(np.unique(np.r_[np.geomspace(1e-4,1e6,29),100.]))
TUNED_COLUMN = 'five_starts__route_ridge_tuned'

def local_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)

def local_parquet(path,frame):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix('.partial')
    frame.to_parquet(temp,index=False)
    temp.replace(path)

def local_npz(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix('.partial')
    with temp.open('wb') as stream:
        np.savez_compressed(stream,**arrays)
    temp.replace(path)

def local_initialize():
    global INPUT,TRAIN_PATH,TEST_PATH,LOCAL_ROOT,MANIFEST,xp,cp,cupyx,BACKEND
    if MODE not in ('oof','evaluate','submit') or not HORIZONS or any(h not in range(1,9) for h in HORIZONS):
        raise ValueError('MODE: oof/evaluate/submit; HORIZONS: 1–8')
    if BATCH_SIZE not in range(1,17):
        raise ValueError('BATCH_SIZE должен быть от 1 до 16, рекомендуется 4')
    INPUT = Path(INPUT_DIR).resolve()
    TRAIN_PATH,TEST_PATH = locate('train_solo_track.parquet'),locate('test_solo_track.parquet')
    packages = {name:version(name) for name in ('numpy','pandas','scikit-learn','pyarrow')}
    fingerprints = [file_digest(TRAIN_PATH),file_digest(TEST_PATH)]
    identity = dict(code_id=TUNED_CODE_ID,base_code_id=BASE_CODE_ID,profile=PROFILE,
                    fingerprints=fingerprints,alphas=ALPHAS.tolist(),packages=packages)
    run_id = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
    LOCAL_ROOT = Path(OUTPUT_DIR).resolve()/f'ridge_tuned_{run_id}'
    LOCAL_ROOT.mkdir(parents=True,exist_ok=True)
    MANIFEST = dict(**identity,run_id=run_id,column=TUNED_COLUMN,tuning_end='2025-09-20',
                    tuning_starts=[str(s) for s in TUNING_STARTS],oof_weeks=list(STACK_WEEKS))
    local_json(LOCAL_ROOT/'manifest.json',MANIFEST)
    cp,cupyx = None,None
    if DEVICE=='cuda':
        try:
            import cupy as cp
            import cupyx
            xp,BACKEND = cp,'cuda'
            cp.linalg.solve(cp.eye(2),cp.ones(2))
            cp.linalg.eigh(cp.eye(3)[None,:,:])
            cp.cuda.runtime.deviceSynchronize()
        except Exception as error:
            raise RuntimeError('CUDA не готова. Проверьте CuPy, драйвер и выбранное окружение Jupyter. См. инструкцию Windows.') from error
        gpu = str(cp.cuda.runtime.getDeviceProperties(0)['name'])
        free,total = cp.cuda.runtime.memGetInfo()
        print(f'GPU: {gpu}; свободно {free/2**30:.2f}/{total/2**30:.2f} ГиБ',flush=True)
    elif DEVICE=='cpu':
        xp,BACKEND = np,'cpu'
    else:
        raise ValueError('DEVICE: cuda или cpu')
    local_json(LOCAL_ROOT/'environment.json',dict(device=DEVICE,platform=platform.platform(),
               packages=packages,cupy=cp.__version__ if cp else None))
    print(f'RAM свободно: {psutil.virtual_memory().available/2**30:.2f} ГиБ; результаты: {LOCAL_ROOT}',flush=True)

def release_device():
    gc.collect()
    if BACKEND=='cuda':
        cp.get_default_memory_pool().free_all_blocks()

def close_horizon(data):
    for name in ('frame','corr'):
        array = getattr(data,name,None)
        if array is not None and hasattr(array,'_mmap'):
            array.flush()
            array._mmap.close()
            setattr(data,name,None)

def timed_log(event,started,**details):
    synchronize()
    row = dict(event=event,seconds=time.monotonic()-started,**details)
    with (LOCAL_ROOT/'timings.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(row,flush=True)

def tune_items(items):
    batch = DeviceBatch(items,spectral=True)
    scores = np.empty((len(items),len(ALPHAS)))
    zero = np.array([item['query_y'].sum()==0 for item in items])
    for j,alpha in enumerate(ALPHAS):
        coef = batch.coefficients(np.full(len(items),alpha),True)
        predictions = to_host(xp.maximum((batch.q@coef[:,:,None])[:,:,0]+batch.y_mean[:,None],0.))
        for r,item in enumerate(items):
            actual = item['query_y']
            error = predictions[r,:len(actual)]-actual
            volume = actual.sum()
            scores[r,j] = (np.abs(error).sum()+abs(error.sum()))/volume if volume>0 else np.abs(error).mean()
    if not np.isfinite(scores).all():
        raise ValueError('Нечисловые оценки подбора')
    return scores,zero

def fit_items(items,alphas,reference=False):
    batch = DeviceBatch(items,spectral=False)
    coef = batch.coefficients(alphas,False)
    metrics,predictions = batch.metrics(coef)
    if reference:
        for r,item in enumerate(items[:2]):
            expected = np.maximum(0.,Ridge(alpha=alphas[r],solver='cholesky').fit(item['x'],item['y']).predict(item['query_x']))
            if not np.allclose(expected,predictions[r,:len(expected)],rtol=1e-6,atol=1e-7*max(1.,np.max(np.abs(item['y'])))):
                raise ValueError('Ridge расходится с эталоном sklearn')
    return predictions,metrics

def retry_compute(items,alphas=None,reference=False):
    try:
        return tune_items(items) if alphas is None else fit_items(items,alphas,reference)
    except Exception as error:
        oom = isinstance(error,MemoryError) or (BACKEND=='cuda' and isinstance(error,cp.cuda.memory.OutOfMemoryError))
        if not oom or len(items)==1:
            raise
    release_device()
    middle = len(items)//2
    left = retry_compute(items[:middle],None if alphas is None else alphas[:middle],reference)
    right = retry_compute(items[middle:],None if alphas is None else alphas[middle:],reference)
    if alphas is None:
        return np.concatenate([left[0],right[0]]),np.concatenate([left[1],right[1]])
    return np.concatenate([left[0],right[0]]),{k:left[1][k]+right[1][k] for k in left[1]}

def tuning_fold(data,fold,boundary):
    origins = train_indices(data.h,boundary,'five_starts')
    query = query_indices(boundary,7)
    folder = LOCAL_ROOT/f'h{data.h}/tuning/fold{fold}'
    groups = [(start,min(start+BATCH_SIZE,len(routes))) for start in range(0,len(routes),BATCH_SIZE)]
    if any(not (folder/f'{start}_{end}.npz').is_file() for start,end in groups):
        started = time.monotonic()
        data.prepare(boundary,int(query[-1])+1)
        timed_log('tuning_lags',started,horizon=data.h,fold=fold)
    scores,zero = [],[]
    for start,end in groups:
        path = folder/f'{start}_{end}.npz'
        if not path.is_file():
            started = time.monotonic()
            items = [route_item(data,origins,query,r,'tuning') for r in range(start,end)]
            result,flags = retry_compute(items)
            local_npz(path,scores=result,zero_volume=flags,route_ids=routes[start:end])
            del items
            release_device()
            timed_log('tuning_batch',started,horizon=data.h,fold=fold,completed=end,total=len(routes))
        with np.load(path,allow_pickle=False) as batch:
            if not np.array_equal(batch['route_ids'],routes[start:end]) or batch['scores'].shape!=(end-start,len(ALPHAS)):
                raise ValueError(f'Некорректный checkpoint {path}')
            scores.append(batch['scores'].copy())
            zero.append(batch['zero_volume'].copy())
    return np.concatenate(scores),np.concatenate(zero)

def tuned_alphas(data):
    path = LOCAL_ROOT/f'h{data.h}/alphas.parquet'
    if path.is_file():
        table = pd.read_parquet(path)
        if not np.array_equal(table.route_id,routes):
            raise ValueError('Несовпадение маршрутов в alpha')
        return table.alpha.to_numpy()
    results,flags = [],[]
    for fold,boundary in enumerate(TUNING_STARTS,1):
        result,zero = tuning_fold(data,fold,boundary)
        results.append(result)
        flags.append(zero)
    scores = np.stack(results).mean(axis=0)
    selected = scores.argmin(axis=1)
    alpha = ALPHAS[selected]
    table = pd.DataFrame(dict(route_id=routes,horizon=data.h,alpha=alpha,
                 mean_tuning_score=scores[np.arange(len(routes)),selected],zero_volume_windows=np.stack(flags).sum(axis=0)))
    local_parquet(path,table)
    local_npz(path.with_suffix('.npz'),scores=scores,alphas=ALPHAS,route_ids=routes)
    display(table.alpha.describe())
    return alpha

def prediction_path(h,stage):
    return LOCAL_ROOT/f'predictions/h{h}/{stage}.parquet'

def predict_stage(data,stage,alphas):
    path = prediction_path(data.h,stage)
    info_path = path.with_suffix('.json')
    if path.is_file() and info_path.is_file():
        return
    boundary,days = stage_spec(stage)
    origins = train_indices(data.h,boundary,'five_starts')
    query = query_indices(boundary,days)
    if not (times[origins]+data.h*STEP < boundary).all():
        raise ValueError('Утечка целевых меток')
    folder = LOCAL_ROOT/f'h{data.h}/batches/{stage}'
    groups = [(start,min(start+BATCH_SIZE,len(routes))) for start in range(0,len(routes),BATCH_SIZE)]
    if any(not (folder/f'{start}_{end}.json').is_file() for start,end in groups):
        started = time.monotonic()
        data.prepare(boundary,int(query[-1])+1)
        timed_log('prediction_lags',started,horizon=data.h,stage=stage)
    predictions,train_rows = [],[]
    for start,end in groups:
        checkpoint = folder/f'{start}_{end}.json'
        if not checkpoint.is_file():
            started = time.monotonic()
            items = [route_item(data,origins,query,r,stage) for r in range(start,end)]
            out,metrics = retry_compute(items,alphas[start:end],start==0)
            rows = {key:combine([totals(v['n'],v['actual_sum'],v['absolute_error'],v['signed_error']) for v in val]) for key,val in metrics.items()}
            local_json(checkpoint,dict(predictions=out.tolist(),metrics=rows,route_ids=routes[start:end].tolist()))
            del items
            release_device()
            timed_log('prediction_batch',started,horizon=data.h,stage=stage,completed=end,total=len(routes))
        batch = json.loads(checkpoint.read_text(encoding='utf-8'))
        if not np.array_equal(batch['route_ids'],routes[start:end]):
            raise ValueError('Несовпадение маршрутов checkpoint')
        predictions.append(np.asarray(batch['predictions']))
        train_rows.append(batch['metrics']['train'])
    pred = np.concatenate(predictions).T.ravel()
    frame = result_frame(data.h,query).assign(prediction=pred)
    if not np.isfinite(pred).all() or (pred<0).any() or frame.duplicated(KEY).any():
        raise ValueError('Некорректный прогноз')
    rows = dict(train=combine(train_rows))
    if 'y' in frame:
        rows['held_out'] = metric(frame.y,pred)
    local_parquet(path,frame)
    local_json(info_path,dict(run_id=MANIFEST['run_id'],base_code_id=BASE_CODE_ID,boundary=str(boundary),
               fit_through=str(times[origins[-1]]+data.h*STEP),tuning_end='2025-09-20',metrics=rows))
    display(pd.DataFrame(rows).T.assign(horizon=data.h,stage=stage))

def export_results():
    destination = LOCAL_ROOT.parent/f'{LOCAL_ROOT.name}_export.zip'
    temp = destination.with_suffix('.partial')
    files = [LOCAL_ROOT/'manifest.json',LOCAL_ROOT/'environment.json']
    files += list(LOCAL_ROOT.glob('h*/alphas.parquet'))+list(LOCAL_ROOT.glob('predictions/h*/*'))
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path,arcname=str(Path(LOCAL_ROOT.name)/path.relative_to(LOCAL_ROOT)))
    temp.replace(destination)
    print(f'Экспорт для Kaggle: {destination}',flush=True)

def run_local():
    from threadpoolctl import threadpool_limits
    stages = [f'oof{i}' for i in STACK_WEEKS] if MODE=='oof' else ['test'] if MODE=='evaluate' else ['extra','full']
    with threadpool_limits(limits=2):
        for h in HORIZONS:
            pending = [stage for stage in stages if not (prediction_path(h,stage).is_file() and prediction_path(h,stage).with_suffix('.json').is_file())]
            if not pending:
                continue
            if MODE!='oof' and not (LOCAL_ROOT/f'h{h}/alphas.parquet').is_file():
                raise ValueError(f'Сначала выполните oof для h{h}, чтобы зафиксировать alpha')
            stop = max(query_indices(*stage_spec(stage))[-1] for stage in pending)+1
            scratch = Path(CACHE_DIR).resolve()
            scratch.mkdir(parents=True,exist_ok=True)
            if psutil.virtual_memory().available<2*2**30:
                raise MemoryError('Перед расчётом горизонта нужно освободить хотя бы 2 ГиБ RAM')
            with TemporaryDirectory(dir=scratch) as folder:
                started = time.monotonic()
                data = HorizonData(folder,h,int(stop))
                try:
                    timed_log('horizon_features',started,horizon=h)
                    alphas = tuned_alphas(data)
                    for stage in pending:
                        predict_stage(data,stage,alphas)
                finally:
                    close_horizon(data)
                    del data
                    release_device()
            export_results()
    export_results()
