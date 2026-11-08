class HorizonData:
    def __init__(self, folder, h, stop=None):
        self.h = h
        self.stop = len(times) if stop is None else stop
        self.folder = Path(folder)
        self.corr_names = None
        self.corr = None
        self.frame = None
        self.baseline = np.empty((self.stop,len(routes)),np.float32)
        self.origins = np.arange(self.stop)
        if PROFILE=='fast':
            grid = times[:self.stop]
            self.origins = self.origins[np.isin(grid.hour*60+grid.minute,TRAIN_MINUTES)]
        self.slots = np.full(self.stop,-1,dtype=int)
        self.slots[self.origins] = np.arange(len(self.origins))
        for start in range(0,len(self.origins),32):
            idx = self.origins[start:start+32]
            frame = horizon_dataset(h,idx)
            if self.frame is None:
                self.cols = [c for c in frame if c not in ('cutoff','timestamp','route_id','profile28','overlap')
                             and not (c.startswith(('selected_status6','status6_lag_valid','lagged_status6','residual_')))]
                self.frame = np.memmap(self.folder/'static.bin', mode='w+', dtype='float32',
                                       shape=(len(self.origins),len(routes),len(self.cols)))
            self.frame[self.slots[idx]] = frame[self.cols].to_numpy().reshape(len(idx),len(routes),-1)
            self.baseline[idx] = frame.profile28.to_numpy().reshape(len(idx),len(routes))
        self.frame.flush()

    def prepare(self, boundary, stop):
        table = fit_lags(boundary)
        origins = self.origins[self.origins<stop]
        for start in range(0,len(origins),64):
            idx = origins[start:start+64]
            keys = result_frame(self.h,idx)[['cutoff','route_id']]
            corr = add_correlation_features(keys,self.h,table).drop(columns=['cutoff','route_id'])
            if self.corr is None:
                self.corr_names = corr.columns.tolist()
                self.corr = np.memmap(self.folder/'correlations.bin',mode='w+',dtype='float32',
                                      shape=(len(self.origins),len(routes),len(self.corr_names)))
            self.corr[self.slots[idx]] = corr.to_numpy().reshape(len(idx),len(routes),-1)

    def matrix(self, origins, route=None):
        slots = self.slots[origins]
        if (slots<0).any():
            raise ValueError('Запрошен старт вне выбранного профиля')
        if route is None:
            a,b = self.frame[slots], self.corr[slots]
        else:
            a,b = self.frame[slots,route], self.corr[slots,route]
        return np.concatenate([a,b],axis=-1).reshape(-1,len(self.cols)+len(self.corr_names))

def global_data(data, origins, label, only_finite):
    shape = (len(origins)*len(routes),len(data.cols)+len(data.corr_names))
    matrix = np.memmap(data.folder/f'{label}.bin',mode='w+',dtype='float32',shape=shape,order='F')
    count = 0
    keys = []
    for start in range(0,len(origins),32):
        chunk = origins[start:start+32]
        rows = result_frame(data.h,chunk)
        rows['profile28'] = data.baseline[chunk].ravel()
        keep = np.isfinite(rows.profile28.to_numpy())
        if not only_finite and not keep.all():
            raise ValueError('Неполный сезонный прогноз на оценке')
        batch = data.matrix(chunk)[keep]
        matrix[count:count+len(batch)] = batch
        count += len(batch)
        keys.append(rows.loc[keep])
    if count == 0:
        raise ValueError('Пустой train')
    return matrix[:count],pd.concat(keys,ignore_index=True)

def numeric_data(x, q, train_ids, query_ids, folder, scale):
    width = x.shape[1]
    bits = max(1,len(routes).bit_length())
    medians = np.empty(width)
    means = np.zeros(width)
    scales = np.ones(width)
    for col in range(width):
        values = x[:,col]
        valid = values[np.isfinite(values)]
        medians[col] = np.median(valid) if len(valid) else 0.
        if scale:
            filled = np.where(np.isfinite(values),values,medians[col])
            means[col] = np.mean(filled,dtype=np.float64)
            scales[col] = np.std(filled,dtype=np.float64) or 1.
    outputs = []
    for label,matrix,ids in [('train',x,train_ids),('query',q,query_ids)]:
        out = np.memmap(folder/f'encoded_{label}.bin',mode='w+',dtype='float32',
                        shape=(len(ids),width+bits))
        codes = pd.Index(routes).get_indexer(ids)+1
        for start in range(0,len(ids),16384):
            rows = slice(start,start+16384)
            block = matrix[rows]
            filled = np.where(np.isfinite(block),block,medians)
            out[rows,:width] = (filled-means)/scales
            for bit in range(bits):
                out[rows,width+bit] = (codes[rows] >> bit) & 1
        outputs.append(out)
    return outputs

def ridge_coefficients(x,y,alpha,backend):
    mean = np.mean(x,axis=0,dtype=np.float64)
    y_mean = np.mean(y,dtype=np.float64)
    gram = backend.zeros((x.shape[1],x.shape[1]),dtype=backend.float64)
    rhs = backend.zeros(x.shape[1],dtype=backend.float64)
    for start in range(0,len(x),16384):
        rows = slice(start,start+16384)
        a = backend.asarray(x[rows],dtype=backend.float64)-backend.asarray(mean)
        b = backend.asarray(y[rows]-y_mean,dtype=backend.float64)
        gram += a.T @ a
        rhs += a.T @ b
    gram += alpha*backend.eye(len(mean))
    coef = backend.linalg.solve(gram,rhs)
    coef = coef.get() if hasattr(coef,'get') else np.asarray(coef)
    return coef,float(y_mean - mean @ coef)

def linear_predict(x, coef, intercept):
    result = np.empty(len(x))
    for start in range(0,len(x),16384):
        result[start:start+16384] = x[start:start+16384] @ coef + intercept
    return result

def verify_global_ridge(x,y,alpha):
    rows = np.linspace(0,len(x)-1,min(1024,len(x)),dtype=int)
    a,b = np.asarray(x[rows],dtype=np.float64),y[rows]
    coef,offset = ridge_coefficients(a,b,alpha,xp)
    expected = Ridge(alpha=alpha,solver='lsqr',tol=1e-5).fit(a,b).predict(a)
    if not np.allclose(a @ coef+offset,expected,rtol=2e-4,atol=2e-4*max(1.,np.max(np.abs(b)))):
        raise ValueError('GPU Ridge расходится с sklearn')

def tree_frame(matrix, columns, ids, catboost):
    frame = pd.DataFrame(matrix,columns=columns,copy=False)
    frame.insert(0,'route_id',ids.astype(str).to_numpy() if catboost else pd.Categorical(ids,categories=routes))
    return frame

def fit_global(name, cfg, x,q,train,query, columns, folder, encoded=None):
    family = name.rsplit('_',1)[0]
    residual = name.endswith('_residual')
    baseline = train.profile28.to_numpy() if residual else 0.
    y = (train.y.to_numpy(dtype=np.float64)-baseline)/TARGET_SCALE
    params = cfg['params']
    if family in ('ridge','rf'):
        a,b = encoded if encoded is not None else numeric_data(x,q,train.route_id,query.route_id,folder,family=='ridge')
        if family == 'ridge':
            verify_global_ridge(a,y,params['alpha'])
            coef,intercept = ridge_coefficients(a,y,params['alpha'],xp)
            train_pred, pred = (linear_predict(z,coef,intercept) for z in (a,b))
        else:
            from cuml.ensemble import RandomForestRegressor
            model = RandomForestRegressor(random_state=42,n_streams=4,output_type='numpy',**params)
            model.fit(a,y.astype(np.float32))
            train_pred,pred = (np.concatenate([np.asarray(model.predict(z[i:i+32768])).reshape(-1)
                                               for i in range(0,len(z),32768)]) for z in (a,b))
        device = 'cuda'
    else:
        a,b = (tree_frame(z,columns,frame.route_id,family=='catboost') for z,frame in ((x,train),(q,query)))
        if family == 'xgb':
            from xgboost import XGBRegressor
            model = XGBRegressor(n_estimators=cfg['trees'],device='cuda',tree_method='hist',enable_categorical=True,
                                 max_cat_to_onehot=1,objective='reg:squarederror',random_state=42,n_jobs=THREADS,**params)
            model.fit(a,y)
            device = json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
            if not device.startswith('cuda'):
                raise RuntimeError('XGBoost не использует GPU')
        elif family == 'catboost':
            from catboost import CatBoostRegressor
            model = CatBoostRegressor(iterations=cfg['trees'],loss_function='RMSE',task_type='GPU',random_seed=42,
                                      bootstrap_type='Bayesian',one_hot_max_size=1,allow_writing_files=False,
                                      thread_count=THREADS,verbose=False,**params)
            model.fit(a,y,cat_features=['route_id'])
            device = 'cuda'
        else:
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(n_estimators=cfg['trees'],objective='regression',device_type=LGB_DEVICE,
                                  subsample_freq=1,max_cat_to_onehot=1,verbosity=-1,random_state=42,n_jobs=THREADS,**params)
            model.fit(a,y,categorical_feature=['route_id'])
            device = LGB_DEVICE
        train_pred,pred = (np.concatenate([model.predict(z.iloc[i:i+32768]) for i in range(0,len(z),32768)])
                           for z in (a,b))
    train_pred = np.maximum(0.,train_pred*TARGET_SCALE+baseline)
    pred = np.maximum(0.,pred*TARGET_SCALE+(query.profile28.to_numpy() if residual else 0.))
    return metric(train.y,train_pred),pred,device
