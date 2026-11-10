import ast
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
folder=BASE/'notebooks/stepa/stepa_checkpoint_0328_bundle/code'
texts=[]
for name in ('baseline_common_features.py','baseline_correlations.py','baseline_horizon.py'):
    tree=ast.parse((folder/name).read_text())
    for node in ast.walk(tree):
        if isinstance(node,(ast.Module,ast.FunctionDef,ast.ClassDef)) and node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):
            node.body.pop(0)
    texts.append(ast.unparse(tree))
source='\n'.join(texts)
addon='''original_fit_factors_recent = fit_factors


def fit_factors(frame, strategy):
    if strategy.startswith('recent_global_'):
        half_life = float(strategy.rsplit('_',1)[-1])
        age = (frame.cutoff.max() - frame.cutoff).dt.total_seconds().to_numpy() / 86400
        weight = np.exp2(-age / half_life)
        ratio = float((frame.y * weight).sum() / (frame.y_pred * weight).sum())
        if not np.isfinite(ratio) or ratio <= 0:
            raise ValueError('Некорректная взвешенная калибровка')
        return {str(h):float(np.clip(ratio,*SCALE_BOUNDS)) for h in HORIZONS}
    return original_fit_factors_recent(frame,strategy)


original_features_rich = features
original_segment_rich = segment_routes


def segment_routes(boundary, stage):
    global FEATURE_NS
    result = original_segment_rich(boundary, stage)
    FEATURE_NS = dict(np=np, pd=pd, values=VALUES, TARGET=TARGET, STATUSES=STATUSES,
                      times=TIMES, routes=ROUTES, STEP=STEP, WINDOW_OFFSETS=tuple(range(-4,5)))
    exec(RICH_SOURCE, FEATURE_NS)
    lag_table = FEATURE_NS['calculate_dynamic_shoulder'](VALUES[TARGET], FEATURE_NS['hub_signal'], TIMES, ROUTES, fit_before=boundary)
    FEATURE_NS['validation_lag_table'] = lag_table
    FEATURE_NS['network_values'] = {name:array.sum(axis=1,dtype=np.float64).astype(np.float32) for name,array in VALUES.items()}
    lag_table.to_csv(RUN / f'rich_status6_lags_{stage}.csv', index=False)
    return result


def features(h, origins, segmentation):
    base, names = original_features_rich(h, origins, segmentation)
    frame = FEATURE_NS['horizon_dataset'](h, origins)
    cols = [c for c in frame if c not in ('route_id','cutoff','timestamp','y','id','overlap')]
    values = frame[cols].to_numpy(dtype=np.float32)
    scale = np.tile(segmentation['scale'],len(origins)).astype(np.float32)
    dimensional = [i for i,c in enumerate(cols) if c not in ('hour','minute','hour_float','dayofweek','is_weekend') and not any(v in c for v in ('ratio','rate','correlation','pair_count','lag_steps','valid','available','_sin','_cos'))]
    for i in dimensional:
        values[:,i] /= scale
    available = np.isfinite(values).astype(np.float32)
    values = np.where(np.isfinite(values),values,0)
    shoulder_columns, shoulder_names = [], []
    for label, days in [('daily7',range(1,8)),('weekly4',(7,14,21,28))]:
        profiles = []
        for offset in (-4,-2,0,2,4):
            total = np.zeros((len(origins),len(ROUTES)),dtype=np.float32)
            count = np.zeros((len(origins),1),dtype=np.float32)
            for day in days:
                positions = origins + h - day*48 + offset
                valid = positions >= 0
                total += np.where(valid[:,None],VALUES[TARGET][np.maximum(positions,0)],0)
                count += valid[:,None]
            profile = np.divide(total,count,out=np.zeros_like(total),where=count>0) / segmentation['scale'][None,:]
            profiles.append(profile.ravel())
            shoulder_columns.append(profile.ravel())
            shoulder_names.append(f'rich_{label}_shoulder_{offset}')
        window = np.column_stack(profiles)
        shoulder_columns.extend([window.max(axis=1)-window.min(axis=1),np.argmax(window,axis=1)*60-120])
        shoulder_names.extend([f'rich_{label}_shoulder_range',f'rich_{label}_peak_minutes'])
    for signal, array in FEATURE_NS['network_values'].items():
        for lag in (0,2):
            signal_values = np.where(origins>=lag,array[np.maximum(origins-lag,0)],0) / max(float(segmentation['scale'].sum()),1.)
            shoulder_columns.append(np.repeat(signal_values,len(ROUTES)))
            shoulder_names.append(f'rich_network_{signal}_back_{lag}')
        change = (array[origins]-array[np.maximum(origins-2,0)]) / max(float(segmentation['scale'].sum()),1.)
        shoulder_columns.append(np.repeat(change,len(ROUTES)))
        shoulder_names.append(f'rich_network_{signal}_change_1h')
    shoulder_columns.append(np.tile(np.arange(len(ROUTES),dtype=np.float32),len(origins)))
    shoulder_names.append('rich_route_index')
    share = VALUES[TARGET][origins] / np.maximum(FEATURE_NS['network_values'][TARGET][origins,None],1)
    shoulder_columns.append(share.ravel())
    shoulder_names.append('rich_route_target_network_share')
    matrix = np.column_stack([base,values,available,*[np.asarray(c,dtype=np.float32) for c in shoulder_columns]]).astype(np.float32,copy=False)
    names = names + ['rich_'+c for c in cols] + ['rich_available__'+c for c in cols] + shoulder_names
    if not np.isfinite(matrix).all():
        raise ValueError('Неконечные расширенные признаки')
    return matrix,names
'''
night_source=(BASE/'scripts/cluster_experiment/nightly.py').read_text()
night_tree=ast.parse(night_source)
fit_node=next(x for x in night_tree.body if getattr(x,'name','')=='fit_model')
fit_source=ast.get_source_segment(night_source,fit_node)
fit_source=fit_source.replace("allow_const_label=True, bootstrap_type='Bayesian')", "allow_const_label=True, bootstrap_type='Bayesian', cat_features=['route_code'])")
fit_source=fit_source.replace("tree_method='hist', device='cuda:0', n_jobs=THREADS, random_state=SEED)", "tree_method='hist', device='cuda:0', n_jobs=THREADS, random_state=SEED, enable_categorical=True, max_cat_to_onehot=1)")
fit_source=fit_source.replace("model.fit(train['x'][select], target[select], sample_weight=weights[select])", "model.fit(model_frame(train['x'][select],train['names'],ACTIVE['algorithm']), target[select], sample_weight=weights[select])")
fit_source=fit_source.replace('    models = {}',"    if ACTIVE.get('fixed_params'):\n        params=dict(ACTIVE['fixed_params'])\n        trees=int(ACTIVE['fixed_trees'])\n    models = {}")
fit_source=fit_source.replace('    weights = scales / scales.mean()',"    weights = scales / scales.mean()\n    if ACTIVE.get('time_half_life'):\n        age=(TIMES[train['origins']].max()-TIMES[train['origins']]).total_seconds().to_numpy()/86400\n        weights *= np.repeat(np.exp2(-age/ACTIVE['time_half_life']),len(ROUTES))\n        weights /= weights.mean()")
addon += "\n\ndef model_frame(x,names,algorithm):\n    frame=pd.DataFrame(x,columns=names)\n    route_codes=x[:,names.index('rich_route_index')].astype(np.int64)\n    frame['route_code']=pd.Categorical(route_codes,categories=np.arange(len(ROUTES))) if algorithm=='xgb' else route_codes.astype(str)\n    return frame\n\n\ndef predict_model(bundle,data,segmentation):\n    labels=segmentation['labels'][bundle['k']][data['route_index']]\n    pred=np.empty(len(labels),dtype=np.float64)\n    for group,item in bundle['models'].items():\n        indices=np.flatnonzero(labels==group)\n        for start in range(0,len(indices),PREDICT_BATCH):\n            rows=indices[start:start+PREDICT_BATCH]\n            frame=model_frame(data['x'][rows],bundle['feature_names'],bundle['experiment']['algorithm'])\n            pred[rows]=item['model'].predict(frame)\n    if bundle['residual']:\n        pred+=seasonal_base(data['x'],bundle['feature_names'])\n    return np.maximum(pred*segmentation['scale'][data['route_index']],0)\n\n\nexec(RICH_TRAINING_SOURCE)\n"
n=json.loads((BASE/'notebooks/clustering/nightly_gpu.ipynb').read_text())
config=''.join(n['cells'][1]['source'])
node=next(x for x in ast.parse(config).body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='EXPERIMENTS' for t in x.targets))
old=ast.get_source_segment(config,node)
exps=ast.literal_eval(node.value)
exps=[dict(e,name=e['name']+'_rich') for e in exps if e['compact']]
exps += [dict(e,name=e['name']+'_mse',loss='RMSE' if e['algorithm']=='cat' else 'reg:squarederror') for e in exps.copy()]
config=config.replace(old,'EXPERIMENTS = '+repr(exps))
config+='\nRICH_TRAINING_SOURCE = '+repr(fit_source)+'\n'
config+='\nRICH_SOURCE = '+repr(source)+'\nCODE_HASH = '+repr(hashlib.sha256((source+addon+config).encode()).hexdigest())+'\n'
n['cells'][1]['source']=config.splitlines(True)
cal=''.join(n['cells'][3]['source']).replace("'group_horizon_shrink50', 'group_horizon')","'group_horizon_shrink50', 'group_horizon', 'recent_global_2', 'recent_global_3', 'recent_global_7')")
cal=cal.replace('    display(summary)',"    display(summary)\n    print(summary[['variant','strategy','period','WAPE','relative_bias','WAPE + |Relative Bias|']].to_string(index=False),flush=True)")
cal=cal.replace("winner, choices[winner]['strategy'], flush=True","winner, choices[winner]['strategy'], 'selection_score', choices[winner]['selection_score'], flush=True")
n['cells'][3]['source']=cal.splitlines(True)
config += '\nCODE_HASH = '+repr(hashlib.sha256((source+fit_source+addon+cal+config+''.join(n['cells'][4]['source'])).encode()).hexdigest())+'\n'
n['cells'][1]['source']=config.splitlines(True)
n['cells'][0]['source']=['# GPU: признаки из запуска 0.328\n','Компактные CatBoost/XGBoost с признаками baseline из пакета 0.328: комбинации статусов, лаги, недельные сезонные профили, корреляционные лаги только по обучению. Компактные признаки сохраняются дополнительно. Это перенос признаков в новую нормировку и калибровку, а не точное воспроизведение 0.328. Дедлайн и временная оценка такие же, как в ночном цикле.\n']
n['cells'].insert(-1,dict(cell_type='code',id='rich-features',metadata={},source=addon.splitlines(True),execution_count=None,outputs=[]))
for c in n['cells']:
    if c['cell_type']=='code':compile(''.join(c['source']),'rich','exec')
out=BASE/'notebooks/clustering/nightly_rich_gpu.ipynb';out.write_text(json.dumps(n,ensure_ascii=False,indent=1));print(out)
