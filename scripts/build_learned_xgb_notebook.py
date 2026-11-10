import ast
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
n = json.loads((BASE / 'notebooks/clustering/learned_residual_gpu.ipynb').read_text())
source = ''.join(n['cells'][-2]['source'])
tree = ast.parse(source)
node = next(x for x in tree.body if isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'META_CONFIGS' for t in x.targets))
configs = [dict(max_depth=d, n_estimators=t, learning_rate=.03, reg_lambda=l, objective=o, min_child_weight=20, subsample=.9, colsample_bytree=.9) for d, t, l, o in [(3, 300, 50, 'reg:absoluteerror'), (4, 450, 30, 'reg:absoluteerror'), (5, 900, 50, 'reg:absoluteerror'), (3, 300, 50, 'reg:squarederror')]]
source = source.replace(ast.get_source_segment(source, node), 'META_CONFIGS = ' + repr(configs))
tree = ast.parse(source)
node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'meta_fit')
replacement = '''def meta_matrix(frame, columns):
    data = frame.loc[:, columns].copy()
    data['route_code'] = pd.Categorical(data.route_code, categories=[str(r) for r in ROUTES])
    data['horizon_code'] = pd.Categorical(data.horizon_code, categories=[str(h) for h in HORIZONS])
    return data


def meta_fit(frame, k, params):
    from xgboost import XGBRegressor
    models = {}
    labels = np.zeros(len(frame), dtype=int) if k == 1 else frame.group.to_numpy()
    columns = meta_columns(frame)
    target = ((frame.y - frame.y_pred) / frame.scale).to_numpy()
    weights = (frame.scale / frame.scale.mean()).to_numpy()
    for group in np.unique(labels):
        selected = np.ones(len(frame), dtype=bool) if group == -1 else labels == group
        model = XGBRegressor(**params, tree_method='hist', device='cuda:0', enable_categorical=True, max_cat_to_onehot=1, n_jobs=THREADS, random_state=SEED, verbosity=0)
        model.fit(meta_matrix(frame.loc[selected], columns), target[selected], sample_weight=weights[selected])
        device = json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
        if not device.startswith('cuda'):
            raise RuntimeError('Метамодель XGBoost обучена без CUDA')
        models[int(group)] = model
    return dict(k=k, params=params, columns=columns, models=models)
'''
source = source.replace(ast.get_source_segment(source, node), replacement)
source = source.replace("estimator.predict(frame.loc[selected,model['columns']])", "estimator.predict(meta_matrix(frame.loc[selected],model['columns']))")
n['cells'][-2]['source'] = source.splitlines(True)
config = ''.join(n['cells'][1]['source'])
config += '\nCODE_HASH = ' + repr(hashlib.sha256((config + source).encode()).hexdigest()) + '\n'
n['cells'][1]['source'] = config.splitlines(True)
n['cells'][0]['source'] = ['# XGBoost: обучаемая поправка к OOF CatBoost\n', 'Тот же временной протокол и сохранённая база, что в learned_residual_gpu. Поправку учит XGBoost CUDA с категориальными маршрутом и горизонтом, общую либо по 8 группам. Четыре компактные конфигурации MAE/RMSE, фиксированная смесь двух глубин, raw/ratio/rolling7. Выбор до test, затем полное переобучение метамоделей на доступных OOF. Rolling использует только уже доступные прошлые метки timestamp < cutoff − 4 часа. Базы для submission заранее переобучены на всей совпадающей истории.\n']
n['cells'][-1]['source'] += ['\narchive = Path(OUTPUT_ROOT) / (RUN.name + ".zip")\n', 'with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:\n', '    for path in sorted(RUN.rglob("*")):\n', '        if path.is_file():\n', '            bundle.write(path, path.relative_to(RUN))\n', 'print("Архив:", archive, flush=True)\n']
for cell in n['cells']:
    if cell['cell_type'] == 'code':
        compile(''.join(cell['source']), 'learned_xgb', 'exec')
out = BASE / 'notebooks/clustering/learned_xgb_gpu.ipynb'
out.write_text(json.dumps(n, ensure_ascii=False, indent=1))
print(out)
