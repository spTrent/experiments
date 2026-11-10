import ast
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
n=json.loads((BASE/'notebooks/clustering/nightly_rich_gpu.ipynb').read_text())
config=''.join(n['cells'][1]['source'])
node=next(x for x in ast.parse(config).body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='EXPERIMENTS' for t in x.targets))
experiment=dict(name='cat_reference_0328_rich',algorithm='cat',residual=False,parameter_set='catboost_direct',loss='MAE',compact=False,
                fixed_trees=1500,fixed_params=dict(depth=7,learning_rate=.05,l2_leaf_reg=3,random_strength=1,bagging_temperature=1))
config=config.replace(ast.get_source_segment(config,node),'EXPERIMENTS = '+repr([experiment,dict(experiment,name=experiment['name']+'_recent28',time_half_life=28)]))
addon="def calibration_variants():\n    return [('cat_reference_k1',1)]\n"
config+='\nCODE_HASH = '+repr(hashlib.sha256((config+addon+''.join(''.join(c['source']) for c in n['cells'] if c['cell_type']=='code')).encode()).hexdigest())+'\n'
n['cells'][1]['source']=config.splitlines(True)
n['cells'].insert(-1,dict(cell_type='code',id='reference-variant',metadata={},source=addon.splitlines(True),outputs=[],execution_count=None))
n['cells'][0]['source']=['# CatBoost: параметры базового запуска 0.328\n','Общая модель на горизонт, 1500 деревьев, глубина 7, learning_rate=0.05, MAE, категориальный маршрут. Добавлен расширенный набор признаков, нормировка маршрутов и новая схема калибровки. Это перенос базовой конфигурации, а не точное воспроизведение всей старой стратегии с двумя последующими моделями остатков и EWMA. Подбор калибровки до test, полный refit перед сабмитом, GPU обязателен.\n']
for c in n['cells']:
    if c['cell_type']=='code':compile(''.join(c['source']),'reference','exec')
out=BASE/'notebooks/clustering/reference_cat_gpu.ipynb';out.write_text(json.dumps(n,ensure_ascii=False,indent=1));print(out)
