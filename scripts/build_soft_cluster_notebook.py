import ast
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
n=json.loads((BASE/'notebooks/clustering/nightly_rich_gpu.ipynb').read_text())
config=''.join(n['cells'][1]['source'])
node=next(x for x in ast.parse(config).body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='EXPERIMENTS' for t in x.targets))
experiments=ast.literal_eval(node.value)
experiments=[dict(e,name=e['name']+'_soft',soft_cluster=True) for e in experiments if not e['name'].endswith('_mse')]
config=config.replace(ast.get_source_segment(config,node),'EXPERIMENTS = '+repr(experiments))
addon='''original_soft_features = features


def calibration_variants():
    return [(ACTIVE['algorithm']+'_soft_k1',1)]


def features(h, origins, segmentation):
    x,names=original_soft_features(h,origins,segmentation)
    labels=segmentation['labels'][max(CLUSTERS)]
    columns=[]
    for group in range(-1,max(CLUSTERS)):
        columns.append(np.tile((labels==group).astype(np.float32),len(origins)))
        names.append(f'soft_cluster_{group}')
    return np.column_stack([x,*columns]).astype(np.float32,copy=False),names
'''
config+='\nCODE_HASH = '+repr(hashlib.sha256((config+addon+''.join(''.join(c['source']) for c in n['cells'] if c['cell_type']=='code')).encode()).hexdigest())+'\n'
n['cells'][1]['source']=config.splitlines(True)
n['cells'].insert(-1,dict(cell_type='code',id='soft-cluster',metadata={},source=addon.splitlines(True),outputs=[],execution_count=None))
n['cells'][0]['source']=['# Общая модель с признаками кластеров\n','CatBoost/XGBoost MAE: 216 расширенных признаков и one-hot группы маршрута. Все маршруты обучают одну модель на горизонт, группа — дополнительный признак. Это контроль против жёсткого разделения данных по кластерам. Те же временные блоки, выбор калибровки до test, GPU и дедлайн.\n']
for c in n['cells']:
    if c['cell_type']=='code':compile(''.join(c['source']),'soft','exec')
out=BASE/'notebooks/clustering/soft_cluster_gpu.ipynb';out.write_text(json.dumps(n,ensure_ascii=False,indent=1));print(out)
