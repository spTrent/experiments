import ast
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
n = json.loads((BASE / 'notebooks/clustering/nightly_gpu.ipynb').read_text())
config = ''.join(n['cells'][1]['source'])
tree = ast.parse(config)
node = next(x for x in tree.body if isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'EXPERIMENTS' for t in x.targets))
experiments = [dict(e, name=e['name'] + '_k16') for e in ast.literal_eval(node.value) if e['compact']]
config = config.replace(ast.get_source_segment(config, node), 'EXPERIMENTS = ' + repr(experiments))
node = next(x for x in tree.body if isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CLUSTERS' for t in x.targets))
config = config.replace(ast.get_source_segment(config, node), 'CLUSTERS = (1, 16)')
config += '\nCODE_HASH = ' + repr(hashlib.sha256((config + ''.join(''.join(c['source']) for c in n['cells'] if c['cell_type'] == 'code')).encode()).hexdigest()) + '\n'
n['cells'][1]['source'] = config.splitlines(True)
n['cells'][0]['source'] = ['# CatBoost и XGBoost на 16 кластерах\n', 'Контроль 16 групп против общей модели на исходных компактных признаках: число групп меняется при тех же признаках и компактных параметрах, что в ночных k8-запусках. GPU обязателен, MAE. Настройки и калибровка выбираются до test, затем полный refit перед сабмитом. Группы для калибровки фиксируются до val.\n']
for cell in n['cells']:
    if cell['cell_type'] == 'code':
        compile(''.join(cell['source']), 'cluster16', 'exec')
out = BASE / 'notebooks/clustering/cluster16_gpu.ipynb'
out.write_text(json.dumps(n, ensure_ascii=False, indent=1))
print(out)
