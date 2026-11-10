import ast
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
n = json.loads((BASE / 'notebooks/clustering/nightly_gpu.ipynb').read_text())
config = ''.join(n['cells'][1]['source'])
node = next(x for x in ast.parse(config).body if isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'EXPERIMENTS' for t in x.targets))
experiments = [dict(e, name=e['name'] + '_mse', loss='RMSE' if e['algorithm'] == 'cat' else 'reg:squarederror') for e in ast.literal_eval(node.value) if e['compact']]
config = config.replace(ast.get_source_segment(config, node), 'EXPERIMENTS = ' + repr(experiments))
config += '\nCODE_HASH = ' + repr(hashlib.sha256((config + ''.join(''.join(c['source']) for c in n['cells'] if c['cell_type'] == 'code')).encode()).hexdigest()) + '\n'
n['cells'][1]['source'] = config.splitlines(True)
n['cells'][0]['source'] = ['# CatBoost/XGBoost: контроль RMSE на компактных признаках\n', 'Те же исходные признаки49, подобранные параметры с лимитом900 деревьев и глубины6, k1/k8. Для direct используется RMSE, как в исходной оптимизации из artifacts. Контроль влияния objective при остальных прежних признаках. GPU обязателен, выбор калибровки до test, полный refit перед сабмитом.\n']
for cell in n['cells']:
    if cell['cell_type'] == 'code':
        compile(''.join(cell['source']), 'compact_mse', 'exec')
out = BASE / 'notebooks/clustering/compact_mse_gpu.ipynb'
out.write_text(json.dumps(n, ensure_ascii=False, indent=1))
print(out)
