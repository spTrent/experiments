import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
original = json.loads((root/'stacking/blending/exact_session6_blending_kaggle.ipynb').read_text())
runtime = (root/'scripts/cluster_experiment/session6_server_restore.py').read_text()
cells=[]


def add(kind,text):
    if kind=='code':
        tree = ast.parse(text)
        gating = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'GatingNetwork'), None)
        tree.body = [node for node in tree.body if node is not gating and not (
            isinstance(node, ast.Import) and len(node.names) == 1 and node.names[0].name == 'torch')]
        if gating is not None:
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name in ('fit_gating', 'gate_weights'):
                    node.body.insert(0, ast.Import(names=[ast.alias(name='torch')]))
                    if node.name == 'fit_gating':
                        node.body.insert(1, gating)
        class Paths(ast.NodeTransformer):
            def visit_Call(self, node):
                self.generic_visit(node)
                if isinstance(node.func, ast.Name) and node.func.id == 'compile' and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    payload = node.args[0].value.replace("Path('/kaggle/working/session6_exact_replay')", "WORK_ROOT / 'session6_exact_replay'").replace("Path('/kaggle/working')", 'WORK_ROOT').replace("Path('/kaggle/input')", 'INPUT_ROOT')
                    payload = payload.replace('del d, extra, cutoffs, new, comp, f, z, model, item',
                                              'session6_verified_light = new\ndel d, extra, cutoffs, new, comp, f, z, model, item')
                    node.args[0] = ast.Constant(payload)
                return node
        text = ast.unparse(ast.fix_missing_locations(Paths().visit(tree)))
        compile(text,'server','exec')
    cell=dict(cell_type=kind,id=f'server-{len(cells)}',metadata={},source=(text.strip()+'\n').splitlines(True))
    if kind=='code':cell.update(execution_count=None,outputs=[])
    cells.append(cell)


add('markdown','''# Session6 recovery and best validated blend — server

Бленд cat_d4__ridge выбран до test. Сначала проверяется архивный Session6. Оценочные h1/h3/h4/h7 загружаются, h2/h5/h6/h8 обучаются по исходным признакам и фиксированным числам деревьев. GPU-переобучение может дать другие прогнозы, поэтому сохраняются построчные сравнения с эталоном. Score 0.317416 не гарантирован.

Первый этап использует исторический train+val-рецепт Session6: CatBoost до 4 октября, подбор числа деревьев исторически на 4–17 октября, поправки по OOS 4–17 октября, замороженный LightGBM86. Оценка 18–31 октября и конкурсный прогноз используют одинаковые обученные веса. EWMA причинно обновляется по прошедшим фактам с задержкой четыре часа, как в оригинале.

После оценки и первого CSV CatBoost и остаточные модели Session6 переобучаются. Остатки обучаются на OOS первого этапа. LightGBM86 сохраняется согласно исходному рецепту Session6, это явное исключение из полного refit всех компонентов для воспроизведения выбранного решения. Остальные базы и метамодель используют полную разрешённую историю. Новая независимая оценка test получена до полного переобучения, сам test ранее изучался.

Требуется NVIDIA GPU и среда с CuPy, cuML, CatBoost, LightGBM, XGBoost. Пути задаются в следующей ячейке. Все обучения запускает пользователь.''')
add('code','''from pathlib import Path
import os
WORK_ROOT = Path(os.environ.get('BLEND_WORK', '/root/experiments/blend_session6_server'))
INPUT_ROOT = Path(os.environ.get('BLEND_INPUT', '/root/experiments/blend_inputs'))
WORK_ROOT.mkdir(parents=True, exist_ok=True)
EXACT_ROOT = INPUT_ROOT / 'Session6_exact'
RECOVERED_ROOT = INPUT_ROOT / 'Session6_recovered_h4'
for path in [EXACT_ROOT/'ANSWERS_RU.md', RECOVERED_ROOT/'evaluation/h4/model.cbm']:
    if not path.is_file():
        raise FileNotFoundError(path)
import hashlib
import json
for relative, expected in json.loads((RECOVERED_ROOT/'SHA256.json').read_text()).items():
    if hashlib.sha256((RECOVERED_ROOT/relative).read_bytes()).hexdigest() != expected:
        raise ValueError(f'Повреждён файл h4: {relative}')
''')
for c in original['cells'][1:]:
    text=''.join(c['source'])
    if c['cell_type']=='code' and 'EXACT_DATA_DIR = None' in text:
        text="""import shutil
exact_work = WORK_ROOT / 'session6_exact_replay'
shutil.copytree(EXACT_ROOT, exact_work, dirs_exist_ok=True)
verification_scope = {'__name__': '__session6_verification__'}
"""
    text=text.replace("Path('/kaggle/working/session6_exact_replay')",'WORK_ROOT / \'session6_exact_replay\'')
    if "exec(compile(" in text and 'verification_scope)' in text:
        text=text.replace('verification_scope)', 'verification_scope)')
    text=text.replace("Path('/kaggle/working')",'WORK_ROOT').replace("Path('/kaggle/input')",'INPUT_ROOT')
    text=text.replace("verification_scope = {'__name__': '__session6_verification__'}", "verification_scope = {'__name__': '__session6_verification__', 'WORK_ROOT': WORK_ROOT}")
    if c['cell_type']=='code' and 'def enter_submission_scope(' in text:
        text = text.replace('EWMA uses only pre-boundary residuals', 'EWMA uses already observable historical residuals with a four-hour delay')
    if c['cell_type']=='code' and text.startswith('local = pd.read_parquet(exact_local)'):
        text=runtime+'\nsession_future = restored_session6()\n'
    if c['cell_type']=='code' and 'SUBMIT_CODE_HASH =' in text:
        text=text.replace('EVALUATION_DIR = None', "EVALUATION_DIR = INPUT_ROOT / 'blend_best_output'")
        text=text.replace('SESSION_DATA_DIR = None', "SESSION_DATA_DIR = INPUT_ROOT / 'session6'")
        text=text.replace('XGB_DATA_DIR = None', "XGB_DATA_DIR = INPUT_ROOT / 'xgb_rich'")
        text=text.replace("SUBMIT_CODE_HASH =", "SUBMIT_CODE_HASH =")
        lines=text.splitlines();lines[0]='SUBMIT_CODE_HASH = '+repr(hashlib.sha256(Path(__file__).read_bytes()+runtime.encode()+json.dumps(original).encode()).hexdigest());text='\n'.join(lines)
    add(c['cell_type'],text)
output=root/'stacking/blending/session6_recovered_blend_server.ipynb'
output.write_text(json.dumps(dict(nbformat=4,nbformat_minor=5,metadata=original['metadata'],cells=cells),ensure_ascii=False,indent=1)+'\n')
print(output)
