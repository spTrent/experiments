import ast
import json
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
archive = root / 'stacking/blending/session6_exact/Session6_exact_20261111.zip'
source = json.loads((root / 'stacking/blending/best_blend_submission_kaggle.ipynb').read_text())
with zipfile.ZipFile(archive) as z:
    verification = json.loads(z.read('Session6_exact/notebooks/verify_session6.ipynb'))


class Clean(ast.NodeTransformer):
    def visit_Assert(self, node):
        return ast.If(test=ast.UnaryOp(op=ast.Not(), operand=node.test), body=[ast.Raise(
            exc=ast.Call(func=ast.Name(id='ValueError', ctx=ast.Load()), args=[node.msg or ast.Constant('Session6 verification failed')], keywords=[]), cause=None)], orelse=[])


cells = []


def add(kind, text):
    if kind == 'code':
        text = ast.unparse(ast.fix_missing_locations(Clean().visit(ast.parse(text))))
        compile(text, '<exact>', 'exec')
    cell = dict(cell_type=kind, id=f'exact-{len(cells)}', metadata={}, source=(text.strip()+'\n').splitlines(True))
    if kind == 'code':
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)


add('markdown', '''# Точный Session6 и блендинг

Подключите распакованный `Session6_exact_20261111.zip`, исходный output `blend-best-2` и прежние два датасета Session6 и XGB rich. Включите GPU для остальных баз бленда.

Сначала исполняется проверочный рецепт из архива. Он восстанавливает финальные CatBoost из raw-данных и весов, модели остатков, LightGBM и формулу Session6. Локальные и конкурсные прогнозы сравниваются построчно с каноническими. Часть оценочных CatBoost-весов отсутствует, поэтому локальное воспроизведение использует сохранённые OOS-прогнозы этих баз. Это воспроизведение сохранённого решения, а не повторение всего обучения с нуля. Public 0.326 сообщён автором, повторная отправка автоматически не выполняется.

После проверки строится бленд по прежнему фиксированному cat_d4__ridge. Для локального test используется проверенный точный Session6. Для конкурсного прогноза используется точный финальный Session6 из архива. Исторический рецепт замораживает LightGBM и переобучает CatBoost и модели остатков. Этот режим явно запрошен для точного воспроизведения.

Два CSV бленда сохраняют прежние имена train_val и full_history для совместимости. Эти названия теперь описывают обучение остальных баз и метамодели. В обоих конкурсных CSV Session6 использует свой канонический финальный комплект. Поэтому первый CSV представляет гибрид с финальным Session6, а его локальная оценка использует исторический оценочный комплект Session6. Оценка test получена до полного переобучения остальных компонентов. Период test уже изучался.

В рабочую папку копируется комплект архива, чтобы исходные датасеты оставались доступными для проверки контрольных сумм. Потребуется место под распакованный архив. Ошибка любого построчного сравнения останавливает сборку.
''')
add('code', '''from pathlib import Path
import shutil
import os
EXACT_DATA_DIR = None
candidates = [Path(EXACT_DATA_DIR)] if EXACT_DATA_DIR else [p.parent for p in Path('/kaggle/input').rglob('ANSWERS_RU.md') if (p.parent/'session6_bundle/manifest.json').is_file()]
if len(candidates) != 1:
    raise ValueError(f'Укажите EXACT_DATA_DIR, найдено {candidates}')
exact_work = Path('/kaggle/working/session6_exact_replay')
shutil.copytree(candidates[0], exact_work, dirs_exist_ok=True)
verification_scope = {'__name__': '__session6_verification__'}
''')
for c in verification['cells']:
    if c['cell_type'] != 'code':
        continue
    text = ''.join(c['source'])
    text = text.replace("BASE = next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'session6_bundle/manifest.json').is_file())", "BASE = Path('/kaggle/working/session6_exact_replay')")
    cleaned = ast.unparse(ast.fix_missing_locations(Clean().visit(ast.parse(text))))
    add('code', f'exec(compile({cleaned!r}, "session6_verify", "exec"), verification_scope)')
add('code', '''exact_local = verification_scope['BASE'] / 'validation_predictions.parquet'
exact_future = verification_scope['BASE'] / 'outputs/submission_session6_reproduced.csv'
print('Session6 проверен. Локальный test и конкурсный прогноз совпали с архивом.')
''')
for c in source['cells'][1:]:
    text = ''.join(c['source'])
    if c['cell_type']=='code' and text.startswith('session_runtime ='):
        continue
    if c['cell_type']=='code' and text.startswith("session_future = session_runtime"):
        text = '''local = pd.read_parquet(exact_local).rename(columns={'session_prediction': 'session6'})
future = pd.read_csv(exact_future)
future = REQUESTS.merge(future, on='id', validate='one_to_one')
future['cutoff'] = FUTURE_CUTOFF
future['horizon'] = ((future.timestamp - FUTURE_CUTOFF) / STEP).astype(int)
future = future.rename(columns={'y_pred': 'session6'})
if SUBMISSION_SCOPE == 'train_val':
    original = read_session6_test()
    comparison = original.merge(local[KEYS + ['session6']], on=KEYS, validate='one_to_one', suffixes=('_old', '_exact'))
    if len(comparison) != len(original) or not np.allclose(comparison.session6_old, comparison.session6_exact, rtol=1e-12, atol=1e-6):
        raise ValueError('Архив Session6 отличается от входа исходного бленда')
    session_future = pd.concat([local[KEYS + ['session6']], future[KEYS + ['session6']]], ignore_index=True)
else:
    session_future = future[KEYS + ['session6']]
write_json(ARTIFACTS / 'session6_exact_protocol.json', dict(local_source=str(exact_local), future_source=str(exact_future), contest_uses_canonical_final_session6=True, full_retraining_from_scratch=False))
'''
    add(c['cell_type'], text)
output=root/'stacking/blending/exact_session6_blending_kaggle.ipynb'
output.write_text(json.dumps(dict(nbformat=4, nbformat_minor=5, metadata=source['metadata'], cells=cells),ensure_ascii=False,indent=1)+'\n')
print(output)
