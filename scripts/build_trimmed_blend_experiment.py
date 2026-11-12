import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
folder = root / 'stacking/blending'
source = json.loads((folder / 'session6_recovered_blend_server.ipynb').read_text())
helpers = (root / 'scripts/cluster_experiment/trimmed_blend_experiment.py').read_text()
code_hash = hashlib.sha256(Path(__file__).read_bytes() + helpers.encode() + json.dumps(source).encode()).hexdigest()


def replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Expected one match for {old}')
    return text.replace(old, new)


class FitWindows(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)
        if not (isinstance(node.func, ast.Name) and node.func.id == 'compile' and isinstance(node.args[0], ast.Constant)):
            return node
        text = node.args[0].value
        if 'def fit_rich_cat(' in text:
            old = 'train_origins = np.flatnonzero(TIMES[:len(TIMES) - h] + h * STEP < start)'
            text = replace(text, old, old + "\n            train_origins = keep_recent_origins(train_origins, h, 'CatBoost rich')")
        if 'def fit_rich_xgb_full(' in text:
            old = 'origins = np.flatnonzero(TIMES[:len(TIMES) - h] + h * STEP < FULL_BOUNDARY)'
            text = replace(text, old, old + "\n        origins = keep_recent_origins(origins, h, 'XGB rich')")
        if 'def fit_rf(' in text:
            text = replace(text, 'train_origins = np.flatnonzero(np.isin(minute, TRAIN_MINUTES) & (times + h * STEP < start))',
                           "train_origins = keep_recent_origins(np.flatnonzero(np.isin(minute, rf_start_minutes(RF_START_COUNT)) & (times + h * STEP < start)), h, f'RF {RF_START_COUNT} slots')")
            checkpoint = next(line for line in text.splitlines() if "folder / f'h{h}_model.joblib'" in line)
            text = replace(text, checkpoint, "            if split != 'val':\n    " + checkpoint)
        if 'def fit_clean_xgb(' in text:
            text = replace(text, 'train = origins[TIMES[origins + h] < start]',
                           "train = keep_recent_origins(origins[TIMES[origins + h] < start], h, 'XGB clean')")
        compile(text, 'embedded', 'exec')
        node.args[0] = ast.Constant(text)
        return node


rf_definition = next(cell for cell in source['cells'] if ''.join(cell['source']).startswith('rf_runtime = dict(globals())'))
meta_source = next(''.join(cell['source']) for cell in source['cells']
                   if ''.join(cell['source']).startswith('import joblib') and 'def predict_shape(' in ''.join(cell['source']))

for cell in source['cells']:
    text = ''.join(cell['source'])
    if cell['cell_type'] == 'markdown':
        if cell is source['cells'][0]:
            text = '''# Бленд с последними 75% стартов: выбор окна RF

Отдельный эксперимент. CatBoost rich, XGB rich, XGB clean и RF отбрасывают первые floor(25%) допустимых стартов по времени внутри каждого горизонта. Для RF сравниваются 5 стартов (09:30–11:30), 9 стартов (08:30–12:30) и все 48 слотов. Его признаки, гиперпараметры и медианы остаются прежними. Отсечение применяется после выбора слотов и проверки границы цели.

На неделях 4–10 и 11–17 октября сравниваются три версии бленда с одинаковыми архивными прогнозами остальных баз, одинаковым объёмом Session6 и одной метамоделью, обученной на более ранних архивных OOF. Меняется только прогноз RF. Выбор по среднему WAPE + |Bias| фиксируется до test. Только победитель обучается для оценки test и обоих сабмитов. Результаты выбора находятся в rf_window_selection/validation_ranking.csv внутри папки запуска. Это сравнение замены RF при фиксированных OOF, поэтому общий выигрыш в новом бленде проверяется отдельно на test. Для признаков доступна более ранняя история до соответствующего старта. Лаги и статистики маршрутов по-прежнему оцениваются до границы этапа.

Внутри Session6 исходный CatBoost уже использовал последние 75% стартов. LightGBM86 здесь переобучается с таким же отсечением на сохранённой таблице его признаков, с исходными параметрами и числом деревьев. На первом этапе его обучающие цели строго до 4 октября, на финальном используются все доступные строки этой таблицы. Историческое расписание LightGBM сохраняется. Модели остатков, веса бленда, CatBoost-метамодель и Ridge используют прежние периоды OOS. Ранние OOF взяты из исходного запуска, поэтому их распределение может отличаться от прогнозов новых баз.

Сначала проводится локальная оценка и создаётся train_val CSV, затем выполняется финальное переобучение. Для каждого этапа создаётся дополнительный CSV с объёмами архивного Session6, как в варианте на 0.3258. Для train_val печатается сравнение на одинаковом локальном test. Эта оценка получена до финального переобучения. Test уже изучался. Настройки фиксированы до запуска.
'''
        cell['source'] = text.splitlines(True)
        continue
    if cell is rf_definition:
        text = meta_source + '\n' + text
    if text.startswith('SUBMIT_CODE_HASH ='):
        text = text.replace(text.splitlines()[0], f'SUBMIT_CODE_HASH = {code_hash!r}', 1)
        text += '\n' + helpers
    if text.startswith('def restored_session6():'):
        first = text.index("    light = verification_scope['session6_verified_light']")
        last = text.index("    result = all_predictions", first)
        text = text[:first] + '''    from predict_round100 import Predictor
    predictor = Predictor(base / 'session6_bundle', bundle / 'data', base / 'session6_bundle/reference_0328.csv', WORK_ROOT / 'lightgbm_future_cache')
    light = fit_recent_lightgbm(predictor, base, boundary)
''' + text[last:]
    if text.startswith("rf_future = rf_runtime['fit_rf']"):
        text = 'rf_future = fit_selected_rf(rf_runtime)\n'
    if text.startswith('finish_submission('):
        text += '\nsave_exact_volume_variant()\n'
    tree = ast.fix_missing_locations(FitWindows().visit(ast.parse(text)))
    text = ast.unparse(tree) + '\n'
    compile(text, 'experiment', 'exec')
    cell['source'] = text.splitlines(True)

output = folder / 'session6_trim25_rf_all_slots_server.ipynb'
output.write_text(json.dumps(source, ensure_ascii=False, indent=1) + '\n')
print(output)
