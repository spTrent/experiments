import ast
import hashlib
import json
from pathlib import Path


root = Path(__file__).resolve().parents[1]
template = json.loads((root / 'notebooks/clustering/xgboost_oof_smoothed_dual.ipynb').read_text())
legacy = json.loads((root / 'best/raw-vs-clean.ipynb').read_text())
runtime = (root / 'scripts/cluster_experiment/session6_xgb_starts.py').read_text()
cells = template['cells'][:-1]
cells[0]['source'] = '''# Session6 + XGBoost rich. Пять утренних стартов против всех получасов

Цель — снизить WAPE формы при фиксированном объёме и проверить, приблизит ли затем калибровка итог к 0.318. В обоих вариантах заменяется 25% вклада LightGBM в Session6, то есть XGBoost занимает примерно 12.5% всей смеси. Признаки, параметры, число деревьев и KMeans с 8 кластерами одинаковы. В новой ветке обучающие старты ограничены 09:30, 10:00, 10:30, 11:00 и 11:30. Оценка обеих веток идёт из 10:30 на восемь горизонтов.

Подключить в Kaggle `cleaning_comparison_inputs.zip` из `best/` и `cluster_experiment_4741653856669b1e.zip` из прогона XGBoost с метамоделью. ZIP-файлы должны быть доступны как входные файлы. Из первого берутся train/test parquet, параметры LightGBM и архив Session6. Из второго читаются только сохранённые raw OOF-прогнозы XGBoost со всех стартов. Включить GPU и выполнить Run All. XGBoost обучается на CUDA QuantileDMatrix, CatBoost базы и поправки тоже на GPU. LightGBM восстанавливается на CPU по рецепту Session6.

Для val восстанавливаем сопоставимый Session6 по его рецепту. CatBoost и LightGBM для этой ветки обучаются на целях до 27 сентября. Прогнозы 27 сентября–3 октября нужны для первого fit остатка. Для недель 4–10 и 11–17 октября остаточные модели обучаются только до начала своей недели. EWMA объёма учитывает лишь уже доступные факты с запасом четыре часа. Так архивные оценки калибратора на его собственном train не используются для выбора.

Победитель формы определяется по средней WAPE смеси на двух неделях val при одинаковых объёмах. После этого подбирается сила усадки единого множителя объёма к 1 по WAPE + |bias|. Множитель оценивается на первой неделе, кандидаты проверяются на второй. Выбранная сила применяется к коэффициенту, оценённому по обеим неделям, и фиксируется до test. На локальном test 18–31 октября обе формы оцениваются с одним архивным Session6, который использует исходный причинный протокол объёма. Восстановленный Session6 на val и архивный Session6 на test различаются обучающей историей, поэтому перенос калибровки проверяется отдельно. Параметры фиксируются до чтения прогнозов test, период ранее использовался для диагностики.

Таблицы содержат WAPE, signed bias и конкурсный score по этапам, дням и горизонтам. Итоговый ZIP включает модели новой ветки, восстановленный контроль, построчные прогнозы, протокол и таблицы. Этот эксперимент оценивает локальный test до полного переобучения. Для конкурсного сабмита нужен отдельный полный refit выбранных компонентов.
'''.splitlines(True)


def add(source, name):
    compile(source, name, 'exec')
    cells.append(dict(cell_type='code', id=name, metadata={}, source=source.splitlines(True), execution_count=None, outputs=[]))


anchor_parts = [''.join(legacy['cells'][index]['source']) for index in (5, 6, 7, 8, 9, 10, 11, 13)]
anchor_parts[4] = anchor_parts[4].replace('def features(', 'def session_shape_features(')
anchor_parts[6] = anchor_parts[6].replace(' = features(', ' = session_shape_features(')
anchor_parts[6] = anchor_parts[6].replace('frame.cutoff >= start_test', 'frame.cutoff >= VAL_FIRST').replace("'test',frame.loc[test", "'val',frame.loc[test")
anchor_parts[7] = anchor_parts[7].replace('frame.cutoff >= start_test', 'frame.cutoff >= fit_before').replace("stage,'all','test'", "stage,'all','val'")
anchor_parts[7] = anchor_parts[7].replace("thread_count=4,allow_writing_files=False,verbose=False)", "thread_count=4,task_type='GPU',devices='0',allow_writing_files=False,verbose=False)")
anchor = '\n\n'.join(ast.unparse(ast.parse(part)) for part in anchor_parts)
add(anchor, 'session6-causal-anchor')
add(runtime, 'starts-comparison')
setup = '''from io import BytesIO
from tempfile import TemporaryDirectory
import lightgbm as lgb

STEP = pd.Timedelta(minutes=30)
ANCHOR_SEED = pd.Timestamp('2025-09-27')
VAL_FIRST = pd.Timestamp('2025-10-04')
VAL_SECOND = pd.Timestamp('2025-10-11')
MORNING_STARTS = (570, 600, 630, 660, 690)
keys = ['cutoff', 'horizon', 'route_id']
ACTIVE = EXPERIMENTS[0]
PIPELINE_VERSION = 'session6_xgb_5_vs_all_v1'
CODE_HASH = __CODE_HASH__
MODEL_CLUSTERS = (8,)
CAT_CONFIGS = [dict(experiment=ACTIVE, parameters=PARAMETERS, morning_starts=list(MORNING_STARTS))]
search = Path('/kaggle/input')


def locate(filename):
    matches = list(search.rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError('Подключите ровно один архив ' + filename)
    return matches[0]


SESSION_ARCHIVE = locate('cleaning_comparison_inputs.zip')
XGB_ARCHIVE = locate('cluster_experiment_4741653856669b1e.zip')
with TemporaryDirectory() as directory:
    with zipfile.ZipFile(SESSION_ARCHIVE) as archive:
        for filename in ('train_solo_track.parquet', 'test_solo_track.parquet'):
            (Path(directory) / filename).write_bytes(archive.read(filename))
    TRAIN_PATH = str(Path(directory) / 'train_solo_track.parquet')
    TEST_PATH = str(Path(directory) / 'test_solo_track.parquet')
    run_starts_comparison()
'''
digest = hashlib.sha256((''.join(''.join(c['source']) for c in cells if c['cell_type'] == 'code') + setup).encode()).hexdigest()
add(setup.replace('__CODE_HASH__', repr(digest)), 'run-starts-comparison')
for item in cells:
    if item['cell_type'] == 'code':
        item.update(execution_count=None, outputs=[])
metadata = {key: template['metadata'][key] for key in ('kernelspec', 'language_info') if key in template['metadata']}
output = root / 'notebooks/clustering/session6_xgb_five_starts.ipynb'
output.write_text(json.dumps(dict(nbformat=4, nbformat_minor=5, metadata=metadata, cells=cells), ensure_ascii=False, indent=1))
print(output)
