import hashlib
import json
from pathlib import Path


root = Path(__file__).resolve().parents[1]
notebook = json.loads((root / 'notebooks/clustering/xgboost_oof_meta_dual.ipynb').read_text())
runtime = (root / 'scripts/cluster_experiment/xgb_starts_calibration_search.py').read_text()
cells = notebook['cells'][:-1]
cells[0]['source'] = '''# XGBoost rich. Пять стартов и все получасы с отдельной калибровкой

Подключить `case_rwb`, включить GPU и выполнить Run All. Сравниваются две ветки XGBoost rich с одинаковыми признаками, параметрами и 8 кластерами. Первая обучается на стартах 09:30, 10:00, 10:30, 11:00 и 11:30. Вторая использует все доступные получасовые старты. Обе обучаются заново. Обучающие матрицы XGBoost создаются из CuPy-батчей и хранятся в сжатом виде на GPU. Подготовка признаков, KMeans и простые калибровки выполняются на CPU. Метамодели обучаются на CUDA.

Первые 21 день val дают три последовательных недельных OOF-блока. Калибровки обучаются на этих OOF, а последние 7 дней val независимо для каждой ветки выбирают тип и параметры по WAPE + |Relative Bias|. Для каждой ветки заново обучаем выбранную калибровку на всех 28 днях OOF val. Затем обе базы переобучаются на всех допустимых стартах train + val с целями строго до test. Калибровки замораживаются до оценки test. На test факты доступны только для метрик и причинных признаков после наступления соответствующего времени.

Перебор включает raw, постоянные множители, обученный общий множитель, варианты по горизонту, группе и группе × горизонту, иерархическую поправку маршрут × горизонт, свежие и сглаженные множители, сдвиг, affine, изотоническую нелинейную и простые XGBoost-метамодели коэффициента. Сетка от 0.50 до 1.50 включительно с шагом 0.05. Для общего affine отдельно перебирается наклон, сдвиг обучается на OOF. Для обучаемых групповых и других калибровок сетка задаёт общий множитель поверх обученной поправки. Это сохраняет относительные различия коэффициентов групп.

Последняя таблица содержит две строки. Для сопоставимости train и test оцениваются из одного старта 10:30, на восьми горизонтах. Train — внутривыборочная оценка базы после train + val. Test — оценка до полного обучения на всей истории. Локальный период уже использовался для диагностики. Таблицы перебора, прогнозы, модели и итоговая таблица сохраняются в ZIP. Сабмит в этом эксперименте не формируется.
'''.splitlines(True)
config = '''STEP = pd.Timedelta(minutes=30)
ACTIVE = EXPERIMENTS[0]
PIPELINE_VERSION = 'xgb_five_vs_all_independent_calibration_v1'
CODE_HASH = __CODE_HASH__
MODEL_CLUSTERS = (8,)
CAT_CONFIGS = [dict(experiment=ACTIVE, parameters=PARAMETERS, model_starts=MODEL_STARTS, grid=list(FACTOR_GRID))]
run_independent_calibrations()
'''
digest = hashlib.sha256((runtime + config + ''.join(''.join(cell['source']) for cell in cells if cell['cell_type'] == 'code')).encode()).hexdigest()
for name, source in [('independent-calibration-search', runtime), ('run-independent-search', config.replace('__CODE_HASH__', repr(digest)))]:
    cells.append(dict(cell_type='code', id=name, metadata={}, source=source.splitlines(True), execution_count=None, outputs=[]))
for cell in cells:
    if cell['cell_type'] == 'code':
        compile(''.join(cell['source']), cell['id'], 'exec')
        cell.update(execution_count=None, outputs=[])
metadata = {key: notebook['metadata'][key] for key in ('kernelspec', 'language_info') if key in notebook['metadata']}
output = root / 'notebooks/clustering/xgb_five_vs_all_calibrations.ipynb'
output.write_text(json.dumps(dict(nbformat=4, nbformat_minor=5, metadata=metadata, cells=cells), ensure_ascii=False, indent=1))
print(output)
