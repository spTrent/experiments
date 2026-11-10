import ast
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
n = json.loads((BASE / 'notebooks/clustering/xgboost_rich_calibration.ipynb').read_text())
addon = (BASE / 'scripts/cluster_experiment/xgb_gpu_shrinkage.py').read_text()
config = ''.join(n['cells'][1]['source'])
for node in reversed(ast.parse(config).body):
    if isinstance(node, ast.Assign):
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if 'VAL_DAYS' in names:
            config = config.replace(ast.get_source_segment(config, node), 'VAL_DAYS = 28')
        if 'CAL_FIT_DAYS' in names:
            config = config.replace(ast.get_source_segment(config, node), 'CAL_FIT_DAYS = 21')
        if 'CODE_HASH' in names:
            config = config.replace(ast.get_source_segment(config, node), '')
config += '\nMODEL_CLUSTERS = (1, 8)\nGPU_BATCH_ROWS = 65536\nGPU_MAX_BIN = 256\nGPU_QUANTILE_BATCHES = 8\n'
config += 'CODE_HASH = ' + repr(hashlib.sha256((config + addon + ''.join(''.join(c['source']) for c in n['cells'] if c['cell_type'] == 'code')).encode()).hexdigest()) + '\n'
n['cells'][1]['source'] = config.splitlines(True)
addon = 'original_shrinkage_phase = phase_forecasts\n\n' + addon
n['cells'].insert(-1, dict(cell_type='code', id='gpu-shrinkage', metadata={}, execution_count=None, outputs=[], source=addon.splitlines(True)))
n['cells'][0]['source'] = '''# XGBoost rich: свежая OOF-калибровка с усадкой, CUDA-данные

Подключить case_rwb, включить GPU T4×2 и Run All. Обучение запускает пользователь. Параметры refit_params по восьми горизонтам встроены из xgb_tuning/best_params. 216 rich-признаков плюс категория маршрута, MAE, все получасовые старты обучения. Общая модель и8 кластеров, MODEL_CLUSTERS можно ограничить (8,) для сокращения времени. CLUSTERS=(1,8) оставьте для статистик сегментации.

Проверка CUDA обязательна, CPU fallback запрещён. CuPy передаёт признаки, таргет и веса на GPU через DataIter. QuantileDMatrix хранит сжатую обучающую матрицу на CUDA и строится батчами, без размещения всей исходной float32-матрицы на GPU одновременно. Модель обучается на CUDA; inplace_predict получает и возвращает CUDA-массивы. Исходный parquet, расчёт rich-признаков, KMeans и таблицы калибровки остаются на CPU. Это не перенос всего пайплайна на GPU. Используется GPU0; второй T4 автоматически не задействуется. При нехватке VRAM уменьшите GPU_BATCH_ROWS; уменьшение GPU_MAX_BIN также экономит память, но меняет модель. Нужна версия XGBoost с max_quantile_batches и GPU QuantileDMatrix(DataIter), CuPy и рабочая CUDA. Автоматической установки пакетов нет.

Локальный test14дней тот же, что раньше. Val расширен до28дней: первые21день дают три последовательных недельных OOF-блока с переобучением только по прошлой истории; последние7дней выбирают правило калибровки и вариант модели. Коэффициенты сравнения обучаются только на первых21 OOF-днях. Долгосрочный множитель sum(y)/sum(pred); свежий — отношение сумм с экспоненциальными весами и полупериодом3/7/14дней. Усаженный множитель = (1−alpha)×long + alpha×recent, alpha=.25/.5/.75. Также raw, fixed1.05, global и чистые recent. Всего15 правил, границы множителя0.8…1.25. Прогнозы и ошибки берутся в исходных единицах target, не по нормализованному таргету.

Правило выбирается по WAPE + |Relative Bias| на последней неделе val, без test. Затем коэффициенты пересчитываются на всех28 OOF-днях и фиксируются на весь test. Rolling-обновления по ответам test отсутствуют. Это одна отложенная неделя выбора, а не доказательство устойчивости на нескольких независимых неделях. Сравнение всех способов на test только отчётное; выбор после просмотра test запрещён. После fit печатаются доступные train/проверочные метрики, затем таблица train/test всех способов.

Перед submission выбранные модели переобучаются на всей размеченной истории, коэффициент выбранного правила обновляется по всем доступным OOF val+бывший test. Независимая от полного refit оценка test получена до переобучения; сам test уже изучался в проекте. Сохраняются submission.csv, submission_raw.csv, модели, OOF, коэффициенты и ZIP. GPU-обучение при подготовке ноутбука не запускалось.
'''.splitlines(True)
for cell in n['cells']:
    if cell['cell_type'] == 'code':
        compile(''.join(cell['source']), 'xgb_gpu_shrinkage', 'exec')
out = BASE / 'notebooks/clustering/xgboost_rich_gpu_shrinkage.ipynb'
out.write_text(json.dumps(n, ensure_ascii=False, indent=1))
print(out)
