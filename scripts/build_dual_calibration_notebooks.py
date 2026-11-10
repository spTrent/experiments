import ast
import copy
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
template = json.loads((BASE / 'notebooks/clustering/xgboost_rich_gpu_shrinkage.ipynb').read_text())
runtime = (BASE / 'scripts/cluster_experiment/calibration_dual_submissions.py').read_text()
for mode in ('scalar', 'meta'):
    notebook = copy.deepcopy(template)
    config = ''.join(notebook['cells'][1]['source'])
    for node in reversed(ast.parse(config).body):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CODE_HASH' for t in node.targets):
            config = config.replace(ast.get_source_segment(config, node), '')
    config += '\nCORRECTION_MODE = ' + repr(mode) + '\n'
    config += 'META_MIN_PRED_SCALE = 0.01\nMETA_STRENGTHS = (0.5, 1.0)\n'
    config += "META_CONFIGS = [dict(max_depth=1, trees=50, learning_rate=0.03, reg_lambda=100., min_child_weight=100.), dict(max_depth=1, trees=100, learning_rate=0.03, reg_lambda=100., min_child_weight=100.), dict(max_depth=2, trees=150, learning_rate=0.03, reg_lambda=100., min_child_weight=100.)]\n"
    config += 'CODE_HASH = ' + repr(hashlib.sha256((mode + config + runtime + ''.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')).encode()).hexdigest()) + '\n'
    notebook['cells'][1]['source'] = config.splitlines(True)
    notebook['cells'].insert(-1, dict(cell_type='code', id='dual-submission-runtime', metadata={}, execution_count=None, outputs=[], source=runtime.splitlines(True)))
    notebook['cells'][-1]['source'] = 'STEP = pd.Timedelta(minutes=30)\nrun_dual_submissions()\n'.splitlines(True)
    detail = 'Только скалярная калибровка: 15 правил, включая долгосрочный, свежий и смесь long/recent с полупериодами3/7/14дней и alpha=.25/.5/.75. Расчёт отношения взвешенных сумм на CuPy. Фиксированный множитель на весь test.' if mode == 'scalar' else 'Дополнительно обучается XGBoost CUDA-метамодель коэффициента. Входы: raw-прогноз в нормированном/логарифмическом виде, логарифм масштаба маршрута, текущий таргет, средние за2/24часа, изменение таргета зачас, статусы1–3, категориальные горизонт/группа. Всего12 признаков, без ID маршрута и полного rich-набора. Таргет y/raw, веса raw/mean(raw), objective squarederror; это даёт объёмный вес ошибкам коэффициента. Исключаются raw <= max(0.01×route_scale,1), для таких строк действует глобальный коэффициент. Прогноз множителя ограничен0.8…1.25. Три регуляризованные конфигурации depth1/50, depth1/100, depth2/150; L2=100, min_child_weight=100 и усадка предсказанного множителя к глобальному с силой.5/1. Сравнивается со всеми15 скалярными правилами. Если метамодель проиграет на val, для сабмитов автоматически выбирается скалярный baseline. Нельзя гарантировать уменьшение bias или итог0.318.'
    notebook['cells'][0]['source'] = f'''# XGBoost rich: {mode}, GPU-данные и два режима сабмита

{detail}

Подключить только case_rwb, включить GPU и Run All. Обучение запускает пользователь в Kaggle. Параметры XGBoost из xgb_tuning/best_params встроены по8 горизонтам с selected_trees без обрезки. 216 rich-признаков плюс категория маршрута, MAE у базовых моделей, все получасовые старты. MODEL_CLUSTERS=(1,8) сравнивает общую модель и8 кластеров; можно оставить(8,), CLUSTERS=(1,8) для статистик не менять.

Каждый fit базовой модели получает CuPy-батчи признаков/таргета/весов; QuantileDMatrix строится на CUDA и хранит сжатую обучающую матрицу в VRAM. Fit метамодели, если включён, получает CuPy-массивы целиком и CUDA QuantileDMatrix. Inplace predict баз и метамодели получает/возвращает CUDA-массивы. CPU fallback запрещён и проверяется конфигурация CUDA каждой модели. Исходные parquet, подготовка rich-признаков, KMeans, хранение OOF и отчёты остаются на CPU. Не требуется держать всю исходную float32-панель в VRAM одновременно. Используется GPU0; второй T4 автоматически не используется. Нужны CuPy/CUDA и XGBoost с GPU QuantileDMatrix(DataIter,max_quantile_batches). Ошибка VRAM не переключает обучение на CPU. GPU_BATCH_ROWS можно уменьшить, но вся сжатая матрица группы всё равно должна помещаться в VRAM. GPU_MAX_BIN можно уменьшить ценой изменения модели.

Хронология: val28дней до неизменного local test14дней. Первые21день дают три недельных OOF-блока: каждый обучается только на истории до своей недели. Последние7дней val выбирают вариант модели и калибровку по WAPE+|Relative Bias|. Коэффициенты/метамодель для выбора обучаются только на первых21 OOF-днях. После выбора калибровка переобучается на всех28 OOF-днях, фиксируется перед local test. Модели train+val обучаются на всех строках с target timestamp < TEST_START. Метки test не входят в fit или выбор, rolling-обновления на test отсутствуют. Любые метапризнаки считаются без чтения y и используют фиксированные обучаемые статистики до val. Известная на cutoff история доступна для причинных признаков. Test уже изучался в проекте и не является новым слепым holdout.

После каждого base fit печатаются train и доступный проверочный период; во время OOF test остаётся закрытым. При оценке train+val печатаются train/test raw. Метамодели печатают OOF train/selection. independent_metrics.csv содержит calibrated/raw local test и отдельно meta_train_oof_in_sample; эти OOF in-sample метрики калибратора не равны in-sample метрикам базы на всей train+val. Все отчёты, включая базовые train/test по горизонтам, сохраняются. Финальный отчёт повторяет независимую от full refit оценку.

Два основных файла: submission_train_val.csv от модели target timestamp < TEST_START и калибровки только по OOF val; submission_full.csv после переобучения всех выбранных баз на всей размеченной истории и калибратора на всех доступных OOF val+бывший test. Правило/параметры не выбираются повторно по test. Также сохраняются submission_train_val_raw.csv и submission_full_raw.csv. Первый сабмит является диагностическим исключением из обычного full-refit режима по прямому запросу пользователя. На конкурсном cutoff обоим доступны свежие причинные признаки всей известной истории.

Все базовые и метамодели, статистики, выбор, параметры, OOF, коэффициенты каждой строки test/future, отчёты и четыре CSV сабмитов сохраняются в /kaggle/working/cluster_experiment_…. В самом конце формируется ZIP со всеми файлами. Checkpoint проверяет hash и границу fit; запуск в той же working-папке может возобновить базовые модели. При сбое файлы до сбоя остаются в working, финальный ZIP появляется только при полном завершении. Независимая оценка local test получена до полного переобучения. GPU-обучение при подготовке не запускалось.
'''.splitlines(True)
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), mode, 'exec')
    name = 'xgboost_oof_smoothed_dual.ipynb' if mode == 'scalar' else 'xgboost_oof_meta_dual.ipynb'
    output = BASE / 'notebooks/clustering' / name
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1))
    print(output)
