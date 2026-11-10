import ast
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
notebook = json.loads((BASE / 'notebooks/clustering/catboost_rich_calibration.ipynb').read_text())
source = (BASE / 'scripts/cluster_experiment/train_val_submission.py').read_text()
config = ''.join(notebook['cells'][1]['source'])
for node in reversed(ast.parse(config).body):
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CODE_HASH' for t in node.targets):
        config = config.replace(ast.get_source_segment(config, node), '')
config += '\nSUBMISSION_MULTIPLIER = 1.05\nCODE_HASH = ' + repr(hashlib.sha256((config + source + ''.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')).encode()).hexdigest()) + '\n'
notebook['cells'][1]['source'] = config.splitlines(True)
notebook['cells'][-1]['source'] = 'STEP = pd.Timedelta(minutes=30)\nrun_train_val_submission()\n'.splitlines(True)
notebook['cells'].insert(-1, dict(cell_type='code', id='train-val-submission', metadata={}, source=source.splitlines(True), execution_count=None, outputs=[]))
notebook['cells'][0]['source'] = '''# CatBoost rich: сабмит от модели train + val

Диагностический эксперимент по явному запросу пользователя: исключить локальный test из обучения. Подключите case_rwb, включите GPU T4×2 и Run All. Внешние Outputs или ZIP не нужны. Обучение запускает пользователь в Kaggle.

Те же параметры CatBoost rich, 216 числовых признаков плюс категориальный маршрут, 8 кластеров, восемь горизонтов, MAE, все получасовые старты обучения. Никакого повторного подбора параметров/калибровки. Фиксированный множитель1.05 взят из предыдущего выбора по val.

История и конкурсные запросы загружаются полностью. Локальные val/test по14дней определяются той же процедурой, что в catboost_rich_calibration. Для исходных данных TEST_START — 18октября2025 00:00. Модели обучаются только на примерах target timestamp < TEST_START. Масштабы, KMeans и корреляционные лаги тоже оцениваются до этой границы. Локальный test не используется при fit и early stopping. На последнем доступном cutoff для конкурсных дат используются все известные на этот момент значения истории, включая local test, исключительно как причинные признаки.

Для каждого горизонта после обучения печатаются train/test WAPE, Relative Bias и WAPE + |Relative Bias|, raw и ×1.05. Затем сводная таблица по всем горизонтам. Эти оценки не используются для выбора модели или множителя. Test уже изучался в проекте: это не новый слепой holdout.

Сохраняются submission_raw.csv и submission.csv (=raw×1.05), восемь моделей train_val_model_h*.joblib, обученная сегментация, прогнозы локального test, метрики, аудит границ и ZIP в /kaggle/working/cluster_experiment_…. Checkpoint имеет отдельный hash и проверяет границу обучения, поэтому модели полного refit не переиспользуются. При повторном Run All в том же working-каталоге можно возобновить готовые горизонты.

Полного переобучения на train + val + local test здесь намеренно нет по запросу пользователя. Оценочная и конкурсная модели — одни и те же обученные веса; отличаются доступные на момент прогноза признаки. Это контроль влияния полного refit, а не замена стандартного full-history пайплайна. В соревнование файлы автоматически не отправляются.
'''.splitlines(True)
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        compile(''.join(cell['source']), 'train_val_submission', 'exec')
output = BASE / 'notebooks/clustering/catboost_rich_train_val_submission.ipynb'
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1))
print(output)
