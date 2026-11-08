from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]
features = (ROOT / 'scripts/five_starts_features.py').read_text()
models = (ROOT / 'scripts/five_starts_models.py').read_text()
setup = '''from pathlib import Path
from datetime import datetime, timezone
import json
import gc
import os
import importlib.util
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from IPython.display import display
ALGORITHM = {algorithm!r}
RESIDUAL = {residual}
DEVICE = 'cuda' if ALGORITHM in ('xgb', 'catboost') else 'cpu'
N_TRIALS = 12
N_JOBS = min(4, os.cpu_count() or 1)
SEED = 42
MAX_TREES = 2500
EARLY_STOPPING_ROUNDS = 100
PREDICTION_BATCH = 4096
RUN_LABEL = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
required_modules = ['sklearn', 'optuna', 'joblib', 'psutil']
required_modules += [{{'xgb': 'xgboost', 'catboost': 'catboost', 'lightgbm': 'lightgbm', 'prophet': 'prophet'}}.get(ALGORITHM, 'sklearn')]
missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
if missing:
    raise ImportError(f'Установите недостающие пакеты до расчёта признаков: {{sorted(set(missing))}}')
if N_TRIALS < 1 or PREDICTION_BATCH < 1:
    raise ValueError('Число trials и размер блока должны быть положительными')
DATA = Path('data')
TRAIN_FILE = 'train_solo_track.parquet'
TEST_FILE = 'test_solo_track.parquet'
TRAIN_PATH, TEST_PATH = DATA / TRAIN_FILE, DATA / TEST_FILE
if not (TRAIN_PATH.is_file() and TEST_PATH.is_file()) and Path('/kaggle/input').exists():
    train_matches = list(Path('/kaggle/input').rglob(TRAIN_FILE))
    test_matches = list(Path('/kaggle/input').rglob(TEST_FILE))
    if len(train_matches) != 1 or len(test_matches) != 1:
        raise ValueError('Укажите TRAIN_PATH и TEST_PATH вручную')
    TRAIN_PATH, TEST_PATH = train_matches[0], test_matches[0]
BASE = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()
MODEL_NAME = {experiment!r}
RUN_DIR = BASE / 'artifacts' / f'{{MODEL_NAME}}_stack'
if RUN_DIR.exists():
    raise FileExistsError(f'Папка {{RUN_DIR}} уже существует. Перенесите предыдущий результат или измените MODEL_NAME.')
OUT = RUN_DIR / 'features'
MODEL_DIR = RUN_DIR / 'models'
IMPORTANCE_DIR = RUN_DIR / 'importance'
POSTPROCESS_DIR = RUN_DIR / 'postprocessing'
DOCS_DIR = RUN_DIR / 'docs'
PARAMS_DIR = RUN_DIR / 'model_params'
for directory in (OUT, MODEL_DIR, IMPORTANCE_DIR, POSTPROCESS_DIR, DOCS_DIR, PARAMS_DIR):
    directory.mkdir(parents=True)
TARGET = 'target_1h'
STATUSES = [f'status_{{i}}' for i in range(1, 7)]
STEP = pd.Timedelta(minutes=30)
TARGET_SCALE = 100000
TRAIN_MINUTES = (570, 600, 630, 660, 690)
WINDOW_OFFSETS = tuple(range(-4, 5))
LAG_MAX_LOOKBACK_HOURS = 24
LAG_MIN_PAIRS = 336
LAG_MIN_CORRELATION = 0.05
FEATURE_PIPELINE_VERSION = 'five_starts_windows_v4_stage_safe_hpo'
'''
feature_blocks = [
    ('Данные и временное разбиение', 'history = pd.read_parquet(TRAIN_PATH'),
    ('Общие признаки пяти стартов', 'import pyarrow as pa'),
    ('Корреляционные лаги, обучаемые внутри train', 'def calculate_dynamic_shoulder'),
    ('Матрицы восьми горизонтов', 'def horizon_dataset'),
    ('Метрика соревнования', 'def mean_daily_score')
]
model_blocks = [
    ('Модели, поиск и сохранение', 'import importlib.metadata'),
    ('Подбор до test', "if ALGORITHM == 'prophet':\n    full_rows"),
    ('Независимая оценка', 'evaluation_lags = fit_lags'),
    ('Переобучение на всей истории и submission', 'full_lags = fit_lags')
]
for algorithm in ('xgb', 'lightgbm', 'random_forest', 'ridge', 'prophet', 'catboost'):
    for residual in (False, True):
        variant = 'residual' if residual else 'direct'
        name = f'{algorithm}_{variant}'
        intro = f'''# {algorithm} {variant}: пять стартов, подбор и test

Восемь общих моделей по горизонтам, все маршруты вместе. Признаки исходных XGBoost сохранены. Train: 09:30, 10:00, 10:30, 11:00, 11:30. Val, tuning, test и submission: 10:30. Direct предсказывает y / 100000, residual — (y − profile28) / 100000 с восстановлением уровня перед отсечением отрицательных прогнозов.

Optuna использует три недельных окна с 30 августа по 19 сентября 2025 года. Все метки train, val и tuning строго раньше 20 сентября. Параметры можно использовать для временных OOF со старта 20 сентября. Локальный test занимает 18–31 октября. XGBoost, LightGBM и CatBoost выбирают число деревьев на предшествующем val. Optuna выбирает параметры отдельно для каждого горизонта по среднему WAPE + |Relative Bias| трёх tuning-окон. Test не читается в objective и не используется для выбора параметров, признаков или постобработки. После фиксации всех настроек модель обучается на всей истории до test; затем оценивается на test. Перед submission каждая модель переобучается на всей доступной истории с пригодными метками и profile28, как в исходных ноутбуках. Первые строки без доступного profile28 исключены в обеих ветках.

Во время поиска выводятся train/tuning; test выводится только после фиксации настроек. Независимая оценка test получена до финального переобучения. Этот временной период ранее использовался в проекте, поэтому он не является новым слепым тестом.

Запускайте один ноутбук за раз. N_TRIALS — число конфигураций на горизонт; каждая проверяется на трёх окнах. DEVICE = 'cpu' включает CPU для бустингов. Заполнение пропусков и масштабирование обучаются только на train текущего fit. Справочник маршрутов общий и не использует таргет. Корреляционные колонки в базовых parquet оставлены NaN и заполняются только после выбора границы обучения. Артефакты: artifacts/<алгоритм>_<direct или residual>_stack/{{features, importance, models, postprocessing, docs, model_params}}. Лучшие параметры каждого горизонта: model_params/best_hN.json. Инструкция: docs/model_experiments.md.
'''
        if algorithm == 'lightgbm':
            intro += '\nLightGBM использует нативные категории route_id, пропуски остаются NaN. По умолчанию DEVICE=cpu; gpu требует OpenCL-сборки, cuda требует CUDA-сборки LightGBM. Остановка по MAE на val, subsample_freq=1 включает подбираемое семплирование строк.\n'
        if algorithm in ('random_forest', 'ridge', 'prophet'):
            intro += '\nМаршрут кодируется двоичным представлением позиции в отсортированном справочнике маршрутов плюс один: 10 колонок для 1000 маршрутов. Код 0 зарезервирован для неизвестного маршрута. Справочник не использует таргет и сохраняется в metadata.json; он одинаков для всех этапов. Для Ridge и Prophet это ограничивает независимость эффектов маршрутов.\n'
        if algorithm == 'prophet':
            intro += '\nProphet — одна общая модель на горизонт, не отдельные модели маршрутов. Время ds относится к целевому слоту; одинаковые даты разных маршрутов сохраняются. Числовые признаки — дополнительные регрессоры, route_id — двоичный код из 10 колонок для 1000 маршрутов. Тренд и сезонность общие для маршрутов, эффекты регрессоров аддитивные. Двоичные признаки в аддитивной модели не дают независимого смещения каждому маршруту; это ограничение выразительности модели. Проверка RAM выполняется до поиска по размеру полного refit; полная матрица может потребовать существенно больше 16 ГБ. GPU здесь не используется.\n'
        cells = [nbformat.v4.new_markdown_cell(intro), nbformat.v4.new_code_cell(setup.format(algorithm=algorithm, residual=residual, experiment=name))]
        for source, blocks in ((features, feature_blocks), (models, model_blocks)):
            offsets = [source.index(marker) for _, marker in blocks] + [len(source)]
            for i, (title, _) in enumerate(blocks):
                cells.extend([nbformat.v4.new_markdown_cell('## ' + title), nbformat.v4.new_code_cell(source[offsets[i]:offsets[i+1]].strip())])
        notebook = nbformat.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}})
        nbformat.write(notebook, ROOT / f'{name}.ipynb')
