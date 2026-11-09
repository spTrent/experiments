from pathlib import Path
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import time
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from catboost import CatBoostRegressor
from catboost.utils import get_gpu_device_count
from IPython.display import display

DATA = Path('/kaggle/input/datasets/stepzak/rwb-ds')
RUN_ROOT = Path('/kaggle/working/baseline_00')
RESTORE_DIR = None
GPU_DEVICES = '0'
THREAD_COUNT = 4
RUN_FULL_REFIT = True
WEEKLY_BLOCKS = 6
ITERATIONS = 1500
TARGET_SCALE = 100_000
RECENT_TRAIN_FRACTION = 0.75
TRAIN_FILE = 'train_solo_track.parquet'
TEST_FILE = 'test_solo_track.parquet'
TARGET = 'target_1h'
STATUSES = [f'status_{i}' for i in range(1, 7)]
STEP = pd.Timedelta(minutes=30)
LAG_MAX_LOOKBACK_HOURS = 24
LAG_MIN_PAIRS = 336
LAG_MIN_CORRELATION = 0.05
PIPELINE_HASH = '__PIPELINE_HASH__'

if not (DATA / TRAIN_FILE).is_file():
    matches = [p.parent for p in Path('/kaggle/input').rglob(TRAIN_FILE)
               if (p.parent / TEST_FILE).is_file()]
    if len(matches) != 1:
        raise ValueError(f'Укажите DATA вручную: найдено {matches}')
    DATA = matches[0]
if get_gpu_device_count() < 1:
    raise RuntimeError('Включите GPU в настройках Kaggle. CPU fallback отключён.')
print('GPU available:', get_gpu_device_count(), 'devices:', GPU_DEVICES)
if WEEKLY_BLOCKS != 6:
    raise ValueError('Этот протокол требует шесть недель: warmup / tuning / holdout по две недели')

if RESTORE_DIR is not None:
    import shutil
    source = Path(RESTORE_DIR)
    if RUN_ROOT.exists():
        raise ValueError('RESTORE_DIR применяется только к отсутствующей RUN_ROOT')
    if not (source / 'manifest.json').is_file():
        raise ValueError('RESTORE_DIR должен содержать manifest.json и папки результатов')
    shutil.copytree(source, RUN_ROOT)
for name in ['models', 'predictions', 'reports', 'lags', 'checkpoints']:
    (RUN_ROOT / name).mkdir(parents=True, exist_ok=True)

def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()

def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))
    temporary.replace(path)

manifest = dict(
    pipeline_hash=PIPELINE_HASH,
    input_hashes={name: file_hash(DATA / name) for name in [TRAIN_FILE, TEST_FILE]},
    versions={name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'pyarrow', 'catboost']},
    python=platform.python_version(), devices=GPU_DEVICES, thread_count=THREAD_COUNT,
    weekly_blocks=WEEKLY_BLOCKS, iterations=ITERATIONS, target_scale=TARGET_SCALE,
    recent_fraction=RECENT_TRAIN_FRACTION,
    model=dict(depth=7, learning_rate=0.05, loss_function='MAE', random_seed=42),
    lag=dict(max_hours=LAG_MAX_LOOKBACK_HOURS, min_pairs=LAG_MIN_PAIRS, min_corr=LAG_MIN_CORRELATION),
)
manifest_path = RUN_ROOT / 'manifest.json'
if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
    raise ValueError('Данные, код, параметры или версии изменились. Используйте новую RUN_ROOT.')
save_json(manifest_path, manifest)
RUN_ID = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
print('run_id:', RUN_ID, 'output:', RUN_ROOT)
