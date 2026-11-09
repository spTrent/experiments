from pathlib import Path
from tempfile import TemporaryDirectory
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

START = pd.Timestamp('2025-09-20')
TEST_START = pd.Timestamp('2025-10-18')
DAY = pd.Timedelta(days=1)
STEP = pd.Timedelta(minutes=30)
TARGET = 'target_1h'
STATUSES = [f'status_{i}' for i in range(1, 7)]
WINDOW_OFFSETS = tuple(range(-4, 5))
LAG_MAX_LOOKBACK_HOURS = 24
LAG_MIN_PAIRS = 336
LAG_MIN_CORRELATION = 0.05
TRAIN_MINUTES = (570, 600, 630, 660, 690)
REGIMES = {'five_starts':('five_starts',), 'recent75':('recent_all',)}[PROFILE]
STACK_WEEKS = (2, 3, 4)
VALIDATION_WEEKS = STACK_WEEKS[1:]
FAMILIES = ('rf', 'light_gbm', 'catboost', 'xgb', 'ridge')
MODEL_NAMES = [f'{f}_{v}' for f in FAMILIES for v in ('direct', 'residual')]
KEY = ['cutoff', 'route_id', 'horizon']

TARGET_SCALE = 100000
ETA_CONTEXT = None
THREADS = min(4, os.cpu_count() or 1)

def file_digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def locate(name):
    hits = list(INPUT.rglob(name))
    if len(hits) != 1:
        raise ValueError(f'Ожидается один {name}, найдено {hits}')
    return hits[0]

def initialize():
    global INPUT, OUTPUT, ROOT, TRAIN_PATH, TEST_PATH, RUN_ID, CHECKPOINTS
    INPUT = Path(INPUT_DIR)
    TRAIN_PATH = locate('train_solo_track.parquet')
    TEST_PATH = locate('test_solo_track.parquet')
    fingerprints = [file_digest(TRAIN_PATH),file_digest(TEST_PATH)]
    identity = [CODE_ID,PROFILE,str(ROUTE_ALPHA),*fingerprints]
    if not np.isfinite(ROUTE_ALPHA) or ROUTE_ALPHA<=0:
        raise ValueError('ROUTE_ALPHA должен быть положительным')
    RUN_ID = hashlib.sha256('|'.join(identity).encode()).hexdigest()[:24]
    OUTPUT = Path(OUTPUT_DIR)
    ROOT = OUTPUT / f'stack_{RUN_ID}'
    ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS = [p for p in INPUT.rglob(ROOT.name) if p.is_dir() and p.resolve() != ROOT.resolve()]
    print(f'Конфигурация {RUN_ID}, часть {PART}, режим {MODE}, роль {ROLE}')

def profile_path(rel):
    return str(rel)

def existing(rel):
    rel = profile_path(rel)
    dest = ROOT / rel
    if dest.is_file():
        return dest
    hits = [base / rel for base in CHECKPOINTS if (base / rel).is_file()]
    if not hits:
        return None
    if len(hits) > 1 and len({file_digest(p) for p in hits}) != 1:
        raise ValueError(f'Несогласованные копии {rel}: {hits}')
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix('.copying')
    shutil.copyfile(hits[0], temp)
    temp.replace(dest)
    return dest

def require(rel):
    path = existing(rel)
    if path is None:
        raise FileNotFoundError(f'Подключите результат {rel} из запуска {RUN_ID}')
    return path

def save(rel, value):
    path = ROOT / profile_path(rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.partial')
    if isinstance(value, pd.DataFrame):
        value.to_parquet(temp, index=False)
    else:
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)

def load_json(rel):
    return json.loads(require(rel).read_text())

def train_indices(h, boundary, regime):
    possible = np.flatnonzero(times[:len(times)-h] + h * STEP < boundary)
    days = times[possible].normalize()
    unique = days.unique()
    early = days.isin(unique[:len(unique)//4])
    five = np.isin(times[possible].hour * 60 + times[possible].minute, TRAIN_MINUTES)
    masks = {'five_starts': five, 'mixed_starts': ~early | five, 'recent_all': ~early}
    return possible[masks[regime]]

def query_indices(boundary, days):
    if days == 0:
        return np.array([len(times)-1])
    grid = pd.date_range(boundary + pd.Timedelta(hours=10, minutes=30), periods=days, freq='D')
    result = times.get_indexer(grid)
    if (result < 0).any():
        raise ValueError(f'Нет стартов для {grid}')
    return result

def metric(y, p):
    y = np.asarray(y, dtype=np.float64)
    error = np.asarray(p, dtype=np.float64) - y
    if not len(y) or not np.isfinite(error).all():
        raise ValueError('Пустая выборка или нечисловой прогноз')
    volume = float(y.sum())
    return totals(len(y), volume, float(np.abs(error).sum()), float(error.sum()))

def totals(n, volume, absolute, signed):
    score = (absolute+abs(signed))/volume if volume else None
    return dict(rows=int(n), volume=volume, absolute=absolute, signed=signed,
                WAPE=absolute/volume if volume else None, relative_bias=signed/volume if volume else None,
                score=score, **{'WAPE + |Relative Bias|':score})

def combine(rows):
    return totals(*(sum(row[k] for row in rows) for k in ('rows','volume','absolute','signed')))

def task_path(h, stage, regime, name):
    return f'h{h}/{stage}/{regime}/{name}.parquet'

def complete(rel):
    return existing(rel + '.json') is not None and existing(rel) is not None

def result_frame(h, origins):
    frame = pd.DataFrame(dict(cutoff=times[origins].repeat(len(routes)), route_id=np.tile(routes, len(origins))))
    frame['horizon'] = h
    frame['timestamp'] = frame.cutoff + h * STEP
    if origins[-1] + h < len(times):
        frame['y'] = target[origins+h].ravel()
    return frame

def finish_fit(rel, frame, pred, train_row, boundary, fit_through, seconds, device):
    if not np.isfinite(pred).all() or np.min(pred) < 0:
        raise ValueError(f'Некорректный прогноз {rel}')
    rows = {'train':train_row}
    if 'y' in frame:
        rows['test' if '/test/' in rel else 'held_out'] = metric(frame.y, pred)
    save(rel, frame.assign(prediction=pred))
    save(rel+'.json',dict(run_id=RUN_ID, boundary=str(boundary), fit_through=str(fit_through),
                          seconds=seconds, backend=device, metrics=rows))
    display(pd.DataFrame(rows).T.assign(task=rel))
    show_eta()

def show_eta():
    if not ETA_CONTEXT:
        return
    known = ETA_CONTEXT['completed']
    oof = [v for (h,s),v in known.items() if s.startswith('oof')]
    if not oof:
        return
    current = ETA_CONTEXT['current']
    remaining = 0.
    for h,stage in ETA_CONTEXT['pending']:
        if (h,stage) in known:
            continue
        similar = [v for (other,s),v in known.items() if s==stage]
        if not similar:
            similar = [v for (other,s),v in known.items() if s[:3]==stage[:3]]
        estimate = np.mean(similar or oof)
        if (h,stage)==current:
            estimate = max(0.,estimate-(time.monotonic()-ETA_CONTEXT['start']))
        remaining += estimate
    remaining += ETA_CONTEXT['future_features']*ETA_CONTEXT['features']
    print(f'ETA части по измеренным этапам: {remaining/3600:.2f} ч')

def stage_spec(stage):
    if stage.startswith('oof'):
        return START + (int(stage[-1])-1)*7*DAY, 7
    if stage == 'test':
        return TEST_START, 14
    if stage == 'extra':
        return TEST_START + 7*DAY, 7
    if stage == 'full':
        return times[-1] + STEP, 0
    raise ValueError(stage)

def device_setup():
    global xp, cp, cupyx, BACKEND, LGB_DEVICE
    try:
        import cupy as cp
        import cupyx
        from cuml.ensemble import RandomForestRegressor
        device_name = cp.cuda.runtime.getDeviceProperties(0)['name']
    except (ImportError,RuntimeError) as error:
        raise RuntimeError('Для расчёта баз включите NVIDIA GPU и подключите CuPy и cuML, совместимые с CUDA сессии.') from error
    xp, BACKEND = cp, 'cuda'
    cp.linalg.solve(cp.eye(2),cp.ones(2))
    LGB_DEVICE = 'cpu'
    for candidate in ['cuda','gpu']:
        code = f"""import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
r = np.tile(np.arange({len(routes)}),32)
rng = np.random.default_rng(42)
data = pd.DataFrame(dict(route_id=pd.Categorical(r),volume=rng.random(len(r))))
model = LGBMRegressor(device_type='{candidate}',n_estimators=2,max_cat_to_onehot=1,verbosity=-1,n_jobs={THREADS})
model.fit(data,r%17+data.volume,categorical_feature=['route_id'])
"""
        probe = subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        if probe.returncode == 0:
            LGB_DEVICE = candidate
            break
        print(candidate, (probe.stderr or probe.stdout)[-1200:])
    print('GPU:',device_name,'LightGBM:',LGB_DEVICE)
    from importlib.metadata import version
    save(f'part{PART}/{MODE}_devices.json',dict(gpu=str(device_name),lightgbm_device=LGB_DEVICE,
         packages={k:version(k) for k in ('numpy','pandas','xgboost','lightgbm','catboost','scikit-learn')},
         cupy=cp.__version__,cuml=__import__('cuml').__version__))
