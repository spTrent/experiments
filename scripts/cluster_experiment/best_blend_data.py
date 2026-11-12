from pathlib import Path
import gc
import hashlib
import json
import zipfile

import numpy as np
import pandas as pd
from IPython.display import display


HORIZONS = tuple(range(1, 9))
STEP = pd.Timedelta(minutes=30)
OOF_START = pd.Timestamp('2025-09-20')
LOCAL_TEST_START = pd.Timestamp('2025-10-18')
LOCAL_TEST_END = pd.Timestamp('2025-11-01')
OOF_WINDOWS = list(zip(pd.date_range(OOF_START, periods=4, freq='7D'),
                       pd.date_range(OOF_START + pd.Timedelta(days=7), periods=4, freq='7D')))
KEYS = ['cutoff', 'horizon', 'route_id']
GROUP_KEYS = KEYS[:2]
BASES = ['session6', 'rf', 'cat_rich', 'xgb_rich', 'xgb_clean', 'profile28']
TARGET = 'target_1h'
STATUSES = [f'status_{i}' for i in range(1, 7)]
SEED = 42
THREADS = 4


SESSION_FILES = ('comparison_meta.json', 'train_solo_track.parquet', 'test_solo_track.parquet',
                 'validation_predictions.parquet')
XGB_STAGES = ('oof_week_0', 'oof_week_1', 'oof_week_2', 'calibration_select', 'train_val')
XGB_FILES = ('manifest.json', 'protocol.json', 'evaluation_complete.json',
             *(f'{stage}_h{h}_xgb_k8.parquet' for stage in XGB_STAGES for h in HORIZONS))


def find_dataset(label, path, marker, required):
    if path is None:
        root = Path('/kaggle/input')
        candidates = sorted({file.parent for file in root.rglob(marker)})
        complete = [folder for folder in candidates if all((folder / name).is_file() for name in required)]
        if len(complete) == 1:
            return complete[0]
        if not complete and len(candidates) == 1:
            path = candidates[0]
        else:
            raise FileNotFoundError(f'{label}: нужен один полный набор. Укажите путь к папке в ячейке настроек. '
                                    f'Найденные полные наборы: {complete}; папки с {marker}: {candidates}')
    path = Path(path)
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f'{label}: в {path} отсутствуют файлы {missing}')
    return path


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


def save_parquet(path, data):
    temp = path.with_suffix('.partial')
    data.to_parquet(temp, index=False)
    temp.replace(path)


def initialize(session_path, xgb_path):
    global ARTIFACTS, TIMES, ROUTES, VALUES, SESSION_META, SESSION_DATA, XGB_DATA
    SESSION_DATA = find_dataset('Session6', session_path, 'comparison_meta.json', SESSION_FILES)
    XGB_DATA = find_dataset('XGBoost rich k8', xgb_path, 'oof_week_0_h1_xgb_k8.parquet', XGB_FILES)
    train_path = SESSION_DATA / 'train_solo_track.parquet'
    SESSION_META = json.loads((SESSION_DATA / 'comparison_meta.json').read_text())
    xgb_manifest = json.loads((XGB_DATA / 'manifest.json').read_text())
    XGB_PROTOCOL = json.loads((XGB_DATA / 'protocol.json').read_text())
    xgb_evaluation = json.loads((XGB_DATA / 'evaluation_complete.json').read_text())
    contract = dict(code=NOTEBOOK_HASH, seed=SEED,
                    session_sha256={name: digest_file(SESSION_DATA / name) for name in SESSION_FILES},
                    xgb_sha256={name: digest_file(XGB_DATA / name) for name in XGB_FILES})
    if (xgb_manifest['train_hash'] != contract['session_sha256']['train_solo_track.parquet'] or
            xgb_manifest['test_hash'] != contract['session_sha256']['test_solo_track.parquet']):
        raise ValueError('Датасеты XGBoost rich и Session6 используют разные входные файлы')
    if (pd.Timestamp(XGB_PROTOCOL['val_start']) != OOF_START or
            pd.Timestamp(XGB_PROTOCOL['calibration_split']) != OOF_WINDOWS[-1][0] or
            pd.Timestamp(XGB_PROTOCOL['test_start']) != LOCAL_TEST_START or
            pd.Timestamp(xgb_evaluation['train_val_model_target_end_exclusive']) != LOCAL_TEST_START or
            XGB_PROTOCOL['model_clusters'] != [8] or XGB_PROTOCOL['test_used_for_selection'] or
            not xgb_evaluation['test_evaluated_before_full_refit']):
        raise ValueError('Границы OOF или протокол архива XGB rich отличаются')
    run_id = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()[:12]
    ARTIFACTS = Path('/kaggle/working') / f'best_blend_{run_id}'
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    write_json(ARTIFACTS / 'inputs.json', dict(contract, session_path=str(SESSION_DATA), xgb_path=str(XGB_DATA)))
    print('Session6:', SESSION_DATA)
    print('XGBoost rich k8:', XGB_DATA)
    history = pd.read_parquet(train_path, columns=['timestamp', 'route_id', TARGET, *STATUSES])
    history = history.sort_values(['timestamp', 'route_id'])
    TIMES = pd.DatetimeIndex(history.timestamp.unique())
    ROUTES = np.sort(history.route_id.unique())
    if (len(history) != len(TIMES) * len(ROUTES) or history.isna().any().any() or
            history.duplicated(['timestamp', 'route_id']).any() or
            not TIMES.equals(pd.date_range(TIMES[0], TIMES[-1], freq=STEP))):
        raise ValueError('Требуется полная получасовая панель')
    VALUES = {name: history[name].to_numpy(dtype=np.float32).reshape(len(TIMES), len(ROUTES))
              for name in (TARGET, *STATUSES)}
    if not all(np.isfinite(array).all() for array in VALUES.values()) or np.any(VALUES[TARGET] < 0):
        raise ValueError('Некорректные наблюдения исходной панели')
    display(pd.DataFrame([dict(split=f'oof_{i}', start=start, end_exclusive=stop)
                          for i, (start, stop) in enumerate(OOF_WINDOWS, 1)] +
                         [dict(split='test', start=LOCAL_TEST_START, end_exclusive=LOCAL_TEST_END)]))


def metric(y, prediction):
    actual = np.asarray(y, dtype=np.float64).ravel()
    predicted = np.asarray(prediction, dtype=np.float64).ravel()
    volume = actual.sum()
    if actual.shape != predicted.shape or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError('Некорректный вход метрики')
    if volume == 0:
        return dict(rows=len(actual), WAPE=np.nan, relative_bias=np.nan, score=np.nan)
    error = predicted - actual
    wape = np.abs(error).sum() / volume
    bias = error.sum() / volume
    return dict(rows=len(actual), WAPE=wape, relative_bias=bias, score=wape + abs(bias))


def fit_report(model, boundary, horizon, group, split, train_y, train_pred, query_y, query_pred):
    report = pd.DataFrame([
        dict(model=model, fit_before=boundary, horizon=horizon, group=group, period=period, **metric(y, pred))
        for period, y, pred in [('train', train_y, train_pred), (split, query_y, query_pred)]
    ])
    display(report)
    path = ARTIFACTS / f'fits_{model}_{boundary:%Y%m%d}_h{horizon}_{group}.csv'
    report.to_csv(path, index=False)


def read_xgb(stage):
    if stage not in XGB_STAGES:
        raise ValueError(f'Неизвестный этап XGBoost rich: {stage}')
    frames = [pd.read_parquet(XGB_DATA / f'{stage}_h{h}_xgb_k8.parquet',
                              columns=['cutoff', 'timestamp', 'route_id', 'h', 'y', 'y_pred']) for h in HORIZONS]
    result = pd.concat(frames, ignore_index=True).rename(columns={'h': 'horizon', 'y_pred': 'xgb_rich'})
    if not result.timestamp.eq(result.cutoff + result.horizon * STEP).all():
        raise ValueError('Неверный timestamp в XGB rich')
    return result


def read_session6_test():
    frame = pd.read_parquet(SESSION_DATA / 'validation_predictions.parquet',
                            columns=KEYS + ['y', 'session_prediction'])
    return frame.rename(columns={'session_prediction': 'session6'})


def context(start, stop):
    origins = np.flatnonzero((TIMES >= start) & (TIMES < stop) & (TIMES.hour == 10) & (TIMES.minute == 30))
    expected_days = (stop - start).days
    if len(origins) != expected_days or origins[-1] + max(HORIZONS) >= len(TIMES):
        raise ValueError('Не хватает дней или целевых горизонтов')
    y = VALUES[TARGET]
    windows = np.arange(-47, 1)
    level24 = np.stack([y[pos + windows].mean(axis=0, dtype=np.float64) for pos in origins])
    seasonal_origins = origins[:, None] - 48 * np.arange(1, 29)
    if seasonal_origins.min() + windows.min() < 0:
        raise ValueError('Не хватает наблюдений для profile28')
    seasonal_level = np.stack([y[pos + windows].mean(axis=0, dtype=np.float64)
                               for pos in seasonal_origins.ravel()]).reshape(*seasonal_origins.shape, len(ROUTES))
    ratio = level24[:, None, :] / np.maximum(seasonal_level, 1)
    route_count = len(ROUTES)
    common = dict(cutoff=TIMES[origins].repeat(route_count), route_id=np.tile(ROUTES, len(origins)),
                  level24=level24.ravel(), current=y[origins].ravel(),
                  delta1h=(y[origins] - y[origins - 2]).ravel(),
                  delta2h=(y[origins] - y[origins - 4]).ravel(),
                  network_current=np.repeat(y[origins].sum(axis=1, dtype=np.float64), route_count))
    return pd.concat([pd.DataFrame(dict(common, horizon=h, y=y[origins + h].ravel(),
                                        profile28=(y[seasonal_origins + h] * ratio).mean(axis=1).ravel()))
                      for h in HORIZONS], ignore_index=True).sort_values(KEYS).reset_index(drop=True)


def join_base(frame, incoming, column):
    relevant = incoming[KEYS + ['y', column]]
    merged = frame.merge(relevant, on=KEYS, how='left', validate='one_to_one', suffixes=('', '_incoming'))
    if len(relevant) != len(frame) or not np.array_equal(merged.y, merged.y_incoming):
        raise ValueError(f'Не совпали ключи или факты для {column}')
    if not np.isfinite(merged[column]).all() or merged[column].lt(0).any():
        raise ValueError(f'Некорректные прогнозы {column}')
    return merged.drop(columns='y_incoming')


def assemble(start, stop, session, rf, cat, xgb, xgb_clean):
    frame = context(start, stop)
    for source, name in [(session, 'session6'), (rf, 'rf'), (cat, 'cat_rich'),
                         (xgb, 'xgb_rich'), (xgb_clean, 'xgb_clean')]:
        part = source.loc[source.cutoff.ge(start) & source.cutoff.lt(stop)]
        frame = join_base(frame, part, name)
    return frame


def archive_report():
    path = ARTIFACTS.with_suffix('.zip')
    temp = path.with_suffix('.partial')
    with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=2) as archive:
        for file in sorted(ARTIFACTS.rglob('*')):
            if file.is_file() and not file.name.endswith('.partial'):
                archive.write(file, file.relative_to(ARTIFACTS.parent))
    temp.replace(path)
    print('Архив результатов', path)
