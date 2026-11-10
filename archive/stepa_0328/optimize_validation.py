"""Хронологическая калибровка сохранённых OOS-прогнозов и журнал 10 итераций.

Запускать на сервере из /root/experiments: .venv/bin/python stepa/optimize_validation.py.
Стратегия находится в active_strategy.py. Все её fit получают только tuning.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import shutil
import subprocess
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def metric(frame, prediction):
    y = frame.y.to_numpy(dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    if y.shape != p.shape or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('Некорректные прогнозы')
    if y.sum() <= 0:
        raise ValueError('Нулевой знаменатель')
    error = p - y
    wape = float(np.abs(error).sum() / y.sum())
    bias = float(error.sum() / y.sum())
    daily = pd.DataFrame({'day': frame.cutoff.to_numpy(), 'y': y,
                          'ae': np.abs(error), 'error': error}).groupby('day').sum()
    return {'WAPE': wape, 'Rbias': bias, 'score': wape + abs(bias),
            'daily_score': float(((daily.ae + daily.error.abs()) / daily.y).mean()),
            'rows': len(y), 'days': len(daily)}


def write_json(path, value):
    temp = path.with_suffix('.partial.json')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


def validate(frame):
    if frame.duplicated(['cutoff', 'horizon', 'route_id']).any():
        raise ValueError('Повторы ключей')
    if not np.isfinite(frame[['y', 'raw_blend', 'model_prediction', 'profile28']]).all().all():
        raise ValueError('NaN/inf в данных')
    if not (frame.groupby(['cutoff', 'horizon']).size() == 1000).all():
        raise ValueError('Неполная панель')
    if frame.horizon.nunique() != 8 or frame.cutoff.nunique() != 28:
        raise ValueError('Ожидаются 28 дней и 8 горизонтов')
    if not (frame.timestamp == frame.cutoff + pd.to_timedelta(frame.horizon * 30, unit='min')).all():
        raise ValueError('Нарушен горизонт')


def main():
    started = time.monotonic()
    state_path = ROOT.parent / 'optimization_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'iterations': 0, 'best_score': .330, 'history': []}
    if state['iterations'] >= 10:
        raise RuntimeError('Лимит 10 запусков исчерпан')
    i = state['iterations'] + 1
    result = ROOT / 'results' / f'iteration_{i:02}'
    result.mkdir(parents=True, exist_ok=False)
    state['iterations'] = i
    write_json(state_path, state)
    source = ROOT / 'active_strategy.py'
    shutil.copy2(source, result / 'strategy.py')
    spec = importlib.util.spec_from_file_location('strategy', source)
    strategy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(strategy)
    frame = pd.read_parquet(ROOT / 'inputs/selected_evaluation.parquet').sort_values(['cutoff', 'horizon', 'route_id']).reset_index(drop=True)
    validate(frame)
    tuning = frame.loc[frame.split.eq('tuning')].copy()
    holdout = frame.loc[frame.split.eq('holdout')].copy()
    if tuning.timestamp.max() + pd.Timedelta(hours=4) >= holdout.cutoff.min():
        raise ValueError('Недостаточный purge для калибровки')
    print(f'ITERATION {i}/10: {strategy.NAME}', flush=True)
    print('BASELINE_SELECTED', metric(holdout, holdout.prediction), flush=True)
    model = strategy.fit(tuning, ROOT)
    prediction = np.maximum(np.asarray(strategy.predict(model, frame, ROOT), dtype=np.float64), 0)
    reports = []
    for split, part in [('tuning', tuning), ('holdout', holdout)]:
        mask = frame.index.isin(part.index)
        p = prediction[mask]
        reports.append({'split': split, 'horizon': 0, **metric(part, p)})
        for h in range(1, 9):
            hmask = part.horizon.eq(h).to_numpy()
            reports.append({'split': split, 'horizon': h, **metric(part.loc[hmask], p[hmask])})
        if split == 'holdout':
            for week, dates in enumerate(np.array_split(np.sort(part.cutoff.unique()), 2), 1):
                wm = part.cutoff.isin(dates).to_numpy()
                reports.append({'split': f'holdout_week{week}', 'horizon': 0, **metric(part.loc[wm], p[wm])})
    report = pd.DataFrame(reports)
    report.to_csv(result / 'metrics.csv', index=False)
    frame['calibrated_prediction'] = prediction
    frame.to_parquet(result / 'predictions.parquet', index=False)
    import joblib
    joblib.dump(model, result / 'calibrator.joblib')
    score = float(report.loc[(report.split == 'holdout') & (report.horizon == 0), 'score'].iloc[0])
    info = {'iteration': i, 'name': strategy.NAME, 'holdout_score': score,
            'elapsed_seconds': time.monotonic() - started,
            'strategy_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'calibration_fit_first': str(tuning.cutoff.min()), 'calibration_fit_last_label': str(tuning.timestamp.max()),
            'holdout_first': str(holdout.cutoff.min()), 'holdout_last': str(holdout.cutoff.max()),
            'note': 'Повторно просматриваемый локальный holdout; внешний score неизвестен.'}
    write_json(result / 'report.json', info)
    write_json(ROOT / 'latest_result.json', info)
    # Обязательный checkpoint выполняется немедленно после фиксации нового score.
    if score < state['best_score']:
        destination = f'stepa_checkpoint_{score:.12f}'
        print(f'NEW_BEST {score:.12f}; cp -r stepa/ {destination}', flush=True)
        subprocess.run(['cp', '-r', 'stepa/', destination], cwd=ROOT.parent, check=True)
        state['best_score'] = score
        state['best_checkpoint'] = destination
        state['best_iteration'] = i
    state['history'].append(info)
    write_json(state_path, state)
    print(report.loc[report.horizon.eq(0)].to_string(index=False), flush=True)
    print('BEST', state['best_score'], 'ITERATIONS', i, flush=True)


if __name__ == '__main__':
    main()

