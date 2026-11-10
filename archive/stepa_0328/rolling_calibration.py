"""Итерация 4: калибровка по предыдущим семи дням с доступными метками.

Каждый дневной origin — отдельный фолд. Tuning score диагностический.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

NAME = 'rolling7_per_horizon_shift_route_shrink025'
WINDOW = 7
STRENGTH = .25


def fit_static(frame):
    residual = frame.y - frame.raw_blend
    shifts = residual.groupby(frame.horizon).mean().to_dict()
    p = np.maximum(frame.raw_blend + frame.horizon.map(shifts), 0).to_numpy()
    totals = pd.DataFrame({'route': frame.route_id.to_numpy(), 'y': frame.y.to_numpy(), 'p': p}).groupby('route').sum()
    ratio = (totals.y / totals.p).clip(.5, 2.)
    factors = (1 + STRENGTH * (ratio - 1)).clip(.8, 1.2).to_dict()
    p = p * frame.route_id.map(factors).to_numpy()
    y = frame.y.to_numpy(dtype=float)
    def objective(k):
        e = p*k-y
        return (np.abs(e).sum()+abs(e.sum()))/y.sum()
    opt = minimize_scalar(objective, bounds=(.8, 1.2), method='bounded', options={'xatol':1e-10})
    k = min([float(opt.x), float(y.sum()/p.sum())], key=objective)
    return {'shifts': shifts, 'factors': factors, 'multiplier': k,
            'fit_through': str(frame.timestamp.max())}


def apply_static(model, frame):
    p = np.maximum(frame.raw_blend.to_numpy() + frame.horizon.map(model['shifts']).to_numpy(), 0)
    return p * frame.route_id.map(model['factors']).fillna(1).to_numpy() * model['multiplier']


def fit(frame, root):
    return {'initial':fit_static(frame), 'tuning':frame.copy()}


def predict(model, frame, root):
    history = pd.concat([model['tuning'], frame.loc[frame.get('split', pd.Series('',index=frame.index)).eq('holdout')]], ignore_index=True)
    history = history.drop_duplicates(['cutoff','horizon','route_id'])
    output = np.empty(len(frame))
    audits = []
    for day in np.sort(frame.cutoff.unique()):
        cutoff = pd.Timestamp(day)
        available = history.loc[(history.timestamp < cutoff-pd.Timedelta(hours=4)) &
                                 (history.cutoff >= cutoff-pd.Timedelta(days=WINDOW))]
        current = frame.cutoff.eq(cutoff).to_numpy()
        if available.cutoff.nunique() >= 3:
            calibrator = fit_static(available)
            assert pd.Timestamp(calibrator['fit_through']) < cutoff-pd.Timedelta(hours=4)
        else:
            calibrator = model['initial']
        if cutoff >= pd.Timestamp('2025-10-18'):
            assert pd.Timestamp(calibrator['fit_through']) < cutoff-pd.Timedelta(hours=4)
        output[current] = apply_static(calibrator, frame.loc[current])
        audits.append({'cutoff':str(cutoff),'fit_through':calibrator['fit_through'],
                      'days':available.cutoff.nunique(),'multiplier':calibrator['multiplier']})
    print('CAUSAL_ROLLING_AUDIT', audits[-14:], flush=True)
    model['rolling_audit'] = audits
    return output

