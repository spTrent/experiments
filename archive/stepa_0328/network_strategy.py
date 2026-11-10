"""Итерация 7: MAE-поправка и независимый прогноз общего объёма сети."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import residual_ewma as residual
from network_volume import fit_predict

NAME = 'residual_ewma7_plus_causal_network_volume_depth4'


def fit(frame, root):
    base = residual.fit(frame,root)
    network = fit_predict(root)
    # Общий множитель сетевого прогноза — только по ранним 14 датам.
    totals = frame.groupby('cutoff').y.sum()
    net = network['predictions'].set_index('timestamp').network_prediction
    factor = float(totals.sum()/net.reindex(totals.index).sum())
    # Вес выбирается по ошибке общего объёма на tuning; holdout labels сюда не передаются.
    initial = residual.unshifted(base,frame,root)+base['initial_offset']
    baseline = pd.Series(initial,index=frame.index).groupby(frame.cutoff).sum().reindex(totals.index).to_numpy()
    desired = net.reindex(totals.index).to_numpy()*factor
    truth = totals.to_numpy()
    def objective(w):
        return float(np.mean(np.abs((1-w)*baseline+w*desired-truth)/truth))
    optimum = minimize_scalar(objective,bounds=(0,1),method='bounded',options={'xatol':1e-9})
    weight = min([0.,.25,.5,.75,1.,float(optimum.x)],key=objective)
    print('NETWORK_FACTOR',factor,'WEIGHT',weight,'TUNING_VOLUME_ERROR',objective(weight),flush=True)
    return {'base':base,'network':network,'network_factor':factor,'weight':weight}


def predict(model, frame, root):
    p = residual.predict(model['base'],frame,root)
    dates = pd.DatetimeIndex(frame.cutoff)
    net = model['network']['predictions'].set_index('timestamp').network_prediction
    sums = pd.Series(p).groupby(frame.cutoff.reset_index(drop=True)).sum()
    desired = net.reindex(sums.index)*model['network_factor']
    if desired.isna().any():
        raise ValueError('Отсутствует сетевой прогноз')
    ratio = (1-model['weight'])+model['weight']*desired/sums
    return p*pd.Series(dates).map(ratio).to_numpy()

