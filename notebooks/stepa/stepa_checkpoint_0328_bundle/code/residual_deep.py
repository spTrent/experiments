"""Итерация 8: MAE-поправка и экспоненциальное обновление bias, полупериод 7 дней."""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

NAME = 'residual_catboost_depth5_900_ewma7_bias'
WINDOW = 7
ANCHOR_COLUMNS = ['target_back_0','target_back_1','target_back_2','target_back_4',
 'target_1h_mean_4','target_1h_mean_12','level24_feature','target_1h_lag_48','target_1h_lag_96',
 'target_vs_mean_6h','morning_level_ratio','warehouse_processing_rate','target_acceleration']
ANCHOR_COLUMNS += [f'status_{j}_{v}' for j in range(1,7) for v in ['latest','mean_4','mean_12']]


def features(frame, root):
    anchor = pd.read_parquet(root/'inputs/anchor_features.parquet')
    d = frame[['cutoff','route_id','horizon','raw_blend','model_prediction','profile28']].merge(
        anchor, on=['cutoff','route_id'], how='left', validate='many_to_one', sort=False)
    if d[ANCHOR_COLUMNS].isna().any().any():
        raise ValueError('Отсутствуют причинные anchor-признаки')
    d['dayofweek'] = d.cutoff.dt.dayofweek
    d['route_id'] = d.route_id.astype(str)
    d['horizon'] = d.horizon.astype(str)
    for col in ['model_prediction','profile28','target_back_0','target_1h_mean_4',
                'target_1h_mean_12','level24_feature','target_1h_lag_48','target_1h_lag_96']:
        d[f'{col}_minus_base'] = d[col] - d.raw_blend
    return d.drop(columns='cutoff')


def fit(frame, root):
    X = features(frame, root)
    residual = (frame.y.to_numpy() - frame.raw_blend.to_numpy()) / 100000.
    model = CatBoostRegressor(iterations=900, depth=5, learning_rate=.03,
        loss_function='MAE', l2_leaf_reg=50, random_seed=42, thread_count=4,
        cat_features=['route_id','horizon'], allow_writing_files=False, verbose=150)
    model.fit(X, residual)
    initial = frame.raw_blend.to_numpy() + model.predict(X)*100000.
    offset = float(np.mean(frame.y.to_numpy()-initial))
    return {'model':model, 'initial_offset':offset, 'tuning':frame.copy(),
            'fit_through':str(frame.timestamp.max())}


def unshifted(model, frame, root):
    return frame.raw_blend.to_numpy() + model['model'].predict(features(frame, root))*100000.


def predict(model, frame, root):
    p = unshifted(model, frame, root)
    all_known = pd.concat([model['tuning'],frame.loc[frame.get('split',pd.Series('',index=frame.index)).eq('holdout')]],ignore_index=True)
    all_known = all_known.drop_duplicates(['cutoff','horizon','route_id'])
    oldp = unshifted(model, all_known, root)
    residual = all_known.y.to_numpy()-oldp
    out = np.empty(len(frame)); audits=[]
    for day in np.sort(frame.cutoff.unique()):
        cutoff = pd.Timestamp(day)
        history = (all_known.timestamp < cutoff-pd.Timedelta(hours=4)).to_numpy()
        if all_known.loc[history,'cutoff'].nunique() >= 3:
            age = (cutoff-all_known.loc[history,'cutoff']).dt.total_seconds().to_numpy()/86400
            weights = np.exp2(-age/7.)
            offset = float(np.average(residual[history],weights=weights))
            through = all_known.loc[history,'timestamp'].max()
        else:
            offset = model['initial_offset'];through = pd.Timestamp(model['fit_through'])
        if cutoff >= pd.Timestamp('2025-10-18'):
            assert through < cutoff-pd.Timedelta(hours=4)
        mask = frame.cutoff.eq(cutoff).to_numpy()
        out[mask] = np.maximum(p[mask]+offset,0)
        audits.append({'cutoff':str(cutoff),'bias_fit_through':str(through),'offset':offset})
    model['rolling_audit'] = audits
    print('ROLLING_BIAS',audits[-14:],flush=True)
    return out

