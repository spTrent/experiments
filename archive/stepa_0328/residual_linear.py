"""Итерация 9: регуляризованная линейная MAE-поправка; fit только tuning."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import residual_ewma as residual

NAME = 'residual_smooth_mae_ridge001_ewma7'


def features(frame,root):
    x=residual.features(frame,root).drop(columns=['route_id','horizon'])
    for h in range(1,9):x[f'horizon_{h}']=frame.horizon.eq(h).to_numpy(dtype=float)
    return x


def fit(frame,root):
    x=features(frame,root)
    scaler=StandardScaler()
    X=scaler.fit_transform(x)
    y=(frame.y.to_numpy()-frame.raw_blend.to_numpy())/100000.
    X=np.column_stack([X,np.ones(len(X))])
    alpha=.01;epsilon=.02
    def objective(beta):
        e=X@beta-y
        smooth=np.sqrt(e*e+epsilon**2)
        penalty=alpha*np.dot(beta[:-1],beta[:-1])
        gradient=X.T@(e/smooth)/len(e)
        gradient[:-1]+=2*alpha*beta[:-1]
        return float(smooth.mean()+penalty),gradient
    result=minimize(objective,np.zeros(X.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':400,'ftol':1e-10})
    print('LINEAR',result.message,'ITER',result.nit,'LOSS',result.fun,flush=True)
    raw=frame.raw_blend.to_numpy()+X@result.x*100000.
    return {'scaler':scaler,'coef':result.x,'columns':x.columns.tolist(),
            'initial_offset':float(np.mean(frame.y.to_numpy()-raw)),
            'tuning':frame.copy(),'fit_through':str(frame.timestamp.max())}


def unshifted(model,frame,root):
    X=model['scaler'].transform(features(frame,root)[model['columns']])
    return frame.raw_blend.to_numpy()+(X@model['coef'][:-1]+model['coef'][-1])*100000.


def predict(model,frame,root):
    p=unshifted(model,frame,root)
    history=pd.concat([model['tuning'],frame.loc[frame.get('split',pd.Series('',index=frame.index)).eq('holdout')]],ignore_index=True)
    history=history.drop_duplicates(['cutoff','horizon','route_id'])
    oldp=unshifted(model,history,root)
    errors=history.y.to_numpy()-oldp
    out=np.empty(len(frame));audits=[]
    for day in np.sort(frame.cutoff.unique()):
        cutoff=pd.Timestamp(day)
        mask=(history.timestamp<cutoff-pd.Timedelta(hours=4)).to_numpy()
        if history.loc[mask,'cutoff'].nunique()>=3:
            age=(cutoff-history.loc[mask,'cutoff']).dt.total_seconds().to_numpy()/86400
            offset=float(np.average(errors[mask],weights=np.exp2(-age/7)))
            through=history.loc[mask,'timestamp'].max()
        else:
            offset=model['initial_offset'];through=pd.Timestamp(model['fit_through'])
        if cutoff>=pd.Timestamp('2025-10-18'):assert through<cutoff-pd.Timedelta(hours=4)
        current=frame.cutoff.eq(cutoff).to_numpy()
        out[current]=np.maximum(p[current]+offset,0)
        audits.append({'cutoff':str(cutoff),'fit_through':str(through),'offset':offset})
    model['rolling_audit']=audits
    return out

