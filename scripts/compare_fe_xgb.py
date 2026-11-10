import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ['cutoff', 'route_id', 'horizon']


def daily_metrics(frame, pred):
    error = np.asarray(pred) - frame.y.to_numpy(dtype=float)
    daily = pd.DataFrame({
        'day': frame.cutoff.to_numpy(), 'y': frame.y.to_numpy(dtype=float),
        'absolute': np.abs(error), 'signed': error,
    }).groupby('day').sum()
    if (daily.y <= 0).any():
        raise ValueError('Нулевой дневной объём')
    return {
        'WAPE': daily.absolute.sum() / daily.y.sum(),
        'signed_bias': daily.signed.sum() / daily.y.sum(),
        'daily_score': ((daily.absolute + daily.signed.abs()) / daily.y).mean(),
    }


def compare(fe_file, xgb_file, fe_column, legacy_start, output):
    fe = pd.read_parquet(fe_file)
    xgb = pd.read_parquet(xgb_file)
    if legacy_start:
        start = pd.Timestamp(legacy_start).normalize()
        fe = fe.loc[fe.cutoff.ge(start) & fe.cutoff.lt(start + pd.Timedelta(days=14))]
    elif 'model_info_through' not in fe:
        raise ValueError('Для старого файла укажите --legacy-test-start с первой датой ранее отложенного test')
    days = pd.DatetimeIndex(fe.cutoff.unique()).sort_values()
    if len(days) != 14 or not days.equals(pd.date_range(days[0], periods=14, freq='D')) or (days[0].hour,days[0].minute)!=(10,30):
        raise ValueError('Нужны две последовательные недели стартов 10:30')
    xgb = xgb.loc[xgb.cutoff.isin(days)]
    for frame in (fe, xgb):
        if 'model_info_through' in frame and not (frame.model_info_through < frame.cutoff).all():
            raise ValueError('Прогноз использует информацию после старта')
        expected = len(days)*8*frame.route_id.nunique()
        if len(frame)!=expected or set(frame.horizon)!=set(range(1,9)) or frame.duplicated(KEY).any():
            raise ValueError('Неполная панель прогнозов')
    merged = fe[KEY+['y',fe_column]].rename(columns={fe_column:'fe'}).merge(
        xgb[KEY+['y','xgboost']],on=KEY, how='outer',validate='one_to_one',indicator=True)
    if not merged._merge.eq('both').all() or not np.array_equal(merged.y_x,merged.y_y):
        raise ValueError('Не совпадают ключи или метки FE и XGBoost')
    merged = merged.rename(columns={'y_x':'y'})
    if not np.isfinite(merged[['fe','xgboost','y']]).all().all():
        raise ValueError('Неконечные прогнозы или метки')
    train = merged.loc[merged.cutoff < days[7]]
    val = merged.loc[merged.cutoff >= days[7]]
    def evaluate(part, weight):
        return daily_metrics(part,(1-weight)*part.fe + weight*part.xgboost)
    weights = [0.,.05,.10,.20,.30]
    weight = min(weights,key=lambda w:evaluate(train,w)['daily_score'])
    rows = []
    for name,part in [('train',train),('val',val)]:
        for label,w in [('FE',0.),('XGBoost',1.),('смесь',weight)]:
            rows.append(dict(period=name,first=part.cutoff.min(),last=part.cutoff.max(),
                             model=label,xgb_weight=w,**evaluate(part,w)))
    report = pd.DataFrame(rows)
    report['gain_vs_FE'] = report.groupby('period').daily_score.transform('first')-report.daily_score
    output.parent.mkdir(parents=True,exist_ok=True)
    report.to_csv(output,index=False)
    print(report.to_string(index=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Вес XGBoost выбирается по первой неделе и переносится на вторую')
    parser.add_argument('fe_file',type=Path)
    parser.add_argument('--xgb',type=Path,default=Path('artifacts/local_stack/base_predictions.parquet'))
    parser.add_argument('--fe-column',default='fe_prediction')
    parser.add_argument('--legacy-test-start',help='Первая дата 14-дневного test старого FE с настройками, зафиксированными до этой даты')
    parser.add_argument('--output',type=Path,default=Path('artifacts/fe_xgb_blend.csv'))
    args = parser.parse_args()
    compare(args.fe_file,args.xgb,args.fe_column,args.legacy_test_start,args.output)
