import argparse
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('submission', type=Path)
    parser.add_argument('--session6', required=True, type=Path)
    parser.add_argument('--requests', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    keys = pd.read_parquet(args.requests, columns=['id', 'timestamp'])
    keys['id'] = keys.id.astype(str)
    blend = pd.read_csv(args.submission, dtype={'id': str})
    if args.session6.suffix == '.zip':
        with ZipFile(args.session6) as archive:
            with archive.open('Session6_exact/submission_session6.csv') as stream:
                session6 = pd.read_csv(stream, dtype={'id': str})
    else:
        session6 = pd.read_csv(args.session6, dtype={'id': str})
    for forecasts in [blend, session6]:
        if (len(forecasts) != len(keys) or list(forecasts) != ['id', 'y_pred'] or
                not np.isfinite(forecasts.y_pred).all() or forecasts.y_pred.lt(0).any()):
            raise ValueError('Некорректный исходный CSV')
    frame = keys.merge(blend, on='id', how='left', sort=False, validate='one_to_one')
    frame = frame.merge(session6.rename(columns={'y_pred': 'session6'}),
                        on='id', how='left', sort=False, validate='one_to_one')
    groups = frame.groupby('timestamp')
    if len(frame) != 8000 or groups.ngroups != 8 or not groups.size().eq(1000).all():
        raise ValueError('Ожидалось восемь горизонтов по 1000 строк')
    totals = groups[['y_pred', 'session6']].transform('sum')
    if frame[['y_pred', 'session6']].isna().any().any() or totals.le(0).any().any():
        raise ValueError('Пропущенные ID или нулевой объём горизонта')
    submission = frame[['id']].assign(y_pred=frame.y_pred / totals.y_pred * totals.session6)
    path = args.output or args.submission.with_name(f'{args.submission.stem}_exact_session6_volume.csv')
    if path.exists():
        raise FileExistsError(path)
    pending = path.with_suffix('.partial')
    submission.to_csv(pending, index=False)
    checked = pd.read_csv(pending, dtype={'id': str})
    if not checked.id.equals(keys.id) or not np.isfinite(checked.y_pred).all():
        raise ValueError('Некорректный итоговый CSV')
    if not np.allclose(checked.y_pred / totals.session6, frame.y_pred / totals.y_pred, rtol=1e-12, atol=1e-15):
        raise ValueError('Изменились доли маршрутов')
    audit = frame.assign(adjusted=checked.y_pred).groupby('timestamp')[['y_pred', 'session6', 'adjusted']].sum()
    if not np.allclose(audit.adjusted, audit.session6, rtol=1e-12, atol=1e-6):
        raise ValueError('Объёмы по горизонтам отличаются от CSV с опорными объёмами')
    pending.replace(path)
    print(path)
    print(audit.rename(columns={'y_pred': 'before', 'session6': 'anchor'}).to_string())


if __name__ == '__main__':
    main()
