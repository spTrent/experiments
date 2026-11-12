import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


GROUP_KEYS = ['cutoff', 'horizon']


def remove_ridge(frame, column):
    columns = ['session6', column]
    values = frame[columns].astype('float64')
    if not np.isfinite(values.to_numpy()).all() or values.lt(0).any().any():
        raise ValueError('Прогнозы должны быть конечными и неотрицательными')
    totals = values.groupby([frame[key] for key in GROUP_KEYS]).transform('sum')
    if (totals[column].eq(0) & totals.session6.gt(0)).any():
        raise ValueError('Нельзя восстановить объём для группы с нулевым прогнозом')
    factors = totals.session6.div(totals[column].where(totals[column].ne(0), 1))
    return values[column] * factors


def score(y, prediction):
    actual = y.to_numpy(dtype=np.float64)
    error = prediction.to_numpy(dtype=np.float64) - actual
    total = actual.sum()
    if not np.isfinite(actual).all() or total <= 0:
        raise ValueError('Некорректные цели локального test')
    wape = np.abs(error).sum() / total
    bias = error.sum() / total
    return dict(WAPE=wape, relative_bias=bias, score=wape + abs(bias))


def write_csv(frame, path):
    pending = path.with_suffix('.partial')
    frame.to_csv(pending, index=False)
    pending.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_val_dir', nargs='?', type=Path)
    args = parser.parse_args()
    folder = args.train_val_dir
    if folder is None:
        candidates = [p.parent for p in Path('output').glob('blend_submit_*/train_val/future_predictions.parquet')]
        if len(candidates) != 1:
            raise ValueError(f'Укажи точную папку train_val. Найдены: {candidates}')
        folder = candidates[0]
    frozen = json.loads((folder / 'frozen_selection.json').read_text())
    if frozen['variant'] != 'cat_d4__ridge':
        raise ValueError('Ожидается сохранённый бленд cat_d4__ridge')
    test = pd.read_parquet(folder / 'test_predictions.parquet', columns=GROUP_KEYS + ['session6', 'blend_prediction', 'y'])
    metrics = pd.DataFrame([
        dict(variant='cat_d4__ridge', **score(test.y, test.blend_prediction)),
        dict(variant='cat_d4__fixed', **score(test.y, remove_ridge(test, 'blend_prediction'))),
    ])
    future = pd.read_parquet(folder / 'future_predictions.parquet', columns=GROUP_KEYS + ['route_id', 'timestamp', 'session6', 'y_pred'])
    request_path = Path('blend_inputs/session6/test_solo_track.parquet')
    requests = pd.read_parquet(request_path, columns=['id', 'route_id', 'timestamp'])
    future['without_ridge'] = remove_ridge(future, 'y_pred')
    joined = requests.merge(future, on=['route_id', 'timestamp'], how='left', sort=False, validate='one_to_one')
    original = pd.read_csv(folder / 'submission.csv', dtype={'id': str})
    if (len(future) != len(requests) or len(joined) != 8000 or not joined.id.is_unique or
            not joined.id.astype(str).equals(original.id) or
            not np.allclose(joined.y_pred, original.y_pred, rtol=1e-12, atol=1e-6) or
            not np.isfinite(joined.without_ridge).all()):
        raise ValueError('Исходный CSV и сохранённые конкурсные прогнозы не совпадают')
    submission = joined[['id', 'without_ridge']].rename(columns={'without_ridge': 'y_pred'})
    destination = folder.parent.parent / 'submission_train_val_without_ridge.csv'
    write_csv(metrics, folder / 'ridge_comparison_test_metrics.csv')
    write_csv(submission, destination)
    print(metrics.to_string(index=False))
    print(destination)


if __name__ == '__main__':
    main()
