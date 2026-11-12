import ast
import json
from pathlib import Path


root = Path(__file__).resolve().parents[1]
template = json.loads((root / 'best/raw-vs-clean.ipynb').read_text())
cells = []


def cell(kind, source):
    item = dict(cell_type=kind, id=f'session6-clusters-{len(cells)}', metadata={}, source=source.splitlines(True))
    if kind == 'code':
        compile(source, item['id'], 'exec')
        item.update(execution_count=None, outputs=[])
    cells.append(item)


def function(source, name):
    node = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name)
    return ast.unparse(node)


cell('markdown', '''# Session6. Кластеры для LightGBM

Контроль 0.326 на закрытом тесте использует две части. CatBoost с поправками задаёт общий объём горизонта, LightGBM уточняет его распределение по маршрутам. Здесь проверяем кластеризацию только для второй части. Признаки, параметры каждого горизонта, целевая переменная и веса берутся из Session6. Пять обучающих стартов в день, с 09:30 до 11:30, как в исходном решении. LightGBM обучается на CPU, как в исходном тренере.

В Kaggle подключите архив `best/cleaning_comparison_inputs.zip` как приватный input и выполните Run All. Если файлы уже распакованы, они должны лежать рядом с `comparison_meta.json`. Для локального запуска из репозитория используется тот же архив. Требуются NumPy, pandas, PyArrow, scikit-learn, joblib и LightGBM.

Обе ветки обучаются на сырой истории с целями строго до 4 октября. KMeans также видит только эту историю. Сравниваем общую модель и модели по 8 кластерам, небольшие кластеры используют общую модель. Срезы val и test включают прогнозы от 10:30 за 4–17 и 18–31 октября соответственно. Параметры и число кластеров фиксированы заранее, по test ничего не подбирается.

В основной таблице обе формы приведены к сумме новой общей LightGBM по каждому cutoff и горизонту. Поэтому разница WAPE показывает эффект распределения при одинаковом bias. Дополнительная таблица для test подставляет обе формы в архивный Session6 с его прежним весом и объёмом. Она сравнивает две заново обученные ветки в одном протоколе, а архивные 0.319190 служат отдельным ориентиром. Архивная ветка объёма использовала причинные поправки по доступным прошлым фактам. Локальный test уже изучался. Этот диагностический опыт сохраняет метрики и прогнозы и не собирает новый сабмит.
''')
cell('code', '''from pathlib import Path
from tempfile import TemporaryDirectory
import gc
import json
import zipfile

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

work = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()
search = Path('/kaggle/input') if Path('/kaggle/input').exists() else Path.cwd() / 'best'
RUN = work / 'session6_lightgbm_clusters'
RUN.mkdir(exist_ok=True)
fit_end = pd.Timestamp('2025-10-04')
test_start = pd.Timestamp('2025-10-18')
test_end = pd.Timestamp('2025-11-01')
STEP = pd.Timedelta(minutes=30)
CLUSTERS = (1, 8)
MIN_CLUSTER_ROUTES = 10
SEED = 42
TARGET = 'target_1h'
names = [TARGET, *[f'status_{i}' for i in range(1, 7)]]
keys = ['cutoff', 'horizon', 'route_id']
group_keys = ['cutoff', 'horizon']
required = ['comparison_meta.json', 'train_solo_track.parquet', 'validation_predictions.parquet', 'evaluation_prediction_values.parquet']
temporary = TemporaryDirectory()
folders = [p.parent for p in search.rglob('comparison_meta.json') if all((p.parent / f).is_file() for f in required)]
if len(folders) == 1:
    data = folders[0]
else:
    archives = list(search.rglob('cleaning_comparison_inputs.zip'))
    if len(archives) != 1:
        raise ValueError('Подключите один архив cleaning_comparison_inputs.zip')
    data = Path(temporary.name)
    with zipfile.ZipFile(archives[0]) as archive:
        for filename in required:
            (data / filename).write_bytes(archive.read(filename))
meta = json.loads((data / 'comparison_meta.json').read_text())
history = pd.read_parquet(data / 'train_solo_track.parquet', columns=['timestamp', 'route_id', *names])
history = history.sort_values(['timestamp', 'route_id'])
times = pd.DatetimeIndex(history.timestamp.unique())
routes = np.sort(history.route_id.unique())
if len(history) != len(times) * len(routes) or history.duplicated(['timestamp', 'route_id']).any() or history.isna().any().any():
    raise ValueError('Неполная панель')
if not times.equals(pd.date_range(times[0], times[-1], freq='30min')):
    raise ValueError('Неверная сетка времени')
values = {name: history[name].to_numpy(dtype=np.float32).reshape(len(times), len(routes)) for name in names}
del history
target = values[TARGET]
TIMES, ROUTES, VALUES = times, routes, values
''')
source = lambda index: ''.join(template['cells'][index]['source'])
runtime = (root / 'scripts/cluster_experiment/runtime.py').read_text()
cell('code', '\n\n'.join([
    function(source(5), 'rolling'), function(source(8), 'extend_robust'), function(source(9), 'features'),
    function(source(11), 'metric'), function(runtime, 'segment_routes'),
]))
cell('code', '''means = {(name, steps): rolling(array, steps) for name, array in values.items() for steps in (4, 12, 48)}
level24 = means[TARGET, 48]
long_week = {name: rolling(values[name], 336) for name in names[1:]}
deviations = {w: pd.DataFrame(target).rolling(w, min_periods=2).std().to_numpy(dtype=np.float32) for w in (4, 12)}
with threadpool_limits(limits=4):
    segmentation = segment_routes(fit_end, 'before_val')
route_cluster = pd.Series(segmentation['labels'][8], index=routes)
positions = np.arange(len(times))
minutes = times.hour * 60 + times.minute
train_starts = np.isin(minutes, [570, 600, 630, 660, 690]) & (positions >= 336)
query_starts = (times >= fit_end) & (times < test_end) & (minutes == 630)
fit_reports = []
predictions = []

for h in range(1, 9):
    train_origins = positions[train_starts & (times + h * STEP < fit_end)]
    query_origins = positions[query_starts & (positions + h < len(times))]
    train = features(train_origins, h, target, level24)
    query = features(query_origins, h, target, level24)
    config = meta[str(h)]
    columns = config['columns']
    if set(columns) & {'y', 'cutoff', 'timestamp'}:
        raise ValueError('Недопустимые колонки LightGBM')
    train_scale = train.scale.to_numpy(dtype=float)
    query_scale = query.scale.to_numpy(dtype=float)
    train_group = train.route_id.map(route_cluster).to_numpy()
    query_group = query.route_id.map(route_cluster).to_numpy()
    query_period = np.where(query.cutoff < test_start, 'val', 'test')
    params = dict(config['params'], n_estimators=config['trees'], n_jobs=4)
    global_pred = np.empty(len(query))
    clustered_pred = np.empty(len(query))
    groups = ['global', *sorted(set(train_group) - {-1})]
    for group in groups:
        train_mask = np.ones(len(train), dtype=bool) if group == 'global' else train_group == group
        query_mask = np.ones(len(query), dtype=bool) if group == 'global' else query_group == group
        model = lgb.LGBMRegressor(**params)
        model.fit(train.loc[train_mask, columns], train.y.to_numpy()[train_mask] / train_scale[train_mask],
                  sample_weight=train_scale[train_mask] / train_scale.mean())
        model.booster_.save_model(str(RUN / f'lightgbm_h{h}_{group}.txt'))
        train_pred = np.maximum(model.predict(train.loc[train_mask, columns]) * train_scale[train_mask], 0)
        query_pred = np.maximum(model.predict(query.loc[query_mask, columns]) * query_scale[query_mask], 0)
        if group == 'global':
            global_pred[:] = query_pred
            clustered_pred[:] = query_pred
        else:
            clustered_pred[query_mask] = query_pred
        rows = [dict(h=h, group=group, period='train', **metric(train.y.to_numpy()[train_mask], train_pred))]
        for period in ('val', 'test'):
            mask = query_period[query_mask] == period
            rows.append(dict(h=h, group=group, period=period, **metric(query.y.to_numpy()[query_mask][mask], query_pred[mask])))
        fit_reports.extend(rows)
        display(pd.DataFrame(rows))
        pd.DataFrame(fit_reports).to_csv(RUN / 'metrics_after_fit.csv', index=False)
        del model, train_pred, query_pred
    predictions.append(query[keys + ['timestamp', 'y']].assign(period=query_period, cluster=query_group,
        global_shape=global_pred, clustered_shape=clustered_pred))
    del train, query
    gc.collect()
''')
cell('code', '''def normalize_shape(frame, shape, anchor):
    amounts = frame[group_keys + [shape, anchor]].copy()
    amounts[[shape, anchor]] = amounts[[shape, anchor]].astype(float)
    shape_total = amounts.groupby(group_keys, sort=False)[shape].transform('sum').to_numpy()
    anchor_total = amounts.groupby(group_keys, sort=False)[anchor].transform('sum').to_numpy()
    result = amounts[anchor].to_numpy().copy()
    valid = shape_total > 0
    result[valid] = amounts[shape].to_numpy()[valid] * anchor_total[valid] / shape_total[valid]
    totals = amounts.assign(prediction=result).groupby(group_keys, sort=False)[['prediction', anchor]].sum()
    if not np.allclose(totals.prediction, totals[anchor], rtol=1e-10, atol=1e-5):
        raise ValueError('Не сохранён общий объём')
    return result


frame = pd.concat(predictions, ignore_index=True)
if frame.duplicated(keys).any() or len(frame) != 28 * 8 * len(routes):
    raise ValueError('Неполная панель прогнозов')
frame['clustered_at_global_volume'] = normalize_shape(frame, 'clustered_shape', 'global_shape')
frame.to_parquet(RUN / 'shape_predictions.parquet', index=False)
rows = []
for period, part in frame.groupby('period'):
    for column in ('global_shape', 'clustered_shape', 'clustered_at_global_volume'):
        rows.append(dict(protocol='shape', period=period, variant=column, **metric(part.y, part[column])))

archive_test = pd.read_parquet(data / 'validation_predictions.parquet')
volume = pd.read_parquet(data / 'evaluation_prediction_values.parquet')[keys + ['calibrated_prediction']]
test = frame.loc[frame.period.eq('test')].merge(archive_test[keys + ['y', 'session_prediction']], on=keys,
    validate='one_to_one', how='left', suffixes=('', '_archive')).merge(volume, on=keys, validate='one_to_one', how='left')
if not np.array_equal(test.y, test.y_archive) or test.calibrated_prediction.isna().any():
    raise ValueError('Архивный test не совпадает с оценкой')
weight = float(meta['weight'])
for shape in ('global_shape', 'clustered_shape'):
    test[f'session_{shape}'] = (1 - weight) * test.calibrated_prediction + weight * normalize_shape(test, shape, 'calibrated_prediction')
session_columns = ['session_prediction', 'session_global_shape', 'session_clustered_shape']
for column in session_columns:
    rows.append(dict(protocol='archived_volume', period='test', variant=column, **metric(test.y, test[column])))
    sums = test.groupby(group_keys)[[column, 'calibrated_prediction']].sum()
    if not np.allclose(sums[column], sums.calibrated_prediction, rtol=1e-10, atol=1e-5):
        raise ValueError('Объём Session6 изменился')
scores = pd.DataFrame(rows)
display(scores)
scores.to_csv(RUN / 'scores.csv', index=False)
test.to_parquet(RUN / 'session6_volume_test_predictions.parquet', index=False)
for key in ('cutoff', 'horizon', 'cluster'):
    details = [dict(**{key: value}, variant=column, **metric(part.y, part[column]))
               for value, part in test.groupby(key) for column in session_columns]
    pd.DataFrame(details).to_csv(RUN / f'test_by_{key}.csv', index=False)
(RUN / 'protocol.json').write_text(json.dumps(dict(fit_end_exclusive=str(fit_end), test_start=str(test_start),
    test_end_exclusive=str(test_end), clusters=list(CLUSTERS), min_cluster_routes=MIN_CLUSTER_ROUTES,
    shape_weight=weight, test_used_for_selection=False, evaluation_before_full_refit=True, full_refit=False,
    parameters=meta), ensure_ascii=False, indent=2))
with zipfile.ZipFile(RUN.with_suffix('.zip'), 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(RUN.iterdir()):
        if path.is_file():
            archive.write(path, path.name)
print('Локальный test оценён до полного переобучения. Период ранее изучался.')
print('Архив результатов', RUN.with_suffix('.zip'))
''')

output = root / 'notebooks/clustering/session6_lightgbm_clusters.ipynb'
output.write_text(json.dumps(dict(nbformat=4, nbformat_minor=5, metadata=template['metadata'], cells=cells), ensure_ascii=False, indent=1))
print(output)
