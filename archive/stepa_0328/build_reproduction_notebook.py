"""Собрать переносимый ноутбук восстановления кандидата Public 0.328."""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).resolve().parent
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source):
    cells.append(nbf.v4.new_code_cell(source.strip()))


md(r"""
# Воспроизведение лучшего чекпоинта Stepa — Public 0.328

Снимок **итерации 10**, `stepa_checkpoint_0.321453823624`. Внешний score **0.328**
сообщён пользователем 10 ноября 2026. Цель `< 0.322` на платформе ещё не достигнута.

**Run All** в свежем ядре Python 3.14: ноутбук проверит комплект файлов, построит
финальные признаки из исходного train, применит восемь сохранённых CatBoost-моделей,
затем две модели поправки и сформирует `submission_reproduced.csv`.
GPU для восстановления сохранённого чекпоинта **не нужен**.

Рядом должна лежать папка `stepa_checkpoint_0328_bundle/` либо одноимённый ZIP.
В комплект входят исходные train/test, все веса в формате CBM, исходный код,
OOS-кэш и эталонный отправленный CSV. Абсолютных путей к серверу в исполнении нет.
Не устанавливайте зависимости повторно, если открыли существующее ядро проекта.

Локально до финального refit: pooled **0.321453824**, среднее дневное **0.338300893**.
Эти величины не равны внешнему score: bias может компенсироваться между днями.
Здесь воспроизводится сохранённый результат, а не объявляется новый слепой тест.
""")

md("""
## 1. Настройки и переносимый комплект

По умолчанию восстанавливаются исходные финальные модели и валидация по сохранённым
OOS-прогнозам. `REFIT_CALIBRATORS=True` дополнительно переобучит только две модели
поправки с теми же параметрами; этот режим по умолчанию выключен.
""")
code(r"""
from pathlib import Path
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
import zipfile

BASE_DIR = None  # При необходимости: Path('/папка/с/ноутбуком')
REBUILD_BASE_PREDICTIONS = True
VERIFY_VALIDATION = True
REFIT_CALIBRATORS = False

bundle_name = 'stepa_checkpoint_0328_bundle'
search = [Path(BASE_DIR).expanduser()] if BASE_DIR else [Path.cwd(), Path.cwd() / 'stepa']
matches = [p.resolve() for p in search
           if (p / bundle_name / 'bundle_manifest.json').is_file()
           or (p / f'{bundle_name}.zip').is_file()]
matches = list(dict.fromkeys(matches))
if len(matches) != 1:
    raise FileNotFoundError('Укажите BASE_DIR: нужна папка с комплектом ' + bundle_name)
BASE = matches[0]
BUNDLE = BASE / bundle_name
if not (BUNDLE / 'bundle_manifest.json').is_file():
    with zipfile.ZipFile(BASE / f'{bundle_name}.zip') as archive:
        for item in archive.infolist():
            name = Path(item.filename)
            if name.is_absolute() or '..' in name.parts or not name.parts or name.parts[0] != bundle_name:
                raise ValueError('Некорректный путь в архиве')
        archive.extractall(BASE)
OUT = BASE / 'reproduced_0328'
OUT.mkdir(exist_ok=True)
DATA = BUNDLE / 'data'
manifest = json.loads((BUNDLE / 'bundle_manifest.json').read_text())

def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

missing = [name for name in ['numpy', 'pandas', 'pyarrow', 'catboost', 'scipy']
           if importlib.util.find_spec(name) is None]
if missing:
    raise RuntimeError('В ядре отсутствуют ' + ', '.join(missing)
                       + '. Выберите ядро проекта или установите requirements-reproduce.txt из комплекта.')

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from IPython.display import display

for relative, expected in manifest['files'].items():
    path = BUNDLE / relative
    if not path.is_file() or sha256(path) != expected:
        raise ValueError('Файл отсутствует или повреждён: ' + relative)
assert sha256(BUNDLE / 'reference/submission_0328.csv') == manifest['original_submission_sha256']
print('Комплект проверен:', BUNDLE)
print('Чекпоинт:', manifest['checkpoint'], '| Public:', manifest['public_score_reported_by_user'])
original_environment = json.loads((BUNDLE / 'environment.json').read_text())
display(pd.DataFrame([
    {'пакет': p, 'версия сервера': v, 'текущее ядро': importlib.metadata.version(p)}
    for p, v in original_environment['packages'].items()
    if importlib.util.find_spec(p) is not None
]))
""")

md("""
## 2. Рецепт и загрузка моделей поправки

Основа: recent75 CatBoost; control на h1/h2/h3/h4/h7, сезонные признаки на h5/h6/h8.
Смесь: h1 — CatBoost, h2–h8 — 70% CatBoost + 30% profile28.
Далее две MAE-модели остатка: depth=4 / 450 деревьев и depth=5 / 900 деревьев,
фиксированная смесь 50/50. Общий bias обновляется экспоненциально, полупериод 7 дней.

Для оценки модели поправки обучены на 4–17 октября; для submission — переобучены
на OOS-кэше 4–31 октября 2025. Эти веса хранятся отдельно. Восстановление использует
CBM и JSON, не зависит от pickle-представления pandas на сервере.
""")
code(r"""
sys.path.insert(0, str(BUNDLE / 'code'))
import winning_strategy as strategy

history = pd.read_parquet(BUNDLE / 'inputs/selected_evaluation.parquet').sort_values(
    ['cutoff', 'horizon', 'route_id']).reset_index(drop=True)
assert len(history) == 224000 and history.cutoff.nunique() == 28
assert not history.duplicated(['cutoff', 'horizon', 'route_id']).any()
assert (history.groupby(['cutoff', 'horizon']).size() == 1000).all()
tuning = history.loc[history.split.eq('tuning')].copy()
holdout = history.loc[history.split.eq('holdout')].copy()
assert tuning.timestamp.max() + pd.Timedelta(hours=4) < holdout.cutoff.min()

def load_calibrator(phase):
    folder = BUNDLE / 'models' / f'{phase}_calibrator'
    info = json.loads((folder / 'metadata.json').read_text())
    assert info['combination_weights'] == {'shallow': .5, 'deep': .5}
    known = tuning if phase == 'evaluation' else history
    model = {}
    for name, metadata in info['components'].items():
        estimator = CatBoostRegressor()
        estimator.load_model(str(folder / f'{name}.cbm'))
        assert estimator.feature_names_ == metadata['feature_names']
        model[name] = dict(model=estimator, tuning=known.copy(),
                           initial_offset=metadata['initial_offset'],
                           fit_through=metadata['fit_through'])
    return model

evaluation_calibrator = load_calibrator('evaluation')
final_calibrator = load_calibrator('final')
display(pd.DataFrame([
    {'этап': phase, 'модель': name, 'деревья': item['model'].tree_count_,
     'последняя метка fit': item['fit_through']}
    for phase, model in [('оценка', evaluation_calibrator), ('submission', final_calibrator)]
    for name, item in model.items()
]))
""")

md("""
## 3. Признаки из исходных данных и восемь базовых моделей

Восстанавливаем признаки последнего origin **1 ноября 2025, 10:30** и прогнозы
на **11:00–14:30**, 1000 маршрутов × 8 горизонтов. Используются сохранённый финальный
словарь лагов и те же формулы, что при обучении. Исходные parquet не изменяются.
Этот этап выполняет inference сохранённых моделей и не запускает их обучение.
""")
code(r"""
future = pd.read_parquet(BUNDLE / 'inputs/final_predictions.parquet')
keys = pd.read_parquet(BUNDLE / 'inputs/test_keys.parquet')
assert len(future) == 8000 and future.id.is_unique

if REBUILD_BASE_PREDICTIONS:
    for name, expected in manifest['data_sha256'].items():
        if sha256(DATA / name) != expected:
            raise ValueError('Не совпал исходный parquet: ' + name)
    base_work = OUT / 'base_inference'
    base_work.mkdir(exist_ok=True)
    scope = {'__name__': '__baseline_reproduction__'}
    exec(compile((BUNDLE / 'code/baseline_definitions.py').read_text(), 'baseline_definitions.py', 'exec'), scope)
    scope.update(ROOT=base_work, TRAIN_PATH=DATA / 'train_solo_track.parquet',
                 TEST_PATH=DATA / 'test_solo_track.parquet')
    scope['load_panel']()
    for name in ['baseline_common_features.py', 'baseline_correlations.py', 'baseline_horizon.py']:
        exec(compile((BUNDLE / 'code' / name).read_text(), name, 'exec'), scope)
    scope['validation_lag_table'] = pd.read_csv(BUNDLE / 'inputs/status6_lags_final.csv')
    rebuilt = []
    origin = np.array([len(scope['times']) - 1])
    for h in range(1, 9):
        folder = BUNDLE / 'models/baseline' / f'h{h}'
        metadata = json.loads((folder / 'complete.json').read_text())
        features = scope['add_pilot_features'](scope['horizon_dataset'](h, origin), h, origin)
        requests = scope['test'].loc[scope['test'].timestamp.eq(scope['end'] + h * scope['STEP'])]
        features = requests.merge(features, on=['route_id', 'timestamp'], validate='one_to_one', sort=False)
        estimator = CatBoostRegressor()
        estimator.load_model(str(folder / 'model.cbm'))
        assert estimator.feature_names_ == metadata['features']
        part = features[['id', 'route_id', 'timestamp', 'profile28']].copy()
        part['model_prediction'] = np.maximum(estimator.predict(features[metadata['features']]) * 100000., 0)
        rebuilt.append(part)
        print(f'h{h}: {len(part)} маршрутов, {metadata["trees"]} деревьев')
    rebuilt = pd.concat(rebuilt, ignore_index=True)
    checked = future[['id', 'model_prediction', 'profile28']].merge(
        rebuilt[['id', 'model_prediction', 'profile28']], on='id', validate='one_to_one', suffixes=('_saved', '_new'))
    assert len(checked) == 8000
    np.testing.assert_allclose(checked.model_prediction_saved, checked.model_prediction_new, rtol=1e-10, atol=1e-6)
    np.testing.assert_allclose(checked.profile28_saved, checked.profile28_new, rtol=1e-7, atol=.05)
    print('Максимальное отличие базового прогноза:', float(np.abs(checked.model_prediction_saved - checked.model_prediction_new).max()))
    lookup = rebuilt.set_index('id').reindex(future.id)
    future['model_prediction'] = lookup.model_prediction.to_numpy()
    future['profile28'] = lookup.profile28.to_numpy()
    future['raw_blend'] = np.where(future.horizon.eq(1), future.model_prediction,
                                    .7 * future.model_prediction + .3 * future.profile28)
    # Большие промежуточные матрицы больше не нужны.
    del scope, rebuilt, lookup, features, estimator
    import gc
    gc.collect()
else:
    print('Используются сохранённые финальные базовые прогнозы из проверенного комплекта.')
""")

md("""
## 4. Валидация сохранённого OOS-кэша

Оценочные базовые модели обучались по меткам до **4 октября 2025, 00:00**;
early stopping использовал tuning. Модели поправки обучались только на tuning
4–17 октября. Для каждого holdout-origin обновление bias использует исключительно
метки с `timestamp < cutoff − 4 часа`. Предыдущие завершившиеся holdout-дни доступны
следующим origin — это последовательная оценка с обратной связью.

Метрика: `WAPE = sum(abs(p-y))/sum(y)`, `Rbias = sum(p-y)/sum(y)`,
`score = WAPE + abs(Rbias)`. Знаменатель должен быть положительным.
Приводим pooled score, дневную комбинацию и две последовательные недели отдельно.
Holdout уже просматривался при разработке; сохранённый результат не является
независимой оценкой моделей после финального refit.
""")
code(r"""
def metric(frame, prediction):
    y = frame.y.to_numpy(dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    assert y.shape == p.shape and np.isfinite(p).all() and (p >= 0).all()
    if y.sum() <= 0:
        raise ValueError('Нулевой знаменатель')
    error = p - y
    daily = pd.DataFrame({'day': frame.cutoff.to_numpy(), 'y': y,
                          'absolute': np.abs(error), 'signed': error}).groupby('day').sum()
    if (daily.y <= 0).any():
        raise ValueError('Нулевой знаменатель дневной метрики')
    return dict(WAPE=float(np.abs(error).sum()/y.sum()), Rbias=float(error.sum()/y.sum()),
                score=float((np.abs(error).sum()+abs(error.sum()))/y.sum()),
                daily_score=float(((daily.absolute+daily.signed.abs())/daily.y).mean()))

if VERIFY_VALIDATION:
    evaluation_prediction = np.maximum(strategy.predict(evaluation_calibrator, history, BUNDLE), 0)
    reference = pd.read_parquet(BUNDLE / 'reference/evaluation_predictions.parquet').sort_values(
        ['cutoff', 'horizon', 'route_id']).reset_index(drop=True)
    assert np.array_equal(reference[['cutoff', 'horizon', 'route_id']].to_numpy(),
                          history[['cutoff', 'horizon', 'route_id']].to_numpy())
    np.testing.assert_allclose(evaluation_prediction, reference.calibrated_prediction, rtol=1e-12, atol=1e-6)
    rows = []
    mask = history.split.eq('holdout').to_numpy()
    for label, days in [('holdout', np.sort(holdout.cutoff.unique())),
                        ('неделя 1', np.sort(holdout.cutoff.unique())[:7]),
                        ('неделя 2', np.sort(holdout.cutoff.unique())[7:])]:
        selected = history.cutoff.isin(days).to_numpy()
        rows.append({'период': label, 'горизонт': 0, **metric(history.loc[selected], evaluation_prediction[selected])})
    for h in range(1, 9):
        selected = mask & history.horizon.eq(h).to_numpy()
        rows.append({'период': 'holdout', 'горизонт': h, **metric(history.loc[selected], evaluation_prediction[selected])})
    validation_metrics = pd.DataFrame(rows)
    np.testing.assert_allclose(validation_metrics.iloc[0]['score'], manifest['local_pooled_score'], rtol=0, atol=1e-12)
    np.testing.assert_allclose(validation_metrics.iloc[0]['daily_score'], manifest['local_daily_score'], rtol=0, atol=1e-12)
    display(validation_metrics)
    validation_metrics.to_csv(OUT / 'validation_metrics_reproduced.csv', index=False)
    # Origin 25 октября: текущие метки неизвестны, будущие строки исключены.
    cutoff = pd.Timestamp('2025-10-25 10:30')
    prefix = history.loc[history.cutoff.le(cutoff)].copy()
    prefix.loc[prefix.cutoff.eq(cutoff), 'y'] = np.nan
    prefix_prediction = strategy.predict(evaluation_calibrator, prefix, BUNDLE)
    np.testing.assert_allclose(prefix_prediction[prefix.cutoff.eq(cutoff)],
                               evaluation_prediction[history.cutoff.eq(cutoff)], rtol=1e-12, atol=1e-6)
    print('Проверка неизвестных текущих и будущих меток пройдена.')
""")

md("""
## 5. Submission из восстановленного чекпоинта

Финальные модели поправки используют 28 OOS-дней. Их качество нельзя оценивать
как holdout на тех же обучающих датах. Перенос OOS-поправки на базовые модели после
полного refit — допущение, проверенное здесь только внешним результатом 0.328.
CSV сравнивается с эталонным файлом по `id`, значениям и SHA-256.
""")
code(r"""
future['split'] = 'submission'
assert history.timestamp.max() + pd.Timedelta(hours=4) < future.cutoff.min()
prediction = np.maximum(strategy.predict(final_calibrator, future, BUNDLE), 0)
candidate = future[['id']].copy()
candidate['y_pred'] = prediction
submission = keys[['id']].merge(candidate, on='id', how='left', sort=False, validate='one_to_one')
assert len(submission) == 8000 and submission.id.is_unique
assert submission.id.equals(keys.id.reset_index(drop=True))
assert np.isfinite(submission.y_pred).all() and (submission.y_pred >= 0).all()
reference_submission = pd.read_csv(BUNDLE / 'reference/submission_0328.csv')
assert submission.id.equals(reference_submission.id)
np.testing.assert_allclose(submission.y_pred, reference_submission.y_pred, rtol=1e-12, atol=1e-6)
destination = OUT / 'submission_reproduced.csv'
submission.to_csv(destination, index=False)
verification = dict(checkpoint=manifest['checkpoint'], public_score_reported_by_user=.328,
                    rows=len(submission), numeric_predictions_match=True,
                    max_absolute_difference=float(np.abs(submission.y_pred-reference_submission.y_pred).max()),
                    original_sha256=manifest['original_submission_sha256'], reproduced_sha256=sha256(destination))
verification['byte_identical'] = verification['original_sha256'] == verification['reproduced_sha256']
(OUT / 'verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2))
display(submission.head())
print('Сохранено:', destination)
print(json.dumps(verification, ensure_ascii=False, indent=2))
""")

md("""
## 6. Необязательное переобучение моделей поправки

Чтобы проверить воспроизводимость **обучения поправки**, установите
`REFIT_CALIBRATORS=True` в первой ячейке и выполните Run All. Две оценочные модели
обучатся на tuning, затем две финальные — на всех 28 OOS-днях. Базовые модели
останутся сохранёнными: их полное GPU-переобучение этот ноутбук не запускает.
Результаты будут в отдельной папке `refit/`; исходный чекпоинт не перезаписывается.
Небольшие отличия между версиями библиотек и CPU возможны.
""")
code(r"""
if REFIT_CALIBRATORS:
    refit_dir = OUT / 'refit'
    refit_dir.mkdir(exist_ok=True)
    refit_evaluation = strategy.fit(tuning, BUNDLE)
    refit_p = np.maximum(strategy.predict(refit_evaluation, history, BUNDLE), 0)
    selected = history.split.eq('holdout').to_numpy()
    print('Оценка переобученной поправки до final refit:', metric(history.loc[selected], refit_p[selected]))
    refit_final = strategy.fit(history, BUNDLE)
    refit_prediction = np.maximum(strategy.predict(refit_final, future, BUNDLE), 0)
    result = future[['id']].copy()
    result['y_pred'] = refit_prediction
    result = keys[['id']].merge(result, on='id', how='left', sort=False, validate='one_to_one')
    assert len(result) == 8000 and np.isfinite(result.y_pred).all() and (result.y_pred >= 0).all()
    result.to_csv(refit_dir / 'submission_refit.csv', index=False)
    for phase, model in [('evaluation', refit_evaluation), ('final', refit_final)]:
        for name, item in model.items():
            item['model'].save_model(str(refit_dir / f'{phase}_{name}.cbm'))
    print('Сохранены новые веса поправки и submission:', refit_dir)
else:
    print('Переобучение выключено; восстановление чекпоинта завершено.')
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata.kernelspec = dict(display_name='Python 3 (ipykernel)', language='python', name='python3')
nb.metadata.language_info = dict(name='python', version='3.14')
nb.metadata['checkpoint'] = dict(iteration=10, public_score=.328, local_pooled_score=.3214538236236013)
nbf.validate(nb)
destination = HERE / 'reproduce_best_checkpoint_0328.ipynb'
nbf.write(nb, destination)
print(destination)
