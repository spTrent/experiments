import ast
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root / 'scripts/cluster_experiment'
meta = ast.parse((source / 'best_blend_meta.py').read_text())
names = {'with_volume', 'shape_matrix', 'fit_weights', 'meta_features', 'fit_shape', 'predict_shape'}
functions = ast.unparse(ast.Module(body=[node for node in meta.body if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[]))
cells = []


def add(kind, text):
    text = text.strip() + '\n'
    cell = dict(cell_type=kind, id=f'volume-{len(cells)}', metadata={}, source=text.splitlines(True))
    if kind == 'code':
        compile(text, cell['id'], 'exec')
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)


add('markdown', '''# Три способа прогнозировать объём сети

Все варианты используют доли маршрутов одного CatBoost depth 4 на исходных OOF. Контроль сохраняет объём Session6. Ridge прогнозирует отношение к Session6 с общими признаками утра, поправками горизонтов и расхождениями баз. Третий вариант прогнозирует объём независимо от Session6: отдельный Ridge корректирует сезонный сетевой профиль. Прогноз смешивается с Session6 в пропорциях 25%, 50% и 100%.

На неделях 4–10 и 11–17 октября выбираем настройки двух семейств и общий вариант по среднему WAPE + |Bias|. Train содержит только цели до соответствующей недели. На test сравниваем контроль и по одному заранее выбранному варианту каждого семейства. Доступные до 10:30 наблюдения используются как признаки. Метрика объёма считается по парам день × горизонт, поэтому ошибки не гасятся между горизонтами.

Источники — исходный blend_best_output и исходная размеченная история. Оценка использует архивные прогнозы баз, поэтому её числа не являются оценкой эксперимента trim25 или сабмита на Kaggle. Локальный test уже изучался. Оценка test получена до полного переобучения. Ноутбук запускается на CPU и не создаёт конкурсные CSV.
''')
add('code', (source / 'best_blend_data.py').read_text())
add('code', 'from catboost import CatBoostRegressor\nfrom sklearn.linear_model import Ridge\nfrom sklearn.preprocessing import StandardScaler\n\n' + functions)
add('code', (source / 'network_volume_experiment.py').read_text())
add('code', '''import os

INPUT_ROOT = Path(os.environ.get('BLEND_INPUT', 'blend_inputs'))
EVALUATION_DIR = INPUT_ROOT / 'blend_best_output'
HISTORY_PATH = INPUT_ROOT / 'session6/train_solo_track.parquet'
ARTIFACTS = Path(os.environ.get('VOLUME_OUTPUT', 'output_volume_experiment'))
ARTIFACTS.mkdir(parents=True, exist_ok=True)
contract = json.loads((EVALUATION_DIR / 'inputs.json').read_text())
if digest_file(HISTORY_PATH) != contract['session_sha256'][HISTORY_PATH.name]:
    raise ValueError('История отличается от исходного запуска')
network = network_features(load_network(HISTORY_PATH))
columns = KEYS + ['y', *BASES, 'level24', 'current', 'delta1h', 'delta2h', 'network_current']
oof = pd.read_parquet(EVALUATION_DIR / 'base_oof.parquet', columns=columns).sort_values(KEYS).reset_index(drop=True)
targets = oof.cutoff + oof.horizon * STEP
if oof.duplicated(KEYS).any() or not targets.lt(LOCAL_TEST_START).all():
    raise ValueError('Некорректные ключи или границы OOF')
''')
add('code', '''blocks = []
for start, stop in OOF_WINDOWS[2:]:
    train = oof.loc[targets.lt(start)]
    val = oof.loc[oof.cutoff.ge(start) & oof.cutoff.lt(stop)]
    table, _ = score_variants(train, val, network, start, 'val')
    table['week'] = str(start.date())
    display(table)
    blocks.append(table)
validation = pd.concat(blocks, ignore_index=True)
validation.to_csv(ARTIFACTS / 'validation_metrics.csv', index=False)
ranking = validation.loc[validation.period.eq('val')].groupby('variant', as_index=False).agg(
    mean_score=('score', 'mean'), worst_score=('score', 'max'), mean_volume_WAPE=('volume_WAPE', 'mean'), weeks=('score', 'count'))
if not ranking.weeks.eq(2).all():
    raise ValueError('Неполная validation')
ranking = ranking.sort_values(['mean_score', 'worst_score', 'variant'])
selected = ['session6', *[ranking.loc[ranking.variant.str.startswith(prefix), 'variant'].iloc[0]
                         for prefix in ['coefficient_', 'direct_']]]
frozen = dict(variants=selected, winner=ranking.variant.iloc[0], selected_before_test=True)
write_json(ARTIFACTS / 'frozen_selection.json', frozen)
ranking.to_csv(ARTIFACTS / 'validation_ranking.csv', index=False)
display(ranking)
''')
add('code', '''frozen = json.loads((ARTIFACTS / 'frozen_selection.json').read_text())
test = pd.read_parquet(EVALUATION_DIR / 'test_predictions.parquet', columns=columns).sort_values(KEYS).reset_index(drop=True)
if (test.duplicated(KEYS).any() or not test.cutoff.ge(LOCAL_TEST_START).all() or
        not (test.cutoff + test.horizon*STEP).lt(LOCAL_TEST_END).all()):
    raise ValueError('Некорректные ключи или границы test')
report, predictions = score_variants(oof, test, network, LOCAL_TEST_START, 'test', frozen['variants'])
report['winner_on_val'] = report.variant.eq(frozen['winner'])
report.to_csv(ARTIFACTS / 'train_test_metrics.csv', index=False)
display(report)
rows = test[KEYS + ['y']].assign(**predictions)
save_parquet(ARTIFACTS / 'test_predictions.parquet', rows)
volumes = rows.groupby(GROUP_KEYS, as_index=False)[['y', *predictions]].sum()
volumes.to_csv(ARTIFACTS / 'test_volume_by_day_horizon.csv', index=False)
print('Оценка test получена до полного переобучения. Конкурсный прогноз не создавался.')
''')
output = root / 'stacking/blending/network_volume_experiment_server.ipynb'
output.write_text(json.dumps(dict(nbformat=4, nbformat_minor=5, metadata=dict(
    kernelspec=dict(name='python3', display_name='Python 3', language='python')), cells=cells), ensure_ascii=False, indent=1) + '\n')
print(output)
