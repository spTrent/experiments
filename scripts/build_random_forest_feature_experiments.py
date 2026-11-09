from pathlib import Path
import hashlib
import json
import nbformat
import build_random_forest_notebooks as base

ROOT = Path(__file__).resolve().parents[1]
additions = (ROOT / 'scripts/random_forest_experimental_features.py').read_text()
experiments = (ROOT / 'scripts/random_forest_feature_experiments.py').read_text()
feature_source = base.features.replace('def horizon_dataset(h, origins):', additions + '\n\ndef horizon_dataset(h, origins):')
feature_source = feature_source.replace("selection_end = pd.Timestamp('2025-09-20')", "selection_end = pd.Timestamp('2025-10-11')")
feature_source = feature_source.replace('len(selection_dates) <= 28', 'len(selection_dates) <= 49')
feature_source = feature_source.replace('len(selection_dates) - 21', 'len(selection_dates) - 42')
feature_source = feature_source.replace("    numeric = frame.columns.difference(['cutoff', 'timestamp', 'route_id'])", "    frame = add_experimental_features(frame, h, origins)\n    numeric = frame.columns.difference(['cutoff', 'timestamp', 'route_id'])")
setup = base.setup.format(algorithm='random_forest',residual=False,experiment='random_forest_feature_experiments')
setup = setup.replace("f'{MODEL_NAME}_search_eval_v2'", "f'{MODEL_NAME}_rf_direct_seed_v4'")
setup = setup.replace('N_TRIALS = 10', 'N_TRIALS = 20')
setup = setup.replace('RF_DEPTHS = [10, 16]', 'RF_DEPTHS = [6, 8, 10, 12, 16]')
setup = setup.replace('rf_shared_features_v2','rf_experiment_features_robust_v3')
config = '''
VARIANTS = ['direct']
HORIZONS = list(range(1, 9))
EXPERIMENT_NAMES = ['baseline', 'thin_offsets', 'compact_with_additions']
'''
source_parameters = {}
for h in range(1, 9):
    path = ROOT / 'artifacts' / 'rf_direct' / 'model_params' / f'best_h{h}.json'
    record = json.loads(path.read_text())
    if record.get('algorithm') != 'random_forest' or record.get('residual') is not False or record.get('horizon') != h:
        raise ValueError(f'Некорректный источник параметров: {path}')
    source_parameters[h] = record['params']
config += '\nBASE_PARAMS_BY_HORIZON = ' + repr(source_parameters) + '\n'
config += "BASE_PARAMS_SOURCE = 'artifacts/rf_direct/model_params/best_hN.json'\n"
config += "\ndef base_params_for_horizon(h):\n    return dict(BASE_PARAMS_BY_HORIZON[h])\n"
model_functions = base.models[:base.models.index('manifest = dict(')]
model_functions = model_functions.replace("'min_samples_leaf', 2, 64", "'min_samples_leaf', 8, 256")
model_functions = model_functions.replace("'max_features', 0.25, 0.9", "'max_features', 0.3, 0.9")
model_functions = model_functions.replace("'max_samples', 0.5, 1", "'max_samples', 0.5, 1")
model_functions = model_functions.replace("    for trial in study.get_trials(states=(optuna.trial.TrialState.RUNNING,)):", "    if not study.trials:\n        study.enqueue_trial(base_params_for_horizon(h))\n    for trial in study.get_trials(states=(optuna.trial.TrialState.RUNNING,)):")

evaluation = base.models[base.models.index('evaluation_rows = []'):]
evaluation = evaluation.replace('range(1, 9)', 'HORIZONS')
evaluation = evaluation.replace("        metadata.update(algorithm=ALGORITHM", "        encoded_names = list(joblib.load(folder / 'preprocess.joblib').feature_names_in_)\n        selected_names = experiment_columns(encoded_names, EXPERIMENTS[ACTIVE_EXPERIMENT])\n        metadata.update(experiment=ACTIVE_EXPERIMENT, selected_features=selected_names, selected_indices=[encoded_names.index(n) for n in selected_names])\n        metadata.update(algorithm=ALGORITHM")
evaluation = evaluation.replace("RUN_DIR / 'reports.zip'", "RUN_DIR / variant / 'reports.zip'")
evaluation = evaluation.replace('path.relative_to(RUN_DIR)', 'path.relative_to(RUN_DIR / variant)')
final = '''for variant in VARIANTS:
    select_context(variant, winners[variant])
    selected = all_selected[variant]
    manifest = dict(variant=variant, experiment=winners[variant], contract=RUN_CONTRACT,
                    test_note='Независимая оценка test получена после фиксации признаков и параметров, до какого-либо переобучения на всей истории. Переобучение на всей истории и submission не выполняются. Период test ранее использовался в проекте и не является новым слепым test.')
    atomic_json(DOCS_DIR / 'manifest.json', manifest)
'''
final += '\n'.join('    '+line if line else '' for line in evaluation.splitlines())
setup += config
feature_hash = hashlib.sha256(feature_source.encode()).hexdigest()
model_hash = hashlib.sha256((model_functions+experiments+final).encode()).hexdigest()
setup += f'\nFEATURE_SOURCE_HASH = {feature_hash!r}\nMODEL_SOURCE_HASH = {model_hash!r}\n'
cache = base.cache_setup.replace('residual=RESIDUAL,', 'variants=VARIANTS, horizons=HORIZONS, experiments=EXPERIMENT_NAMES, base_params=BASE_PARAMS_BY_HORIZON,')
setup += cache
intro = '''# Direct Random Forest: устойчивость признаков и параметров

Самодостаточный ноутбук Kaggle: NVIDIA GPU, cuML и CuPy. По умолчанию direct и все восемь горизонтов. Обучение запускает пользователь.

## Протокол и временные границы

Сравниваются три кандидата: baseline, thin_offsets и compact_with_additions. Первый сохраняет исходный набор, второй прореживает несезонные смещения (центр и −2 вместо пяти значений), третий сокращает окна, календарь, сетевые смещения и флаги доступности, удаляет selected_status6_pair_count и добавляет время старта, profile28, level24, аномалии, тренды, изменчивость и сезонный разброс.

Сначала все кандидаты обучаются с BASE_PARAMS_BY_HORIZON: лучшие параметры каждого горизонта из artifacts/rf_direct/model_params/best_hN.json встроены в первую ячейку. Для данного горизонта параметры одинаковы у всех кандидатов. Старые tuning-score не используются: конфигурации заново оцениваются на шести окнах. Train использует пять стартов 09:30–11:30, tuning — 10:30. Шесть недельных tuning-окон: 30 августа — 5 сентября, 6–12 сентября, 13–19 сентября, 20–26 сентября, 27 сентября — 3 октября, 4–10 октября 2025 года. Последние три окна — дополнительная проверка переноса во времени. Перед каждым окном оставлена неделя val-зазора. Корреляционные лаги и медианы обучаются только на train соответствующего окна. Новые признаки используют данные не позднее cutoff.

Общий набор выбирается по средней метрике шести окон; в каждом окне сначала объединяются ошибки по всем включённым горизонтам и маршрутам. Отдельно выводятся результаты и изменения к baseline для каждого горизонта и окна, число улучшений, разброс и худшее окно. Небольшое среднее улучшение не доказывает устойчивость при одном seed.

Затем для выбранного набора запускается Optuna отдельно на каждом горизонте: 20 trials, включая исходные параметры соответствующего горизонта первым trial. Диапазоны: 150/300 деревьев, глубина [6,8,10,12,16], min_samples_leaf 8–256 (log), max_features 0.3–0.9, max_samples 0.5–1. Критерий — среднее WAPE + |Relative Bias| по шести окнам. Глубина 16 и листья от 8 сохранены: поиск проверяет строгие и гибкие модели на новых tuning-окнах, без запрета конфигурации по просмотренному test. После поиска таблица base_vs_optuna сравнивает исходные и лучшие параметры, WAPE, signed Bias и |Bias| на каждом окне. Первый trial входит в общий бюджет. Подбор не должен проиграть собственному первому trial по средней tuning-метрике; улучшение на test не гарантируется.

Все метки подбора строго раньше 11 октября. После фиксации признаков и параметров модель обучается на всей пригодной истории до 18 октября, включая train и val, и оценивается на test 18–31 октября. Во время поиска показываются train/tuning; после фиксации — train/test. Независимая оценка получена до какого-либо переобучения на всей истории. Переобучения на всей истории, importance и submission нет. Test уже просмотрен и использовался в проекте: это контрольная оценка переноса, а не новый слепой test. Он не участвует в автоматическом отборе.

## Бюджет, кэш и продолжение

По умолчанию: 3 набора × 8 горизонтов × 6 окон = 144 обучения сравнения; 20 × 8 × 6 = 960 обучений Optuna; 8 для test. Всего 1112 обучений. Для пилота можно изменить HORIZONS на [1,4,6] и N_TRIALS на 10 до запуска. Число обучений и время отражаются в отчётах. n_streams=4, seed=42, битовая воспроизводимость GPU не гарантируется. PREDICTION_BATCH=32768; уменьшите при нехватке памяти.

Новая папка результатов: artifacts/random_forest_feature_experiments_rf_direct_seed_v4. Для изменённых временных границ используется отдельный кэш rf_experiment_features_robust_v3. Старые результаты автоматически не смешиваются с новым протоколом. Подготовленные матрицы хранятся на диске, train-only медианы общие для кандидатов. Повторный Run All продолжает завершённые сравнения, trials и оценку; прерванное обучение повторяется. Запускайте один экземпляр. Для другой Kaggle-сессии сохраните artifacts как dataset и укажите RESUME_FROM='/kaggle/input/.../artifacts'. При изменении конфигурации выберите новую RUN_DIR.

## Артефакты

feature_experiments_summary.csv — сравнение наборов. feature_experiments_by_fold.csv и feature_experiments_by_horizon.csv — детализация. feature_stability_by_horizon.csv — устойчивость по окнам и горизонтам; feature_deltas_by_window.csv — изменения к baseline. В direct/docs: base_vs_optuna_by_fold.csv, base_vs_optuna_summary.csv, trials_hN.csv, train/test метрики по горизонтам и в целом. В каждой таблице сохраняются WAPE и Bias отдельно, чтобы видеть взаимную компенсацию ошибок. feature_selection_frozen.json и all_parameters_frozen.json фиксируют выбор до test.

Model bundle содержит model, preprocess полного набора, metadata.selected_features и metadata.selected_indices: после preprocess.transform необходимо отобрать selected_indices перед model.predict. Модели оценки и прогнозы сохраняются по горизонтам. reports.zip содержит отчёты.
'''

cells = [nbformat.v4.new_markdown_cell(intro), nbformat.v4.new_code_cell(setup)]
for source, blocks in [(feature_source,base.feature_blocks)]:
    markers = [(title, 'def stable_ratio' if marker=='def horizon_dataset' else marker) for title,marker in blocks]
    offsets = [source.index(marker) for _,marker in markers]+[len(source)]
    for i,(title,_) in enumerate(markers):
        cells.extend([nbformat.v4.new_markdown_cell('## '+title),nbformat.v4.new_code_cell(source[offsets[i]:offsets[i+1]].strip())])
cells.extend([nbformat.v4.new_markdown_cell('## Кэш, модель и Optuna'),nbformat.v4.new_code_cell(model_functions),
              nbformat.v4.new_markdown_cell('## Сравнение признаков и подбор выбранных моделей'),nbformat.v4.new_code_cell(experiments),
              nbformat.v4.new_markdown_cell('## Независимая оценка после фиксации выбора'),nbformat.v4.new_code_cell(final)])
notebook = nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
for i,cell in enumerate(notebook.cells):
    cell.id=f'rf-experiments-{i:02d}'
nbformat.write(notebook,ROOT/'notebooks/features/random_forest_feature_experiments.ipynb')
