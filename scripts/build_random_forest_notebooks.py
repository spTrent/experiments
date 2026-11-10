from pathlib import Path
import ast
import hashlib
import nbformat

ROOT = Path(__file__).resolve().parents[1]
features = (ROOT / 'scripts/five_starts_features.py').read_text()
models = (ROOT / 'scripts/random_forest_models.py').read_text()
builder = ast.parse((ROOT / 'scripts/build_model_notebooks.py').read_text())
setup = next(ast.literal_eval(node.value) for node in builder.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'setup' for t in node.targets))
setup = setup.replace('N_TRIALS = 20', 'N_TRIALS = 10\nN_STREAMS = 4\nRF_TREES = [150, 300]\nRF_DEPTHS = [10, 16]\nRESUME_FROM = None')
setup = setup.replace('PREDICTION_BATCH = 4096', 'PREDICTION_BATCH = 32768')
setup = setup.replace("RUN_DIR = BASE / 'artifacts' / f'{{MODEL_NAME}}_stack'\nif RUN_DIR.exists():\n    raise FileExistsError(f'Папка {{RUN_DIR}} уже существует. Перенесите предыдущий результат или измените MODEL_NAME.')", "RUN_DIR = BASE / 'artifacts' / f'{{MODEL_NAME}}_search_eval_v2'")
setup = setup.replace("OUT = RUN_DIR / 'features'", "OUT = BASE / 'artifacts' / 'rf_shared_features_v2'")
setup = setup.replace("IMPORTANCE_DIR = RUN_DIR / 'importance'\n", '')
setup = setup.replace('(OUT, MODEL_DIR, IMPORTANCE_DIR, POSTPROCESS_DIR, DOCS_DIR, PARAMS_DIR)', '(OUT, MODEL_DIR, POSTPROCESS_DIR, DOCS_DIR, PARAMS_DIR)')
setup = setup.replace('directory.mkdir(parents=True)', 'directory.mkdir(parents=True, exist_ok=True)')
setup = setup.replace('MAX_TREES = 3000\nEARLY_STOPPING_ROUNDS = 100\n', '')
setup = setup.replace("TRAIN_PATH, TEST_PATH = DATA / TRAIN_FILE, DATA / TEST_FILE", "TRAIN_PATH = DATA / TRAIN_FILE")
a = setup.index('if not (TRAIN_PATH.is_file()')
b = setup.index("BASE = Path", a)
setup = setup[:a] + '''if not TRAIN_PATH.is_file() and Path('/kaggle/input').exists():
    train_matches = list(Path('/kaggle/input').rglob(TRAIN_FILE))
    if len(train_matches) != 1:
        raise ValueError('Укажите TRAIN_PATH вручную')
    TRAIN_PATH = train_matches[0]
''' + setup[b:]
setup = setup.replace("TEST_FILE = 'test_solo_track.parquet'\n", '')
setup = setup.replace('for directory in (OUT, MODEL_DIR, POSTPROCESS_DIR, DOCS_DIR, PARAMS_DIR):', '''if RESUME_FROM is not None:
    import shutil
    source = Path(RESUME_FROM)
    if not source.is_dir():
        raise ValueError('RESUME_FROM должен указывать на сохранённую папку artifacts')
    for folder_name in (RUN_DIR.name, OUT.name):
        source_folder = source / folder_name
        destination = BASE / 'artifacts' / folder_name
        if source_folder.exists() and not destination.exists():
            shutil.copytree(source_folder, destination)
for directory in (OUT, MODEL_DIR, POSTPROCESS_DIR, DOCS_DIR, PARAMS_DIR):''')
cache_setup = '''
import hashlib
import importlib.metadata

def file_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_contract(path, contract):
    if path.exists():
        if json.loads(path.read_text()) != contract:
            raise ValueError(f'Настройки или данные изменились: {path}. Используйте новую папку результатов/кэша.')
    else:
        temporary = path.with_suffix('.partial')
        temporary.write_text(json.dumps(contract, ensure_ascii=False, indent=2))
        temporary.replace(path)

FEATURE_CONTRACT = dict(source=FEATURE_SOURCE_HASH, data=file_digest(TRAIN_PATH), starts=list(TRAIN_MINUTES),
                        offsets=list(WINDOW_OFFSETS), lookback=LAG_MAX_LOOKBACK_HOURS,
                        min_pairs=LAG_MIN_PAIRS, min_correlation=LAG_MIN_CORRELATION)
verify_contract(OUT / 'contract.json', FEATURE_CONTRACT)
PREPARED = OUT / ('prepared_' + importlib.metadata.version('scikit-learn') + '_' + MODEL_SOURCE_HASH[:16])
PREPARED.mkdir(parents=True, exist_ok=True)
RUN_CONTRACT = dict(features=FEATURE_CONTRACT, source=MODEL_SOURCE_HASH, residual=RESIDUAL,
                    n_trials=N_TRIALS, seed=SEED, n_streams=N_STREAMS, trees=RF_TREES, depths=RF_DEPTHS,
                    versions={name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'scikit-learn', 'optuna')},
                    cuml=__import__('cuml').__version__)
verify_contract(RUN_DIR / 'contract.json', RUN_CONTRACT)
'''
features = features.replace("test = pd.read_parquet(TEST_PATH, columns=['id', 'route_id', 'timestamp'])\n", '')
features = features.replace('history.notna().all().all() and test.notna().all().all()', 'history.notna().all().all()')
a = features.index("if not (test.id.is_unique")
b = features.index('values = ', a)
features = features[:a] + 'end = times[-1]\n' + features[b:]
a = features.index('    features = horizon_dataset(h, np.array([len(times) - 1]))')
b = features.index('for h in range(1, 9):\n    write_horizon(h)', a)
features = features[:a] + "    print(f'h={h}: история {len(origins)*len(routes):,}, модельных признаков {len(frame.columns) - 5}')\n" + features[b:]
features = features.replace("    path = OUT / f'history_h{h}.parquet'\n", "    path = OUT / f'history_h{h}.parquet'\n    if path.exists():\n        print(f'h={h}: общие признаки загружены из кэша')\n        return\n")
feature_blocks = [('Данные и временное разбиение', 'history = pd.read_parquet(TRAIN_PATH'),
                  ('Общие признаки пяти стартов', 'import pyarrow as pa'),
                  ('Корреляционные лаги внутри train', 'def calculate_dynamic_shoulder'),
                  ('Общие матрицы восьми горизонтов', 'def horizon_dataset'),
                  ('Метрика соревнования', 'def mean_daily_score')]
model_blocks = [('Кэширование, модели и продолжение поиска', 'import time'),
                ('Подбор параметров до test', 'selected = {}'),
                ('Обучение на train + val и независимая оценка test', 'evaluation_rows = []')]
intro = '''# Random Forest {variant}: подбор и независимая оценка

Восемь моделей по горизонтам +30 минут — +4 часа, все маршруты вместе. Train использует старты 09:30, 10:00, 10:30, 11:00, 11:30; tuning и test — 10:30. Direct обучается на y / 100000, residual — на (y − profile28) / 100000. Уровень восстанавливается перед отсечением отрицательных прогнозов.

Optuna: 10 конфигураций на горизонт, каждая проверяется на трёх недельных tuning-окнах 30 августа — 19 сентября 2025 года. Критерий — среднее WAPE + |Relative Bias| по окнам. Подбор: 150/300 деревьев, глубина 10/16, min_samples_leaf 2–64 (log), max_features 0.25–0.9, max_samples 0.5–1. Seed=42. Итого 240 обучений при подборе и 8 для оценки. Early stopping отсутствует; недельный val перед каждым tuning сохранён как временной зазор и отдельно не вычисляется.

После фиксации всех настроек каждый горизонт обучается на всей пригодной размеченной истории до 18 октября, включая train, val и прошлые tuning-периоды. Test: 18–31 октября. Во время подбора выводятся train/tuning; независимые train/test — после фиксации настроек. Test не используется в objective. Независимая оценка test получена без переобучения на всей истории. Период test ранее использовался в проекте и не является новым слепым тестом. Финальное переобучение, submission и importance отсутствуют.

В Kaggle включите NVIDIA GPU; нужны совместимые cuML и CuPy. Обучение использует один GPU, n_streams=4; bitwise-воспроизводимость не гарантируется. Прогнозы идут блоками 32768 строк. При нехватке памяти уменьшите PREDICTION_BATCH. Предобработка выполняется на CPU; медианы и корреляционные лаги определяются только на train текущего окна. Маршруты кодируются двоичным индексом в общем справочнике без таргета.

Запускайте один ноутбук за раз. Базовые признаки и подготовленные матрицы общие для direct/residual: artifacts/rf_shared_features_v2. Кэш матриц хранится на диске и читается через mmap, поэтому не требуется держать три fold в RAM. Для обмена кэшем между отдельными Kaggle-сессиями сохраните папку artifacts как output/dataset и укажите RESUME_FROM = '/kaggle/input/.../artifacts'. В той же сессии оставьте None и повторно запустите ноутбук. Продолжение использует завершённые trials и готовые модели оценки; прерванный trial или горизонт оценки выполняется заново. Параллельные запуски в одну папку не поддерживаются. При изменении данных или настроек используйте новые папки RUN_DIR и OUT. Старые результаты первой версии автоматически не используются.

В docs/trials_hN.csv сохраняются длительности trials, в docs/metrics.csv — метрики и длительности fold. Артефакты: параметры model_params/best_hN.json, модели models/evaluation/hN/model.joblib (model, preprocess, metadata), прогнозы postprocessing, отчёты docs и reports.zip. Время полного прогона измеряется на Kaggle; сокращение поиска может изменить качество.
'''

def build():
    for residual in (False, True):
        variant = 'residual' if residual else 'direct'
        code = setup.format(algorithm='random_forest', residual=residual, experiment=f'random_forest_{variant}')
        feature_hash = hashlib.sha256(features.encode()).hexdigest()
        model_hash = hashlib.sha256((models + cache_setup).encode()).hexdigest()
        code += f'\nFEATURE_SOURCE_HASH = {feature_hash!r}\nMODEL_SOURCE_HASH = {model_hash!r}\n' + cache_setup
        cells = [nbformat.v4.new_markdown_cell(intro.format(variant=variant)), nbformat.v4.new_code_cell(code)]
        for source, blocks in ((features, feature_blocks), (models, model_blocks)):
            offsets = [source.index(marker) for _, marker in blocks] + [len(source)]
            for i, (title, _) in enumerate(blocks):
                cells.extend([nbformat.v4.new_markdown_cell('## ' + title), nbformat.v4.new_code_cell(source[offsets[i]:offsets[i+1]].strip())])
        notebook = nbformat.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}})
        for index, cell in enumerate(notebook.cells):
            cell.id = f'rf-{variant}-{index:02d}'
        nbformat.write(notebook, ROOT / f'notebooks/models/random_forest_{variant}.ipynb')


if __name__ == '__main__':
    build()
