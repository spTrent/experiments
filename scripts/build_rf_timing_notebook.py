import ast
import json
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
original = json.loads((BASE / 'stacking/stacking_part_1.ipynb').read_text())
cells = []


def cell(kind, source):
    entry = dict(cell_type=kind, metadata={}, source=source.strip().splitlines(True), id=f'rf-time-{len(cells):02}')
    if kind == 'code':
        compile(source, entry['id'], 'exec')
        entry.update(execution_count=None, outputs=[])
    cells.append(entry)


cell('markdown', '''# Замер Random Forest: последние 75% train

В Kaggle включите GPU T4 x2, подключите train_solo_track.parquet и test_solo_track.parquet и выполните Save & Run All.
Нужны установленные CuPy, cuML, pandas, PyArrow и sklearn. Ноутбук автономный, другие файлы проекта не нужны.

По умолчанию: h1, direct и residual, параметры из stacking_part_1; без подбора.
Train содержит только метки до 18 октября 2025. Первые floor(N / 4) обучающих дат исключены,
на оставшихся датах используются все получасовые старты. Более ранние факты остаются доступными для лагов.
Проверка: 18–31 октября, только 10:30. Это замер времени, не основание для выбора модели по test.
Можно задать BENCHMARK_HORIZONS=(1,8), чтобы дополнительно замерить h8, либо BOUNDARY='2025-10-11'
и QUERY_DAYS=7 для точного воспроизведения последней OOF-недели.

Отдельно измеряются загрузка, общие признаки, признаки горизонта, корреляционные лаги,
сборка и обработка матриц, fit, predict train и predict test. GPU синхронизируется на границах замеров.
Результаты каждого законченного этапа сразу записываются в /kaggle/working/rf_benchmark.
Это диагностический ноутбук: он не продолжает прерванный fit и не создаёт результаты для сборщика stacking.
''')
setup = ''.join(original['cells'][1]['source'])
tree = ast.parse(setup)
params = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PARAMS' for t in n.targets))
params = {k: v for k, v in params.items() if k.startswith('rf_')}
cell('code', f'''import time
from pathlib import Path
BENCHMARK_START = time.monotonic()
BENCHMARK_HORIZONS = (1,)
VARIANTS = ('direct', 'residual')
BOUNDARY = '2025-10-18'
QUERY_DAYS = 14
INPUT_DIR = '/kaggle/input'
OUTPUT_DIR = '/kaggle/working/rf_benchmark'
PROFILE = 'recent75'
MODE = 'oof'
ROLE = 'worker'
PARAMS = {params!r}
''')
runtime = ast.literal_eval(ast.parse(''.join(original['cells'][2]['source'])).body[0].value.args[0])
cell('code', runtime)
cell('code', '''import cupy as cp
import cuml
from cuml.ensemble import RandomForestRegressor
xp = cp
BACKEND = 'cuda'
INPUT = Path(INPUT_DIR)
ROOT = Path(OUTPUT_DIR)
ROOT.mkdir(parents=True, exist_ok=True)
TRAIN_PATH = locate('train_solo_track.parquet')
TEST_PATH = locate('test_solo_track.parquet')
records = []
def record(stage, started, **details):
    cp.cuda.runtime.deviceSynchronize()
    row = dict(stage=stage, seconds=time.monotonic()-started,
               elapsed_seconds=time.monotonic()-BENCHMARK_START, **details)
    records.append(row)
    temp = ROOT / 'timings.partial.csv'
    pd.DataFrame(records).to_csv(temp, index=False)
    temp.replace(ROOT / 'timings.csv')
    print(json.dumps(row, ensure_ascii=False), flush=True)

def tick():
    cp.cuda.runtime.deviceSynchronize()
    return time.monotonic()

devices = [str(cp.cuda.runtime.getDeviceProperties(i)['name']) for i in range(cp.cuda.runtime.getDeviceCount())]
(ROOT/'environment.json').write_text(json.dumps(dict(gpus=devices, cpu_count=os.cpu_count(),
    cupy=cp.__version__, cuml=cuml.__version__, horizons=BENCHMARK_HORIZONS,
    boundary=BOUNDARY, query_days=QUERY_DAYS, params=PARAMS), indent=2))
print('GPU:', devices, 'Random Forest использует одну GPU на fit', flush=True)
load_start = tick()
''')
for index, before, after in [(4, '', "record('load_panel', load_start)"), (5, 'feature_start = tick()', "record('shared_features', feature_start)"), (6, '', '')]:
    source = ast.literal_eval(ast.parse(''.join(original['cells'][index]['source'])).body[0].value.args[0])
    cell('code', '\n'.join([before, source, after]))
cell('code', '''boundary = pd.Timestamp(BOUNDARY)
query = query_indices(boundary, QUERY_DAYS)
metrics = []
for h in BENCHMARK_HORIZONS:
    if str(h) not in PARAMS['rf_direct']:
        raise ValueError('Встроены параметры только h1 и h8')
    origins = train_indices(h, boundary, 'recent_all')
    possible = np.flatnonzero(times[:len(times)-h] + h*STEP < boundary)
    all_days = times[possible].normalize().unique()
    dropped = len(all_days)//4
    if len(origins) == 0 or not (times[origins] + h*STEP < boundary).all():
        raise ValueError('Неверная граница train')
    if not times[origins].normalize().isin(all_days[dropped:]).all():
        raise ValueError('В train остались ранние даты')
    print(dict(h=h, total_dates=len(all_days), excluded_dates=dropped,
               first_train=str(times[origins[0]]), last_label=str(times[origins[-1]]+h*STEP),
               potential_rows=len(origins)*len(routes)), flush=True)
    scratch = '/kaggle/temp' if Path('/kaggle/temp').is_dir() else None
    with TemporaryDirectory(dir=scratch) as folder:
        started = tick()
        data = HorizonData(folder, h, int(query[-1])+1)
        record('horizon_features', started, horizon=h)
        started = tick()
        data.prepare(boundary, int(query[-1])+1)
        record('correlation_features', started, horizon=h)
        started = tick()
        x, train = global_data(data, origins, 'train', True)
        q, valid = global_data(data, query, 'query', False)
        record('matrices', started, horizon=h, train_rows=len(train), test_rows=len(valid), features=x.shape[1])
        started = tick()
        a, b = numeric_data(x, q, train.route_id, valid.route_id, data.folder, False)
        record('imputation_encoding', started, horizon=h, features=a.shape[1], matrix_gib=a.nbytes/2**30)
        for variant in VARIANTS:
            name = 'rf_' + variant
            baseline = train.profile28.to_numpy() if variant == 'residual' else 0.
            y = ((train.y.to_numpy(dtype=np.float64)-baseline)/TARGET_SCALE).astype(np.float32)
            started = tick()
            model = RandomForestRegressor(random_state=42, n_streams=4, output_type='numpy',
                                           **PARAMS[name][str(h)]['params'])
            model.fit(a, y)
            record('fit', started, horizon=h, model=name)
            predictions = []
            for split, matrix, frame in [('train', a, train), ('test', b, valid)]:
                started = tick()
                pred = np.concatenate([np.asarray(model.predict(matrix[i:i+32768])).reshape(-1)
                                       for i in range(0,len(matrix),32768)])
                record('predict_'+split, started, horizon=h, model=name, rows=len(matrix))
                pred = np.maximum(0., pred*TARGET_SCALE + (frame.profile28.to_numpy() if variant=='residual' else 0.))
                if not np.isfinite(pred).all():
                    raise ValueError('Нечисловой прогноз')
                metrics.append(dict(horizon=h, model=name, split=split, **metric(frame.y, pred)))
                predictions.append(pred)
            valid.assign(prediction=predictions[1]).to_parquet(ROOT/f'h{h}_{name}_test.parquet', index=False)
            pd.DataFrame(metrics).to_csv(ROOT/'metrics.csv', index=False)
            display(pd.DataFrame(metrics).tail(2))
            del model, y, predictions, pred
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
        del data, x, q, train, valid, a, b, matrix, frame
    gc.collect()
record('total', BENCHMARK_START)
display(pd.DataFrame(records))
print('Для оценки полного stacking нужны timings.csv и environment.json из', ROOT)
''')
result = dict(nbformat=4, nbformat_minor=5, metadata=original['metadata'], cells=cells)
destination = BASE / 'stacking/rf_recent75_benchmark.ipynb'
destination.write_text(json.dumps(result, ensure_ascii=False, indent=1)+'\n')
print(destination)
