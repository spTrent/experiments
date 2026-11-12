import ast
import json
import zipfile
from pathlib import Path


root = Path(__file__).resolve().parents[1]
clustering = json.loads((root / 'notebooks/clustering/session6_lightgbm_clusters.ipynb').read_text())
legacy = json.loads((root / 'best/raw-vs-clean.ipynb').read_text())
runtime = (root / 'scripts/cluster_experiment/session6_full_submissions.py').read_text()
cells = []


def add(kind, source):
    cell = dict(cell_type=kind, id=f'session6-full-{len(cells)}', metadata={}, source=source.splitlines(True))
    if kind == 'code':
        compile(source, cell['id'], 'exec')
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)


def legacy_code(index):
    return ''.join(legacy['cells'][index]['source'])


def function(source, name):
    node = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name)
    return ast.unparse(node)


add('markdown', '''# Session6. Сабмиты после полного переобучения

Продолжение эксперимента `clustering326`. На локальном test архивный Session6 дал 0.319190, новая общая LightGBM при его объёме дала 0.319163, LightGBM по кластерам дала 0.319496. Общая LightGBM также выиграла на val. Эти оценки получены до полного переобучения, в предыдущем запуске. Локальный период ранее изучался.

В Kaggle оставить подключённые датасеты `case_rwb`, `cleaning_inputs` и `s6_residual_inputs`, включить GPU и выполнить Run All. `case_rwb` даёт train/test parquet, `cleaning_inputs` — параметры LightGBM, `s6_residual_inputs` — OOF-прогнозы CatBoost и признаки для поправки. Файлы можно хранить распакованными или внутри ZIP. Если `case_rwb` отсутствует, train/test читаются из `cleaning_inputs`.

Восемь CatBoost заново обучаются на всех доступных получасовых стартах с известной целью, включая бывшие val и локальный test. Это расширяет окно по сравнению с recent75 в исходном архивном CatBoost. LightGBM обучается на пяти утренних стартах за всю доступную размеченную историю с нужным прогревом признаков. KMeans, масштабы маршрутов и корреляционные лаги используют историю до конкурсного cutoff. Две модели поправки CatBoost переобучаются на всех доступных исходных OOF за 4–31 октября. Остатки этой истории дают причинный EWMA-сдвиг на конкурсный cutoff. CatBoost обучается на GPU, LightGBM на CPU.

`submission.csv` использует общую LightGBM, `submission_clusters.csv` использует восемь кластеров с общей моделью для малых групп. Обе формы нормируются к объёму заново обученного компонента Session6 с исходным весом 0.4997259. Обе сохраняют одинаковые суммы горизонтов. После каждого обучения выводятся метрики на обучении полного refit, а архивная независимая оценка показана отдельной таблицей. Финальный ZIP содержит модели, метрики, прогнозы и оба CSV. Новый результат на платформе станет известен после отправки файла.
''')
initialization = ''.join(clustering['cells'][1]['source'])
initialization = initialization.replace("RUN = work / 'session6_lightgbm_clusters'", "RUN = work / 'session6_full_submissions'")
before, marker, previous_inputs = initialization.partition('required = ')
if not marker or 'history = pd.read_parquet' not in previous_inputs:
    raise ValueError('Изменился шаблон загрузки Session6')
loader = '''from shutil import copyfileobj


def unique_path(paths, label):
    if len(paths) != 1:
        raise ValueError(f'{label}: ожидался один путь, найдено {len(paths)}: {paths}')
    return paths[0]


def dataset_mount(slug):
    normalized = slug.replace('_', '-').lower()
    matches = [path for path in search.rglob('*') if path.is_dir() and path.name.replace('_', '-').lower() == normalized]
    return unique_path(matches, slug) if matches else None


def dataset_files(slug, filenames, archive_name=None, mount=None):
    mount = mount if mount is not None else dataset_mount(slug)
    resolved = {}
    if mount is not None:
        for filename in filenames:
            matches = [path for path in mount.rglob(filename) if path.is_file()]
            if matches:
                resolved[filename] = unique_path(matches, f'{slug}/{filename}')
                print(f'{slug}/{filename} <- {resolved[filename]}')
    missing = [filename for filename in filenames if filename not in resolved]
    if not missing:
        return resolved
    if archive_name is None:
        raise FileNotFoundError(f'В датасете {slug} отсутствуют файлы {missing}')
    archives = list((mount or search).rglob(archive_name))
    if not archives and mount is not None:
        archives = list(mount.rglob('*.zip'))
    if not archives:
        raise FileNotFoundError(f'Подключите датасет {slug} с файлами {missing} или архивом {archive_name}')
    archive_path = unique_path(archives, f'ZIP датасета {slug}')
    destination = Path(temporary.name) / slug
    destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for filename in missing:
            members = [item for item in archive.infolist() if not item.is_dir() and Path(item.filename).name == filename]
            member = unique_path(members, f'{archive_path.name}/{filename}')
            target_path = destination / filename
            with archive.open(member) as source, target_path.open('wb') as output:
                copyfileobj(source, output)
            resolved[filename] = target_path
            print(f'{slug}/{filename} <- {archive_path}/{member.filename}')
    return resolved


temporary = TemporaryDirectory()
data_names = ('train_solo_track.parquet', 'test_solo_track.parquet')
case_mount = dataset_mount('case_rwb')
cleaning_names = ('comparison_meta.json',) if case_mount is not None else ('comparison_meta.json', *data_names)
cleaning_inputs = dataset_files('cleaning_inputs', cleaning_names, 'cleaning_comparison_inputs.zip')
case_inputs = dataset_files('case_rwb', data_names, mount=case_mount) if case_mount is not None else cleaning_inputs
RESIDUAL_INPUTS = dataset_files('s6_residual_inputs', ('selected_evaluation.parquet', 'anchor_features.parquet'),
                                'session6_residual_oof_inputs.zip')
TRAIN_INPUT, TEST_INPUT = (case_inputs[filename] for filename in data_names)
meta = json.loads(cleaning_inputs['comparison_meta.json'].read_text())
'''
initialization = before + loader + 'history = pd.read_parquet' + previous_inputs.split('history = pd.read_parquet', 1)[1]
initialization = initialization.replace("data / 'train_solo_track.parquet'", 'TRAIN_INPUT')
initialization += '''
from catboost import CatBoostRegressor
from catboost.utils import get_gpu_device_count

STATUSES = names[1:]
REFERENCE_SCORES = [
    dict(variant='session_prediction', WAPE=.3178266755292927, relative_bias=-.0013635244235125578, score=.3191901999528053),
    dict(variant='session_global_shape', WAPE=.3177996914357809, relative_bias=-.0013635244235125508, score=.31916321585929347),
    dict(variant='session_clustered_shape', WAPE=.3181329194743572, relative_bias=-.001363524423512551, score=.31949644389786974),
]
'''
add('code', initialization)
parts = [legacy_code(index) for index in (5, 6, 7, 8, 9, 10)]
parts[4] = parts[4].replace('def features(', 'def session_shape_features(')
legacy_settings = legacy_code(11).split('log = []')[0]
parts.extend([legacy_settings, function(legacy_code(13), 'residual_features'),
              function(''.join(clustering['cells'][2]['source']), 'segment_routes'),
              function(''.join(clustering['cells'][4]['source']), 'normalize_shape')])
for part in parts:
    add('code', ast.unparse(ast.parse(part)))
add('code', runtime)
add('code', 'run_full_submissions()\n')
metadata = {key: clustering['metadata'][key] for key in ('kernelspec', 'language_info') if key in clustering['metadata']}
output = root / 'notebooks/clustering/session6_cluster_submissions.ipynb'
output.write_text(json.dumps(dict(nbformat=4, nbformat_minor=5, metadata=metadata, cells=cells), ensure_ascii=False, indent=1))

input_dir = root / 'notebooks/stepa/stepa_checkpoint_0328_bundle/inputs'
output_archive = root / 'best/session6_residual_oof_inputs.zip'
with zipfile.ZipFile(output_archive, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for filename in ('selected_evaluation.parquet', 'anchor_features.parquet'):
        archive.write(input_dir / filename, filename)
print(output)
print(output_archive)
