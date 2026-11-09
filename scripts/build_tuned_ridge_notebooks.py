import ast
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE/'scripts/tuned_ridge'
DEST = BASE/'stacking/tuned_ridge'
DEST.mkdir(parents=True,exist_ok=True)


def read_part(part):
    nb = json.loads((BASE/f'stacking/stacking_part_{part}.ipynb').read_text())
    setup = {}
    exec(''.join(nb['cells'][1]['source']),setup)
    return nb,setup


def unpack(cell):
    node = ast.parse(''.join(cell['source'])).body[0].value
    return ast.literal_eval(node.args[0])


def make_cell(kind,source,index):
    if kind=='code':
        compile(source,f'cell{index}','exec')
    item = dict(cell_type=kind,metadata={},source=source.strip().splitlines(True),id=f'tuned-{index:02}')
    if kind=='code':
        item.update(execution_count=None,outputs=[])
    return item


def write_nb(path,entries,metadata):
    cells = [make_cell(kind,source,i) for i,(kind,source) in enumerate(entries)]
    path.write_text(json.dumps(dict(nbformat=4,nbformat_minor=5,metadata=metadata,cells=cells),ensure_ascii=False,indent=1)+'\n')
    print(path)


nb,setup = read_part(1)
runtime,panel,features,bases,backend = [unpack(nb['cells'][i]) for i in (2,4,5,6,7)]
routes = unpack(nb['cells'][8])
route_tree = ast.parse(routes)
route_helpers = '\n\n'.join(ast.get_source_segment(routes,n) for n in route_tree.body if getattr(n,'name','') in ('route_features','route_item'))
base_tree = ast.parse(bases)
horizon = ast.get_source_segment(bases,next(n for n in base_tree.body if getattr(n,'name','')=='HorizonData'))
horizon = horizon.replace(',32)',',8)').replace('start:start+32','start:start+8').replace(',64)',',16)').replace('start:start+64','start:start+16')
local = (SOURCE/'local.py').read_text()
digest = hashlib.sha256((''.join([runtime,panel,features,horizon,backend,route_helpers,local])+setup['CODE_ID']).encode()).hexdigest()
intro = '''# Подбор маршрутных Ridge · Windows · RTX 4060 Laptop

Автономный ноутбук для 16 ГБ RAM и 8 ГБ VRAM. Только Ridge: cuML, XGBoost и CatBoost не нужны.
Положите рядом папку data с train_solo_track.parquet и test_solo_track.parquet. В VS Code выберите Jupyter kernel окружения с CuPy.
Сначала выполните проверку CUDA. Не устанавливайте одновременно несколько вариантов CuPy.

По умолчанию MODE='oof', HORIZONS=(1,…,8), BATCH_SIZE=4, DEVICE='cuda'. Все горизонты идут последовательно.
Для каждой пары маршрут × горизонт выбирается alpha по 30 фиксированным кандидатам от 1e-4 до 1e6 (включая 100)
на неделях 30 августа–5 сентября, 6–12 и 13–19 сентября. Это воспроизводимый перебор, без Optuna.
Критерий — средняя недельная WAPE + |Bias|, при нулевом объёме недели — MAE с отметкой.
Train — пять стартов 09:30–11:30 по всей доступной истории, метки строго до соответствующей недели.
После подбора alpha фиксируется и строятся OOF-2, OOF-3, OOF-4 (27 сентября–17 октября).

Прогресс сохраняется после каждого пакета. При остановке повторите Run All с теми же путями и окружением:
готовые пакеты и горизонты будут пропущены, временные признаки могут строиться заново.
Не удаляйте output; cache содержит только временные файлы. Перед сном компьютера остановите расчёт;
для длительного запуска подключите питание и отключите автоматический сон.

После OOF перенесите архив ridge_tuned_*_export.zip в Kaggle Dataset и используйте новые stacking_tuned_part_*.ipynb.
Старые четыре Kaggle-ноутбука продолжают работу без изменений. Новые Ridge добавляются отдельной колонкой.
Позже MODE='evaluate' построит test, MODE='submit' — extra и full; alpha не переизбирается.
После каждого этапа обновите Kaggle Dataset новым архивом целиком, не подключайте несколько версий одновременно.
Выбирайте конфигурацию стекинга до просмотра test. Локальные поздние этапы могут считаться заранее, но не используйте их метрики для выбора.
'''
config = f'''PROFILE = 'five_starts'
MODE = 'oof'
ROLE = 'worker'
DEVICE = 'cuda'
HORIZONS = (1,2,3,4,5,6,7,8)
BATCH_SIZE = 4
INPUT_DIR = './data'
OUTPUT_DIR = './ridge_output'
CACHE_DIR = './ridge_cache'
BASE_CODE_ID = {setup['CODE_ID']!r}
TUNED_CODE_ID = {digest!r}
'''
entries = [('markdown',intro),('code',config),('code',runtime),('code',local),('code','local_initialize()'),
           ('code',panel),('code',features),('code',horizon),('code',backend),('code',route_helpers),('code','run_local()')]
write_nb(BASE/'stacking/ridge_tuned_local.ipynb',entries,nb['metadata'])
bridge = (SOURCE/'bridge.py').read_text()
for part in range(1,5):
    original,settings = read_part(part)
    if settings['CODE_ID']!=setup['CODE_ID']:
        raise ValueError('Несогласованные исходные версии')
    text = ''.join(original['cells'][1]['source']).replace("MODE = 'oof'","MODE = 'submit'").replace("PROFILE = 'recent75'","PROFILE = 'five_starts'")
    bridge_intro = f'''# Подключение подобранных Ridge · часть {part}

Сохраняет CODE_ID и базовые выходы уже запущенной stacking_part_{part}. Сами исходные ноутбуки не изменены.
Подключите исходные данные, все готовые выходы баз и один актуальный экспорт ridge_tuned_*.
PROFILE='five_starts'. OOF баз здесь не запускается.

1. Часть 1: MODE='assemble' — сравнение без Ridge, с фиксированным Ridge, с tuned Ridge и с обоими.
   Полный поиск: 28 исходных конфигураций и 18 вариантов калибровок победителя (46 кандидатов).
2. Все части: MODE='evaluate', ROLE='worker' — досчитать только отсутствующие исходные базы победителя.
   На локальном компьютере выполнить MODE='evaluate' и обновить экспорт Ridge.
3. Часть 1: MODE='evaluate', ROLE='collect' — независимая оценка победителя.
4. Все части: MODE='submit', ROLE='worker' — только full, пять стартов по всей размеченной истории. Для выбранного fixed локальный экспорт не требуется.
5. Часть 1: MODE='submit', ROLE='collect' — submission_tuned.csv.

Новые метамодели сохраняются в meta_tuned; исходный meta не заменяется. На каждом этапе подключайте полный
output предшествующих этапов и исходных частей. Готовые базовые задачи с совпадающим CODE_ID не переобучаются.
Сессия по-прежнему ограничена 11 часами. Обычные Kaggle worker-ноутбуки не читают meta_tuned: после новой сборки используйте эти четыре версии.
'''
    engine = [unpack(c) for c in original['cells'][2:-2]]
    if part!=1:
        engine.append(unpack(nb['cells'][10]))
    engine = [source.replace("path = ROOT/f'meta/{stage}_post_calibration.joblib'", "path = ROOT/profile_path(f'meta/{stage}_post_calibration.joblib')")
              .replace("path = ROOT/'submission.csv'", "path = ROOT/'submission_tuned.csv'")
              .replace("f'baseline_{len(cols)}_{key}'", "f'baseline_{label}_{key}'") for source in engine]
    engine = [source.replace("stages = ['test'] if MODE=='evaluate' else ['extra','full']",
                             "stages = ['test'] if MODE=='evaluate' else ['full']")
              .replace("train = pd.concat([oof,test.loc[test.cutoff<TEST_START+7*DAY],collect_stage('extra',cols)],ignore_index=True)",
                       "train = pd.concat([oof,test],ignore_index=True)") for source in engine]
    engine.append(f'EXPECTED_TUNED_CODE_ID = {digest!r}\n'+bridge)
    engine.append("if MODE=='oof':\n    raise ValueError('OOF уже вычисляется исходными ноутбуками')\n"+unpack(original['cells'][-2]))
    entries = [('markdown',bridge_intro),('code',text)]
    for source in engine:
        compile(source,'bridge','exec')
        entries.append(('code','ENGINE.append('+repr(source)+')'))
    entries.append(('code',''.join(original['cells'][-1]['source'])))
    write_nb(DEST/f'stacking_tuned_part_{part}.ipynb',entries,original['metadata'])
