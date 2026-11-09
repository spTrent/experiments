import ast
import hashlib
import json
import pprint
import textwrap
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE / 'scripts' / 'stacking'

def template(name):
    return (SOURCE/name).read_text()

def route_backend():
    notebook = json.loads((BASE/'ridge_by_route_horizon.ipynb').read_text())
    source = ''.join(notebook['cells'][9]['source'])
    names = {'to_host','synchronize','linalg_guard','totals_to_metrics','DeviceBatch'}
    return '\n\n'.join(ast.get_source_segment(source,node) for node in ast.parse(source).body if getattr(node,'name','') in names)

def frozen():
    result = {}
    for directory in sorted((BASE/'artifacts').glob('*/model_params')):
        params = json.loads((directory/'selection_frozen.json').read_text())
        for h in range(1,9):
            best = json.loads((directory/f'best_h{h}.json').read_text())
            if best!=params[str(h)] or best['horizon']!=h or best['residual']!=directory.parent.name.endswith('_residual'):
                raise ValueError(f'Несовпадение параметров {directory}, h{h}')
        result[directory.parent.name] = params
    if not result:
        result = json.loads((SOURCE/'model_params.json').read_text())
    if len(result)!=10 or any(set(v)!=set(map(str,range(1,9))) for v in result.values()):
        raise ValueError('Ожидается 10 семейств конфигураций')
    return result

def code(source):
    return dict(cell_type='code',metadata={},source=source.strip().splitlines(True),execution_count=None,outputs=[])

def markdown(source):
    return dict(cell_type='markdown',metadata={},source=source.strip().splitlines(True))

def build():
    output_dir = BASE/'stacking'
    output_dir.mkdir(exist_ok=True)
    features = (BASE/'scripts/five_starts_features.py').read_text()
    panel = features.split('dates = times[')[0]
    features = features[features.index('target = values[TARGET]'):features.index('def write_horizon')]
    route = route_backend()
    config = frozen()
    sources = [template('runtime.py'),panel,features,template('bases.py'),route,
               template('routes.py'),template('worker.py'),template('calibration.py')+'\n'+template('meta.py')]
    execute = """if MODE not in ('oof','assemble','evaluate','submit') or ROLE not in ('worker','collect'):
    raise ValueError('Проверьте MODE и ROLE')
if MODE=='assemble' or ROLE=='collect':
    if PART!=1 or MODE=='oof':
        raise ValueError('Сборку выполняет первая часть в assemble/evaluate/submit')
    {'assemble':assemble,'evaluate':evaluate,'submit':submit}[MODE]()
else:
    run_worker()
"""
    digest = hashlib.sha256((''.join(sources)+execute+json.dumps(config,sort_keys=True)).encode()).hexdigest()
    for part,horizons in enumerate([(1,8),(2,7),(3,6),(4,5)],1):
        subset = {k:{str(h):v[str(h)] for h in horizons} for k,v in config.items()}
        setup = f"""from time import monotonic
SESSION_START = monotonic()
SESSION_HOURS = 11.0
PROFILE = 'five_starts'
ROUTE_ALPHA = 100.0
ENGINE = []
PART = {part}
HORIZONS = {horizons}
MODE = 'oof'
ROLE = 'worker'
OOF_WEEKS = (2, 3, 4)
INPUT_DIR = '/kaggle/input'
OUTPUT_DIR = '/kaggle/working'
CODE_ID = '{digest}'
PARAMS = {pprint.pformat(subset,width=100)}
"""
        intro = f"""# Стекинг, часть {part} · h{horizons[0]} и h{horizons[1]}

Включите NVIDIA GPU и подключите исходные train/test parquet. В GPU-окружении нужны CuPy, cuML, XGBoost, LightGBM, CatBoost, sklearn и PyArrow. Для сравнения метамоделей нужны также SciPy и PyTorch. Запустите Save & Run All.

Первый запуск: PROFILE='five_starts'. Обучающие старты — 09:30, 10:00, 10:30, 11:00, 11:30 по всей доступной истории train, без отсечения первых 25% дат. Признаки горизонта строятся только для этих пяти стартов; полная получасовая история остаётся доступной для лагов. Дополнительный профиль PROFILE='recent75' сохраняет вариант с последними 75% дат и всеми получасами. Результаты разных профилей изолированы. OOF_WEEKS=(2,3,4): 27 сентября – 3 октября, 4–10 октября и 11–17 октября 2025 года. Число OOF-строк — 168000, прогнозных признаков — 12. После добавления двух октябрьских недель финальная метамодель обучается на 280000 строках.

Параметры глобальных моделей зафиксированы. Маршрутные Ridge используют ROUTE_ALPHA=100 на стандартизованных признаках, без подбора. Метамодели сравниваются на двух временных проверках: OOF-2 → OOF-3 и OOF-2+3 → OOF-4.

SESSION_HOURS=11 ограничивает время вычислений одной сессии. При остановке сохраните output и подключите сохранённые выходы целиком как Kaggle inputs. Незавершённый fit и временные признаки вычисляются заново. Для короткого первого запуска задайте OOF_WEEKS=(2,), затем подключите результат и верните (2,3,4). Выходы прежних версий с другим идентификатором кода не следует смешивать с новым запуском.

1. Во всех четырёх частях выполните `MODE='oof'`, `ROLE='worker'`.
2. В части 1 подключите четыре результата, выберите `MODE='assemble'` и выполните Save & Run All. Получите OOF, результаты сравнения и winner.json. Для победителя среди исходных ансамблей сохраняются общие поправки 50% и 100% по базовым колонкам и итоговому прогнозу. Дополнительно проверяются семь калибровок итогового прогноза с силой 50%/100%: по горизонту, группам маршрутов, иерархическая, сдвиг, аффинная, монотонная нелинейная и ситуационная. Всего 32 кандидата с учётом 14 исходных конфигураций. Новые калибровки обучаются только на предыдущих отложенных прогнозах метамодели; на первой проверочной неделе поправка нейтральная. Базы повторно не обучаются. Сохраняются времена проверок и история калибровки; передавайте весь output assemble в следующие этапы.
3. Во всех четырёх частях подключите победителя и свои OOF-выходы, выполните `MODE='evaluate'`, `ROLE='worker'`. В части 1 подключите все четыре результата оценки и вывод assemble, выполните `MODE='evaluate'`, `ROLE='collect'`.
4. Во всех четырёх частях подключите отчёт оценки, победителя и свои OOF-выходы, выполните `MODE='submit'`, `ROLE='worker'`. В части 1 подключите четыре результата этого этапа и выходы assemble и evaluate/collect, установите `MODE='submit'`, `ROLE='collect'`. В каталоге stack появится submission.csv.

Для recent75 четверть train определяется по датам текущего фолда. Для five_starts ранние даты не исключаются. Все оценки используют старт 10:30. Независимая оценка test сохраняется до финального переобучения. Время и метрика конкурса выводятся по ходу запуска.
"""
        cells = [markdown(intro),code(setup),code(sources[0]),code('initialize()'),
                 code(panel),code("target = values[TARGET]\nif MODE != 'assemble' and ROLE != 'collect':\n"+textwrap.indent(features,'    ')),code(sources[3]),code('from contextlib import nullcontext\n'+route),
                 code(sources[5]),code(sources[6])]
        if part==1:
            cells.append(code(sources[7]))
        cells.append(code(execute))
        for cell in cells[2:]:
            source = ''.join(cell['source'])
            delimiter = "'" * 3
            if delimiter in source:
                raise ValueError('Конфликт разделителя в исходнике')
            cell.update(code(f"ENGINE.append(r{delimiter}\n{source}\n{delimiter})"))
        cells.append(code(template('session.py')))
        for i,cell in enumerate(cells):
            cell['id'] = f'stack-{i:02}'
            if cell['cell_type']=='code':
                compile(''.join(cell['source']),f'part{part}-cell{i}','exec')
        notebook = dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',language='python',display_name='Python 3')),cells=cells)
        (output_dir/f'stacking_part_{part}.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')

if __name__=='__main__':
    build()
