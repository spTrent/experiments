import ast
import copy
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
template = json.loads((BASE / 'notebooks/clustering/nightly_rich_gpu.ipynb').read_text())
extension = (BASE / 'scripts/cluster_experiment/rich_calibrations.py').read_text()


def replace_assignment(source, name, value):
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)]
    for node in reversed(nodes):
        source = source.replace(ast.get_source_segment(source, node), name + ' = ' + repr(value))
    return source


for algorithm in ('cat', 'xgb'):
    notebook = copy.deepcopy(template)
    parameters, provenance = {}, {}
    for h in range(1, 9):
        relative = f'artifacts/catboost_direct/model_params/best_h{h}.json' if algorithm == 'cat' else f'notebooks/xgb_tuning/best_params/best_h{h}.json'
        path = BASE / relative
        item = json.loads(path.read_text())
        if algorithm == 'cat':
            params = dict(item['params'])
            params['depth'] = min(params['depth'], 6)
            trees = min(item['trees'], 900)
        else:
            params = dict(item['refit_params'])
            trees = params.pop('n_estimators')
        parameters[str(h)] = dict(params=params, trees=int(trees))
        provenance[str(h)] = dict(path=relative, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    name = 'catboost_rich_calibration' if algorithm == 'cat' else 'xgboost_rich_calibration'
    family = 'selected_cat' if algorithm == 'cat' else 'selected_xgb'
    experiment = dict(name=name, algorithm=algorithm, residual=False, parameter_set=family,
                      loss='MAE' if algorithm == 'cat' else 'reg:absoluteerror', compact=False)
    config = ''.join(notebook['cells'][1]['source'])
    config = replace_assignment(config, 'PARAMETERS', {family: parameters})
    config = replace_assignment(config, 'EXPERIMENTS', [experiment])
    config = replace_assignment(config, 'CREATE_ARCHIVE', True)
    tree = ast.parse(config)
    for node in reversed(tree.body):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('STOP_NEW_FITS_UTC', 'CODE_HASH') for t in node.targets):
            config = config.replace(ast.get_source_segment(config, node), '')
    if algorithm == 'xgb':
        node = next(node for node in ast.parse(config).body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'RICH_TRAINING_SOURCE' for t in node.targets))
        training = ast.literal_eval(node.value).replace('enable_categorical=True, max_cat_to_onehot=1)', "enable_categorical=True, max_cat_to_onehot=1, eval_metric='mae')")
        config = replace_assignment(config, 'RICH_TRAINING_SOURCE', training)
    config += '\nAFFINE_SLOPE_BOUNDS = (0.5, 1.5)\nPARAMETER_SOURCES = ' + repr(provenance) + '\n'
    phase = ''.join(notebook['cells'][3]['source'])
    anchor = "                    report.append(dict(variant=name, h=h, strategy='selected', period=period, **metrics(data['y'], pred)))"
    addition = "\n                    for strategy, candidate in frozen['comparison_factors'][name].items():\n                        if strategy == 'raw':\n                            continue\n                        candidate_pred = apply_factors(raw, h, data['route_index'], candidate)\n                        report.append(dict(variant=name, h=h, strategy=strategy, period=period, **metrics(data['y'], candidate_pred)))"
    if phase.count(anchor) != 1:
        raise ValueError('Не найден участок сравнения калибровок')
    phase = phase.replace(anchor, anchor + addition)
    notebook['cells'][3]['source'] = phase.splitlines(True)
    runtime = ''.join(notebook['cells'][4]['source'])
    for node in reversed(ast.parse(runtime).body):
        if isinstance(node, ast.FunctionDef) and node.name in ('run_nightly', 'check_deadline'):
            runtime = runtime.replace(ast.get_source_segment(runtime, node), '')
    notebook['cells'][4]['source'] = runtime.splitlines(True)
    config += '\nCODE_HASH = ' + repr(hashlib.sha256((config + phase + runtime + extension + ''.join(notebook['cells'][5]['source'])).encode()).hexdigest()) + '\n'
    notebook['cells'][1]['source'] = config.splitlines(True)
    notebook['cells'].insert(-1, dict(cell_type='code', id='calibration-comparison', metadata={}, source=extension.splitlines(True), execution_count=None, outputs=[]))
    notebook['cells'][-1]['source'] = 'STEP = pd.Timedelta(minutes=30)\nrun_rich_calibration()\n'.splitlines(True)
    detail = 'CatBoost: точная конфигурация rich-запуска с WAPE0.31815 — параметры artifacts/catboost_direct по горизонтам, глубина не выше6, деревьев не больше900, MAE. Эти ограничения уже применены в таблице PARAMETERS; дополнительных ограничений при fit нет.' if algorithm == 'cat' else 'XGBoost: refit_params каждого горизонта из notebooks/xgb_tuning/best_params/best_h1.json…best_h8.json. Используются selected_trees, без ограничения глубины и числа деревьев. MAE, CUDA hist, категориальный маршрут. Параметры переносятся в rich-признаки и нормировку по маршрутам; исходный результат XGB tuning этим не воспроизводится.'
    text = f'''# {name}

{detail}

Самодостаточный Kaggle-ноутбук: подключить только case_rwb, включить GPU T4×2 и Run All. Источники параметров и их SHA-256 встроены. Обучение запускает пользователь. Нет ночного дедлайна или CPU fallback. 216 rich-признаков плюс категориальный ID маршрута, все48 получасов обучения, отдельная модель на каждый из8 горизонтов, варианты общей модели и8 кластеров. Размер малой группы меньше10 маршрутов использует общую fallback-модель.

История делится по времени: train, val14дней, test14дней. В первой неделе val обучаются коэффициенты калибровки; во второй выбирается способ по WAPE + |Relative Bias|. Базы для второго блока переобучаются на ранее доступной истории. Все18 способов затем переоценивают коэффициенты на val14 и замораживаются до test. После каждого fit выводятся доступные train/проверочные метрики; после фиксации процедуры — train/test для всех способов. Печатается единая таблица all_calibrations_train_test.csv. Сравнение способов на test служит отчётности: победитель остаётся выбранным по val.

Способы: raw (без множителя и сдвига); заданные постоянные множители1.025/1.05; общий обученный множитель; множители по горизонту, группе и группе×горизонту, со сглаживанием50% и без; общий множитель с давностью2/3/7дней; только аддитивный сдвиг; affine a×prediction+b — общая, по горизонтам, по группам, групповая со сглаживанием. Affine обучается по WAPE + |Relative Bias| с неотрицательным прогнозом и ограничением наклона0.5…1.5. Для малых групп используется глобальная поправка. Коэффициенты постоянны на test, будущие метки test в калибровку не поступают. Группы для калибровки зафиксированы до val.

Перед submission все выбранные базовые модели и обучаемые статистики переобучаются на всей доступной размеченной истории. Финальные коэффициенты обновляются на доступных OOF val + бывший test без повторного выбора способа. Независимая оценка test получена до полного переобучения. Этот test уже изучался в проекте. Сохраняются raw и calibrated submission, модели, метрики, коэффициенты, параметры и ZIP. Эти два ноутбука не запускались в Kaggle при подготовке.
'''
    notebook['cells'][0]['source'] = text.splitlines(True)
    parameter_note = dict(cell_type='code', id='show-effective-params', metadata={}, execution_count=None, outputs=[], source="display(pd.DataFrame([dict(h=int(h), trees=p['trees'], **p['params']) for h, p in PARAMETERS[EXPERIMENTS[0]['parameter_set']].items()]))\n".splitlines(True))
    notebook['cells'].insert(3, parameter_note)
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            code = ''.join(cell['source'])
            compile(code, name, 'exec')
            for node in ast.walk(ast.parse(code)):
                if isinstance(node, ast.Assert):
                    raise ValueError('assert запрещён')
    output = BASE / 'notebooks/clustering' / (name + '.ipynb')
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1))
    print(output)
