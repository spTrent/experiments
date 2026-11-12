import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
folder = root / 'scripts/cluster_experiment'
cells = []


def add(kind, text):
    if kind == 'code':
        compile(text, '<cell>', 'exec')
    cell = dict(cell_type=kind, id=f'metric-{len(cells)}', metadata={}, source=(text.strip() + '\n').splitlines(True))
    if kind == 'code':
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)


add('markdown', """# Три эксперимента со структурой бленда

Подключите только полный распакованный **output исходного blend-best-2** через Add Input. Обучение запускает пользователь в Kaggle. CPU достаточно; повторного обучения тяжёлых баз нет.

Контроль — прежний `cat_d4__ridge`. Проверяем 11 заранее заданных кандидатов:

1. **Калибровка горизонтов**: поверх Ridge добавляются относительные поправки h1–h8 с усадкой 0.25, 0.5 или 1.0. Оцениваются только на обучающих прогнозах; ограничены ±3%, итоговый множитель относительно объёма Session6 остаётся в пределах ±5%. Это поправка отношения сумм фактов и прогнозов по горизонту относительно общего отношения, а не отдельный подбор по test.
2. **Распределение между маршрутами**: линейная модель или MLP с 16 нейронами, обученная на MAE после дифференцируемой нормировки суммы к Session6 для каждого старта и горизонта. До нормировки поправка ограничена ±10%. Затем применяется тот же рецепт Ridge-калибровки объёма. Существующий CatBoost также сохраняет объём; здесь сравниваются альтернативные обучаемые модели распределения.
3. **Исключение баз**: по очереди убираем RF, CatBoost rich, XGB rich, XGB clean и profile28. Каждый раз заново учим веса, CatBoost-метамодель и Ridge. Исключённая база удаляется и из метапризнаков, и из признаков калибровки. Session6 остаётся обязательным якорем объёма.

На каждом из двух validation-окон (4–10 и 11–17 октября) обучение использует только более ранние OOF. Печатаются WAPE, Relative Bias и итоговый score на train OOF и val. Выбор — по среднему score двух недель, затем по худшей неделе, затем по заранее заданному порядку с приоритетом контроля.

После фиксации выбора на локальном test 18–31 октября оцениваются только победитель и контроль. Test ранее изучался, поэтому это не новый holdout. Метамодели обучаются на собственных train OOF; их train-score не является независимой оценкой.

Результаты: таблицы validation, рейтинг, зафиксированный выбор, train/test-метрики, построчные прогнозы test, дневные срезы и модели. Сабмит не создаётся. Независимая от полного переобучения оценка test получается до переобучения; перед конкурсным сабмитом требуется полный refit всех выбранных компонентов.
""")
add('code', (folder / 'best_blend_data.py').read_text())
add('code', (folder / 'best_blend_meta.py').read_text())
add('code', (folder / 'blend_metric_experiment.py').read_text().replace('correction_bound=.05, candidates=CANDIDATES', "shape_correction_bound=.10, horizon_correction_bound=.03, total_volume_bound=.05, candidates=CANDIDATES"))
add('code', (folder / 'blend_structure_experiment.py').read_text())
add('code', '''EVALUATION_DIR = None
torch.set_num_threads(4)
EVALUATION = find_previous_run(EVALUATION_DIR)
CANDIDATES = ['cat_d4__ridge', 'horizon_0.25', 'horizon_0.5', 'horizon_1.0', 'shape_linear', 'shape_mlp', *[f'without_{name}' for name in BASES if name != 'session6']]
contract = dict(code=CODE_HASH, oof=digest_file(EVALUATION / 'base_oof.parquet'))
run_id = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()[:12]
ARTIFACTS = Path('/kaggle/working') / f'blend_structure_{run_id}'
ARTIFACTS.mkdir(parents=True, exist_ok=True)
write_json(ARTIFACTS / 'inputs.json', dict(contract, source=str(EVALUATION)))
oof = pd.read_parquet(EVALUATION / 'base_oof.parquet')
verify_frame(oof)
if oof.cutoff.min() != OOF_START + pd.Timedelta(hours=10, minutes=30) or not (oof.cutoff + oof.horizon * STEP).lt(LOCAL_TEST_START).all():
    raise ValueError('Неверный период OOF')
frozen = experiment_select(oof)
''')
add('code', '''if json.loads((ARTIFACTS / 'frozen_selection.json').read_text()) != frozen:
    raise ValueError('Выбор должен быть зафиксирован до test')
test = pd.read_parquet(EVALUATION / 'test_predictions.parquet')
verify_frame(test)
if not test.cutoff.ge(LOCAL_TEST_START).all() or not (test.cutoff + test.horizon * STEP).lt(LOCAL_TEST_END).all():
    raise ValueError('Неверный период test')
experiment_test(oof, test, frozen)
''')
digest = hashlib.sha256(json.dumps(cells, sort_keys=True).encode()).hexdigest()
cells[5]['source'].insert(0, f'CODE_HASH = {digest!r}\n')
output = root / 'stacking/blending/blend_structure_experiment_kaggle.ipynb'
output.write_text(json.dumps(dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python')), cells=cells), ensure_ascii=False, indent=1)+'\n')
print(output)
