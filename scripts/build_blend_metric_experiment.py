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


add('markdown', '''# Метамодель с функцией потерь соревнования

Отдельный эксперимент на готовых прогнозах: контроль CatBoost depth 4 + Ridge против линейной модели и MLP с 16 скрытыми нейронами. Для каждой новой архитектуры сравниваются MAE и WAPE + |Relative Bias|. Функция потерь считается по всему обучающему периоду, со всеми маршрутами и горизонтами; это не средняя дневная метрика. Поправка к взвешенной смеси ограничена ±5%, штраф стягивает её к нулю. Число эпох и настройки фиксированы заранее.

Подключите только **полный распакованный output blend-best-2** через Add Input. Базовые модели повторно не обучаются. Запуск на CPU, GPU не требуется. Обучение запускает пользователь в Kaggle.

Выбор по среднему score на неделях 4–10 и 11–17 октября. Каждому окну предшествует своё обучение. После фиксации выбора на test 18–31 октября оцениваются только выбранный вариант и контроль. Метамодели обучаются на OOF до test. Этот test уже изучался: результат является проверкой гипотезы на прежнем периоде, а не новым holdout.

В таблицах train_oof — метамодель на собственном обучении, val/test — следующий временной период. Сохраняются WAPE, Relative Bias, итоговый score и дневные срезы. Результаты в `/kaggle/working/blend_metric_*`. Сабмит не собирается: сначала проверяем гипотезу. Независимая от полного переобучения оценка test получается до переобучения.
''')
add('code', (folder / 'best_blend_data.py').read_text())
add('code', (folder / 'best_blend_meta.py').read_text())
add('code', (folder / 'blend_metric_experiment.py').read_text())
add('code', '''EVALUATION_DIR = None
torch.set_num_threads(4)
EVALUATION = find_previous_run(EVALUATION_DIR)
CANDIDATES = ['cat_d4__ridge', 'linear_mae', 'linear_metric', 'mlp_mae', 'mlp_metric']
contract = dict(code=CODE_HASH, oof=digest_file(EVALUATION / 'base_oof.parquet'))
run_id = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()[:12]
ARTIFACTS = Path('/kaggle/working') / f'blend_metric_{run_id}'
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
cells[4]['source'].insert(0, f'CODE_HASH = {digest!r}\n')
output = root / 'stacking/blending/blend_metric_experiment_kaggle.ipynb'
output.write_text(json.dumps(dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python')), cells=cells), ensure_ascii=False, indent=1)+'\n')
print(output)
