# Запуск восстановленного Session6 и бленда на сервере

Используйте `session6_recovered_blend_server.ipynb`. Проверена структура ноутбука и синтаксис встроенного кода. Полное GPU-обучение этой версии ещё не выполнялось. Лучший ранее проверенный рецепт бленда — cat_d4__ridge с локальным score 0.317416. Повторение этого score после восстановления весов требует проверки.

## 1. Подготовка файлов

На сервере нужны пять распакованных комплектов. Вложенные папки следует привести к следующей структуре:

```text
/root/experiments/blend_inputs/
  Session6_exact/
    ANSWERS_RU.md
    session6_bundle/
    stepa_checkpoint_0328_bundle/
    ...
  Session6_recovered_h4/
    evaluation/h4/model.cbm
    ...
  blend_best_output/
    inputs.json
    frozen_selection.json
    base_oof.parquet
    test_predictions.parquet
    train_test_metrics.csv
    ...
  session6/
    comparison_meta.json
    train_solo_track.parquet
    test_solo_track.parquet
    validation_predictions.parquet
  xgb_rich/
    manifest.json
    protocol.json
    evaluation_complete.json
    oof_week_0_h1_xgb_k8.parquet
    ...
```

Архив Session6_exact уже находится на сервере в `/root/experiments/Session6_exact_20261111.zip`. Остальные архивы и ноутбук перенесите с Mac. В переменной окружения `SERVER_USER_HOST` укажите пользователя и адрес вашего сервера. Запускайте эти команды на Mac, подставив свой SSH-порт, если он отличается от 22:

```bash
scp ~/Downloads/Session6_recovered_h4.zip "$SERVER_USER_HOST":/root/experiments/
scp stacking/blending/session6_recovered_blend_server.ipynb "$SERVER_USER_HOST":/root/experiments/
scp stacking/blending/session6_blend_input.zip "$SERVER_USER_HOST":/root/experiments/
scp stacking/blending/xgb_rich_k8_blend_input.zip "$SERVER_USER_HOST":/root/experiments/
```

Также загрузите полный output исходного blend-best-2 — `best_blend_bf2ef3a96ffc.zip`. Этот файл берётся из Kaggle Output. Скачанный ipynb его не заменяет.

## 2. Распаковка на сервере

```bash
ssh "$SERVER_USER_HOST"
cd /root/experiments
mkdir -p blend_inputs/session6 blend_inputs/xgb_rich
unzip -n Session6_exact_20261111.zip -d blend_inputs
unzip -n Session6_recovered_h4.zip -d blend_inputs
unzip -n session6_blend_input.zip -d blend_inputs/session6
unzip -n xgb_rich_k8_blend_input.zip -d blend_inputs/xgb_rich
unzip -n best_blend_bf2ef3a96ffc.zip -d blend_inputs
ln -s best_blend_bf2ef3a96ffc blend_inputs/blend_best_output
```

Проверьте расположение `comparison_meta.json` и `manifest.json`. Если распаковка создала дополнительный вложенный каталог, укажите его в ячейке с `SESSION_DATA_DIR` или `XGB_DATA_DIR` ноутбука.

## 3. Окружение

Активируйте имеющееся GPU-окружение сервера. Ноутбуку нужны CatBoost, LightGBM, XGBoost, CuPy, RAPIDS cuML, NumPy, pandas, scipy, scikit-learn, joblib, PyArrow, IPython и Jupyter. Обычного окружения из архива Session6 недостаточно для GPU Random Forest из бленда.

```bash
nvidia-smi
python - <<'PY'
import sys
import numpy, pandas, scipy, sklearn, joblib, pyarrow
import catboost, lightgbm, xgboost, cupy, cuml
from catboost.utils import get_gpu_device_count
print(sys.executable)
print('CUDA devices', cupy.cuda.runtime.getDeviceCount())
print('CatBoost GPUs', get_gpu_device_count())
print('XGBoost', xgboost.__version__)
print('cuML', cuml.__version__)
PY
```

Если импорт `cuml` или `cupy` не проходит, сначала требуется совместимое CUDA/RAPIDS-окружение. Версии зависят от Python и драйвера сервера. Сохранённые модели могут требовать версии библиотек из `Session6_exact/reports/environment.json`. Не обновляйте рабочее окружение сервера целиком без проверки совместимости.

Установите средства выполнения в выбранное окружение при их отсутствии:

```bash
python -m pip install nbformat nbclient ipykernel jupyter-client
python -m ipykernel install --user --name blend-server --display-name 'Blend server'
```

## 4. Запуск с сохранением вывода

Сохраните рядом с ноутбуком файл `run_blend.py` со следующим кодом:

```python
from pathlib import Path
import nbformat
from nbclient import NotebookClient

source = Path('/root/experiments/session6_recovered_blend_server.ipynb')
result = source.with_name('session6_recovered_blend_server.executed.ipynb')
notebook = nbformat.read(source, as_version=4)
client = NotebookClient(notebook, timeout=None, kernel_name='blend-server', resources={'metadata': {'path': str(source.parent)}})
try:
    client.execute()
finally:
    nbformat.write(notebook, result)
```

Затем:

```bash
cd /root/experiments
export BLEND_INPUT=/root/experiments/blend_inputs
export BLEND_WORK=/root/experiments/blend_session6_server
nohup python -u run_blend.py > blend_server.log 2>&1 &
echo $! > blend_server.pid
```

Процесс продолжит работать после закрытия SSH. `nbclient` сохраняет вывод ячеек в выполненном ноутбуке после завершения или исключения. Лог оболочки может оставаться коротким во время обучения. Для наблюдения используйте:

```bash
tail -n 50 /root/experiments/blend_server.log
nvidia-smi
find /root/experiments/blend_session6_server -name '*.parquet' -o -name '*.csv'
```

## 5. Последовательность и результаты

1. Проверяется исходный архивный Session6, включая построчное совпадение исторического локального test и конкурсного CSV.
2. Загружаются оценочные CatBoost h1/h3/h4/h7. Восстанавливаются h2/h5/h6/h8 по исходному рецепту, без повторного подбора по test.
3. Восстанавливаются поправки Session6, обучаются остальные базы до 18 октября и метамодель на прежних OOF до test.
4. Новые прогнозы локального test оцениваются. Затем те же обученные веса делают первый сабмит.
5. После сохранения оценки и первого CSV запускается обучение на всей истории. Для моделей второго уровня используются OOS первого этапа.

Файлы первого и второго сабмитов:

```text
/root/experiments/blend_session6_server/submission_train_val.csv
/root/experiments/blend_session6_server/submission_full_history.csv
```

Внутри `blend_submit_*/train_val/` смотрите `train_test_metrics.csv`, `evaluation_complete.json`, `test_predictions.parquet`. В `session6_restored/` находятся `base_recovery_comparison.csv` и `session6_reference_comparison.parquet`. Нулевая разница по сохранённым весам ожидается, расхождение восстановленных весов измеряется фактически. Новый score нельзя выдавать за исходные 0.317416 до проверки.

Исторический Session6 оставлял LightGBM86 замороженным перед конкурсом. Этот ноутбук сохраняет это правило, поэтому во втором этапе переобучаются CatBoost Session6, модели остатков, остальные базы бленда и метамодель, а LightGBM86 применяется из исходного комплекта.

Независимая от финального переобучения оценка test получена до переобучения. Период test уже изучался в предыдущих экспериментах.

## 6. Скачать результат

На Mac:

```bash
scp "$SERVER_USER_HOST":/root/experiments/blend_session6_server/submission_train_val.csv ~/Downloads/
scp "$SERVER_USER_HOST":/root/experiments/blend_session6_server/submission_full_history.csv ~/Downloads/
scp "$SERVER_USER_HOST":/root/experiments/session6_recovered_blend_server.executed.ipynb ~/Downloads/
```
