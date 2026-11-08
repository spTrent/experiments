# Эксперименты: шесть алгоритмов, direct и residual

Подготовлены `xgb_direct/residual.ipynb`, `lightgbm_direct/residual.ipynb`, `random_forest_direct/residual.ipynb`, `ridge_direct/residual.ipynb`, `prophet_direct/residual.ipynb`, `catboost_direct/residual.ipynb`. Каждый ноутбук автономный: загружает исходные parquet, строит признаки, подбирает параметры, оценивает test и создаёт submission. Обучение запускает пользователь. Ноутбуки генерируются из `scripts/five_starts_features.py` и `scripts/five_starts_models.py` командой `python scripts/build_model_notebooks.py`; эти два файла — шаблоны ячеек, отдельно их запускать не нужно.

## Протокол

Сохранены признаки текущих XGBoost, включая окна ±2 часа, сетевые суммы, сезонные профили и корреляционные лаги. Восемь моделей на горизонты 30–240 минут. Пять стартов train: 09:30–11:30 с шагом 30 минут. Оценка и submission стартуют в 10:30. `profile28` и `overlap` не входят в признаки модели. `profile28` служит смещением residual и отдельным ориентиром качества; смесь с ним не применяется к submission.

Размеченная история содержит 1000 маршрутов, 4 630 000 строк с 28 июля по 1 ноября 2025 года, 10:30. Submission содержит 8000 строк на 1 ноября, 11:00–14:30.

Последние 14 полных дней — локальный test. Подбор заканчивается 19 сентября 2025 года. Три окна tuning занимают 30 августа–5 сентября, 6–12 и 13–19 сентября. Optuna читает только строки с временем метки строго раньше 20 сентября. Для OOF с 20 сентября по 17 октября используют сохранённые параметры и заново обучают базы на прошлом каждого окна. Каждому предшествует отдельная неделя val для early stopping бустингов. Train каждого окна находится строго до val; метки тоже проверяются по времени. Random Forest, Ridge и Prophet не используют early stopping. Подбор параметров каждого горизонта минимизирует среднее трёх значений WAPE + |Relative Bias| на tuning. Дополнительно выводится среднее метрики по дням; оно не является целью подбора.

Для каждого fit заполнение пропусков, масштабирование и категории обучаются заново только по его train. Корреляционные лаги пересчитываются до соответствующей границы обучения. В базовых FE-parquet их колонки содержат NaN: обученные значения добавляются только при подготовке конкретного train/val/tuning/test/full. Это исключает ошибочное использование словаря из более позднего окна при повторном чтении базовых файлов. Исторические признаки каждой строки используют только факты до её собственного cutoff. При оценке нескольких последовательных дней факты предыдущего дня могут входить в лаги следующего: это ежедневный прогноз со свежей историей, не единый прогноз сразу на 14 дней.

Во время поиска выводятся train/tuning, test не показывается. После фиксации настроек всех восьми горизонтов модели обучаются на всей доступной истории до test, без early stopping на test. Число деревьев бустингов — медиана лучших чисел по трём val-окнам выбранной конфигурации. Выводятся train/test по горизонтам и общая метрика по восьми горизонтам. В агрегированной train-метрике остаются все пять стартов, в test — только 10:30.

**Независимая оценка test получена до переобучения на всей истории.** Затем все модели и корреляционные лаги переобучаются на полной доступной размеченной истории для submission. Как в исходных XGBoost, обе ветки исключают начальные строки без доступного profile28. Сохранённые модели evaluation и full разделены. Последний период test уже использовался в этом проекте, поэтому он не является новым слепым holdout; повторный выбор решений по нему уменьшает независимость оценки.

## Ресурсы и настройки

Запускайте один ноутбук за раз. По умолчанию `N_TRIALS=12` на горизонт: 288 обучений на окнах плюс 8 оценочных и 8 финальных моделей на ноутбук. Это стартовый объём поиска, не гарантия достижения оптимума. Повышать бюджет следует по tuning, до просмотра test. `N_JOBS` ограничен четырьмя потоками; Optuna выполняет конфигурации последовательно. Параметры GPU: `DEVICE='cuda'`; для CPU измените на `'cpu'`. XGBoost/CatBoost используют GPU, остальные алгоритмы — CPU.

Random Forest ограничен глубиной и числом деревьев, но остаётся тяжёлым на этих данных. Ridge использует стандартизацию, компактное двоичное кодирование маршрутов и решатель lsqr. Random Forest, Ridge и Prophet получают 10 двоичных колонок маршрута вместо 999 one-hot колонок. Код — позиция в отсортированном справочнике маршрутов плюс один; нулевой код зарезервирован для неизвестного маршрута. Справочник строится без таргета, сохраняется в metadata.json и используется одинаково на всех этапах. В линейных Ridge и Prophet такие признаки не позволяют задать независимый коэффициент каждому маршруту. XGBoost, LightGBM и CatBoost сохраняют нативные категории. Сохранение моделей в joblib использует сжатие.

Prophet реализован строго как одна общая модель на горизонт. Все маршруты имеют общий тренд и сезонность; route_id кодируется двоично, остальные признаки становятся дополнительными линейными регрессорами. Это не 1000 независимых временных моделей. Пропуски заполняются медианами train, постоянные регрессоры исключаются внутри fit. Prophet строит плотную матрицу: 16 ГБ могут оказаться недостаточными, и более крупная Kaggle-среда тоже не гарантирует успех. Перед поиском проверяется приблизительный запас памяти для полного refit, дополнительно — перед каждым fit. Оценка консервативная (шесть плотных матриц), фактический пик зависит от версии библиотек; проверка не является гарантией. При недостатке RAM запуск прекращается с понятной ошибкой. Данные автоматически не сокращаются. Удаление one-hot уменьшает ширину матрицы, но не гарантирует, что все временные копии данных поместятся в 16 ГБ.

Каждый ноутбук сохраняет результаты в `artifacts/{model_name}_stack`, например `artifacts/xgb_direct_stack` или `artifacts/prophet_residual_stack`. Внутри находятся ровно шесть рабочих папок: `features`, `importance`, `models`, `postprocessing`, `docs`, `model_params`. Вложенной папки времени запуска нет. Если папка эксперимента уже существует, запуск остановится до записи: перенесите предыдущую папку или измените `MODEL_NAME` в первой ячейке. Частичные метрики, лучшие tuning-прогнозы и Optuna SQLite сохраняются по ходу работы. Автоматического продолжения прерванного обучения нет: повторный запуск начинает новый эксперимент после переноса старой папки или изменения MODEL_NAME. Архив создаётся после независимой оценки и обновляется после финального refit; промежуточные файлы доступны и до упаковки.

LightGBM запускается на CPU по умолчанию. DEVICE='gpu' требует сборки с OpenCL, DEVICE='cuda' — сборки с CUDA. LightGBM использует нативный категориальный route_id, objective='regression', metric='l1', early stopping на val и медиану лучших чисел деревьев трёх окон. Optuna подбирает num_leaves=15–127, learning_rate=0,02–0,15, min_child_samples=20–200, subsample=0,65–1, colsample_bytree=0,5–1, reg_alpha=0,00001–10 и reg_lambda=0,1–100. subsample_freq=1 включает семплирование строк. Итоговая модель сохраняется в model.txt.

Tuning-прогнозы выбранной конфигурации не являются независимым OOF: их параметры выбраны по всем трём tuning-неделям. Поэтому в parquet is_unbiased_oof=False, а model_info_through учитывает последний момент всех данных подбора. Эти прогнозы нельзя использовать как доказательство независимого качества стекинга на тех же датах. Независимая оценка test получена до полного refit; уже просмотренный test не превращается в новый слепой holdout.

## Локальный запуск на ноутбуке

Для отдельного окружения удобно использовать Python 3.11 или 3.12. Основной `pyproject.toml` проекта требует Python 3.14 и содержит другие ограничения, поэтому экспериментальное окружение создаётся независимо через `requirements-experiments.txt`.

Windows PowerShell, из корня репозитория:

```powershell
py -3.12 -m venv .venv-experiments
.\.venv-experiments\Scripts\python.exe -m pip install -r requirements-experiments.txt
.\.venv-experiments\Scripts\python.exe -m jupyterlab
```

Linux/macOS:

```bash
python3.12 -m venv .venv-experiments
.venv-experiments/bin/python -m pip install -r requirements-experiments.txt
.venv-experiments/bin/python -m jupyterlab
```

Положите исходные parquet в `data/`. В Jupyter откройте один ноутбук и выполните ячейки по порядку. На NVIDIA нужен рабочий драйвер; при ошибке GPU проверьте окружение или явно выберите CPU. На Mac для бустингов выберите CPU. Обучение всех двенадцати вариантов не запускается автоматически. Для воспроизведения используйте версии из `manifest.json`: requirements задаёт нижние границы, а не фиксированное окружение.

## Kaggle → основной Mac

1. Создайте Kaggle Notebook, импортируйте нужный `.ipynb` и подключите dataset с двумя исходными parquet. Поиск файлов внутри `/kaggle/input` автоматический; если найдено несколько копий, укажите пути вручную в первой ячейке.
2. Для XGBoost/CatBoost включите GPU. Для остальных важнее CPU и фактическая доступная RAM. Не запускайте несколько экспериментов одновременно в одном процессе.
3. Если пакетов не хватает, установите их в отдельной первой ячейке и перезапустите kernel до основного запуска:

```python
%pip install "scikit-learn>=1.6" "xgboost>=2.1" "catboost>=1.2.7" "lightgbm>=4.0" "prophet>=1.1.6" "optuna>=4" pyarrow joblib psutil
```

4. Выполните **Save Version → Save & Run All** и дождитесь завершения сохранённой версии. Результаты записываются под `/kaggle/working/artifacts/<model_name>_stack/`. Код ноутбука не публикует и не отправляет submission автоматически.
5. В выходных файлах сохранённой версии скачайте `reports.zip` и `results.zip`. Первый содержит компактные отчёты; второй — отчёты, модели, преобразования, прогнозы и submission без тяжёлых воспроизводимых FE-parquet. При необходимости сами FE-parquet скачиваются из папки `features` отдельно.
6. Распакуйте `results.zip` на Mac в `artifacts/<model_name>_stack/`. Для просмотра CSV и JSON обучать модели повторно не нужно. Для загрузки joblib используйте совместимые версии из manifest; нативные XGBoost/CatBoost/LightGBM и Prophet JSON также сохранены.

Можно скачать весь вывод сохранённого ноутбука через официальный Kaggle CLI после настройки авторизации:

```bash
python3 -m pip install kaggle
kaggle kernels output USER/NOTEBOOK-SLUG -p artifacts/kaggle-download
```

`USER/NOTEBOOK-SLUG` берётся из ссылки вашего ноутбука. Команда скачивает выходные файлы, включая FE-parquet, поэтому для экономии трафика удобнее отдельно скачать архивы в интерфейсе. Токены Kaggle в Git не добавляйте.

Официальная справка: [Kaggle kernels output](https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/kernels.md).

## Что переносить через Git

`artifacts/` уже исключён из Git. После завершения эксперимента распакуйте только `reports.zip` в `results/<model_name>_stack/`, затем добавьте эту конкретную папку в Git. Через Git переносятся ноутбуки, параметры, метрики, таблицы поиска, важности и manifest. Большие модели, прогнозы и исходные данные переносите архивом через скачивание из Kaggle, диск или файловую синхронизацию. `results.zip` и FE-parquet не нужно коммитить.

Структура результатов на примере XGBoost direct:

```text
artifacts/xgb_direct_stack/
    features/
        history_h1.parquet … history_h8.parquet
        submission_h1.parquet … submission_h8.parquet
    importance/
        importance_h1.csv … importance_h8.csv
    models/
        evaluation/h1/ … evaluation/h8/
        full/h1/ … full/h8/
    postprocessing/
        search.sqlite3
        tuning_predictions_h1.parquet … tuning_predictions_h8.parquet
        train_predictions_h1.parquet … train_predictions_h8.parquet
        test_predictions_h1.parquet … test_predictions_h8.parquet
        test_predictions.parquet
    docs/
        feature_importance.csv
        feature_pipeline.json
        metrics.csv
        profile_control.csv
        status6_lags_validation.csv
        status6_lags_fold_1.csv … status6_lags_fold_3.csv
        status6_lags_evaluation.csv
        status6_lags_full.csv
        submission.csv
        independent_metrics.csv
        independent_metrics_by_horizon.csv
        trials_h1.csv … trials_h8.csv
        manifest.json
        evaluation_note.txt
    model_params/
        best_h1.json … best_h8.json
        selection_frozen.json
    reports.zip
    results.zip
```

`model_name` включает алгоритм и вариант цели, поэтому direct и residual не перезаписывают друг друга. `model_params/best_hN.json` содержит лучшие параметры Optuna для горизонта N, номер trial, tuning-метрики и выбранное число деревьев для бустингов. `selection_frozen.json` объединяет выбор всех восьми горизонтов.

`docs/feature_importance.csv` содержит колонки h1–h8 и mean_all_8. Для деревьев это нативные важности, для Ridge и Prophet — модули коэффициентов. Вид показателя записан в importance_kind; значения разных алгоритмов не являются одной универсальной шкалой важности. У Prophet сюда входят только дополнительные регрессоры, без тренда и сезонных компонент. Подробные таблицы по горизонтам лежат в importance; у Ridge и Prophet это coefficients_hN.csv с подписанными коэффициентами.

`docs/profile_control.csv` сравнивает независимую оценочную модель с profile28 на train/test. Он не подбирает вес смешивания и не меняет submission.

`reports.zip` сохраняет папки docs, model_params и importance с той же структурой. `results.zip` содержит все шесть рабочих папок, кроме воспроизводимых FE-parquet из features; пустые папки в ZIP не добавляются. Сами признаки остаются в features внутри папки эксперимента и доступны для отдельного скачивания. Архивы обновляются после независимой оценки и после полного refit; feature_importance.csv и submission.csv появляются после полного refit.

Официальная документация моделей: [регрессоры Prophet](https://meta.github.io/prophet/docs/seasonality%2C_holiday_effects%2C_and_regressors.html), [XGBoost API](https://xgboost.readthedocs.io/en/stable/python/python_api.html), [CatBoost fit](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor_fit), [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html).

Проверки и ограничения: [отчёт аудита](model_pipeline_audit.md).
