"""Build a standalone Kaggle baseline from the verified stepa feature definitions."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'stepa' / 'stepa330.ipynb'


def build():
    source = json.loads(SOURCE.read_text())
    cell = lambda i: ''.join(source['cells'][i]['source'])
    config = (ROOT / 'scripts/baseline/config.py').read_text()
    load = cell(3).replace('if len(dates) <= 28:', 'if len(dates) <= 42:')
    features = cell(5)
    correlations = cell(7).split('\nvalidation_lag_table =')[0]
    horizon = cell(9).split('\ndef write_horizon')[0]
    start = horizon.index('    if h == 6:')
    stop = horizon.index('    frame = add_correlation_features', start)
    horizon = horizon[:start] + horizon[stop:]
    horizon = horizon.replace('    frame = add_correlation_features(frame, h, validation_lag_table)\n',
                              '    for name in CORRELATION_COLUMNS:\n        frame[name] = np.float32(np.nan)\n')
    runtime = (ROOT / 'scripts/baseline/runtime.py').read_text()
    report = (ROOT / 'scripts/baseline/report.py').read_text()
    aggregate = '''network_history = pd.DataFrame({'timestamp': times})
for name, array in values.items():
    network_history[name] = array.sum(axis=1, dtype=np.float64)
network_history.to_parquet(RUN_ROOT / 'network_history.parquet', index=False)
del network_history
'''
    bodies = [config, load, features, correlations, horizon, aggregate, runtime, 'run_all()\n', report]
    fingerprint = hashlib.sha256('\n'.join(bodies).encode()).hexdigest()
    bodies[0] = config.replace('__PIPELINE_HASH__', fingerprint)
    descriptions = [
        '''# 00 — контроль stepa330 и прогнозы для следующих экспериментов

Автономный Kaggle Notebook. Включите **GPU** и подключите dataset с
`train_solo_track.parquet` и `test_solo_track.parquet`. Репозиторий загружать не нужно.
Зависимости: numpy, pandas, pyarrow, catboost (обычно уже доступны в Kaggle).
Если import не проходит: `%pip install numpy pandas pyarrow catboost`, затем restart kernel.

Контроль: recent75, CatBoost depth=7 / learning_rate=0.05 / MAE, цель /100000.
h1 — CatBoost; h2–h8 — 70% CatBoost + 30% profile28; общая аддитивная поправка.
Настройки уже выбраны в проекте: здесь нет повторного поиска победителя.
Старые ноутбуки не изменяются. GPU-обучение не гарантирует побитового совпадения.

**Две отдельные проверки:** legacy повторяет прежние границы и early stopping;
weekly строит шесть недель с обновлением модели перед каждой неделей и фиксированными
1500 деревьями. Последующие недели не участвуют в fit или early stopping weekly.
Эти параметры уже выбирались на истории проекта: прогнозы причинны по fit,
но не являются новым слепым экспериментом по выбору параметров.

По умолчанию 64 fit: 8 legacy + 48 weekly + 8 full. Один GPU на fit;
`GPU_DEVICES='0:1'` можно задать для двух GPU, ускорение нужно измерить.
Не запускайте два экземпляра в одну RUN_ROOT.

**Продолжение:** в той же файловой системе просто Run All. После каждой модели
сохраняются checkpoint и results.zip; после горизонта — checkpoint.zip со всеми моделями.
После штатной ошибки обучения checkpoint.zip также пересобирается. При принудительном
завершении сессии доступен последний полностью сохранённый архив.
Для новой сессии сохраните checkpoint.zip, распакуйте его в `/kaggle/working`
(он содержит `baseline_00/`) и запустите с той же RUN_ROOT.
Альтернатива: подключите распакованный checkpoint как dataset и укажите RESTORE_DIR
на папку с manifest.json; RUN_ROOT должна ещё отсутствовать.
Версии библиотек, хеши данных и кода проверяются, несовместимый запуск не продолжится.
Незавершённый fit запускается заново; уже завершённые модели пропускаются.
''',
        '## Данные и границы\nПолная панель проверяется до создания признаков.',
        '## Исторические признаки\nФормулы взяты из stepa330; pilot-признаки не используются контрольной моделью.',
        '## Корреляционные лаги\nОтдельный словарь для каждой границы обучения.',
        '## Признаки горизонта\nВсе источники значений доступны на момент cutoff.',
        '## История общего объёма\nАгрегаты сохраняются для следующего эксперимента. При его обучении также нужны временные границы.',
        '''## Протокол и сохранение
Legacy: train до 4 октября, tuning 4–17 октября, holdout 18–31 октября (для текущих данных).
Weekly: 20 сентября – 3 октября — warmup калибровки; 4–17 октября — tuning;
18–31 октября — ранее просмотренный holdout. Даты вычисляются из истории.
Weekly-контроль использует одну поправку по warmup, зафиксированную до tuning.
Окна 7/14/28 дней будут сравниваться в следующем ноутбуке.
Корреляционные словари fit только по меткам до границы. Train использует все
получасовые старты последних 75% доступных cutoffs отдельно для каждого горизонта.
Legacy tuning участвует в early stopping: его нельзя использовать как независимый OOF.
Показываем in-sample train и query метрики; query legacy объединяет tuning и holdout,
раздельная итоговая оценка появится в последней ячейке.
''',
        '## Обучение на GPU\nПризнаки обрабатываются по одному горизонту. Результаты каждой модели сохраняются атомарно.',
        '''## Итоги и архивы
Основной критерий — средняя дневная WAPE + |Bias| по всем маршрутам и восьми горизонтам.
Отдельно сохраняются pooled score, signed bias, WAPE, дни и горизонты.
Для submission контрольная поправка переоценивается на 28 legacy-датах,
число деревьев берётся из legacy, все доступные метки участвуют в final recent75 refit.
Это воспроизведение текущего подхода, не автоматический выбор новой калибровки.

Сохраните выполненный `.ipynb`, **results.zip**, **models.zip**.
Для переноса незавершённого запуска сохраните **checkpoint.zip**.
В results.zip есть manifest, прогнозы, метрики, тайминги, лаги, агрегированная история
и контрольный submission. Полные матрицы признаков в архивы не входят.
''',
    ]
    cells = []
    for description, body in zip(descriptions, bodies):
        ast.parse(body)
        cells.append(dict(cell_type='markdown', metadata={}, source=description.splitlines(keepends=True)))
        cells.append(dict(cell_type='code', execution_count=None, metadata={}, outputs=[], source=body.splitlines(keepends=True)))
    notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                    language_info=dict(name='python', version='3.11'), accelerator='GPU'), nbformat=4, nbformat_minor=4)
    destination = ROOT / 'stepa' / '00_baseline.ipynb'
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n')
    print(destination)


if __name__ == '__main__':
    build()
