from pathlib import Path
import json
import pandas as pd

root = Path(__file__).resolve().parents[1]
out = root / 'artifacts/xgb_cleaning_audit/threshold_sweep'
summary = pd.read_csv(out / 'comparison.csv')
features = pd.read_csv(out / 'feature_changes.csv')
metadata = json.loads((out / 'metadata.json').read_text())
fields = ['target_1h'] + [f'status_{i}' for i in range(1, 7)]


def fmt(value):
    if 0 < value < .001:
        return '<0,001'
    return f'{value:.3f}'.replace('.', ',')


def table(frame, index, value, label):
    pivot = frame.pivot(index=index, columns='threshold', values=value)
    if index == 'field':
        pivot = pivot.reindex(fields)
    lines = [f'| {label} | 3 | 4 | 5 | 6 |', '|---|---:|---:|---:|---:|']
    for name, row in pivot.iterrows():
        lines.append('| ' + str(name) + ' | ' + ' | '.join(fmt(row[t]) for t in [3, 4, 5, 6]) + ' |')
    return '\n'.join(lines)


parts = ['# Доля замен при порогах MAD 3, 4, 5, 6', '', 'Все значения в таблицах — проценты. Модели не обучались; это аудит очистки и преобразования признаков, а не подбор порога по test.', '', '## Исходный датасет', '', 'Знаменатель для каждого поля — все 4 630 000 исходных наблюдений, включая начальный период без достаточной истории. Фильтр использует log1p, медиану и точный MAD предыдущих 28 дней того же получасового слота, минимум 14 дней, масштаб max(1.4826 × MAD, 0.25). Текущая точка исключена. Все пороги применены к одной и той же исходной истории; очищенные наблюдения не используются для оценки следующих окон.', '', table(summary, 'field', 'raw_replaced_pct', 'Поле'), '', '## Ячейки признаков после лагов и окон', '', 'Ниже знаменатель — доступные ячейки соответствующего семейства входных признаков по всем восьми горизонтам и всем доступным размеченным origins. Для девяти стартов берутся 08:30–12:30. Это состав полного refit, без конкурсных будущих строк. Исходная и очищенная матрицы пересчитаны реальным кодом признаков ноутбука. Сравниваются float32 значения: два NaN равны и исключаются из знаменателя, появление/исчезновение NaN считается изменением.', '', 'Семейство поля включает его собственные лаги, скользящие средние, нормировки, сезонные и сетевые признаки, если они зависят только от этого поля. Поэтому доля не является процентом исходных строк и не обязана совпадать с долей прямых замен. Признаки с несколькими исходными полями показаны отдельно.', '', '### Все получасы', '', table(summary, 'field', 'all_halfhours_changed_cells_pct', 'Семейство'), '', '### Девять стартов', '', table(summary, 'field', 'nine_starts_changed_cells_pct', 'Семейство'), '', '## Строки с хотя бы одним изменившимся признаком семейства', '', 'Это отдельная мера распространения влияния: достаточно изменения одной ячейки семейства, чтобы строка считалась затронутой. Одна исходная замена может затронуть много последующих строк через лаги, профили и сетевые суммы.', '', '### Все получасы', '', table(summary, 'field', 'all_halfhours_changed_rows_pct', 'Семейство'), '', '### Девять стартов', '', table(summary, 'field', 'nine_starts_changed_rows_pct', 'Семейство'), '', '## Все модельные признаки и смешанные комбинации', '', '| Порог | Режим | Семейство | Изменено ячеек, % | Строк с изменением, % |', '|---:|---|---|---:|---:|']
for family in ['mixed_statuses', 'correlation_target_status6', 'all_model_features']:
    for variant in ['all_halfhours', 'nine_starts']:
        for threshold in [3, 4, 5, 6]:
            row = features.loc[features.family.eq(family) & features.variant.eq(variant) & features.threshold.eq(threshold)].iloc[0]
            parts.append(f'| {threshold} | {variant} | {family} | {fmt(row.feature_cells_changed_pct)} | {fmt(row.rows_with_change_pct)} |')
parts += ['', 'Смешанные статусы: warehouse_total_load, warehouse_processing_rate, hub_buffer, hub_gate_bottleneck и лаг общей нагрузки. Корреляционные признаки зависят от target и status_6; словарь отдельно переоценивается до 7 октября 2025 года на исходных/очищенных рядах, как в оценочной стадии ноутбука. Поэтому их изменение может затронуть даже строки без прямой замены в соответствующем окне. Календарь и route_id не очищаются; числовой календарь включён только в общий знаменатель всех модельных признаков. Служебный overlap и цели y/y_clean исключены из входных признаков.', '', 'Сетевые признаки рассчитаны по полной панели 1 000 маршрутов, даже при технической обработке данных блоками маршрутов. Распределение колонок по семействам дано в feature_families.json; результат отдельно для каждой колонки — individual_feature_changes.csv. Счётчики и точные доли без округления — raw_replacements.csv, feature_changes.csv и comparison.csv.', '', f'Время аудита: {metadata["elapsed_seconds"]:.1f} с. Обучение не запускалось. Порог рабочего ноутбука остаётся 4; аудит его не меняет.', '']
(out / 'report.md').write_text('\n'.join(parts))
print('\n'.join(parts[:19]))
raw = pd.read_csv(out / 'raw_replacements.csv')
totals = []
for threshold in [3, 4, 5, 6]:
    block = raw.loc[raw.threshold.eq(threshold)]
    row = {'threshold': threshold, 'sum_field_percentages_pp': block.raw_replaced_pct.sum(), 'all_raw_cells_replaced_pct': 100 * block.replaced.sum() / block.raw_values.sum()}
    for variant in ['all_halfhours', 'nine_starts']:
        total = features.loc[features.threshold.eq(threshold) & features.variant.eq(variant) & features.family.eq('all_model_features')].iloc[0]
        row[variant + '_all_feature_cells_changed_pct'] = total.feature_cells_changed_pct
    totals.append(row)
pd.DataFrame(totals).to_csv(out / 'totals.csv', index=False)
lines = ['', '## Итоги по всем полям для каждого порога', '', 'Σ долей полей — сумма процентов семи исходных колонок, в процентных пунктах. Общая доля исходных замен рассчитана относительно всех 32 410 000 ячеек семи полей. Итоги по признакам взвешены по числу доступных ячеек и включают смешанные, корреляционные и календарные признаки.', '', '| Порог | Σ долей полей, п.п. | Все исходные ячейки, % | Все ячейки признаков: получасы, % | Все ячейки признаков: 9 стартов, % |', '|---:|---:|---:|---:|---:|']
for row in totals:
    lines.append('| ' + str(row['threshold']) + ' | ' + ' | '.join(fmt(value) for key, value in row.items() if key != 'threshold') + ' |')
with (out / 'report.md').open('a') as stream:
    stream.write('\n'.join(lines) + '\n')
