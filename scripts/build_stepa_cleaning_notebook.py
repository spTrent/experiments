import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = json.loads((ROOT / 'notebooks/stepa/stepa330.ipynb').read_text())
CELLS = []


class StripDocs(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body.pop(0)
        return node


def code(source):
    tree = StripDocs().visit(ast.parse(source))
    CELLS.append({'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': (ast.unparse(tree) + '\n').splitlines(keepends=True)})


def markdown(source):
    CELLS.append({'cell_type': 'markdown', 'metadata': {}, 'source': source.splitlines(keepends=True)})


def original(index):
    return ''.join(SOURCE['cells'][index]['source'])


def functions(index, names):
    tree = ast.parse(original(index))
    return '\n\n'.join(ast.unparse(node) for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names)


markdown('''# stepa330: проверка очистки обучающих целей

Контроль использует признаки и CatBoost из stepa330. Меняется только вес редких очень больших обучающих целей: 1, 0.25 или 0. Порог оценивается отдельно для маршрута и получасового слота только на разрешённой обучающей истории. Это гипотеза о загрязнении, а не установленные ошибки данных. Нули, исходная история для лагов/профиля и факты tuning/test сохраняются. Этот опыт не проверяет очистку входных признаков.

Train: до 4 октября; tuning: 4–17 октября; test: 18–31 октября 2025. Даты определяются из истории так же, как в stepa330. Октябрьский test уже просматривался в проекте и не является новым слепым периодом. Вариант выбирается только по tuning; независимая от этого выбора оценка test получается до финального переобучения. Одинаковая постобработка выбирается на контроле и затем фиксируется для всех вариантов, чтобы измерить эффект весов.

Импортируйте ноутбук в Kaggle, подключите train/test parquet, включите GPU и запустите все ячейки. Обучение выполняет пользователь. Это новый прогон контроля, численное совпадение с прежним GPU-запуском не гарантируется.
''')
s = original(1).replace('artifacts/team_scaled_experiment_v1/features', 'artifacts/stepa_cleaning_v1/features')
s = s[:s.index('# Challenger')]
s += '\nPILOT_H6_FEATURES = ["start_sin", "start_cos", "peak7_mean_max", "peak7_mean_level", "peak7_share_-1", "peak7_share_0", "peak7_share_1"]\n'
s += '\nCLEANING_WEIGHTS = {"baseline": 1.0, "soft_spikes": 0.25, "drop_spikes": 0.0}\n'
code(s)
code(original(3))
markdown('''## Контекстный порог

Для каждой пары маршрут/слот порог равен максимуму из 8 медиан, медианы + 12 × 1.4826 × MAD и 4 общих медиан маршрута. Требуется минимум 28 наблюдений пары. Все статистики обучаются до tuning; при финальном refit — на всей доступной размеченной истории с тем же правилом. Порог намеренно консервативный, константы в этом опыте не перебираются.
''')
code('''def fit_spike_limits(frame):
    work = frame[["route_id", "timestamp", "target_1h"]].copy()
    work["slot"] = work.timestamp.dt.hour * 2 + work.timestamp.dt.minute // 30
    keys = ["route_id", "slot"]
    grouped = work.groupby(keys).target_1h
    work["median"] = grouped.transform("median")
    work["deviation"] = (work.target_1h - work["median"]).abs()
    limits = work.groupby(keys).agg(median=("target_1h", "median"), count=("target_1h", "size"), mad=("deviation", "median")).reset_index()
    typical = work.groupby("route_id").target_1h.median()
    limits["limit"] = np.maximum.reduce([8 * limits["median"], limits["median"] + 12 * 1.4826 * limits["mad"], 4 * limits.route_id.map(typical)])
    limits.loc[limits["count"] < 28, "limit"] = np.inf
    return limits

def spike_mask(frame, limits, target_column="y"):
    lookup = limits.set_index(["route_id", "slot"])["limit"]
    slot = frame.timestamp.dt.hour * 2 + frame.timestamp.dt.minute // 30
    index = pd.MultiIndex.from_arrays([frame.route_id, slot])
    bound = lookup.reindex(index).to_numpy()
    return frame[target_column].to_numpy() > bound

raw_labels = pd.read_parquet(DATA / TRAIN_FILE, columns=["route_id", "timestamp", TARGET])
label_train = raw_labels.loc[raw_labels.timestamp < tune_start.normalize()].copy()
validation_limits = fit_spike_limits(label_train)
validation_limits.to_csv(OUT.parent / "spike_limits_train.csv", index=False)
flags = spike_mask(label_train, validation_limits, TARGET)
label_train.loc[flags].to_csv(OUT.parent / "suspicious_train_labels.csv", index=False)
display(pd.DataFrame([{"train_rows": len(label_train), "flagged": int(flags.sum()), "row_share": float(flags.mean()), "target_volume_share": float(label_train.loc[flags, TARGET].sum() / label_train[TARGET].sum())}]))
del label_train, flags
''')
markdown('## Исходные признаки stepa330\n\nЛаги и сезонный профиль строятся из неизменённых наблюдений. Позиции временной сетки сохраняются.\n')
for i in (5, 7, 9):
    code(original(i))
code(original(12))
markdown('''## Контроль и два варианта очистки

После каждого fit выводятся WAPE, |Relative Bias| и их сумма на одинаковых исходных train/test строках, в том числе на исключённых из обучения целях. Метрика на train также считается отдельно по стартам 10:30. Число деревьев выбирается по tuning MAE. После refit сохраняется прежняя test-оценка, новый test после refit не вычисляется.
''')
code('''validation_parts = []
train_parts = []
tree_counts = {}
evaluation_tables = {}
for h in range(1, 9):
    data = pd.read_parquet(OUT / f"history_h{h}.parquet")
    columns = model_columns(data)
    train = data.timestamp < tune_start.normalize()
    evaluate = data.cutoff.isin(dates[-28:])
    tune = evaluate & (data.cutoff < holdout_start)
    train_frame = data.loc[train]
    flagged = spike_mask(train_frame, validation_limits)
    meta_columns = ["cutoff", "timestamp", "route_id", "y", "profile28", "overlap", "slot_daily7", "slot_weekly4"]
    evaluation = data.loc[evaluate, meta_columns].copy()
    evaluation["horizon"] = h
    train1030 = train & data.cutoff.isin(dates[:-28])
    training = data.loc[train1030, meta_columns].copy()
    training["horizon"] = h
    for mode, flagged_weight in CLEANING_WEIGHTS.items():
        weights = np.where(flagged, flagged_weight, 1.0)
        model = make_model(1500)
        model.fit(train_frame[columns], train_frame.y / TARGET_SCALE, sample_weight=weights,
                  eval_set=(data.loc[tune, columns], data.loc[tune, "y"] / TARGET_SCALE),
                  early_stopping_rounds=100, use_best_model=True)
        tree_counts[(mode, h)] = model.tree_count_
        model.save_model(str(MODEL_DIR / f"{mode}_h{h}_evaluation.cbm"))
        evaluation[mode] = np.maximum(model.predict(data.loc[evaluate, columns]) * TARGET_SCALE, 0)
        train_prediction = np.maximum(model.predict(train_frame[columns]) * TARGET_SCALE, 0)
        training[mode] = train_prediction[train_frame.cutoff.isin(dates[:-28]).to_numpy()]
        test_rows = evaluation.loc[evaluation.cutoff >= holdout_start]
        table = pd.DataFrame([score_row(train_frame, train_prediction, "train_all_starts"),
                              score_row(training, training[mode], "train_1030"),
                              score_row(test_rows, test_rows[mode], "test")])
        table.insert(0, "horizon", h)
        table.insert(0, "mode", mode)
        evaluation_tables[(mode, h)] = table
        print(mode, h, "trees", model.tree_count_, "flagged", int(flagged.sum()))
        display(table)
        del model, train_prediction
    validation_parts.append(evaluation)
    train_parts.append(training)
    del data, train_frame
    gc.collect()
validation = pd.concat(validation_parts, ignore_index=True)
training_predictions = pd.concat(train_parts, ignore_index=True)
validation.to_parquet(OUT.parent / "validation_predictions.parquet", index=False)
training_predictions.to_parquet(OUT.parent / "train_1030_predictions.parquet", index=False)
pd.concat(evaluation_tables.values(), ignore_index=True).to_csv(OUT.parent / "model_metrics.csv", index=False)
''')
code(functions(15, {'get_raw_full_prediction', 'apply_final_calibration', 'fit_calibration_value', 'search_postprocessing'}))
code('''def mode_frame(frame, mode):
    result = frame.copy()
    result["catboost"] = result[mode]
    result["seasonal"] = result[["slot_daily7", "slot_weekly4"]].mean(axis=1)
    return result

baseline_frame = mode_frame(validation, "baseline")
_, config, search, _ = search_postprocessing(baseline_frame)
search.to_csv(OUT.parent / "baseline_postprocessing_search.csv", index=False)

def predict_processed(frame, mode, calibration):
    prepared = mode_frame(frame, mode)
    raw = get_raw_full_prediction(prepared, config["w"], config["overlap"], config["p_weight"])
    return apply_final_calibration(raw, config["calib_type"], calibration)

tuning_rows = []
for mode in CLEANING_WEIGHTS:
    tuning_frame = validation.loc[validation.cutoff < holdout_start]
    pred = predict_processed(tuning_frame, mode, config["calib_val"])
    tuning_rows.append({"mode": mode, **score_row(tuning_frame, pred, "tuning")})
tuning_report = pd.DataFrame(tuning_rows)
selected_mode = tuning_report.sort_values("среднее по дням", kind="stable").iloc[0]["mode"]
print("Выбор только по tuning:", selected_mode)
display(tuning_report)
tuning_report.to_csv(OUT.parent / "tuning_selection.csv", index=False)
report_rows = []
for mode in CLEANING_WEIGHTS:
    for split, frame in [("train_1030", training_predictions), ("test", validation.loc[validation.cutoff >= holdout_start])]:
        pred = predict_processed(frame, mode, config["calib_val"])
        report_rows.append({"mode": mode, "selected_on_tuning": mode == selected_mode, **score_row(frame, pred, split)})
report = pd.DataFrame(report_rows)
display(report)
report.to_csv(OUT.parent / "independent_metrics_before_refit.csv", index=False)
print("Независимая от выбора по tuning оценка test получена до переобучения. Test уже использовался в предыдущих экспериментах проекта.")
settings = {"selected_mode": selected_mode, "cleaning_weights": CLEANING_WEIGHTS, "fixed_baseline_postprocessing": config,
            "tune_start": str(tune_start), "test_start": str(holdout_start), "test_evaluated_before_refit": True,
            "spike_rule": {"median_multiplier": 8, "mad_multiplier": 12, "route_median_multiplier": 4, "min_count": 28}}
(OUT.parent / "settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2))
''')
markdown('''## Финальное переобучение и сабмит

Выбранный по tuning режим переобучается на всей доступной размеченной истории, включая прежние tuning/test. Если победит исключение всплесков, вся история участвует в построении признаков и статистик, а отмеченные цели получают нулевой вес. Отчёт независимой оценки test выше относится к моделям до этого переобучения. Калибровка фиксируется по tuning, чтобы опыт менял только очистку. Все восемь моделей refit обучаются заново; число деревьев зафиксировано до test.
''')
code('''full_limits = fit_spike_limits(raw_labels)
full_limits.to_csv(OUT.parent / "spike_limits_full.csv", index=False)
final_lag_table = calculate_dynamic_shoulder(target, hub_signal, times, routes, fit_before=end + STEP,
    max_lookback_hours=LAG_MAX_LOOKBACK_HOURS, min_pairs=LAG_MIN_PAIRS, min_correlation=LAG_MIN_CORRELATION)
final_lag_table.to_csv(OUT.parent / "status6_lags_full.csv", index=False)
submission_parts = []
for h in range(1, 9):
    data = add_correlation_features(pd.read_parquet(OUT / f"history_h{h}.parquet"), h, final_lag_table)
    future = add_correlation_features(pd.read_parquet(OUT / f"submission_h{h}.parquet"), h, final_lag_table)
    columns = model_columns(data)
    flags = spike_mask(data, full_limits)
    weights = np.where(flags, CLEANING_WEIGHTS[selected_mode], 1.0)
    model = make_model(tree_counts[(selected_mode, h)])
    model.fit(data[columns], data.y / TARGET_SCALE, sample_weight=weights)
    model.save_model(str(MODEL_DIR / f"{selected_mode}_h{h}_full.cbm"))
    result = future[["id", "profile28", "overlap", "slot_daily7", "slot_weekly4"]].copy()
    result["horizon"] = h
    result[selected_mode] = np.maximum(model.predict(future[columns]) * TARGET_SCALE, 0)
    submission_parts.append(result)
    full_prediction = np.maximum(model.predict(data[columns]) * TARGET_SCALE, 0)
    full_row = pd.DataFrame([score_row(data, full_prediction, "full_history_after_refit")])
    display(pd.concat([evaluation_tables[(selected_mode, h)], full_row], ignore_index=True))
    del model, data, future, full_prediction
    gc.collect()
predictions = pd.concat(submission_parts, ignore_index=True)
predictions["y_pred"] = predict_processed(predictions, selected_mode, config["calib_val"])
submission = test[["id"]].merge(predictions[["id", "y_pred"]], on="id", how="left", validate="one_to_one")
if len(submission) != len(test) or not submission.id.equals(test.id.reset_index(drop=True)) or not submission.id.is_unique or not np.isfinite(submission.y_pred).all() or (submission.y_pred < 0).any():
    raise ValueError("Некорректный сабмит")
submission.to_csv("submission_stepa_cleaning.csv", index=False)
print("submission_stepa_cleaning.csv", len(submission), "строк")
display(submission.head())
''')
notebook = {'cells': CELLS, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.11'}, 'accelerator': 'GPU'}, 'nbformat': 4, 'nbformat_minor': 5}
path = ROOT / 'notebooks/stepa/stepa_cleaning.ipynb'
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
print(path)
