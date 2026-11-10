import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
n = json.loads((BASE / 'notebooks/clustering/nightly_rich_gpu.ipynb').read_text())
source = '''import shutil

STOP_NEW_FITS_UTC = '2026-11-10T08:25:00+00:00'
roots = [p.parent for p in Path('/kaggle/input').rglob('nightly_completed.json') if 'nightly-rich-gpu' in p.parts]
if len(roots) != 1:
    raise ValueError('Подключите Output nightly-rich-gpu и case_rwb')
source_root = roots[0]
completed = json.loads((source_root / 'nightly_completed.json').read_text())
if not completed:
    raise ValueError('Нет полностью оценённых вариантов')
best = min(completed, key=lambda row: row['selection_score'])
source_run = source_root / Path(best['directory']).name
original_manifest = json.loads((source_run / 'manifest.json').read_text())
ACTIVE = original_manifest['cat_configs'][0]['experiment']
PIPELINE_VERSION = 'nightly_gpu_v1_' + ACTIVE['name']
CAT_CONFIGS = [dict(experiment=ACTIVE, parameters=PARAMETERS)]
STEP = pd.Timedelta(minutes=30)
with threadpool_limits(limits=THREADS):
    initialize_calibration()
    current_manifest = json.loads((RUN / 'manifest.json').read_text())
    if any(original_manifest[key] != current_manifest[key] for key in ('train_hash', 'test_hash', 'code')):
        raise ValueError('Данные или код отличаются от оценённой модели')
    shutil.copytree(source_run, RUN, dirs_exist_ok=True)
    group_table = pd.read_csv(RUN / 'calibration_groups.csv').set_index('route_id').reindex(ROUTES)
    if group_table.calibration_group.isna().any():
        raise ValueError('Не найдены фиксированные группы')
    CAL_GROUP_LABELS = group_table.calibration_group.to_numpy(dtype=int)
    frozen = json.loads((RUN / 'calibration_frozen.json').read_text())
    print('Выбор по val, test до полного refit:', best, flush=True)
    print(pd.read_csv(RUN / 'independent_calibration_metrics.csv').to_string(index=False), flush=True)
    finalize_calibration(frozen)
'''
n['cells'][-1]['source'] = source.splitlines(True)
n['cells'][0]['source'] = ['# Полный refit лучшего rich-варианта\n', 'Подключить Output nightly-rich-gpu и case_rwb. Используется только победитель ранее завершённой оценки по val. Test повторно не участвует в выборе. Все модели переобучаются на всей размеченной истории; завершённые финальные горизонты можно возобновить. Независимая оценка test получена в исходном запуске до полного refit. Обучение только на GPU, остановка новых fit в11:25 МСК. Этот ноутбук нужен, если исходный длинный цикл не успел собрать submission.\n']
for cell in n['cells']:
    if cell['cell_type'] == 'code':
        compile(''.join(cell['source']), 'rich_refit', 'exec')
out = BASE / 'notebooks/clustering/rich_refit_gpu.ipynb'
out.write_text(json.dumps(n, ensure_ascii=False, indent=1))
print(out)
