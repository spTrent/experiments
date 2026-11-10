import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
source = json.loads((BASE/'notebooks/clustering/cluster_catboost_calibration.ipynb').read_text())
config = ''.join(source['cells'][1]['source'])
parameters = {name: {str(h): json.loads((BASE/f'artifacts/{name}/model_params/best_h{h}.json').read_text()) for h in range(1,9)} for name in ('catboost_direct','catboost_residual','xgb_direct','xgb_residual')}
experiments = []
for algorithm, prefix in [('cat','catboost'),('xgb','xgb')]:
    for residual in (False,True):
        parameter_set = prefix + ('_residual' if residual else '_direct')
        loss = ('RMSE' if residual else 'MAE') if algorithm=='cat' else ('reg:squarederror' if residual else 'reg:absoluteerror')
        experiments.append(dict(name=parameter_set+'_tuned',algorithm=algorithm,residual=residual,parameter_set=parameter_set,loss=loss,compact=False))
for algorithm,prefix in [('cat','catboost'),('xgb','xgb')]:
    experiments.append(dict(name=prefix+'_direct_compact',algorithm=algorithm,residual=False,parameter_set=prefix+'_direct',loss='MAE' if algorithm=='cat' else 'reg:absoluteerror',compact=True))
nightly = (BASE/'scripts/cluster_experiment/nightly.py').read_text()
config = config.replace("TRAIN_PATH = 'data/train_solo_track.parquet'", "TRAIN_PATH = '/kaggle/input/datasets/iskanderyamalov/case-rwb/train_solo_track.parquet'").replace("TEST_PATH = 'data/test_solo_track.parquet'", "TEST_PATH = '/kaggle/input/datasets/iskanderyamalov/case-rwb/test_solo_track.parquet'")
config += '\nSTOP_NEW_FITS_UTC = "2026-11-10T08:00:00+00:00"\n'
config += 'PARAMETERS = '+repr(parameters)+'\nEXPERIMENTS = '+repr(experiments)+'\n'
config += 'CODE_HASH = '+repr(hashlib.sha256((''.join(source['cells'][3]['source'])+''.join(source['cells'][4]['source'])+nightly+json.dumps(parameters)).encode()).hexdigest())+'\n'
entries=[('markdown','# Ночной цикл GPU\n\nCatBoost и XGBoost, direct и residual относительно среднего доступных сезонных лагов 7/14/28 дней. Параметры из artifacts включены в ноутбук. Компактные варианты ограничены глубиной 6 и 900 деревьями. Обучение на всех получасах, группы 1/8. Восемь способов калибровки, выбор на второй неделе val, test только после фиксации. Параметры перенесены в новый набор признаков и нормировку; это проверка переноса, а не точное воспроизведение исходных моделей. GPU обязателен. Новые fit останавливаются с 11:00 МСК, оставляя запас до 11:30; долгий текущий fit может закончиться позднее. Каждый эксперимент возобновляется по checkpoint. Итоговый победитель выбирается по val после калибровки; дополнительно сохраняется WAPE. Полный refit всех участвующих моделей перед сабмитом. Независимая оценка test получена до переобучения. Этот test уже изучался в проекте.'),('code',config),('code',''.join(source['cells'][3]['source'])),('code',''.join(source['cells'][4]['source'])),('code',nightly),('code','STEP = pd.Timedelta(minutes=30)\nrun_nightly()')]
cells=[]
for i,(kind,text) in enumerate(entries):
    c=dict(cell_type=kind,id=f'nightly-{i}',metadata={},source=text.splitlines(True))
    if kind=='code':
        compile(text,f'cell{i}','exec')
        c.update(outputs=[],execution_count=None)
    cells.append(c)
out=BASE/'notebooks/clustering/nightly_gpu.ipynb'
out.write_text(json.dumps(dict(nbformat=4,nbformat_minor=5,metadata=source['metadata'],cells=cells),ensure_ascii=False,indent=1))
print(out)
