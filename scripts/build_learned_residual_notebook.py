import json
import hashlib
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
n=json.loads((BASE/'notebooks/clustering/cluster_catboost_calibration.ipynb').read_text())
config=''.join(n['cells'][1]['source']).replace("'data/train_solo_track.parquet'","'/kaggle/input/datasets/iskanderyamalov/case-rwb/train_solo_track.parquet'").replace("'data/test_solo_track.parquet'","'/kaggle/input/datasets/iskanderyamalov/case-rwb/test_solo_track.parquet'")
runtime=''.join(n['cells'][3]['source'])
meta=(BASE/'scripts/cluster_experiment/learned_residual.py').read_text()
config+="\nPIPELINE_VERSION = 'learned_residual_gpu_v1'\nCODE_HASH = "+repr(hashlib.sha256((runtime+meta).encode()).hexdigest())+'\n'
entries=[('markdown','# Обучаемая поправка по кластерам на OOF\n\nПодключите case_rwb и Output clustering-calibration (версия 367876306). Используются сохранённые прогнозы CatBoost k8. Дополнительная модель обучается на остатках OOF, общая либо на фиксированных группах маршрутов. Признаки компактного прогноза, нормированный базовый прогноз, категориальные маршрут и горизонт. Четыре небольшие конфигурации CatBoost MAE/RMSE и фиксированная смесь 50/50 моделей глубины 4 и 5, как в стратегии 0.328. Проверяются исходная поправка, общий множитель и rolling EWMA-сдвиг с полупериодом 7 дней. Rolling использует только метки timestamp < cutoff − 4 часа, включая уже наблюдённые прошлые дни проверочного блока; это последовательная оценка заранее зафиксированного правила, а не полностью замороженный коэффициент. GPU обязателен. Первые 7 дней val обучают поправку, следующие 7 выбирают конфигурацию. Затем обучение на всех 14 днях OOF до test и замороженная оценка test. После оценки поправка переобучается на всех доступных OOF val + бывший test; базовый прогноз для сабмита уже получен моделями, переобученными на всей той же размеченной истории в исходном запуске. Источники проверяются по хешам данных. Независимая оценка test получена до полного переобучения; test уже изучался ранее. Базовые модели обучены на всех получасах, метамодели — только на доступных OOF-прогнозах в 10:30.'),('code',config),('code',runtime),('code',meta),('code','STEP = pd.Timedelta(minutes=30)\nwith threadpool_limits(limits=THREADS):\n    run_meta()')]
cells=[]
for i,(kind,s) in enumerate(entries):
 c=dict(cell_type=kind,id=f'meta-{i}',metadata={},source=s.splitlines(True))
 if kind=='code':compile(s,f'meta{i}','exec');c.update(outputs=[],execution_count=None)
 cells.append(c)
out=BASE/'notebooks/clustering/learned_residual_gpu.ipynb';out.write_text(json.dumps(dict(nbformat=4,nbformat_minor=5,metadata=n['metadata'],cells=cells),ensure_ascii=False,indent=1));print(out)
