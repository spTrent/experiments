"""Сборка submission выбранного рецепта; новых гипотез и оценки после refit нет."""
from pathlib import Path
import hashlib
import importlib.util
import json
import shutil
import sys
import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
state=json.loads((ROOT.parent/'optimization_state.json').read_text())
assert state['iterations']<=10
best=int(state['best_iteration'])
source=ROOT/'results'/f'iteration_{best:02}'/'strategy.py'
spec=importlib.util.spec_from_file_location('winning_strategy',source)
strategy=importlib.util.module_from_spec(spec);spec.loader.exec_module(strategy)
final=ROOT/'final';final.mkdir(exist_ok=False)
shutil.copy2(source,final/'strategy.py')
shutil.copy2(source,ROOT/'active_strategy.py')
history=pd.read_parquet(ROOT/'inputs/selected_evaluation.parquet').sort_values(['cutoff','horizon','route_id']).reset_index(drop=True)
evaluation=pd.read_parquet(ROOT/'results'/f'iteration_{best:02}'/'predictions.parquet')
eval_model=joblib.load(ROOT/'results'/f'iteration_{best:02}'/'calibrator.joblib')
# Проверка: прогноз origin не зависит от текущих/будущих labels, в том числе при NaN.
for cutoff in pd.to_datetime(['2025-10-18 10:30','2025-10-25 10:30']):
    prefix=history.loc[history.cutoff<=cutoff].copy()
    prefix.loc[prefix.cutoff.eq(cutoff),'y']=np.nan
    p=strategy.predict(eval_model,prefix,ROOT)
    actual=p[prefix.cutoff.eq(cutoff).to_numpy()]
    expected=evaluation.loc[evaluation.cutoff.eq(cutoff),'calibrated_prediction'].to_numpy()
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-7)
print('PASS_PREFIX_AND_UNKNOWN_LABELS_INVARIANCE',flush=True)
# Рецепт фиксирован; для submission только переобучаем ту же поправку на всех 28 OOS-днях.
# Новая поправка не используется задним числом для holdout score.
model=strategy.fit(history,ROOT)
future=pd.read_parquet(ROOT/'inputs/final_predictions.parquet')
future['split']='submission'
assert history.timestamp.max()+pd.Timedelta(hours=4)<future.cutoff.min()
prediction=np.maximum(strategy.predict(model,future,ROOT),0)
future['y_pred']=prediction
keys=pd.read_parquet(ROOT/'inputs/test_keys.parquet')
submission=keys[['id']].merge(future[['id','y_pred']],on='id',how='left',validate='one_to_one')
assert len(submission)==8000 and submission.id.is_unique
assert submission.id.equals(keys.id.reset_index(drop=True))
assert np.isfinite(submission.y_pred).all() and (submission.y_pred>=0).all()
assert future.groupby('horizon').size().to_dict()=={h:1000 for h in range(1,9)}
submission.to_csv(final/'submission_candidate.csv',index=False)
future.to_parquet(final/'prediction_details.parquet',index=False)
joblib.dump(model,final/'calibrator.joblib')
# Повторная загрузка и inference подтверждают воспроизводимость сохранённого артефакта.
restored=joblib.load(final/'calibrator.joblib')
np.testing.assert_allclose(strategy.predict(restored,future,ROOT),prediction,rtol=1e-12,atol=1e-7)

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

summary=[]
for i in range(1,state['iterations']+1):
    path=ROOT/'results'/f'iteration_{i:02}'/'report.json'
    if path.exists():summary.append(json.loads(path.read_text()))
    else:summary.append({'iteration':i,'name':'failed_readonly_numpy_array','holdout_score':None})
pd.DataFrame(summary).to_csv(final/'iterations.csv',index=False)
metrics=pd.read_csv(ROOT/'results'/f'iteration_{best:02}'/'metrics.csv')
metrics.to_csv(final/'validation_metrics.csv',index=False)
rows=[]
for day,part in evaluation.loc[evaluation.split.eq('holdout')].groupby('cutoff'):
    y=part.y.to_numpy();p=part.calibrated_prediction.to_numpy();e=p-y
    b=part.prediction.to_numpy()-y
    rows.append({'cutoff':day,'WAPE':np.abs(e).sum()/y.sum(),'Rbias':e.sum()/y.sum(),
                 'score':(np.abs(e).sum()+abs(e.sum()))/y.sum(),
                 'baseline_score':(np.abs(b).sum()+abs(b.sum()))/y.sum()})
pd.DataFrame(rows).to_csv(final/'validation_by_day.csv',index=False)
base=Path('/root/rwb-stepa-submission/output/submit_a2bcc4a0ad40e1c50257')
old=pd.read_csv(base/'submission.csv')
manifest={
 'best_iteration':best,'strategy':strategy.NAME,'checkpoint':state['best_checkpoint'],
 'validation':metrics.loc[(metrics.split=='holdout')&(metrics.horizon==0)].to_dict('records')[0],
 'autonomous_validation_runs':state['iterations'],'failed_runs':[3],
 'public_baseline_reported_by_user':.330,'public_candidate_score':None,'submitted':False,
 'evaluation_calibrator_fit':'2025-10-04..2025-10-17; per-origin EWMA uses only labels before cutoff minus 4h',
 'final_calibrator_refit':'same selected hyperparameters; OOS labels 2025-10-04..2025-10-31',
 'final_model_base':'ready full-history recent75 CatBoost; control h1/h2/h3/h4/h7, seasonal h5/h6/h8',
 'final_calibrator_last_label':str(history.timestamp.max()),
 'forecast_origin':str(future.cutoff.min()),'forecast_first':str(future.timestamp.min()),'forecast_last':str(future.timestamp.max()),
 'submission_rows':len(submission),'submission_columns':submission.columns.tolist(),
 'submission_sum':float(submission.y_pred.sum()),'prior_server_submission_sum':float(old.y_pred.sum()),
 'submission_sha256':digest(final/'submission_candidate.csv'),
 'reloaded_model_predictions_identical':True,'prefix_and_unknown_labels_invariance_passed':True,
 'inputs_sha256':{p.name:digest(p) for p in (ROOT/'inputs').glob('*.parquet')},
 'source_sha256':{p.name:digest(p) for p in ROOT.glob('*.py')},
 'base_model_sha256':{f'h{h}':digest(base/f'final/h{h}/model.cbm') for h in range(1,9)},
 'limitations':['External score is unmeasured; user submits manually.',
  'Holdout has been examined repeatedly during project development and these 10 runs.',
  'Pooled bias can cancel between days; daily and two weekly scores are reported separately.',
  'Final refit is not evaluated on its own training dates; transfer of OOS correction to full-history base models is uncertain.']}
(final/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False))
shutil.copy2(ROOT.parent/'optimization_state.json',final/'optimization_state.json')
print('FINAL_SUBMISSION',str(final/'submission_candidate.csv'),flush=True)
print('BEST_LOCAL',manifest['validation'],flush=True)
print('SUBMISSION_SUM',manifest['submission_sum'],'SHA256',manifest['submission_sha256'],flush=True)

