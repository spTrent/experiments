"""Exercise standalone notebook flow with real features and a lightweight model double."""
import ast
import contextlib
import gc
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


class ModelDouble:
    fits = 0

    def __init__(self, **kwargs):
        assert kwargs['task_type'] == 'GPU'
        self.tree_count_ = kwargs['iterations']

    def fit(self, x, y, **kwargs):
        assert len(x) == len(y) > 0
        assert np.isfinite(y).all()
        self.names = list(x)
        ModelDouble.fits += 1

    def predict(self, x):
        assert list(x) == self.names
        return x.target_back_0.to_numpy()/100_000

    def save_model(self, path):
        Path(path).write_text(json.dumps(self.names))

    def get_feature_importance(self):
        return np.ones(len(self.names))/len(self.names)


def main():
    n = json.loads((ROOT/'stepa/00_baseline.ipynb').read_text())
    codes = [''.join(c['source']) for c in n['cells'] if c['cell_type'] == 'code']
    for code in codes:
        ast.parse(code)
    with tempfile.TemporaryDirectory() as folder:
        run = Path(folder)/'baseline_00'
        for sub in ['models', 'predictions', 'reports', 'lags', 'checkpoints']:
            (run/sub).mkdir(parents=True, exist_ok=True)
        env = dict(Path=Path, np=np, pd=pd, pa=pa, pq=pq, gc=gc, time=time, json=json,
                   hashlib=hashlib, zipfile=zipfile, display=lambda *a: None,
                   RUN_ROOT=run, RUN_ID='test', RUN_FULL_REFIT=True,
                   CatBoostRegressor=ModelDouble, GPU_DEVICES='0', THREAD_COUNT=1,
                   WEEKLY_BLOCKS=6, ITERATIONS=1500, RECENT_TRAIN_FRACTION=.75,
                   TARGET_SCALE=100_000, TARGET='target_1h', STATUSES=[f'status_{i}' for i in range(1,7)],
                   STEP=pd.Timedelta(minutes=30), LAG_MAX_LOOKBACK_HOURS=24,
                   LAG_MIN_PAIRS=336, LAG_MIN_CORRELATION=.05)
        for node in ast.parse(codes[0]).body:
            if isinstance(node, ast.FunctionDef):
                exec(compile(ast.Module(body=[node], type_ignores=[]), '<helpers>', 'exec'), env)
        env['save_json'](run/'manifest.json', {'test': True})
        times = pd.date_range('2025-08-25', '2025-11-01 10:30', freq='30min')
        routes = np.arange(3)
        rng = np.random.default_rng(11)
        values = {name:rng.uniform(1000, 90000, (len(times),len(routes))).astype('float32')
                  for name in [env['TARGET'], *env['STATUSES']]}
        dates = times[(times.hour == 10)&(times.minute == 30)&(times+8*env['STEP']<=times[-1])]
        test = pd.MultiIndex.from_product([routes, times[-1]+np.arange(1,9)*env['STEP']],
                                         names=['route_id','timestamp']).to_frame(index=False)
        test.insert(0, 'id', np.arange(len(test)))
        env.update(times=times, routes=routes, values=values, dates=dates, end=times[-1], test=test,
                   tune_start=dates[-28], holdout_start=dates[-14])
        for code in codes[2:5]:
            exec(code, env)
        origin = np.array([len(times)-100])
        before = [env['horizon_dataset'](h, origin) for h in range(1,9)]
        reference = json.loads((ROOT/'stepa/stepa330.ipynb').read_text())
        old_env = env.copy()
        old_code = ''.join(reference['cells'][9]['source']).split('\ndef write_horizon')[0]
        old_env['validation_lag_table'] = env['calculate_dynamic_shoulder'](
            env['target'],env['hub_signal'],times,routes,fit_before=dates[-28].normalize())
        exec(old_code, old_env)
        pilot = ['start_sin','start_cos','peak7_mean_max','peak7_mean_level','peak7_share_-1','peak7_share_0','peak7_share_1']
        for h, actual in enumerate(before,1):
            expected = old_env['horizon_dataset'](h,origin).drop(columns=pilot, errors='ignore')
            actual = env['add_correlation_features'](actual.copy(),h,old_env['validation_lag_table'])
            pd.testing.assert_frame_equal(actual,expected)
        print('PASS: all eight feature sets and column order match stepa330')
        lag_before = env['calculate_dynamic_shoulder'](env['target'],env['hub_signal'],times,routes,
                                                       fit_before=times[origin[0]])
        for a in values.values():
            a[origin[0]+1:] *= 7
        exec(codes[2],env)
        for h, expected in enumerate(before,1):
            pd.testing.assert_frame_equal(env['horizon_dataset'](h,origin),expected)
        lag_after = env['calculate_dynamic_shoulder'](env['target'],env['hub_signal'],times,routes,
                                                      fit_before=times[origin[0]])
        pd.testing.assert_frame_equal(lag_before,lag_after)
        print('PASS: future mutation does not affect past features or lag fitting')
        with contextlib.redirect_stdout(io.StringIO()):
            for code in codes[5:]:
                exec(code,env)
        assert ModelDouble.fits == 64
        weekly = pd.read_parquet(run/'weekly_predictions.parquet')
        assert len(weekly)==42*8*len(routes)
        assert (weekly.model_info_through<weekly.cutoff).all()
        scored = weekly.query("period != 'warmup'")
        assert (scored.calibration_info_through<scored.cutoff).all()
        submission = pd.read_csv(run/'submission.csv')
        assert submission.id.tolist()==test.id.tolist()
        assert np.isfinite(submission.y_pred).all() and (submission.y_pred>=0).all()
        with contextlib.redirect_stdout(io.StringIO()):
            env['run_all']()
        assert ModelDouble.fits == 64
        for name in ['results.zip','models.zip','checkpoint.zip']:
            with zipfile.ZipFile(Path(folder)/name) as z:
                assert z.testzip() is None
        assert not list(run.glob('*.partial*'))
        (run/'models/legacy_h1.cbm').write_text('corrupted')
        try:
            env['completed']('legacy',1)
        except ValueError:
            pass
        else:
            raise AssertionError('Corrupted checkpoint was accepted')
        print('PASS: 64-stage fits, causal weekly metadata, reports, submission, archives, resume and corruption detection')
        print('Model fits used a test double; real GPU performance is not tested here.')


if __name__ == '__main__':
    main()
