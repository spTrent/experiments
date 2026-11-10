import sys
import types
from pathlib import Path
import numpy as np


class DummyXGB:
    def __init__(self, **params):
        if params['device'] != 'cuda:0' or not params['enable_categorical']:
            raise ValueError('CUDA or categories disabled')

    def fit(self, x, y, sample_weight=None):
        self.categories = {c: list(x[c].cat.categories) for c in ('route_code', 'horizon_code')}
        self.value = float(np.average(y, weights=sample_weight))
        return self

    def predict(self, x):
        if any(list(x[c].cat.categories) != v for c, v in self.categories.items()):
            raise ValueError('Category dictionary changed')
        return np.full(len(x), self.value)

    def get_booster(self):
        return types.SimpleNamespace(save_config=lambda: '{"learner":{"generic_param":{"device":"cuda:0"}}}')


sys.modules['xgboost'] = types.SimpleNamespace(XGBRegressor=DummyXGB)
source = Path('scripts/check_learned_residual.py').read_text().replace('notebooks/clustering/learned_residual_gpu.ipynb', 'notebooks/clustering/learned_xgb_gpu.ipynb')
exec(compile(source, 'check_learned_xgb', 'exec'))
print('XGBoost CUDA configuration and stable categorical dictionaries: OK, test double only.')
