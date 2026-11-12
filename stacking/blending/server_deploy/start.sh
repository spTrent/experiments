set -eu
cd "$(dirname "$0")"
python_bin=${BLEND_PYTHON:-python3}
export BLEND_INPUT="$PWD/blend_inputs"
export BLEND_WORK="${BLEND_WORK:-$PWD/output}"
export BLEND_KERNEL=blend-server
export JUPYTER_PATH="$PWD/.jupyter${JUPYTER_PATH:+:$JUPYTER_PATH}"
"$python_bin" - <<'PY'
import catboost
import cupy
import cuml
import ipykernel
import lightgbm
import nbclient
import nbformat
import xgboost
from catboost.utils import get_gpu_device_count

if cupy.cuda.runtime.getDeviceCount() < 1 or get_gpu_device_count() < 1:
    raise RuntimeError('Нужен доступный NVIDIA GPU в текущем Python-окружении')
print('Проверка Python и GPU прошла', flush=True)
PY
"$python_bin" -m ipykernel install --prefix "$PWD/.kernel" --name "$BLEND_KERNEL" --display-name 'Blend server'
export JUPYTER_PATH="$PWD/.kernel/share/jupyter${JUPYTER_PATH:+:$JUPYTER_PATH}"
exec "$python_bin" -u run_blend.py
