import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from threading import Thread
from time import monotonic

if not 0 < SESSION_HOURS <= 11.5:
    raise ValueError('SESSION_HOURS должен быть в диапазоне (0, 11.5]')
configuration = ('PART','HORIZONS','MODE','ROLE','OOF_WEEKS','INPUT_DIR','OUTPUT_DIR',
                 'CODE_ID','PARAMS','PROFILE','ROUTE_ALPHA')
setup = '\n'.join(f'{name} = {globals()[name]!r}' for name in configuration)
scratch = Path('/kaggle/temp') if Path('/kaggle/temp').is_dir() else None
with tempfile.TemporaryDirectory(dir=scratch) as folder:
    script = Path(folder)/'run_stack.py'
    script.write_text('\n\n'.join([setup,*ENGINE]))
    process = subprocess.Popen([sys.executable,'-u',str(script)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                               env=dict(os.environ,TMPDIR=folder),text=True,errors='replace',start_new_session=True)
    def show_output():
        for line in process.stdout:
            print(line,end='',flush=True)
    reader = Thread(target=show_output,daemon=True)
    reader.start()
    try:
        process.wait(timeout=max(0.,SESSION_HOURS*3600-(monotonic()-SESSION_START)))
    except subprocess.TimeoutExpired:
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGKILL)
        process.wait()
        status = 'time_limit'
        print('Бюджет сессии исчерпан. Подключите сохранённый output и предыдущие части, затем продолжите тот же этап.')
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGKILL)
        process.wait()
        raise
    else:
        status = 'complete'
        if process.returncode:
            raise RuntimeError(f'Этап завершился с кодом {process.returncode}. См. вывод выше.')
    finally:
        reader.join()
    status_path = Path(OUTPUT_DIR)/f'session_part{PART}_{PROFILE}_{MODE}_{ROLE}.json'
    status_path.parent.mkdir(parents=True,exist_ok=True)
    status_path.write_text(json.dumps(dict(status=status,elapsed_hours=(monotonic()-SESSION_START)/3600)))
