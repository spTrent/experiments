import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

folder = Path(__file__).resolve().parent
os.environ.setdefault('RWB_ROOT', str(folder))
parser = argparse.ArgumentParser()
parser.add_argument('parts', nargs='*', type=int)
parser.add_argument('--worker', type=int, choices=(2,3,4))
args = parser.parse_args()
if any(part not in (2,3,4) for part in args.parts):
    parser.error('Допустимые части: 2, 3, 4')
if args.worker:
    part = args.worker
    notebook = json.loads((folder / f'evaluate_part_{part}.ipynb').read_text())
    scope = {'__name__':'__main__'}
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            exec(compile(''.join(cell['source']), f'part{part}-{cell["id"]}', 'exec'), scope)
else:
    for part in args.parts or (2,3,4):
        print(f'Запуск части {part}',flush=True)
        subprocess.run([sys.executable,'-u',__file__,'--worker',str(part)],check=True)
