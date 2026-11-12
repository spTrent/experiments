import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


folder = Path(__file__).resolve().parent
source = folder / os.environ.get('BLEND_NOTEBOOK', 'session6_recovered_blend_server.ipynb')
destination = source.with_suffix('.executed.ipynb')
notebook = nbformat.read(source, as_version=4)
client = NotebookClient(notebook, timeout=None,
                        kernel_name=os.environ.get('BLEND_KERNEL', 'blend-server'),
                        resources={'metadata': {'path': str(source.parent)}})


def progress(cell, cell_index, **kwargs):
    print(f'Cell {cell_index + 1}/{len(notebook.cells)}', flush=True)


client.on_cell_execute = progress
try:
    client.execute()
finally:
    nbformat.write(notebook, destination)
