import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/stepa_cleaning_audit'
OUT.mkdir(parents=True, exist_ok=True)
frame = pd.read_parquet(ROOT / 'data/train_solo_track.parquet').sort_values(['timestamp', 'route_id'])
requests = pd.read_parquet(ROOT / 'data/test_solo_track.parquet')
times = pd.DatetimeIndex(frame.timestamp.unique())
routes = np.sort(frame.route_id.unique())
origins = times[(times.hour == 10) & (times.minute == 30) & (times + pd.Timedelta(hours=4) <= times[-1])]
tune_start, test_start = origins[-28].normalize(), origins[-14].normalize()
summary = {
    'rows': len(frame), 'routes': len(routes), 'timestamps': len(times),
    'start': str(times[0]), 'end': str(times[-1]),
    'nulls': int(frame.isna().sum().sum()),
    'duplicate_route_timestamps': int(frame.duplicated(['route_id', 'timestamp']).sum()),
    'regular_30min_grid': times.equals(pd.date_range(times[0], times[-1], freq='30min')),
    'complete_panel': len(frame) == len(times) * len(routes),
    'test_rows': len(requests), 'unique_test_ids': bool(requests.id.is_unique),
    'tune_start': str(tune_start), 'local_test_start': str(test_start),
}
notebook = json.loads((ROOT / 'notebooks/stepa/stepa_cleaning.ipynb').read_text())
namespace = {'np': np, 'pd': pd}
for cell in notebook['cells']:
    if cell['cell_type'] != 'code':
        continue
    tree = ast.parse(''.join(cell['source']))
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {'fit_spike_limits', 'spike_mask'}]
    if definitions:
        exec(compile(ast.Module(body=definitions, type_ignores=[]), '<cleaning_rules>', 'exec'), namespace)
train = frame.loc[frame.timestamp < tune_start].copy()
limits = namespace['fit_spike_limits'](train)
limits.to_csv(OUT / 'spike_limits_train.csv', index=False)
flags = namespace['spike_mask'](train, limits, 'target_1h')
suspicious = train.loc[flags].copy()
slot = suspicious.timestamp.dt.hour * 2 + suspicious.timestamp.dt.minute // 30
suspicious['slot'] = slot
suspicious = suspicious.merge(limits, on=['route_id', 'slot'], how='left', validate='many_to_one')
suspicious['ratio_to_slot_median'] = suspicious.target_1h / suspicious['median'].replace(0, np.nan)
suspicious.sort_values('target_1h', ascending=False).to_csv(OUT / 'suspicious_train_rows.csv', index=False)
summary.update({'train_rows': len(train), 'suspicious_train_rows': int(flags.sum()),
                'suspicious_train_row_share': float(flags.mean()),
                'suspicious_train_volume_share': float(train.loc[flags, 'target_1h'].sum() / train.target_1h.sum())})
rows = []
for split, part in [('train', train), ('tuning', frame.loc[frame.timestamp.ge(tune_start) & frame.timestamp.lt(test_start)]), ('test', frame.loc[frame.timestamp.ge(test_start) & frame.timestamp.lt(origins[-1].normalize() + pd.Timedelta(days=1))])]:
    for name in [f'status_{i}' for i in range(1, 7)] + ['target_1h']:
        values = part[name]
        rows.append({'split': split, 'column': name, 'rows': len(part), 'zeros': int(values.eq(0).sum()),
                     'negative': int(values.lt(0).sum()), 'nonfinite': int((~np.isfinite(values)).sum()),
                     'median': float(values.median()), 'p999': float(values.quantile(.999)), 'max': int(values.max())})
pd.DataFrame(rows).to_csv(OUT / 'numeric_quality_by_split.csv', index=False)
suspicious.groupby('route_id').agg(rows=('target_1h', 'size'), volume=('target_1h', 'sum')).sort_values('rows', ascending=False).to_csv(OUT / 'suspicious_routes.csv')
context = train.set_index(['timestamp', 'route_id']).target_1h
context_rows = []
for row in suspicious.nlargest(20, 'target_1h').itertuples():
    for offset in range(-2, 3):
        timestamp = row.timestamp + pd.Timedelta(minutes=30 * offset)
        context_rows.append({'event_timestamp': row.timestamp, 'route_id': row.route_id, 'offset_steps': offset,
                             'timestamp': timestamp, 'target_1h': context.get((timestamp, row.route_id), np.nan)})
pd.DataFrame(context_rows).to_csv(OUT / 'top_spike_context.csv', index=False)
(OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(summary, ensure_ascii=False, indent=2))
