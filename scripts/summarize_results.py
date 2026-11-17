"""Print measured results alongside the official repository's reference scores."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
print('| Dataset | Measured mAP / Rank-1 | Official mAP / Rank-1 |')
print('| --- | --- | --- |')
for dataset, reference in [('market1501', '89.0% / 95.1%'), ('msmt17', '67.8% / 85.3%')]:
    path = root / 'logs' / f'{dataset}_transreid_stride' / 'metrics.json'
    if path.exists():
        metrics = json.loads(path.read_text())
        measured = f"{metrics['mAP']:.1%} / {metrics['rank1']:.1%}"
    else:
        measured = 'Pending'
    print(f'| {dataset} | {measured} | {reference} |')
print('\nReference: https://github.com/damo-cv/TransReID#trained-models-and-logs-size-256')
