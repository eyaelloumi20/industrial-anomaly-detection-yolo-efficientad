#!/usr/bin/env bash
set -euo pipefail
cd /home/ahmed/two-stage-efficientad-inspection

echo "=== DATASET ==="
python - <<'PY'
import csv
from collections import defaultdict, Counter
p='dataset_multi_bg_fast/split_csv/1cls.csv'
d=defaultdict(Counter)
with open(p, encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        d[r['split']][r['label']] += 1
for s in ('train','test','validation'):
    print(f"{s:12s}: {sum(d[s].values()):5d} -> {dict(d[s])}")
PY

echo
echo "=== NEW CHECKPOINT ==="
find ./ckpt_lowres_192_multibg_fast -maxdepth 1 -type f -printf "%f\n" 2>/dev/null | sort || true

echo
echo "=== LAST TRAINING LINES ==="
tail -40 ./lowres192_multibg_fast_train.log 2>/dev/null || true
