"""Print PushT visual-shift success rates from a run_visual.sh sweep.

Reads ``<root>/<model>/pusht-<visual>/results.txt`` and prints the last
``success_rate`` in each file. Missing runs are left blank.

Usage:
    python scripts/plan/visual_table.py outputs/run_many
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SUCCESS_RE = re.compile(
    r"['\"]success_rate['\"]\s*:\s*"
    r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)'
)
MODELS = ('lewm', 'mopwm', 'mopwm-full')
VISUALS = (
    'none',
    'blur',
    'snp1',
    'snp5',
    'dark',
    'red-agent',
    'red-block',
    'red-anchor',
)


def success_rate(path: Path) -> float | None:
    if not path.is_file():
        return None
    found = SUCCESS_RE.findall(path.read_text(encoding='utf-8', errors='replace'))
    if not found:
        return None
    return float(found[-1])


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else 'outputs/run_many')
    rates = {
        visual: {
            model: success_rate(root / model / f'pusht-{visual}' / 'results.txt')
            for model in MODELS
        }
        for visual in VISUALS
    }

    label_width = max(len('visual'), *(len(name) for name in VISUALS))
    col_width = max(8, *(len(model) for model in MODELS))
    header = f"{'visual':<{label_width}}" + ''.join(
        f'{model:>{col_width}}' for model in MODELS
    )
    print(f'success rate (%)  {root}')
    print(header)
    print('-' * len(header))
    for visual in VISUALS:
        row = f'{visual:<{label_width}}'
        for model in MODELS:
            rate = rates[visual][model]
            cell = '-' if rate is None else f'{rate:.1f}'
            row += f'{cell:>{col_width}}'
        print(row)

    missing = [
        f'{model}/pusht-{visual}'
        for visual in VISUALS
        for model in MODELS
        if rates[visual][model] is None
    ]
    if missing:
        print()
        print('missing: ' + ', '.join(missing))


if __name__ == '__main__':
    main()
