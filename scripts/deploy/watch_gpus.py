#!/usr/bin/env python3
"""Print Runpod pod GPUs that currently have stock, every 10 seconds.

Reads RUNPOD_API_KEY from the environment. Availability is per cloud, so each
refresh queries secure and community cloud for a single-GPU pod.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime

API_URL = 'https://api.runpod.io/v2/catalog/gpus'
INTERVAL_S = 10
CLOUDS = ('SECURE', 'COMMUNITY')
STOCK_RANK = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}


def fetch_gpus(api_key: str, cloud: str) -> list[dict]:
    query = urllib.parse.urlencode(
        {
            'include': 'AVAILABILITY',
            'product': 'POD',
            'cloud': cloud,
            'count': 1,
        }
    )
    request = urllib.request.Request(
        f'{API_URL}?{query}',
        headers={
            'Authorization': f'Bearer {api_key}',
            'Accept': 'application/json',
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    return payload['gpus']


def available_rows(gpus: list[dict], cloud: str) -> list[dict]:
    rows = []
    for gpu in gpus:
        stock = gpu.get('availability') or 'NONE'
        if stock == 'NONE':
            continue
        datacenters = [
            f"{dc['id']}:{dc['availability']}"
            for dc in gpu.get('dataCenters') or []
            if dc.get('availability') not in (None, 'NONE')
        ]
        price = (gpu.get('price') or {}).get(cloud.lower())
        rows.append(
            {
                'name': gpu.get('name') or gpu['id'],
                'gpu_id': gpu['id'],
                'memory': gpu.get('memory') or 0,
                'cloud': cloud.lower(),
                'stock': stock,
                'price': None if price is None else f'{price:.2f}',
                'datacenters': ', '.join(datacenters) or '-',
            }
        )
    return rows


def print_table(rows: list[dict]) -> None:
    rows.sort(
        key=lambda row: (
            STOCK_RANK.get(row['stock'], 9),
            -row['memory'],
            row['name'],
            row['cloud'],
        )
    )
    stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'{stamp}  pod GPUs with stock: {len(rows)}')
    if not rows:
        print('  none')
        print(flush=True)
        return

    headers = ('name', 'gpu_id', 'vram', 'cloud', 'stock', '$/hr', 'datacenters')
    body = [
        (
            row['name'],
            row['gpu_id'],
            str(row['memory']),
            row['cloud'],
            row['stock'],
            row['price'] or '-',
            row['datacenters'],
        )
        for row in rows
    ]
    widths = [
        max(len(header), *(len(line[index]) for line in body))
        for index, header in enumerate(headers)
    ]
    print('  '.join(header.ljust(width) for header, width in zip(headers, widths)))
    for line in body:
        print('  '.join(cell.ljust(width) for cell, width in zip(line, widths)))
    print(flush=True)


def main() -> None:
    api_key = os.environ.get('RUNPOD_API_KEY')
    if not api_key:
        sys.exit('RUNPOD_API_KEY is not set.')

    while True:
        started = time.monotonic()
        try:
            rows = []
            for cloud in CLOUDS:
                rows.extend(available_rows(fetch_gpus(api_key, cloud), cloud))
            print_table(rows)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors='replace').strip()
            print(f'HTTP {exc.code}: {detail}', file=sys.stderr)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
            print(f'request failed: {exc}', file=sys.stderr)

        remaining = INTERVAL_S - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(remaining)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print()
