"""
Wayback Machine enrichment script — uses Yelp alias (URL slug) for lookups.

Reads data/yelp_aliases.csv (produced by fetch_aliases.py) to get the correct
Wayback-crawlable URL for each business, e.g.:
    www.yelp.com/biz/crossfit-provo-provo

Adds `first_seen_year` and `years_on_yelp` to data/utah_fitness_v2.csv.
Checkpoints to data/wayback_progress.csv every 50 businesses.

Usage:
    python wayback_enrich.py
"""

import pandas as pd
import numpy as np
import requests
import time
import os

CURRENT_YEAR   = 2026
CDX_URL        = 'http://web.archive.org/cdx/search/cdx'
DATA_PATH      = 'data/utah_fitness_v2.csv'
ALIAS_PATH     = 'data/yelp_aliases.csv'
PROGRESS_PATH  = 'data/wayback_progress.csv'
SLEEP_SEC      = 0.5
REQUEST_TIMEOUT = 10


def get_first_wayback_year(alias, retries=1):
    """
    Queries Wayback CDX API for the earliest snapshot of a Yelp business page
    using its URL slug (alias), e.g. 'crossfit-provo-provo'.
    Returns the year as an int, or None if no snapshot exists.
    """
    url = f'www.yelp.com/biz/{alias}'
    params = {
        'url':    url,
        'output': 'json',
        'limit':  1,
        'fl':     'timestamp',
    }
    for attempt in range(retries + 1):
        try:
            resp = requests.get(CDX_URL, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                # data[0] = header ['timestamp'], data[1] = first result
                if len(data) >= 2:
                    return int(data[1][0][:4])
                return None
            elif resp.status_code == 403:
                print(f'  403 rate-limited — sleeping 60s...')
                time.sleep(60)
            else:
                time.sleep(2)
        except Exception:
            pass
        if attempt < retries:
            time.sleep(3)
    return None


def main():
    df = pd.read_csv(DATA_PATH, dtype={'zip_code': str})
    aliases = pd.read_csv(ALIAS_PATH)

    # Merge aliases in; drop businesses with no alias (can't look up)
    merged = df.merge(aliases, on='id', how='left')
    has_alias = merged['alias'].notna()
    print(f'Businesses with alias:    {has_alias.sum()}')
    print(f'Businesses without alias: {(~has_alias).sum()} (will be median-imputed)')

    # ── Resume from checkpoint ─────────────────────────────────────────────────
    if os.path.exists(PROGRESS_PATH):
        progress = pd.read_csv(PROGRESS_PATH)
        done_ids = set(progress['id'].tolist())
        print(f'Resuming: {len(done_ids)} businesses already done.')
    else:
        progress = pd.DataFrame(columns=['id', 'first_seen_year'])
        done_ids = set()

    # Only query businesses that have an alias and haven't been done yet
    todo = merged[has_alias & ~merged['id'].isin(done_ids)][['id', 'alias']].values.tolist()
    print(f'{len(todo)} remaining. Est. time: ~{len(todo) * 0.6 / 60:.0f} min\n')

    new_rows = []
    no_snapshot = 0

    for i, (biz_id, alias) in enumerate(todo):
        year = get_first_wayback_year(alias)
        new_rows.append({'id': biz_id, 'first_seen_year': year})
        if year is None:
            no_snapshot += 1

        if (i + 1) % 50 == 0:
            chunk = pd.DataFrame(new_rows)
            progress = pd.concat([progress, chunk], ignore_index=True)
            progress.to_csv(PROGRESS_PATH, index=False)
            new_rows = []
            found = (i + 1) - no_snapshot
            print(f'  {i+1}/{len(todo)} | found: {found} | no snapshot: {no_snapshot}')

        time.sleep(SLEEP_SEC)

    # Save remainder
    if new_rows:
        chunk = pd.DataFrame(new_rows)
        progress = pd.concat([progress, chunk], ignore_index=True)
        progress.to_csv(PROGRESS_PATH, index=False)

    # ── Merge back + derive features ──────────────────────────────────────────
    print('\nMerging back into main dataset...')
    df = df.merge(progress[['id', 'first_seen_year']], on='id', how='left')

    median_year = df['first_seen_year'].median()
    missing = df['first_seen_year'].isna().sum()
    print(f'Imputing {missing} missing with median ({median_year:.0f})')
    df['first_seen_year'] = df['first_seen_year'].fillna(median_year).clip(lower=2004, upper=CURRENT_YEAR)
    df['years_on_yelp']   = CURRENT_YEAR - df['first_seen_year']

    if 'has_hours' in df.columns:
        df = df.drop(columns=['has_hours'])
        print('Dropped has_hours.')

    df.to_csv(DATA_PATH, index=False)
    print(f'\nSaved {len(df)} rows to {DATA_PATH}')
    print(f'years_on_yelp:\n{df["years_on_yelp"].describe().round(2)}')
    print(f'\nFirst seen year distribution:')
    print(df['first_seen_year'].astype(int).value_counts().sort_index())


if __name__ == '__main__':
    main()
