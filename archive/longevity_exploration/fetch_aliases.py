"""
Fetches the Yelp `alias` (URL slug) for each business in utah_fitness_v2.csv
using the Yelp Fusion detail endpoint.

The alias is the actual URL path used by Wayback Machine, e.g.:
    yelp.com/biz/crossfit-provo-provo

Saves results to data/yelp_aliases.csv (id, alias) so we don't need to
re-hit the Yelp API again.

Usage:
    python fetch_aliases.py
"""

import pandas as pd
import requests
import time
import os

YELP_API_KEY = 'vwVDFljHUMs6UCKeSs-EaLRynI7cTmvGSTXEBRn6PdTApVWT5w1DpW1EFWFobl5FDgNfQ6t3T43e6j3GInTe_v9mkSNz7NRMPJxL4uVsegh8Fl-CBtk9PomtEyvIaXYx'
HEADERS      = {'Authorization': f'Bearer {YELP_API_KEY}'}
DETAIL_URL   = 'https://api.yelp.com/v3/businesses/{}'
ALIAS_PATH   = 'data/yelp_aliases.csv'
DATA_PATH    = 'data/utah_fitness_v2.csv'
SLEEP_SEC    = 0.3


def get_alias(biz_id, retries=1):
    for attempt in range(retries + 1):
        try:
            resp = requests.get(DETAIL_URL.format(biz_id), headers=HEADERS, timeout=8)
            if resp.status_code == 200:
                return resp.json().get('alias')
            elif resp.status_code == 429:
                print(f'  Rate limited on Yelp — sleeping 30s...')
                time.sleep(30)
        except Exception:
            pass
        if attempt < retries:
            time.sleep(1)
    return None


def main():
    df = pd.read_csv(DATA_PATH, dtype={'zip_code': str})
    biz_ids = df['id'].tolist()
    print(f'Fetching aliases for {len(biz_ids)} businesses...')

    # Resume if partial file exists
    if os.path.exists(ALIAS_PATH):
        done = pd.read_csv(ALIAS_PATH)
        done_ids = set(done['id'].tolist())
        print(f'Resuming: {len(done_ids)} already done.')
        rows = done.to_dict('records')
    else:
        done_ids = set()
        rows = []

    todo = [b for b in biz_ids if b not in done_ids]
    print(f'{len(todo)} remaining. Est. time: ~{len(todo) * 0.4 / 60:.0f} min\n')

    for i, biz_id in enumerate(todo):
        alias = get_alias(biz_id)
        rows.append({'id': biz_id, 'alias': alias})

        if (i + 1) % 50 == 0:
            pd.DataFrame(rows).to_csv(ALIAS_PATH, index=False)
            found = sum(1 for r in rows if r['alias'] is not None)
            print(f'  {i+1}/{len(todo)} done | aliases found: {found}')

        time.sleep(SLEEP_SEC)

    # Final save
    alias_df = pd.DataFrame(rows)
    alias_df.to_csv(ALIAS_PATH, index=False)

    found = alias_df['alias'].notna().sum()
    print(f'\nDone. {found}/{len(alias_df)} aliases retrieved.')
    print(f'Saved to {ALIAS_PATH}')
    print(alias_df.head(10))


if __name__ == '__main__':
    main()
