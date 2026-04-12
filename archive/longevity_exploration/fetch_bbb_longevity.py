"""
Fetches "Business Started" year from the Better Business Bureau for each
fitness business in utah_fitness_v2.csv.

Strategy:
  1. Search BBB for each business name + city
  2. Fuzzy-match the top result to confirm it's the right business
  3. Load the profile page and extract "Business Started" year
  4. Checkpoint every 50 businesses to data/bbb_longevity.csv
  5. Merge `first_biz_year` and `years_active` into utah_fitness_v2.csv

Usage:
    python fetch_bbb_longevity.py
"""

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import time
import os
from rapidfuzz import fuzz

CURRENT_YEAR  = 2026
DATA_PATH     = 'data/utah_fitness_v2.csv'
OUTPUT_PATH   = 'data/bbb_longevity.csv'
SLEEP_SEC     = 0.5
FUZZY_THRESH  = 60   # minimum name similarity score to accept a match

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
                  'Chrome/120.0.0.0 Safari/537.36'
}


def search_bbb(name, city):
    """Search BBB and return list of (profile_url, biz_name) results."""
    query = f'{name} {city} UT'
    try:
        resp = requests.get(
            'https://www.bbb.org/search',
            params={'find_text': name, 'find_loc': f'{city}, UT'},
            headers=HEADERS, timeout=10
        )
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, 'html.parser')
        # Extract profile links and their text labels
        results = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            if '/profile/' in href and href.startswith('/us/ut/'):
                label = a.get_text(strip=True)
                if label:
                    results.append((href, label))
        return results[:5]
    except Exception:
        return []


def get_business_started_year(profile_path):
    """Load a BBB profile page and extract the Business Started year."""
    url = f'https://www.bbb.org{profile_path}'
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return None
        text = resp.get_text() if hasattr(resp, 'get_text') else BeautifulSoup(resp.text, 'html.parser').get_text()
        soup = BeautifulSoup(resp.text, 'html.parser')
        page_text = soup.get_text(separator=' ', strip=True)

        # Look for "Business Started: M/D/YYYY" or "Business Started: YYYY"
        match = re.search(r'Business Started[:\s|]+(\d{1,2}/\d{1,2}/(\d{4}))', page_text)
        if match:
            return int(match.group(2))

        match = re.search(r'Business Started[:\s|]+(\d{4})', page_text)
        if match:
            return int(match.group(1))

        # Fallback: "Years in Business: N" → subtract from current year
        match = re.search(r'Years in Business[:\s|]+(\d+)', page_text)
        if match:
            years = int(match.group(1))
            return CURRENT_YEAR - years

        return None
    except Exception:
        return None


def find_best_match(biz_name, results, threshold=FUZZY_THRESH):
    """Fuzzy-match business name against BBB results, return best profile path."""
    best_score = 0
    best_path  = None
    for path, label in results:
        score = fuzz.token_set_ratio(biz_name.lower(), label.lower())
        if score > best_score:
            best_score = score
            best_path  = path
    return best_path if best_score >= threshold else None


def main():
    df = pd.read_csv(DATA_PATH, dtype={'zip_code': str})
    print(f'Loaded {len(df)} businesses.')

    # Install rapidfuzz if missing
    try:
        from rapidfuzz import fuzz
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'rapidfuzz', '-q'])
        from rapidfuzz import fuzz

    # Resume from checkpoint
    if os.path.exists(OUTPUT_PATH):
        done_df = pd.read_csv(OUTPUT_PATH)
        done_ids = set(done_df['id'].tolist())
        rows = done_df.to_dict('records')
        print(f'Resuming: {len(done_ids)} already done.')
    else:
        done_ids = set()
        rows = []

    todo = df[~df['id'].isin(done_ids)][['id', 'name', 'city']].values.tolist()
    print(f'{len(todo)} remaining. Est. time: ~{len(todo) * 1.2 / 60:.0f} min\n')

    found_count = 0

    for i, (biz_id, name, city) in enumerate(todo):
        year = None
        try:
            results = search_bbb(name, city)
            if results:
                best_path = find_best_match(name, results)
                if best_path:
                    time.sleep(0.3)  # polite gap between search and profile
                    year = get_business_started_year(best_path)
        except Exception:
            pass

        rows.append({'id': biz_id, 'first_biz_year': year})
        if year:
            found_count += 1

        if (i + 1) % 50 == 0:
            pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
            print(f'  {i+1}/{len(todo)} | years found: {found_count}')

        time.sleep(SLEEP_SEC)

    # Final save
    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT_PATH, index=False)
    print(f'\nDone. {found_count}/{len(result_df)} businesses with a year.')

    # ── Merge back into main CSV ───────────────────────────────────────────────
    df = df.merge(result_df[['id', 'first_biz_year']], on='id', how='left')

    median_year = df['first_biz_year'].median()
    missing = df['first_biz_year'].isna().sum()
    print(f'Imputing {missing} missing with median ({median_year:.0f})')
    df['first_biz_year'] = (
        df['first_biz_year']
        .fillna(median_year)
        .clip(lower=1950, upper=CURRENT_YEAR)
    )
    df['years_active'] = CURRENT_YEAR - df['first_biz_year']

    if 'has_hours' in df.columns:
        df = df.drop(columns=['has_hours'])

    df.to_csv(DATA_PATH, index=False)
    print(f'Saved {len(df)} rows to {DATA_PATH}')
    print(f'\nyears_active stats:')
    print(df['years_active'].describe().round(2))
    print(f'\nYear distribution:')
    print(df['first_biz_year'].astype(int).value_counts().sort_index())


if __name__ == '__main__':
    main()
