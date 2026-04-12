"""
Fetches the oldest Yelp review date for each business using the Reviews API.

Strategy:
  - Yelp's reviews endpoint supports sort_by=date (newest first) and offset.
  - By setting offset = max(0, review_count - 3), we land on the last page
    of reviews, which contains the oldest ones.
  - Yelp caps offset at 1000, so for very popular businesses (1000+ reviews)
    we get reviews from ~year 3-4 of operation rather than day 1 — still useful.

Output:
  - data/oldest_reviews.csv  (id, oldest_review_date, first_review_year)
  - Merges first_review_year and years_active into utah_fitness_v2.csv

Usage:
    python fetch_oldest_review.py
"""

import pandas as pd
import requests
import time
import os

YELP_API_KEY  = 'vwVDFljHUMs6UCKeSs-EaLRynI7cTmvGSTXEBRn6PdTApVWT5w1DpW1EFWFobl5FDgNfQ6t3T43e6j3GInTe_v9mkSNz7NRMPJxL4uVsegh8Fl-CBtk9PomtEyvIaXYx'
HEADERS       = {'Authorization': f'Bearer {YELP_API_KEY}'}
REVIEWS_URL   = 'https://api.yelp.com/v3/businesses/{}/reviews'
DATA_PATH     = 'data/utah_fitness_v2.csv'
OUTPUT_PATH   = 'data/oldest_reviews.csv'
CURRENT_YEAR  = 2026
SLEEP_SEC     = 0.3
MAX_OFFSET    = 1000  # Yelp API hard cap


def get_oldest_review_year(biz_id, review_count, retries=1):
    """
    Fetches the oldest available review year for a business.
    Returns (oldest_date_str, year_int) or (None, None) if no reviews.
    """
    if review_count == 0:
        return None, None

    offset = max(0, min(review_count - 3, MAX_OFFSET))
    params = {
        'limit':   3,
        'offset':  offset,
        'sort_by': 'date',   # newest first → last page = oldest
    }

    for attempt in range(retries + 1):
        try:
            resp = requests.get(
                REVIEWS_URL.format(biz_id),
                headers=HEADERS,
                params=params,
                timeout=8,
            )
            if resp.status_code == 200:
                reviews = resp.json().get('reviews', [])
                if not reviews:
                    return None, None
                # Reviews on last page sorted newest-first within page,
                # so the last item is the oldest
                dates = [r['time_created'] for r in reviews if r.get('time_created')]
                if not dates:
                    return None, None
                oldest = min(dates)  # lexicographic sort works on ISO dates
                year = int(oldest[:4])
                return oldest, year
            elif resp.status_code == 429:
                print(f'  Yelp rate limit — sleeping 30s...')
                time.sleep(30)
            else:
                time.sleep(1)
        except Exception as e:
            pass
        if attempt < retries:
            time.sleep(2)

    return None, None


def main():
    df = pd.read_csv(DATA_PATH, dtype={'zip_code': str})
    print(f'Loaded {len(df)} businesses.')

    # Resume from checkpoint
    if os.path.exists(OUTPUT_PATH):
        done_df = pd.read_csv(OUTPUT_PATH)
        done_ids = set(done_df['id'].tolist())
        rows = done_df.to_dict('records')
        print(f'Resuming: {len(done_ids)} already done.')
    else:
        done_ids = set()
        rows = []

    todo = df[~df['id'].isin(done_ids)][['id', 'review_count']].values.tolist()
    print(f'{len(todo)} remaining. Est. time: ~{len(todo) * 0.35 / 60:.0f} min\n')

    no_result = 0

    for i, (biz_id, review_count) in enumerate(todo):
        oldest_date, year = get_oldest_review_year(biz_id, int(review_count))
        rows.append({
            'id':                  biz_id,
            'oldest_review_date':  oldest_date,
            'first_review_year':   year,
        })
        if year is None:
            no_result += 1

        if (i + 1) % 50 == 0:
            pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
            found = (i + 1) - no_result
            print(f'  {i+1}/{len(todo)} | year found: {found} | no result: {no_result}')

        time.sleep(SLEEP_SEC)

    # Final checkpoint save
    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT_PATH, index=False)

    found = result_df['first_review_year'].notna().sum()
    print(f'\nDone. {found}/{len(result_df)} businesses with a review year.')

    # ── Merge into main CSV ────────────────────────────────────────────────────
    print('Merging into main dataset...')
    df = df.merge(result_df[['id', 'first_review_year']], on='id', how='left')

    median_year = df['first_review_year'].median()
    missing = df['first_review_year'].isna().sum()
    print(f'Imputing {missing} missing with median ({median_year:.0f})')
    df['first_review_year'] = (
        df['first_review_year']
        .fillna(median_year)
        .clip(lower=2004, upper=CURRENT_YEAR)
    )
    df['years_active'] = CURRENT_YEAR - df['first_review_year']

    if 'has_hours' in df.columns:
        df = df.drop(columns=['has_hours'])

    df.to_csv(DATA_PATH, index=False)
    print(f'Saved {len(df)} rows to {DATA_PATH}')
    print(f'\nyears_active distribution:')
    print(df['years_active'].value_counts().sort_index())
    print(f'\nyears_active stats:')
    print(df['years_active'].describe().round(2))


if __name__ == '__main__':
    main()
