#!/bin/bash
# Run this after fetch_oldest_review.py finishes (or instead of wayback).
# Retrains features and models end-to-end with years_active included.

set -e
cd "$(dirname "$0")"

echo "=== Step 1: Feature engineering ==="
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=300 \
  02_feature_engineering.ipynb

echo ""
echo "=== Step 2: Modeling ==="
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 \
  03_modeling.ipynb

echo ""
echo "=== ALL DONE ==="
