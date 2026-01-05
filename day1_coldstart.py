#!/usr/bin/env python3
"""Day 1: Pure popularity-based recommendations (cold start)"""
import pandas as pd, numpy as np
from pathlib import Path
from tqdm import tqdm

print("\n" + "="*80)
print("DAY 1: PURE POPULARITY (COLD START SOLUTION)")
print("="*80 + "\n")

# Load data
train = pd.read_csv('./data/train_click_log.csv')
test_users_df = pd.read_csv('./data/testA_click_log.csv')
test_users = test_users_df['user_id'].unique()

# All test users are NEW (not in training)
print(f"[1/3] Data:")
print(f"  Test users: {len(test_users)}")
print(f"  All users are NEW (no training history)")

# Get top 50 popular items
print(f"\n[2/3] Computing popularity...")
item_popularity = train['click_article_id'].value_counts()
top_items = item_popularity.head(50).index.tolist()
print(f"  Top 50 items selected")

# Generate recommendations
print(f"\n[3/3] Generating recommendations...")
recommendations = []
for rank, item_id in enumerate(top_items, 1):
    for user_id in test_users:
        recommendations.append((user_id, item_id, rank))

df = pd.DataFrame(recommendations, columns=['user_id', 'article_id', 'rank'])
df.to_csv('./temp_results/submission_coldstart.csv', index=False)

print(f"✓ Generated {len(recommendations)} records")
print(f"✓ Saved: temp_results/submission_coldstart.csv")
print("\n" + "="*80)
