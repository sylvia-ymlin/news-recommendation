#!/usr/bin/env python3
"""Day 1 Fixed: Better recommendation logic"""
import os, sys, subprocess
from pathlib import Path

os.chdir("/Users/ymlin/Library/CloudStorage/OneDrive-Uppsalauniversitet/100-Study/130-CS/136 搜广推/天池新闻推荐/coding")

print("\n" + "="*80)
print("DAY 1 IMPROVED: Multi-Strategy Recall (Fixed)")
print("="*80 + "\n")

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from collections import defaultdict
from tqdm import tqdm

data_path = Path("data")
save_path = Path("temp_results")

print("[Stage 1/3] Loading data...")
train_df = pd.read_csv(data_path / "train_click_log.csv")
test_df = pd.read_csv(data_path / "testA_click_log.csv")

all_click = pd.concat([train_df, test_df])
all_click = all_click.drop_duplicates(['user_id', 'click_article_id', 'click_timestamp'])

print(f"  Data: {len(all_click)} total clicks")

# Get all items and compute popularity
print("[Stage 2/3] Computing item popularity...")
item_popularity = train_df['click_article_id'].value_counts().to_dict()
all_items = sorted(all_click['click_article_id'].unique())

print(f"  Total unique articles: {len(all_items)}")

# Load pre-computed ItemCF if available
has_itemcf = (save_path / "itemcf_i2i_sim.pkl").exists()
if has_itemcf:
    itemcf_i2i = pickle.load(open(save_path / "itemcf_i2i_sim.pkl", "rb"))
    print(f"  Using cached ItemCF: {len(itemcf_i2i)} items")
else:
    itemcf_i2i = {}

print("\n[Stage 3/3] Generating recommendations for all users...")
test_users = test_df['user_id'].unique()
print(f"  Target: {len(test_users)} test users")

submissions = []

for user_id in tqdm(test_users, desc="  Processing", unit="user"):
    # User history from train + test
    user_hist = set(all_click[all_click['user_id'] == user_id]['click_article_id'].unique())
    
    # Collect candidates from ItemCF
    candidates = defaultdict(float)
    for hist_item in user_hist:
        if hist_item in itemcf_i2i:
            for rec_item, score in sorted(itemcf_i2i[hist_item].items(), 
                                         key=lambda x: x[1], reverse=True)[:100]:
                if rec_item not in user_hist:
                    candidates[rec_item] += score
    
    # Sort by score
    recs = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    
    # If less than 50, pad with popular unseen items
    if len(recs) < 50:
        unseen = [a for a in all_items if a not in user_hist and a not in [r[0] for r in recs]]
        for item in sorted(unseen, key=lambda x: item_popularity.get(x, 0), reverse=True):
            if len(recs) >= 50:
                break
            recs.append((item, 0))
    
    # Take top 50
    for rank, (item_id, score) in enumerate(recs[:50], 1):
        submissions.append({
            'user_id': user_id,
            'article_id': item_id,
            'rank': rank
        })

submission_df = pd.DataFrame(submissions)
output_file = save_path / "submission_multi_strategy.csv"
submission_df.to_csv(output_file, index=False)

print(f"\n  ✓ Generated: {len(submission_df)} records")
print(f"  ✓ Saved: {output_file}")
print(f"  ✓ Unique items: {submission_df['article_id'].nunique()}")

print("\n" + "="*80)
print("✅ DAY 1 IMPROVED COMPLETE!")
print("="*80 + "\n")
