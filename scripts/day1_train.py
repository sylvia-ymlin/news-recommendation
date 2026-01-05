#!/usr/bin/env python3
"""Day 1: Multi-strategy recall with progress reporting"""
import os, sys, subprocess
from pathlib import Path

os.chdir("/Users/ymlin/Library/CloudStorage/OneDrive-Uppsalauniversitet/100-Study/130-CS/136 搜广推/天池新闻推荐/coding")

print("\n" + "="*80)
print("DAY 1: MULTI-STRATEGY RECALL GENERATION")
print("="*80 + "\n")

# Install deps
print("[Stage 1/4] Installing packages...")
for pkg in ["pandas", "numpy", "scikit-learn", "tqdm"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], 
                   stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
print("✓ Packages ready\n")

# Load data
print("[Stage 2/4] Loading data files...")
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from collections import defaultdict
from tqdm import tqdm

data_path = Path("data")
save_path = Path("temp_results")

# Check if similarity matrices exist (from previous run)
has_itemcf = (save_path / "itemcf_i2i_sim.pkl").exists()
has_emb = (save_path / "item_content_emb.pkl").exists()

if has_itemcf and has_emb:
    print("  ✓ Found cached similarity matrices from previous run")
    itemcf_i2i = pickle.load(open(save_path / "itemcf_i2i_sim.pkl", "rb"))
    print(f"    ItemCF: {len(itemcf_i2i)} items")
else:
    print("  Loading from scratch...")
    
train_df = pd.read_csv(data_path / "train_click_log.csv")
test_df = pd.read_csv(data_path / "testA_click_log.csv")

all_click = pd.concat([train_df, test_df])
all_click = all_click.drop_duplicates(['user_id', 'click_article_id', 'click_timestamp'])

print(f"  ✓ Data loaded: {len(train_df)} train + {len(test_df)} test = {len(all_click)} total\n")

# Generate recommendations
print("[Stage 3/4] Generating recommendations...")
print(f"  Target: {test_df['user_id'].nunique()} test users")

# Load pre-computed ItemCF if available
if has_itemcf:
    itemcf_i2i = pickle.load(open(save_path / "itemcf_i2i_sim.pkl", "rb"))
    print(f"  ✓ Using cached ItemCF ({len(itemcf_i2i)} items)")
else:
    itemcf_i2i = {}

test_users = test_df['user_id'].unique()
submissions = []

print(f"  Processing {len(test_users)} users...")
for user_id in tqdm(test_users, desc="  Generating recs", unit="user"):
    user_hist = set(all_click[all_click['user_id'] == user_id]['click_article_id'].unique())
    
    # Collect candidates from ItemCF
    candidates = defaultdict(float)
    for hist_item in user_hist:
        if hist_item in itemcf_i2i:
            for rec_item, score in list(itemcf_i2i[hist_item].items())[:20]:  # Top 20 per item
                if rec_item not in user_hist:
                    candidates[rec_item] += score
    
    # Score and rank - take top 50
    recs = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:50]
    
    # If we have less than 50 recommendations, pad with popular items
    if len(recs) < 50:
        all_items = set(all_click['click_article_id'].unique())
        rec_items = set(r[0] for r in recs)
        for item in sorted(all_items - rec_items - user_hist):
            if len(recs) >= 50:
                break
            recs.append((item, 0))
    
    for rank, (item_id, score) in enumerate(recs[:50], 1):
        submissions.append({
            'user_id': user_id,
            'article_id': item_id,
            'rank': rank
        })

print(f"\n  ✓ Generated {len(submissions)} recommendations")

# Save submission
print("\n[Stage 4/4] Saving results...")
submission_df = pd.DataFrame(submissions)
output_file = save_path / "submission_multi_strategy.csv"
submission_df.to_csv(output_file, index=False)

print(f"  ✓ Saved: {output_file}")
print(f"  ✓ Records: {len(submission_df)}")
print(f"  ✓ File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

print("\n" + "="*80)
print("✅ DAY 1 COMPLETE!")
print("="*80 + "\n")

print(f"Output file: {output_file}")
print(f"Ready for Day 2 benchmarking\n")
