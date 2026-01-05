#!/usr/bin/env python3
"""
Direct Day 1 execution: Multi-strategy recall for news recommendation
Simplified from notebook without complex dependencies
"""
import os
import sys
from pathlib import Path
import subprocess
import json

# Ensure we're in the right directory
os.chdir("/Users/ymlin/Library/CloudStorage/OneDrive-Uppsalauniversitet/100-Study/130-CS/136 搜广推/天池新闻推荐/coding")

print("=" * 80)
print("DAY 1: MULTI-STRATEGY RECALL EXECUTION")
print("=" * 80)

# Install dependencies if not available
print("\n[1/3] Installing dependencies...")
deps = ["pandas", "numpy", "scikit-learn", "tqdm", "faiss-cpu"]
for pkg in deps:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        print(f"  Installing {pkg}...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

print("✓ Dependencies ready\n")

# Import core libraries
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
import math
from tqdm import tqdm
import faiss

print("[2/3] Loading and processing data...")

data_path = Path("data") / ""
save_path = Path("temp_results") / ""
data_path.parent.mkdir(parents=True, exist_ok=True)
save_path.parent.mkdir(parents=True, exist_ok=True)

# Load data
train_df = pd.read_csv(data_path / "train_click_log.csv")
test_df = pd.read_csv(data_path / "testA_click_log.csv")
articles_df = pd.read_csv(data_path / "articles.csv")
articles_emb_df = pd.read_csv(data_path / "articles_emb.csv")

print(f"  Train clicks: {len(train_df)}")
print(f"  Test clicks: {len(test_df)}")
print(f"  Articles: {len(articles_df)}")

# Combine for recall
all_click_df = pd.concat([train_df, test_df])
all_click_df = all_click_df.drop_duplicates(['user_id', 'click_article_id', 'click_timestamp'])

# Normalize embeddings
emb_cols = [c for c in articles_emb_df.columns if 'emb' in c]
emb_array = np.ascontiguousarray(articles_emb_df[emb_cols].values)
emb_array = emb_array / np.linalg.norm(emb_array, axis=1, keepdims=True)

# Build item embedding dict
item_emb_dict = dict(zip(articles_emb_df['article_id'], emb_array))
pickle.dump(item_emb_dict, open(save_path / "item_content_emb.pkl", "wb"))

print("✓ Data loaded\n")

print("[3/3] Computing recall strategies...")

# ItemCF: Item-to-item similarity using co-occurrence
print("  - ItemCF (collaborative filtering)...")
item_user_dict = defaultdict(set)
for _, row in all_click_df.iterrows():
    item_user_dict[row['click_article_id']].add(row['user_id'])

i2i_sim = {}
for item_a in tqdm(item_emb_dict.keys(), desc="ItemCF", leave=False):
    i2i_sim[item_a] = {}
    users_a = item_user_dict.get(item_a, set())
    if not users_a:
        continue
    for item_b in item_emb_dict.keys():
        if item_a == item_b:
            continue
        users_b = item_user_dict.get(item_b, set())
        common = len(users_a & users_b)
        if common > 0:
            sim = common / (len(users_a) + len(users_b) - common + 1)
            i2i_sim[item_a][item_b] = sim

pickle.dump(i2i_sim, open(save_path / "itemcf_i2i_sim.pkl", "wb"))

# Content-based: Embedding similarity
print("  - Content-based (embedding similarity)...")
index = faiss.IndexFlatIP(len(emb_cols))
items = list(item_emb_dict.keys())
index.add(np.array([item_emb_dict[i] for i in items]).astype('float32'))

# Final submission: For each test user, generate top 50 recommendations
print("  - Generating final recommendations...")
test_users = test_df['user_id'].unique()
submissions = []

for user_id in tqdm(test_users, desc="Submissions", leave=False):
    user_hist = set(all_click_df[all_click_df['user_id'] == user_id]['click_article_id'].unique())
    
    # Collect candidates from ItemCF
    candidates = defaultdict(float)
    for hist_item in user_hist:
        if hist_item in i2i_sim:
            for rec_item, score in i2i_sim[hist_item].items():
                if rec_item not in user_hist:
                    candidates[rec_item] += score
    
    # Score and rank
    recs = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:50]
    
    for rank, (item_id, score) in enumerate(recs, 1):
        submissions.append({
            'user_id': user_id,
            'article_id': item_id,
            'rank': rank
        })

submission_df = pd.DataFrame(submissions)
submission_df.to_csv(save_path / "submission_multi_strategy.csv", index=False)

print(f"\n✓ Submission created: {len(submission_df)} records")
print("=" * 80)
print("DAY 1 COMPLETE!")
print("=" * 80)
