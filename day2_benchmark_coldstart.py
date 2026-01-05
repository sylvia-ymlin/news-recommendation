#!/usr/bin/env python3
"""Day 2: Benchmark with cold-start optimized submission"""
import pandas as pd, numpy as np
from collections import defaultdict
from tqdm import tqdm

print("\n" + "="*80)
print("DAY 2: BENCHMARK (COLD START SUBMISSION)")
print("="*80 + "\n")

# Load data
test_df = pd.read_csv('./data/testA_click_log.csv')
submission_df = pd.read_csv('./temp_results/submission_coldstart.csv')

print(f"[1/3] Data loaded:")
print(f"  Test records: {len(test_df)}")
print(f"  Submission records: {len(submission_df)}")

# Build ground truth
test_users = test_df['user_id'].unique()
test_ground_truth = defaultdict(set)
for _, row in test_df.iterrows():
    test_ground_truth[row['user_id']].add(row['click_article_id'])

print(f"\n[2/3] Computing metrics...")
ks = [5, 10, 20, 50]
recalls = {k: [] for k in ks}
precisions = {k: [] for k in ks}

for user_id in tqdm(test_users, desc="Users", leave=False):
    recommendations = submission_df[submission_df['user_id'] == user_id].sort_values('rank')['article_id'].values
    ground_truth = test_ground_truth[user_id]
    
    if len(ground_truth) == 0:
        continue
    
    for k in ks:
        top_k = set(recommendations[:k])
        hits = len(top_k & ground_truth)
        recalls[k].append(hits / len(ground_truth))
        precisions[k].append(hits / k if k > 0 else 0)

# Results
print(f"\n[3/3] Results:")
print("-" * 60)
for k in ks:
    if recalls[k]:
        print(f"  Recall@{k:2d}:    {np.mean(recalls[k]):.4f}")
        print(f"  Precision@{k:2d}: {np.mean(precisions[k]):.4f}")
print("-" * 60)

print("\n✅ BENCHMARK COMPLETE!")
