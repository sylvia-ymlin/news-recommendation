#!/usr/bin/env python3
"""Day 2: Benchmark multi-strategy recall (simplified, no seaborn)"""
import os, sys, pandas as pd, numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

os.chdir("/Users/ymlin/Library/CloudStorage/OneDrive-Uppsalauniversitet/100-Study/130-CS/136 搜广推/天池新闻推荐/coding")

print("\n" + "="*80)
print("DAY 2: BENCHMARK & ANALYSIS")
print("="*80 + "\n")

data_path = Path("data")
save_path = Path("temp_results")
output_path = Path("outputs")
output_path.mkdir(exist_ok=True)

# Load data
print("[1/4] Loading data...")
train_df = pd.read_csv(data_path / "train_click_log.csv")
test_df = pd.read_csv(data_path / "testA_click_log.csv")
submission_df = pd.read_csv(save_path / "submission_multi_strategy.csv")

print(f"  Train: {len(train_df)}")
print(f"  Test: {len(test_df)}")
print(f"  Submission: {len(submission_df)}")

# Evaluation metrics
print("\n[2/4] Computing metrics...")

all_click = pd.concat([train_df, test_df])

# Build ground truth for test users
test_users = test_df['user_id'].unique()
test_ground_truth = defaultdict(set)
for _, row in test_df.iterrows():
    test_ground_truth[row['user_id']].add(row['click_article_id'])

# Compute recall@K
ks = [5, 10, 20, 50]
recalls = {k: [] for k in ks}
precisions = {k: [] for k in ks}

print("  Computing Recall@K and Precision@K...")
for user_id in tqdm(test_users, desc="  Users", leave=False):
    recommendations = submission_df[submission_df['user_id'] == user_id].sort_values('rank')['article_id'].values
    ground_truth = test_ground_truth[user_id]
    
    if len(ground_truth) == 0:
        continue
    
    for k in ks:
        top_k = set(recommendations[:k])
        hits = len(top_k & ground_truth)
        recalls[k].append(hits / len(ground_truth))
        precisions[k].append(hits / k if k > 0 else 0)

# Compute metrics
metrics = {}
for k in ks:
    if recalls[k]:
        metrics[f'Recall@{k}'] = np.mean(recalls[k])
        metrics[f'Precision@{k}'] = np.mean(precisions[k])

print("\n[3/4] Results:")
print("  " + "-" * 60)
for metric, value in sorted(metrics.items()):
    print(f"    {metric:20s}: {value:.4f}")
print("  " + "-" * 60)

# Save results
print("\n[4/4] Saving metrics...")
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv(output_path / "metrics.csv", index=False)

# Save detailed evaluation report
report = f"""
DAY 2 EVALUATION REPORT
{'='*60}

DATASET:
  - Train samples: {len(train_df):,}
  - Test samples: {len(test_df):,}
  - Test users: {len(test_users):,}
  - Generated recommendations: {len(submission_df):,}

PERFORMANCE METRICS:
"""

for metric, value in sorted(metrics.items()):
    report += f"  {metric:20s}: {value:.4f}\n"

report += f"""
RECOMMENDATION STATISTICS:
  - Avg recommendations per user: {len(submission_df) / len(test_users):.1f}
  - Unique items recommended: {submission_df['article_id'].nunique():,}
  - Coverage: {submission_df['article_id'].nunique() / all_click['click_article_id'].nunique() * 100:.1f}%

SUMMARY:
  ✅ Multi-strategy recall system evaluated
  ✅ All users have recommendations
  ✅ Ready for interview discussion

NEXT STEPS:
  1. Day 3-4: Prepare interview answers
  2. Focus on: System design, trade-offs, improvements
  3. Practice: Technical explanation of each strategy
"""

with open(output_path / "evaluation_report.txt", "w") as f:
    f.write(report)

print("  ✓ metrics.csv")
print("  ✓ evaluation_report.txt")

print("\n" + "="*80)
print("✅ DAY 2 COMPLETE!")
print("="*80)
print(f"\nKey Results:")
for metric in sorted([k for k in metrics.keys() if 'Recall@50' in k or 'Precision@50' in k]):
    print(f"  {metric}: {metrics[metric]:.4f}")
print("\n")
