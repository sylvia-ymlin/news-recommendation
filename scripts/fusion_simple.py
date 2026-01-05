#!/usr/bin/env python3
"""
简化版融合召回：仅使用已有的文件（ItemCF + Embedding）
"""
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm
import time

print('\n' + '='*80)
print('Simplified Fusion (ItemCF + Embedding + Popularity)')
print('='*80 + '\n')

start_time = time.time()

SAVE_PATH = '/root/autodl-tmp/news-rec-data/'

# Load data
print('[1/4] Loading data...')
train = pd.read_csv('./data/train_click_log.csv')
test = pd.read_csv('./data/testA_click_log.csv')

# Global popularity
print('[2/4] Computing popularity...')
popularity = train['click_article_id'].value_counts().index.tolist()[:2000]
print(f'  Top-2000 popular articles')

# Load ItemCF and Embedding
print('[3/4] Loading retrieval results...')
with open(SAVE_PATH + 'itemcf_sim.pkl', 'rb') as f:
    itemcf_sim = pickle.load(f)
print(f'  ItemCF: {len(itemcf_sim)} articles')

with open(SAVE_PATH + 'emb_sim_faiss.pkl', 'rb') as f:
    emb_sim = pickle.load(f)
print(f'  Embedding: {len(emb_sim)} articles')

# User history
test_user_hist = test.groupby('user_id')['click_article_id'].apply(list).to_dict()

# Fusion weights (只用3种方法)
WEIGHTS = {'hot': 0.25, 'itemcf': 0.4, 'embedding': 0.35}

def fuse_recalls(user_id):
    """Simplified fusion with 3 methods"""
    scores = defaultdict(float)
    
    # 1. Popularity
    for rank, item in enumerate(popularity[:300], 1):
        scores[item] += WEIGHTS['hot'] / rank
    
    # 2. ItemCF from user history
    user_history = test_user_hist.get(user_id, [])
    if user_history and itemcf_sim:
        for hist_item in user_history[-15:]:
            if hist_item in itemcf_sim:
                for rank, sim_item in enumerate(itemcf_sim[hist_item][:100], 1):
                    scores[sim_item] += WEIGHTS['itemcf'] / rank
    
    # 3. Embedding from user history
    if user_history and emb_sim:
        for hist_item in user_history[-15:]:
            if hist_item in emb_sim:
                for rank, sim_item in enumerate(emb_sim[hist_item][:100], 1):
                    scores[sim_item] += WEIGHTS['embedding'] / rank
    
    # Fallback
    if not scores:
        return popularity[:200]
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:200]

# Fusion
print('[4/4] Fusing...')
test_users = test['user_id'].unique()
fused_recalls = {}

for user_id in tqdm(test_users, desc='Progress'):
    result = fuse_recalls(user_id)
    fused_recalls[user_id] = [aid for aid, score in result] if result else []

# Save
output_path = SAVE_PATH + 'fused_recalls.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(fused_recalls, f)

# Stats
avg_candidates = np.mean([len(v) for v in fused_recalls.values()])
unique = len(set(sum(fused_recalls.values(), [])))

print('\n' + '='*80)
print(f'✓ Users: {len(fused_recalls):,}')
print(f'✓ Avg candidates: {avg_candidates:.0f}')
print(f'✓ Unique articles: {unique:,}')
print(f'✓ Output: {output_path}')
print(f'✓ Time: {time.time() - start_time:.1f}s')
print('='*80)
