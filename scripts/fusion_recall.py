#!/usr/bin/env python3
"""Retrieval fusion: Combine multiple recall channels with weighted strategy"""
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm
import time
import os

print('\n' + '='*80)
print('Retrieval Fusion System (Weighted Combination)')
print('='*80 + '\n')

start_time = time.time()

SAVE_PATH = '/root/autodl-tmp/news-rec-data/'
os.makedirs(SAVE_PATH, exist_ok=True)

# Load test users
test = pd.read_csv('./data/testA_click_log.csv')
test_users = sorted(test['user_id'].unique())
print(f'[1/4] Loading test users: {len(test_users):,}')

# Fusion weights: [popularity, ItemCF, Embedding, UserCF]
WEIGHTS = {
    'hot': 0.2,
    'itemcf': 0.3,
    'embedding': 0.3,
    'usercf': 0.2
}

# Load retrieval results
print('\n[2/4] Loading retrieval results...')

# Popularity
try:
    with open(SAVE_PATH + 'hot_list.pkl', 'rb') as f:
        hot_list = pickle.load(f)
    print(f'  ✓ Popularity: {len(hot_list)} items')
except FileNotFoundError:
    hot_list = []
    print('  ✗ Popularity: Not found')

# ItemCF
try:
    with open(SAVE_PATH + 'itemcf_sim.pkl', 'rb') as f:
        itemcf_sim = pickle.load(f)
    print(f'  ✓ ItemCF: {len(itemcf_sim)} items with similarity')
except FileNotFoundError:
    itemcf_sim = {}
    print('  ✗ ItemCF: Not found')

# Embedding
try:
    with open(SAVE_PATH + 'emb_sim_faiss.pkl', 'rb') as f:
        emb_sim = pickle.load(f)
    print(f'  ✓ Embedding: {len(emb_sim)} items with similarity')
except FileNotFoundError:
    emb_sim = {}
    print('  ✗ Embedding: Not found')

# UserCF
try:
    with open(SAVE_PATH + 'usercf_sim.pkl', 'rb') as f:
        usercf_sim = pickle.load(f)
    print(f'  ✓ UserCF: {len(usercf_sim)} user pairs')
except FileNotFoundError:
    usercf_sim = {}
    print('  ✗ UserCF: Not found (will be skipped)')

# Load user history from test set
test_user_hist = test.groupby('user_id')['click_article_id'].apply(list).to_dict()
print(f'\n[3/4] User history: {len(test_user_hist):,}')

# Fusion function with better ranking
def fuse_recalls(user_id):
    """
    Weighted combination of multiple retrieval channels using inverse rank scoring
    """
    scores = defaultdict(float)
    
    # 1. Popularity (0.2 weight)
    if hot_list:
        for rank, item in enumerate(hot_list[:200], 1):
            # Inverse rank: rank 1 -> 1.0, rank 2 -> 0.5, etc.
            scores[item] += WEIGHTS['hot'] / rank
    
    # 2. ItemCF (0.3 weight) - based on user history
    user_history = test_user_hist.get(user_id, [])
    if user_history and itemcf_sim:
        for hist_item in user_history[-10:]:  # Recent 10 items
            if hist_item in itemcf_sim:
                similar_items = itemcf_sim[hist_item]
                for rank, sim_item in enumerate(similar_items[:100], 1):
                    scores[sim_item] += WEIGHTS['itemcf'] / rank
    
    # 3. Embedding (0.3 weight) - content similarity
    if user_history and emb_sim:
        for hist_item in user_history[-10:]:  # Recent 10 items
            if hist_item in emb_sim:
                similar_items = emb_sim[hist_item]
                for rank, sim_item in enumerate(similar_items[:100], 1):
                    scores[sim_item] += WEIGHTS['embedding'] / rank
    
    # 4. UserCF (0.2 weight) - similar users' items
    if user_id in usercf_sim and usercf_sim:
        similar_users = usercf_sim[user_id]
        for sim_user in similar_users[:20]:  # Top-20 similar users
            sim_user_hist = test_user_hist.get(sim_user, [])
            for hist_item in sim_user_hist[-5:]:
                # Discounted weight for collaborative filtering
                scores[hist_item] += WEIGHTS['usercf'] * 0.3 / 20  # Divided by num similar users
    
    # Fallback: ensure minimum diversity with popularity
    if not scores and hot_list:
        return hot_list[:200]
    
    # Return top candidates by weighted score (sorted descending)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:200]

# Fuse recalls for each test user
print('\n[4/4] Fusing retrieval results...')
fused_recalls = {}

for user_id in tqdm(test_users, desc='Fusion progress'):
    fused_result = fuse_recalls(user_id)
    fused_recalls[user_id] = [aid for aid, score in fused_result] if fused_result else []

# Save fused recalls
output_path = os.path.join(SAVE_PATH, 'fused_recalls.pkl')
with open(output_path, 'wb') as f:
    pickle.dump(fused_recalls, f)

# Statistics
avg_candidates = np.mean([len(v) for v in fused_recalls.values()])
total_unique = len(set(sum(fused_recalls.values(), [])))

print('\n' + '='*80)
print('Fusion Complete')
print('='*80)
print(f'✓ Test users with fused recalls: {len(fused_recalls):,}')
print(f'✓ Avg candidates per user: {avg_candidates:.0f}')
print(f'✓ Total unique articles: {total_unique:,}')
print(f'✓ Output file: {output_path}')
print(f'✓ Runtime: {(time.time() - start_time):.1f} seconds')
print('='*80)
