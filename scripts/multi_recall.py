#!/usr/bin/env python3
"""多路召回系统 - ItemCF + Embedding + UserCF + 热度"""
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict, Counter
from tqdm import tqdm
import time
import os

print('\n' + '='*80)
print('多路召回系统')
print('='*80 + '\n')

start_time = time.time()

# 配置
SAVE_PATH = '/root/autodl-tmp/news-rec-data/'
os.makedirs(SAVE_PATH, exist_ok=True)

# ==================== 1. 加载数据 ====================
print('[1/6] 加载数据...')
train = pd.read_csv('./data/train_click_log.csv')
test = pd.read_csv('./data/testA_click_log.csv')
articles = pd.read_csv('./data/articles.csv')
articles_emb = pd.read_csv('./data/articles_emb.csv')

print(f'  训练: {len(train):,} 条')
print(f'  测试: {len(test):,} 条')
print(f'  文章: {len(articles):,} 篇')
print(f'  Embedding: {len(articles_emb):,} 篇')

# 用户历史
train_user_hist = train.groupby('user_id')['click_article_id'].apply(list).to_dict()
test_user_hist = test.groupby('user_id')['click_article_id'].apply(list).to_dict()
test_users = test['user_id'].unique()

# ==================== 2. 热度召回 ====================
print('\n[2/6] 热度召回...')
popularity = train['click_article_id'].value_counts()
hot_articles = popularity.index.tolist()[:200]
print(f'  Top-200热门文章')

# ==================== 3. ItemCF召回 ====================
print('\n[3/6] ItemCF协同过滤...')

# 构建物品-物品共现矩阵
item_sim = defaultdict(lambda: defaultdict(int))
user_items = defaultdict(set)

for uid, items in train_user_hist.items():
    user_items[uid] = set(items)
    for i in range(len(items)):
        for j in range(i+1, len(items)):
            item_sim[items[i]][items[j]] += 1
            item_sim[items[j]][items[i]] += 1

# 计算Jaccard相似度并取Top-100
itemcf_topk = {}
for item_i, sims in tqdm(item_sim.items(), desc='  计算相似度'):
    # 标准化：除以sqrt(count_i * count_j)
    count_i = popularity.get(item_i, 1)
    sim_scores = []
    for item_j, cooccur in sims.items():
        count_j = popularity.get(item_j, 1)
        sim = cooccur / np.sqrt(count_i * count_j)
        sim_scores.append((item_j, sim))
    # Top-100
    sim_scores.sort(key=lambda x: x[1], reverse=True)
    itemcf_topk[item_i] = [item for item, _ in sim_scores[:100]]

print(f'  ItemCF矩阵: {len(itemcf_topk)} items')

# 保存ItemCF
with open(SAVE_PATH + 'itemcf_sim.pkl', 'wb') as f:
    pickle.dump(itemcf_topk, f)

# ==================== 4. Embedding召回 ====================
print('\n[4/6] Embedding相似度...')

# 转换embedding为矩阵
emb_cols = [c for c in articles_emb.columns if c.startswith('emb_')]
emb_matrix = articles_emb[emb_cols].values
article_ids = articles_emb['article_id'].values
article_id_to_idx = {aid: idx for idx, aid in enumerate(article_ids)}

# 计算每篇文章的Top-100相似文章
emb_topk = {}
print('  计算cosine相似度...')
for idx, aid in enumerate(tqdm(article_ids[:10000], desc='  采样计算')):  # 只计算1万篇节省时间
    vec = emb_matrix[idx]
    # cosine相似度
    sims = np.dot(emb_matrix, vec) / (np.linalg.norm(emb_matrix, axis=1) * np.linalg.norm(vec) + 1e-10)
    top_idx = np.argsort(sims)[::-1][1:101]  # 排除自己，取Top-100
    emb_topk[aid] = article_ids[top_idx].tolist()

print(f'  Embedding相似度: {len(emb_topk)} items')

# 保存Embedding相似度
with open(SAVE_PATH + 'emb_sim.pkl', 'wb') as f:
    pickle.dump(emb_topk, f)

# ==================== 5. UserCF召回 ====================
print('\n[5/6] UserCF协同过滤...')

# 构建用户-用户相似度（基于共同点击的文章）
user_sim = defaultdict(lambda: defaultdict(int))

# 倒排：文章→点击用户
item_users = defaultdict(set)
for uid, items in train_user_hist.items():
    for item in items:
        item_users[item].add(uid)

# 计算用户相似度（只计算测试用户与训练用户的相似度）
test_train_user_sim = {}
for test_uid in tqdm(test_users, desc='  测试用户相似度'):
    if test_uid not in test_user_hist:
        continue
    test_items = set(test_user_hist[test_uid])
    sim_users = defaultdict(int)
    # 找到共同点击文章的训练用户
    for item in test_items:
        if item in item_users:
            for train_uid in item_users[item]:
                if train_uid in train_user_hist:
                    sim_users[train_uid] += 1
    # Jaccard相似度
    if sim_users:
        sim_scores = []
        for train_uid, overlap in sim_users.items():
            train_items = set(train_user_hist[train_uid])
            union = len(test_items | train_items)
            jaccard = overlap / union if union > 0 else 0
            sim_scores.append((train_uid, jaccard))
        # Top-50相似用户
        sim_scores.sort(key=lambda x: x[1], reverse=True)
        test_train_user_sim[test_uid] = [(uid, score) for uid, score in sim_scores[:50]]

print(f'  UserCF: {len(test_train_user_sim)} 测试用户有相似训练用户')

# 保存UserCF
with open(SAVE_PATH + 'usercf_sim.pkl', 'wb') as f:
    pickle.dump(test_train_user_sim, f)

# ==================== 6. 保存召回结果摘要 ====================
print('\n[6/6] 保存召回资源...')

recall_summary = {
    'hot_articles': hot_articles,
    'itemcf_count': len(itemcf_topk),
    'emb_count': len(emb_topk),
    'usercf_count': len(test_train_user_sim),
}

with open(SAVE_PATH + 'recall_summary.pkl', 'wb') as f:
    pickle.dump(recall_summary, f)

elapsed = time.time() - start_time

print('\n' + '='*80)
print('✅ 多路召回完成')
print('='*80)
print(f'热度召回: {len(hot_articles)} 篇')
print(f'ItemCF: {len(itemcf_topk)} items')
print(f'Embedding: {len(emb_topk)} items')
print(f'UserCF: {len(test_train_user_sim)} 测试用户')
print(f'执行时间: {elapsed/60:.1f} 分钟')
print(f'保存路径: {SAVE_PATH}')
print('='*80)
