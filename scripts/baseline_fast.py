#!/usr/bin/env python3
"""Baseline推荐系统 - 热度多样化版 (10秒快速版)"""
import pandas as pd
import numpy as np
import time
from collections import Counter
from tqdm import tqdm

print('\n' + '='*80)
print('Baseline推荐系统 - 快速热度版')
print('='*80 + '\n')

start_time = time.time()

# 配置
K_RECOMMENDATIONS = 50
SAVE_PATH = './temp_results/'

# ==================== 加载数据 ====================
print('[1/4] 加载数据...')
train = pd.read_csv('./data/train_click_log.csv')
test_users = pd.read_csv('./data/testA_click_log.csv')['user_id'].unique()
articles = pd.read_csv('./data/articles.csv')
article_cats = articles.set_index('article_id')['category_id'].to_dict()

print(f'  训练用户: {train["user_id"].nunique():,}')
print(f'  测试用户: {len(test_users):,}')
print(f'  文章数: {len(articles):,}')

# ==================== 构建向量 ====================
print('\n[2/4] 构建特征...')

# 全局热度
global_pop = train['click_article_id'].value_counts()
top_articles = global_pop.index.values

# 用户历史 & 类别偏好
user_history = train.groupby('user_id')['click_article_id'].apply(list).to_dict()
user_cats = {}
for uid in train['user_id'].unique():
    cat_counts = Counter()
    for aid in user_history.get(uid, []):
        if aid in article_cats:
            cat_counts[article_cats[aid]] += 1
    user_cats[uid] = cat_counts.most_common(1)[0][0] if cat_counts else 0

# 类别热度
cat_pop = {}
for cat_id in articles['category_id'].unique():
    cat_articles = articles[articles['category_id'] == cat_id]['article_id'].values
    cat_clicks = train[train['click_article_id'].isin(cat_articles)]['click_article_id'].value_counts()
    cat_pop[cat_id] = cat_clicks.index.values

print(f'  类别: {len(cat_pop)}')
print(f'  全局热度: {len(global_pop)} items')

# ==================== 生成推荐 ====================
print('\n[3/4] 生成推荐...')

all_recs = []
for user_id in tqdm(test_users, desc='  用户'):
    recs = []
    seen = set()
    
    # 1. 用户偏好类别热度
    if user_id in user_cats and user_id in user_history:
        fav_cat = user_cats[user_id]
        if fav_cat in cat_pop:
            for aid in cat_pop[fav_cat]:
                if aid not in seen and aid not in user_history.get(user_id, []):
                    recs.append(aid)
                    seen.add(aid)
                    if len(recs) >= K_RECOMMENDATIONS:
                        break
    
    # 2. 全局热度补充
    if len(recs) < K_RECOMMENDATIONS:
        for aid in top_articles:
            if aid not in seen and aid not in user_history.get(user_id, []):
                recs.append(aid)
                seen.add(aid)
                if len(recs) >= K_RECOMMENDATIONS:
                    break
    
    # 记录
    for rank, aid in enumerate(recs, 1):
        all_recs.append((user_id, int(aid), rank))

# ==================== 保存结果 ====================
print('\n[4/4] 保存结果...')
df = pd.DataFrame(all_recs, columns=['user_id', 'article_id', 'rank'])
df.to_csv(SAVE_PATH + 'submission_baseline.csv', index=False)

elapsed = time.time() - start_time

print('\n' + '='*80)
print('✅ 完成！')
print('='*80)
print(f'推荐记录: {len(df):,}')
print(f'独特文章: {df["article_id"].nunique():,}')
print(f'执行时间: {elapsed:.1f}秒')
print(f'吞吐量: {len(test_users)/elapsed:.0f} 用户/秒')
print('='*80)
