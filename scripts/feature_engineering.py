#!/usr/bin/env python3
"""特征工程 - 提取用户/文章/交叉特征"""
import pandas as pd
import numpy as np
import pickle
from collections import Counter, defaultdict
from tqdm import tqdm
import time

print('\n' + '='*80)
print('特征工程 - 提取用户/文章/交叉特征')
print('='*80 + '\n')

start_time = time.time()

# ==================== 加载数据 ====================
print('[1/6] 加载数据...')
train = pd.read_csv('./data/train_click_log.csv')
articles = pd.read_csv('./data/articles.csv')

# 转换时间戳
train['click_timestamp'] = pd.to_datetime(train['click_timestamp'], unit='ms')
articles['created_at_ts'] = pd.to_datetime(articles['created_at_ts'], unit='ms')

print(f'  训练记录: {len(train):,}')
print(f'  文章数: {len(articles):,}')

# ==================== 用户特征 ====================
print('\n[2/6] 提取用户特征...')

user_features = {}

for user_id, group in tqdm(train.groupby('user_id'), desc='  用户'):
    # 基础统计
    click_count = len(group)
    unique_articles = group['click_article_id'].nunique()
    
    # 时间特征
    time_span = (group['click_timestamp'].max() - group['click_timestamp'].min()).total_seconds() / 3600  # 小时
    click_times = group['click_timestamp'].dt.hour.values
    hour_dist = Counter(click_times)
    
    # 类别偏好
    article_ids = group['click_article_id'].values
    categories = articles[articles['article_id'].isin(article_ids)]['category_id'].values
    cat_counts = Counter(categories)
    top_cat = cat_counts.most_common(1)[0][0] if cat_counts else 0
    cat_entropy = -sum((c/len(categories)) * np.log(c/len(categories) + 1e-10) 
                       for c in cat_counts.values()) if len(cat_counts) > 0 else 0
    
    user_features[user_id] = {
        'click_count': click_count,
        'unique_articles': unique_articles,
        'diversity_ratio': unique_articles / click_count if click_count > 0 else 0,
        'time_span_hours': time_span,
        'avg_hour': np.mean(click_times),
        'most_active_hour': hour_dist.most_common(1)[0][0] if hour_dist else 12,
        'top_category': top_cat,
        'category_count': len(cat_counts),
        'category_entropy': cat_entropy,
    }

print(f'  用户特征: {len(user_features)} users x {len(list(user_features.values())[0])} features')

# ==================== 文章特征 ====================
print('\n[3/6] 提取文章特征...')

# 点击统计
article_clicks = train['click_article_id'].value_counts().to_dict()
article_users = train.groupby('click_article_id')['user_id'].nunique().to_dict()

# 时间特征
article_first_click = train.groupby('click_article_id')['click_timestamp'].min().to_dict()
article_last_click = train.groupby('click_article_id')['click_timestamp'].max().to_dict()

article_features = {}

for _, row in tqdm(articles.iterrows(), total=len(articles), desc='  文章'):
    aid = row['article_id']
    
    # 基础特征
    words = row['words_count'] if pd.notna(row['words_count']) else 0
    category = row['category_id'] if pd.notna(row['category_id']) else 0
    created_at = row['created_at_ts']
    
    # 热度特征
    click_cnt = article_clicks.get(aid, 0)
    user_cnt = article_users.get(aid, 0)
    
    # 时间特征
    if aid in article_first_click:
        first_click = article_first_click[aid]
        last_click = article_last_click[aid]
        lifespan = (last_click - first_click).total_seconds() / 3600
        age_at_first_click = (first_click - created_at).total_seconds() / 3600
    else:
        lifespan = 0
        age_at_first_click = 0
    
    article_features[aid] = {
        'words_count': words,
        'category_id': category,
        'click_count': click_cnt,
        'unique_users': user_cnt,
        'click_rate': click_cnt / user_cnt if user_cnt > 0 else 0,
        'lifespan_hours': lifespan,
        'age_at_first_click_hours': age_at_first_click,
    }

print(f'  文章特征: {len(article_features)} articles x {len(list(article_features.values())[0])} features')

# ==================== 类别特征 ====================
print('\n[4/6] 提取类别特征...')

category_stats = {}

for cat_id, group in articles.groupby('category_id'):
    cat_articles = group['article_id'].values
    cat_clicks = train[train['click_article_id'].isin(cat_articles)]
    
    category_stats[cat_id] = {
        'article_count': len(cat_articles),
        'total_clicks': len(cat_clicks),
        'unique_users': cat_clicks['user_id'].nunique() if len(cat_clicks) > 0 else 0,
        'avg_clicks_per_article': len(cat_clicks) / len(cat_articles) if len(cat_articles) > 0 else 0,
    }

print(f'  类别特征: {len(category_stats)} categories')

# ==================== 用户-文章历史交互 ====================
print('\n[5/6] 构建交互历史...')

user_history = train.groupby('user_id')['click_article_id'].apply(set).to_dict()
print(f'  用户历史: {len(user_history)} users')

# ==================== 保存特征 ====================
print('\n[6/6] 保存特征...')

features = {
    'user_features': user_features,
    'article_features': article_features,
    'category_stats': category_stats,
    'user_history': user_history,
}

with open('./temp_results/features.pkl', 'wb') as f:
    pickle.dump(features, f)

# 保存统计摘要
summary = {
    'user_count': len(user_features),
    'article_count': len(article_features),
    'category_count': len(category_stats),
    'user_feature_dim': len(list(user_features.values())[0]),
    'article_feature_dim': len(list(article_features.values())[0]),
}

with open('./outputs/feature_summary.txt', 'w') as f:
    f.write('特征工程摘要\n')
    f.write('='*60 + '\n\n')
    for key, value in summary.items():
        f.write(f'{key}: {value:,}\n')

elapsed = time.time() - start_time

print('\n' + '='*80)
print('✅ 完成！')
print('='*80)
print(f'用户特征: {len(user_features):,} users x {len(list(user_features.values())[0])} dims')
print(f'文章特征: {len(article_features):,} articles x {len(list(article_features.values())[0])} dims')
print(f'类别特征: {len(category_stats):,} categories')
print(f'执行时间: {elapsed:.1f}秒')
print(f'保存路径: temp_results/features.pkl')
print('='*80)
