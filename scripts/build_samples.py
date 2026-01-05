#!/usr/bin/env python3
"""构建训练样本 - 正负样本采样"""
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm
import time
import random
import os

print('\n' + '='*80)
print('构建训练样本 - 正负样本采样')
print('='*80 + '\n')

start_time = time.time()

# 配置
NEG_SAMPLE_RATIO = 4  # 每个正样本采样4个负样本
MAX_POSITIVE_SAMPLES = 1112623  # 使用全部正样本！现在有100GB数据盘可用
RANDOM_SEED = 42
SAVE_PATH = '/root/autodl-tmp/news-rec-data/'  # 使用100GB数据盘
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ==================== 加载数据 ====================
print('[1/5] 加载数据...')

# 创建保存目录
os.makedirs(SAVE_PATH, exist_ok=True)

train = pd.read_csv('./data/train_click_log.csv')

# 加载特征
with open('./temp_results/features.pkl', 'rb') as f:
    features = pickle.load(f)

user_features = features['user_features']
article_features = features['article_features']
category_stats = features['category_stats']
user_history = features['user_history']

print(f'  训练记录: {len(train):,}')
print(f'  用户特征: {len(user_features):,}')
print(f'  文章特征: {len(article_features):,}')

# ==================== 准备文章池 ====================
print('\n[2/5] 准备文章池...')

# 所有文章
all_articles = set(article_features.keys())
print(f'  文章池: {len(all_articles):,}')

# 类别文章索引
cat_articles = defaultdict(list)
for aid, feat in article_features.items():
    cat_articles[feat['category_id']].append(aid)
print(f'  类别数: {len(cat_articles)}')

# ==================== 构建正样本 ====================
print('\n[3/5] 构建正样本...')

positive_samples = []

# 采样部分正样本
sampled_indices = np.random.choice(len(train), min(MAX_POSITIVE_SAMPLES, len(train)), replace=False)

for idx in tqdm(sampled_indices, desc='  正样本'):
    row = train.iloc[idx]
    user_id = row['user_id']
    article_id = row['click_article_id']
    
    if user_id not in user_features or article_id not in article_features:
        continue
    
    positive_samples.append({
        'user_id': user_id,
        'article_id': article_id,
        'label': 1
    })

print(f'  正样本: {len(positive_samples):,}')

# ==================== 负采样 ====================
print('\n[4/5] 负采样...')

negative_samples = []

for sample in tqdm(positive_samples, desc='  负样本'):
    user_id = sample['user_id']
    
    # 用户点击历史
    clicked = user_history.get(user_id, set())
    
    # 用户偏好类别
    user_top_cat = user_features[user_id]['top_category']
    
    # 负采样策略：80%从同类别采样（hard negative），20%随机采样
    neg_count = 0
    attempts = 0
    max_attempts = NEG_SAMPLE_RATIO * 20
    
    while neg_count < NEG_SAMPLE_RATIO and attempts < max_attempts:
        attempts += 1
        
        # 80%同类别，20%随机
        if random.random() < 0.8 and user_top_cat in cat_articles:
            neg_article = random.choice(cat_articles[user_top_cat])
        else:
            neg_article = random.choice(list(all_articles))
        
        # 确保未被点击
        if neg_article not in clicked:
            negative_samples.append({
                'user_id': user_id,
                'article_id': neg_article,
                'label': 0
            })
            neg_count += 1

print(f'  负样本: {len(negative_samples):,}')

# ==================== 合并并构造特征 ====================
print('\n[5/5] 构造特征向量...')

all_samples = positive_samples + negative_samples
random.shuffle(all_samples)

X = []
y = []

for sample in tqdm(all_samples, desc='  特征'):
    user_id = sample['user_id']
    article_id = sample['article_id']
    
    # 用户特征
    uf = user_features[user_id]
    # 文章特征
    af = article_features[article_id]
    # 类别特征
    cat_id = af['category_id']
    cf = category_stats.get(cat_id, {
        'article_count': 0,
        'total_clicks': 0,
        'unique_users': 0,
        'avg_clicks_per_article': 0
    })
    
    # 交叉特征
    category_match = 1 if uf['top_category'] == cat_id else 0
    
    # 特征向量
    feature_vec = [
        # 用户特征 (9)
        uf['click_count'],
        uf['unique_articles'],
        uf['diversity_ratio'],
        uf['time_span_hours'],
        uf['avg_hour'],
        uf['most_active_hour'],
        uf['category_count'],
        uf['category_entropy'],
        uf['top_category'],
        # 文章特征 (7)
        af['words_count'],
        af['category_id'],
        af['click_count'],
        af['unique_users'],
        af['click_rate'],
        af['lifespan_hours'],
        af['age_at_first_click_hours'],
        # 类别特征 (4)
        cf['article_count'],
        cf['total_clicks'],
        cf['unique_users'],
        cf['avg_clicks_per_article'],
        # 交叉特征 (1)
        category_match,
    ]
    
    X.append(feature_vec)
    y.append(sample['label'])

X = np.array(X)
y = np.array(y)

print(f'  特征矩阵: {X.shape}')
print(f'  标签: {y.shape}')
print(f'  正样本: {y.sum():,} ({y.mean()*100:.1f}%)')

# ==================== 保存样本 ====================
print('\n保存样本...')

with open(SAVE_PATH + 'training_samples.pkl', 'wb') as f:
    pickle.dump({'X': X, 'y': y}, f)

elapsed = time.time() - start_time

print('\n' + '='*80)
print('✅ 完成！')
print('='*80)
print(f'总样本: {len(y):,}')
print(f'正样本: {y.sum():,} ({y.mean()*100:.1f}%)')
print(f'负样本: {(1-y).sum():,} ({(1-y.mean())*100:.1f}%)')
print(f'执行时间: {elapsed:.1f}秒')
print(f'保存路径: {SAVE_PATH}training_samples.pkl')
print('='*80)
