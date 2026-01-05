#!/usr/bin/env python3
"""提取测试用户特征（基于testA历史）"""
import pandas as pd
import numpy as np
import pickle
from collections import Counter
from tqdm import tqdm
import os

print('\n提取测试用户特征...')

# 加载数据
test = pd.read_csv('./data/testA_click_log.csv')
articles = pd.read_csv('./data/articles.csv')
article_cats = articles.set_index('article_id')['category_id'].to_dict()

# 提取测试用户特征
test_user_features = {}
test_user_history = {}

for uid, group in tqdm(test.groupby('user_id'), desc='  测试用户'):
    # 历史
    history = set(group['click_article_id'].values)
    test_user_history[uid] = history
    
    # 类别偏好
    cat_counts = Counter()
    for aid in history:
        if aid in article_cats:
            cat_counts[article_cats[aid]] += 1
    
    top_cat = cat_counts.most_common(1)[0][0] if cat_counts else 0
    
    test_user_features[uid] = {
        'click_count': len(group),
        'unique_articles': len(history),
        'top_category': top_cat,
        'category_count': len(cat_counts),
    }

# 保存
os.makedirs('./temp_results', exist_ok=True)
with open('./temp_results/test_user_features.pkl', 'wb') as f:
    pickle.dump({
        'user_features': test_user_features,
        'user_history': test_user_history,
    }, f)

print(f'测试用户: {len(test_user_features):,}')
print(f'保存: temp_results/test_user_features.pkl')
