#!/usr/bin/env python3
"""生成提交文件 - XGBoost 排序（CPU/GPU自适应）"""
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from collections import Counter, defaultdict
from tqdm import tqdm
import time
import os

print('\n' + '='*80)
print('生成提交文件 - XGBoost 排序')
print('='*80 + '\n')

start_time = time.time()

# 路径配置
MODEL_PATH = '/root/autodl-tmp/news-rec-data/xgb_ranker.json'
FEATURE_PATH = './temp_results/features.pkl'
TEST_FEATURE_PATH = './temp_results/test_user_features.pkl'  # 测试用户特征
OUTPUT_PATH = '/root/autodl-tmp/news-rec-data/submission_ranker_top5_v2.csv'
CANDIDATES_PER_USER = 150  # 增加候选多样性
TOP_K = 5                  # 最终提交Top-5
BATCH_SIZE = 200_000       # 每批构造与预测的行数

# ========== 加载数据 ==========
print('[1/5] 加载数据与特征...')
train = pd.read_csv('./data/train_click_log.csv')
test = pd.read_csv('./data/testA_click_log.csv')
articles = pd.read_csv('./data/articles.csv')

with open(FEATURE_PATH, 'rb') as f:
    feats = pickle.load(f)
user_features = feats['user_features']
article_features = feats['article_features']
category_stats = feats['category_stats']
user_history = feats['user_history']

# 加载测试用户特征（基于testA历史）
with open(TEST_FEATURE_PATH, 'rb') as f:
    test_feats = pickle.load(f)
test_user_features = test_feats['user_features']
test_user_history = test_feats['user_history']

print(f'  训练用户: {len(user_features):,}')
print(f'  测试用户: {len(test_user_features):,}')
print(f'  文章特征: {len(article_features):,}')
# ========== 预备集合 ==========
print('\n[2/5] 准备候选...')
# 全局热度
popularity = train['click_article_id'].value_counts().index.tolist()
global_top = popularity[:CANDIDATES_PER_USER]

# 类别热度
cat_pop = defaultdict(list)
for cat_id, group in articles.groupby('category_id'):
    aids = group['article_id'].values
    clicks = train[train['click_article_id'].isin(aids)]['click_article_id'].value_counts()
    cat_pop[cat_id] = clicks.index.tolist()[:CANDIDATES_PER_USER]

test_users = test['user_id'].unique()

# ========== 加载模型 ==========
print('\n[3/5] 加载模型...')
bst = xgb.Booster()
bst.load_model(MODEL_PATH)

# ========== 生成推荐（批量） ==========
print('\n[4/5] 批量打分...')

# 预先准备空类别特征默认值
default_cat_feat = {
    'article_count': 0,
    'total_clicks': 0,
    'unique_users': 0,
    'avg_clicks_per_article': 0,
}

# 1) 构造平铺候选
cand_rows = []
for uid in test_users:
    seen = test_user_history.get(uid, set())  # 用测试集历史
    user_cat = test_user_features.get(uid, {}).get('top_category', 0)  # 用测试集偏好
    candidates = []
    if user_cat in cat_pop:
        candidates.extend(cat_pop[user_cat])
    candidates.extend(global_top)
    uniq = []
    s = set()
    for aid in candidates:
        if aid in s or aid in seen:
            continue
        if aid not in article_features:
            continue
        uniq.append(aid)
        s.add(aid)
        if len(uniq) >= CANDIDATES_PER_USER:
            break
    for aid in uniq:
        cand_rows.append((uid, aid))

cand_df = pd.DataFrame(cand_rows, columns=['user_id', 'article_id'])

# 2) 分批构造特征并预测
scores_all = []
for start in tqdm(range(0, len(cand_df), BATCH_SIZE), desc='  predict-batch'):
    end = min(start + BATCH_SIZE, len(cand_df))
    batch = cand_df.iloc[start:end]
    uids = batch['user_id'].values
    aids = batch['article_id'].values
    feats_mat = []
    for uid, aid in zip(uids, aids):
        uf = user_features.get(uid, {
            'click_count': 0,
            'unique_articles': 0,
            'diversity_ratio': 0,
            'time_span_hours': 0,
            'avg_hour': 12,
            'most_active_hour': 12,
            'category_count': 0,
            'category_entropy': 0,
            'top_category': 0,
        })
        af = article_features[aid]
        cat_id = af['category_id']
        cf = category_stats.get(cat_id, default_cat_feat)
        category_match = 1 if uf['top_category'] == cat_id else 0
        feats_mat.append([
            uf['click_count'], uf['unique_articles'], uf['diversity_ratio'],
            uf['time_span_hours'], uf['avg_hour'], uf['most_active_hour'],
            uf['category_count'], uf['category_entropy'], uf['top_category'],
            af['words_count'], af['category_id'], af['click_count'],
            af['unique_users'], af['click_rate'], af['lifespan_hours'],
            af['age_at_first_click_hours'],
            cf['article_count'], cf['total_clicks'], cf['unique_users'], cf['avg_clicks_per_article'],
            category_match,
        ])
    Xb = np.array(feats_mat, dtype=np.float32)
    scores = bst.predict(xgb.DMatrix(Xb))
    scores_all.append(scores)

scores_all = np.concatenate(scores_all)
cand_df['score'] = scores_all

# 3) 按用户取Top-K并转为提交格式
sub_rows = []
for uid, grp in cand_df.groupby('user_id'):
    topk = grp.nlargest(TOP_K, 'score')
    articles_top = topk['article_id'].tolist()
    # 不足补热门
    if len(articles_top) < TOP_K:
        for aid in global_top:
            if aid not in articles_top:
                articles_top.append(aid)
            if len(articles_top) >= TOP_K:
                break
    sub_rows.append([uid] + articles_top[:TOP_K])

sub_df = pd.DataFrame(sub_rows, columns=['user_id'] + [f'article_{i}' for i in range(1, TOP_K+1)])
sub_df.to_csv(OUTPUT_PATH, index=False)

elapsed = time.time() - start_time
print('\n' + '='*80)
print('✅ 提交文件完成')
print('='*80)
print(f'提交行数: {len(sub_df):,}, 每行Top-{TOP_K}')
print(f'输出: {OUTPUT_PATH}')
print(f'耗时: {elapsed/60:.1f} 分钟')
print('='*80)
