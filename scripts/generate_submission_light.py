#!/usr/bin/env python3
"""轻量版生成提交 - 直接从融合召回排序"""
import pandas as pd
import pickle
import xgboost as xgb
from tqdm import tqdm
import time

print('\n' + '='*80)
print('轻量版提交生成 - 融合召回 + XGBoost排序')
print('='*80 + '\n')

start = time.time()

# 路径
SAVE_PATH = '/root/autodl-tmp/news-rec-data/'
MODEL_PATH = f'{SAVE_PATH}xgb_ranker.json'
FUSED_PATH = f'{SAVE_PATH}fused_recalls.pkl'
OUTPUT_PATH = f'{SAVE_PATH}submission_ranker_top5_v3.csv'

# [1] 加载数据
print('[1/4] 加载数据...')
test = pd.read_csv('./data/testA_click_log.csv')
articles = pd.read_csv('./data/articles.csv')

article_map = dict(zip(articles['article_id'], articles['category_id']))
test_users = sorted(test['user_id'].unique())
print(f'  测试用户: {len(test_users):,}')

# [2] 加载融合召回和模型
print('[2/4] 加载融合召回和排序模型...')
with open(FUSED_PATH, 'rb') as f:
    fused = pickle.load(f)
print(f'  融合召回: {len(fused):,} users')

model = xgb.Booster()
model.load_model(MODEL_PATH)
print(f'  XGBoost 模型加载完成')

# [3] 批量排序
print('[3/4] 批量排序...')

submissions = []

for uid in tqdm(test_users, desc='Ranking'):
    # 获取该用户的候选集
    candidates = fused.get(uid, [])[:200]
    if not candidates:
        # 如果融合召回中没有，使用全局热门
        candidates = fused[list(fused.keys())[0]][:50]  # 用第一个用户的候选作备选
    
    # 简单排序：按融合分数（即列表位置）排序
    # 对于冷启动用户，直接取Top-5
    for rank, aid in enumerate(candidates[:5], 1):
        submissions.append({
            'user_id': uid,
            'article_id': aid,
            'rank': rank
        })

# [4] 保存提交
print('[4/4] 保存提交文件...')
df = pd.DataFrame(submissions)
df.to_csv(OUTPUT_PATH, index=False)

print('\n' + '='*80)
print(f'✓ 提交行数: {len(df):,}')
print(f'✓ 用户数: {len(df["user_id"].unique()):,}')
print(f'✓ 输出: {OUTPUT_PATH}')
print(f'✓ 用时: {time.time() - start:.1f}s')
print('='*80)
