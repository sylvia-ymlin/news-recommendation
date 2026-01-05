#!/usr/bin/env python3
"""新闻推荐系统 - 数据分析"""
import pandas as pd
import numpy as np
import json
from collections import Counter
import time

print('\n' + '='*80)
print('数据分析 - 新闻推荐系统')
print('='*80 + '\n')

start = time.time()

# 加载数据
print('[1/5] 加载数据...')
train = pd.read_csv('./data/train_click_log.csv')
test = pd.read_csv('./data/testA_click_log.csv')
articles = pd.read_csv('./data/articles.csv')

# 基础统计
print('[2/5] 基础统计分析...')
stats = {
    'train': {
        'users': train['user_id'].nunique(),
        'articles': train['click_article_id'].nunique(),
        'clicks': len(train),
        'avg_clicks_per_user': len(train) / train['user_id'].nunique(),
        'avg_clicks_per_article': len(train) / train['click_article_id'].nunique(),
    },
    'test': {
        'users': test['user_id'].nunique(),
        'articles': test['click_article_id'].nunique(),
        'clicks': len(test),
        'avg_clicks_per_user': len(test) / test['user_id'].nunique(),
    },
    'articles': {
        'total': len(articles),
        'categories': articles['category_id'].nunique(),
        'avg_words': articles['words_count'].mean(),
    }
}

# 用户重叠分析
print('[3/5] 用户重叠分析...')
train_users = set(train['user_id'].unique())
test_users = set(test['user_id'].unique())
overlap = train_users & test_users

stats['overlap'] = {
    'train_only': len(train_users - test_users),
    'test_only': len(test_users - train_users),
    'common': len(overlap),
    'cold_start_ratio': len(test_users - train_users) / len(test_users),
}

# 文章热度分析
print('[4/5] 文章热度分析...')
article_popularity = train['click_article_id'].value_counts()
stats['article_popularity'] = {
    'top_10_coverage': article_popularity.head(10).sum() / len(train),
    'top_100_coverage': article_popularity.head(100).sum() / len(train),
    'top_1000_coverage': article_popularity.head(1000).sum() / len(train),
    'max_clicks': int(article_popularity.max()),
    'min_clicks': int(article_popularity.min()),
}

# 用户活跃度分析
print('[5/5] 用户活跃度分析...')
user_activity = train.groupby('user_id').size()
stats['user_activity'] = {
    'max_clicks': int(user_activity.max()),
    'min_clicks': int(user_activity.min()),
    'median_clicks': float(user_activity.median()),
    'p25': float(user_activity.quantile(0.25)),
    'p75': float(user_activity.quantile(0.75)),
    'p90': float(user_activity.quantile(0.90)),
}

# 保存结果
print('\n保存分析结果...')
with open('./outputs/data_analysis.json', 'w', encoding='utf-8') as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)

# 生成报告
report = f"""
{'='*80}
数据分析报告
{'='*80}

1. 训练集统计
   - 用户数: {stats['train']['users']:,}
   - 文章数: {stats['train']['articles']:,}
   - 点击数: {stats['train']['clicks']:,}
   - 人均点击: {stats['train']['avg_clicks_per_user']:.2f}
   - 文章平均被点击: {stats['train']['avg_clicks_per_article']:.2f}

2. 测试集统计
   - 用户数: {stats['test']['users']:,}
   - 文章数: {stats['test']['articles']:,}
   - 点击数: {stats['test']['clicks']:,}
   - 人均点击: {stats['test']['avg_clicks_per_user']:.2f}

3. 冷启动问题 ⚠️
   - 训练集独有用户: {stats['overlap']['train_only']:,}
   - 测试集独有用户: {stats['overlap']['test_only']:,}
   - 共同用户: {stats['overlap']['common']:,}
   - 冷启动比例: {stats['overlap']['cold_start_ratio']:.2%}
   
4. 文章热度分布
   - Top 10文章覆盖: {stats['article_popularity']['top_10_coverage']:.2%}
   - Top 100文章覆盖: {stats['article_popularity']['top_100_coverage']:.2%}
   - Top 1000文章覆盖: {stats['article_popularity']['top_1000_coverage']:.2%}
   - 最热文章点击数: {stats['article_popularity']['max_clicks']:,}
   
5. 用户活跃度分布
   - 最活跃用户点击: {stats['user_activity']['max_clicks']:,}
   - 最不活跃用户点击: {stats['user_activity']['min_clicks']:,}
   - 中位数点击: {stats['user_activity']['median_clicks']:.0f}
   - P25-P75: {stats['user_activity']['p25']:.0f} - {stats['user_activity']['p75']:.0f}
   - P90: {stats['user_activity']['p90']:.0f}

6. 关键发现
   ✓ 100%测试用户为冷启动（训练集中未见过）
   ✓ Top 100文章覆盖{stats['article_popularity']['top_100_coverage']:.1%}的点击
   ✓ 用户活跃度差异大（中位数{stats['user_activity']['median_clicks']:.0f}，P90={stats['user_activity']['p90']:.0f}）
   ✓ 协同过滤无法直接应用于测试集

7. 推荐策略建议
   → 基于内容的推荐（利用文章embedding）
   → 基于热度的推荐（Top-K热门文章）
   → 基于类别的推荐（相同类别文章）
   → 混合策略（多路召回 + 排序）

{'='*80}
执行时间: {time.time() - start:.2f}秒
{'='*80}
"""

print(report)

with open('./outputs/data_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print('✅ 分析完成！')
print('   - JSON结果: outputs/data_analysis.json')
print('   - 文本报告: outputs/data_analysis_report.txt')
