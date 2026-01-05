#!/usr/bin/env python3
"""训练排序模型（XGBoost GPU）"""
import os
import pickle
import time
import numpy as np
import xgboost as xgb

print('\n' + '='*80)
print('训练排序模型 - XGBoost GPU')
print('='*80 + '\n')

start_time = time.time()

DATA_PATH = '/root/autodl-tmp/news-rec-data/training_samples.pkl'
MODEL_PATH = '/root/autodl-tmp/news-rec-data/xgb_ranker.json'

# ========== 加载数据 ==========
print('[1/4] 加载特征...')
with open(DATA_PATH, 'rb') as f:
    data = pickle.load(f)
X = data['X'].astype(np.float32)
y = data['y'].astype(np.float32)
print(f'  X: {X.shape}, y: {y.shape}, 正样本占比: {y.mean()*100:.2f}%')

# ========== 划分训练/验证 ==========
print('\n[2/4] 划分训练/验证...')
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.05, random_state=42, stratify=y
)
print(f'  训练集: {X_train.shape}, 验证集: {X_valid.shape}')

# 释放原数组引用，加速GC
X = None; y = None

# ========== 构建DMatrix ==========
print('\n[3/4] 构建DMatrix...')
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)

# ========== 训练 ==========
print('\n[4/4] 训练模型（GPU）...')
params = {
    'objective': 'binary:logistic',
    'eval_metric': ['auc', 'aucpr'],
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'max_depth': 8,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'gamma': 0.0,
}
num_boost_round = 500
early_stopping_rounds = 50

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=watchlist,
    early_stopping_rounds=early_stopping_rounds,
    verbose_eval=20,
)

# 保存模型
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
bst.save_model(MODEL_PATH)

elapsed = time.time() - start_time
print('\n' + '='*80)
print('✅ 训练完成')
print('='*80)
print(f'最佳迭代: {bst.best_iteration}')
print(f'验证AUC: {bst.best_score:.4f}')
print(f'模型保存: {MODEL_PATH}')
print(f'耗时: {elapsed/60:.1f} 分钟')
print('='*80)
