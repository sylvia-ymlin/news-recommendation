#!/usr/bin/env python3
"""Embedding召回 (Faiss GPU 优先, CPU回退)
- 读取 articles_emb.csv (article_id + emb_*)
- 归一化向量, 建立 IVF+Flat 索引
- 对热门文章（按训练点击热度Top MAX_QUERY）做TopK相似召回
- 保存相似列表到 /root/autodl-tmp/news-rec-data/emb_sim_faiss.pkl
"""
import os
import time
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm

SAVE_PATH = '/root/autodl-tmp/news-rec-data/'
DATA_EMB = './data/articles_emb.csv'
TRAIN_CSV = './data/train_click_log.csv'
MAX_QUERY = 50_000       # 热门文章查询数
TOPK = 100               # 每篇文章召回数
NLIST = 4096             # IVF簇数
NPROBE = 16              # 搜索probe数
SEED = 42

print('\n' + '='*80)
print('Embedding召回 - Faiss')
print('='*80 + '\n')

os.makedirs(SAVE_PATH, exist_ok=True)
np.random.seed(SEED)
start = time.time()

# 1) 读取数据
print('[1/4] 读取embedding...')
emb_df = pd.read_csv(DATA_EMB)
emb_cols = [c for c in emb_df.columns if c.startswith('emb_')]
vecs = emb_df[emb_cols].values.astype('float32')
# 清理异常值以避免 Faiss 训练报 NaN/Inf
bad_mask = ~np.isfinite(vecs)
if bad_mask.any():
    bad_cnt = int(bad_mask.sum())
    print(f'  发现 {bad_cnt} 个 NaN/Inf，已替换为0')
    vecs = np.nan_to_num(vecs, nan=0.0, posinf=0.0, neginf=0.0)
vecs = np.ascontiguousarray(vecs)
article_ids = emb_df['article_id'].values.astype('int64')
ntotal, dim = vecs.shape
print(f'  向量数: {ntotal}, 维度: {dim}')

# 读取训练热度, 选热门文章作为查询
print('[2/4] 读取训练点击热度...')
train = pd.read_csv(TRAIN_CSV, usecols=['click_article_id'])
pop = train['click_article_id'].value_counts()
query_ids = pop.index.values[: min(MAX_QUERY, len(pop))]
print(f'  查询Top: {len(query_ids)}')

# 归一化用于内积=cosine
faiss.normalize_L2(vecs)

# 3) 建索引 (IVF Flat)
print('[3/4] 构建Faiss索引 (IVF Flat)...')
quantizer = faiss.IndexFlatIP(dim)
index = faiss.IndexIVFFlat(quantizer, dim, NLIST, faiss.METRIC_INNER_PRODUCT)

# 训练
train_size = min(200_000, ntotal)
train_sample = vecs[np.random.choice(ntotal, train_size, replace=False)]
index.train(train_sample)

# 尝试GPU
try:
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)
    print('  使用GPU')
except Exception as e:
    print('  GPU不可用, 回退CPU:', e)

# 添加向量
index.add(vecs)
index.nprobe = NPROBE

# 查询向量
print('[4/4] 搜索TopK...')
# 为查询文章取向量
aid_to_idx = {aid: idx for idx, aid in enumerate(article_ids)}
query_vecs = np.vstack([vecs[aid_to_idx[aid]] for aid in query_ids if aid in aid_to_idx]).astype('float32')
# search
_, I = index.search(query_vecs, TOPK + 1)  # 含自身

# 组装结果
emb_topk = {}
for qid, neighbors in zip(query_ids, I):
    # 去掉自身id
    neighbor_ids = []
    for nidx in neighbors:
        nid = article_ids[nidx]
        if nid == qid:
            continue
        neighbor_ids.append(int(nid))
        if len(neighbor_ids) >= TOPK:
            break
    emb_topk[int(qid)] = neighbor_ids

# 保存
out_pkl = os.path.join(SAVE_PATH, 'emb_sim_faiss.pkl')
import pickle
with open(out_pkl, 'wb') as f:
    pickle.dump(emb_topk, f)

elapsed = time.time() - start
print('\n' + '='*80)
print('✅ Embedding召回完成')
print('='*80)
print(f'索引: IVF{NLIST} Flat, nprobe={NPROBE}, 查询数={len(query_ids)}, TopK={TOPK}')
print(f'保存: {out_pkl}')
print(f'耗时: {elapsed/60:.1f} 分钟')
print('='*80)
