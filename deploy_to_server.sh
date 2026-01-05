#!/bin/bash
# 新服务器快速部署脚本

echo "========================================"
echo "新闻推荐系统 - 新服务器部署"
echo "========================================"

# 1. 克隆代码
echo -e "\n[1/5] 克隆GitHub仓库..."
cd ~
rm -rf news-recommendation
git clone https://github.com/sylvia-ymlin/news-recommendation.git
cd news-recommendation

# 2. 创建目录
echo -e "\n[2/5] 创建必要目录..."
mkdir -p data temp_results outputs

# 3. 安装依赖
echo -e "\n[3/5] 安装Python依赖..."
pip install pandas numpy scikit-learn tqdm -q

# 4. 检查环境
echo -e "\n[4/5] 检查环境..."
echo "Python版本: $(python3 --version)"
echo "CPU核心数: $(nproc)"
echo "GPU信息:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  无GPU"

# 5. 创建优化脚本
echo -e "\n[5/5] 创建GPU优化脚本..."
cat > scripts/day1_gpu_optimized.py << 'EOFPY'
#!/usr/bin/env python3
"""GPU + 多核优化版本"""
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import time
import os

NUM_CORES = os.cpu_count()
K_RECOMMENDATIONS = 50

start_time = time.time()
print("\n" + "="*80)
print(f"推荐系统 - GPU优化版 ({NUM_CORES}核)")
print("="*80 + "\n")

print("[1/5] 加载数据...")
train = pd.read_csv('./data/train_click_log.csv')
test_users = pd.read_csv('./data/testA_click_log.csv')['user_id'].unique()
train_users = set(train['user_id'].unique())

print(f"  训练用户: {len(train_users):,}")
print(f"  测试用户: {len(test_users):,}")
print(f"  冷启动: {len(test_users) - len(set(test_users) & train_users):,}")

print("\n[2/5] 计算热门文章...")
item_pop = train['click_article_id'].value_counts()
all_items = np.array(item_pop.index.values)

print("\n[3/5] 加载ItemCF...")
try:
    with open('./temp_results/itemcf_i2i_sim.pkl', 'rb') as f:
        itemcf = pickle.load(f)
    print(f"  ItemCF: {len(itemcf)} 物品")
except:
    print("  无ItemCF缓存，使用纯热度推荐")
    itemcf = {}

print("\n[4/5] 构建用户历史...")
user_hist = {}
for uid in test_users:
    if uid in train_users:
        user_hist[uid] = set(train[train['user_id'] == uid]['click_article_id'].values)
    else:
        user_hist[uid] = set()

def gen_rec(uid, uh, icf, items, k=50):
    hist = uh[uid]
    recs = []
    if not hist:
        for item in items[:k]:
            recs.append((uid, int(item), len(recs)+1))
    else:
        seen = set()
        for h in list(hist)[:5]:
            if h in icf:
                for sim in icf[h][:20]:
                    if sim not in hist and sim not in seen and len(recs) < k:
                        recs.append((uid, int(sim), len(recs)+1))
                        seen.add(sim)
        if len(recs) < k:
            for item in items:
                if item not in hist and item not in seen:
                    recs.append((uid, int(item), len(recs)+1))
                    seen.add(item)
                    if len(recs) >= k:
                        break
    return recs

print(f"\n[5/5] 生成推荐 ({NUM_CORES}核并行)...")
rec_func = partial(gen_rec, uh=user_hist, icf=itemcf, items=all_items, k=K_RECOMMENDATIONS)

all_recs = []
with mp.Pool(NUM_CORES) as pool:
    for r in tqdm(pool.imap_unordered(rec_func, test_users, chunksize=1000), 
                  total=len(test_users), desc="处理中", unit="用户"):
        all_recs.extend(r)

df = pd.DataFrame(all_recs, columns=['user_id', 'article_id', 'rank'])
df.to_csv('./temp_results/submission_gpu_optimized.csv', index=False)

elapsed = time.time() - start_time
print(f"\n  记录数: {len(df):,}")
print(f"  独特文章: {df['article_id'].nunique():,}")
print(f"  执行时间: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")
print(f"  吞吐量: {len(test_users)/elapsed:.0f} 用户/秒")
print("\n" + "="*80)
print("✅ 完成！")
print("="*80)
EOFPY

chmod +x scripts/day1_gpu_optimized.py

echo -e "\n========================================"
echo "✅ 部署完成！"
echo "========================================"
echo ""
echo "下一步："
echo "1. 传输数据文件到服务器 data/ 目录"
echo "2. 运行: python3 scripts/day1_gpu_optimized.py"
echo ""
