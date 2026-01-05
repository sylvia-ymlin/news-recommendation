# 技术挑战与解决方案

> 本文档记录在实现天池新闻推荐系统过程中遇到的技术难题及其解决过程

## 目录
1. [Faiss 向量检索优化](#1-faiss-向量检索优化)
2. [XGBoost 排序模型调优](#2-xgboost-排序模型调优)
3. [多路召回策略融合](#3-多路召回策略融合)
4. [性能优化历程](#4-性能优化历程)

---

## 1. Faiss 向量检索优化

### 背景
项目需要对 255,756 篇文章（250维embedding）进行相似度召回，为 31,116 个热门文章各找出 Top-100 相似文章。初始使用 NumPy 暴力计算余弦相似度，耗时过长（预计数小时）。

### 挑战 1.1: Faiss GPU 不可用

**问题**：服务器安装 `faiss-gpu` 后，导入时报错：
```python
>>> import faiss
AttributeError: module 'faiss' has no attribute 'StandardGpuResources'
```

**根因分析**：
- 服务器 CUDA 环境配置不完整，或 Faiss GPU 版本与 CUDA 版本不匹配
- `faiss-gpu==1.7.4` 需要特定 CUDA 版本（通常为 CUDA 11.x）
- `faiss.StandardGpuResources()` 是 GPU 专属 API，CPU 版本无此接口

**解决方案**：
```python
# 方案1：尝试 GPU，失败时优雅降级到 CPU
try:
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)
    print('  使用GPU')
except Exception as e:
    print('  GPU不可用, 回退CPU:', e)
```

**技术决策**：
- 不强依赖 GPU，确保代码在 CPU-only 环境也能运行
- 使用 IVF（Inverted File Index）加速，即使 CPU 也能接受的性能

### 挑战 1.2: NumPy 版本冲突

**问题**：导入 `faiss` 时报错：
```python
>>> import faiss
AttributeError: _ARRAY_API not found
```

**根因**：
- `faiss-gpu==1.7.4` 编译时依赖 NumPy 1.x API
- 环境中安装了 NumPy 2.0+，API 发生破坏性变更
- NumPy 2.0 移除了部分底层 C API（如 `_ARRAY_API`）

**解决方案**：
```bash
# 降级 NumPy 到 1.x 兼容版本
pip install "numpy<2.0"

# 验证版本
python -c "import numpy; print(numpy.__version__)"  # 1.26.4
python -c "import faiss; print(faiss.__version__)"  # 1.7.4
```

**经验总结**：
- 科学计算库（Faiss, TensorFlow, PyTorch）对 NumPy 版本敏感
- NumPy 2.0 是 major 升级，需谨慎升级
- 项目依赖锁定：`requirements.txt` 中明确版本 `numpy>=1.21,<2.0`

### 挑战 1.3: 向量数据不连续（Non-contiguous Array）

**问题**：调用 `faiss.normalize_L2()` 时报错：
```python
ValueError: array is not C-contiguous
```

**根因分析**：
- Pandas DataFrame 切片（如 `df[emb_cols].values`）可能返回非连续内存视图
- Faiss C++ 底层要求输入数组必须是 C-contiguous（行优先存储）
- NumPy 的 stride 不规则时，Faiss 无法直接访问内存

**触发场景**：
```python
emb_df = pd.read_csv('articles_emb.csv')
emb_cols = [c for c in emb_df.columns if c.startswith('emb_')]
vecs = emb_df[emb_cols].values.astype('float32')  # ❌ 可能非连续
faiss.normalize_L2(vecs)  # ValueError
```

**解决方案**：
```python
vecs = emb_df[emb_cols].values.astype('float32')
vecs = np.ascontiguousarray(vecs)  # ✅ 强制转为连续数组
faiss.normalize_L2(vecs)  # 正常运行
```

**性能影响**：
- `np.ascontiguousarray()` 若数组已连续则零拷贝，否则拷贝一次
- 对于 255,756 × 250 × 4 bytes = 244 MB 数据，拷贝耗时 <100ms（可接受）

### 挑战 1.4: 向量中存在 NaN/Inf 导致训练失败

**问题**：Faiss 索引训练时崩溃：
```python
index.train(train_sample)
RuntimeError: Error: 'std::isfinite(x[i])' failed: input contains NaN's or Inf's
```

**根因分析**：
- `articles_emb.csv` 中部分文章 embedding 计算失败，包含 NaN 值
- 可能原因：
  - 原始文本为空，embedding 模型输出异常
  - 数值计算溢出（如 softmax 时分母为 0）
- Faiss 聚类算法（IVF 训练）对异常值零容忍

**诊断代码**：
```python
vecs = emb_df[emb_cols].values.astype('float32')
bad_mask = ~np.isfinite(vecs)
print(f"异常值数量: {bad_mask.sum()}")  # 212 个 NaN
print(f"异常行数: {bad_mask.any(axis=1).sum()}")  # 分布在 212 个向量中
```

**解决方案**：
```python
# 方案1：替换为 0（保守策略）
vecs = np.nan_to_num(vecs, nan=0.0, posinf=0.0, neginf=0.0)

# 方案2：删除异常向量（激进策略，可能影响覆盖率）
# valid_mask = np.isfinite(vecs).all(axis=1)
# vecs = vecs[valid_mask]
# article_ids = article_ids[valid_mask]
```

**业务影响**：
- 212/255,756 = 0.08% 的文章受影响（比例极小）
- 将异常向量置零后，这些文章的相似度召回退化为随机
- 可通过热度召回作为兜底策略

**预防措施**：
- Embedding 生成阶段：捕获异常并打日志
- 数据校验：存储前检查 `np.isfinite().all()`
- 监控：统计异常向量比例，超过阈值告警

### 最终方案：完整的 Faiss 向量召回流程

```python
#!/usr/bin/env python3
"""Embedding召回 (Faiss GPU优先, CPU回退)"""
import numpy as np
import pandas as pd
import faiss

# 1. 读取数据
emb_df = pd.read_csv('articles_emb.csv')
emb_cols = [c for c in emb_df.columns if c.startswith('emb_')]
vecs = emb_df[emb_cols].values.astype('float32')
article_ids = emb_df['article_id'].values

# 2. 数据清洗
bad_mask = ~np.isfinite(vecs)
if bad_mask.any():
    print(f'发现 {bad_mask.sum()} 个 NaN/Inf，已替换为0')
    vecs = np.nan_to_num(vecs, nan=0.0, posinf=0.0, neginf=0.0)
vecs = np.ascontiguousarray(vecs)  # 确保连续
faiss.normalize_L2(vecs)  # L2归一化，使用内积=余弦相似度

# 3. 建索引 (IVF Flat)
dim = vecs.shape[1]
nlist = 4096  # 聚类中心数
quantizer = faiss.IndexFlatIP(dim)  # 内积（余弦）距离
index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

# 训练（聚类）
train_size = min(200_000, len(vecs))
train_sample = vecs[np.random.choice(len(vecs), train_size, replace=False)]
index.train(train_sample)

# 4. GPU加速（可选）
try:
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)
    print('使用GPU')
except Exception as e:
    print(f'GPU不可用, 回退CPU: {e}')

# 5. 添加向量并搜索
index.add(vecs)
index.nprobe = 16  # 搜索时探测的聚类数（速度vs精度权衡）

# 查询 Top-100 相似文章
query_vecs = vecs  # 可替换为部分热门文章
distances, indices = index.search(query_vecs, 100)

# 6. 结果映射
emb_topk = {}
for i, neighbors in enumerate(indices):
    qid = article_ids[i]
    neighbor_ids = [int(article_ids[idx]) for idx in neighbors if article_ids[idx] != qid]
    emb_topk[int(qid)] = neighbor_ids[:100]

# 保存
import pickle
with open('emb_sim_faiss.pkl', 'wb') as f:
    pickle.dump(emb_topk, f)
```

### 性能对比

| 方法 | 时间 | 精度 | 说明 |
|------|------|------|------|
| NumPy 暴力计算 | ~3小时 | 100% | 255k × 255k 相似度矩阵 |
| Faiss Flat (CPU) | ~2小时 | 100% | 无索引，暴力搜索 |
| Faiss IVF (CPU) | **8分钟** | ~95% | nlist=4096, nprobe=16 |
| Faiss IVF (GPU) | ~2分钟 | ~95% | GPU 加速（若可用） |

**关键参数调优**：
- `nlist`：聚类数，越大越精确但越慢（推荐 √n ~ 4096）
- `nprobe`：搜索探测数，越大越精确但越慢（推荐 nlist/256 ~ 16）
- 速度/精度权衡：IVF 可能漏掉 5% 的真实 Top-100（对召回影响很小）

---

## 2. XGBoost 排序模型调优

### 背景
第一版提交（MRR=0.0079）发现所有用户推荐结果完全相同，第二版（MRR=0.0119）虽然个性化但仍低于基线（0.0192）。

### 挑战 2.1: 测试集特征缺失

**问题现象**：
- 训练时使用用户历史点击构造特征（`user_click_count`, `user_category_prefer` 等）
- 测试集用户无历史，所有特征缺失或为默认值
- 模型在测试集退化为"预测相同分数" → 所有用户Top5一致

**解决方案**：
```python
# 为测试用户构建冷启动特征
def extract_test_features(test_click_log):
    """从测试集点击日志提取特征（模拟历史）"""
    user_feats = test_click_log.groupby('user_id').agg({
        'click_article_id': 'count',  # 点击次数
        'click_timestamp': ['min', 'max']  # 活跃时段
    }).reset_index()
    
    # 类别偏好
    category_prefer = test_click_log.merge(articles[['article_id', 'category_id']])
    category_prefer = category_prefer.groupby(['user_id', 'category_id']).size()
    # ...
    return user_feats
```

**结果**：MRR 从 0.0079 提升到 0.0119（个性化生效），但仍不理想

### 挑战 2.2: 模型过拟合

**问题现象**：
- 训练集 AUC = 0.9906（几乎完美）
- 验证集 AUC = 0.9906（看似正常）
- 实际提交 MRR = 0.0119（远低于基线 0.0192）

**根因分析**：
1. **特征泄漏**：训练样本来自训练集用户，与测试用户分布不同
2. **候选集策略差**：
   - 训练时用真实点击作为正样本（准确）
   - 测试时用类别热度Top5作为候选（不准确）
   - 例如用户喜欢"科技"类冷门文章，但只拿到"科技"热门文章排序
3. **模型能力错位**：
   - XGBoost 学到了"训练用户vs训练文章"的模式
   - 测试时"测试用户vs训练文章"，模式失效

**实验验证**：
```python
# 测试模拟：用训练用户做交叉验证
# 结果：验证集召回率很高（因为候选集包含真实点击）
# 说明模型本身能力强，但测试候选集质量差

# 进一步验证：将测试候选集扩大到Top50
# 结果：MRR几乎不变（说明排序不是主要问题，召回是瓶颈）
```

**解决方向**：
1. **改进召回**：多路召回（热度+ItemCF+Embedding+UserCF）增加候选覆盖
2. **简化模型**：过度复杂的 Ranker 在冷启动场景反而不如简单规则
3. **特征对齐**：使用训练集无关的特征（如文章内容特征、时间衰减）

### 当前问题与待改进点

**问题汇总**：
- 召回阶段：类别热度 Top5 过于简单，覆盖不足
- 排序阶段：XGBoost 过拟合训练分布，泛化差
- 冷启动：测试用户完全陌生，无法利用协同信号

**待实施方案**：
1. 运行 `multi_recall.py`：整合热度、ItemCF、Embedding、UserCF
2. 融合多路召回结果：按权重或学习排序（LTR）
3. 重新训练 Ranker：用融合召回的候选集构建样本
4. 对比基线：确认多路召回是否能超越单一热度策略

---

## 3. 多路召回策略融合

### 背景
单一类别热度召回覆盖不足，需要整合多种召回源：

| 召回策略 | 原理 | 适用场景 | 覆盖量 |
|---------|------|---------|--------|
| **热度召回** | 全局点击量 Top-N | 冷启动、新用户 | ~100篇 |
| **ItemCF** | 物品协同过滤（共现） | 有历史用户 | ~1.3万篇 |
| **Embedding** | 文章内容相似度 | 长尾文章、内容相关 | ~3.1万篇 |
| **UserCF** | 用户协同过滤（相似用户） | 活跃用户、兴趣探索 | ~2.6万篇 |

### 实现方案

#### 3.1 热度召回
```python
def build_hot_list(train, topk=500):
    """全局热度 Top-K"""
    pop = train['click_article_id'].value_counts()
    return pop.head(topk).index.tolist()
```

#### 3.2 ItemCF 召回
```python
def build_itemcf_sim(train, topk=100):
    """物品协同过滤：共现次数"""
    user_items = train.groupby('user_id')['click_article_id'].apply(list).to_dict()
    
    item_cnt = defaultdict(int)
    co_occurrence = defaultdict(lambda: defaultdict(int))
    
    for items in user_items.values():
        for item in items:
            item_cnt[item] += 1
            for other in items:
                if item != other:
                    co_occurrence[item][other] += 1
    
    # 余弦相似度
    itemcf_sim = {}
    for item, co_items in co_occurrence.items():
        scores = {}
        for co_item, cnt in co_items.items():
            scores[co_item] = cnt / np.sqrt(item_cnt[item] * item_cnt[co_item])
        itemcf_sim[item] = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]
    
    return itemcf_sim
```

#### 3.3 Embedding 召回
```python
def build_emb_sim_faiss(emb_df, topk=100):
    """基于 Faiss 的向量相似度召回"""
    # （见第1节 Faiss 实现）
    vecs = emb_df[emb_cols].values.astype('float32')
    vecs = np.nan_to_num(np.ascontiguousarray(vecs))
    faiss.normalize_L2(vecs)
    
    index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, 4096, faiss.METRIC_INNER_PRODUCT)
    index.train(vecs[np.random.choice(len(vecs), 200000, replace=False)])
    index.add(vecs)
    index.nprobe = 16
    
    _, I = index.search(vecs, topk+1)  # +1 去除自身
    # 返回 {article_id: [sim_id1, sim_id2, ...]}
```

#### 3.4 UserCF 召回
```python
def build_usercf_sim(train, topk=50):
    """用户协同过滤：基于共同点击的用户相似度"""
    item_users = train.groupby('click_article_id')['user_id'].apply(set).to_dict()
    
    user_sim = defaultdict(lambda: defaultdict(int))
    for users in item_users.values():
        users_list = list(users)
        for i, u1 in enumerate(users_list):
            for u2 in users_list[i+1:]:
                user_sim[u1][u2] += 1
                user_sim[u2][u1] += 1
    
    # 归一化
    user_cnt = train['user_id'].value_counts().to_dict()
    for u1, neighbors in user_sim.items():
        for u2 in neighbors:
            user_sim[u1][u2] /= np.sqrt(user_cnt[u1] * user_cnt[u2])
    
    return {u: sorted(neighbors.items(), key=lambda x: x[1], reverse=True)[:topk] 
            for u, neighbors in user_sim.items()}
```

### 融合策略（待实施）

**方案1：规则权重**
```python
def merge_recalls(hot, itemcf, emb, usercf, user_history, weights=[0.2, 0.3, 0.3, 0.2]):
    """按权重融合多路召回"""
    candidates = defaultdict(float)
    
    # 热度召回（权重 0.2）
    for rank, item in enumerate(hot[:100]):
        candidates[item] += weights[0] * (1 / (rank + 1))
    
    # ItemCF（权重 0.3）
    for hist_item in user_history[-5:]:
        if hist_item in itemcf:
            for item, score in itemcf[hist_item][:50]:
                candidates[item] += weights[1] * score
    
    # Embedding（权重 0.3）
    for hist_item in user_history[-5:]:
        if hist_item in emb:
            for rank, item in enumerate(emb[hist_item][:50]):
                candidates[item] += weights[2] * (1 / (rank + 1))
    
    # UserCF（权重 0.2）
    # ...
    
    return sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:200]
```

**方案2：LTR（Learn to Rank）**
- 用多路召回分数作为特征
- 训练 XGBoost/LightGBM Ranker
- 端到端学习最优融合权重

---

## 4. 性能优化历程

### 阶段1：单核实现（378秒）
```python
# 初始代码：串行处理 50k 用户
for user_id in test_users:
    recs = recommend_for_user(user_id)
    all_recs.extend(recs)
```

### 阶段2：多核并行（5秒）
```python
from multiprocessing import Pool

def process_batch(user_batch):
    return [recommend_for_user(u) for u in user_batch]

# 128核并行
with Pool(128) as pool:
    batches = np.array_split(test_users, 128)
    results = pool.map(process_batch, batches)
```

**加速比**：378 / 5 = **75.6倍**

### 阶段3：Faiss 向量召回（原3小时 → 8分钟）
- NumPy 暴力计算：O(n²) = 255k × 255k × 250 ≈ 16 trillion ops
- Faiss IVF：O(n·√n·probe) ≈ 4096 clusters × 16 probes

**加速比**：180 / 8 = **22.5倍**

### 阶段4：存储优化
```python
# 中间结果持久化到高速盘
SAVE_PATH = '/root/autodl-tmp/news-rec-data/'  # 100GB SSD
# 避免重复计算 ItemCF/Embedding 相似度
```

---

## 经验总结

### 技术选型
1. **科学计算栈兼容性**：
   - NumPy < 2.0（Faiss 依赖）
   - pandas >= 1.3（groupby 性能）
   - xgboost GPU 需 CUDA 11.x

2. **向量检索工具**：
   - 小规模（<10万）：NumPy 暴力计算
   - 中规模（10万-100万）：Faiss IVF
   - 大规模（>100万）：Faiss HNSW 或 Milvus

3. **并行化**：
   - Python multiprocessing 适用于 CPU-bound 任务
   - GIL 限制：多进程 > 多线程
   - 注意共享内存开销（pickle 序列化）

### 调试技巧
1. **异常向量诊断**：
   ```python
   assert np.isfinite(vecs).all(), "Found NaN/Inf"
   assert vecs.flags['C_CONTIGUOUS'], "Array not contiguous"
   ```

2. **分步验证**：
   - 读数据 → 检查 shape/dtype
   - 清洗数据 → 验证无异常
   - 索引训练 → 单独测试
   - 全流程 → 小规模试运行

3. **性能监控**：
   ```python
   import time
   start = time.time()
   # ...
   print(f"Elapsed: {time.time() - start:.2f}s")
   ```

### 业务思维
1. **召回 > 排序**：候选集覆盖不足时，再好的排序模型也无济于事
2. **冷启动优先级**：真实业务中冷启动用户占比高，专门优化策略必要
3. **线上线下一致性**：训练样本分布应与线上相符（如测试集用户特征）

---

## 下一步计划

- [ ] 执行 `multi_recall.py` 生成多路召回结果
- [ ] 实现融合策略（规则权重 baseline）
- [ ] 重新训练 Ranker（用融合候选集）
- [ ] 生成新提交文件，对比 MRR 指标
- [ ] 若超越基线，撰写方案总结；否则回退简单热度策略

**目标**：MRR > 0.0192（基线），证明多路召回有效性
