# 如何将多路召回集成到现有Notebook

本文档说明如何将多路召回代码集成到你现有的Jupyter Notebook中。

## 方案A: 直接在Notebook中实现（推荐）

### Step 1: 在notebook开头添加类定义

打开 `新闻推荐系统-多路召回.ipynb`，在最前面添加几个新的代码单元格：

```python
# Cell 1: 导入库
import pandas as pd
import numpy as np
from collections import defaultdict
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
```

```python
# Cell 2: ItemCF类定义
# 复制 multi_strategy_recall.py 中的 ItemCFRecall 类
class ItemCFRecall:
    # ... (完整代码见 multi_strategy_recall.py)
    pass
```

```python
# Cell 3: Embedding类定义
class EmbeddingRecall:
    # ... (完整代码)
    pass
```

```python
# Cell 4: Popularity类定义
class PopularityRecall:
    # ... (完整代码)
    pass
```

```python
# Cell 5: Fusion类定义
class RecallFusion:
    # ... (完整代码)
    pass
```

### Step 2: 替换原有的ItemCF部分

找到你原来的ItemCF实现部分，替换为：

```python
# Cell: 训练ItemCF
itemcf = ItemCFRecall(sim_item_topk=100, recall_item_number=100)
itemcf.fit(all_click_df)  # 使用你的点击数据变量名
```

### Step 3: 添加新的召回策略

```python
# Cell: 训练Embedding召回
# 假设你已经加载了 articles_emb.csv
embedding = EmbeddingRecall(recall_item_number=100, use_faiss=False)
embedding.fit(all_click_df, articles_emb_df)
```

```python
# Cell: 训练Popularity召回
popularity = PopularityRecall(recall_item_number=100)
popularity.fit(all_click_df)
```

### Step 4: 创建融合

```python
# Cell: 创建多路融合
fusion = RecallFusion(
    recalls={
        'itemcf': itemcf,
        'embedding': embedding,
        'popularity': popularity
    },
    weights={
        'itemcf': 0.6,
        'embedding': 0.3,
        'popularity': 0.1
    },
    method='weighted_avg'
)

print("✓ 多路召回融合创建完成")
print(f"策略数量: {len(fusion.recalls)}")
```

### Step 5: 生成预测（替换原有逻辑）

找到你原来生成预测的部分，替换为：

```python
# Cell: 批量生成预测
# 构建用户历史字典
user_hist_dict = {}
for user_id, hist_df in all_click_df.groupby('user_id'):
    user_hist_dict[user_id] = list(zip(
        hist_df['click_article_id'].values,
        hist_df['click_timestamp'].values
    ))

# 获取所有需要预测的用户
test_users = test_df['user_id'].unique()
print(f"需要预测的用户数: {len(test_users)}")

# 批量预测
predictions = {}
batch_size = 10000

for i in range(0, len(test_users), batch_size):
    batch = test_users[i:i+batch_size]
    batch_pred = fusion.predict_batch(
        batch, 
        num_candidates=50,
        user_history_dict=user_hist_dict
    )
    predictions.update(batch_pred)
    
    if (i % 50000) == 0:
        print(f"进度: {i}/{len(test_users)}")

print(f"✓ 预测完成: {len(predictions)} 个用户")
```

### Step 6: 生成提交文件

```python
# Cell: 生成提交文件
submission_rows = []
for user_id in test_users:
    candidates = predictions.get(user_id, [])
    
    # 补齐到5个
    while len(candidates) < 5:
        candidates.append(0)
    
    submission_rows.append({
        'user_id': user_id,
        'article_1': candidates[0],
        'article_2': candidates[1],
        'article_3': candidates[2],
        'article_4': candidates[3],
        'article_5': candidates[4],
    })

submission = pd.DataFrame(submission_rows)
submission.to_csv('submission_multi_strategy.csv', index=False)

print(f"✓ 提交文件已保存")
print(submission.head(10))
```

---

## 方案B: 使用外部Python文件（更清晰）

### Step 1: 保存类定义到文件

将 `multi_strategy_recall.py` 保存到你的项目目录。

### Step 2: 在Notebook中导入

```python
# Cell 1: 导入自定义模块
import sys
sys.path.append('.')  # 添加当前目录到路径

from multi_strategy_recall import (
    ItemCFRecall,
    EmbeddingRecall,
    PopularityRecall,
    RecallFusion
)

print("✓ 多路召回模块导入成功")
```

### Step 3: 直接使用类

```python
# Cell 2: 训练和使用
itemcf = ItemCFRecall(sim_item_topk=100, recall_item_number=100)
itemcf.fit(all_click_df)

embedding = EmbeddingRecall(recall_item_number=100)
embedding.fit(all_click_df, articles_emb_df)

popularity = PopularityRecall(recall_item_number=100)
popularity.fit(all_click_df)

fusion = RecallFusion(
    recalls={'itemcf': itemcf, 'embedding': embedding, 'popularity': popularity},
    weights={'itemcf': 0.6, 'embedding': 0.3, 'popularity': 0.1}
)

# 预测
user_hist = get_user_hist(200001)  # 你的用户历史获取函数
candidates = fusion.predict(200001, num_candidates=50, user_history=user_hist)
print(candidates[:10])
```

---

## 方案C: 渐进式集成（最稳妥）

如果你不想一次性改动太大，可以分步骤集成：

### 第1阶段：只添加Popularity（最简单）

```python
# 在原有ItemCF基础上添加热门召回
class PopularityRecall:
    def __init__(self, recall_item_number=100):
        self.recall_item_number = recall_item_number
        self.popular_items = []
    
    def fit(self, click_df):
        item_counts = click_df['click_article_id'].value_counts()
        self.popular_items = item_counts.head(self.recall_item_number).index.tolist()
        return self
    
    def predict(self, user_id):
        return self.popular_items

# 训练
popularity = PopularityRecall(recall_item_number=50)
popularity.fit(all_click_df)

# 简单融合：ItemCF + Popularity
def simple_fusion(itemcf_results, popularity_results, alpha=0.7):
    """
    alpha: ItemCF的权重，1-alpha是Popularity的权重
    """
    fused = {}
    
    # ItemCF结果
    for rank, item in enumerate(itemcf_results):
        fused[item] = alpha * (1.0 / (rank + 1))
    
    # Popularity结果
    for rank, item in enumerate(popularity_results):
        if item not in fused:
            fused[item] = 0.0
        fused[item] += (1 - alpha) * (1.0 / (rank + 1))
    
    # 排序
    sorted_items = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return [item for item, _ in sorted_items]

# 使用
user_id = 200001
itemcf_cand = itemcf.predict(user_id, user_history[user_id])
pop_cand = popularity.predict(user_id)
final_cand = simple_fusion(itemcf_cand, pop_cand, alpha=0.8)

print(f"ItemCF前5: {itemcf_cand[:5]}")
print(f"Popularity前5: {pop_cand[:5]}")
print(f"融合后前5: {final_cand[:5]}")
```

### 第2阶段：添加Embedding

在第1阶段稳定后，添加Embedding召回（代码略，参考上面的EmbeddingRecall类）

### 第3阶段：使用完整的RecallFusion

当前两个阶段都验证没问题后，引入完整的RecallFusion类。

---

## 数据变量名对应关系

你的Notebook中可能使用不同的变量名，这里是对应关系：

| 本文档 | 你的Notebook可能叫 |
|--------|-------------------|
| `all_click_df` | `train_click_log`, `click_df`, `trn_click` |
| `articles_emb_df` | `articles_emb`, `item_emb_df`, `emb_df` |
| `test_df` | `tst_click`, `test_click_log` |
| `user_hist_dict` | `user_item_time_dict`, `user_history` |

**适配示例**:
```python
# 如果你的变量叫 trn_click
itemcf.fit(trn_click)  # 而不是 all_click_df

# 如果你的用户历史字典叫 user_item_time_dict
predictions = fusion.predict_batch(
    test_users,
    user_history_dict=user_item_time_dict  # 直接使用你的变量
)
```

---

## 常见问题

### Q1: 我的Embedding文件列名不是 emb_0, emb_1...

**解决**:
修改 `EmbeddingRecall.fit()` 中的列名提取逻辑：

```python
# 原代码
emb_cols = [col for col in embeddings_df.columns if col.startswith('emb_')]

# 改为你的列名模式，例如 dim_0, dim_1...
emb_cols = [col for col in embeddings_df.columns if col.startswith('dim_')]

# 或者直接指定列索引范围
emb_cols = embeddings_df.columns[1:251]  # 假设第1-250列是向量
```

### Q2: 运行很慢怎么办？

**优化建议**:

1. **减少sim_item_topk**:
```python
# 从100降到50
itemcf = ItemCFRecall(sim_item_topk=50, recall_item_number=100)
```

2. **使用采样**:
```python
# 只用10%的数据训练
sample_click = all_click_df.sample(frac=0.1)
itemcf.fit(sample_click)
```

3. **启用FAISS** (需要安装):
```bash
pip install faiss-cpu
```
```python
embedding = EmbeddingRecall(recall_item_number=100, use_faiss=True)
```

### Q3: 内存不足怎么办？

**分批处理**:

```python
# 原来：一次性处理所有用户
predictions = fusion.predict_batch(all_users, user_history_dict=user_hist)

# 改为：分批处理
batch_size = 5000  # 根据你的内存调整
predictions = {}

for i in range(0, len(all_users), batch_size):
    batch = all_users[i:i+batch_size]
    batch_pred = fusion.predict_batch(batch, user_history_dict=user_hist)
    predictions.update(batch_pred)
    
    # 清理内存
    import gc
    gc.collect()
```

### Q4: 如何验证效果？

**添加评估代码**:

```python
def evaluate_recall_at_k(predictions, ground_truth, k=5):
    """
    predictions: {user_id: [predicted_items]}
    ground_truth: {user_id: [true_items]}
    """
    hits = 0
    total = 0
    
    for user_id, true_items in ground_truth.items():
        if user_id not in predictions:
            continue
        
        pred_items = predictions[user_id][:k]
        true_set = set(true_items)
        
        for item in pred_items:
            if item in true_set:
                hits += 1
        
        total += len(true_set)
    
    return hits / total if total > 0 else 0.0

# 使用
# 假设你有验证集
val_truth = val_click_df.groupby('user_id')['click_article_id'].apply(list).to_dict()

# ItemCF单独
itemcf_pred = itemcf.predict_batch(val_users, user_hist)
itemcf_recall = evaluate_recall_at_k(itemcf_pred, val_truth, k=5)

# 多路融合
fusion_pred = fusion.predict_batch(val_users, user_history_dict=user_hist)
fusion_recall = evaluate_recall_at_k(fusion_pred, val_truth, k=5)

print(f"ItemCF Recall@5: {itemcf_recall:.4f}")
print(f"Fusion Recall@5: {fusion_recall:.4f}")
print(f"提升: {(fusion_recall - itemcf_recall) / itemcf_recall * 100:.2f}%")
```

---

## 完整示例：最小改动集成

如果你想以最小的改动集成多路召回，只需在原有代码后面添加：

```python
# ========== 在你原有ItemCF代码之后添加 ==========

# 1. 添加Popularity
popularity = PopularityRecall(50)
popularity.fit(all_click_df)

# 2. 创建简单融合
recalls = {'itemcf': itemcf, 'popularity': popularity}
fusion = RecallFusion(recalls, weights={'itemcf': 0.8, 'popularity': 0.2})

# 3. 替换预测部分
# 原来: 
# candidates = itemcf.predict(user_id, user_history)

# 现在:
candidates = fusion.predict(user_id, num_candidates=50, user_history=user_history)

# 4. 生成提交（保持不变）
# ... 你原有的提交代码 ...
```

就这样！只需要4步，你就完成了多路召回的集成。

---

## 总结

推荐路径：
1. **新手/赶时间**: 使用方案A的渐进式集成，先加Popularity
2. **有经验**: 直接使用方案B，导入完整模块
3. **求稳妥**: 使用方案C，分阶段验证

选择适合你的方式，祝你项目顺利！🚀
