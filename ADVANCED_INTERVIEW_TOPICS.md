# 进阶面试话题 - 推荐系统深入讨论

本文档涵盖高级面试中可能被问到的深度话题。

---

## 1. 深度学习召回

### 双塔模型 (Two-Tower Model)

**原理**:
```
User Features → User Tower → User Embedding (128-dim)
                                    ↓
                            Cosine Similarity
                                    ↓
Item Features → Item Tower → Item Embedding (128-dim)
```

**优势**:
- 支持复杂特征（文本、图片、序列）
- 可以离线预计算item embeddings
- 实时计算user embedding

**实现思路**:
```python
import torch
import torch.nn as nn

class UserTower(nn.Module):
    def __init__(self, user_features_dim, embedding_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(user_features_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class ItemTower(nn.Module):
    def __init__(self, item_features_dim, embedding_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(item_features_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
    
    def forward(self, x):
        return self.network(x)

# 训练目标: maximize cosine similarity for clicked pairs
# Loss = -log(sigmoid(user_emb · item_pos_emb - user_emb · item_neg_emb))
```

**面试回答模板**:
> "如果要进一步优化，我会考虑引入深度学习。双塔模型可以学习更复杂的用户和物品表示，支持多模态特征（文本标题、图片、用户行为序列）。优势是可以离线预计算物品向量，在线只需计算用户塔，延迟可控。训练时使用负采样，损失函数是对比学习的BPR或InfoNCE。"

---

## 2. 序列建模 (Sequential Recommendation)

### GRU4Rec / SASRec

**问题**: 用户点击是序列行为，如何捕捉时序模式？

**GRU4Rec**:
```python
class GRU4Rec(nn.Module):
    def __init__(self, num_items, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_items)
    
    def forward(self, item_seq):
        # item_seq: [batch, seq_len]
        emb = self.item_embedding(item_seq)  # [batch, seq_len, emb_dim]
        output, hidden = self.gru(emb)       # output: [batch, seq_len, hidden]
        logits = self.fc(output[:, -1, :])  # 最后一个时间步
        return logits
```

**SASRec (Self-Attention)**:
```python
# 使用Transformer的self-attention捕捉序列
# 优势: 可以建模长期依赖，不受RNN的限制
class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.ln1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        return x
```

**面试讨论**:
> "我的项目目前使用的是ItemCF，它隐式地考虑了序列（共现）。如果要显式建模序列，可以用GRU4Rec或SASRec。GRU适合短序列，SASRec的self-attention可以捕捉长期依赖。训练数据是（序列，下一个点击）对，预测下一个物品的概率分布。"

---

## 3. 多目标优化 (Multi-Objective Optimization)

### 场景
真实推荐系统不只关注点击率，还有：
- **点击率 (CTR)**: 用户是否点击
- **停留时长 (Dwell Time)**: 用户看了多久
- **转化率 (CVR)**: 是否购买/订阅
- **多样性 (Diversity)**: 避免信息茧房

### 方法1: 多任务学习 (Multi-Task Learning)

```python
class MultiTaskModel(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        # 共享底层
        self.shared = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        # 任务特定头
        self.ctr_head = nn.Linear(128, 1)
        self.cvr_head = nn.Linear(128, 1)
        self.dwell_head = nn.Linear(128, 1)
    
    def forward(self, x):
        shared_emb = self.shared(x)
        ctr = torch.sigmoid(self.ctr_head(shared_emb))
        cvr = torch.sigmoid(self.cvr_head(shared_emb))
        dwell = self.dwell_head(shared_emb)
        return ctr, cvr, dwell

# 损失函数
loss = w1 * bce_loss(ctr_pred, ctr_true) + \
       w2 * bce_loss(cvr_pred, cvr_true) + \
       w3 * mse_loss(dwell_pred, dwell_true)
```

### 方法2: 帕累托最优 (Pareto Optimization)

找到多个目标的平衡点，不能牺牲某个目标来提升另一个。

**面试回答**:
> "生产环境中，我会考虑多目标优化。比如用MTL同时预测CTR和停留时长，共享底层特征学习。权重可以通过业务目标设定，或者用Pareto优化找最佳平衡点。此外，还要加入多样性约束，比如MMR算法重排序，避免同质化推荐。"

---

## 4. 冷启动的高级解决方案

### Bandit算法 (Exploration vs Exploitation)

**UCB (Upper Confidence Bound)**:
```python
def ucb_select(item_stats, total_trials, c=2):
    """
    选择具有最高UCB的物品
    UCB = mean_reward + c * sqrt(log(total_trials) / item_trials)
    """
    ucb_scores = {}
    for item, (sum_reward, count) in item_stats.items():
        if count == 0:
            ucb_scores[item] = float('inf')  # 未尝试的优先
        else:
            mean = sum_reward / count
            confidence = c * np.sqrt(np.log(total_trials) / count)
            ucb_scores[item] = mean + confidence
    
    return max(ucb_scores, key=ucb_scores.get)
```

**ε-greedy**:
```python
def epsilon_greedy_select(item_scores, epsilon=0.1):
    """
    以概率ε随机探索，1-ε利用当前最优
    """
    if np.random.rand() < epsilon:
        return np.random.choice(list(item_scores.keys()))  # 探索
    else:
        return max(item_scores, key=item_scores.get)  # 利用
```

**Thompson Sampling**:
```python
# 为每个物品维护Beta分布 Beta(α, β)
# α = 成功次数 + 1, β = 失败次数 + 1
# 每次采样一个值，选择采样值最大的物品

def thompson_sampling(item_stats):
    samples = {}
    for item, (alpha, beta) in item_stats.items():
        samples[item] = np.random.beta(alpha, beta)
    return max(samples, key=samples.get)
```

**面试讨论**:
> "对于冷启动，除了用Popularity兜底，还可以用Bandit算法平衡探索和利用。UCB会优先尝试不确定性高的物品，Thompson Sampling从贝叶斯角度采样。这样可以快速学习新物品的质量，同时不牺牲太多用户体验。"

---

## 5. 实时性与增量更新

### 场景
- 新闻有时效性，需要快速捕捉热点
- 用户兴趣动态变化

### 解决方案

**1. Lambda架构**:
```
实时层 (Streaming) → Redis (最近1小时)
                ↓
批处理层 (Batch) → HDFS (历史全量)
                ↓
          服务层 (Serving)
```

**2. 增量更新ItemCF**:
```python
class IncrementalItemCF:
    def __init__(self):
        self.item_cooccur = defaultdict(lambda: defaultdict(int))
        self.item_count = defaultdict(int)
    
    def update(self, new_clicks):
        """增量更新相似度矩阵"""
        for user_id, items in new_clicks.groupby('user_id')['article_id']:
            items_list = list(items)
            for item in items_list:
                self.item_count[item] += 1
            
            for i in range(len(items_list)):
                for j in range(i+1, len(items_list)):
                    item_i, item_j = items_list[i], items_list[j]
                    self.item_cooccur[item_i][item_j] += 1
                    self.item_cooccur[item_j][item_i] += 1
    
    def decay(self, factor=0.95):
        """衰减旧的统计值"""
        for item_i in self.item_cooccur:
            for item_j in self.item_cooccur[item_i]:
                self.item_cooccur[item_i][item_j] *= factor
        
        for item in self.item_count:
            self.item_count[item] *= factor
```

**3. 时间衰减**:
```python
def time_weighted_similarity(clicks_df, decay_days=7):
    """时间加权的相似度计算"""
    now = clicks_df['timestamp'].max()
    clicks_df['weight'] = np.exp(-(now - clicks_df['timestamp']) / (decay_days * 86400))
    
    # 在计算共现时使用权重
    # cooccur[i, j] += weight1 * weight2
```

**面试回答**:
> "对于新闻这种时效性强的场景，我会采用增量更新。维护一个滑动窗口（如最近3天），定期重算相似度。或者用时间衰减因子，旧的点击权重降低。实时层用Redis缓存热门文章，批处理层定期全量更新。这样可以在延迟和准确性之间平衡。"

---

## 6. 规模化与工程优化

### 1. 分布式训练

**数据并行**:
```python
# PyTorch DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

model = UserTower(...)
model = model.to(device)
model = DDP(model, device_ids=[local_rank])

# 训练时数据分片到各GPU
train_sampler = DistributedSampler(train_dataset)
train_loader = DataLoader(train_dataset, sampler=train_sampler, ...)
```

**模型并行**:
```python
# 对于超大模型，不同层放在不同GPU
class BigModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1000, 5000).to('cuda:0')
        self.layer2 = nn.Linear(5000, 1000).to('cuda:1')
    
    def forward(self, x):
        x = self.layer1(x.to('cuda:0'))
        x = self.layer2(x.to('cuda:1'))
        return x
```

### 2. 模型压缩

**量化 (Quantization)**:
```python
# 将float32权重量化为int8
import torch.quantization

model_fp32 = MyModel()
model_fp32.eval()

# 动态量化
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32, {nn.Linear}, dtype=torch.qint8
)

# 模型大小减少4倍，推理加速2-3倍
```

**知识蒸馏 (Knowledge Distillation)**:
```python
# 用大模型(teacher)训练小模型(student)
teacher_model = BigModel()
student_model = SmallModel()

# 蒸馏损失
def distillation_loss(student_logits, teacher_logits, temperature=3.0):
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    kd_loss = F.kl_div(soft_student, soft_targets, reduction='batchmean')
    return kd_loss * (temperature ** 2)
```

### 3. 缓存策略

**多层缓存**:
```python
class MultiLevelCache:
    def __init__(self):
        self.l1_cache = {}  # 本地内存，100ms TTL
        self.l2_cache = redis_client  # Redis，10分钟TTL
        self.l3_storage = db_client  # 数据库
    
    def get(self, user_id):
        # L1
        if user_id in self.l1_cache:
            return self.l1_cache[user_id]
        
        # L2
        result = self.l2_cache.get(f'user:{user_id}')
        if result:
            self.l1_cache[user_id] = result
            return result
        
        # L3
        result = self.l3_storage.query(user_id)
        self.l2_cache.setex(f'user:{user_id}', 600, result)
        self.l1_cache[user_id] = result
        return result
```

**面试讨论**:
> "规模化时，我会考虑：1) 分布式训练加速模型更新；2) 模型量化和蒸馏减小size；3) 多级缓存（本地→Redis→DB）降低延迟；4) 物品向量用FAISS或ScaNN做ANN检索；5) 服务用多副本负载均衡。目标是p99延迟<50ms，QPS>10k。"

---

## 7. A/B测试与在线评估

### Interleaving

**问题**: A/B测试需要长时间才能有显著性

**解决**: Interleaving - 同一用户同时看到两个算法的结果

```python
def team_draft_interleaving(results_A, results_B, k=10):
    """
    Team Draft Interleaving算法
    A和B轮流选择物品加入最终列表
    """
    interleaved = []
    pointer_A, pointer_B = 0, 0
    
    while len(interleaved) < k:
        if len(interleaved) % 2 == 0:  # A的回合
            while pointer_A < len(results_A):
                item = results_A[pointer_A]
                pointer_A += 1
                if item not in interleaved:
                    interleaved.append(item)
                    break
        else:  # B的回合
            while pointer_B < len(results_B):
                item = results_B[pointer_B]
                pointer_B += 1
                if item not in interleaved:
                    interleaved.append(item)
                    break
    
    return interleaved

# 评估: 统计用户点击的物品来自哪个算法
# 如果点击更多来自A，则A胜出
```

### Counterfactual Evaluation

**离线评估在线效果**:

```python
def inverse_propensity_scoring(logged_data, new_policy):
    """
    使用IPS估计新策略的效果
    logged_data: (user, item, reward, propensity)
    """
    estimated_reward = 0.0
    
    for user, item, reward, propensity in logged_data:
        new_prob = new_policy.predict_prob(user, item)
        
        # 重要性采样
        weight = new_prob / propensity
        estimated_reward += weight * reward
    
    return estimated_reward / len(logged_data)
```

**面试回答**:
> "A/B测试是金标准，但耗时长。我会先用Interleaving快速比较两个算法，几小时就能有结论。离线阶段用Counterfactual Evaluation（IPS、DR）估计新策略的在线表现。此外还要监控多个指标：CTR、停留时长、用户满意度、多样性等，避免单一指标优化。"

---

## 8. 推荐系统的伦理与公平性

### Filter Bubble (信息茧房)

**问题**: 推荐系统不断强化用户已有兴趣，导致信息窄化

**解决**:
1. **Diversity Injection**: 10-20%的推荐来自探索
2. **Serendipity**: 推荐意外但相关的内容
3. **Explanability**: 解释为什么推荐这个

```python
def diverse_rerank(candidates, diversity_weight=0.3):
    """
    重排序加入多样性
    """
    selected = []
    remaining = candidates.copy()
    
    while len(selected) < 10 and remaining:
        scores = []
        for item in remaining:
            # 相关性得分
            relevance = item['score']
            
            # 多样性得分（与已选物品的平均距离）
            if selected:
                diversity = np.mean([
                    distance(item, s) for s in selected
                ])
            else:
                diversity = 1.0
            
            # 组合
            final_score = (1 - diversity_weight) * relevance + \
                         diversity_weight * diversity
            scores.append((item, final_score))
        
        # 选择得分最高的
        best_item, _ = max(scores, key=lambda x: x[1])
        selected.append(best_item)
        remaining.remove(best_item)
    
    return selected
```

### Fairness

**问题**: 推荐系统可能对某些群体不公平（性别、年龄、地域）

**评估**:
```python
def demographic_parity(recommendations, user_demographics, protected_attr='gender'):
    """
    检查不同人口群体是否得到类似的推荐质量
    """
    group_metrics = defaultdict(list)
    
    for user_id, rec_items in recommendations.items():
        group = user_demographics[user_id][protected_attr]
        quality = compute_quality(rec_items)  # 如avg CTR
        group_metrics[group].append(quality)
    
    # 比较不同组的平均质量
    for group, qualities in group_metrics.items():
        print(f"{group}: {np.mean(qualities):.4f}")
```

**面试讨论**:
> "推荐系统要考虑伦理问题。1) 避免Filter Bubble，可以用MMR重排序加入多样性；2) 确保公平性，定期审计不同人群的推荐质量；3) 提供可解释性，让用户理解推荐理由；4) 给用户控制权，允许调整兴趣偏好。这些在实际系统中和算法同样重要。"

---

## 总结：面试中的层次

### Level 1: 基础理解
- 能解释ItemCF、UserCF、内容召回
- 知道Recall、Precision、NDCG等指标
- 理解冷启动问题

### Level 2: 工程实现
- 能实现多路召回融合
- 知道如何优化性能（FAISS、缓存）
- 理解A/B测试流程

### Level 3: 系统设计
- 设计完整推荐系统架构
- 考虑规模化、实时性
- 多目标优化

### Level 4: 前沿研究
- 深度学习方法（双塔、序列模型）
- 强化学习、Bandit
- 公平性、可解释性

**你的项目目前在Level 2，通过这些进阶话题可以展示到Level 3**。

在面试中，根据面试官的反应调整深度。如果是算法岗，可以深入讨论模型；如果是工程岗，侧重系统设计和优化。

**关键**: 不要只说"我知道XXX算法"，要说"我的项目用了XXX，如果要进一步优化，可以考虑YYY，因为ZZZ"。展示思考深度和工程sense。
