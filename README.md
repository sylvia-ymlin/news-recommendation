# 天池新闻推荐系统

> Tianchi News Recommendation Competition - Multi-Recall & Ranking System

## 项目概述

本项目是天池新闻推荐算法竞赛的完整解决方案，实现了从数据分析、特征工程、多路召回到排序模型的端到端推荐系统。

**核心技术**：
- 多路召回：热度、ItemCF、Embedding(Faiss)、UserCF
- 排序模型：XGBoost Ranker
- 性能优化：多进程并行、向量索引加速
- 工程实践：远程服务器部署、版本控制

**提交成绩**：
- Baseline（热度）: MRR = 0.0192
- v1（单路召回+Ranker）: MRR = 0.0079 ❌
- v2（测试集特征+Ranker）: MRR = 0.0119 ⚠️
- v3（多路召回融合）: 开发中...

---

## 目录结构

```
coding/
├── data/                          # 数据文件（.gitignore）
│   ├── train_click_log.csv        # 训练集点击日志（200k用户）
│   ├── testA_click_log.csv        # 测试集点击日志（50k用户）
│   ├── articles.csv               # 文章元数据（364k篇）
│   └── articles_emb.csv           # 文章embedding（250维）
│
├── notebooks/                     # Jupyter 分析笔记本
│   ├── 新闻系统推荐-赛题理解.ipynb
│   ├── 新闻推荐系统-数据分析.ipynb
│   ├── 新闻推荐系统-多路召回.ipynb
│   ├── 新闻推荐系统-特征工程.ipynb
│   └── 新闻推荐系统-排序模型.ipynb
│
├── scripts/                       # 生产脚本
│   ├── multi_recall.py            # 多路召回（热度+ItemCF+Emb+UserCF）
│   ├── embedding_recall_faiss.py  # Faiss向量召回（优化版）
│   ├── feature_engineering.py     # 特征提取
│   ├── build_samples.py           # 训练样本构建
│   ├── train_ranker.py            # XGBoost排序模型训练
│   ├── extract_test_features.py   # 测试集特征提取
│   └── generate_submission.py     # 生成提交文件
│
├── docs/                          # 项目文档
│   ├── 03-problem-analysis.md     # 问题分析（冷启动、性能优化）
│   └── 04-technical-challenges.md # 技术挑战（Faiss、XGBoost调优）
│
├── outputs/                       # 输出结果
│   ├── evaluation_report.txt      # 评估报告
│   └── metrics.csv                # 指标汇总
│
├── temp_results/                  # 中间结果（.gitignore）
│   ├── itemcf_i2i_sim.pkl        # ItemCF相似度表
│   ├── usercf_u2u_sim.pkl        # UserCF相似度表
│   └── item_content_emb.pkl      # 文章embedding索引
│
├── requirements.txt               # Python依赖
├── deploy_to_server.sh            # 服务器部署脚本
├── .gitignore                     # Git忽略规则
└── README.md                      # 本文档
```

---

## 快速开始

### 1. 环境配置

**本地开发**（分析、notebook）：
```bash
conda create -n news-rec python=3.10
conda activate news-rec
pip install -r requirements.txt
```

**远程服务器**（训练、推理）：
```bash
# 部署代码和数据
bash deploy_to_server.sh

# SSH到服务器
ssh news-server
cd ~/news-recommendation

# 安装依赖（注意NumPy版本）
pip install pandas numpy scikit-learn xgboost tqdm
pip install "numpy<2.0"  # Faiss兼容性
pip install faiss-cpu    # 或 faiss-gpu（需CUDA）
```

### 2. 数据准备

将竞赛数据放到 `data/` 目录：
```bash
data/
├── train_click_log.csv       # 必需
├── testA_click_log.csv       # 必需
├── articles.csv              # 必需
└── articles_emb.csv          # 必需
```

验证数据完整性：
```python
import pandas as pd

train = pd.read_csv('data/train_click_log.csv')
test = pd.read_csv('data/testA_click_log.csv')
articles = pd.read_csv('data/articles.csv')
emb = pd.read_csv('data/articles_emb.csv')

print(f"训练用户: {train['user_id'].nunique()}")       # 200,000
print(f"测试用户: {test['user_id'].nunique()}")        # 50,000
print(f"文章数: {articles['article_id'].nunique()}")   # 364,048
print(f"Embedding: {emb.shape}")                       # (255756, 251)
```

### 3. 执行流程

#### 阶段1：数据分析（可选）
```bash
jupyter notebook notebooks/新闻推荐系统-数据分析.ipynb
```

#### 阶段2：多路召回
```bash
# 在服务器执行（需大内存）
cd ~/news-recommendation

# 热度 + ItemCF + UserCF + Embedding召回
python3 scripts/multi_recall.py
# 输出: /root/autodl-tmp/news-rec-data/{hot_list,itemcf_sim,usercf_sim}.pkl

# Faiss向量召回（优化版）
python3 scripts/embedding_recall_faiss.py
# 输出: /root/autodl-tmp/news-rec-data/emb_sim_faiss.pkl
```

**预期耗时**：
- `multi_recall.py`: ~10分钟（128核）
- `embedding_recall_faiss.py`: ~8分钟（CPU）

#### 阶段3：特征工程
```bash
# 提取训练特征
python3 scripts/feature_engineering.py
# 输出: temp_results/features.pkl

# 构建训练样本（正负采样）
python3 scripts/build_samples.py
# 输出: /root/autodl-tmp/news-rec-data/training_samples.pkl
```

#### 阶段4：排序模型训练
```bash
# XGBoost Ranker
python3 scripts/train_ranker.py
# 输出: /root/autodl-tmp/news-rec-data/xgb_ranker.json
```

#### 阶段5：测试集推理
```bash
# 提取测试集特征
python3 scripts/extract_test_features.py

# 生成提交文件
python3 scripts/generate_submission.py
# 输出: submission_ranker_top5_v3.csv
```

### 4. 简化版快速验证

如果只想快速生成提交文件（热度baseline）：
```bash
python3 scripts/baseline_fast.py
# 14秒生成50k用户×50条推荐
# MRR ≈ 0.0192
```

---

## 核心技术详解

### 多路召回策略

| 召回路径 | 原理 | 覆盖量 | 适用场景 |
|---------|------|--------|---------|
| **热度召回** | 全局点击Top-N | ~500篇 | 冷启动、新用户 |
| **ItemCF** | 物品协同过滤（共现）| ~13k篇 | 有历史用户，挖掘关联 |
| **Embedding** | Faiss向量检索 | ~31k篇 | 内容相似，长尾覆盖 |
| **UserCF** | 用户协同过滤 | ~26k篇 | 兴趣探索，群体偏好 |

**融合策略**：
- 规则权重：`score = 0.2×hot + 0.3×ItemCF + 0.3×Emb + 0.2×UserCF`
- LTR（待实现）：用XGBoost学习最优权重

### Faiss向量召回优化

**问题**：255k篇文章两两计算相似度需 255k² × 250 ≈ 16 trillion 次浮点运算（~3小时）

**解决方案**：
1. IVF索引：将向量聚类到4096个簇，搜索时仅探测16个簇
2. 向量归一化：L2归一化后用内积代替余弦相似度
3. 数据清洗：处理NaN/Inf，确保C-contiguous

**加速效果**：3小时 → 8分钟（22.5倍）

**关键代码**：
```python
import faiss
import numpy as np

# 读取并清洗embedding
vecs = pd.read_csv('articles_emb.csv', header=None).values.astype('float32')
vecs = np.nan_to_num(vecs, nan=0.0)
vecs = np.ascontiguousarray(vecs)
faiss.normalize_L2(vecs)

# 构建IVF索引
quantizer = faiss.IndexFlatIP(dim)
index = faiss.IndexIVFFlat(quantizer, dim, 4096, faiss.METRIC_INNER_PRODUCT)
index.train(vecs[np.random.choice(len(vecs), 200000)])
index.add(vecs)
index.nprobe = 16

# 搜索Top-100
distances, indices = index.search(vecs, 100)
```

### XGBoost排序模型

**特征体系**（21维）：
- 用户特征（9维）：点击次数、活跃天数、类别偏好分布
- 文章特征（7维）：热度、发布时间、字数、类别热度
- 交互特征（5维）：用户-类别偏好匹配、时间衰减

**训练配置**：
```python
params = {
    'objective': 'rank:pairwise',
    'eval_metric': 'auc',
    'tree_method': 'gpu_hist',  # GPU加速
    'max_depth': 8,
    'eta': 0.1,
    'subsample': 0.8
}
```

**当前问题**：
- 训练AUC=0.99（过拟合）
- 测试MRR=0.0119（低于baseline）
- 根因：召回候选集质量差 + 测试用户冷启动

---

## 已知问题与改进计划

### 已解决问题 ✅

1. **Faiss导入错误**（AttributeError: _ARRAY_API）
   - 原因：NumPy 2.0 不兼容
   - 解决：降级到 `numpy<2.0`

2. **向量非连续错误**（array is not C-contiguous）
   - 原因：pandas切片返回非连续内存
   - 解决：`np.ascontiguousarray(vecs)`

3. **NaN/Inf训练失败**（input contains NaN）
   - 原因：embedding数据异常（212个向量）
   - 解决：`np.nan_to_num(vecs, nan=0.0)`

4. **测试集特征缺失**（所有用户推荐相同）
   - 原因：未用测试点击日志构造特征
   - 解决：`extract_test_features.py`

### 待改进问题 ⚠️

1. **多路召回未融合**
   - 现状：各路召回已生成pkl，但未融合使用
   - 计划：实现规则权重融合 → 重训Ranker

2. **排序模型泛化差**
   - 现状：训练过拟合，测试不如baseline
   - 计划：简化模型 or 改进候选集质量

3. **冷启动覆盖不足**
   - 现状：测试用户100%冷启动，协同过滤失效
   - 计划：增强内容召回（Embedding）+ 热度兜底

### 下一步计划 📋

- [ ] 执行多路召回脚本（已完成：embedding_recall_faiss.py）
- [ ] 实现融合策略代码
- [ ] 用融合召回重建训练样本
- [ ] 重新训练Ranker并提交v3
- [ ] 对比v3 vs baseline，决定最终方案

**目标MRR**：> 0.0192（超越baseline）

---

## 性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| **训练数据量** | 1,112,623 | 点击记录 |
| **训练样本量** | 5,563,115 | 正样本 + 4倍负采样 |
| **测试用户数** | 50,000 | 全冷启动 |
| **文章覆盖率** | 255,756 / 364,048 | 70% 有embedding |
| **Embedding维度** | 250 | 文章内容向量 |
| **ItemCF覆盖** | 13,897 | 有共现关系的文章 |
| **Faiss召回耗时** | 8分钟 | 255k向量，Top-100 |
| **多核推理耗时** | 5秒 | 50k用户×50条（128核）|

---

## 技术栈

**编程语言**：
- Python 3.10

**核心库**：
- pandas 1.5+ : 数据处理
- numpy 1.21-1.26 : 数值计算（<2.0兼容Faiss）
- scikit-learn 1.3+ : 特征归一化
- xgboost 2.0+ : 排序模型
- faiss-cpu/gpu 1.7.4 : 向量检索
- tqdm : 进度条

**开发工具**：
- Jupyter Notebook : 数据分析
- Git : 版本控制
- SSH/SCP : 远程服务器部署

**计算资源**：
- 本地：Mac（分析、开发）
- 服务器：128核CPU + 100GB SSD（训练、推理）

---

## 文档索引

- [问题分析与解决方案](docs/03-problem-analysis.md) - 冷启动、性能优化
- [技术挑战详解](docs/04-technical-challenges.md) - Faiss、XGBoost调优过程
- [Jupyter Notebooks](notebooks/) - 数据探索与实验

---

## 项目亮点（面试素材）

1. **端到端推荐系统**
   - 从原始数据到提交文件的完整pipeline
   - 多路召回 + 精排的工业界标准架构

2. **性能工程实践**
   - 多进程并行：378秒 → 5秒（75倍加速）
   - Faiss向量索引：3小时 → 8分钟（22倍加速）
   - 存储优化：大文件迁移到高速SSD

3. **问题诊断能力**
   - 冷启动识别：分析测试用户100%新用户
   - 特征泄漏发现：训练分布 vs 测试分布不匹配
   - 调试技巧：NaN检测、内存连续性验证

4. **技术深度**
   - Faiss IVF原理与参数调优（nlist, nprobe）
   - NumPy底层：C-contiguous、stride理解
   - XGBoost Ranker：pairwise loss、AUC优化

5. **工程规范**
   - 模块化设计：召回、排序、评估分离
   - 文档驱动：Markdown记录技术决策
   - 版本控制：Git管理代码和实验

---

## 作者

**ymlin** - Uppsala University  
竞赛时间：2026年1月  
联系方式：[GitHub](https://github.com/yourusername)

---

## 许可证

本项目仅用于学习和竞赛目的，数据版权归天池平台所有。
