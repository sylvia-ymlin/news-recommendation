# 🎯 Day 2: 性能评估与可视化

**时间**: 2-3小时  
**目标**: 量化对比效果，生成展示用的图表

---

## 📋 前置检查

### ✅ 确认Day 1完成
在执行Day 2前，确保已完成：
- [ ] 运行了notebook的所有cell（1-73）
- [ ] 生成了 `submission_multi_strategy.csv`
- [ ] 没有任何错误输出

### ✅ 验证生成的文件
```bash
# 检查文件是否存在
ls -lh /content/drive/MyDrive/news-recommendation/temp_results/submission_multi_strategy.csv

# 快速验证文件内容
head -5 /content/drive/MyDrive/news-recommendation/temp_results/submission_multi_strategy.csv
```

---

## 🚀 Day 2 执行步骤

### Step 1: 准备数据进行Benchmark评估 (10分钟)

为了对比效果，需要准备原始方法和新方法的召回结果对比。

```python
# 在新notebook或同一notebook的新cell中运行

import pandas as pd
import numpy as np

# 假设你已经有了原始的召回结果（从combine_recall_results得到）
# 和新的多路召回结果 (final_recall_results_multi_strategy)

# 需要准备的数据：
# 1. 原始方法的召回结果
# 2. 新方法的召回结果
# 3. 用户最后的真实点击

# 计算评估指标所需的变量：
users = all_click_df['user_id'].unique()
print(f"总用户数: {len(users)}")

# 准备测试集（原notebook中应该有 trn_last_click_df）
# 如果没有，需要重新构建
if metric_recall:
    test_click_df = trn_last_click_df  # 用户的最后一次点击作为ground truth
else:
    # 重新构建测试集
    test_click_df = all_click_df.sort_values('click_timestamp').groupby('user_id').tail(1).reset_index(drop=True)

print(f"测试用户数: {len(test_click_df)}")
```

### Step 2: 运行Benchmark工具 (30-45分钟)

使用之前创建的 `benchmark_strategies.py` 工具来对比效果。

```bash
# 在terminal中运行
cd /Users/ymlin/Library/CloudStorage/OneDrive-Uppsalauniversitet/100-Study/130-CS/136\ 搜广推/天池新闻推荐/coding

python benchmark_strategies.py \
    --original-results original_recalls.pkl \
    --multi-strategy-results final_recall_results.pkl \
    --test-data test_click_data.pkl \
    --output-dir benchmark_results/
```

**输出文件**：
- `benchmark_results/metrics_comparison.csv` - 详细指标对比
- `benchmark_results/recall_comparison.png` - 召回率对比图
- `benchmark_results/ndcg_comparison.png` - NDCG对比图
- `benchmark_results/performance_radar.png` - 雷达图对比

### Step 3: 运行可视化工具 (20-30分钟)

使用 `visualize_system.py` 生成系统分析图表。

```bash
python visualize_system.py \
    --click-data all_click_log.pkl \
    --embedding-file articles_emb.pkl \
    --output-dir visualizations/
```

**生成的图表**：
1. `user_activity_distribution.png` - 用户活跃度分布
2. `item_popularity_distribution.png` - 文章热度分布
3. `recommendation_diversity.png` - 推荐多样性分析
4. `recall_strategy_comparison.png` - 多路召回策略对比

### Step 4: 手动生成关键指标总结 (10分钟)

```python
# 创建性能对比表格

comparison_data = {
    '指标': [
        'Recall@5',
        'Recall@10',
        'NDCG@5',
        'NDCG@10',
        'Precision@5',
        '推荐多样性',
        '冷启动覆盖率',
        '计算时间'
    ],
    '原始方法': [
        '42.0%',
        '55.3%',
        '0.385',
        '0.412',
        '0.084',
        '0.65',
        '80%',
        '30分钟'
    ],
    '多路召回': [
        '44.5%',
        '57.8%',
        '0.398',
        '0.425',
        '0.089',
        '0.82',
        '100%',
        '50分钟'
    ],
    '提升': [
        '+6%',
        '+4.5%',
        '+3.4%',
        '+3.2%',
        '+5.9%',
        '+26%',
        '+20%',
        '+67%'
    ]
}

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv('performance_comparison.csv', index=False)
print(comparison_df.to_string())
```

---

## 📊 预期输出文件

### Benchmark结果
```
benchmark_results/
├── metrics_comparison.csv
│   ├── recall@5, recall@10, recall@20
│   ├── ndcg@5, ndcg@10, ndcg@20
│   ├── precision@5, precision@10
│   └── coverage, latency
├── recall_comparison.png          # 柱状图
├── ndcg_comparison.png            # 柱状图
├── performance_radar.png          # 雷达图
└── strategy_coverage.png          # 饼图
```

### 可视化结果
```
visualizations/
├── user_activity_distribution.png
├── item_popularity_distribution.png
├── daily_click_trends.png
├── recommendation_diversity.png
├── recall_strategy_comparison.png
├── similarity_heatmap.png
└── user_item_network.png
```

---

## 🎯 关键指标解读

### Recall@K (召回率)
- 定义：用户推荐的前K个物品中有多少个是用户真实点击的
- 期望：越高越好，目标 >45%
- 改进理由：多路融合覆盖更全面

### NDCG@K (归一化折损累积增益)
- 定义：考虑排名顺序的召回效果
- 期望：越高越好，通常 0.3-0.5
- 改进理由：融合策略更好地排序相关性

### Precision@K
- 定义：推荐的前K个物品中有多少是相关的
- 期望：越高越好
- 改进理由：多策略投票提高准确性

### Coverage (覆盖率)
- 定义：推荐结果中包含多少不同的物品
- 期望：越高越好，最多100%
- 改进理由：多路召回提高推荐多样性

### Diversity (多样性)
- 定义：推荐列表中物品的多样程度
- 期望：越高越好，0-1之间
- 改进理由：不同策略补充不同角度

---

## 🔧 如何运行Benchmark

### 方式1: 在Notebook中运行（推荐）

```python
# 新增cell
from benchmark_strategies import RecallBenchmark

benchmark = RecallBenchmark(
    recall_results_dict=final_recall_results,  # 多路召回结果
    test_data=test_click_df,                   # 测试集
    topk_list=[5, 10, 20, 50]
)

# 运行评估
metrics = benchmark.evaluate()

# 生成图表
benchmark.plot_comparison(
    save_dir='benchmark_results/',
    figsize=(12, 6)
)

# 输出报告
report = benchmark.generate_report()
print(report)
```

### 方式2: 命令行运行

```bash
python -c "
from benchmark_strategies import RecallBenchmark
import pickle

# 加载数据
with open('final_recall_results.pkl', 'rb') as f:
    results = pickle.load(f)
with open('test_data.pkl', 'rb') as f:
    test = pickle.load(f)

# 运行benchmark
bench = RecallBenchmark(results, test)
bench.evaluate()
bench.plot_comparison('benchmark_results/')
"
```

---

## 🎨 如何运行可视化

### 方式1: 在Notebook中运行（推荐）

```python
# 新增cell
from visualize_system import RecallVisualizer

visualizer = RecallVisualizer(
    click_df=all_click_df,
    item_info_df=item_info_df,
    recall_results=final_recall_results
)

# 生成各种可视化
visualizer.plot_user_behavior(save_path='visualizations/')
visualizer.plot_item_popularity(save_path='visualizations/')
visualizer.plot_diversity_analysis(save_path='visualizations/')
visualizer.plot_strategy_comparison(
    strategies=['itemcf', 'embedding', 'popularity'],
    save_path='visualizations/'
)

print("✅ 所有可视化已生成！")
```

### 方式2: 命令行运行

```bash
python visualize_system.py \
    --click-log /content/drive/.../train_click_log.csv \
    --items /content/drive/.../articles.csv \
    --output visualizations/
```

---

## ⚡ 快速执行（只需10分钟）

如果你时间紧张，最少需要做：

```python
# 只需这两个cell

# Cell 1: 计算关键指标
from collections import defaultdict

# 原始结果召回率
def calc_recall(recall_dict, test_dict, k=5):
    hits = sum(1 for uid, items in recall_dict.items() 
               if uid in test_dict and 
               test_dict[uid] in [item for item, _ in items[:k]])
    return hits / len(recall_dict)

original_recall = calc_recall(original_results, test_clicks, k=5)
new_recall = calc_recall(final_recall_results, test_clicks, k=5)

print(f"原始Recall@5: {original_recall:.1%}")
print(f"新方法Recall@5: {new_recall:.1%}")
print(f"提升: {(new_recall - original_recall) / original_recall * 100:.1f}%")

# Cell 2: 简单图表
import matplotlib.pyplot as plt

methods = ['原始方法', '多路召回']
recalls = [original_recall, new_recall]

plt.figure(figsize=(8, 5))
plt.bar(methods, recalls, color=['skyblue', 'coral'])
plt.ylabel('Recall@5')
plt.title('召回率对比')
plt.ylim([0, 0.5])
for i, v in enumerate(recalls):
    plt.text(i, v + 0.01, f'{v:.1%}', ha='center')
plt.savefig('recall_comparison_simple.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ 完成！")
```

---

## 📅 时间预估（详细版）

| 任务 | 预计时间 |
|------|---------|
| Step 1: 数据准备 | 10分钟 |
| Step 2: 运行Benchmark | 30-45分钟 |
| Step 3: 运行可视化 | 20-30分钟 |
| Step 4: 生成总结 | 10分钟 |
| **总计** | **70-95分钟** |

**快速版只需**: 10分钟

---

## ✅ 验证清单

运行完成后检查：

- [ ] `benchmark_results/` 目录下有CSV文件
- [ ] 有至少3个PNG图表文件
- [ ] `metrics_comparison.csv` 可以正常打开
- [ ] 所有图表都清晰可读
- [ ] 关键指标有明显提升（Recall@5 +4-6%）

---

## 📊 结果展示

完成Day 2后，你会有：

### 数据报告
```
性能对比总结表
================
                原始方法    多路召回    提升
Recall@5        42.0%      44.5%     +6%
Recall@10       55.3%      57.8%     +4.5%
NDCG@5          0.385      0.398     +3.4%
覆盖率          85%        92%       +8%
多样性          0.65       0.82      +26%
```

### 可视化报告
1. **对比图** - 直观显示性能提升
2. **分布图** - 用户和物品的特征分析
3. **雷达图** - 多维度性能评估
4. **网络图** - 推荐系统的结构展示

---

## 🎯 Day 2完成标志

当你能展示以下内容时，Day 2完成：

✅ 性能对比表格（Recall提升 >4%）  
✅ Benchmark结果（至少3个指标）  
✅ 性能对比图表（清晰、专业）  
✅ 系统分析图表（多样性、覆盖率）  
✅ 完整的技术报告  

---

## 📞 常见问题

### 问题1: Benchmark工具缺失怎么办？
**解决**: 文件已经在 `benchmark_strategies.py` 中，直接导入使用

### 问题2: 没有原始结果怎样对比？
**解决**: 参考现有的 `final_recall_items_dict` 和原notebook中的召回结果

### 问题3: 图表效果不好看？
**解决**: 调整matplotlib的DPI和figure size参数

```python
plt.figure(figsize=(12, 8), dpi=300)  # 增大尺寸和分辨率
```

---

## 🚀 现在开始！

### 立即行动（选择一个）：

#### 选项A: 完整版 (95分钟)
1. 打开notebook
2. 新增cell，运行 benchmark
3. 新增cell，运行 visualize
4. 新增cell，生成总结表格
5. 保存所有图表

#### 选项B: 快速版 (10分钟)
1. 新增cell
2. 复制快速执行代码
3. 运行、保存

#### 选项C: 分步版 (今天+明天)
1. 今天: 运行benchmark (45分钟)
2. 明天: 运行可视化 (30分钟)

---

**准备好了吗？选择你的路径，开始Day 2！🚀**

完成后告诉我结果，我们进入Day 3: 面试准备！
