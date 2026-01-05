# Day 1 执行指南 - 多路召回集成

## 🎯 目标
将多路召回策略集成到现有的Jupyter Notebook中，生成优化后的推荐结果。

## ✅ 已完成的准备工作
1. ✅ 多路召回代码已创建 (`multi_strategy_recall.py`)
2. ✅ 已在notebook末尾添加8个新的代码单元格
3. ✅ 所有集成代码已准备就绪

## 📝 执行步骤

### Step 1: 打开Notebook
```bash
# 在VS Code中打开
新闻推荐系统-多路召回.ipynb
```

### Step 2: 运行原有的数据加载部分
**重要**: 必须先运行notebook前面的cell，确保以下变量已加载：
- `all_click_df` - 点击数据
- `item_emb_dict` - 文章embedding字典
- `save_path` - 保存路径

**需要运行的关键cell**：
1. Google Drive挂载 (如果使用Colab)
2. 导入库的cell
3. 数据路径配置
4. 读取数据的cell (get_all_click_df, get_item_emb_dict等)

### Step 3: 运行新增的多路召回代码
滚动到notebook最底部，你会看到新增的部分：
- **标题**: "🎯 多路召回策略升级"
- **7个代码cell** + 1个总结cell

**按顺序运行**：
```python
# Cell 1: 导入模块 ✅
# Cell 2: 训练ItemCF ✅
# Cell 3: 训练Embedding ✅
# Cell 4: 训练Popularity ✅
# Cell 5: 创建融合器 ✅
# Cell 6: 批量召回 ✅
# Cell 7: 生成提交文件 ✅
```

### Step 4: 检查输出文件
```bash
# 检查生成的文件
ls -lh /content/drive/MyDrive/news-recommendation/temp_results/submission_multi_strategy.csv
```

**预期输出**：
- 文件名: `submission_multi_strategy.csv`
- 列: `user_id, article_1, article_2, article_3, article_4, article_5`
- 行数: 约200,000行 (所有测试用户)

## ⏱️ 预计执行时间

| 步骤 | 预计时间 |
|------|---------|
| Step 2: 数据加载 | 5-10分钟 |
| Cell 2: ItemCF训练 | 15-20分钟 |
| Cell 3: Embedding训练 | 10-15分钟 |
| Cell 4: Popularity训练 | 2-3分钟 |
| Cell 5-7: 融合&生成 | 5-10分钟 |
| **总计** | **40-60分钟** |

## 🐛 常见问题

### 问题1: 导入模块失败
```python
ModuleNotFoundError: No module named 'multi_strategy_recall'
```

**解决方案**：
```python
# 确认文件路径正确
import os
print(os.getcwd())  # 查看当前目录

# 手动添加路径
import sys
sys.path.insert(0, '/你的实际路径/coding')
```

### 问题2: item_emb_dict 未定义
```python
NameError: name 'item_emb_dict' is not defined
```

**解决方案**：
必须先运行前面的cell加载embedding：
```python
# 找到并运行这个cell
item_emb_dict = get_item_emb_dict(data_path, save_path)
```

### 问题3: 内存不足
```python
MemoryError: Unable to allocate ...
```

**解决方案**：
降低召回数量参数：
```python
# 修改 Cell 2-4 的参数
itemcf_recall = ItemCFRecall(
    sim_item_topk=50,         # 从100降到50
    recall_item_number=50     # 从100降到50
)
```

### 问题4: 运行时间过长
**解决方案**：
- 确认运行环境（本地/Colab）
- 本地运行建议使用GPU
- Colab建议升级到Pro获得更多资源

## 📊 验证结果

运行完成后，检查以下指标：

### 1. 训练日志
```
✅ ItemCF训练完成
   - 物品数量: 364047
   - 平均相似物品数: 85.3

✅ Embedding召回训练完成
   - 文章数量: 364047
   - Embedding维度: 250

✅ Popularity召回训练完成
   - 热门文章数: 364047

✅ 召回完成
   - 召回用户数: 200000
   - 平均每用户召回数: 150.0

✅ 提交文件已生成
   - 用户数: 200000
```

### 2. 提交文件格式
```python
# 验证代码
import pandas as pd
submission = pd.read_csv(save_path + 'submission_multi_strategy.csv')

print(f"形状: {submission.shape}")  # 应该是 (200000, 6)
print(f"列名: {submission.columns.tolist()}")  # ['user_id', 'article_1', ..., 'article_5']
print(f"是否有空值: {submission.isnull().sum().sum()}")  # 应该是 0

# 检查前5行
print(submission.head())
```

### 3. 数据质量检查
```python
# 检查文章ID是否合法
all_articles = set(all_click_df['click_article_id'].unique())

for col in ['article_1', 'article_2', 'article_3', 'article_4', 'article_5']:
    invalid = submission[~submission[col].astype(int).isin(all_articles)]
    print(f"{col} 非法文章数: {len(invalid)}")  # 应该都是 0
```

## 🎉 成功标志

当你看到以下输出时，说明Day 1任务完成：

```
✅ 提交文件已生成
   - 文件路径: /content/drive/MyDrive/news-recommendation/temp_results/submission_multi_strategy.csv
   - 用户数: 200000
   - 文件大小: 15.23 KB

📊 前5行预览:
   user_id  article_1  article_2  article_3  article_4  article_5
0   123456     789012     456789     234567     890123     567890
1   234567     901234     567890     345678     012345     678901
...
```

## 📅 下一步 (Day 2)

完成Day 1后，明天的任务：
1. ✅ 运行benchmarking工具对比效果
2. ✅ 生成可视化图表
3. ✅ 记录关键指标

---

## 📞 需要帮助？

如果遇到任何问题：
1. 查看 `NOTEBOOK_INTEGRATION_GUIDE.md`
2. 查看 `MULTI_STRATEGY_QUICKSTART.md`
3. 检查 `multi_strategy_recall.py` 中的文档字符串

## 🔧 调试技巧

### 启用详细日志
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 单独测试每个策略
```python
# 测试ItemCF
test_user = all_click_df['user_id'].iloc[0]
result = itemcf_recall.predict(test_user, all_click_df)
print(f"ItemCF结果: {result[:5]}")

# 测试Embedding
result = embedding_recall.predict(test_user, all_click_df)
print(f"Embedding结果: {result[:5]}")

# 测试Popularity
result = popularity_recall.predict(test_user, all_click_df)
print(f"Popularity结果: {result[:5]}")
```

### 检查数据一致性
```python
# 检查用户数
print(f"点击数据用户数: {all_click_df['user_id'].nunique()}")
print(f"召回结果用户数: {len(final_recall_results)}")

# 检查文章数
print(f"点击数据文章数: {all_click_df['click_article_id'].nunique()}")
print(f"Embedding文章数: {len(item_emb_dict)}")
```

---

**祝你执行顺利！🚀**
