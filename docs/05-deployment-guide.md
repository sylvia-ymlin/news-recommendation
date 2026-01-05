# 部署与使用指南

本文档提供远程服务器部署和日常开发流程指导。

---

## 一、服务器环境配置

### 1.1 首次部署

**SSH配置**（本地 `~/.ssh/config`）：
```bash
Host news-server
    HostName your-server-ip
    User root
    Port 22
    IdentityFile ~/.ssh/id_rsa
```

**测试连接**：
```bash
ssh news-server
# 应能免密登录
```

### 1.2 创建项目目录

```bash
ssh news-server
mkdir -p ~/news-recommendation
mkdir -p /root/autodl-tmp/news-rec-data  # 大文件存储（100GB SSD）
```

### 1.3 安装依赖

```bash
# 基础依赖
pip3 install pandas numpy scikit-learn xgboost tqdm

# NumPy版本控制（Faiss兼容性）
pip3 install "numpy>=1.21,<2.0"

# Faiss（向量检索）
pip3 install faiss-cpu
# 或GPU版本（需CUDA 11.x）
# pip3 install faiss-gpu
```

**验证安装**：
```bash
python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python3 -c "import faiss; print(f'Faiss: {faiss.__version__}, GPUs: {faiss.get_num_gpus()}')"
python3 -c "import xgboost; print(f'XGBoost: {xgboost.__version__}')"
```

预期输出：
```
NumPy: 1.26.4
Faiss: 1.7.4, GPUs: 0  (或1+如果GPU可用)
XGBoost: 2.0.3
```

---

## 二、代码与数据同步

### 2.1 使用部署脚本

**本地执行**（推荐）：
```bash
cd /path/to/coding/
bash deploy_to_server.sh
```

脚本内容：
```bash
#!/bin/bash
# 同步代码
rsync -avz --exclude='.git' --exclude='data/' --exclude='temp_results/' \
    ./ news-server:~/news-recommendation/

# 同步数据（首次或数据更新时）
scp data/*.csv news-server:~/news-recommendation/data/

echo "✅ 部署完成"
```

### 2.2 手动同步（按需）

**同步脚本**：
```bash
scp scripts/multi_recall.py news-server:~/news-recommendation/scripts/
```

**同步数据**：
```bash
scp data/articles_emb.csv news-server:~/news-recommendation/data/
```

**下载结果**：
```bash
scp news-server:~/news-recommendation/submission_*.csv ./
scp news-server:/root/autodl-tmp/news-rec-data/*.pkl ./temp_results/
```

---

## 三、完整执行流程

### 3.1 多路召回生成

```bash
# SSH到服务器
ssh news-server
cd ~/news-recommendation

# 热度 + ItemCF + UserCF
python3 scripts/multi_recall.py
# 输出: /root/autodl-tmp/news-rec-data/{hot_list,itemcf_sim,usercf_sim}.pkl
# 耗时: ~10分钟（128核）

# Faiss向量召回
python3 scripts/embedding_recall_faiss.py
# 输出: /root/autodl-tmp/news-rec-data/emb_sim_faiss.pkl
# 耗时: ~8分钟（CPU）
```

**验证输出**：
```bash
ls -lh /root/autodl-tmp/news-rec-data/
# 应看到：
# hot_list.pkl (~50KB)
# itemcf_sim.pkl (~50MB)
# usercf_sim.pkl (~75MB)
# emb_sim_faiss.pkl (~200MB)
```

### 3.2 特征工程

```bash
# 提取训练特征
python3 scripts/feature_engineering.py
# 输出: temp_results/features.pkl

# 构建训练样本（正负采样）
python3 scripts/build_samples.py
# 输出: /root/autodl-tmp/news-rec-data/training_samples.pkl
# 耗时: ~15分钟
# 样本量: 556万（111万正 + 444万负）
```

### 3.3 排序模型训练

```bash
# XGBoost Ranker
python3 scripts/train_ranker.py
# 输出: /root/autodl-tmp/news-rec-data/xgb_ranker.json
# 耗时: ~30分钟（GPU）或2小时（CPU）
```

**监控训练**：
```bash
tail -f train_ranker.log  # 如果重定向了输出
# 或直接观察终端输出的AUC曲线
```

### 3.4 测试集推理

```bash
# 提取测试特征
python3 scripts/extract_test_features.py
# 输出: temp_results/test_features.pkl

# 生成提交文件
python3 scripts/generate_submission.py
# 输出: submission_ranker_top5_v3.csv
# 耗时: ~5秒（多核并行）
```

### 3.5 下载提交文件

```bash
# 本地执行
scp news-server:~/news-recommendation/submission_*.csv ./
```

---

## 四、快速验证（Baseline）

如果只需快速测试pipeline：

```bash
ssh news-server
cd ~/news-recommendation

# 热度baseline（14秒生成）
python3 scripts/baseline_fast.py
# 输出: submission_baseline.csv
# MRR: ~0.0192
```

---

## 五、常见问题排查

### 5.1 Faiss导入错误

**症状**：
```python
>>> import faiss
AttributeError: module 'faiss' has no attribute 'StandardGpuResources'
# 或
AttributeError: _ARRAY_API not found
```

**解决**：
```bash
# 检查NumPy版本
python3 -c "import numpy; print(numpy.__version__)"
# 如果≥2.0，降级
pip3 install --force-reinstall "numpy<2.0"

# 重新安装Faiss
pip3 uninstall faiss-cpu faiss-gpu -y
pip3 install faiss-cpu
```

### 5.2 内存不足

**症状**：
```
MemoryError: Unable to allocate array
```

**排查**：
```bash
# 检查内存使用
free -h
# 或
htop
```

**解决**：
- 减少批量大小（`build_samples.py` 中的负采样倍数）
- 使用 `/root/autodl-tmp/` 存储大文件（100GB可用）
- 分批处理：`np.array_split(data, n_batches)`

### 5.3 XGBoost GPU不可用

**症状**：
```
[WARN] GPU training not available, falling back to CPU
```

**验证GPU**：
```bash
nvidia-smi
# 检查CUDA版本和GPU状态
```

**解决**：
- 安装GPU版本：`pip3 install xgboost[gpu]`
- 或在代码中改为CPU：`tree_method='hist'`（而非`'gpu_hist'`）

### 5.4 文件路径错误

**症状**：
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/train_click_log.csv'
```

**排查**：
```bash
pwd  # 确认当前目录
ls -lh data/  # 检查数据文件
```

**解决**：
- 确保在 `~/news-recommendation/` 目录执行
- 或使用绝对路径：`/root/news-recommendation/data/train_click_log.csv`

---

## 六、性能优化Tips

### 6.1 多核并行

脚本已默认使用128核（如 `baseline_fast.py`），如果核数不同：

```python
# 修改脚本中的
NUM_CORES = 128  # → 改为实际核数
# 或自动检测
import multiprocessing
NUM_CORES = multiprocessing.cpu_count()
```

### 6.2 Faiss参数调优

**速度优先**：
```python
nlist = 2048    # 减少聚类数
nprobe = 8      # 减少搜索探测
TOPK = 50       # 减少召回数
```

**精度优先**：
```python
nlist = 8192    # 增加聚类数
nprobe = 32     # 增加搜索探测
TOPK = 200      # 增加召回数
```

平衡点（当前配置）：
```python
nlist = 4096
nprobe = 16
TOPK = 100
```

### 6.3 存储加速

**使用高速SSD**：
```python
SAVE_PATH = '/root/autodl-tmp/news-rec-data/'  # 100GB SSD
# 而非
SAVE_PATH = './temp_results/'  # 可能在较慢的挂载盘
```

**批量写入**：
```python
# ❌ 低效：逐行写入
for rec in recommendations:
    f.write(f"{rec}\n")

# ✅ 高效：批量写入
pd.DataFrame(recommendations).to_csv('output.csv', index=False)
```

---

## 七、开发工作流

### 7.1 本地开发

```bash
# 1. 在Jupyter中探索和原型
jupyter notebook notebooks/

# 2. 将成熟代码迁移到scripts/
# 例如：notebook → scripts/new_recall.py

# 3. 本地测试（小数据）
python3 scripts/new_recall.py  # 用data/的前1000行测试
```

### 7.2 服务器训练

```bash
# 4. 同步到服务器
bash deploy_to_server.sh

# 5. SSH执行
ssh news-server "cd ~/news-recommendation && python3 scripts/new_recall.py"

# 6. 下载结果分析
scp news-server:~/news-recommendation/outputs/* ./outputs/
```

### 7.3 Git版本控制

```bash
# 每日提交
git add .
git commit -m "feat: add faiss embedding recall"
git push origin main

# 查看变更
git status
git diff
```

---

## 八、检查清单（推送前）

### 代码完整性
- [ ] 所有脚本有 shebang：`#!/usr/bin/env python3`
- [ ] 导入顺序规范：标准库 → 第三方 → 本地
- [ ] 函数有docstring
- [ ] 硬编码路径改为配置变量

### 数据安全
- [ ] `.gitignore` 包含 `data/*.csv`
- [ ] 大文件（pkl > 100MB）在 `.gitignore`
- [ ] 提交文件只保留最新版本

### 文档更新
- [ ] `README.md` 反映最新功能
- [ ] `docs/04-technical-challenges.md` 记录新问题
- [ ] 部署脚本 `deploy_to_server.sh` 可用

### 服务器状态
- [ ] 远程代码与本地同步
- [ ] 中间结果已下载备份
- [ ] 不需要的临时文件已清理

---

## 九、联系与支持

**开发者**：ymlin  
**项目路径**：
- 本地：`/Users/ymlin/.../天池新闻推荐/coding`
- 服务器：`~/news-recommendation`

**问题反馈**：
- 提交 GitHub Issue
- 或发送邮件至 [your-email]

---

**最后更新**：2026-01-05  
**文档版本**：v1.0
