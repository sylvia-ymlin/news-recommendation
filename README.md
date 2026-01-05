# News Recommendation System

## Overview

This project implements an end-to-end recommendation system for news articles. The system addresses a cold-start scenario with 50,000 test users and approximately 360,000 news articles. It follows a standard retrieval-ranking pipeline: multi-channel retrieval generates candidates, feature engineering constructs input representations, and a learned ranking model produces final recommendations.

### Data Summary
- Training users: 200,000
- Training interactions: 1,112,623
- Test users: 50,000 (100% new users)
- Articles: 255,756 with embeddings
- Training samples: 5.56M (1.1M positive + 4.4M negative)

### Performance
- Baseline (popularity): MRR = 0.0192
- Current best (v2 ranker): MRR = 0.0119
- Bottleneck identified: Retrieval quality

## System Architecture

### Component Breakdown

```
RETRIEVAL PHASE
├─ Popularity: Global frequency ranking
├─ ItemCF: Item-item collaborative filtering
├─ Embedding: Vector similarity via Faiss
└─ UserCF: User-user collaborative filtering

CANDIDATE FUSION (pending)
│
FEATURE ENGINEERING
├─ User features (9D)
├─ Article features (7D)
└─ Interaction features (5D)

RANKING MODEL
├─ Algorithm: XGBoost (binary classification)
├─ Training samples: 5.56M
└─ Validation AUC: 0.9906

FINAL OUTPUT
└─ CSV submission format
```

## File Organization

```
scripts/
├── multi_recall.py              Multi-channel retrieval orchestration
├── embedding_recall_faiss.py    Faiss vector similarity (8 min)
├── feature_engineering.py       21D feature extraction
├── build_samples.py             5.56M training sample generation
├── train_ranker.py              XGBoost model training (2 hours CPU)
├── extract_test_features.py     Test set features
├── generate_submission.py       CSV output format
└── baseline_fast.py             Popularity baseline (14 sec)

docs/
├── 04-technical-challenges.md   Implementation details
└── 06-server-data-persistence.md Data management and GPU notes

notebooks/
├── 赛题理解.ipynb               Problem and data exploration
├── 数据分析.ipynb               Distribution analysis
└── 特征工程.ipynb               Feature engineering walkthrough
```

## Execution Workflow

### Baseline Verification (14 seconds)
```bash
python scripts/baseline_fast.py
# Output: popularity-based Top-5 (MRR ~0.0192)
```

### Full Pipeline (CPU, ~2.5 hours)

1. Multi-channel retrieval:
   ```bash
   python scripts/multi_recall.py                 # 10 min
   python scripts/embedding_recall_faiss.py      # 8 min
   ```

2. Feature engineering:
   ```bash
   python scripts/feature_engineering.py         # 5 min
   python scripts/build_samples.py               # 15 min
   ```

3. Ranking model:
   ```bash
   python scripts/train_ranker.py                # 2 hours
   python scripts/extract_test_features.py       # 3 min
   ```

4. Submission:
   ```bash
   python scripts/generate_submission.py         # 5 sec
   ```

## Key Technical Solutions

### 1. Efficient Vector Similarity Search

**Problem**: 255K articles × 255K similarity = 16 trillion FLOPs

**Solution**: Faiss IVF with 4096 clusters, probe=16
- Reduces computation from 3 hours to 8 minutes
- 22.5x acceleration with acceptable recall (~95%)
- Handles NaN/Inf vectors and array contiguity

### 2. XGBoost with Auto GPU Detection

**Configuration**:
```python
tree_method = 'gpu_hist' if gpu_available else 'hist'
predictor = 'gpu_predictor' if gpu_available else 'cpu_predictor'
```

No GPU required; CPU-only mode fully supported.

### 3. Negative Sampling Strategy

- 1.1M positive samples from training
- 4.4M negative samples (4x ratio)
- 80-20 train-validation split
- Total training samples: 5.56M

## Current Status and Issues

### Completed ✅
- Retrieval infrastructure: All four strategies implemented
- Feature engineering: 21D representation
- Ranking model: Trained XGBoost with validation AUC=0.9906
- Data persistence: All outputs stored in `/root/autodl-tmp/`
- **Retrieval fusion**: Implemented weighted combination (ItemCF 40% + Embedding 35% + Popularity 25%)
  - Generated fused_recalls.pkl (47MB, 50k users × ~200 candidates)
  - fusion_simple.py: Efficient multi-method fusion
- **v3 Submission**: Generated submission_ranker_top5_v3.csv (3.8MB, 250k rows)
  - Direct Top-5 extraction from fused candidates
  - generate_submission_light.py: Memory-efficient submission generation

### Performance Analysis

**v2 (MRR=0.0119) vs Baseline (MRR=0.0192):**
- Validation AUC near-perfect (0.9906)
- Test performance degraded
- Root cause: Feature distribution mismatch (training users ≠ test users)
- **Key insight**: Ranking quality not the bottleneck; retrieval is

**v3 Expected Improvement:**
- Added ItemCF similarity-based recommendations
- Added Embedding-based vector recommendations  
- Maintains Popularity for diversity
- Should outperform v2 due to multi-method fusion

## Dependencies

```
pandas >= 1.3
numpy >= 1.21, < 2.0
scikit-learn >= 1.0
xgboost >= 2.0
faiss-cpu >= 1.7
tqdm
```

## Design Decisions

1. **NumPy < 2.0**: Faiss compatibility requirement
2. **Faiss over NumPy**: 22.5x faster for similarity search
3. **IVF over HNSW**: Memory efficiency (255K vectors, 250D)
4. **Negative sampling ratio 4:1**: Balances positive-biased dataset
5. **XGBoost over LightGBM**: Better AUC on this dataset in preliminary tests

## Next Steps

1. ✅ Implement explicit retrieval fusion (weighted combination) 
2. ✅ Regenerate v3 submission using fused candidates
3. Evaluate v3 against baseline threshold (0.0192)
4. If v3 > 0.0192: Submit to competition
5. If v3 ≤ 0.0192: Explore alternative fusion weights or ranking adjustments

## References

- Competition: https://www.tianchi.aliyun.com/
- Faiss: https://faiss.ai/
- XGBoost: https://xgboost.readthedocs.io/

---

**Status**: 95% pipeline completion  
**Current Task**: v3 submission ready for evaluation  
**Last Modified**: January 6, 2026 03:25 UTC
