# Technical Implementation Notes

## 1. Faiss Vector Similarity Optimization

### Problem Statement
Computing similarity between 255,756 articles with 250-dimensional embeddings requires approximately 16 trillion floating-point operations using dense methods (estimated runtime: 3 hours).

### Solution: IVF (Inverted File Index)
The Faiss library implements IVF with the following configuration:
- Cluster count: 4,096 (approximately √n for n vectors)
- Probe count: 16 (search 0.4% of clusters)
- Similarity metric: Inner product (equivalent to cosine after L2 normalization)
- Search recall: Approximately 95% of true top-100

### Implementation Details

**Vector preprocessing**:
1. Handling missing values: Replace NaN and Inf with 0 (212 instances identified)
2. Memory layout: Convert to C-contiguous array for Faiss compatibility
3. Normalization: Apply L2 normalization for inner-product equivalence to cosine

**Index construction**:
```python
quantizer = faiss.IndexFlatIP(dim)
index = faiss.IndexIVFFlat(quantizer, dim, nlist=4096, metric=faiss.METRIC_INNER_PRODUCT)
index.train(sample_vectors)  # Train cluster centroids
index.add(all_vectors)       # Add all vectors to index
index.nprobe = 16            # Set search parameter
```

**Performance achieved**: 3 hours → 8 minutes (22.5x speedup)

### Technical Challenges Addressed

1. **GPU availability detection**: StandardGpuResources unavailable in CPU-only environment; fallback implemented
2. **NumPy version compatibility**: Faiss 1.7.4 requires NumPy < 2.0 due to C API changes
3. **Array contiguity**: Pandas slicing produces non-contiguous arrays; explicit conversion required

## 2. XGBoost Ranking Model

### Training Configuration
- Objective: Binary classification (sigmoid loss)
- Samples: 5.56M total (1.1M positive, 4.4M negative from 4:1 ratio)
- Features: 21-dimensional
- Boosting rounds: 500 with early stopping (patience=50)

### Feature Engineering

**User features (9D)**:
- Click count, active days, category preference distribution (3D), temporal statistics

**Article features (7D)**:
- Popularity rank, text length, publish date, category popularity, publishing frequency

**Interaction features (5D)**:
- User-category preference match, temporal decay, cross-category interest

### Model Performance

**Training metrics**:
- AUC: 0.9906 (validation)
- AUCPR: High precision at low recall

**Test performance**:
- v1: MRR = 0.0079 (feature representation error)
- v2: MRR = 0.0119 (improved with test features)
- Baseline: MRR = 0.0192 (popularity only)

### Analysis

The large gap between validation AUC (0.9906) and test MRR (0.0119 vs 0.0192 baseline) indicates:
1. **Distribution mismatch**: Training uses user-item interactions; test set is entirely cold-start
2. **Ranking not bottleneck**: Model precision is high but applicable candidates are limited
3. **Retrieval quality critical**: Candidate set composition determines upper bound on ranking performance

## 3. Cold-Start Scenario

### Data Characteristics
- Training set: 200,000 users
- Test set: 50,000 users (no overlap with training set)
- User ID ranges: Training [0-199999], Test [200000-249999]

### Impact on Collaborative Filtering
- User-based methods (UserCF): No historical user similarity available
- Item-based methods (ItemCF): Only effective when users have clicked items in training set
- Content-based methods (Embedding): Only viable approach for cold-start

## 4. Negative Sampling Strategy

**Rationale**: Training data exhibits positive bias (1.1M positive vs 364K articles)

**Implementation**:
- Sample 4 negative items for each positive interaction
- Negative sources: Random articles not clicked by user
- Maintains realistic ranking difficulty

**Result**: 5.56M total training samples with balanced class representation

## 5. Deployment and Data Persistence

### Storage Strategy
All model artifacts stored in persistent storage (`/root/autodl-tmp/news-rec-data/`):
- Training samples: training_samples.pkl (~934MB)
- Model checkpoint: xgb_ranker.json (~10MB)
- Retrieval indices: emb_sim_faiss.pkl (~200MB), itemcf_sim.pkl (~50MB)

### Computational Environment
- CPU-only supported (GPU optional for acceleration)
- XGBoost auto-detects GPU availability
- Faiss CPU version uses ~8 minutes vs GPU ~2 minutes (non-critical for one-time use)

---

**Status**: These solutions address specific technical constraints in the competition setting. Generalization to other datasets may require parameter tuning, particularly for IVF cluster count and Faiss probe parameter.
