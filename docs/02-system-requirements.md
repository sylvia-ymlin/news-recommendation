# System Requirements and Data Persistence

## Environment Requirements

### Python Dependencies
```
pandas >= 1.3
numpy >= 1.21, < 2.0
scikit-learn >= 1.0
xgboost >= 2.0
faiss-cpu >= 1.7.0
tqdm
```

### Hardware Specifications
- **Minimum**: 16GB RAM, 100GB storage
- **Recommended**: 32GB RAM, 200GB SSD for intermediate files
- **GPU**: Optional (CPU-only fully supported)

### Installation
```bash
pip install pandas numpy scikit-learn xgboost faiss-cpu "numpy<2.0"
```

## Data Storage Architecture

### Persistent Storage
All critical outputs stored in `/root/autodl-tmp/news-rec-data/`:
- Does not require GPU
- Survives system shutdown
- Not deleted during maintenance

**Files**:
- `training_samples.pkl` (934MB): 5.56M training instances
- `xgb_ranker.json` (10MB): Trained model
- `emb_sim_faiss.pkl` (200MB): Embedding similarity results
- `itemcf_sim.pkl` (50MB): Item-item collaborative filtering

### Temporary Storage
Files in `./temp_results/` and `./outputs/` are transient and may be regenerated as needed.

## GPU Dependency Analysis

### Current Usage
- **XGBoost**: Tree building (optional GPU acceleration)
  - GPU mode: ~30 minutes training
  - CPU mode: ~2 hours training
  - Model performance: Identical

- **Faiss**: Vector clustering and search (optional GPU acceleration)
  - GPU mode: ~2 minutes retrieval
  - CPU mode: ~8 minutes retrieval
  - Recall: Approximately 95% (IVF approximate search)

- **Other components**: Entirely CPU-based
  - Feature engineering: CPU only
  - Sampling: CPU only
  - Model inference: CPU only

### Operational Decision
GPU is optional for current pipeline. CPU-only mode fully functional with acceptable runtimes.

## Computational Workflow Timeline (CPU)

| Step | Script | Time | Notes |
|------|--------|------|-------|
| Retrieval | multi_recall.py | 10 min | ItemCF, UserCF, popularity |
| Retrieval | embedding_recall_faiss.py | 8 min | Faiss IVF index search |
| Features | feature_engineering.py | 5 min | 21D feature extraction |
| Sampling | build_samples.py | 15 min | 5.56M sample generation |
| Training | train_ranker.py | 120 min | XGBoost with early stopping |
| Features | extract_test_features.py | 3 min | Test set feature prep |
| Submission | generate_submission.py | 5 sec | CSV output format |
| **Total** | | **161 min** | ~2.7 hours |

## Environment Auto-Detection

The system implements automatic GPU detection:

```python
# XGBoost
try:
    gpu_available = len(xgb.device.cuda().get_device_properties()) > 0
except:
    gpu_available = False

tree_method = 'gpu_hist' if gpu_available else 'hist'

# Faiss
try:
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)
except Exception as e:
    print(f'GPU unavailable, using CPU: {e}')
```

No manual configuration required; runtime automatically selects available hardware.

## Data Backup Strategy

Recommended procedure for offline environments:

```bash
# Download model and results
scp news-server:/root/autodl-tmp/news-rec-data/xgb_ranker.json ./backups/
scp news-server:/root/autodl-tmp/news-rec-data/*.pkl ./backups/

# Download input data
scp news-server:~/news-recommendation/data/*.csv ./backups/
```

These files enable reproduction or retraining on alternate systems.

---

**Design Note**: This architecture prioritizes data durability and computational flexibility. The separation of persistent storage (predictable location) from temporary caches (regenerable if needed) balances reliability with resource efficiency.
