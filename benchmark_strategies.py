"""
性能基准测试脚本 - 对比不同召回策略
Benchmark different recall strategies and fusion configurations
"""

import pandas as pd
import numpy as np
import time
from collections import defaultdict
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# 假设已经导入了召回策略
from multi_strategy_recall import (
    ItemCFRecall, EmbeddingRecall, PopularityRecall, RecallFusion
)


class RecallBenchmark:
    """召回策略性能基准测试"""
    
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                 embeddings_df: pd.DataFrame = None):
        """
        Args:
            train_df: 训练集点击日志
            test_df: 测试集点击日志
            embeddings_df: 文章向量
        """
        self.train_df = train_df
        self.test_df = test_df
        self.embeddings_df = embeddings_df
        
        # 构建ground truth
        self.ground_truth = test_df.groupby('user_id')['click_article_id'].apply(set).to_dict()
        self.test_users = list(self.ground_truth.keys())
        
        # 用户历史
        self.user_history = {}
        for user_id, group in train_df.groupby('user_id'):
            self.user_history[user_id] = list(zip(
                group['click_article_id'].values,
                group['click_timestamp'].values
            ))
        
        self.results = []
    
    def recall_at_k(self, predictions: Dict[int, List[int]], k: int = 5) -> float:
        """计算Recall@K"""
        hits = 0
        total = 0
        
        for user_id, true_items in self.ground_truth.items():
            if user_id not in predictions:
                continue
            
            pred_items = set(predictions[user_id][:k])
            hits += len(pred_items & true_items)
            total += len(true_items)
        
        return hits / total if total > 0 else 0.0
    
    def ndcg_at_k(self, predictions: Dict[int, List[int]], k: int = 5) -> float:
        """计算NDCG@K"""
        ndcg_sum = 0.0
        count = 0
        
        for user_id, true_items in self.ground_truth.items():
            if user_id not in predictions:
                continue
            
            pred_items = predictions[user_id][:k]
            
            # DCG
            dcg = 0.0
            for i, item in enumerate(pred_items):
                if item in true_items:
                    dcg += 1.0 / np.log2(i + 2)
            
            # IDCG
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(true_items), k)))
            
            if idcg > 0:
                ndcg_sum += dcg / idcg
            count += 1
        
        return ndcg_sum / count if count > 0 else 0.0
    
    def precision_at_k(self, predictions: Dict[int, List[int]], k: int = 5) -> float:
        """计算Precision@K"""
        precision_sum = 0.0
        count = 0
        
        for user_id, true_items in self.ground_truth.items():
            if user_id not in predictions:
                continue
            
            pred_items = set(predictions[user_id][:k])
            precision_sum += len(pred_items & true_items) / k
            count += 1
        
        return precision_sum / count if count > 0 else 0.0
    
    def coverage(self, predictions: Dict[int, List[int]]) -> float:
        """计算覆盖率（推荐了多少不同的物品）"""
        all_items = set()
        for items in predictions.values():
            all_items.update(items)
        
        total_items = self.train_df['click_article_id'].nunique()
        return len(all_items) / total_items
    
    def evaluate_strategy(self, strategy_name: str, recall_model, 
                         measure_time: bool = True) -> Dict:
        """评估单个策略"""
        print(f"\n{'='*60}")
        print(f"评估策略: {strategy_name}")
        print(f"{'='*60}")
        
        # 预测
        start_time = time.time()
        
        predictions = {}
        if strategy_name == 'itemcf':
            predictions = recall_model.predict_batch(self.test_users, self.user_history)
        else:
            predictions = recall_model.predict_batch(self.test_users)
        
        latency = (time.time() - start_time) / len(self.test_users) * 1000  # ms per user
        
        # 计算指标
        metrics = {
            'strategy': strategy_name,
            'recall@5': self.recall_at_k(predictions, k=5),
            'recall@10': self.recall_at_k(predictions, k=10),
            'ndcg@5': self.ndcg_at_k(predictions, k=5),
            'ndcg@10': self.ndcg_at_k(predictions, k=10),
            'precision@5': self.precision_at_k(predictions, k=5),
            'coverage': self.coverage(predictions),
            'latency_ms': latency if measure_time else None
        }
        
        self.results.append(metrics)
        
        # 打印结果
        print(f"Recall@5:     {metrics['recall@5']:.4f}")
        print(f"Recall@10:    {metrics['recall@10']:.4f}")
        print(f"NDCG@5:       {metrics['ndcg@5']:.4f}")
        print(f"NDCG@10:      {metrics['ndcg@10']:.4f}")
        print(f"Precision@5:  {metrics['precision@5']:.4f}")
        print(f"Coverage:     {metrics['coverage']:.4f}")
        if measure_time:
            print(f"Latency:      {metrics['latency_ms']:.2f} ms/user")
        
        return metrics
    
    def run_full_benchmark(self):
        """运行完整的基准测试"""
        print("\n" + "="*60)
        print("开始完整基准测试")
        print("="*60)
        print(f"训练集: {len(self.train_df)} 条点击")
        print(f"测试集: {len(self.test_df)} 条点击")
        print(f"测试用户: {len(self.test_users)} 个")
        
        # 1. ItemCF
        print("\n训练 ItemCF...")
        itemcf = ItemCFRecall(sim_item_topk=100, recall_item_number=100)
        itemcf.fit(self.train_df)
        self.evaluate_strategy('itemcf', itemcf)
        
        # 2. Embedding (如果有)
        if self.embeddings_df is not None:
            print("\n训练 Embedding...")
            embedding = EmbeddingRecall(recall_item_number=100, use_faiss=False)
            embedding.fit(self.train_df, self.embeddings_df)
            self.evaluate_strategy('embedding', embedding)
        else:
            print("\n⚠️  跳过Embedding（未提供向量数据）")
            embedding = None
        
        # 3. Popularity
        print("\n训练 Popularity...")
        popularity = PopularityRecall(recall_item_number=100)
        popularity.fit(self.train_df)
        self.evaluate_strategy('popularity', popularity)
        
        # 4. Fusion (如果有embedding)
        if embedding is not None:
            print("\n测试 Fusion 配置...")
            
            # 配置1: 均衡权重
            fusion1 = RecallFusion(
                recalls={'itemcf': itemcf, 'embedding': embedding, 'popularity': popularity},
                weights={'itemcf': 0.5, 'embedding': 0.3, 'popularity': 0.2},
                method='weighted_avg'
            )
            metrics1 = self.evaluate_strategy('fusion_balanced', fusion1)
            
            # 配置2: ItemCF为主
            fusion2 = RecallFusion(
                recalls={'itemcf': itemcf, 'embedding': embedding, 'popularity': popularity},
                weights={'itemcf': 0.7, 'embedding': 0.2, 'popularity': 0.1},
                method='weighted_avg'
            )
            metrics2 = self.evaluate_strategy('fusion_itemcf_heavy', fusion2)
            
            # 配置3: 投票法
            fusion3 = RecallFusion(
                recalls={'itemcf': itemcf, 'embedding': embedding, 'popularity': popularity},
                weights={'itemcf': 0.5, 'embedding': 0.3, 'popularity': 0.2},
                method='voting'
            )
            metrics3 = self.evaluate_strategy('fusion_voting', fusion3)
        
        return self.results
    
    def plot_results(self, save_path: str = 'benchmark_results.png'):
        """可视化结果"""
        if not self.results:
            print("没有结果可供可视化")
            return
        
        df = pd.DataFrame(self.results)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Recall对比
        ax1 = axes[0, 0]
        df[['strategy', 'recall@5', 'recall@10']].set_index('strategy').plot(
            kind='bar', ax=ax1, rot=45
        )
        ax1.set_title('Recall Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Recall')
        ax1.legend(['Recall@5', 'Recall@10'])
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. NDCG对比
        ax2 = axes[0, 1]
        df[['strategy', 'ndcg@5', 'ndcg@10']].set_index('strategy').plot(
            kind='bar', ax=ax2, rot=45, color=['orange', 'red']
        )
        ax2.set_title('NDCG Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('NDCG')
        ax2.legend(['NDCG@5', 'NDCG@10'])
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Precision vs Coverage
        ax3 = axes[1, 0]
        scatter = ax3.scatter(df['coverage'], df['precision@5'], 
                            s=200, alpha=0.6, c=range(len(df)), cmap='viridis')
        for idx, row in df.iterrows():
            ax3.annotate(row['strategy'], (row['coverage'], row['precision@5']),
                        fontsize=8, ha='center')
        ax3.set_xlabel('Coverage', fontsize=12)
        ax3.set_ylabel('Precision@5', fontsize=12)
        ax3.set_title('Precision vs Coverage', fontsize=14, fontweight='bold')
        ax3.grid(alpha=0.3)
        
        # 4. Latency对比
        ax4 = axes[1, 1]
        if df['latency_ms'].notna().any():
            df_latency = df[df['latency_ms'].notna()]
            df_latency[['strategy', 'latency_ms']].set_index('strategy').plot(
                kind='barh', ax=ax4, color='green', legend=False
            )
            ax4.set_xlabel('Latency (ms/user)', fontsize=12)
            ax4.set_title('Latency Comparison', fontsize=14, fontweight='bold')
            ax4.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ 结果图表已保存: {save_path}")
        plt.close()
    
    def export_results(self, csv_path: str = 'benchmark_results.csv'):
        """导出结果到CSV"""
        if not self.results:
            print("没有结果可供导出")
            return
        
        df = pd.DataFrame(self.results)
        df.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"✓ 结果已导出: {csv_path}")


# ==================== 使用示例 ====================
def main():
    """主函数 - 运行基准测试"""
    
    # 加载数据
    print("加载数据...")
    train_clicks = pd.read_csv('../data/train_click_log.csv')
    
    # 分割训练/测试集（时间排序）
    train_clicks = train_clicks.sort_values('click_timestamp')
    split_idx = int(len(train_clicks) * 0.8)
    
    train_df = train_clicks.iloc[:split_idx]
    test_df = train_clicks.iloc[split_idx:]
    
    print(f"训练集: {len(train_df)} 条")
    print(f"测试集: {len(test_df)} 条")
    
    # 加载向量（可选）
    try:
        embeddings_df = pd.read_csv('../data/articles_emb.csv')
        print(f"向量数据: {embeddings_df.shape}")
    except:
        print("⚠️  未找到向量数据，将跳过Embedding策略")
        embeddings_df = None
    
    # 运行基准测试
    benchmark = RecallBenchmark(train_df, test_df, embeddings_df)
    results = benchmark.run_full_benchmark()
    
    # 可视化和导出
    benchmark.plot_results('benchmark_results.png')
    benchmark.export_results('benchmark_results.csv')
    
    # 打印最佳策略
    print("\n" + "="*60)
    print("最佳策略排名 (by Recall@5)")
    print("="*60)
    df = pd.DataFrame(results)
    df_sorted = df.sort_values('recall@5', ascending=False)
    print(df_sorted[['strategy', 'recall@5', 'ndcg@5', 'coverage']].to_string(index=False))


if __name__ == '__main__':
    main()
