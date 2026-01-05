# 多路召回策略完整实现代码
# Complete Multi-Strategy Recall Implementation
# 可直接复制到 新闻推荐系统-多路召回.ipynb

import pandas as pd
import numpy as np
from collections import defaultdict
import logging
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==================== 1. ItemCF 协同过滤召回 ====================
class ItemCFRecall:
    """基于物品的协同过滤召回策略
    
    通过用户点击序列构建物品相似度矩阵，为每个用户推荐与其历史点击相似的物品
    """
    
    def __init__(self, sim_item_topk: int = 100, recall_item_number: int = 100):
        """
        Args:
            sim_item_topk: 每个物品保留的最相似物品数量
            recall_item_number: 每个用户召回的候选物品数量
        """
        self.sim_item_topk = sim_item_topk
        self.recall_item_number = recall_item_number
        self.item_sim_dict = {}  # {item_id: [(similar_item, similarity_score), ...]}
        self.is_fitted = False
    
    def fit(self, click_df: pd.DataFrame) -> 'ItemCFRecall':
        """训练ItemCF模型，构建物品相似度矩阵
        
        Args:
            click_df: 包含 user_id, click_article_id 列的DataFrame
        """
        logger.info("开始训练ItemCF模型...")
        
        # 统计物品共现次数
        item_cnt = defaultdict(int)  # 物品点击次数
        item_pair_cnt = defaultdict(int)  # 物品对共现次数
        
        for user_id, items in click_df.groupby('user_id')['click_article_id']:
            items_list = list(items)
            
            # 统计单个物品
            for item in items_list:
                item_cnt[item] += 1
            
            # 统计物品对（窗口内共现）
            for i in range(len(items_list)):
                for j in range(i + 1, len(items_list)):
                    item_i, item_j = items_list[i], items_list[j]
                    item_pair_cnt[(item_i, item_j)] += 1
                    item_pair_cnt[(item_j, item_i)] += 1
        
        logger.info(f"统计完成: {len(item_cnt)} 个物品, {len(item_pair_cnt)} 个物品对")
        
        # 计算物品相似度 (余弦相似度)
        for (item_i, item_j), co_count in item_pair_cnt.items():
            similarity = co_count / np.sqrt(item_cnt[item_i] * item_cnt[item_j])
            
            if item_i not in self.item_sim_dict:
                self.item_sim_dict[item_i] = []
            self.item_sim_dict[item_i].append((item_j, similarity))
        
        # 只保留top-K相似物品
        for item, sim_list in self.item_sim_dict.items():
            self.item_sim_dict[item] = sorted(sim_list, key=lambda x: x[1], reverse=True)[:self.sim_item_topk]
        
        self.is_fitted = True
        logger.info(f"✓ ItemCF训练完成，覆盖 {len(self.item_sim_dict)} 个物品")
        return self
    
    def predict(self, user_id: int, user_history: List[Tuple[int, int]]) -> List[int]:
        """为单个用户生成候选物品
        
        Args:
            user_id: 用户ID
            user_history: 用户历史点击 [(article_id, timestamp), ...]
            
        Returns:
            候选物品ID列表
        """
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用 fit()")
        
        # 聚合用户历史中每个物品的相似物品得分
        scores = defaultdict(float)
        clicked_items = set(item for item, _ in user_history)
        
        for item, timestamp in user_history:
            if item in self.item_sim_dict:
                for similar_item, sim_score in self.item_sim_dict[item]:
                    # 过滤掉已点击物品
                    if similar_item not in clicked_items:
                        scores[similar_item] += sim_score
        
        # 按得分排序，返回top-N
        candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in candidates[:self.recall_item_number]]
    
    def predict_batch(self, user_ids: List[int], user_history_dict: Dict[int, List[Tuple[int, int]]]) -> Dict[int, List[int]]:
        """批量预测
        
        Args:
            user_ids: 用户ID列表
            user_history_dict: {user_id: [(article_id, timestamp), ...]}
            
        Returns:
            {user_id: [candidate_articles]}
        """
        results = {}
        for user_id in user_ids:
            history = user_history_dict.get(user_id, [])
            results[user_id] = self.predict(user_id, history)
        return results


# ==================== 2. Embedding 向量召回 ====================
class EmbeddingRecall:
    """基于文章向量的内容召回策略
    
    通过计算用户向量（历史点击文章的平均向量）与所有文章向量的相似度进行召回
    """
    
    def __init__(self, recall_item_number: int = 100, use_faiss: bool = False):
        """
        Args:
            recall_item_number: 召回数量
            use_faiss: 是否使用FAISS加速（需要安装 faiss-cpu 或 faiss-gpu）
        """
        self.recall_item_number = recall_item_number
        self.use_faiss = use_faiss
        self.user_embeddings = {}  # {user_id: embedding}
        self.article_embeddings = None  # np.array
        self.article_ids = None  # np.array
        self.article_to_idx = {}
        self.is_fitted = False
        
        if use_faiss:
            try:
                import faiss
                self.faiss_index = None
                logger.info("FAISS已启用")
            except ImportError:
                logger.warning("FAISS未安装，将使用原生计算")
                self.use_faiss = False
    
    def fit(self, click_df: pd.DataFrame, embeddings_df: pd.DataFrame) -> 'EmbeddingRecall':
        """训练向量召回模型
        
        Args:
            click_df: 点击日志
            embeddings_df: 文章向量，需包含 article_id 和 emb_* 列
        """
        logger.info("开始训练Embedding召回模型...")
        
        # 提取文章向量
        self.article_ids = embeddings_df['article_id'].values
        emb_cols = [col for col in embeddings_df.columns if col.startswith('emb_')]
        self.article_embeddings = embeddings_df[emb_cols].values.astype('float32')
        self.article_to_idx = {aid: idx for idx, aid in enumerate(self.article_ids)}
        
        logger.info(f"文章向量维度: {self.article_embeddings.shape}")
        
        # 计算用户向量（历史点击文章的平均向量）
        for user_id, items in click_df.groupby('user_id')['click_article_id']:
            item_indices = [self.article_to_idx[item] for item in items if item in self.article_to_idx]
            
            if item_indices:
                user_emb = self.article_embeddings[item_indices].mean(axis=0)
                # L2 归一化
                norm = np.linalg.norm(user_emb)
                if norm > 1e-10:
                    user_emb = user_emb / norm
                self.user_embeddings[user_id] = user_emb
        
        # 如果使用FAISS，构建索引
        if self.use_faiss:
            import faiss
            # 归一化文章向量（用于余弦相似度）
            faiss.normalize_L2(self.article_embeddings)
            
            # 构建索引
            dimension = self.article_embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner Product (余弦相似度)
            self.faiss_index.add(self.article_embeddings)
            logger.info("✓ FAISS索引构建完成")
        
        self.is_fitted = True
        logger.info(f"✓ Embedding召回训练完成，覆盖 {len(self.user_embeddings)} 个用户")
        return self
    
    def predict(self, user_id: int) -> List[int]:
        """为用户生成候选"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        if user_id not in self.user_embeddings:
            return []  # 冷启动用户
        
        user_emb = self.user_embeddings[user_id]
        
        if self.use_faiss:
            # FAISS快速检索
            D, I = self.faiss_index.search(user_emb.reshape(1, -1), self.recall_item_number)
            return self.article_ids[I[0]].tolist()
        else:
            # 原生计算余弦相似度
            similarities = self.article_embeddings @ user_emb
            top_indices = np.argsort(similarities)[-self.recall_item_number:][::-1]
            return self.article_ids[top_indices].tolist()
    
    def predict_batch(self, user_ids: List[int]) -> Dict[int, List[int]]:
        """批量预测"""
        return {uid: self.predict(uid) for uid in user_ids}


# ==================== 3. Popularity 热门召回 ====================
class PopularityRecall:
    """基于热度的召回策略
    
    返回全局最热门的文章，用于冷启动场景和保底策略
    """
    
    def __init__(self, recall_item_number: int = 100):
        self.recall_item_number = recall_item_number
        self.popular_items = []
        self.is_fitted = False
    
    def fit(self, click_df: pd.DataFrame) -> 'PopularityRecall':
        """统计热门文章"""
        logger.info("开始训练Popularity召回...")
        
        item_counts = click_df['click_article_id'].value_counts()
        self.popular_items = item_counts.head(self.recall_item_number).index.tolist()
        
        logger.info(f"✓ 最热门文章: {self.popular_items[0]} (点击{item_counts.iloc[0]}次)")
        self.is_fitted = True
        return self
    
    def predict(self, user_id: int) -> List[int]:
        """返回热门文章（所有用户相同）"""
        return self.popular_items
    
    def predict_batch(self, user_ids: List[int]) -> Dict[int, List[int]]:
        """批量预测"""
        return {uid: self.popular_items for uid in user_ids}


# ==================== 4. RecallFusion 多路召回融合 ====================
class RecallFusion:
    """多路召回融合策略
    
    将多个召回策略的结果进行融合，支持加权平均、投票等方法
    """
    
    def __init__(self, 
                 recalls: Dict[str, object], 
                 weights: Optional[Dict[str, float]] = None,
                 method: str = 'weighted_avg'):
        """
        Args:
            recalls: 召回策略字典 {'itemcf': itemcf_model, 'embedding': emb_model, ...}
            weights: 权重字典 {'itemcf': 0.6, 'embedding': 0.3, ...}，默认均等权重
            method: 融合方法 'weighted_avg', 'voting', 'ranking'
        """
        self.recalls = recalls
        self.method = method
        
        # 设置权重
        if weights is None:
            # 默认均等权重
            self.weights = {name: 1.0 / len(recalls) for name in recalls}
        else:
            # 归一化权重
            total = sum(weights.values())
            self.weights = {name: w / total for name, w in weights.items()}
        
        logger.info(f"召回融合初始化: {len(recalls)} 个策略")
        for name, weight in self.weights.items():
            logger.info(f"  - {name}: {weight:.2%}")
    
    def predict(self, user_id: int, num_candidates: int = 100, 
                user_history: Optional[List[Tuple[int, int]]] = None) -> List[int]:
        """融合多路召回结果
        
        Args:
            user_id: 用户ID
            num_candidates: 最终返回的候选数量
            user_history: 用户历史（ItemCF需要）
            
        Returns:
            融合后的候选物品列表
        """
        all_candidates = {}  # {item_id: fused_score}
        
        # 获取各路召回结果
        for strategy_name, recall_model in self.recalls.items():
            try:
                # 针对ItemCF传入user_history
                if strategy_name == 'itemcf' and user_history is not None:
                    candidates = recall_model.predict(user_id, user_history)
                else:
                    candidates = recall_model.predict(user_id)
                
                # 根据融合方法计算得分
                if self.method in ['weighted_avg', 'ranking']:
                    # 排名加权：rank 1 得分最高
                    for rank, item in enumerate(candidates):
                        if item not in all_candidates:
                            all_candidates[item] = 0.0
                        # 分数 = 1/(rank+1) * 权重
                        score = (1.0 / (rank + 1)) * self.weights[strategy_name]
                        all_candidates[item] += score
                
                elif self.method == 'voting':
                    # 投票法：出现次数加权
                    for item in candidates:
                        if item not in all_candidates:
                            all_candidates[item] = 0.0
                        all_candidates[item] += self.weights[strategy_name]
                
            except Exception as e:
                logger.warning(f"策略 {strategy_name} 预测失败: {e}")
                continue
        
        # 按融合得分排序
        sorted_items = sorted(all_candidates.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in sorted_items[:num_candidates]]
    
    def predict_batch(self, user_ids: List[int], num_candidates: int = 100,
                     user_history_dict: Optional[Dict[int, List[Tuple[int, int]]]] = None) -> Dict[int, List[int]]:
        """批量融合预测"""
        results = {}
        for user_id in user_ids:
            history = user_history_dict.get(user_id, []) if user_history_dict else None
            results[user_id] = self.predict(user_id, num_candidates, history)
        return results


# ==================== 5. 使用示例 ====================
def run_multi_strategy_recall_pipeline():
    """完整的多路召回流程"""
    
    # ===== 步骤1: 加载数据 =====
    logger.info("=" * 50)
    logger.info("步骤1: 加载数据")
    logger.info("=" * 50)
    
    # 替换为你的数据路径
    clicks = pd.read_csv('../data/train_click_log.csv')
    embeddings = pd.read_csv('../data/articles_emb.csv')
    
    print(f"点击日志: {clicks.shape}")
    print(f"文章向量: {embeddings.shape}")
    print(f"用户数: {clicks['user_id'].nunique()}")
    print(f"文章数: {clicks['click_article_id'].nunique()}")
    
    # ===== 步骤2: 训练各路召回 =====
    logger.info("\n" + "=" * 50)
    logger.info("步骤2: 训练各路召回策略")
    logger.info("=" * 50)
    
    # 2.1 ItemCF
    itemcf = ItemCFRecall(sim_item_topk=100, recall_item_number=100)
    itemcf.fit(clicks)
    
    # 2.2 Embedding
    embedding = EmbeddingRecall(recall_item_number=100, use_faiss=False)
    embedding.fit(clicks, embeddings)
    
    # 2.3 Popularity
    popularity = PopularityRecall(recall_item_number=100)
    popularity.fit(clicks)
    
    # ===== 步骤3: 创建融合策略 =====
    logger.info("\n" + "=" * 50)
    logger.info("步骤3: 创建多路召回融合")
    logger.info("=" * 50)
    
    fusion = RecallFusion(
        recalls={
            'itemcf': itemcf,
            'embedding': embedding,
            'popularity': popularity
        },
        weights={
            'itemcf': 0.6,       # 协同过滤权重60%
            'embedding': 0.3,    # 向量召回权重30%
            'popularity': 0.1    # 热门召回权重10%
        },
        method='weighted_avg'
    )
    
    # ===== 步骤4: 生成预测 =====
    logger.info("\n" + "=" * 50)
    logger.info("步骤4: 生成所有用户的预测")
    logger.info("=" * 50)
    
    # 构建用户历史字典
    user_history_dict = {}
    for user_id, group in clicks.groupby('user_id'):
        user_history_dict[user_id] = list(zip(
            group['click_article_id'].values,
            group['click_timestamp'].values
        ))
    
    # 批量预测
    all_users = clicks['user_id'].unique()
    print(f"开始为 {len(all_users)} 个用户生成预测...")
    
    predictions = {}
    batch_size = 10000
    
    for i in range(0, len(all_users), batch_size):
        batch_users = all_users[i:i + batch_size]
        batch_preds = fusion.predict_batch(
            batch_users, 
            num_candidates=50,
            user_history_dict=user_history_dict
        )
        predictions.update(batch_preds)
        
        if (i + batch_size) % 50000 == 0:
            logger.info(f"  进度: {i + batch_size}/{len(all_users)}")
    
    print(f"✓ 完成 {len(predictions)} 个用户的预测")
    
    # ===== 步骤5: 生成提交文件 =====
    logger.info("\n" + "=" * 50)
    logger.info("步骤5: 生成提交文件")
    logger.info("=" * 50)
    
    submission_rows = []
    for user_id, candidates in predictions.items():
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
    
    print(f"✓ 提交文件已保存: submission_multi_strategy.csv")
    print(f"提交文件形状: {submission.shape}")
    print("\n前5行预览:")
    print(submission.head())
    
    return predictions, submission


# ==================== 运行 ====================
if __name__ == '__main__':
    predictions, submission = run_multi_strategy_recall_pipeline()
    print("\n✓ 多路召回流程执行完成！")
