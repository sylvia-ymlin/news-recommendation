"""
可视化工具 - 展示推荐系统的内部机制
Visualization tools for understanding recommendation system internals
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import networkx as nx


class RecallVisualizer:
    """召回策略可视化工具"""
    
    @staticmethod
    def plot_user_behavior_distribution(click_df: pd.DataFrame, 
                                        save_path: str = 'user_behavior.png'):
        """可视化用户行为分布"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 用户点击次数分布
        user_clicks = click_df.groupby('user_id').size()
        axes[0, 0].hist(user_clicks, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('点击次数', fontsize=12)
        axes[0, 0].set_ylabel('用户数量', fontsize=12)
        axes[0, 0].set_title('用户活跃度分布', fontsize=14, fontweight='bold')
        axes[0, 0].axvline(user_clicks.median(), color='red', linestyle='--', 
                          label=f'中位数: {user_clicks.median():.0f}')
        axes[0, 0].legend()
        
        # 2. 文章点击次数分布
        article_clicks = click_df.groupby('click_article_id').size()
        axes[0, 1].hist(article_clicks, bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[0, 1].set_xlabel('点击次数', fontsize=12)
        axes[0, 1].set_ylabel('文章数量', fontsize=12)
        axes[0, 1].set_title('文章热度分布', fontsize=14, fontweight='bold')
        axes[0, 1].axvline(article_clicks.median(), color='red', linestyle='--',
                          label=f'中位数: {article_clicks.median():.0f}')
        axes[0, 1].legend()
        axes[0, 1].set_yscale('log')
        
        # 3. 用户活跃度分类
        activity_bins = [0, 5, 20, 50, float('inf')]
        activity_labels = ['新用户\n(1-5次)', '普通用户\n(6-20次)', 
                          '活跃用户\n(21-50次)', '超级用户\n(>50次)']
        user_activity = pd.cut(user_clicks, bins=activity_bins, labels=activity_labels)
        activity_counts = user_activity.value_counts()
        
        axes[1, 0].bar(range(len(activity_counts)), activity_counts.values, 
                      color=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3'])
        axes[1, 0].set_xticks(range(len(activity_counts)))
        axes[1, 0].set_xticklabels(activity_labels, fontsize=10)
        axes[1, 0].set_ylabel('用户数量', fontsize=12)
        axes[1, 0].set_title('用户分层统计', fontsize=14, fontweight='bold')
        
        # 添加百分比标签
        total = activity_counts.sum()
        for i, v in enumerate(activity_counts.values):
            axes[1, 0].text(i, v, f'{v}\n({v/total*100:.1f}%)', 
                           ha='center', va='bottom', fontsize=10)
        
        # 4. 时间序列趋势
        click_df['date'] = pd.to_datetime(click_df['click_timestamp'], unit='s').dt.date
        daily_clicks = click_df.groupby('date').size()
        
        axes[1, 1].plot(daily_clicks.index, daily_clicks.values, marker='o', linewidth=2)
        axes[1, 1].set_xlabel('日期', fontsize=12)
        axes[1, 1].set_ylabel('点击次数', fontsize=12)
        axes[1, 1].set_title('每日点击趋势', fontsize=14, fontweight='bold')
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 用户行为可视化已保存: {save_path}")
        plt.close()
    
    @staticmethod
    def plot_item_similarity_matrix(item_sim_dict: dict, top_n: int = 20,
                                   save_path: str = 'similarity_matrix.png'):
        """可视化物品相似度矩阵（Top-N物品）"""
        # 选择最热门的N个物品
        all_items = list(item_sim_dict.keys())[:top_n]
        
        # 构建相似度矩阵
        sim_matrix = np.zeros((len(all_items), len(all_items)))
        item_to_idx = {item: idx for idx, item in enumerate(all_items)}
        
        for i, item_i in enumerate(all_items):
            if item_i in item_sim_dict:
                for item_j, sim in item_sim_dict[item_i]:
                    if item_j in item_to_idx:
                        j = item_to_idx[item_j]
                        sim_matrix[i, j] = sim
        
        # 绘制热力图
        plt.figure(figsize=(12, 10))
        sns.heatmap(sim_matrix, cmap='YlOrRd', square=True, 
                   xticklabels=all_items, yticklabels=all_items,
                   cbar_kws={'label': '相似度'})
        plt.title(f'物品相似度矩阵 (Top-{top_n})', fontsize=16, fontweight='bold')
        plt.xlabel('物品ID', fontsize=12)
        plt.ylabel('物品ID', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 相似度矩阵已保存: {save_path}")
        plt.close()
    
    @staticmethod
    def plot_recommendation_diversity(predictions: dict, articles_df: pd.DataFrame = None,
                                     save_path: str = 'diversity_analysis.png'):
        """分析推荐结果的多样性"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 1. 推荐物品频率分布
        all_recommended = []
        for items in predictions.values():
            all_recommended.extend(items[:10])  # Top-10
        
        item_freq = Counter(all_recommended)
        top_items = item_freq.most_common(50)
        
        axes[0].barh(range(len(top_items)), [count for _, count in top_items])
        axes[0].set_yticks(range(len(top_items)))
        axes[0].set_yticklabels([f'Item {item}' for item, _ in top_items], fontsize=8)
        axes[0].set_xlabel('推荐次数', fontsize=12)
        axes[0].set_title('Top-50 最常被推荐的物品', fontsize=14, fontweight='bold')
        axes[0].invert_yaxis()
        
        # 2. 用户推荐列表的唯一物品数分布
        unique_counts = [len(set(items[:10])) for items in predictions.values()]
        axes[1].hist(unique_counts, bins=range(1, 12), edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Top-10中的唯一物品数', fontsize=12)
        axes[1].set_ylabel('用户数', fontsize=12)
        axes[1].set_title('推荐列表多样性分布', fontsize=14, fontweight='bold')
        axes[1].axvline(np.mean(unique_counts), color='red', linestyle='--',
                       label=f'平均: {np.mean(unique_counts):.2f}')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 多样性分析已保存: {save_path}")
        plt.close()
    
    @staticmethod
    def plot_strategy_comparison(results: list, save_path: str = 'strategy_comparison.png'):
        """对比不同策略的雷达图"""
        strategies = [r['strategy'] for r in results]
        metrics = ['recall@5', 'ndcg@5', 'precision@5', 'coverage']
        
        # 归一化指标到0-1
        df = pd.DataFrame(results)
        df_norm = df[metrics].copy()
        for col in metrics:
            max_val = df_norm[col].max()
            if max_val > 0:
                df_norm[col] = df_norm[col] / max_val
        
        # 雷达图
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(strategies)))
        
        for idx, strategy in enumerate(strategies):
            values = df_norm.iloc[idx][metrics].tolist()
            values += values[:1]  # 闭合
            
            ax.plot(angles, values, 'o-', linewidth=2, label=strategy, color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(['Recall@5', 'NDCG@5', 'Precision@5', 'Coverage'], 
                          fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_title('策略性能雷达对比 (归一化)', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 策略对比雷达图已保存: {save_path}")
        plt.close()
    
    @staticmethod
    def plot_user_item_graph(click_df: pd.DataFrame, sample_users: int = 10,
                            save_path: str = 'user_item_graph.png'):
        """可视化用户-物品交互图（采样）"""
        # 采样用户
        sampled = click_df.groupby('user_id').head(5)  # 每个用户最多5条
        user_samples = sampled['user_id'].value_counts().head(sample_users).index
        subgraph_data = sampled[sampled['user_id'].isin(user_samples)]
        
        # 构建图
        G = nx.Graph()
        
        # 添加节点和边
        for _, row in subgraph_data.iterrows():
            user = f"U{row['user_id']}"
            item = f"I{row['click_article_id']}"
            G.add_edge(user, item)
        
        # 布局
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # 绘图
        plt.figure(figsize=(14, 10))
        
        # 分离用户和物品节点
        users = [n for n in G.nodes() if n.startswith('U')]
        items = [n for n in G.nodes() if n.startswith('I')]
        
        nx.draw_networkx_nodes(G, pos, nodelist=users, node_color='lightblue', 
                              node_size=500, label='用户')
        nx.draw_networkx_nodes(G, pos, nodelist=items, node_color='lightcoral',
                              node_size=300, label='物品')
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        plt.title(f'用户-物品交互图 (采样{sample_users}个用户)', 
                 fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 用户-物品交互图已保存: {save_path}")
        plt.close()


# ==================== 使用示例 ====================
def main():
    """主函数 - 生成所有可视化"""
    
    # 加载数据
    print("加载数据...")
    clicks = pd.read_csv('../data/train_click_log.csv')
    
    visualizer = RecallVisualizer()
    
    # 1. 用户行为分布
    print("\n生成用户行为可视化...")
    visualizer.plot_user_behavior_distribution(clicks, 'viz_user_behavior.png')
    
    # 2. 用户-物品交互图
    print("\n生成用户-物品交互图...")
    visualizer.plot_user_item_graph(clicks, sample_users=15, save_path='viz_user_item_graph.png')
    
    print("\n✓ 所有可视化生成完成！")
    print("\n生成的文件:")
    print("  - viz_user_behavior.png")
    print("  - viz_user_item_graph.png")


if __name__ == '__main__':
    main()
