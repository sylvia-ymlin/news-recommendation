# Cell 1
from pathlib import Path

base_path = Path.cwd()

data_path = str(base_path / 'data') + '/'

save_path = str(base_path / 'temp_results') + '/'

Path(data_path).mkdir(parents=True, exist_ok=True)

Path(save_path).mkdir(parents=True, exist_ok=True)

# Cell 5
import subprocess

import sys



subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'faiss-cpu', 'tqdm'])



import os, math, warnings, math, pickle, random

from pathlib import Path

from datetime import datetime

from collections import defaultdict

import logging

warnings.filterwarnings('ignore')

os.environ['OMP_NUM_THREADS'] = '1'

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)



import faiss

import pandas as pd

import numpy as np

from tqdm import tqdm





from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import LabelEncoder

# Cell 6
from pathlib import Path

base_path = Path.cwd()

data_path = str(base_path / 'data') + '/'

save_path = str(base_path / 'temp_results') + '/'

Path(data_path).mkdir(parents=True, exist_ok=True)

Path(save_path).mkdir(parents=True, exist_ok=True)

# 做召回评估的一个标志, 如果不进行评估就是直接使用全量数据进行召回

metric_recall = False

# Cell 8
# debug mode
def get_all_click_sample(data_path, sample_nums=10000):
    all_click = pd.read_csv(data_path + 'train_click_log.csv')
    all_user_ids = all_click.user_id.unique() # 获得去重的 user_id

    # randomly choose, don't put back
    random.seed(42)
    sample_user_ids = np.random.choice(all_user_ids, size=sample_nums, replace=False)
    all_click = all_click[all_click['user_id'].isin(sample_user_ids)]

    # drop repeated records
    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))

    return all_click

# Cell 9
# online and off-line
def get_all_click_df(data_path, offline=True):
  if offline:
    all_click = pd.read_csv(data_path + 'train_click_log.csv')
  else:
    trn_click = pd.read_csv(data_path + 'train_click_log.csv')
    tst_click = pd.read_csv(data_path + 'testA_click_log.csv')

    all_click = pd.concat([trn_click, tst_click])

  # 去重
  all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))

  return all_click

# Cell 10
# load from article.csv
def get_item_info_df(data_path):
  item_info_df = pd.read_csv(data_path + 'articles.csv')

  # 为了方便与训练集中的 click_article_id 对应，需要把 article_id 修改
  item_info_df = item_info_df.rename(columns={'article_id': 'click_article_id'})

  return item_info_df

# Cell 11
# load embedding
def get_item_emb_dict(data_path, save_path):
  item_emb_df = pd.read_csv(data_path + 'articles_emb.csv')

  # 通过列名里包含 'emb' 来定位向量的每一维（如 emb_0, emb_1, ...）
  item_emb_cols = [x for x in item_emb_df.columns if 'emb' in x]
  # 转成连续内存的 NumPy 矩阵
  item_emb_np = np.ascontiguousarray(item_emb_df[item_emb_cols])

  # L2 归一化
  # 用每行的范数把该行除一下，使每个向量长度为 1，后续 内积 就是余弦相似度
  # 若用 FAISS 的 IndexFlatIP (内积作相似度)，必须先做这一步
  item_emb_np = item_emb_np / np.linalg.norm(item_emb_np, axis=1, keepdims=True)
  # 组装成字典：生成 {article_id → 向量} 的查找表
  item_emb_dict = dict(zip(item_emb_df['article_id'], item_emb_np))
  # 持久化到本地
  pickle.dump(item_emb_dict, open(save_path + 'item_content_emb.pkl', 'wb'))

  return item_emb_dict

# Cell 12
# 最小-最大归一化
max_min_scaler = lambda x: (x - np.min(x))/(np.max(x) - np.min(x))

# Cell 13
all_click_df = get_all_click_df(data_path, offline=False)
# 对时间戳归一化
all_click_df['click_timestamp'] = all_click_df[['click_timestamp']].apply(max_min_scaler)

# Cell 14
item_info_df = get_item_info_df(data_path)
# 对文章创建时间归一化
item_info_df['created_at_ts'] = item_info_df[['created_at_ts']].apply(max_min_scaler)

# Cell 15
item_emb_dict = get_item_emb_dict(data_path, save_path)

# Cell 17
# 获取 用户-文章-时间
#  {user1: [(item1, time1), (item2, time2)..]...}
# 用于基于关联规则的 用户协同过滤
def get_user_item_time(click_df):
  click_df = click_df.sort_values('click_timestamp')

  def make_item_time_pair(df):
    return list(zip(df['click_article_id'], df['click_timestamp']))

  # group by user_id, takeout 'click_article_id', 'click_timestamp'
  user_item_time_df = click_df.groupby('user_id')[['click_article_id', 'click_timestamp']] \
      .apply(lambda x: make_item_time_pair(x)).reset_index().rename(columns={0: 'item_time_list'})

  # 转换为字典
  user_item_time_dict = dict(zip(user_item_time_df['user_id'], user_item_time_df['item_time_list']))

  return user_item_time_dict

# Cell 18
# 获取 文章-用户-时间
# {item1: [(user1, time1), (user2, time2)...]...}
# 用于基于关联规则的 物品协同过滤
def get_item_user_time(click_df):
  click_df = click_df.sort_values('click_timestamp')

  def make_user_time_pair(df):
    return list(zip(df['user_id'], df['click_timestamp']))

  item_user_time_df = click_df.groupby('click_article_id')[['user_id', 'click_timestamp']] \
    .apply(lambda x: make_user_time_pair(x)).reset_index().rename(columns={0: 'user_time_list'})

  item_user_time_dict = dict(zip(item_user_time_df['click_article_id'], item_user_time_df['user_time_list']))

  return item_user_time_dict

# Cell 19
# 获取当前数据的历史点击和最后一次点击
# 用户评估召回结果
# 用于特征工程
# 用于制作标签，转成监督学习测试集
def get_hist_and_last_click(click_df):
  # 按用户排序，同一个用户按照时间升序
  all_click = click_df.sort_values(by=['user_id', 'click_timestamp'])
  # 获取每个用户的最后一次点击
  click_last_df = all_click.groupby('user_id').tail(1)

  # 用户最后一次点击要从日志中剔除，但是如果用户本身只有一次点击，为了避免数据缺失，则不剔除
  def hist_func(user_df):
    if len(user_df) == 1:
      return user_df
    else:
      return user_df[:-1]

  click_hist_df = all_click.groupby('user_id').apply(hist_func).reset_index(drop=True)

  return click_hist_df, click_last_df

  # 获取每个用户的历史点击
  # drop=True: 把当前索引丢弃，重新生成从 0 开始的连续行号索引
  click_hist_df = all_click.groupby('user_id').apply(hist_func).reset_index(drop=True)

  return click_hist_df, click_last_df


# Cell 20
# 获取文章属性
def get_item_info_dict(item_info_df):
  # 将文章和文章各个属性都做一个字典
  item_type_dict = dict(zip(item_info_df['click_article_id'], item_info_df['category_id']))
  item_words_dict = dict(zip(item_info_df['click_article_id'], item_info_df['words_count']))
  item_created_time_dict = dict(zip(item_info_df['click_article_id'], item_info_df['created_at_ts']))

  return item_type_dict, item_words_dict, item_created_time_dict

# Cell 21
# 获取用户历史点击文章的信息
def get_user_hist_item_info_dict(all_click):
  # 用户历史点击的文章的类型
  user_hist_item_typs = all_click.groupby('user_id')['category_id'].agg(set).reset_index()
  # 转为字典
  user_hist_item_typs_dict = dict(zip(user_hist_item_typs['user_id'], user_hist_item_typs['category_id']))

  # 文章的集合
  user_hist_item_ids_dict = all_click.groupby('user_id')['click_article_id'].agg(set).reset_index()
  user_hist_item_ids_dict = dict(zip(user_hist_item_ids_dict['user_id'], user_hist_item_ids_dict['click_article_id']))

  # 文章平均字数
  user_hist_item_words = all_click.groupby('user_id')['words_count'].agg('mean').reset_index()
  user_hist_item_words_dict = dict(zip(user_hist_item_words['user_id'], user_hist_item_words['words_count']))

  # 最后一次点击的文章的创建时间
  all_click_ = all_click.sort_values('click_timestamp')
  user_last_item_created_time = all_click_.groupby('user_id')['created_at_ts'].apply(lambda x: x.iloc[-1]).reset_index()
  user_last_item_created_time_dict = dict(zip(user_last_item_created_time['user_id'], \
                                                user_last_item_created_time['created_at_ts']))

  return user_hist_item_typs_dict, user_hist_item_ids_dict, user_hist_item_words_dict, user_last_item_created_time_dict


# Cell 22
# 点击次数最多的topk个文章
def get_item_topk_click(click_df, k):
  topk_click = click_df['click_article_id'].value_counts().index[:k]
  return topk_click

# Cell 24
# 这里实现了五个召回通道
user_multi_recall_dict =  {'itemcf_sim_itemcf_recall': {},
                           'embedding_sim_item_recall': {},
                           'youtubednn_recall': {},
                           'youtubednn_usercf_recall': {},
                           'cold_start_recall': {}}

# Cell 25
item_type_dict, item_words_dict, item_created_time_dict = get_item_info_dict(item_info_df)

# Cell 26
# 提取数据，offline, 最后一次点击用作召回评估
trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)

# Cell 28
# 依次评估召回中前 10， 20， 30， 40， 50 个文章中的击中率
# 越靠前，说明召回的效果越好，不是一定要控制召回的数量，但是如果多篇召回都没有命中，召回算法是有问题的
def metrics_recall(user_recall_items_dict, trn_last_click_df, topk=5):
  # 需要 (user_id, click_article_id)
  last_click_item_dict = dict(zip(trn_last_click_df['user_id'], trn_last_click_df['click_article_id']))

  user_num = len(user_recall_items_dict)

  for k in range(10, topk+1, 10):
    hit_num = 0
    for user, item_list in user_recall_items_dict.items():
      # 获取前 k 召回结果
      tmp_recall_items = [x[0] for x in user_recall_items_dict[user][:k]]
      if last_click_item_dict[user] in set(tmp_recall_items):
        hit_num += 1

  # 有多少用户命中
  hit_rate = round(hit_num * 100.0 / user_num, 2)
  logger.info('topk: {}, hit_num: {}, hit_rate: {}'.format(k, hit_num, hit_rate))
  return hit_rate # 最后返回的是 传入参数 topk 的命中率

# Cell 31
def itemcf_sim(df, item_created_time_dict):
  # item_created_time_dict: 其实就是索引
  user_item_time_dict = get_user_item_time(df)

  # 存放物品相似度
  i2i_sim = {}
  # 统计物品被点击次数
  item_cnt = defaultdict(int)
  for user, item_time_list in tqdm(user_item_time_dict.items()):
    # 考虑时间因素 [loc1 其实是 index]
    for loc1, (item1, time1) in enumerate(item_time_list):
      item_cnt[item1] += 1
      i2i_sim.setdefault(item1, {})
      for loc2, (item2, time2) in enumerate(item_time_list):
        if item1 == item2:
          continue
        # 文章点击顺序: 顺序偏好，sim(i, j) != sim(j, i)
        loc_alpha = 1.0 if loc1 > loc2 else 0.7
        # 位置权重: 相邻（间距=1）权重最大 1，间距越远越小
        loc_weight = loc_alpha * (0.9 ** (np.abs(loc1 - loc2) - 1))
        # 时间权重
        click_time_weight = np.exp(0.7 ** np.abs(time1 - time2))
        # 文章创建时间权重
        created_time_weight = np.exp(0.7 ** np.abs(item_created_time_dict[item1] - item_created_time_dict[item2]))

        # item1 -> item2
        i2i_sim[item1].setdefault(item2, 0)
        # 融合，然后做归一化
        i2i_sim[item1][item2] += loc_weight * click_time_weight * created_time_weight / math.log(len(item_time_list) + 1)

  i2i_sim_ = i2i_sim.copy()
  for i, related_items in i2i_sim.items():
      for j, wij in related_items.items():
          i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])

  # 将得到的相似性矩阵保存到本地
  pickle.dump(i2i_sim_, open(save_path + 'itemcf_i2i_sim.pkl', 'wb'))

  return i2i_sim_

i2i_sim = itemcf_sim(all_click_df, item_created_time_dict)

# Cell 33
def get_user_activate_degree_dict(all_click_df):
  cnt = all_click_df.groupby('user_id')['click_article_id'].count().reset_index(name='click_cnt')
  mm = MinMaxScaler()
  cnt['activity_score'] = mm.fit_transform(cnt[['click_cnt']])
  user_activate_degree_dict = dict(zip(cnt['user_id'], cnt['activity_score']))

  return user_activate_degree_dict


# Cell 34
def usercf_sim(all_click_df, user_activate_degree_dict):
  item_user_time_dict = get_item_user_time(all_click_df)

  u2u_sim = {}
  user_cnt = defaultdict(int)
  for item, user_time_list in tqdm(item_user_time_dict.items()):
    for u, click_time in user_time_list:
      user_cnt[u] += 1
      u2u_sim.setdefault(u, {})

      for v, click_time in user_time_list:
        u2u_sim[u].setdefault(v, 0)
        if u == v:
          continue

        # 用户活跃度差值有关
        activate_weight = 100 * 0.5 * (user_activate_degree_dict[u] + user_activate_degree_dict[v])
        u2u_sim[u][v] += activate_weight / math.log(len(user_time_list) + 1)

  u2u_sim_ = u2u_sim.copy()
  for u, related_users in u2u_sim.items():
    for v, wij in related_users.items():
      u2u_sim_[u][v] = wij / math.sqrt(user_cnt[u] * user_cnt[v])

  # save
  pickle.dump(u2u_sim_, open(save_path + 'usercf_u2u_sim.pkl', 'wb'))

  return u2u_sim_

sample = get_all_click_sample(data_path)
user_activate_degree_dict = get_user_activate_degree_dict(sample)
u2u_sim = usercf_sim(sample, user_activate_degree_dict)

# Cell 36
def embedding_sim(click_df, item_emb_dict, save_path, topk):
  # click_df: 计算哪些物品的相似度
  # item_emb_df：物品的 embedding
  # topk: 每个物品计算 topk 相似

  # 建立文章索引和文章id的字典映射
  ids = np.array(list(item_emb_dict.keys()))
  item_emb_dict = np.ascontiguousarray(np.stack([item_emb_dict[i] for i in ids]).astype(np.float32))
  item_idx_2_rawid_dict = dict(enumerate(ids))

  # 建立faiss索引
  item_index = faiss.IndexFlatIP(item_emb_dict.shape[1])
  item_index.add(item_emb_dict)

  # 相似度查询，给每个索引位置上的向量返回topk个item以及相似度
  sim, idx = item_index.search(item_emb_dict, topk) # 返回的是列表

  # 将向量检索的结果，保存为原始 id 的对应关系
  item_sim_dict = collections.defaultdict(dict)
  for target_idx, sim_value_list, rele_idx_list in tqdm(zip(range(len(item_emb_dict)), sim, idx)):
    target_raw_id = item_idx_2_rawid_dict[target_idx]
    # 从1开始是为了去掉商品本身, 所以最终获得的相似商品只有topk-1
    for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):
        rele_raw_id = item_idx_2_rawid_dict[rele_idx]
        item_sim_dict[target_raw_id][rele_raw_id] = item_sim_dict.get(target_raw_id, {}).get(rele_raw_id, 0) + sim_value

  pickle.dump(item_sim_dict, open(save_path + 'emb_i2i_sim.pkl', 'wb'))

  return item_sim_dict

# Cell 37
emb_i2i_sim = embedding_sim(all_click_df, item_emb_dict, save_path, topk=10) # topk可以自行设置

# Cell 40
# 获取双塔召回时的训练验证数据
# negsample指的是通过滑窗构建样本的时候，负样本的数量
def gen_data_set(data, negsample=0):
    # 按时间/点击顺序排序
    data.sort_values("click_timestamp", inplace=True)
    # 获取所有文章 id
    item_ids = data['click_article_id'].unique()

    # 存放训练集
    train_set = []
    # 存放测试集
    test_set = []
    # 按用户分组，逐个用户处理
    for reviewerID, hist in tqdm(data.groupby('user_id'), disable=not logger.isEnabledFor(logging.DEBUG)):
        # 正样本：用户点击过的文章
        pos_list = hist['click_article_id'].tolist()

        # 需要做负采样
        if negsample > 0:
            # 用户没有点击过的文章列表
            candidate_set = list(set(item_ids) - set(pos_list))
            # 从用户没有点击过的文章里抽选负样本，根据正样本的数量，抽取标准是每个正样本对应 negsample 个负样本
            neg_list = np.random.choice(candidate_set,size=len(pos_list)*negsample,replace=True)  # 对于每个正样本，选择n个负样本

        # 如果用户只点击过一篇文章，则该文章需要也放入训练集，否则会有信息缺失
        if len(pos_list) == 1:
            train_set.append((reviewerID, [pos_list[0]], pos_list[0],1,len(pos_list)))
            test_set.append((reviewerID, [pos_list[0]], pos_list[0],1,len(pos_list)))

        # 滑窗构造正负样本，点击次数大于 1；因为用户点击是一个时间序列，从用户点击的序列中，构建多个正样本
        '''
          用户点击 [A, B, C, D]
          不滑窗: 历史=[A,B,C] → 预测 D
          滑窗：
            历史=[A]   → 预测 B
            历史=[A,B] → 预测 C
            历史=[A,B,C] → 预测 D
        '''
        for i in range(1, len(pos_list)):
            # hist: 历史，前 i 篇
            # pos_item：预测目标，第 i 篇

            hist = pos_list[:i]

            # 不是最后一次点击
            if i != len(pos_list) - 1:
                # 添加正样本
                train_set.append((reviewerID, hist[::-1], pos_list[i], 1, len(hist[::-1])))  # 正样本 [user_id, his_item, pos_item, label, len(his_item)]
                # 添加 n 个负样本
                for negi in range(negsample):
                    train_set.append((reviewerID, hist[::-1], neg_list[i*negsample+negi], 0,len(hist[::-1]))) # 负样本 [user_id, his_item, neg_item, label, len(his_item)]
            # 最后一次点击，测试集
            else:
                test_set.append((reviewerID, hist[::-1], pos_list[i],1,len(hist[::-1])))

    # 打乱样本顺序
    random.shuffle(train_set)
    random.shuffle(test_set)

    return train_set, test_set

# 将输入的数据进行padding，使得序列特征的长度都一致
def gen_model_input(train_set,user_profile,seq_max_len):

    train_uid = np.array([line[0] for line in train_set])
    train_seq = [line[1] for line in train_set]
    train_iid = np.array([line[2] for line in train_set])
    train_label = np.array([line[3] for line in train_set])
    train_hist_len = np.array([line[4] for line in train_set])

    # 因为每个用户历史点击数不一样，需要统一长度，短的补0，长的截断。
    train_seq_pad = pad_sequences(train_seq, maxlen=seq_max_len, padding='post', truncating='post', value=0)
    # 组装为字典
    train_model_input = {"user_id": train_uid, "click_article_id": train_iid, "hist_article_id": train_seq_pad,
                         "hist_len": train_hist_len}
    # 返回输入特征，对应标签
    return train_model_input, train_label

# Cell 42
# funrec youtubeDNN召回
def youtubednn_u2i_dict(data, topk=20):
    """
    使用 FunRec 的 YouTubeDNN 两塔模型进行召回，保持与当前逻辑一致的预处理：
    - 标签/目标为正样本采样（sampled softmax 内部使用 item_id 作为 label）
    - 通过滑窗构造训练/测试样本，使用最近序列作为测试
    - 历史序列长度固定为 SEQ_LEN，并做 post-padding
    - 训练完成后提取 user/item embedding，使用 FAISS 基于内积做 TopK 近邻召回
    - 返回 {user_raw_id: [(item_raw_id, score), ...]} 的召回结果字典
    """
    import sys
    import numpy as np
    import pickle
    from tqdm import tqdm
    from sklearn.preprocessing import LabelEncoder

    from funrec.features.feature_column import FeatureColumn
    from funrec.training.trainer import train_model

    # 内联配置（参考 config_youtubednn.yaml，并适配当前数据列名）
    SEQ_LEN = 30
    emb_dim = 16
    neg_sample = 20
    dnn_units = [32]
    label_name = 'click_article_id'

    # 拷贝并做类别编码（与现有逻辑保持一致）
    df = data.copy()
    user_profile_raw = df[["user_id"]].drop_duplicates('user_id')
    item_profile_raw = df[["click_article_id"]].drop_duplicates('click_article_id')

    encoders = {}
    feature_max_idx = {}
    for col in ["user_id", "click_article_id"]:
        lbe = LabelEncoder()
        df[col] = lbe.fit_transform(df[col])
        encoders[col] = lbe
        feature_max_idx[col] = int(df[col].max()) + 1

    # 画像（仅用于 id 回退映射）
    user_profile = df[["user_id"]].drop_duplicates('user_id')
    item_profile = df[["click_article_id"]].drop_duplicates('click_article_id')
    user_index_2_rawid = dict(zip(user_profile['user_id'], user_profile_raw['user_id']))
    item_index_2_rawid = dict(zip(item_profile['click_article_id'], item_profile_raw['click_article_id']))

    # 按当前逻辑构造训练/测试样本
    train_set, test_set = gen_data_set(df, 0)
    train_model_input, _ = gen_model_input(train_set, user_profile, SEQ_LEN)
    test_model_input, _ = gen_model_input(test_set, user_profile, SEQ_LEN)

    # 仅保留模型实际需要的输入键
    input_keys = ['user_id', 'click_article_id', 'hist_article_id']
    train_X = {k: np.asarray(train_model_input[k], dtype=np.int32) for k in input_keys}
    test_X = {k: np.asarray(test_model_input[k], dtype=np.int32) for k in input_keys}

    # 手动定义特征列（不依赖外部数据字典）
    feature_columns = [
        FeatureColumn(name='user_id', group=['user_dnn'], type='sparse', vocab_size=feature_max_idx['user_id'], emb_dim=emb_dim),
        FeatureColumn(name='click_article_id', group=['target_item'], type='sparse', vocab_size=feature_max_idx['click_article_id'], emb_dim=emb_dim),
        FeatureColumn(name='hist_article_id', emb_name='click_article_id', group=['raw_hist_seq'], type='varlen_sparse', max_len=SEQ_LEN, combiner='mean', emb_dim=emb_dim, vocab_size=feature_max_idx['click_article_id']),
    ]

    # 组装 processed_data（与 FunRec 训练器期望的结构一致）
    processed_data = {
        'train': {
            'features': train_X,
            'labels': None  # 由 positive_sampling_labels 规则内部替换为全 1
        },
        'test': {
            'features': test_X,
            'labels': None,
            'eval_data': {}
        },
        'all_items': {
            'click_article_id': np.arange(feature_max_idx['click_article_id'], dtype=np.int32)
        },
        'feature_dict': {
            'user_id': feature_max_idx['user_id'],
            'click_article_id': feature_max_idx['click_article_id']
        }
    }

    # 训练配置（内联）
    training_config = {
        'build_function': 'funrec.models.youtubednn.build_youtubednn_model',
        'data_preprocessing': [
            {'type': 'positive_sampling_labels'}
        ],
        'model_params': {
            'emb_dim': emb_dim,
            'neg_sample': neg_sample,
            'dnn_units': dnn_units,
            'label_name': label_name
        },
        'optimizer': 'adam',
        'optimizer_params': {
            'learning_rate': 1e-4
        },
        'loss': 'sampledsoftmaxloss',
        'batch_size': 128,
        'epochs': 5,
        'verbose': 0
    }

    # 训练模型（返回 main_model, user_model, item_model）
    model, user_model, item_model = train_model(training_config, feature_columns, processed_data)

    # 提取 embedding
    user_inputs_for_pred = {k: test_X[k] for k in ['user_id', 'hist_article_id']}
    user_embs = user_model.predict(user_inputs_for_pred, batch_size=2 ** 12, verbose=0)
    item_embs = item_model.predict(processed_data['all_items'], batch_size=2 ** 12, verbose=0)

    # 归一化（与现有逻辑一致）
    user_embs = user_embs / np.linalg.norm(user_embs, axis=1, keepdims=True)
    item_embs = item_embs / np.linalg.norm(item_embs, axis=1, keepdims=True)

    # 保存 embedding（与现有逻辑一致，注意 id 回退）
    raw_user_id_emb_dict = {user_index_2_rawid[k]: v for k, v in zip(test_X['user_id'], user_embs)}
    raw_item_id_emb_dict = {item_index_2_rawid[k]: v for k, v in zip(processed_data['all_items']['click_article_id'], item_embs)}
    pickle.dump(raw_user_id_emb_dict, open(save_path / 'user_youtube_emb.pkl', 'wb'))
    pickle.dump(raw_item_id_emb_dict, open(save_path / 'item_youtube_emb.pkl', 'wb'))

    # 使用 FAISS 做向量检索召回
    index = faiss.IndexFlatIP(emb_dim)
    index.add(item_embs.astype(np.float32))
    sim, idx = index.search(np.ascontiguousarray(user_embs.astype(np.float32)), topk)

    user_recall_items_dict = defaultdict(dict)
    for target_idx, sim_value_list, rele_idx_list in tqdm(zip(test_X['user_id'], sim, idx), disable=not logger.isEnabledFor(logging.DEBUG)):
        target_raw_id = user_index_2_rawid[int(target_idx)]
        # 从 1 开始去掉最相似的第一个（通常为本身或极近邻）
        for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):
            rele_raw_id = item_index_2_rawid[int(rele_idx)]
            user_recall_items_dict[target_raw_id][rele_raw_id] = user_recall_items_dict.get(target_raw_id, {}).get(rele_raw_id, 0) + float(sim_value)

    # 排序并保存
    user_recall_items_dict = {k: sorted(v.items(), key=lambda x: x[1], reverse=True) for k, v in user_recall_items_dict.items()}
    pickle.dump(user_recall_items_dict, open(save_path / 'youtube_u2i_dict.pkl', 'wb'))
    return user_recall_items_dict

# Cell 43
# 由于这里需要做召回评估，所以讲训练集中的最后一次点击都提取了出来
if not metric_recall:
    user_multi_recall_dict['youtubednn_recall'] = youtubednn_u2i_dict(all_click_df, topk=20)
else:
    trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
    user_multi_recall_dict['youtubednn_recall'] = youtubednn_u2i_dict(trn_hist_click_df, topk=20)
    # 召回效果评估
    metrics_recall(user_multi_recall_dict['youtubednn_recall'], trn_last_click_df, topk=20)

# Cell 45
# 基于商品的召回i2i
def item_based_recommend(user_id, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num, item_topk_click, item_created_time_dict, emb_i2i_sim):
    """
        基于文章协同过滤的召回
        :param user_id: 用户id
        :param user_item_time_dict: 字典, 根据点击时间获取用户的点击文章序列   {user1: {item1: time1, item2: time2..}...}
        :param i2i_sim: 字典，文章相似性矩阵
        :param sim_item_topk: 整数， 选择与当前文章最相似的前k篇文章
        :param recall_item_num: 整数， 最后的召回文章数量
        :param item_topk_click: 列表，点击次数最多的文章列表，用户召回补全
        :param emb_i2i_sim: 字典基于内容embedding算的文章相似矩阵

        return: 召回的文章列表 {item1:score1, item2: score2...}

    """
    # 获取用户历史交互的文章
    user_hist_items = user_item_time_dict[user_id]

    item_rank = {}
    for loc, (i, click_time) in enumerate(user_hist_items):
        for j, wij in sorted(i2i_sim[i].items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]:
            if j in user_hist_items:
                continue

            # 文章创建时间差权重
            created_time_weight = np.exp(0.8 ** np.abs(item_created_time_dict[i] - item_created_time_dict[j]))
            # 相似文章和历史点击文章序列中历史文章所在的位置权重
            loc_weight = (0.9 ** (len(user_hist_items) - loc))

            content_weight = 1.0
            if emb_i2i_sim.get(i, {}).get(j, None) is not None:
                content_weight += emb_i2i_sim[i][j]
            if emb_i2i_sim.get(j, {}).get(i, None) is not None:
                content_weight += emb_i2i_sim[j][i]

            item_rank.setdefault(j, 0)
            item_rank[j] += created_time_weight * loc_weight * content_weight * wij

    # 不足10个，用热门商品补全
    if len(item_rank) < recall_item_num:
        for i, item in enumerate(item_topk_click):
            if item in item_rank.items(): # 填充的item应该不在原来的列表中
                continue
            item_rank[item] = - i - 100 # 随便给个负数就行
            if len(item_rank) == recall_item_num:
                break

    item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]

    return item_rank

# Cell 46
# 先进行itemcf召回, 为了召回评估，所以提取最后一次点击

if metric_recall:
    trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
else:
    trn_hist_click_df = all_click_df

user_recall_items_dict = defaultdict(dict)
user_item_time_dict = get_user_item_time(trn_hist_click_df)

i2i_sim = pickle.load(open(save_path / 'itemcf_i2i_sim.pkl', 'rb'))
emb_i2i_sim = pickle.load(open(save_path / 'emb_i2i_sim.pkl', 'rb'))

sim_item_topk = 20
recall_item_num = 10
item_topk_click = get_item_topk_click(trn_hist_click_df, k=50)

for user in tqdm(trn_hist_click_df['user_id'].unique(), disable=not logger.isEnabledFor(logging.DEBUG)):
    user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict, \
                                                        i2i_sim, sim_item_topk, recall_item_num, \
                                                        item_topk_click, item_created_time_dict, emb_i2i_sim)

user_multi_recall_dict['itemcf_sim_itemcf_recall'] = user_recall_items_dict
pickle.dump(user_multi_recall_dict['itemcf_sim_itemcf_recall'], open(save_path / 'itemcf_recall_dict.pkl', 'wb'))

if metric_recall:
    # 召回效果评估
    metrics_recall(user_multi_recall_dict['itemcf_sim_itemcf_recall'], trn_last_click_df, topk=recall_item_num)

# Cell 48
# 这里是为了召回评估，所以提取最后一次点击
if metric_recall:
    trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
else:
    trn_hist_click_df = all_click_df

user_recall_items_dict = defaultdict(dict)
user_item_time_dict = get_user_item_time(trn_hist_click_df)
i2i_sim = pickle.load(open(save_path / 'emb_i2i_sim.pkl','rb'))

sim_item_topk = 20
recall_item_num = 10

item_topk_click = get_item_topk_click(trn_hist_click_df, k=50)

for user in tqdm(trn_hist_click_df['user_id'].unique(), disable=not logger.isEnabledFor(logging.DEBUG)):
    user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict, i2i_sim, sim_item_topk,
                                                        recall_item_num, item_topk_click, item_created_time_dict, emb_i2i_sim)

user_multi_recall_dict['embedding_sim_item_recall'] = user_recall_items_dict
pickle.dump(user_multi_recall_dict['embedding_sim_item_recall'], open(save_path / 'embedding_sim_item_recall.pkl', 'wb'))

if metric_recall:
    # 召回效果评估
    metrics_recall(user_multi_recall_dict['embedding_sim_item_recall'], trn_last_click_df, topk=recall_item_num)

# Cell 50
# 基于用户的召回 u2u2i
def user_based_recommend(user_id, user_item_time_dict, u2u_sim, sim_user_topk, recall_item_num,
                         item_topk_click, item_created_time_dict, emb_i2i_sim):
    """
        基于文章协同过滤的召回
        :param user_id: 用户id
        :param user_item_time_dict: 字典, 根据点击时间获取用户的点击文章序列   {user1: {item1: time1, item2: time2..}...}
        :param u2u_sim: 字典，文章相似性矩阵
        :param sim_user_topk: 整数， 选择与当前用户最相似的前k个用户
        :param recall_item_num: 整数， 最后的召回文章数量
        :param item_topk_click: 列表，点击次数最多的文章列表，用户召回补全
        :param item_created_time_dict: 文章创建时间列表
        :param emb_i2i_sim: 字典基于内容embedding算的文章相似矩阵

        return: 召回的文章列表 {item1:score1, item2: score2...}
    """
    # 历史交互
    user_item_time_list = user_item_time_dict[user_id]    # {item1: time1, item2: time2...}
    user_hist_items = set([i for i, t in user_item_time_list])   # 存在一个用户与某篇文章的多次交互， 这里得去重

    items_rank = {}
    for sim_u, wuv in sorted(u2u_sim[user_id].items(), key=lambda x: x[1], reverse=True)[:sim_user_topk]:
        for i, click_time in user_item_time_dict[sim_u]:
            if i in user_hist_items:
                continue
            items_rank.setdefault(i, 0)

            loc_weight = 1.0
            content_weight = 1.0
            created_time_weight = 1.0

            # 当前文章与该用户看的历史文章进行一个权重交互
            for loc, (j, click_time) in enumerate(user_item_time_list):
                # 点击时的相对位置权重
                loc_weight += 0.9 ** (len(user_item_time_list) - loc)
                # 内容相似性权重
                if emb_i2i_sim.get(i, {}).get(j, None) is not None:
                    content_weight += emb_i2i_sim[i][j]
                if emb_i2i_sim.get(j, {}).get(i, None) is not None:
                    content_weight += emb_i2i_sim[j][i]

                # 创建时间差权重
                created_time_weight += np.exp(0.8 * np.abs(item_created_time_dict[i] - item_created_time_dict[j]))

            items_rank[i] += loc_weight * content_weight * created_time_weight * wuv

    # 热度补全
    if len(items_rank) < recall_item_num:
        for i, item in enumerate(item_topk_click):
            if item in items_rank.items(): # 填充的item应该不在原来的列表中
                continue
            items_rank[item] = - i - 100 # 随便给个复数就行
            if len(items_rank) == recall_item_num:
                break

    items_rank = sorted(items_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]

    return items_rank

# Cell 51
# 实际召回

# 这里是为了召回评估，所以提取最后一次点击
# 由于usercf中计算user之间的相似度的过程太费内存了，全量数据这里就没有跑，跑了一个采样之后的数据
if metric_recall:
    trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
else:
    trn_hist_click_df = all_click_df

user_recall_items_dict = defaultdict(dict)
user_item_time_dict = get_user_item_time(trn_hist_click_df)

u2u_sim = pickle.load(open(save_path / 'usercf_u2u_sim.pkl', 'rb'))

sim_user_topk = 20
recall_item_num = 10
item_topk_click = get_item_topk_click(trn_hist_click_df, k=50)

for user in tqdm(trn_hist_click_df['user_id'].unique(), disable=not logger.isEnabledFor(logging.DEBUG)):
    user_recall_items_dict[user] = user_based_recommend(user, user_item_time_dict, u2u_sim, sim_user_topk, \
                                                        recall_item_num, item_topk_click, item_created_time_dict, emb_i2i_sim)

pickle.dump(user_recall_items_dict, open(save_path / 'usercf_u2u2i_recall.pkl', 'wb'))

if metric_recall:
    # 召回效果评估
    metrics_recall(user_recall_items_dict, trn_last_click_df, topk=recall_item_num)

# Cell 53
# 使用Embedding的方式获取u2u的相似性矩阵
# topk指的是每个user, faiss搜索后返回最相似的topk个user
def u2u_embdding_sim(click_df, user_emb_dict, save_path, topk):

    user_list = []
    user_emb_list = []
    for user_id, user_emb in user_emb_dict.items():
        user_list.append(user_id)
        user_emb_list.append(user_emb)

    user_index_2_rawid_dict = {k: v for k, v in zip(range(len(user_list)), user_list)}

    user_emb_np = np.array(user_emb_list, dtype=np.float32)

    # 建立faiss索引
    user_index = faiss.IndexFlatIP(user_emb_np.shape[1])
    user_index.add(user_emb_np)
    # 相似度查询，给每个索引位置上的向量返回topk个item以及相似度
    sim, idx = user_index.search(user_emb_np, topk) # 返回的是列表

    # 将向量检索的结果保存成原始id的对应关系
    user_sim_dict = defaultdict(dict)
    for target_idx, sim_value_list, rele_idx_list in tqdm(zip(range(len(user_emb_np)), sim, idx), disable=not logger.isEnabledFor(logging.DEBUG)):
        target_raw_id = user_index_2_rawid_dict[target_idx]
        # 从1开始是为了去掉商品本身, 所以最终获得的相似商品只有topk-1
        for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):
            rele_raw_id = user_index_2_rawid_dict[rele_idx]
            user_sim_dict[target_raw_id][rele_raw_id] = user_sim_dict.get(target_raw_id, {}).get(rele_raw_id, 0) + sim_value

    # 保存i2i相似度矩阵
    pickle.dump(user_sim_dict, open(save_path / 'youtube_u2u_sim.pkl', 'wb'))
    return user_sim_dict

# Cell 54
# 读取YoutubeDNN过程中产生的user embedding, 然后使用faiss计算用户之间的相似度
# 这里需要注意，这里得到的user embedding其实并不是很好，因为YoutubeDNN中使用的是用户点击序列来训练的user embedding,
# 如果序列普遍都比较短的话，其实效果并不是很好
user_emb_dict = pickle.load(open(save_path / 'user_youtube_emb.pkl', 'rb'))
u2u_sim = u2u_embdding_sim(all_click_df, user_emb_dict, save_path, topk=10)

# Cell 55
# 实际召回
# 使用召回评估函数验证当前召回方式的效果
if metric_recall:
    trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
else:
    trn_hist_click_df = all_click_df

user_recall_items_dict = defaultdict(dict)
user_item_time_dict = get_user_item_time(trn_hist_click_df)
u2u_sim = pickle.load(open(save_path / 'youtube_u2u_sim.pkl', 'rb'))

sim_user_topk = 20
recall_item_num = 10

item_topk_click = get_item_topk_click(trn_hist_click_df, k=50)
for user in tqdm(trn_hist_click_df['user_id'].unique(), disable=not logger.isEnabledFor(logging.DEBUG)):
    user_recall_items_dict[user] = user_based_recommend(user, user_item_time_dict, u2u_sim, sim_user_topk, \
                                                        recall_item_num, item_topk_click, item_created_time_dict, emb_i2i_sim)

user_multi_recall_dict['youtubednn_usercf_recall'] = user_recall_items_dict
pickle.dump(user_multi_recall_dict['youtubednn_usercf_recall'], open(save_path / 'youtubednn_usercf_recall.pkl', 'wb'))

if metric_recall:
    # 召回效果评估
    metrics_recall(user_multi_recall_dict['youtubednn_usercf_recall'], trn_last_click_df, topk=recall_item_num)

# Cell 57
# 先进行itemcf召回，这里不需要做召回评估，这里只是一种策略
trn_hist_click_df = all_click_df

user_recall_items_dict = defaultdict(dict)
user_item_time_dict = get_user_item_time(trn_hist_click_df)
i2i_sim = pickle.load(open(save_path / 'emb_i2i_sim.pkl','rb'))

sim_item_topk = 150
recall_item_num = 100 # 稍微召回多一点文章，便于后续的规则筛选

item_topk_click = get_item_topk_click(trn_hist_click_df, k=50)
for user in tqdm(trn_hist_click_df['user_id'].unique(), disable=not logger.isEnabledFor(logging.DEBUG)):
    user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict, i2i_sim, sim_item_topk,
                                                        recall_item_num, item_topk_click,item_created_time_dict, emb_i2i_sim)
pickle.dump(user_recall_items_dict, open(save_path / 'cold_start_items_raw_dict.pkl', 'wb'))

# Cell 58
# 基于规则进行文章过滤
# 保留文章主题与用户历史浏览主题相似的文章
# 保留文章字数与用户历史浏览文章字数相差不大的文章
# 保留最后一次点击当天的文章
# 按照相似度返回最终的结果

def get_click_article_ids_set(all_click_df):
    return set(all_click_df.click_article_id.values)

def cold_start_items(user_recall_items_dict, user_hist_item_typs_dict, user_hist_item_words_dict, \
                     user_last_item_created_time_dict, item_type_dict, item_words_dict,
                     item_created_time_dict, click_article_ids_set, recall_item_num):
    """
        冷启动的情况下召回一些文章
        :param user_recall_items_dict: 基于内容embedding相似性召回来的很多文章， 字典， {user1: [item1, item2, ..], }
        :param user_hist_item_typs_dict: 字典， 用户点击的文章的主题映射
        :param user_hist_item_words_dict: 字典， 用户点击的历史文章的字数映射
        :param user_last_item_created_time_idct: 字典，用户点击的历史文章创建时间映射
        :param item_tpye_idct: 字典，文章主题映射
        :param item_words_dict: 字典，文章字数映射
        :param item_created_time_dict: 字典， 文章创建时间映射
        :param click_article_ids_set: 集合，用户点击过得文章, 也就是日志里面出现过的文章
        :param recall_item_num: 召回文章的数量， 这个指的是没有出现在日志里面的文章数量
    """

    cold_start_user_items_dict = {}
    for user, item_list in tqdm(user_recall_items_dict.items(), disable=not logger.isEnabledFor(logging.DEBUG)):
        cold_start_user_items_dict.setdefault(user, [])
        for item, score in item_list:
            # 获取历史文章信息
            hist_item_type_set = user_hist_item_typs_dict[user]
            hist_mean_words = user_hist_item_words_dict[user]
            hist_last_item_created_time = user_last_item_created_time_dict[user]
            hist_last_item_created_time = datetime.fromtimestamp(hist_last_item_created_time)

            # 获取当前召回文章的信息
            curr_item_type = item_type_dict[item]
            curr_item_words = item_words_dict[item]
            curr_item_created_time = item_created_time_dict[item]
            curr_item_created_time = datetime.fromtimestamp(curr_item_created_time)

            # 首先，文章不能出现在用户的历史点击中， 然后根据文章主题，文章单词数，文章创建时间进行筛选
            if curr_item_type not in hist_item_type_set or \
                item in click_article_ids_set or \
                abs(curr_item_words - hist_mean_words) > 200 or \
                abs((curr_item_created_time - hist_last_item_created_time).days) > 90:
                continue

            cold_start_user_items_dict[user].append((item, score))      # {user1: [(item1, score1), (item2, score2)..]...}

    # 需要控制一下冷启动召回的数量
    cold_start_user_items_dict = {k: sorted(v, key=lambda x:x[1], reverse=True)[:recall_item_num] \
                                  for k, v in cold_start_user_items_dict.items()}

    pickle.dump(cold_start_user_items_dict, open(save_path / 'cold_start_user_items_dict.pkl', 'wb'))

    return cold_start_user_items_dict

# Cell 59
all_click_df_ = all_click_df.copy()
all_click_df_ = all_click_df_.merge(item_info_df, how='left', on='click_article_id')
user_hist_item_typs_dict, user_hist_item_ids_dict, user_hist_item_words_dict, user_last_item_created_time_dict = get_user_hist_item_info_dict(all_click_df_)
click_article_ids_set = get_click_article_ids_set(all_click_df)
# 需要注意的是
# 这里使用了很多规则来筛选冷启动的文章，所以前面再召回的阶段就应该尽可能的多召回一些文章，否则很容易被删掉
cold_start_user_items_dict = cold_start_items(user_recall_items_dict, user_hist_item_typs_dict, user_hist_item_words_dict, \
                                              user_last_item_created_time_dict, item_type_dict, item_words_dict, \
                                              item_created_time_dict, click_article_ids_set, recall_item_num)

user_multi_recall_dict['cold_start_recall'] = cold_start_user_items_dict

# Cell 61
def combine_recall_results(user_multi_recall_dict, weight_dict=None, topk=25):
    final_recall_items_dict = {}

    # 对每一种召回结果按照用户进行归一化，方便后面多种召回结果，相同用户的物品之间权重相加
    def norm_user_recall_items_sim(sorted_item_list):
        # 如果冷启动中没有文章或者只有一篇文章，直接返回，出现这种情况的原因可能是冷启动召回的文章数量太少了，
        # 基于规则筛选之后就没有文章了, 这里还可以做一些其他的策略性的筛选
        if len(sorted_item_list) < 2:
            return sorted_item_list

        min_sim = sorted_item_list[-1][1]
        max_sim = sorted_item_list[0][1]

        norm_sorted_item_list = []
        for item, score in sorted_item_list:
            if max_sim > 0:
                norm_score = 1.0 * (score - min_sim) / (max_sim - min_sim) if max_sim > min_sim else 1.0
            else:
                norm_score = 0.0
            norm_sorted_item_list.append((item, norm_score))

        return norm_sorted_item_list

    print('多路召回合并...')
    for method, user_recall_items in tqdm(user_multi_recall_dict.items(), disable=not logger.isEnabledFor(logging.DEBUG)):
        print(method + '...')
        # 在计算最终召回结果的时候，也可以为每一种召回结果设置一个权重
        if weight_dict == None:
            recall_method_weight = 1
        else:
            recall_method_weight = weight_dict[method]

        for user_id, sorted_item_list in user_recall_items.items(): # 进行归一化
            user_recall_items[user_id] = norm_user_recall_items_sim(sorted_item_list)

        for user_id, sorted_item_list in user_recall_items.items():
            # print('user_id')
            final_recall_items_dict.setdefault(user_id, {})
            for item, score in sorted_item_list:
                final_recall_items_dict[user_id].setdefault(item, 0)
                final_recall_items_dict[user_id][item] += recall_method_weight * score

    final_recall_items_dict_rank = {}
    # 多路召回时也可以控制最终的召回数量
    for user, recall_item_dict in final_recall_items_dict.items():
        final_recall_items_dict_rank[user] = sorted(recall_item_dict.items(), key=lambda x: x[1], reverse=True)[:topk]

    # 将多路召回后的最终结果字典保存到本地
    pickle.dump(final_recall_items_dict, open(os.path.join(save_path, 'final_recall_items_dict.pkl'),'wb'))

    return final_recall_items_dict_rank

# Cell 62
# 这里直接对多路召回的权重给了一个相同的值，其实可以根据前面召回的情况来调整参数的值
weight_dict = {'itemcf_sim_itemcf_recall': 1.0,
               'embedding_sim_item_recall': 1.0,
               'youtubednn_recall': 1.0,
               'youtubednn_usercf_recall': 1.0,
               'cold_start_recall': 1.0}

# Cell 63
# 最终合并之后每个用户召回150个商品进行排序
final_recall_items_dict_rank = combine_recall_results(user_multi_recall_dict, weight_dict, topk=150)

# Cell 65
# Step 1: 导入多路召回模块
# 直接从同目录下的 multi_strategy_recall.py 导入

import sys
sys.path.insert(0, '/Users/ymlin/Library/CloudStorage/OneDrive-Uppsalauniversitet/100-Study/130-CS/136 搜广推/天池新闻推荐/coding')

from multi_strategy_recall import ItemCFRecall, EmbeddingRecall, PopularityRecall, RecallFusion

print("✅ 多路召回模块导入成功！")
print("  - ItemCFRecall: 协同过滤召回")
print("  - EmbeddingRecall: 向量检索召回")
print("  - PopularityRecall: 热门推荐召回")
print("  - RecallFusion: 多路融合策略")

# Cell 66
# Step 2: 训练 ItemCF 协同过滤召回
# 使用全量数据训练（因为设置了offline=False）

print("=" * 60)
print("🔧 训练 ItemCF 协同过滤召回...")
print("=" * 60)

itemcf_recall = ItemCFRecall(
    sim_item_topk=100,        # 每个物品保留top100相似物品
    recall_item_number=100    # 每个用户召回100个候选
)

# 训练模型
itemcf_recall.fit(all_click_df)

print(f"\n✅ ItemCF训练完成")
print(f"   - 物品数量: {len(itemcf_recall.item_sim_dict)}")
print(f"   - 平均相似物品数: {np.mean([len(v) for v in itemcf_recall.item_sim_dict.values()]):.1f}")

# Cell 67
# Step 3: 训练 Embedding 向量检索召回

print("=" * 60)
print("🔧 训练 Embedding 向量检索召回...")
print("=" * 60)

# 准备 embedding 数据
# 使用已经加载的 item_emb_dict（从之前的cell加载）
embedding_recall = EmbeddingRecall(
    recall_item_number=100,
    use_faiss=False  # 暂时不用FAISS加速，直接计算
)

# 需要准备 DataFrame 格式的 embedding
# 从 item_emb_dict 转换为 DataFrame
article_ids = list(item_emb_dict.keys())
embeddings = np.array([item_emb_dict[aid] for aid in article_ids])

# 创建 embedding DataFrame
emb_dim = embeddings.shape[1]
emb_cols = [f'emb_{i}' for i in range(emb_dim)]
articles_emb_df = pd.DataFrame(embeddings, columns=emb_cols)
articles_emb_df['article_id'] = article_ids

# 训练模型
embedding_recall.fit(all_click_df, articles_emb_df)

print(f"\n✅ Embedding召回训练完成")
print(f"   - 文章数量: {len(embedding_recall.item_emb_dict)}")
print(f"   - Embedding维度: {emb_dim}")

# Cell 68
# Step 4: 训练 Popularity 热门推荐召回

print("=" * 60)
print("🔧 训练 Popularity 热门推荐召回...")
print("=" * 60)

popularity_recall = PopularityRecall(
    recall_item_number=100,
    time_decay_factor=0.95  # 时间衰减因子，越新的文章权重越高
)

# 训练模型
popularity_recall.fit(all_click_df)

print(f"\n✅ Popularity召回训练完成")
print(f"   - 热门文章数: {len(popularity_recall.popular_items)}")
print(f"   - Top 5 热门文章: {popularity_recall.popular_items[:5]}")

# Cell 69
# Step 5: 创建多路召回融合

print("=" * 60)
print("🎯 创建多路召回融合器...")
print("=" * 60)

# 配置权重：根据经验和实验调整
# ItemCF效果最好，给最高权重
# Embedding对新文章效果好
# Popularity用于补充冷启动
fusion = RecallFusion(
    strategies={
        'itemcf': itemcf_recall,
        'embedding': embedding_recall,
        'popularity': popularity_recall
    },
    weights={
        'itemcf': 0.5,       # ItemCF主力，权重50%
        'embedding': 0.35,   # Embedding辅助，权重35%
        'popularity': 0.15   # Popularity兜底，权重15%
    },
    final_topk=150  # 最终每个用户召回150个候选物品
)

print(f"\n✅ 融合器创建完成")
print(f"   - 策略数量: {len(fusion.strategies)}")
print(f"   - 融合方法: {fusion.fusion_method}")
print(f"   - 最终召回数: {fusion.final_topk}")

# Cell 70
# Step 6: 为所有用户生成召回结果

print("=" * 60)
print("🚀 开始批量召回...")
print("=" * 60)

# 获取所有需要召回的用户
all_users = all_click_df['user_id'].unique()
print(f"总用户数: {len(all_users)}")

# 使用融合策略进行批量召回
final_recall_results = fusion.predict_batch(all_users, all_click_df)

print(f"\n✅ 召回完成")
print(f"   - 召回用户数: {len(final_recall_results)}")
print(f"   - 平均每用户召回数: {np.mean([len(items) for items in final_recall_results.values()]):.1f}")

# 显示一个用户的召回示例
sample_user = list(final_recall_results.keys())[0]
sample_items = final_recall_results[sample_user]
print(f"\n📊 示例用户 {sample_user} 的召回结果:")
print(f"   Top 5 召回: {[item for item, score in sample_items[:5]]}")
print(f"   Top 5 分数: {[f'{score:.4f}' for item, score in sample_items[:5]]}")

# Cell 71
# Step 7: 生成提交文件

print("=" * 60)
print("📝 生成提交文件...")
print("=" * 60)

# 创建提交DataFrame
submission_data = []

for user_id, item_list in final_recall_results.items():
    # 取前5个推荐
    top5_items = [str(item) for item, score in item_list[:5]]
    
    # 如果不足5个，用热门文章补足
    while len(top5_items) < 5:
        for pop_item in popularity_recall.popular_items:
            if str(pop_item) not in top5_items:
                top5_items.append(str(pop_item))
                if len(top5_items) == 5:
                    break
    
    submission_data.append({
        'user_id': user_id,
        'article_1': top5_items[0],
        'article_2': top5_items[1],
        'article_3': top5_items[2],
        'article_4': top5_items[3],
        'article_5': top5_items[4]
    })

# 创建DataFrame
submission_df = pd.DataFrame(submission_data)

# 保存文件
output_file = save_path + 'submission_multi_strategy.csv'
submission_df.to_csv(output_file, index=False)

print(f"\n✅ 提交文件已生成")
print(f"   - 文件路径: {output_file}")
print(f"   - 用户数: {len(submission_df)}")
print(f"   - 文件大小: {os.path.getsize(output_file) / 1024:.2f} KB")
print(f"\n📊 前5行预览:")
print(submission_df.head())