import numpy as np
from typing import List, Tuple, Set, Dict
from tqdm import tqdm
import json
from pathlib import Path
from datetime import datetime
import logging
from contextlib import contextmanager
import matplotlib.pyplot as plt
import matplotlib as mpl
import faiss

from tidb_vector_bench.config.config import config
from tidb_vector_bench.db.loader import get_connection
from tidb_vector_bench.data.generate import read_fvecs, read_ivecs
from tidb_vector_bench.benchmark.perf import _vector_to_sql, check_index_usage

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 配置 matplotlib
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
mpl.rcParams['font.family'] = 'Arial Unicode MS'
mpl.rcParams['axes.unicode_minus'] = False

@contextmanager
def get_db_connection():
    """数据库连接的上下文管理器"""
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()

def load_ground_truth(query_id: int) -> Set[int]:
    """从 ground truth 文件中加载真实最近邻"""
    if not hasattr(load_ground_truth, 'ground_truth'):
        logger.info("加载 ground truth 数据...")
        load_ground_truth.ground_truth = read_ivecs("data/sift/sift_groundtruth.ivecs")
    return set(load_ground_truth.ground_truth[query_id])

def exact_knn_search(query_vec: np.ndarray, top_k: int) -> Set[int]:
    """使用 FAISS 执行精确 KNN 搜索"""
    if not hasattr(exact_knn_search, 'index'):
        logger.info("构建 FAISS 精确搜索索引...")
        # 从数据库加载所有向量
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT id, vec FROM {config['db'].table_name}")
            rows = cursor.fetchall()
            
            # 构建 FAISS 索引
            vectors = np.array([np.frombuffer(row[1], dtype=np.float32) for row in rows])
            ids = np.array([row[0] for row in rows])
            
            # 创建精确搜索索引
            exact_knn_search.index = faiss.IndexFlatIP(vectors.shape[1])
            exact_knn_search.index.add(vectors)
            exact_knn_search.ids = ids
    
    # 执行搜索
    distances, indices = exact_knn_search.index.search(query_vec.reshape(1, -1).astype(np.float32), top_k)
    return set(exact_knn_search.ids[indices[0]])

def calculate_mrr(ann_results: List[int], exact_results: Set[int]) -> float:
    """计算 MRR (Mean Reciprocal Rank)
    
    Args:
        ann_results: 近似搜索结果列表
        exact_results: 真实最近邻集合
    
    Returns:
        float: MRR 值，范围 [0, 1]
    """
    for rank, result_id in enumerate(ann_results, 1):
        if result_id in exact_results:
            return 1.0 / rank
    return 0.0

def calculate_ndcg(ann_results: List[int], exact_results: Set[int], k: int) -> float:
    """计算 NDCG (Normalized Discounted Cumulative Gain)
    
    Args:
        ann_results: 近似搜索结果列表
        exact_results: 真实最近邻集合
        k: 评估的 top-k 值
    
    Returns:
        float: NDCG 值，范围 [0, 1]
    """
    # 计算 DCG
    dcg = 0.0
    for i, result_id in enumerate(ann_results[:k]):
        if result_id in exact_results:
            dcg += 1.0 / np.log2(i + 2)
    
    # 计算 IDCG (理想情况下的 DCG)
    # 假设所有相关结果都在最前面
    idcg = 0.0
    num_relevant = min(k, len(exact_results))
    for i in range(num_relevant):
        idcg += 1.0 / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0.0

def plot_recall_distribution(recalls: np.ndarray, save_path: Path):
    """绘制召回率分布图
    
    Args:
        recalls: 召回率数组
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 8))
    
    # 绘制直方图
    n, bins, patches = plt.hist(recalls, bins=20, alpha=0.7, color='blue', 
                              edgecolor='black', linewidth=1.2)
    
    # 添加统计信息
    mean_recall = np.mean(recalls)
    median_recall = np.median(recalls)
    std_recall = np.std(recalls)
    
    stats_text = (
        f"平均召回率: {mean_recall:.4f}\n"
        f"中位数召回率: {median_recall:.4f}\n"
        f"标准差: {std_recall:.4f}\n"
        f"最小值: {np.min(recalls):.4f}\n"
        f"最大值: {np.max(recalls):.4f}"
    )
    
    # 添加统计信息文本框
    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 添加垂直线表示平均值和中位数
    plt.axvline(mean_recall, color='red', linestyle='--', alpha=0.7, 
                label=f'平均值: {mean_recall:.4f}')
    plt.axvline(median_recall, color='green', linestyle='--', alpha=0.7,
                label=f'中位数: {median_recall:.4f}')
    
    # 设置标题和标签
    plt.title('召回率分布', fontsize=14, pad=20)
    plt.xlabel('召回率', fontsize=12)
    plt.ylabel('频次', fontsize=12)
    
    # 添加网格和图例
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def query(cursor, query_vec: np.ndarray, top_k: int) -> List[Tuple[int, np.ndarray]]:
    """执行向量查询"""
    vec_sql = _vector_to_sql(query_vec)
    cursor.execute(
        f"""
        SELECT id, vec
        FROM {config['db'].table_name}
        ORDER BY VEC_COSINE_DISTANCE(vec, %s)
        LIMIT %s
        """,
        (vec_sql, top_k)
    )
    return cursor.fetchall()

def test_recall():
    """测试查询召回率"""
    # 检查索引使用情况
    if not check_index_usage():
        logger.warning("警告: 查询未使用向量索引，召回率可能不理想")
        logger.warning("请确保:")
        logger.warning("1. 表已创建 HNSW 向量索引")
        logger.warning("2. 索引构建已完成")
        logger.warning("3. 查询使用了正确的距离函数")
        return None
    
    # 准备查询向量和基准结果
    logger.info("加载 SIFT1M 查询向量和基准结果...")
    query_vectors = read_fvecs("data/sift/sift_query.fvecs")
    ground_truth = read_ivecs("data/sift/sift_groundtruth.ivecs")
    
    # 验证数据维度
    if query_vectors.shape[1] != config["benchmark"].vector_dim:
        raise ValueError(
            f"查询向量维度不匹配: 期望 {config['benchmark'].vector_dim}，"
            f"实际 {query_vectors.shape[1]}"
        )
    
    # 验证基准结果格式
    logger.info(f"基准结果文件格式: {ground_truth.shape[0]} 个查询，每个查询 {ground_truth.shape[1]} 个最近邻")
    if ground_truth.shape[1] != config["benchmark"].top_k:
        logger.warning(
            f"基准结果中的最近邻数量 ({ground_truth.shape[1]}) 与配置的 top_k "
            f"({config['benchmark'].top_k}) 不匹配，将使用基准结果中的数量"
        )
        config["benchmark"].top_k = ground_truth.shape[1]
    
    num_queries = min(len(query_vectors), config["benchmark"].num_queries)
    logger.info(f"使用 {num_queries} 条查询向量进行测试")
    logger.info(f"查询向量维度: {query_vectors.shape[1]}")
    logger.info(f"每个查询返回的最近邻数量: {config['benchmark'].top_k}")
    
    # 执行查询并计算召回率
    logger.info("开始测试召回率...")
    recalls = []
    mrrs = []
    ndcgs = []
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        for i in tqdm(range(num_queries), desc="测试召回率"):
            # 执行查询
            results = query(cursor, query_vectors[i], config["benchmark"].top_k)
            result_ids = [r[0] for r in results]
            
            # 获取基准结果（ground truth 中的 ID 列表）
            gt_ids = ground_truth[i]
            
            # 计算召回率（正确结果数量 / 总结果数量）
            correct = len(set(result_ids) & set(gt_ids))
            recall = correct / config["benchmark"].top_k
            recalls.append(recall)
            
            # 计算 MRR（第一个正确结果的排名的倒数）
            mrr = calculate_mrr(result_ids, set(gt_ids))
            mrrs.append(mrr)
            
            # 计算 NDCG（考虑排序质量的指标）
            ndcg = calculate_ndcg(result_ids, set(gt_ids), config["benchmark"].top_k)
            ndcgs.append(ndcg)
            
            # 记录一些样本的详细信息
            if i < 3:  # 只记录前3个查询的详细信息
                logger.info(f"\n查询 {i+1} 的详细信息:")
                logger.info(f"近似搜索结果 (top-{config['benchmark'].top_k}): {result_ids}")
                logger.info(f"真实最近邻 (top-{config['benchmark'].top_k}): {gt_ids}")
                logger.info(f"召回率: {recall:.4f} (正确结果数: {correct}/{config['benchmark'].top_k})")
                logger.info(f"MRR: {mrr:.4f}")
                logger.info(f"NDCG: {ndcg:.4f}")
    finally:
        cursor.close()
        conn.close()
    
    # 计算统计信息
    recalls = np.array(recalls)
    mrrs = np.array(mrrs)
    ndcgs = np.array(ndcgs)
    
    stats = {
        "recall": {
            "mean": float(np.mean(recalls)),
            "p50": float(np.percentile(recalls, 50)),
            "p90": float(np.percentile(recalls, 90)),
            "p99": float(np.percentile(recalls, 99)),
            "min": float(np.min(recalls)),
            "max": float(np.max(recalls)),
            "std": float(np.std(recalls))
        },
        "mrr": {
            "mean": float(np.mean(mrrs)),
            "p50": float(np.percentile(mrrs, 50)),
            "p90": float(np.percentile(mrrs, 90)),
            "p99": float(np.percentile(mrrs, 99)),
            "min": float(np.min(mrrs)),
            "max": float(np.max(mrrs)),
            "std": float(np.std(mrrs))
        },
        "ndcg": {
            "mean": float(np.mean(ndcgs)),
            "p50": float(np.percentile(ndcgs, 50)),
            "p90": float(np.percentile(ndcgs, 90)),
            "p99": float(np.percentile(ndcgs, 99)),
            "min": float(np.min(ndcgs)),
            "max": float(np.max(ndcgs)),
            "std": float(np.std(ndcgs))
        },
        "timestamp": datetime.now().isoformat(),
        "config": {
            "vector_dim": config["benchmark"].vector_dim,
            "top_k": config["benchmark"].top_k,
            "num_queries": num_queries
        }
    }
    
    # 打印结果
    logger.info("\n召回率测试结果:")
    logger.info(f"平均召回率: {stats['recall']['mean']:.4f}")
    logger.info(f"P50 召回率: {stats['recall']['p50']:.4f}")
    logger.info(f"P90 召回率: {stats['recall']['p90']:.4f}")
    logger.info(f"P99 召回率: {stats['recall']['p99']:.4f}")
    logger.info(f"最小召回率: {stats['recall']['min']:.4f}")
    logger.info(f"最大召回率: {stats['recall']['max']:.4f}")
    logger.info(f"召回率标准差: {stats['recall']['std']:.4f}")
    
    logger.info("\nMRR 测试结果:")
    logger.info(f"平均 MRR: {stats['mrr']['mean']:.4f}")
    logger.info(f"P50 MRR: {stats['mrr']['p50']:.4f}")
    logger.info(f"P90 MRR: {stats['mrr']['p90']:.4f}")
    logger.info(f"P99 MRR: {stats['mrr']['p99']:.4f}")
    
    logger.info("\nNDCG 测试结果:")
    logger.info(f"平均 NDCG: {stats['ndcg']['mean']:.4f}")
    logger.info(f"P50 NDCG: {stats['ndcg']['p50']:.4f}")
    logger.info(f"P90 NDCG: {stats['ndcg']['p90']:.4f}")
    logger.info(f"P99 NDCG: {stats['ndcg']['p99']:.4f}")
    
    # 保存结果
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = results_dir / f"recall_{timestamp}.json"
    
    with open(result_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"\n结果已保存到: {result_file}")
    
    # 生成召回率分布图
    logger.info("\n生成召回率分布图...")
    plot_recall_distribution(recalls, results_dir / f"recall_dist_{timestamp}.png")
    logger.info(f"分布图已保存到: {results_dir}/recall_dist_{timestamp}.png")
    
    return stats
