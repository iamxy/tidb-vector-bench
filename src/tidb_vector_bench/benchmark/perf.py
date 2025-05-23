import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
from datetime import datetime
import logging

from tidb_vector_bench.config.config import config
from tidb_vector_bench.db.loader import get_connection
from tidb_vector_bench.data.generate import read_fvecs

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def _vector_to_sql(vec: np.ndarray) -> str:
    """将向量转换为 SQL 格式的字符串"""
    return f"[{','.join(map(str, vec))}]"

def check_index_usage():
    """检查向量索引是否被使用"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # 生成测试向量
        test_vec = np.random.randn(config["benchmark"].vector_dim)
        vec_sql = _vector_to_sql(test_vec)
        
        # 使用 EXPLAIN 分析查询计划
        cursor.execute(
            f"""
            EXPLAIN FORMAT='verbose'
            SELECT id, vec
            FROM {config['db'].table_name}
            ORDER BY VEC_COSINE_DISTANCE(vec, %s)
            LIMIT %s
            """,
            (vec_sql, config["benchmark"].top_k)
        )
        
        explain_result = cursor.fetchall()
        
        # 检查是否使用了向量索引
        for row in explain_result:
            if 'annIndex:' in str(row):
                logger.info("✓ 查询使用了 HNSW 向量索引")
                return True
        
        logger.warning("✗ 查询未使用 HNSW 向量索引")
        return False
    finally:
        cursor.close()
        conn.close()

def _execute_query(cursor, query_vec: np.ndarray, top_k: int) -> Tuple[float, List]:
    """执行单个查询并返回延迟和结果"""
    vec_sql = _vector_to_sql(query_vec)
    start_time = time.time()
    cursor.execute(
        f"""
        SELECT id, vec
        FROM {config['db'].table_name}
        ORDER BY VEC_COSINE_DISTANCE(vec, %s)
        LIMIT %s
        """,
        (vec_sql, top_k)
    )
    results = cursor.fetchall()
    latency = time.time() - start_time
    return latency, results

def test_performance():
    """测试查询性能（延迟和吞吐量）"""
    # 检查索引使用情况
    if not check_index_usage():
        logger.warning("警告: 查询未使用向量索引，性能可能不理想")
        logger.warning("请确保:")
        logger.warning("1. 表已创建 HNSW 向量索引")
        logger.warning("2. 索引构建已完成")
        logger.warning("3. 查询使用了正确的距离函数")
        return None
    
    # 准备查询向量
    if config["benchmark"].use_sift1m:
        logger.info("使用 SIFT1M 查询向量...")
        query_vectors = read_fvecs("data/sift/sift_query.fvecs")
        num_queries = min(len(query_vectors), config["benchmark"].num_queries)
        logger.info(f"使用 {num_queries} 条查询向量进行测试")
    else:
        logger.info("生成随机查询向量...")
        query_vectors = [np.random.randn(config["benchmark"].vector_dim) 
                        for _ in range(config["benchmark"].num_queries)]
        num_queries = config["benchmark"].num_queries
    
    # 预热
    logger.info("预热中...")
    conn = get_connection()
    cursor = conn.cursor()
    try:
        # 使用多个不同的查询向量进行预热
        warmup_vectors = query_vectors[:config["benchmark"].warmup_queries]
        for i, query_vec in enumerate(warmup_vectors, 1):
            logger.info(f"预热查询 {i}/{config['benchmark'].warmup_queries}...")
            _execute_query(cursor, query_vec, config["benchmark"].top_k)
    finally:
        cursor.close()
        conn.close()
    
    # 单线程延迟测试
    logger.info("开始单线程延迟测试...")
    latencies = []
    for i in tqdm(range(min(100, num_queries)), desc="测试延迟"):
        conn = get_connection()
        cursor = conn.cursor()
        try:
            latency, _ = _execute_query(cursor, query_vectors[i], config["benchmark"].top_k)
            latencies.append(latency)
        finally:
            cursor.close()
            conn.close()
    
    # 并发吞吐量测试
    logger.info(f"开始并发吞吐量测试 (线程数: {config['benchmark'].threads})...")
    throughput_results = []
    throughput_latencies = []
    
    def worker(query_vec: np.ndarray) -> Tuple[float, List]:
        """工作线程函数"""
        conn = get_connection()
        cursor = conn.cursor()
        try:
            return _execute_query(cursor, query_vec, config["benchmark"].top_k)
        finally:
            cursor.close()
            conn.close()
    
    with ThreadPoolExecutor(max_workers=config["benchmark"].threads) as executor:
        futures = []
        for query_vec in query_vectors[:num_queries]:
            future = executor.submit(worker, query_vec)
            futures.append(future)
        
        for future in tqdm(futures, desc="测试吞吐量"):
            latency, result = future.result()
            throughput_latencies.append(latency)
            throughput_results.append(result)
    
    # 计算统计信息
    latencies = np.array(latencies)
    throughput_latencies = np.array(throughput_latencies)
    total_time = sum(throughput_latencies)
    qps = len(throughput_results) / total_time
    
    stats = {
        "latency": {
            "mean": float(np.mean(latencies) * 1000),  # 转换为毫秒
            "p50": float(np.percentile(latencies, 50) * 1000),
            "p90": float(np.percentile(latencies, 90) * 1000),
            "p99": float(np.percentile(latencies, 99) * 1000),
            "min": float(np.min(latencies) * 1000),
            "max": float(np.max(latencies) * 1000),
            "std": float(np.std(latencies) * 1000)
        },
        "throughput": {
            "total_queries": len(throughput_results),
            "total_time": float(total_time),
            "qps": float(qps),
            "latency": {
                "mean": float(np.mean(throughput_latencies) * 1000),
                "p50": float(np.percentile(throughput_latencies, 50) * 1000),
                "p90": float(np.percentile(throughput_latencies, 90) * 1000),
                "p99": float(np.percentile(throughput_latencies, 99) * 1000),
                "min": float(np.min(throughput_latencies) * 1000),
                "max": float(np.max(throughput_latencies) * 1000),
                "std": float(np.std(throughput_latencies) * 1000)
            }
        },
        "timestamp": datetime.now().isoformat(),
        "config": {
            "vector_dim": config["benchmark"].vector_dim,
            "top_k": config["benchmark"].top_k,
            "num_queries": num_queries,
            "threads": config["benchmark"].threads,
            "use_sift1m": config["benchmark"].use_sift1m
        }
    }
    
    # 打印结果
    logger.info("\n性能测试结果:")
    logger.info("\n单线程延迟测试:")
    logger.info(f"平均延迟: {stats['latency']['mean']:.2f}ms")
    logger.info(f"P50 延迟: {stats['latency']['p50']:.2f}ms")
    logger.info(f"P90 延迟: {stats['latency']['p90']:.2f}ms")
    logger.info(f"P99 延迟: {stats['latency']['p99']:.2f}ms")
    logger.info(f"最小延迟: {stats['latency']['min']:.2f}ms")
    logger.info(f"最大延迟: {stats['latency']['max']:.2f}ms")
    logger.info(f"延迟标准差: {stats['latency']['std']:.2f}ms")
    
    logger.info("\n并发吞吐量测试:")
    logger.info(f"总查询数: {stats['throughput']['total_queries']}")
    logger.info(f"总时间: {stats['throughput']['total_time']:.2f}s")
    logger.info(f"QPS: {stats['throughput']['qps']:.2f}")
    logger.info(f"平均延迟: {stats['throughput']['latency']['mean']:.2f}ms")
    logger.info(f"P50 延迟: {stats['throughput']['latency']['p50']:.2f}ms")
    logger.info(f"P90 延迟: {stats['throughput']['latency']['p90']:.2f}ms")
    logger.info(f"P99 延迟: {stats['throughput']['latency']['p99']:.2f}ms")
    logger.info(f"最小延迟: {stats['throughput']['latency']['min']:.2f}ms")
    logger.info(f"最大延迟: {stats['throughput']['latency']['max']:.2f}ms")
    logger.info(f"延迟标准差: {stats['throughput']['latency']['std']:.2f}ms")
    
    # 保存结果
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = results_dir / f"performance_{timestamp}.json"
    
    with open(result_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"\n结果已保存到: {result_file}")
    
    return stats 