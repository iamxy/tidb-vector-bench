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
from concurrent.futures import as_completed

from tidb_vector_bench.config.config import config
from tidb_vector_bench.db.loader import get_connection
from tidb_vector_bench.data.generate import read_fvecs
from tidb_vector_bench.benchmark.monitor import SystemMonitor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def _vector_to_sql(vec: np.ndarray) -> str:
    """将向量转换为 SQL 格式的字符串"""
    return f"[{','.join(map(str, vec))}]"

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

def query_with_connection(query_vec: np.ndarray, top_k: int) -> Tuple[float, List]:
    """使用新连接执行查询并返回延迟和结果"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        start_time = time.time()
        results = query(cursor, query_vec, top_k)
        latency = (time.time() - start_time) * 1000  # 转换为毫秒
        return latency, results
    finally:
        cursor.close()
        conn.close()

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
    """测试查询性能"""
    # 检查索引使用情况
    if not check_index_usage():
        logger.warning("警告: 查询未使用向量索引，性能可能不理想")
        logger.warning("请确保:")
        logger.warning("1. 表已创建 HNSW 向量索引")
        logger.warning("2. 索引构建已完成")
        logger.warning("3. 查询使用了正确的距离函数")
        return None
    
    # 准备查询向量
    logger.info("加载 SIFT1M 查询向量...")
    query_vectors = read_fvecs("data/sift/sift_query.fvecs")
    
    # 验证数据维度
    if query_vectors.shape[1] != config["benchmark"].vector_dim:
        raise ValueError(
            f"查询向量维度不匹配: 期望 {config['benchmark'].vector_dim}，"
            f"实际 {query_vectors.shape[1]}"
        )
    
    num_queries = min(len(query_vectors), config["benchmark"].num_queries)
    logger.info(f"使用 {num_queries} 条查询向量进行测试")
    
    # 创建结果目录
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # 初始化系统监控器
    monitor = SystemMonitor(interval=1.0)
    monitor.start()
    
    try:
        # 执行预热查询
        logger.info(f"执行 {config['benchmark'].warmup_queries} 次预热查询...")
        conn = get_connection()
        cursor = conn.cursor()
        
        try:
            for i in range(config["benchmark"].warmup_queries):
                query(cursor, query_vectors[i], config["benchmark"].top_k)
                logger.info(f"预热查询 {i+1}/{config['benchmark'].warmup_queries} 完成")
        finally:
            cursor.close()
            conn.close()
        
        # 执行单线程延迟测试
        logger.info("\n开始单线程延迟测试...")
        latencies = []
        
        conn = get_connection()
        cursor = conn.cursor()
        
        try:
            for i in tqdm(range(num_queries), desc="延迟测试"):
                start_time = time.time()
                query(cursor, query_vectors[i], config["benchmark"].top_k)
                latency = (time.time() - start_time) * 1000  # 转换为毫秒
                latencies.append(float(latency))  # 确保是浮点数
        finally:
            cursor.close()
            conn.close()
        
        # 执行并发吞吐量测试
        logger.info("\n开始并发吞吐量测试...")
        throughput_results = []
        
        with ThreadPoolExecutor(max_workers=config["benchmark"].threads) as executor:
            futures = []
            for i in range(num_queries):
                future = executor.submit(
                    query_with_connection,
                    query_vectors[i],
                    config["benchmark"].top_k
                )
                futures.append(future)
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="吞吐量测试"):
                latency, _ = future.result()
                throughput_results.append(float(latency))  # 确保是浮点数
        
        # 计算统计信息
        latencies = np.array(latencies, dtype=np.float64)
        throughput_latencies = np.array(throughput_results, dtype=np.float64)
        
        # 计算 QPS
        total_time = sum(throughput_latencies) / 1000  # 转换为秒
        qps = len(throughput_latencies) / total_time if total_time > 0 else 0
        
        stats = {
            "latency": {
                "mean": float(np.mean(latencies)),
                "p50": float(np.percentile(latencies, 50)),
                "p90": float(np.percentile(latencies, 90)),
                "p99": float(np.percentile(latencies, 99)),
                "min": float(np.min(latencies)),
                "max": float(np.max(latencies)),
                "std": float(np.std(latencies))
            },
            "throughput": {
                "qps": float(qps),  # 每秒查询数
                "mean_latency": float(np.mean(throughput_latencies)),
                "p50_latency": float(np.percentile(throughput_latencies, 50)),
                "p90_latency": float(np.percentile(throughput_latencies, 90)),
                "p99_latency": float(np.percentile(throughput_latencies, 99)),
                "min_latency": float(np.min(throughput_latencies)),
                "max_latency": float(np.max(throughput_latencies)),
                "latency_std": float(np.std(throughput_latencies))
            },
            "timestamp": datetime.now().isoformat(),
            "config": {
                "vector_dim": config["benchmark"].vector_dim,
                "top_k": config["benchmark"].top_k,
                "num_queries": num_queries,
                "threads": config["benchmark"].threads
            }
        }
        
        # 保存系统监控数据
        monitor.stop()
        monitor.save_metrics(results_dir)
        
        # 保存性能测试结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = results_dir / f"perf_{timestamp}.json"
        
        with open(result_file, "w") as f:
            json.dump(stats, f, indent=2)
        
        # 打印结果
        logger.info("\n性能测试结果:")
        logger.info(f"平均延迟: {stats['latency']['mean']:.2f} ms")
        logger.info(f"P50 延迟: {stats['latency']['p50']:.2f} ms")
        logger.info(f"P90 延迟: {stats['latency']['p90']:.2f} ms")
        logger.info(f"P99 延迟: {stats['latency']['p99']:.2f} ms")
        logger.info(f"最小延迟: {stats['latency']['min']:.2f} ms")
        logger.info(f"最大延迟: {stats['latency']['max']:.2f} ms")
        logger.info(f"延迟标准差: {stats['latency']['std']:.2f} ms")
        
        logger.info("\n并发测试结果:")
        logger.info(f"吞吐量 (QPS): {stats['throughput']['qps']:.2f}")
        logger.info(f"平均延迟: {stats['throughput']['mean_latency']:.2f} ms")
        logger.info(f"P50 延迟: {stats['throughput']['p50_latency']:.2f} ms")
        logger.info(f"P90 延迟: {stats['throughput']['p90_latency']:.2f} ms")
        logger.info(f"P99 延迟: {stats['throughput']['p99_latency']:.2f} ms")
        logger.info(f"最小延迟: {stats['throughput']['min_latency']:.2f} ms")
        logger.info(f"最大延迟: {stats['throughput']['max_latency']:.2f} ms")
        logger.info(f"延迟标准差: {stats['throughput']['latency_std']:.2f} ms")
        
        logger.info(f"\n结果已保存到: {result_file}")
        
        return stats
        
    except Exception as e:
        logger.error(f"性能测试失败: {e}")
        return None
    finally:
        # 确保监控器被停止
        if monitor.running:
            monitor.stop()
