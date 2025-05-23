import mysql.connector
from typing import List
import numpy as np
from tqdm import tqdm
import h5py
from pathlib import Path
import struct
import os

from tidb_vector_bench.config.config import config
from tidb_vector_bench.data.generate import read_fvecs

def get_connection():
    """获取数据库连接"""
    return mysql.connector.connect(
        host=config["db"].host,
        port=config["db"].port,
        user=config["db"].user,
        password=config["db"].password,
        database=config["db"].database
    )

def create_table():
    """创建向量表和 HNSW 索引"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # 创建表和索引
        vector_dim = config["benchmark"].vector_dim
        print(f"创建表 {config['db'].table_name}，向量维度: {vector_dim}")
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {config['db'].table_name} (
            id BIGINT PRIMARY KEY,
            vec VECTOR({vector_dim}),
            VECTOR INDEX idx_vec ((VEC_COSINE_DISTANCE(vec)))
        )
        """)
        print(f"创建表 {config['db'].table_name} 和 HNSW 索引")
    except Exception as e:
        print(f"创建表或索引时出错: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

def truncate_table():
    """清空表数据"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        print(f"清空表 {config['db'].table_name}...")
        cursor.execute(f"TRUNCATE TABLE {config['db'].table_name}")
        print(f"表 {config['db'].table_name} 已清空")
    except Exception as e:
        print(f"清空表时出错: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

def _vector_to_sql(vec: np.ndarray) -> str:
    """将向量转换为 SQL 格式的字符串"""
    return f"[{','.join(map(str, vec))}]"

def insert_vectors():
    """从数据文件读取向量数据并插入到数据库"""
    # 先清空表
    truncate_table()
    
    if config["benchmark"].use_sift1m:
        # 读取 SIFT 数据集
        data_file = Path("data/sift/sift_base.fvecs")
        if not data_file.exists():
            raise FileNotFoundError(f"SIFT base vectors file not found: {data_file}")
        
        print("读取 SIFT 基准向量...")
        vectors = read_fvecs(str(data_file))
    else:
        # 读取 HDF5 文件
        data_file = Path("data/vectors.h5")
        if not data_file.exists():
            raise FileNotFoundError(f"Vector data file not found: {data_file}")
        
        print("读取 HDF5 向量数据...")
        with h5py.File(data_file, "r") as f:
            if "vectors" not in f:
                raise KeyError("Dataset 'vectors' not found in HDF5 file")
            vectors = f["vectors"][:]
    
    vector_dim = vectors.shape[1]
    
    # 验证向量维度
    if vector_dim != config["benchmark"].vector_dim:
        raise ValueError(
            f"Vector dimension mismatch: expected {config['benchmark'].vector_dim}, "
            f"got {vector_dim}"
        )
    
    print(f"读取到 {len(vectors)} 条 {vector_dim} 维向量")
    
    # 连接数据库
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # 批量插入向量
        batch_size = config["benchmark"].batch_size
        num_vectors = len(vectors)
        
        for i in tqdm(range(0, num_vectors, batch_size), desc="插入向量"):
            for j in range(min(batch_size, num_vectors - i)):
                vec = vectors[i + j]
                vec_sql = _vector_to_sql(vec)
                cursor.execute(
                    f"INSERT INTO {config['db'].table_name} (id, vec) VALUES (%s, %s)",
                    (i + j, vec_sql)
                )
            conn.commit()
        
        print(f"成功插入 {num_vectors} 条向量")
    except Exception as e:
        print(f"插入向量时出错: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    create_table()
    truncate_table()  # 先清空表
    insert_vectors()
