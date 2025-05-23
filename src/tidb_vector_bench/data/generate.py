import numpy as np
from typing import List, Tuple
from pathlib import Path
import h5py
import urllib.request
import os
import struct
import tarfile
from tqdm import tqdm

from tidb_vector_bench.config.config import config

def generate_random_vector(dim: int) -> List[float]:
    """生成一个随机向量"""
    return np.random.randn(dim).tolist()

def download_file(url: str, filepath: Path, desc: str):
    """下载文件并显示进度条"""
    if url.startswith('ftp://'):
        # 使用 urllib 下载 FTP 文件
        print(f"从 FTP 下载 {desc}...")
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, desc=desc) as pbar:
            def report_progress(block_num, block_size, total_size):
                if total_size > 0:
                    pbar.total = total_size
                pbar.update(block_size)
            
            urllib.request.urlretrieve(url, filepath, report_progress)
    else:
        # 使用 requests 下载 HTTP 文件
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f, tqdm(
            desc=desc,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)

def read_fvecs(filename: str) -> np.ndarray:
    """读取 .fvecs 格式的向量文件
    
    .fvecs 文件格式：
    - 每个向量以 4 字节的维度开始
    - 然后是维度个 4 字节的浮点数
    """
    with open(filename, 'rb') as f:
        # 读取第一个向量的维度
        dim = struct.unpack('i', f.read(4))[0]
        
        # 计算文件大小和向量数量
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(0)
        
        # 每个向量的字节数 = 4(维度) + 4 * dim(数据)
        vector_size = 4 + 4 * dim
        num_vectors = file_size // vector_size
        
        print(f"{filename} 文件包含 {num_vectors} 个 {dim} 维向量")
        
        # 预分配内存
        data = np.zeros((num_vectors, dim), dtype=np.float32)
        
        # 读取所有向量
        for i in tqdm(range(num_vectors), desc="读取向量"):
            # 读取维度（应该是固定的）
            vec_dim = struct.unpack('i', f.read(4))[0]
            assert vec_dim == dim, f"向量维度不一致: {vec_dim} != {dim}"
            
            # 读取向量数据
            vec = struct.unpack('f' * dim, f.read(4 * dim))
            data[i] = vec
        
        return data

def read_ivecs(filename: str) -> np.ndarray:
    """读取 .ivecs 格式的向量文件
    
    .ivecs 文件格式：
    - 每个向量以 4 字节的维度开始
    - 然后是维度个 4 字节的整数
    """
    with open(filename, 'rb') as f:
        # 读取第一个向量的维度
        dim = struct.unpack('i', f.read(4))[0]
        
        # 计算文件大小和向量数量
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(0)
        
        # 每个向量的字节数 = 4(维度) + 4 * dim(数据)
        vector_size = 4 + 4 * dim
        num_vectors = file_size // vector_size
        
        print(f"{filename} 文件包含 {num_vectors} 个 {dim} 维向量")
        
        # 预分配内存
        data = np.zeros((num_vectors, dim), dtype=np.int32)
        
        # 读取所有向量
        for i in tqdm(range(num_vectors), desc="读取向量"):
            # 读取维度（应该是固定的）
            vec_dim = struct.unpack('i', f.read(4))[0]
            assert vec_dim == dim, f"向量维度不一致: {vec_dim} != {dim}"
            
            # 读取向量数据
            vec = struct.unpack('i' * dim, f.read(4 * dim))
            data[i] = vec
        
        return data

def download_sift1m() -> Tuple[np.ndarray, np.ndarray]:
    """下载并加载 SIFT1M 数据集
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (base_vectors, query_vectors)
        - base_vectors: 100万条 128 维向量
        - query_vectors: 1000条查询向量
    """
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # 下载数据集
    tar_file = data_dir / "sift.tar.gz"
    if not tar_file.exists():
        print("下载 SIFT1M 数据集...")
        download_file("ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz", tar_file, "SIFT1M dataset")
        
        print("解压数据集...")
        with tarfile.open(tar_file, "r:gz") as tar:
            tar.extractall(path=data_dir)
        
        # 删除压缩文件
        tar_file.unlink()
    
    # 读取基准向量和查询向量
    print("读取基准向量...")
    base_vectors = read_fvecs(data_dir / "sift/sift_base.fvecs")
    print("读取查询向量...")
    query_vectors = read_fvecs(data_dir / "sift/sift_query.fvecs")
    
    return base_vectors, query_vectors

def generate_vectors():
    """生成或下载测试向量数据"""
    if config["benchmark"].use_sift1m:
        print("使用 SIFT1M 数据集...")
        base_vectors, query_vectors = download_sift1m()
        
        # 保存到 HDF5 文件
        with h5py.File("data/vectors.h5", "w") as f:
            f.create_dataset("vectors", data=base_vectors)
            f.create_dataset("queries", data=query_vectors)
        
        print(f"已保存 {len(base_vectors)} 条基准向量和 {len(query_vectors)} 条查询向量")
    else:
        print("生成随机向量数据...")
        vectors = np.random.randn(
            config["benchmark"].num_vectors,
            config["benchmark"].vector_dim
        ).astype(np.float32)
        
        with h5py.File("data/vectors.h5", "w") as f:
            f.create_dataset("vectors", data=vectors)
        
        print(f"已生成 {len(vectors)} 条随机向量")

def generate_random_vectors(num_vectors: int, dim: int) -> np.ndarray:
    """生成多个随机向量"""
    return np.random.randn(num_vectors, dim).astype(np.float32)

def save_vectors_to_h5(vectors: np.ndarray, filename: str):
    """将向量保存为 HDF5 格式"""
    with h5py.File(filename, 'w') as f:
        f.create_dataset('vectors', data=vectors)

def load_vectors_from_h5(filename: str) -> np.ndarray:
    """从 HDF5 文件加载向量"""
    with h5py.File(filename, 'r') as f:
        return f['vectors'][:]
