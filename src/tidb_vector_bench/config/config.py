from dataclasses import dataclass
from typing import Optional, Dict, Any
import yaml
from pathlib import Path

@dataclass
class DBConfig:
    host: str
    port: int
    user: str
    password: str
    database: str
    table_name: str

@dataclass
class BenchmarkConfig:
    vector_dim: int = 128
    num_vectors: int = 1000000
    batch_size: int = 1000
    num_queries: int = 1000
    top_k: int = 10
    threads: int = 4
    warmup_queries: int = 50

@dataclass
class MonitorConfig:
    collect_interval: float = 1.0  # seconds
    metrics: list[str] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["cpu", "memory", "disk", "network"]

config: Dict[str, Any] = {}

def load_config():
    """加载配置文件"""
    config_file = Path("config/config.yaml")
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_file}")
    
    with open(config_file) as f:
        config_data = yaml.safe_load(f)
    
    # 转换配置为对象
    config["db"] = DBConfig(**config_data["db"])
    config["benchmark"] = BenchmarkConfig(**config_data["benchmark"])
    
    return config

# 在模块级别加载配置
load_config()
