# TiDB Vector Benchmark

这是一个用于测试 TiDB Vector 性能的基准测试工具。

## 功能特点

- 支持大规模向量数据测试（默认 1M 向量，维度 1536）
- 测量关键性能指标：
  - 吞吐量（QPS）
  - 延迟（平均、P50、P90、P99）
  - 召回率
  - 系统资源使用情况（CPU、内存、磁盘、网络）
- 支持并发测试
- 自动生成测试报告

## 安装

```bash
# 克隆仓库
git clone https://github.com/your-username/tidb-vector-bench.git
cd tidb-vector-bench

# 安装依赖
uv pip install -e .
```

## 使用方法

1. 生成测试数据：
```bash
uv run python -m tidb_vector_bench generate
```

2. 将数据导入数据库：
```bash
uv run python -m tidb_vector_bench insert
```

3. 运行性能测试：
```bash
# 运行性能测试（延迟和吞吐量）
uv run python -m tidb_vector_bench perf

# 运行召回率测试
uv run python -m tidb_vector_bench recall

# 运行所有测试
uv run python -m tidb_vector_bench all
```

## 配置

可以通过修改 `config/config.yaml` 文件来调整测试参数：

- 数据库连接信息
- 向量维度
- 数据量
- 并发线程数
- 测试参数（如 top-k 值）

## 测试结果

测试结果将保存在 `results` 目录下，包括：
- JSON 格式的详细测试数据
- 系统资源使用情况
- 性能指标统计

## 开发

```bash
# 安装开发依赖
uv pip install -e ".[dev]"

# 运行测试
uv run pytest

# 代码格式化
uv run ruff format .
```


