# TiDB Vector Benchmark with ann-benchmarks

This is a skeleton project to benchmark TiDB Vector using the `ann-benchmarks` framework.

## Setup

1. Clone ann-benchmarks into this directory:
   git clone https://github.com/erikbern/ann-benchmarks.git ann_benchmarks

2. Place a `tidb.py` file in `ann_benchmarks/algorithms/` to implement TiDB Vector interface.

3. Run:
   ./run_tidb_benchmark.sh

## Requirements

- Python 3.10+
- TiDB started via `tiup playground`
