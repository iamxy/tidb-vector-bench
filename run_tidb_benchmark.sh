#!/bin/bash
python3 ann_benchmarks/main.py   -d sift-128-euclidean   -a tidb tidb   --algorithm-param host=127.0.0.1   --algorithm-param port=4000   --algorithm-param user=root   --algorithm-param database=benchmark   --algorithm-param table=vec_sift   --runs 1
