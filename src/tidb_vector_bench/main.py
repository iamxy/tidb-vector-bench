import argparse
from tidb_vector_bench.data.generate import generate_vectors
from tidb_vector_bench.db.loader import create_table, insert_vectors
from tidb_vector_bench.benchmark.perf import test_performance
from tidb_vector_bench.benchmark.recall import test_recall

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # generate 命令
    generate_parser = subparsers.add_parser("generate", help="Generate vector data")
    
    # insert 命令
    insert_parser = subparsers.add_parser("insert", help="Insert vectors into database")
    
    # perf 命令
    perf_parser = subparsers.add_parser("perf", help="Test query performance (latency and throughput)")
    
    # recall 命令
    recall_parser = subparsers.add_parser("recall", help="Test recall rate")
    
    args = parser.parse_args()
    
    if args.command == "generate":
        generate_vectors()
    elif args.command == "insert":
        create_table()
        insert_vectors()
    elif args.command == "perf":
        test_performance()
    elif args.command == "recall":
        test_recall()
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 