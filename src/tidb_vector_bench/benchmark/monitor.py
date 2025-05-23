import time
import psutil
import threading
from typing import Dict, List
from datetime import datetime
import json
from pathlib import Path

from tidb_vector_bench.config.config import config

class SystemMonitor:
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metrics: List[Dict] = []
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._collect_metrics)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        self._save_metrics()

    def _collect_metrics(self):
        while self.running:
            metric = {
                "timestamp": datetime.now().isoformat(),
                "cpu": {
                    "percent": psutil.cpu_percent(interval=1),
                    "count": psutil.cpu_count(),
                },
                "memory": {
                    "total": psutil.virtual_memory().total,
                    "available": psutil.virtual_memory().available,
                    "percent": psutil.virtual_memory().percent,
                },
                "disk": {
                    "total": psutil.disk_usage('/').total,
                    "used": psutil.disk_usage('/').used,
                    "free": psutil.disk_usage('/').free,
                    "percent": psutil.disk_usage('/').percent,
                },
                "network": {
                    "bytes_sent": psutil.net_io_counters().bytes_sent,
                    "bytes_recv": psutil.net_io_counters().bytes_recv,
                }
            }
            self.metrics.append(metric)
            time.sleep(config["monitor"].collect_interval)

    def _save_metrics(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"system_metrics_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(self.metrics, f, indent=2)

def monitor_system():
    monitor = SystemMonitor()
    monitor.start()
    return monitor
