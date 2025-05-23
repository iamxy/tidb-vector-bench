import time
import psutil
import threading
import logging
from typing import Dict, List
from datetime import datetime
import json
from pathlib import Path

from tidb_vector_bench.config.config import config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemMonitor:
    """系统资源监控器"""
    
    def __init__(self, interval: float = 1.0):
        """初始化监控器
        
        Args:
            interval: 采样间隔（秒）
        """
        self.interval = interval
        self.running = False
        self.monitor_thread = None
        self.metrics: List[Dict] = []
        
        # 初始化基准值
        self._init_baseline()
    
    def _init_baseline(self):
        """初始化基准值"""
        self.cpu_baseline = psutil.cpu_percent(interval=1)
        self.mem_baseline = psutil.virtual_memory().percent
        self.disk_baseline = psutil.disk_io_counters()
        self.net_baseline = psutil.net_io_counters()
    
    def _collect_metrics(self) -> Dict:
        """收集系统指标
        
        Returns:
            Dict: 包含各项指标的字典
        """
        # CPU 使用率
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # 内存使用情况
        mem = psutil.virtual_memory()
        
        # 磁盘 I/O
        disk = psutil.disk_io_counters()
        disk_read = (disk.read_bytes - self.disk_baseline.read_bytes) / self.interval
        disk_write = (disk.write_bytes - self.disk_baseline.write_bytes) / self.interval
        
        # 网络 I/O
        net = psutil.net_io_counters()
        net_recv = (net.bytes_recv - self.net_baseline.bytes_recv) / self.interval
        net_sent = (net.bytes_sent - self.net_baseline.bytes_sent) / self.interval
        
        # 更新基准值
        self.disk_baseline = disk
        self.net_baseline = net
        
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu": {
                "percent": cpu_percent,
                "count": psutil.cpu_count(),
                "freq": psutil.cpu_freq().current if psutil.cpu_freq() else None
            },
            "memory": {
                "total": mem.total,
                "available": mem.available,
                "percent": mem.percent,
                "used": mem.used,
                "free": mem.free
            },
            "disk": {
                "read_bytes_per_sec": disk_read,
                "write_bytes_per_sec": disk_write
            },
            "network": {
                "recv_bytes_per_sec": net_recv,
                "sent_bytes_per_sec": net_sent
            }
        }
    
    def _monitor_loop(self):
        """监控循环"""
        while self.running:
            try:
                metrics = self._collect_metrics()
                self.metrics.append(metrics)
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"监控数据收集失败: {e}")
    
    def start(self):
        """启动监控"""
        if self.running:
            return
        
        logger.info("启动系统资源监控...")
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop(self):
        """停止监控"""
        if not self.running:
            return
        
        logger.info("停止系统资源监控...")
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def save_metrics(self, output_dir: Path):
        """保存监控数据
        
        Args:
            output_dir: 输出目录
        """
        if not self.metrics:
            logger.warning("没有监控数据可保存")
            return
        
        # 计算统计信息
        cpu_percents = [m["cpu"]["percent"] for m in self.metrics]
        mem_percents = [m["memory"]["percent"] for m in self.metrics]
        disk_reads = [m["disk"]["read_bytes_per_sec"] for m in self.metrics]
        disk_writes = [m["disk"]["write_bytes_per_sec"] for m in self.metrics]
        net_recvs = [m["network"]["recv_bytes_per_sec"] for m in self.metrics]
        net_sents = [m["network"]["sent_bytes_per_sec"] for m in self.metrics]
        
        stats = {
            "cpu": {
                "mean": sum(cpu_percents) / len(cpu_percents),
                "max": max(cpu_percents),
                "min": min(cpu_percents)
            },
            "memory": {
                "mean": sum(mem_percents) / len(mem_percents),
                "max": max(mem_percents),
                "min": min(mem_percents)
            },
            "disk": {
                "read": {
                    "mean": sum(disk_reads) / len(disk_reads),
                    "max": max(disk_reads),
                    "min": min(disk_reads)
                },
                "write": {
                    "mean": sum(disk_writes) / len(disk_writes),
                    "max": max(disk_writes),
                    "min": min(disk_writes)
                }
            },
            "network": {
                "recv": {
                    "mean": sum(net_recvs) / len(net_recvs),
                    "max": max(net_recvs),
                    "min": min(net_recvs)
                },
                "sent": {
                    "mean": sum(net_sents) / len(net_sents),
                    "max": max(net_sents),
                    "min": min(net_sents)
                }
            }
        }
        
        # 保存详细数据
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = output_dir / f"system_metrics_{timestamp}.json"
        with open(metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=2)
        
        # 保存统计信息
        stats_file = output_dir / f"system_stats_{timestamp}.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"监控数据已保存到: {metrics_file}")
        logger.info(f"统计信息已保存到: {stats_file}")
        
        # 打印统计信息
        logger.info("\n系统资源使用统计:")
        logger.info(f"CPU 使用率: 平均 {stats['cpu']['mean']:.1f}%, 最大 {stats['cpu']['max']:.1f}%")
        logger.info(f"内存使用率: 平均 {stats['memory']['mean']:.1f}%, 最大 {stats['memory']['max']:.1f}%")
        logger.info(f"磁盘读取: 平均 {stats['disk']['read']['mean']/1024/1024:.1f} MB/s, 最大 {stats['disk']['read']['max']/1024/1024:.1f} MB/s")
        logger.info(f"磁盘写入: 平均 {stats['disk']['write']['mean']/1024/1024:.1f} MB/s, 最大 {stats['disk']['write']['max']/1024/1024:.1f} MB/s")
        logger.info(f"网络接收: 平均 {stats['network']['recv']['mean']/1024/1024:.1f} MB/s, 最大 {stats['network']['recv']['max']/1024/1024:.1f} MB/s")
        logger.info(f"网络发送: 平均 {stats['network']['sent']['mean']/1024/1024:.1f} MB/s, 最大 {stats['network']['sent']['max']/1024/1024:.1f} MB/s")

def monitor_system():
    monitor = SystemMonitor()
    monitor.start()
    return monitor
