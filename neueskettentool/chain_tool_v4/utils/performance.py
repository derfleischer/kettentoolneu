"""
Chain Tool V4 - Performance Monitoring
======================================
Performance tracking und Optimierung für Chain Tool
"""

import time
import traceback
from functools import wraps
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict
import bpy

class PerformanceTracker:
    """Track performance metrics for various operations"""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.start_times: Dict[str, float] = {}
        self.call_counts: Dict[str, int] = defaultdict(int)
        self.enabled = True
    
    def start(self, name: str) -> None:
        """Start timing a section"""
        if self.enabled:
            self.start_times[name] = time.perf_counter()
    
    def stop(self, name: str) -> float:
        """Stop timing and record metric"""
        if not self.enabled or name not in self.start_times:
            return 0.0
        
        elapsed = time.perf_counter() - self.start_times[name]
        self.metrics[name].append(elapsed)
        self.call_counts[name] += 1
        del self.start_times[name]
        return elapsed
    
    def get_average(self, name: str) -> float:
        """Get average time for a metric"""
        if name in self.metrics and self.metrics[name]:
            return sum(self.metrics[name]) / len(self.metrics[name])
        return 0.0
    
    def get_total(self, name: str) -> float:
        """Get total time for a metric"""
        if name in self.metrics:
            return sum(self.metrics[name])
        return 0.0
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get comprehensive stats for a metric"""
        if name not in self.metrics or not self.metrics[name]:
            return {}
        
        times = self.metrics[name]
        return {
            'average': sum(times) / len(times),
            'total': sum(times),
            'min': min(times),
            'max': max(times),
            'count': self.call_counts[name],
        }
    
    def clear(self) -> None:
        """Clear all metrics"""
        self.metrics.clear()
        self.start_times.clear()
        self.call_counts.clear()
    
    def report(self) -> str:
        """Generate performance report"""
        if not self.metrics:
            return "No performance data collected"
        
        report_lines = ["Performance Report", "=" * 50]
        
        for name in sorted(self.metrics.keys()):
            stats = self.get_stats(name)
            report_lines.append(
                f"{name}:\n"
                f"  Calls: {stats['count']}\n"
                f"  Total: {stats['total']:.3f}s\n"
                f"  Average: {stats['average']:.3f}s\n"
                f"  Min: {stats['min']:.3f}s\n"
                f"  Max: {stats['max']:.3f}s"
            )
        
        return "\n".join(report_lines)

class PerformanceMonitor:
    """Real-time performance monitoring"""
    
    def __init__(self):
        self.is_monitoring = False
        self.is_monitoring_edges = False
        self.edge_stats: Dict[str, Any] = {}
        self.fps_history: List[float] = []
        self.memory_usage: List[float] = []
        self.last_update = time.time()
    
    def start_monitoring(self) -> None:
        """Start performance monitoring"""
        self.is_monitoring = True
        self.last_update = time.time()
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring"""
        self.is_monitoring = False
    
    def update(self) -> None:
        """Update monitoring data"""
        if not self.is_monitoring:
            return
        
        current_time = time.time()
        delta = current_time - self.last_update
        
        if delta > 0:
            fps = 1.0 / delta
            self.fps_history.append(fps)
            
            # Keep only last 100 samples
            if len(self.fps_history) > 100:
                self.fps_history.pop(0)
        
        self.last_update = current_time
    
    def get_current_fps(self) -> float:
        """Get current FPS"""
        if self.fps_history:
            return self.fps_history[-1]
        return 60.0
    
    def get_average_fps(self) -> float:
        """Get average FPS"""
        if self.fps_history:
            return sum(self.fps_history) / len(self.fps_history)
        return 60.0
    
    def get_edge_performance_stats(self) -> Dict[str, Any]:
        """Get edge detection performance statistics"""
        if not self.edge_stats:
            return {
                'detection_time': 0.0,
                'reinforcement_time': 0.0,
                'memory_usage': 0.0,
                'bvh_efficiency': 100.0,
                'suggestions': []
            }
        return self.edge_stats
    
    def set_edge_stats(self, stats: Dict[str, Any]) -> None:
        """Set edge performance statistics"""
        self.edge_stats = stats

class OptimizationHelper:
    """Helper class for performance optimization"""
    
    @staticmethod
    def should_use_cache(operation: str, data_size: int) -> bool:
        """Determine if caching should be used"""
        # Cache for expensive operations or large data
        expensive_operations = ['triangulation', 'bvh_tree', 'pattern_generation']
        return operation in expensive_operations or data_size > 1000
    
    @staticmethod
    def get_optimal_chunk_size(total_size: int, max_memory_mb: float = 100) -> int:
        """Calculate optimal chunk size for processing"""
        # Assume each item uses approximately 1KB
        item_size_kb = 1
        max_items = int((max_memory_mb * 1024) / item_size_kb)
        
        if total_size <= max_items:
            return total_size
        
        # Find a good chunk size
        for chunk_size in [1000, 500, 250, 100, 50]:
            if chunk_size <= max_items:
                return chunk_size
        
        return 50
    
    @staticmethod
    def optimize_mesh_data(mesh_data: Any) -> Any:
        """Optimize mesh data structure"""
        # Placeholder for mesh optimization
        return mesh_data

def measure_time(func: Callable) -> Callable:
    """
    Decorator to measure function execution time
    
    Usage:
        @measure_time
        def my_function():
            # Function code
            pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time
            
            # Log to console if in debug mode
            if bpy.app.debug:
                print(f"[Performance] {func.__name__} took {elapsed:.3f} seconds")
            
            # Track in global performance tracker
            performance.metrics[func.__name__].append(elapsed)
            performance.call_counts[func.__name__] += 1
            
            return result
            
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            print(f"[Performance] {func.__name__} failed after {elapsed:.3f} seconds")
            print(f"[Error] {str(e)}")
            raise
    
    return wrapper

def time_it(name: Optional[str] = None) -> Callable:
    """
    Context-aware timing decorator
    
    Usage:
        @time_it("custom_name")
        def my_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        metric_name = name or func.__name__
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            performance.start(metric_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                performance.stop(metric_name)
        
        return wrapper
    return decorator

class ProfileContext:
    """Context manager for profiling code blocks"""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start_time
        performance.metrics[self.name].append(elapsed)
        
        if bpy.app.debug:
            print(f"[Profile] {self.name}: {elapsed:.3f}s")

# Global instances
performance = PerformanceTracker()
optimization = OptimizationHelper()
fps_monitor = PerformanceMonitor()

# Convenience function for profiling
def profile(name: str):
    """Convenience function for profiling code blocks
    
    Usage:
        with profile("my_operation"):
            # Code to profile
            pass
    """
    return ProfileContext(name)

# Export all
__all__ = [
    'PerformanceTracker',
    'PerformanceMonitor', 
    'OptimizationHelper',
    'measure_time',
    'time_it',
    'profile',
    'ProfileContext',
    'performance',
    'optimization',
    'fps_monitor',
]
