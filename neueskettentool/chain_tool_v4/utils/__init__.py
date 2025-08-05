"""
Utils module initialization
Utility functions and helpers
"""

# Import all utilities
from .debug import debug, profiler, visualizer, time_it
from .performance import performance, optimization, fps_monitor
from .caching import cache_manager, clear_all_caches
from .math_utils import (
    calculate_angle, calculate_distance, 
    get_normal_vector, interpolate_points,
    triangulate_points
)

# Convenience imports
from .debug import DebugLogger, PerformanceProfiler
from .performance import PerformanceMonitor, OptimizationHelper
from .caching import CacheManager, BVHCache

__all__ = [
    # Debug
    'debug',
    'profiler', 
    'visualizer',
    'time_it',
    'DebugLogger',
    'PerformanceProfiler',
    
    # Performance
    'performance',
    'optimization',
    'fps_monitor',
    'PerformanceMonitor',
    'OptimizationHelper',
    
    # Caching
    'cache_manager',
    'clear_all_caches',
    'CacheManager',
    'BVHCache',
    
    # Math utilities
    'calculate_angle',
    'calculate_distance',
    'get_normal_vector',
    'interpolate_points',
    'triangulate_points',
]
