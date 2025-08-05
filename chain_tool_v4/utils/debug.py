"""
Chain Tool V4 - Debug Utilities
===============================
Debug und Visualisierungs-Tools für Entwicklung
"""

import bpy
import time
import traceback
from typing import Any, Optional, List, Tuple
from functools import wraps
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

class DebugLogger:
    """Advanced debug logging system"""
    
    LEVELS = {
        'ERROR': 0,
        'WARNING': 1,
        'INFO': 2,
        'DEBUG': 3,
        'TRACE': 4
    }
    
    def __init__(self):
        self.enabled = bpy.app.debug
        self.level = 'INFO'
        self.log_history: List[Tuple[str, str, float]] = []
        self.max_history = 1000
    
    def set_level(self, level: str) -> None:
        """Set debug level"""
        if level in self.LEVELS:
            self.level = level
    
    def should_log(self, level: str) -> bool:
        """Check if message should be logged"""
        if not self.enabled:
            return False
        return self.LEVELS.get(level, 0) <= self.LEVELS.get(self.level, 2)
    
    def log(self, level: str, message: str, category: Optional[str] = None) -> None:
        """Log a message"""
        if not self.should_log(level):
            return
        
        timestamp = time.time()
        
        # Format message
        if category:
            formatted = f"[{level}] [{category}] {message}"
        else:
            formatted = f"[{level}] {message}"
        
        # Print to console
        print(f"Chain Tool V4: {formatted}")
        
        # Store in history
        self.log_history.append((level, formatted, timestamp))
        
        # Limit history size
        if len(self.log_history) > self.max_history:
            self.log_history = self.log_history[-self.max_history:]
    
    def error(self, message: str, category: Optional[str] = None) -> None:
        """Log error message"""
        self.log('ERROR', message, category)
    
    def warning(self, message: str, category: Optional[str] = None) -> None:
        """Log warning message"""
        self.log('WARNING', message, category)
    
    def info(self, message: str, category: Optional[str] = None) -> None:
        """Log info message"""
        self.log('INFO', message, category)
    
    def debug(self, message: str, category: Optional[str] = None) -> None:
        """Log debug message"""
        self.log('DEBUG', message, category)
    
    def trace(self, message: str, category: Optional[str] = None) -> None:
        """Log trace message"""
        self.log('TRACE', message, category)
    
    def log_error(self, error: Exception, context: Optional[str] = None) -> None:
        """Log exception with traceback"""
        error_msg = str(error)
        tb = traceback.format_exc()
        
        if context:
            self.error(f"{context}: {error_msg}")
        else:
            self.error(error_msg)
        
        if self.should_log('DEBUG'):
            print("Traceback:")
            print(tb)
    
    def clear_history(self) -> None:
        """Clear log history"""
        self.log_history.clear()
    
    def get_history(self, level: Optional[str] = None) -> List[Tuple[str, str, float]]:
        """Get log history, optionally filtered by level"""
        if level and level in self.LEVELS:
            min_level = self.LEVELS[level]
            return [
                (l, msg, t) for l, msg, t in self.log_history
                if self.LEVELS.get(l, 0) <= min_level
            ]
        return self.log_history.copy()

class DebugManager:
    """Central debug management"""
    
    def __init__(self):
        self.logger = DebugLogger()
        self.profiler = PerformanceProfiler()
        self.visualizer = DebugVisualizer()
        self.enabled = bpy.app.debug
    
    def enable(self) -> None:
        """Enable debug mode"""
        self.enabled = True
        self.logger.enabled = True
        bpy.app.debug = True
    
    def disable(self) -> None:
        """Disable debug mode"""
        self.enabled = False
        self.logger.enabled = False
        bpy.app.debug = False
    
    def log(self, message: str, level: str = 'INFO') -> None:
        """Log message"""
        self.logger.log(level, message)
    
    def error(self, message: str) -> None:
        """Log error"""
        self.logger.error(message)
    
    def warning(self, message: str) -> None:
        """Log warning"""
        self.logger.warning(message)
    
    def info(self, message: str) -> None:
        """Log info"""
        self.logger.info(message)
    
    def debug(self, message: str) -> None:
        """Log debug"""
        self.logger.debug(message)

class PerformanceProfiler:
    """Performance profiling tools"""
    
    def __init__(self):
        self.timings: dict = {}
        self.enabled = True
    
    def start(self, name: str) -> None:
        """Start timing a section"""
        if self.enabled:
            self.timings[name] = time.perf_counter()
    
    def stop(self, name: str) -> float:
        """Stop timing and return elapsed time"""
        if self.enabled and name in self.timings:
            elapsed = time.perf_counter() - self.timings[name]
            del self.timings[name]
            return elapsed
        return 0.0
    
    def profile(self, name: str):
        """Context manager for profiling"""
        class ProfileContext:
            def __init__(self, profiler, name):
                self.profiler = profiler
                self.name = name
            
            def __enter__(self):
                self.profiler.start(self.name)
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                elapsed = self.profiler.stop(self.name)
                if self.profiler.enabled:
                    print(f"[Profile] {self.name}: {elapsed:.4f}s")
        
        return ProfileContext(self, name)

class DebugVisualizer:
    """3D viewport debug visualization"""
    
    def __init__(self):
        self.draw_handlers = []
        self.debug_objects = []
    
    def draw_sphere(self, location: Vector, radius: float = 0.1, 
                   color: Tuple[float, float, float, float] = (1, 0, 0, 1)) -> None:
        """Draw debug sphere in viewport"""
        # Create temporary sphere object for visualization
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=radius,
            location=location,
            segments=8,
            ring_count=6
        )
        
        sphere = bpy.context.active_object
        sphere.name = "Debug_Sphere"
        sphere.show_wire = True
        
        # Set color
        mat = bpy.data.materials.new(name="Debug_Material")
        mat.diffuse_color = color
        sphere.data.materials.append(mat)
        
        self.debug_objects.append(sphere)
    
    def draw_line(self, start: Vector, end: Vector,
                 color: Tuple[float, float, float, float] = (0, 1, 0, 1)) -> None:
        """Draw debug line in viewport"""
        # Create curve for line
        curve = bpy.data.curves.new(name="Debug_Line", type='CURVE')
        curve.dimensions = '3D'
        
        spline = curve.splines.new(type='POLY')
        spline.points.add(1)  # We need 2 points total
        
        spline.points[0].co = (*start, 1)
        spline.points[1].co = (*end, 1)
        
        # Create object
        line_obj = bpy.data.objects.new("Debug_Line", curve)
        bpy.context.collection.objects.link(line_obj)
        
        # Set color
        mat = bpy.data.materials.new(name="Debug_Line_Material")
        mat.diffuse_color = color
        line_obj.data.materials.append(mat)
        
        self.debug_objects.append(line_obj)
    
    def clear_debug_objects(self) -> None:
        """Remove all debug visualization objects"""
        for obj in self.debug_objects:
            if obj.name in bpy.data.objects:
                bpy.data.objects.remove(obj, do_unlink=True)
        self.debug_objects.clear()

class DebugSystem:
    """Legacy compatibility class"""
    
    def __init__(self):
        self.manager = DebugManager()
    
    def log(self, message: str, level: str = 'INFO') -> None:
        """Log message"""
        self.manager.log(message, level)

def debug_decorator(func):
    """Decorator for debug logging function calls"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        debug_manager = DebugManager()
        
        if debug_manager.enabled:
            debug_manager.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            
            if debug_manager.enabled:
                debug_manager.debug(f"{func.__name__} returned: {result}")
            
            return result
            
        except Exception as e:
            debug_manager.logger.log_error(e, f"Error in {func.__name__}")
            raise
    
    return wrapper

def time_it(func):
    """Decorator to time function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        
        if bpy.app.debug:
            print(f"[Timing] {func.__name__}: {elapsed:.4f}s")
        
        return result
    
    return wrapper

# Global instances
debug = DebugLogger()
profiler = PerformanceProfiler()
visualizer = DebugVisualizer()

# Export
__all__ = [
    'DebugLogger',
    'DebugManager',
    'PerformanceProfiler',
    'DebugVisualizer',
    'DebugSystem',
    'debug_decorator',
    'time_it',
    'debug',
    'profiler',
    'visualizer',
]
