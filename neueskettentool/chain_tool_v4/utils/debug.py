"""
Debug System for Chain Tool V4
Comprehensive logging and debugging utilities
"""

import bpy
import time
import functools
from datetime import datetime
from typing import Any, Callable, Dict, List

from config import DEBUG_ENABLED, DEBUG_LEVEL, DEBUG_CATEGORIES

# ============================================
# DEBUG LOGGER
# ============================================

class DebugLogger:
    """Advanced debug logging system with categories and levels"""
    
    LEVELS = {
        'ERROR': 0,
        'WARNING': 1,
        'INFO': 2,
        'DEBUG': 3,
        'TRACE': 4
    }
    
    # ANSI color codes for terminal output
    COLORS = {
        'ERROR': '\033[91m',      # Red
        'WARNING': '\033[93m',    # Yellow
        'INFO': '\033[92m',       # Green
        'DEBUG': '\033[94m',      # Blue
        'TRACE': '\033[95m',      # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    # Icons for each level
    ICONS = {
        'ERROR': '❌',
        'WARNING': '⚠️',
        'INFO': '✅',
        'DEBUG': '🔧',
        'TRACE': '🔍'
    }
    
    def __init__(self):
        self.enabled = DEBUG_ENABLED
        self.level = DEBUG_LEVEL
        self.categories = DEBUG_CATEGORIES.copy()
        self.log_history = []
        self.performance_data = {}
        
    def log(self, level: str, category: str, message: str, **kwargs):
        """Log a message with level and category filtering"""
        if not self.enabled:
            return
            
        # Check level
        if self.LEVELS.get(level, 5) > self.LEVELS.get(self.level, 2):
            return
            
        # Check category
        if category not in self.categories or not self.categories[category]:
            return
        
        # Format timestamp
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Build log message
        icon = self.ICONS.get(level, '📝')
        color = self.COLORS.get(level, '')
        reset = self.COLORS['RESET']
        
        # Format the message
        formatted_msg = f"{color}[{timestamp}] {icon} [{level}] [{category}] {message}"
        
        # Add extra data if provided
        if kwargs:
            extra_data = " | " + " | ".join(f"{k}={v}" for k, v in kwargs.items())
            formatted_msg += extra_data
            
        formatted_msg += reset
        
        # Print to console
        print(formatted_msg)
        
        # Store in history
        self.log_history.append({
            'timestamp': timestamp,
            'level': level,
            'category': category,
            'message': message,
            'extra': kwargs
        })
        
        # Limit history size
        if len(self.log_history) > 1000:
            self.log_history = self.log_history[-500:]
    
    def error(self, category: str, message: str, **kwargs):
        """Log error message"""
        self.log('ERROR', category, message, **kwargs)
        
    def warning(self, category: str, message: str, **kwargs):
        """Log warning message"""
        self.log('WARNING', category, message, **kwargs)
        
    def info(self, category: str, message: str, **kwargs):
        """Log info message"""
        self.log('INFO', category, message, **kwargs)
        
    def debug(self, category: str, message: str, **kwargs):
        """Log debug message"""
        self.log('DEBUG', category, message, **kwargs)
        
    def trace(self, category: str, message: str, **kwargs):
        """Log trace message"""
        self.log('TRACE', category, message, **kwargs)
    
    def set_level(self, level: str):
        """Change debug level"""
        if level in self.LEVELS:
            self.level = level
            self.info('GENERAL', f"Debug level set to {level}")
    
    def enable_category(self, category: str, enabled: bool = True):
        """Enable/disable a debug category"""
        self.categories[category] = enabled
        status = "enabled" if enabled else "disabled"
        self.info('GENERAL', f"Category '{category}' {status}")
    
    def get_history(self, level: str = None, category: str = None) -> List[Dict]:
        """Get filtered log history"""
        history = self.log_history
        
        if level:
            history = [h for h in history if h['level'] == level]
            
        if category:
            history = [h for h in history if h['category'] == category]
            
        return history
    
    def clear_history(self):
        """Clear log history"""
        self.log_history.clear()
        self.info('GENERAL', "Log history cleared")
    
    def report_to_blender(self, operator, level: str = 'INFO'):
        """Report last messages to Blender operator"""
        if not operator:
            return
            
        # Get recent messages of specified level
        recent = self.get_history(level=level)[-5:]  # Last 5 messages
        
        for entry in recent:
            msg = f"[{entry['category']}] {entry['message']}"
            
            if level == 'ERROR':
                operator.report({'ERROR'}, msg)
            elif level == 'WARNING':
                operator.report({'WARNING'}, msg)
            else:
                operator.report({'INFO'}, msg)

# ============================================
# PERFORMANCE PROFILER
# ============================================

class PerformanceProfiler:
    """Performance profiling utilities"""
    
    def __init__(self, logger: DebugLogger):
        self.logger = logger
        self.timers = {}
        self.counters = {}
        
    def start_timer(self, name: str):
        """Start a named timer"""
        self.timers[name] = time.time()
        self.logger.trace('PERFORMANCE', f"Timer '{name}' started")
        
    def end_timer(self, name: str) -> float:
        """End a timer and return elapsed time"""
        if name not in self.timers:
            self.logger.warning('PERFORMANCE', f"Timer '{name}' was not started")
            return 0.0
            
        elapsed = time.time() - self.timers[name]
        del self.timers[name]
        
        self.logger.debug('PERFORMANCE', f"Timer '{name}' completed", 
                         elapsed=f"{elapsed:.3f}s")
        return elapsed
    
    def increment_counter(self, name: str, amount: int = 1):
        """Increment a named counter"""
        if name not in self.counters:
            self.counters[name] = 0
        self.counters[name] += amount
        
    def get_counter(self, name: str) -> int:
        """Get counter value"""
        return self.counters.get(name, 0)
    
    def reset_counters(self):
        """Reset all counters"""
        self.counters.clear()
        self.logger.debug('PERFORMANCE', "Counters reset")
    
    def profile_function(self, category: str = 'PERFORMANCE'):
        """Decorator to profile function execution time"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                func_name = func.__name__
                self.start_timer(func_name)
                
                try:
                    result = func(*args, **kwargs)
                    elapsed = self.end_timer(func_name)
                    
                    if elapsed > 0.1:  # Only log if > 100ms
                        self.logger.info(category, 
                                       f"Function '{func_name}' took {elapsed:.3f}s")
                    
                    return result
                    
                except Exception as e:
                    self.end_timer(func_name)
                    self.logger.error(category, 
                                    f"Function '{func_name}' failed: {str(e)}")
                    raise
                    
            return wrapper
        return decorator

# ============================================
# DEBUG VISUALIZER
# ============================================

class DebugVisualizer:
    """Visual debugging helpers"""
    
    def __init__(self, logger: DebugLogger):
        self.logger = logger
        self.debug_objects = []
        
    def create_debug_sphere(self, location: tuple, radius: float = 0.1, 
                           name: str = "Debug_Sphere", color: tuple = (1, 0, 0, 1)):
        """Create a debug sphere at location"""
        # Create mesh
        mesh = bpy.data.meshes.new(name)
        obj = bpy.data.objects.new(name, mesh)
        
        # Generate sphere geometry
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=8, radius=radius)
        bm.to_mesh(mesh)
        bm.free()
        
        # Set location
        obj.location = location
        
        # Set color
        mat = bpy.data.materials.new(name=f"{name}_Mat")
        mat.use_nodes = True
        mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (*color[:3], 1)
        obj.data.materials.append(mat)
        
        # Link to scene
        bpy.context.collection.objects.link(obj)
        
        self.debug_objects.append(obj)
        self.logger.trace('GENERAL', f"Created debug sphere at {location}")
        
        return obj
    
    def create_debug_line(self, start: tuple, end: tuple, name: str = "Debug_Line"):
        """Create a debug line between two points"""
        # Create curve
        curve = bpy.data.curves.new(name, 'CURVE')
        curve.dimensions = '3D'
        
        # Create spline
        spline = curve.splines.new('POLY')
        spline.points.add(1)  # Total 2 points
        
        spline.points[0].co = (*start, 1)
        spline.points[1].co = (*end, 1)
        
        # Create object
        obj = bpy.data.objects.new(name, curve)
        bpy.context.collection.objects.link(obj)
        
        self.debug_objects.append(obj)
        self.logger.trace('GENERAL', f"Created debug line from {start} to {end}")
        
        return obj
    
    def clear_debug_objects(self):
        """Remove all debug objects"""
        for obj in self.debug_objects:
            bpy.data.objects.remove(obj, do_unlink=True)
            
        self.debug_objects.clear()
        self.logger.debug('GENERAL', "Debug objects cleared")

# ============================================
# GLOBAL INSTANCES
# ============================================

# Create global instances
debug = DebugLogger()
profiler = PerformanceProfiler(debug)
visualizer = DebugVisualizer(debug)

# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

def log_operator_start(operator_name: str):
    """Log operator start"""
    debug.info('GENERAL', f"Operator '{operator_name}' started")
    profiler.start_timer(operator_name)

def log_operator_end(operator_name: str, status: str = 'FINISHED'):
    """Log operator end"""
    elapsed = profiler.end_timer(operator_name)
    debug.info('GENERAL', f"Operator '{operator_name}' {status}", 
               elapsed=f"{elapsed:.3f}s")

def time_it(category: str = 'PERFORMANCE'):
    """Decorator for timing functions"""
    return profiler.profile_function(category)

# ============================================
# DEBUG OPERATOR
# ============================================

class CHAIN_TOOL_OT_toggle_debug(bpy.types.Operator):
    """Toggle debug mode"""
    bl_idname = "chain_tool.toggle_debug"
    bl_label = "Toggle Debug"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        debug.enabled = not debug.enabled
        status = "enabled" if debug.enabled else "disabled"
        self.report({'INFO'}, f"Debug mode {status}")
        return {'FINISHED'}

class CHAIN_TOOL_OT_clear_debug(bpy.types.Operator):
    """Clear debug objects and logs"""
    bl_idname = "chain_tool.clear_debug"
    bl_label = "Clear Debug"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        visualizer.clear_debug_objects()
        debug.clear_history()
        self.report({'INFO'}, "Debug data cleared")
        return {'FINISHED'}

# Classes to register
CLASSES = [
    CHAIN_TOOL_OT_toggle_debug,
    CHAIN_TOOL_OT_clear_debug,
]
