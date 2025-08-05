"""
Performance Monitoring for Chain Tool V4
Track and optimize performance
"""

import time
import functools
import gc
from typing import Dict, List, Callable, Any
import bpy

from utils.debug import debug

# ============================================
# PERFORMANCE MONITOR
# ============================================

class PerformanceMonitor:
    """Advanced performance monitoring system"""
    
    def __init__(self):
        self.metrics = {}
        self.active_timers = {}
        self.memory_snapshots = []
        self.operation_history = []
        
    def start_operation(self, name: str, category: str = 'GENERAL'):
        """Start tracking an operation"""
        key = f"{category}::{name}"
        
        self.active_timers[key] = {
            'start_time': time.time(),
            'category': category,
            'name': name,
            'memory_start': self._get_memory_usage()
        }
        
        debug.trace('PERFORMANCE', f"Started tracking: {name}", category=category)
        
    def end_operation(self, name: str, category: str = 'GENERAL') -> Dict[str, Any]:
        """End tracking and return metrics"""
        key = f"{category}::{name}"
        
        if key not in self.active_timers:
            debug.warning('PERFORMANCE', f"Operation not tracked: {name}")
            return {}
            
        timer_data = self.active_timers[key]
        elapsed = time.time() - timer_data['start_time']
        memory_end = self._get_memory_usage()
        memory_delta = memory_end - timer_data['memory_start']
        
        # Store metrics
        metrics = {
            'name': name,
            'category': category,
            'elapsed_time': elapsed,
            'memory_delta': memory_delta,
            'timestamp': time.time()
        }
        
        # Update statistics
        if key not in self.metrics:
            self.metrics[key] = {
                'count': 0,
                'total_time': 0,
                'avg_time': 0,
                'min_time': float('inf'),
                'max_time': 0,
                'memory_impact': 0
            }
            
        stats = self.metrics[key]
        stats['count'] += 1
        stats['total_time'] += elapsed
        stats['avg_time'] = stats['total_time'] / stats['count']
        stats['min_time'] = min(stats['min_time'], elapsed)
        stats['max_time'] = max(stats['max_time'], elapsed)
        stats['memory_impact'] += memory_delta
        
        # Add to history
        self.operation_history.append(metrics)
        if len(self.operation_history) > 1000:
            self.operation_history = self.operation_history[-500:]
            
        # Clean up
        del self.active_timers[key]
        
        # Log if slow
        if elapsed > 1.0:
            debug.warning('PERFORMANCE', 
                         f"Slow operation: {name} took {elapsed:.2f}s",
                         category=category, memory_delta=f"{memory_delta:.2f}MB")
        elif elapsed > 0.1:
            debug.info('PERFORMANCE', 
                      f"Operation complete: {name}",
                      elapsed=f"{elapsed:.3f}s")
                      
        return metrics
        
    def track_function(self, category: str = 'GENERAL'):
        """Decorator to track function performance"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                func_name = func.__name__
                
                # Start tracking
                self.start_operation(func_name, category)
                
                try:
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # End tracking
                    metrics = self.end_operation(func_name, category)
                    
                    # Add result info if available
                    if hasattr(result, '__len__') and not isinstance(result, str):
                        debug.trace('PERFORMANCE', 
                                   f"Function returned {len(result)} items",
                                   function=func_name)
                                   
                    return result
                    
                except Exception as e:
                    # Still end tracking on error
                    self.end_operation(func_name, category)
                    debug.error('PERFORMANCE', 
                               f"Function failed: {func_name}",
                               error=str(e))
                    raise
                    
            return wrapper
        return decorator
        
    def batch_operation(self, items: List[Any], operation: Callable,
                       chunk_size: int = 100, name: str = "Batch Operation"):
        """Process items in optimized batches"""
        total_items = len(items)
        
        if total_items == 0:
            return []
            
        self.start_operation(name, 'BATCH')
        results = []
        
        try:
            # Process in chunks
            for i in range(0, total_items, chunk_size):
                chunk = items[i:i + chunk_size]
                chunk_results = []
                
                # Process chunk
                for item in chunk:
                    result = operation(item)
                    chunk_results.append(result)
                    
                results.extend(chunk_results)
                
                # Progress update
                progress = (i + len(chunk)) / total_items
                if progress < 1.0:
                    debug.trace('PERFORMANCE', 
                               f"Batch progress: {progress:.1%}",
                               operation=name)
                               
                # Allow UI updates
                if i % (chunk_size * 10) == 0:
                    bpy.context.view_layer.update()
                    
            self.end_operation(name, 'BATCH')
            return results
            
        except Exception as e:
            self.end_operation(name, 'BATCH')
            raise
            
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        # This is a simplified version
        # In production, use psutil or resource module
        gc.collect()
        
        # Count Blender objects as proxy for memory
        memory_estimate = 0
        memory_estimate += len(bpy.data.meshes) * 0.1
        memory_estimate += len(bpy.data.objects) * 0.01
        memory_estimate += len(bpy.data.materials) * 0.05
        
        return memory_estimate
        
    def get_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        report = {
            'total_operations': sum(m['count'] for m in self.metrics.values()),
            'total_time': sum(m['total_time'] for m in self.metrics.values()),
            'slowest_operations': [],
            'most_frequent': [],
            'memory_impact': sum(m['memory_impact'] for m in self.metrics.values())
        }
        
        # Find slowest operations
        sorted_by_avg = sorted(self.metrics.items(), 
                              key=lambda x: x[1]['avg_time'], 
                              reverse=True)
        report['slowest_operations'] = [
            {
                'name': k.split('::')[1],
                'category': k.split('::')[0],
                'avg_time': v['avg_time'],
                'count': v['count']
            }
            for k, v in sorted_by_avg[:5]
        ]
        
        # Find most frequent
        sorted_by_count = sorted(self.metrics.items(),
                                key=lambda x: x[1]['count'],
                                reverse=True)
        report['most_frequent'] = [
            {
                'name': k.split('::')[1],
                'category': k.split('::')[0],
                'count': v['count'],
                'total_time': v['total_time']
            }
            for k, v in sorted_by_count[:5]
        ]
        
        return report
        
    def reset_metrics(self):
        """Reset all performance metrics"""
        self.metrics.clear()
        self.active_timers.clear()
        self.operation_history.clear()
        debug.info('PERFORMANCE', "Performance metrics reset")

# ============================================
# OPTIMIZATION HELPERS
# ============================================

class OptimizationHelper:
    """Performance optimization utilities"""
    
    @staticmethod
    def should_use_background(vertex_count: int) -> bool:
        """Check if operation should run in background"""
        from config import MAX_VERTICES_REALTIME
        return vertex_count > MAX_VERTICES_REALTIME
        
    @staticmethod
    def optimize_mesh_access(mesh: bpy.types.Mesh):
        """Optimize mesh for faster access"""
        # Ensure mesh is in object mode for faster access
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
            
        # Update mesh data
        mesh.update()
        
        # Calculate normals if needed
        if not mesh.has_custom_normals:
            mesh.calc_normals()
            
    @staticmethod
    def create_bmesh_optimized(mesh: bpy.types.Mesh):
        """Create optimized BMesh"""
        import bmesh
        
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        
        return bm
        
    @staticmethod
    @functools.lru_cache(maxsize=128)
    def cached_calculation(key: str, calculation: Callable) -> Any:
        """Cache expensive calculations"""
        debug.trace('PERFORMANCE', f"Using cached result for: {key}")
        return calculation()

# ============================================
# FRAME RATE MONITOR
# ============================================

class FrameRateMonitor:
    """Monitor viewport frame rate"""
    
    def __init__(self):
        self.last_time = time.time()
        self.frame_times = []
        self.is_monitoring = False
        
    def update(self):
        """Update frame rate calculation"""
        current_time = time.time()
        frame_time = current_time - self.last_time
        self.last_time = current_time
        
        self.frame_times.append(frame_time)
        
        # Keep only recent frames
        if len(self.frame_times) > 60:
            self.frame_times = self.frame_times[-60:]
            
    def get_fps(self) -> float:
        """Get current FPS"""
        if not self.frame_times:
            return 0.0
            
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        
    def is_low_fps(self, threshold: float = 15.0) -> bool:
        """Check if FPS is below threshold"""
        return self.get_fps() < threshold

# ============================================
# GLOBAL INSTANCE
# ============================================

# Create global performance monitor
performance = PerformanceMonitor()
optimization = OptimizationHelper()
fps_monitor = FrameRateMonitor()

# ============================================
# OPERATORS
# ============================================

class CHAIN_TOOL_OT_performance_report(bpy.types.Operator):
    """Show performance report"""
    bl_idname = "chain_tool.performance_report"
    bl_label = "Performance Report"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        report = performance.get_report()
        
        self.report({'INFO'}, f"Total operations: {report['total_operations']}")
        self.report({'INFO'}, f"Total time: {report['total_time']:.2f}s")
        
        # Show slowest operations
        for op in report['slowest_operations']:
            self.report({'INFO'}, 
                       f"Slow: {op['name']} - {op['avg_time']:.3f}s avg")
                       
        return {'FINISHED'}
        
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=400)
        
    def draw(self, context):
        layout = self.layout
        report = performance.get_report()
        
        # Summary
        box = layout.box()
        box.label(text="Performance Summary", icon='TIME')
        col = box.column(align=True)
        col.label(text=f"Total Operations: {report['total_operations']}")
        col.label(text=f"Total Time: {report['total_time']:.2f}s")
        col.label(text=f"Memory Impact: {report['memory_impact']:.2f}MB")
        
        # Slowest operations
        if report['slowest_operations']:
            box = layout.box()
            box.label(text="Slowest Operations", icon='SORTTIME')
            for op in report['slowest_operations'][:3]:
                row = box.row()
                row.label(text=f"{op['name']}:")
                row.label(text=f"{op['avg_time']:.3f}s")

# Classes to register
CLASSES = [
    CHAIN_TOOL_OT_performance_report,
]
