"""
System Operations - Debug, Performance, Cache Management
"""

import bpy
from bpy.types import Operator
from bpy.props import BoolProperty, EnumProperty
import kettentool.core.constants as constants
import kettentool.core.cache_manager as cache_manager
import kettentool.utils.surface as surface

# =========================================
# DEBUG OPERATORS
# =========================================

class KETTE_OT_toggle_debug(Operator):
    """Toggle debug mode"""
    bl_idname = "kette.toggle_debug"
    bl_label = "Toggle Debug"
    bl_options = {'REGISTER'}
    bl_description = "Toggle debug output"
    
    def execute(self, context):
        props = context.scene.chain_construction_props
        props.debug_enabled = not props.debug_enabled
        
        if constants.debug:
            constants.debug.enabled = props.debug_enabled
            status = "enabled" if props.debug_enabled else "disabled"
            self.report({'INFO'}, f"Debug mode {status}")
        
        return {'FINISHED'}

class KETTE_OT_clear_debug_logs(Operator):
    """Clear debug logs"""
    bl_idname = "kette.clear_debug_logs"
    bl_label = "Clear Debug Logs"
    bl_options = {'REGISTER'}
    bl_description = "Clear all debug logs"
    
    def execute(self, context):
        if constants.debug:
            constants.debug.clear()
            self.report({'INFO'}, "Debug logs cleared")
        else:
            self.report({'WARNING'}, "Debug system not initialized")
        
        return {'FINISHED'}

class KETTE_OT_export_debug_logs(Operator):
    """Export debug logs"""
    bl_idname = "kette.export_debug_logs"
    bl_label = "Export Debug Logs"
    bl_options = {'REGISTER'}
    bl_description = "Export debug logs to text block"
    
    def execute(self, context):
        if not constants.debug:
            self.report({'WARNING'}, "Debug system not initialized")
            return {'CANCELLED'}
        
        # Create text block
        text_name = "KettenTool_Debug_Log"
        if text_name in bpy.data.texts:
            text = bpy.data.texts[text_name]
            text.clear()
        else:
            text = bpy.data.texts.new(text_name)
        
        # Write logs
        for log in constants.debug.get_logs():
            line = f"[{log['level']}] {log['category']}: {log['message']}\n"
            text.write(line)
        
        self.report({'INFO'}, f"Debug logs exported to {text_name}")
        return {'FINISHED'}

# =========================================
# PERFORMANCE OPERATORS
# =========================================

class KETTE_OT_performance_report(Operator):
    """Show performance report"""
    bl_idname = "kette.performance_report"
    bl_label = "Performance Report"
    bl_options = {'REGISTER'}
    bl_description = "Show performance statistics"
    
    def execute(self, context):
        if not constants.performance_monitor:
            self.report({'WARNING'}, "Performance monitor not initialized")
            return {'CANCELLED'}
        
        report = constants.performance_monitor.get_report()
        
        # Show in info area
        for line in report.split('\n'):
            self.report({'INFO'}, line)
        
        return {'FINISHED'}

class KETTE_OT_reset_performance(Operator):
    """Reset performance counters"""
    bl_idname = "kette.reset_performance"
    bl_label = "Reset Performance"
    bl_options = {'REGISTER'}
    bl_description = "Reset all performance counters"
    
    def execute(self, context):
        if constants.performance_monitor:
            constants.performance_monitor.reset_all()
            self.report({'INFO'}, "Performance counters reset")
        
        return {'FINISHED'}

# =========================================
# CACHE OPERATORS
# =========================================

class KETTE_OT_clear_caches(Operator):
    """Clear all caches"""
    bl_idname = "kette.clear_caches"
    bl_label = "Clear Caches"
    bl_options = {'REGISTER'}
    bl_description = "Clear all internal caches"
    
    def execute(self, context):
        # Clear all caches
        cache_manager.cleanup_all_caches()
        surface.clear_auras()
        
        self.report({'INFO'}, "All caches cleared")
        return {'FINISHED'}

class KETTE_OT_cache_statistics(Operator):
    """Show cache statistics"""
    bl_idname = "kette.cache_statistics"
    bl_label = "Cache Statistics"
    bl_options = {'REGISTER'}
    bl_description = "Show cache usage statistics"
    
    def execute(self, context):
        stats = cache_manager.get_cache_statistics()
        
        # Report statistics
        self.report({'INFO'}, f"Total caches: {stats['total_caches']}")
        self.report({'INFO'}, f"Total entries: {stats['total_entries']}")
        
        for cache_name, cache_stats in stats['caches'].items():
            self.report({'INFO'}, f"  {cache_name}: {cache_stats['size']} entries")
        
        return {'FINISHED'}

class KETTE_OT_cleanup_invalid(Operator):
    """Cleanup invalid references"""
    bl_idname = "kette.cleanup_invalid"
    bl_label = "Cleanup Invalid"
    bl_options = {'REGISTER'}
    bl_description = "Remove invalid object references from caches"
    
    def execute(self, context):
        removed = cache_manager.cleanup_invalid_references()
        
        self.report({'INFO'}, f"Removed {removed} invalid references")
        return {'FINISHED'}

# =========================================
# SYSTEM INFO OPERATORS
# =========================================

class KETTE_OT_system_info(Operator):
    """Show system information"""
    bl_idname = "kette.system_info"
    bl_label = "System Info"
    bl_options = {'REGISTER'}
    bl_description = "Show KettenTool system information"
    
    def execute(self, context):
        # Gather statistics
        sphere_count = len(constants.get_sphere_registry())
        connector_count = len(constants.get_connector_registry())
        aura_count = len(constants.get_aura_cache())
        
        # Report info
        self.report({'INFO'}, "=== KettenTool System Info ===")
        self.report({'INFO'}, f"Spheres registered: {sphere_count}")
        self.report({'INFO'}, f"Connectors registered: {connector_count}")
        self.report({'INFO'}, f"Auras cached: {aura_count}")
        self.report({'INFO'}, f"Debug enabled: {constants.debug.enabled if constants.debug else False}")
        
        return {'FINISHED'}

# =========================================
# REGISTRATION
# =========================================

classes = [
    KETTE_OT_toggle_debug,
    KETTE_OT_clear_debug_logs,
    KETTE_OT_export_debug_logs,
    KETTE_OT_performance_report,
    KETTE_OT_reset_performance,
    KETTE_OT_clear_caches,
    KETTE_OT_cache_statistics,
    KETTE_OT_cleanup_invalid,
    KETTE_OT_system_info,
]

def register():
    """Register system operators"""
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    """Unregister system operators"""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
