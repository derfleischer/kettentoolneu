"""
Settings and Debug Panels for Chain Construction
Configuration and debugging tools
"""

import bpy
from bpy.types import Panel


class KETTE_PT_settings(Panel):
    """Settings panel"""
    bl_label = "Settings"
    bl_idname = "KETTE_PT_settings"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tools"
    bl_parent_id = "KETTE_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.chain_props
        
        # Sphere Settings
        col = layout.column(align=True)
        col.label(text="Sphere Defaults:", icon='SPHERE')
        
        box = col.box()
        box_col = box.column(align=True)
        
        # Size settings
        row = box_col.row(align=True)
        row.prop(props, "sphere_radius", text="Radius")
        row.operator("kette.reset_sphere_size", text="", icon='LOOP_BACK')
        
        box_col.prop(props, "sphere_subdivisions", text="Subdivisions")
        
        # Material settings
        box_col.separator()
        box_col.prop(props, "sphere_material", text="Material")
        
        if props.sphere_material == 'CUSTOM':
            box_col.prop(props, "sphere_custom_material", text="")
        
        # Connector Settings
        col.separator()
        col.label(text="Connector Defaults:", icon='LINK_DATA')
        
        box = col.box()
        box_col = box.column(align=True)
        
        box_col.prop(props, "connector_segments", text="Segments")
        
        row = box_col.row(align=True)
        row.prop(props, "connector_radius", text="Radius")
        row.operator("kette.match_connector_radius", text="", icon='EYEDROPPER')
        
        box_col.separator()
        box_col.prop(props, "connector_profile", text="Profile")
        
        if props.connector_profile == 'CUSTOM':
            box_col.prop(props, "connector_custom_curve", text="")
        
        # Performance Settings
        col.separator()
        col.label(text="Performance:", icon='MEMORY')
        
        box = col.box()
        box_col = box.column(align=True)
        
        box_col.prop(props, "viewport_display_mode", text="Display")
        
        if props.viewport_display_mode == 'ADAPTIVE':
            box_col.prop(props, "adaptive_threshold", text="Threshold")
        
        box_col.separator()
        box_col.prop(props, "use_instancing", text="Use Instancing")
        
        if props.use_instancing:
            box_col.prop(props, "instance_collection", text="Collection")
        
        # Display Settings
        col.separator()
        col.label(text="Display:", icon='HIDE_OFF')
        
        box = col.box()
        box_col = box.column(align=True)
        
        box_col.prop(props, "show_debug_info", text="Debug Info")
        box_col.prop(props, "show_overlays", text="Show Overlays")
        
        if props.show_overlays:
            box_col.prop(props, "overlay_color", text="")
            box_col.prop(props, "overlay_alpha", text="Alpha")
        
        box_col.separator()
        box_col.prop(props, "auto_update", text="Auto Update")
        
        if props.auto_update:
            box_col.prop(props, "update_delay", text="Delay (ms)")
        
        # Export/Import
        col.separator()
        col.label(text="Data:", icon='FILE')
        
        row = col.row(align=True)
        row.operator("kette.export_chain", text="Export", icon='EXPORT')
        row.operator("kette.import_chain", text="Import", icon='IMPORT')


class KETTE_PT_debug(Panel):
    """Debug panel"""
    bl_label = "Debug"
    bl_idname = "KETTE_PT_debug"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tools"
    bl_parent_id = "KETTE_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        """Only show debug panel if debug mode is enabled"""
        return context.scene.chain_props.show_debug_info
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.chain_props
        
        # Debug Visualization
        col = layout.column(align=True)
        col.label(text="Visualization:", icon='MODIFIER_ON')
        
        box = col.box()
        box_col = box.column(align=True)
        
        box_col.prop(props, "debug_show_normals", text="Normals")
        
        if props.debug_show_normals:
            row = box_col.row(align=True)
            row.prop(props, "debug_normal_size", text="Size")
            row.prop(props, "debug_normal_color", text="")
        
        box_col.prop(props, "debug_show_weights", text="Weights")
        
        if props.debug_show_weights:
            box_col.prop(props, "debug_weight_gradient", text="Gradient")
        
        box_col.prop(props, "debug_show_connections", text="Connections")
        
        if props.debug_show_connections:
            row = box_col.row(align=True)
            row.prop(props, "debug_connection_color", text="")
            row.prop(props, "debug_connection_width", text="Width")
        
        box_col.separator()
        box_col.prop(props, "debug_show_indices", text="Show Indices")
        box_col.prop(props, "debug_show_bounding_box", text="Bounding Boxes")
        
        # Validation Tools
        col.separator()
        col.label(text="Validation:", icon='CHECKMARK')
        
        row = col.row(align=True)
        row.operator("kette.validate_chain", text="Validate", icon='CHECKMARK')
        row.operator("kette.fix_chain", text="Auto Fix", icon='TOOL_SETTINGS')
        
        # Show validation results if available
        if hasattr(props, 'last_validation_result'):
            box = col.box()
            if props.last_validation_result == 'VALID':
                box.label(text="Chain Valid", icon='CHECKMARK')
            else:
                box.alert = True
                box.label(text="Issues Found:", icon='ERROR')
                # List issues here
        
        # Cache Management
        col.separator()
        col.label(text="Cache:", icon='MEMORY')
        
        box = col.box()
        box_col = box.column(align=True)
        
        # Cache stats
        if hasattr(props, 'cache_size'):
            row = box_col.row(align=True)
            row.label(text=f"Size: {props.cache_size:.2f} MB")
            row.label(text=f"Objects: {props.cache_objects}")
        
        box_col.separator()
        row = box_col.row(align=True)
        row.operator("kette.refresh_cache", text="Refresh", icon='FILE_REFRESH')
        row.operator("kette.clear_cache", text="Clear", icon='TRASH')
        
        # Performance Monitor
        col.separator()
        col.label(text="Performance:", icon='TIME')
        
        box = col.box()
        box_col = box.column(align=True)
        
        # Performance metrics
        if hasattr(props, 'last_update_time'):
            box_col.label(text=f"Last Update: {props.last_update_time:.3f}s")
        
        if hasattr(props, 'fps_current'):
            row = box_col.row(align=True)
            row.label(text=f"FPS: {props.fps_current:.1f}")
            row.label(text=f"Target: {props.fps_target:.1f}")
        
        box_col.separator()
        box_col.prop(props, "debug_profile_operations", text="Profile Operations")
        
        if props.debug_profile_operations:
            box_col.operator("kette.show_profiler", text="Show Profiler", icon='GRAPH')
        
        # System Info
        col.separator()
        col.label(text="System:", icon='PREFERENCES')
        
        box = col.box()
        box_col = box.column(align=True)
        box_col.scale_y = 0.8
        
        import sys
        box_col.label(text=f"Python: {sys.version.split()[0]}")
        box_col.label(text=f"Blender: {bpy.app.version_string}")
        
        # Addon version if available
        if hasattr(props, 'addon_version'):
            box_col.label(text=f"Addon: {props.addon_version}")
        
        box_col.separator()
        box_col.operator("kette.report_bug", text="Report Issue", icon='URL')
        
        # Developer Tools
        col.separator()
        col.label(text="Developer:", icon='SCRIPT')
        
        row = col.row(align=True)
        row.operator("kette.reload_addon", text="Reload", icon='FILE_REFRESH')
        row.operator("kette.open_console", text="Console", icon='CONSOLE')
        
        # Test functions
        if props.debug_test_mode:
            col.separator()
            box = col.box()
            box.alert = True
            box.label(text="TEST MODE", icon='ERROR')
            
            box_col = box.column(align=True)
            box_col.operator("kette.run_tests", text="Run Tests", icon='PLAY')
            box_col.operator("kette.benchmark", text="Benchmark", icon='TIME')


# Classes for registration
classes = [
    KETTE_PT_settings,
    KETTE_PT_debug,
]
