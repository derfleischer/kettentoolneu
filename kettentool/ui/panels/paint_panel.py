"""
Paint Mode Panel for Chain Construction
Interactive painting of chain spheres on surfaces
"""

import bpy
from bpy.types import Panel


class KETTE_PT_paint_mode(Panel):
    """Paint Mode Panel"""
    bl_label = "Paint Mode"
    bl_idname = "KETTE_PT_paint_mode"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tools"
    bl_parent_id = "KETTE_PT_main_panel"
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.chain_props
        
        # Check if target exists
        if not props.kette_target_obj:
            box = layout.box()
            box.label(text="No Target Object!", icon='ERROR')
            box.label(text="Select a target mesh first")
            return
        
        col = layout.column(align=True)
        
        # Paint mode toggle button
        row = col.row(align=True)
        row.scale_y = 1.5
        
        if props.paint_mode_active:
            row.operator("kette.toggle_paint_mode", 
                        text="Exit Paint Mode", 
                        icon='BRUSH_DATA',
                        depress=True)
            
            # Active indicator
            col.separator()
            box = col.box()
            box.alert = True
            box.label(text="PAINT MODE ACTIVE", icon='REC')
        else:
            row.operator("kette.toggle_paint_mode", 
                        text="Enter Paint Mode", 
                        icon='BRUSH_DATA')
        
        # Paint settings (always visible for preview)
        col.separator()
        col.label(text="Brush Settings:", icon='BRUSH_DATA')
        
        # Sphere size with preview
        row = col.row(align=True)
        row.prop(props, "paint_sphere_radius", text="Size")
        if props.paint_mode_active:
            row.label(text="", icon='SPHERE')
        
        # Spacing settings
        col.prop(props, "paint_spacing", text="Spacing")
        
        # Advanced brush settings
        box = col.box()
        box_col = box.column(align=True)
        box_col.label(text="Brush Behavior:", icon='SETTINGS')
        
        box_col.prop(props, "paint_smooth_factor", text="Path Smoothing")
        box_col.prop(props, "paint_follow_surface", text="Follow Surface")
        
        if props.paint_follow_surface:
            box_col.prop(props, "paint_surface_offset", text="Surface Offset")
        
        # Stroke settings
        box_col.separator()
        box_col.prop(props, "paint_continuous", text="Continuous Stroke")
        
        if props.paint_continuous:
            box_col.prop(props, "paint_stroke_smooth", text="Stroke Smoothing")
        
        # Auto-connect options
        col.separator()
        row = col.row(align=True)
        row.prop(props, "paint_auto_connect", text="Auto-Connect")
        
        if props.paint_auto_connect:
            row.label(text="", icon='LINKED')
            box = col.box()
            box_col = box.column(align=True)
            box_col.prop(props, "paint_connect_distance", text="Connect Range")
            box_col.prop(props, "paint_connect_chain", text="Chain Mode")
            
            if props.paint_connect_chain:
                box_col.prop(props, "paint_chain_max", text="Max Chain Links")
        
        # Paint patterns
        col.separator()
        col.label(text="Paint Patterns:", icon='TEXTURE')
        
        row = col.row(align=True)
        row.prop(props, "paint_pattern", text="")
        
        if props.paint_pattern != 'NONE':
            box = col.box()
            box_col = box.column(align=True)
            
            if props.paint_pattern == 'DOTS':
                box_col.prop(props, "paint_dot_spacing", text="Dot Spacing")
            elif props.paint_pattern == 'LINE':
                box_col.prop(props, "paint_line_segments", text="Segments")
            elif props.paint_pattern == 'WAVE':
                box_col.prop(props, "paint_wave_amplitude", text="Amplitude")
                box_col.prop(props, "paint_wave_frequency", text="Frequency")
        
        # Controls help
        if props.paint_mode_active:
            col.separator()
            box = col.box()
            box.label(text="Controls:", icon='QUESTION')
            
            col_help = box.column(align=True)
            col_help.scale_y = 0.8
            col_help.label(text="• LMB: Paint spheres")
            col_help.label(text="• Shift: Constrain to axis")
            col_help.label(text="• Ctrl: Snap to surface")
            col_help.label(text="• Alt: Erase mode")
            col_help.label(text="• ESC/RMB: Exit paint mode")
            
            # Undo reminder
            col_help.separator()
            col_help.label(text="• Ctrl+Z: Undo last stroke", icon='LOOP_BACK')
        
        # Quick actions while painting
        if props.paint_mode_active:
            col.separator()
            row = col.row(align=True)
            row.operator("kette.paint_clear_last", text="Clear Last", icon='BACK')
            row.operator("kette.paint_clear_stroke", text="Clear Stroke", icon='X')


# Classes for registration
classes = [
    KETTE_PT_paint_mode,
]
