"""
Tool Panels for Chain Construction
Includes main tools and auto-connect functionality
"""

import bpy
from bpy.types import Panel


class KETTE_PT_tools(Panel):
    """Tools panel"""
    bl_label = "Tools"
    bl_idname = "KETTE_PT_tools"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tools"
    bl_parent_id = "KETTE_PT_main_panel"
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.chain_props
        
        # Create Operations
        col = layout.column(align=True)
        col.label(text="Create:", icon='ADD')
        
        row = col.row(align=True)
        row.scale_y = 1.3
        op = row.operator("kette.create_sphere", text="Add Sphere", icon='SPHERE')
        op.use_3d_cursor = False  # Use click position
        row.operator("kette.create_connector", text="Connect", icon='LINK_DATA')
        
        # Batch Operations
        col.separator()
        col.label(text="Batch Operations:", icon='FULLSCREEN_ENTER')
        
        row = col.row(align=True)
        row.operator("kette.duplicate_selected", text="Duplicate", icon='DUPLICATE')
        row.operator("kette.mirror_selected", text="Mirror", icon='MOD_MIRROR')
        
        col.separator()
        
        # Selection Operations
        col.label(text="Selection:", icon='RESTRICT_SELECT_OFF')
        
        row = col.row(align=True)
        row.operator("kette.select_all_spheres", text="Spheres", icon='SPHERE')
        row.operator("kette.select_all_connectors", text="Connectors", icon='LINK_DATA')
        
        row = col.row(align=True)
        row.operator("kette.select_connected", text="Connected", icon='LINKED')
        row.operator("kette.select_boundary", text="Boundary", icon='BORDER_RECT')
        
        # Cleanup Operations
        col.separator()
        col.label(text="Cleanup:", icon='BRUSH_DATA')
        
        row = col.row(align=True)
        row.operator("kette.remove_doubles", text="Merge Close", icon='AUTOMERGE_ON')
        row.operator("kette.clean_connectors", text="Fix Links", icon='UNLINKED')
        
        col.separator()
        row = col.row(align=True)
        row.scale_y = 1.2
        row.alert = True
        row.operator("kette.clear_all", text="Clear All", icon='TRASH')


class KETTE_PT_auto_connect(Panel):
    """Auto-Connect panel"""
    bl_label = "Auto-Connect"
    bl_idname = "KETTE_PT_auto_connect"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tools"
    bl_parent_id = "KETTE_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.chain_props
        
        col = layout.column(align=True)
        
        # Enable toggle with icon
        row = col.row(align=True)
        row.prop(props, "auto_connect_enabled", text="Enable Auto-Connect")
        
        if props.auto_connect_enabled:
            row.label(text="", icon='LINKED')
        else:
            row.label(text="", icon='UNLINKED')
        
        if props.auto_connect_enabled:
            col.separator()
            
            # Distance settings
            box = col.box()
            box_col = box.column(align=True)
            box_col.label(text="Connection Rules:", icon='SETTINGS')
            
            box_col.prop(props, "connect_distance", text="Max Distance")
            box_col.prop(props, "connect_angle", text="Max Angle")
            
            # Advanced options
            box_col.separator()
            box_col.prop(props, "avoid_crossing", text="Avoid Crossing")
            box_col.prop(props, "prefer_shortest", text="Prefer Shortest Path")
            
            # Manual connect button
            col.separator()
            row = col.row(align=True)
            row.scale_y = 1.3
            
            selected_spheres = [obj for obj in context.selected_objects 
                              if obj.get("is_chain_sphere")]
            
            if len(selected_spheres) >= 2:
                op_text = f"Connect {len(selected_spheres)} Selected"
                row.operator("kette.auto_connect_selected", 
                           text=op_text, icon='LINKED')
            else:
                row.enabled = False
                row.operator("kette.auto_connect_selected", 
                           text="Select 2+ Spheres", icon='UNLINKED')
            
            # Connection preview
            if props.show_debug_info:
                col.separator()
                box = col.box()
                box.label(text="Preview:", icon='VIEW3D')
                box.prop(props, "preview_connections", text="Show Preview")
                
                if props.preview_connections:
                    box.prop(props, "preview_color", text="")


# Classes for registration
classes = [
    KETTE_PT_tools,
    KETTE_PT_auto_connect,
]
