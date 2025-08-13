"""
Main Panel for Chain Construction Tool
Compatible with modular property structure
"""

import bpy
from bpy.types import Panel


class KETTE_PT_main_panel(Panel):
    """Main panel for Chain Construction"""
    bl_label = "Chain Construction"
    bl_idname = "KETTE_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tools"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        
        # Check if properties exist
        if not hasattr(context.scene, 'chain_props'):
            layout.label(text="Properties not initialized!", icon='ERROR')
            return
            
        props = context.scene.chain_props

        # Access modular properties safely
        main_props = props.main if hasattr(props, 'main') else props
        paint_props = props.paint if hasattr(props, 'paint') else props

        # Target Object Section
        col = layout.column(align=True)
        col.label(text="Target Object:", icon='OBJECT_DATA')
        col.prop(props, "target_object", text="")
        
        if props.target_object:
            col.label(text=f"Vertices: {len(props.target_object.data.vertices)}", icon='INFO')
        
        col.separator()
        
        # Chain Type Section
        col.label(text="Chain Settings:", icon='LINKED')
        if hasattr(main_props, 'chain_type'):
            col.prop(main_props, "chain_type")
            col.prop(main_props, "link_count")
            col.prop(main_props, "link_size")
            col.prop(main_props, "link_thickness")
        
        col.separator()
        
        # Custom Link Section
        col.label(text="Custom Link:", icon='MESH_DATA')
        col.prop(props, "custom_link", text="")
        
        # Main Actions
        col.separator()
        row = col.row(align=True)
        row.scale_y = 1.5
        
        # Check if operators exist before adding buttons
        if hasattr(bpy.ops.kette, 'create_chain'):
            row.operator("kette.create_chain", text="Create Chain", icon='ADD')
        else:
            row.label(text="Create operator not found", icon='ERROR')
            
        if hasattr(bpy.ops.kette, 'clear_chain'):
            row.operator("kette.clear_chain", text="Clear", icon='X')
        
        # Preview Toggle
        if hasattr(main_props, 'show_preview'):
            col.separator()
            col.prop(main_props, "show_preview", text="Show Preview", icon='HIDE_OFF')


class KETTE_PT_chain_modifiers(Panel):
    """Sub-panel for chain modifiers"""
    bl_label = "Modifiers"
    bl_idname = "KETTE_PT_chain_modifiers"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tools"
    bl_parent_id = "KETTE_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        
        if not hasattr(context.scene, 'chain_props'):
            layout.label(text="Properties not initialized!", icon='ERROR')
            return
            
        props = context.scene.chain_props
        tools_props = props.tools if hasattr(props, 'tools') else props
        
        # Modifier options
        col = layout.column(align=True)
        
        if hasattr(tools_props, 'use_modifiers'):
            col.prop(tools_props, "use_modifiers", text="Enable Modifiers")
            
            if tools_props.use_modifiers:
                col.separator()
                col.label(text="Modifier Stack:", icon='MODIFIER')
                
                # Add modifier buttons
                row = col.row(align=True)
                if hasattr(bpy.ops.kette, 'add_array_modifier'):
                    row.operator("kette.add_array_modifier", text="Array", icon='MOD_ARRAY')
                if hasattr(bpy.ops.kette, 'add_curve_modifier'):
                    row.operator("kette.add_curve_modifier", text="Curve", icon='MOD_CURVE')
                if hasattr(bpy.ops.kette, 'add_solidify_modifier'):
                    row.operator("kette.add_solidify_modifier", text="Solidify", icon='MOD_SOLIDIFY')
        else:
            col.label(text="Modifier properties not available", icon='INFO')


class KETTE_PT_chain_physics(Panel):
    """Sub-panel for physics settings"""
    bl_label = "Physics"
    bl_idname = "KETTE_PT_chain_physics"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tools"
    bl_parent_id = "KETTE_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        
        col = layout.column(align=True)
        col.label(text="Physics Settings:", icon='PHYSICS')
        
        # Physics options
        row = col.row(align=True)
        if hasattr(bpy.ops.kette, 'add_physics'):
            row.operator("kette.add_physics", text="Add Physics", icon='PHYSICS')
        else:
            row.label(text="Physics not implemented", icon='INFO')
        
        if hasattr(bpy.ops.kette, 'bake_physics'):
            row.operator("kette.bake_physics", text="Bake", icon='REC')
        
        col.separator()
        
        # Simulation settings placeholder
        col.label(text="Gravity:", icon='FORCE_FORCE')
        col.label(text="  X: 0.0  Y: 0.0  Z: -9.8")
        col.label(text="Mass: 1.0 kg")
        col.label(text="Damping: 0.1")


# Registration
def register():
    """Register panels"""
    # Registration is handled by the parent module
    pass

def unregister():
    """Unregister panels"""
    # Unregistration is handled by the parent module
    pass
