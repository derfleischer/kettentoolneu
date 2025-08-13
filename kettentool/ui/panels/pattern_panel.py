"""
Pattern Connector Panel for Chain Construction
Advanced pattern-based connection of chain spheres
"""

import bpy
from bpy.types import Panel


class KETTE_PT_pattern_connector(Panel):
    """Pattern Connector Panel"""
    bl_label = "Pattern Connector"
    bl_idname = "KETTE_PT_pattern_connector"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tools"
    bl_parent_id = "KETTE_PT_tools"  # Child of Tools panel
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.chain_props
        
        # Count spheres
        all_spheres = [obj for obj in bpy.data.objects 
                      if obj.get("is_chain_sphere")]
        selected_spheres = [obj for obj in context.selected_objects 
                          if obj.get("is_chain_sphere")]
        
        # Info box
        box = layout.box()
        col = box.column(align=True)
        
        if not all_spheres:
            col.label(text="No spheres in scene", icon='INFO')
            col.label(text="Create spheres first")
            return
        else:
            row = col.row(align=True)
            row.label(text=f"Total: {len(all_spheres)}", icon='SPHERE')
            
            if selected_spheres:
                row.label(text=f"Selected: {len(selected_spheres)}", icon='RESTRICT_SELECT_OFF')
            
            # Show existing connections
            connector_count = len([obj for obj in bpy.data.objects 
                                 if obj.get("is_chain_connector")])
            if connector_count > 0:
                col.label(text=f"Existing Connectors: {connector_count}", icon='LINK_DATA')
        
        # Pattern type selection
        col = layout.column(align=True)
        col.label(text="Pattern Type:", icon='GEOMETRY_NODES')
        col.prop(props, "pattern_type", text="")
        
        # Pattern preview
        if props.show_debug_info:
            row = col.row(align=True)
            row.prop(props, "pattern_preview", text="Preview Pattern")
            if props.pattern_preview:
                row.label(text="", icon='HIDE_OFF')
        
        # Pattern-specific settings
        col.separator()
        box = col.box()
        
        if props.pattern_type == 'LINEAR':
            self.draw_linear_settings(box, props)
            
        elif props.pattern_type == 'ZIGZAG':
            self.draw_zigzag_settings(box, props)
            
        elif props.pattern_type == 'SPIRAL':
            self.draw_spiral_settings(box, props)
            
        elif props.pattern_type == 'RADIAL':
            self.draw_radial_settings(box, props)
            
        elif props.pattern_type == 'RANDOM':
            self.draw_random_settings(box, props)
            
        elif props.pattern_type == 'CUSTOM':
            self.draw_custom_settings(box, props)
        
        # Common connection settings
        col = layout.column(align=True)
        col.separator()
        col.label(text="Connection Settings:", icon='SETTINGS')
        
        row = col.row(align=True)
        row.prop(props, "pattern_max_distance", text="Max Distance")
        
        # Only show if relevant
        if props.pattern_type not in ['LINEAR', 'CUSTOM']:
            row.prop(props, "pattern_use_distance", text="", icon='CONSTRAINT')
        
        col.prop(props, "pattern_smooth_connections", text="Smooth Paths")
        
        if props.pattern_smooth_connections:
            col.prop(props, "pattern_smooth_iterations", text="Smooth Iterations")
        
        # Advanced options
        box = col.box()
        box_col = box.column(align=True)
        box_col.label(text="Advanced:", icon='PREFERENCES')
        box_col.prop(props, "pattern_avoid_existing", text="Avoid Duplicates")
        box_col.prop(props, "pattern_bidirectional", text="Bidirectional")
        
        # Execution section
        col.separator()
        self.draw_execute_section(col, context, props, selected_spheres, all_spheres)
    
    def draw_linear_settings(self, layout, props):
        """Draw settings for linear pattern"""
        col = layout.column(align=True)
        col.label(text="Linear Pattern:", icon='CON_FOLLOWPATH')
        
        col.prop(props, "pattern_skip", text="Skip Every")
        col.prop(props, "pattern_reverse", text="Reverse Order")
        
        col.separator()
        col.prop(props, "pattern_closed_loop", text="Close Loop")
    
    def draw_zigzag_settings(self, layout, props):
        """Draw settings for zigzag pattern"""
        col = layout.column(align=True)
        col.label(text="Zigzag Pattern:", icon='IPO_SINE')
        
        col.prop(props, "pattern_zigzag_width", text="Width")
        col.prop(props, "pattern_zigzag_count", text="Segments")
        
        col.separator()
        row = col.row(align=True)
        row.prop(props, "pattern_zigzag_sharp", text="Sharp Corners")
        
        if not props.pattern_zigzag_sharp:
            col.prop(props, "pattern_zigzag_smooth", text="Smoothness")
    
    def draw_spiral_settings(self, layout, props):
        """Draw settings for spiral pattern"""
        col = layout.column(align=True)
        col.label(text="Spiral Pattern:", icon='FORCE_VORTEX')
        
        col.prop(props, "pattern_spiral_turns", text="Turns")
        col.prop(props, "pattern_spiral_height", text="Height Factor")
        
        col.separator()
        col.prop(props, "pattern_spiral_tightness", text="Tightness")
        col.prop(props, "pattern_spiral_direction", text="Direction")
    
    def draw_radial_settings(self, layout, props):
        """Draw settings for radial pattern"""
        col = layout.column(align=True)
        col.label(text="Radial Pattern:", icon='FORCE_FORCE')
        
        col.prop(props, "pattern_radial_rings", text="Ring Groups")
        col.prop(props, "pattern_radial_center", text="Center")
        
        col.separator()
        col.prop(props, "pattern_radial_connect_rings", text="Connect Rings")
        
        if props.pattern_radial_connect_rings:
            col.prop(props, "pattern_radial_ring_offset", text="Ring Offset")
    
    def draw_random_settings(self, layout, props):
        """Draw settings for random pattern"""
        col = layout.column(align=True)
        col.label(text="Random Pattern:", icon='RANDOMIZE')
        
        col.prop(props, "pattern_random_seed", text="Seed")
        
        # Probability slider with percentage
        row = col.row(align=True)
        row.prop(props, "pattern_random_probability", text="Connection")
        row.label(text=f"{props.pattern_random_probability:.0%}")
        
        col.separator()
        col.prop(props, "pattern_random_min_distance", text="Min Distance")
    
    def draw_custom_settings(self, layout, props):
        """Draw settings for custom pattern"""
        col = layout.column(align=True)
        col.label(text="Custom Pattern:", icon='GREASEPENCIL')
        
        col.prop(props, "pattern_custom_rule", text="")
        
        col.separator()
        box = col.box()
        box.scale_y = 0.8
        box.label(text="Rules:", icon='INFO')
        box.label(text="• 'nearest': Connect to nearest")
        box.label(text="• 'every:n': Connect every n-th")
        box.label(text="• 'star': Star pattern")
        box.label(text="• 'selection': Use selection order")
    
    def draw_execute_section(self, layout, context, props, selected_spheres, all_spheres):
        """Draw execution controls"""
        col = layout.column(align=True)
        
        # Mode selection
        row = col.row(align=True)
        row.label(text="Apply To:")
        
        sub = row.row(align=True)
        sub.prop(props, "pattern_use_selected", text="Selected Only")
        
        # Main execute button
        row = col.row(align=True)
        row.scale_y = 1.5
        
        # Determine sphere count for operation
        sphere_count = len(selected_spheres) if props.pattern_use_selected and selected_spheres else len(all_spheres)
        
        if sphere_count < 2:
            row.enabled = False
            op_text = "Need 2+ Spheres"
            icon = 'ERROR'
        else:
            op_text = f"Connect {sphere_count} Spheres"
            icon = 'PLAY'
        
        row.operator("kette.apply_pattern", text=op_text, icon=icon)
        
        # Additional actions
        if any(obj.get("is_chain_connector") for obj in bpy.data.objects):
            col.separator()
            row = col.row(align=True)
            row.operator("kette.clear_connectors", text="Clear Connectors", icon='X')
            row.operator("kette.rebuild_connectors", text="Rebuild", icon='FILE_REFRESH')
        
        # Pattern library (future feature)
        if props.show_debug_info:
            col.separator()
            box = col.box()
            box.label(text="Pattern Library:", icon='ASSET_MANAGER')
            
            row = box.row(align=True)
            row.operator("kette.save_pattern", text="Save", icon='FILE_TICK')
            row.operator("kette.load_pattern", text="Load", icon='FILE_FOLDER')


# Classes for registration
classes = [
    KETTE_PT_pattern_connector,
]
