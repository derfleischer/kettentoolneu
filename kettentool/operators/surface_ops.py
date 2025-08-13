"""
Surface Operations - Snapping and Projection
"""

import bpy
from bpy.types import Operator
from bpy.props import BoolProperty, FloatProperty
import kettentool.creation.spheres as spheres
import kettentool.utils.surface as surface
import kettentool.core.constants as constants

# =========================================
# SURFACE SNAPPING OPERATORS
# =========================================

class KETTE_OT_snap_to_surface(Operator):
    """Snap spheres to surface"""
    bl_idname = "kette.snap_to_surface"
    bl_label = "Snap to Surface"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Snap selected spheres to target surface"
    
    @classmethod
    def poll(cls, context):
        props = context.scene.chain_construction_props
        selected_spheres = [obj for obj in context.selected_objects 
                          if obj.name.startswith("Kugel")]
        return props.kette_target_obj and len(selected_spheres) > 0
    
    def execute(self, context):
        props = context.scene.chain_construction_props
        
        if not props.kette_target_obj:
            self.report({'ERROR'}, "No target object set!")
            return {'CANCELLED'}
        
        # Get selected spheres
        selected_spheres = [obj for obj in context.selected_objects 
                          if obj.name.startswith("Kugel")]
        
        if not selected_spheres:
            self.report({'ERROR'}, "No spheres selected!")
            return {'CANCELLED'}
        
        # Calculate offset
        offset = surface.calculate_surface_offset(props)
        
        # Snap each sphere using BVH
        for sphere in selected_spheres:
            surface.snap_to_surface(sphere, props.kette_target_obj, offset)
        
        self.report({'INFO'}, f"Snapped {len(selected_spheres)} spheres to surface")
        return {'FINISHED'}

class KETTE_OT_snap_all_to_surface(Operator):
    """Snap all spheres to surface"""
    bl_idname = "kette.snap_all_to_surface"
    bl_label = "Snap All to Surface"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Snap all spheres to target surface"
    
    @classmethod
    def poll(cls, context):
        props = context.scene.chain_construction_props
        return props.kette_target_obj is not None
    
    def execute(self, context):
        props = context.scene.chain_construction_props
        
        if not props.kette_target_obj:
            self.report({'ERROR'}, "No target object set!")
            return {'CANCELLED'}
        
        # Get all spheres
        all_spheres = spheres.get_all_spheres()
        
        if not all_spheres:
            self.report({'INFO'}, "No spheres in scene")
            return {'CANCELLED'}
        
        # Calculate offset
        offset = surface.calculate_surface_offset(props)
        
        # Snap each sphere
        for sphere in all_spheres:
            surface.snap_to_surface(sphere, props.kette_target_obj, offset)
        
        self.report({'INFO'}, f"Snapped all {len(all_spheres)} spheres to surface")
        return {'FINISHED'}

class KETTE_OT_project_cursor_to_surface(Operator):
    """Project 3D cursor to surface"""
    bl_idname = "kette.project_cursor_to_surface"
    bl_label = "Project Cursor to Surface"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Project 3D cursor to target surface"
    
    @classmethod
    def poll(cls, context):
        props = context.scene.chain_construction_props
        return props.kette_target_obj is not None
    
    def execute(self, context):
        props = context.scene.chain_construction_props
        
        if not props.kette_target_obj:
            self.report({'ERROR'}, "No target object set!")
            return {'CANCELLED'}
        
        # Project cursor
        cursor_loc = context.scene.cursor.location
        new_loc, normal = surface.project_to_surface(
            props.kette_target_obj,
            cursor_loc,
            0.0  # No offset for cursor
        )
        
        context.scene.cursor.location = new_loc
        
        self.report({'INFO'}, "Cursor projected to surface")
        return {'FINISHED'}

# =========================================
# AURA MANAGEMENT OPERATORS
# =========================================

class KETTE_OT_refresh_aura(Operator):
    """Refresh surface cache"""
    bl_idname = "kette.refresh_aura"
    bl_label = "Refresh Surface Cache"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Refresh BVH cache for target object"
    
    @classmethod
    def poll(cls, context):
        props = context.scene.chain_construction_props
        return props.kette_target_obj is not None
    
    def execute(self, context):
        props = context.scene.chain_construction_props
        
        if not props.kette_target_obj:
            self.report({'ERROR'}, "No target object set!")
            return {'CANCELLED'}
        
        # Invalidate and recreate BVH cache
        surface.invalidate_aura(props.kette_target_obj.name)
        
        # Force rebuild BVH (no density parameter needed anymore)
        _ = surface._build_bvh(props.kette_target_obj)
        
        self.report({'INFO'}, f"Refreshed BVH cache for {props.kette_target_obj.name}")
        return {'FINISHED'}

# =========================================
# REGISTRATION
# =========================================

classes = [
    KETTE_OT_snap_to_surface,
    KETTE_OT_snap_all_to_surface,
    KETTE_OT_project_cursor_to_surface,
    KETTE_OT_refresh_aura,
]

def register():
    """Register surface operators"""
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    """Unregister surface operators"""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
