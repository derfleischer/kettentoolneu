"""
Utility Operations - Cleanup and Organization
"""

import bpy
from bpy.types import Operator
from bpy.props import BoolProperty
import kettentool.creation.spheres as spheres
import kettentool.creation.connectors as connectors
import kettentool.utils.collections as collections
import kettentool.utils.materials as materials
import kettentool.core.constants as constants
import kettentool.core.cache_manager as cache_manager

# =========================================
# CLEANUP OPERATORS
# =========================================

class KETTE_OT_clear_all(Operator):
    """Clear all chain objects"""
    bl_idname = "kette.clear_all"
    bl_label = "Clear All"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Delete all spheres and connectors"
    
    clear_spheres: BoolProperty(
        name="Clear Spheres",
        description="Delete all spheres",
        default=True
    )
    
    clear_connectors: BoolProperty(
        name="Clear Connectors",
        description="Delete all connectors",
        default=True
    )
    
    def execute(self, context):
        deleted_spheres = 0
        deleted_connectors = 0
        
        # Delete spheres
        if self.clear_spheres:
            all_spheres = spheres.get_all_spheres()
            deleted_spheres = len(all_spheres)
            spheres.delete_all_spheres()
        
        # Delete connectors
        if self.clear_connectors:
            all_connectors = [obj for obj in bpy.data.objects if 'Connector' in obj.name or 'Verbinder' in obj.name]
            deleted_connectors = len(all_connectors)
            connectors.delete_all_connectors()
        
        # Clear caches
        cache_manager.cleanup_all_caches()
        
        self.report({'INFO'}, 
                   f"Deleted {deleted_spheres} spheres, {deleted_connectors} connectors")
        return {'FINISHED'}
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)
    
    def draw(self, context):
        layout = self.layout
        layout.prop(self, "clear_spheres")
        layout.prop(self, "clear_connectors")

class KETTE_OT_organize_scene(Operator):
    """Organize scene collections"""
    bl_idname = "kette.organize_scene"
    bl_label = "Organize Scene"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Organize all chain objects into proper collections"
    
    def execute(self, context):
        # Organize objects
        organized = collections.organize_scene()
        
        # Cleanup empty collections
        removed = collections.cleanup_empty_collections()
        
        self.report({'INFO'}, 
                   f"Organized {organized} objects, removed {removed} empty collections")
        return {'FINISHED'}

class KETTE_OT_cleanup_unused(Operator):
    """Cleanup unused data"""
    bl_idname = "kette.cleanup_unused"
    bl_label = "Cleanup Unused"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Remove unused materials and data"
    
    def execute(self, context):
        # Cleanup materials
        removed_materials = materials.cleanup_unused_materials()
        
        # Cleanup invalid cache references
        removed_refs = cache_manager.cleanup_invalid_references()
        
        self.report({'INFO'}, 
                   f"Removed {removed_materials} unused materials, {removed_refs} invalid references")
        return {'FINISHED'}

# =========================================
# SELECTION OPERATORS
# =========================================

class KETTE_OT_select_chain_objects(Operator):
    """Select chain objects"""
    bl_idname = "kette.select_chain_objects"
    bl_label = "Select Chain Objects"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Select all chain objects"
    
    object_type: BoolProperty(
        name="Object Type",
        description="Type of objects to select",
        default=True
    )
    
    def execute(self, context):
        bpy.ops.object.select_all(action='DESELECT')
        
        selected = 0
        
        # Select spheres
        for sphere in spheres.get_all_spheres():
            sphere.select_set(True)
            selected += 1
        
        # Select connectors
        for connector in connectors.get_all_connectors():
            connector.select_set(True)
            selected += 1
        
        self.report({'INFO'}, f"Selected {selected} chain objects")
        return {'FINISHED'}

# =========================================
# REGISTRATION
# =========================================

classes = [
    KETTE_OT_clear_all,
    KETTE_OT_organize_scene,
    KETTE_OT_cleanup_unused,
    KETTE_OT_select_chain_objects,
]

def register():
    """Register utility operators"""
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    """Unregister utility operators"""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
