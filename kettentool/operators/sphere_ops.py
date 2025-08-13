"""
Sphere Operations - Creation, Deletion, Management
"""

import bpy
from bpy.types import Operator
from bpy.props import FloatProperty, StringProperty, BoolProperty
import kettentool.creation.spheres as spheres
import kettentool.utils.surface as surface
import kettentool.core.constants as constants

# =========================================
# SPHERE CREATION OPERATORS
# =========================================

class KETTE_OT_place_sphere(Operator):
    """Place sphere at 3D cursor"""
    bl_idname = "kette.place_sphere"
    bl_label = "Place Sphere"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Place a sphere at 3D cursor location"
    
    sphere_type: StringProperty(
        name="Type",
        default="default",
        options={'HIDDEN'}
    )
    
    def execute(self, context):
        props = context.scene.chain_construction_props
        cursor_loc = context.scene.cursor.location.copy()
        
        # Snap to surface if target object exists
        if props.kette_target_obj:
            # Direkte Surface-Projektion mit BVH
            offset = surface.calculate_surface_offset(props)
            cursor_loc, _ = surface.project_to_surface(
                props.kette_target_obj, 
                cursor_loc, 
                offset
            )
        
        # Create sphere
        sphere = spheres.create_sphere(
            location=cursor_loc,
            radius=props.kette_kugelradius,
            sphere_type=self.sphere_type
        )
        
        # Select new sphere
        spheres.select_sphere(sphere)
        
        self.report({'INFO'}, f"Created sphere: {sphere.name}")
        return {'FINISHED'}

class KETTE_OT_place_start_sphere(Operator):
    """Place start sphere at 3D cursor"""
    bl_idname = "kette.place_start_sphere"
    bl_label = "Place Start Sphere"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Place a start sphere at 3D cursor location"
    
    def execute(self, context):
        bpy.ops.kette.place_sphere(sphere_type='start')
        return {'FINISHED'}

# =========================================
# SPHERE DELETION OPERATORS
# =========================================

class KETTE_OT_delete_selected_spheres(Operator):
    """Delete selected spheres"""
    bl_idname = "kette.delete_selected_spheres"
    bl_label = "Delete Selected Spheres"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Delete all selected spheres"
    
    @classmethod
    def poll(cls, context):
        return any(obj.name.startswith("Kugel") for obj in context.selected_objects)
    
    def execute(self, context):
        deleted = 0
        for obj in list(context.selected_objects):
            if obj.name.startswith("Kugel"):
                spheres.delete_sphere(obj)
                deleted += 1
        
        self.report({'INFO'}, f"Deleted {deleted} spheres")
        return {'FINISHED'}

class KETTE_OT_delete_all_spheres(Operator):
    """Delete all spheres"""
    bl_idname = "kette.delete_all_spheres"
    bl_label = "Delete All Spheres"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Delete all spheres in the scene"
    
    def execute(self, context):
        all_spheres = spheres.get_all_spheres()
        count = len(all_spheres)
        spheres.delete_all_spheres()
        
        self.report({'INFO'}, f"Deleted all {count} spheres")
        return {'FINISHED'}
    
    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

# =========================================
# SPHERE SELECTION OPERATORS
# =========================================

class KETTE_OT_select_all_spheres(Operator):
    """Select all spheres"""
    bl_idname = "kette.select_all_spheres"
    bl_label = "Select All Spheres"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Select all spheres in the scene"
    
    def execute(self, context):
        all_spheres = spheres.get_all_spheres()
        spheres.select_spheres(all_spheres)
        
        self.report({'INFO'}, f"Selected {len(all_spheres)} spheres")
        return {'FINISHED'}

class KETTE_OT_select_connected_spheres(Operator):
    """Select connected spheres"""
    bl_idname = "kette.select_connected_spheres"
    bl_label = "Select Connected"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Select spheres connected to active sphere"
    
    @classmethod
    def poll(cls, context):
        return (context.active_object and 
                context.active_object.name.startswith("Kugel"))
    
    def execute(self, context):
        active = context.active_object
        connected = spheres.get_connected_spheres(active)
        
        if connected:
            spheres.select_spheres(connected, deselect_others=False)
            self.report({'INFO'}, f"Selected {len(connected)} connected spheres")
        else:
            self.report({'INFO'}, "No connected spheres found")
        
        return {'FINISHED'}

# =========================================
# REGISTRATION
# =========================================

classes = [
    KETTE_OT_place_sphere,
    KETTE_OT_place_start_sphere,
    KETTE_OT_delete_selected_spheres,
    KETTE_OT_delete_all_spheres,
    KETTE_OT_select_all_spheres,
    KETTE_OT_select_connected_spheres,
]

def register():
    """Register sphere operators"""
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    """Unregister sphere operators"""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
