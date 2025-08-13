"""
Connector Operations - Connection between spheres
"""

import bpy
from bpy.types import Operator
from bpy.props import BoolProperty
import kettentool.creation.connectors as connectors
import kettentool.creation.spheres as spheres
import kettentool.core.constants as constants

# =========================================
# CONNECTOR OPERATORS
# =========================================

class KETTE_OT_connect_selected(Operator):
    """Connect selected spheres with straight or curved connector"""
    bl_idname = "kette.connect_selected"
    bl_label = "Connect Selected Spheres"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        selected = [obj for obj in context.selected_objects 
                   if obj.name.startswith("Kugel")]
        return len(selected) == 2
    
    def execute(self, context):
        props = context.scene.chain_construction_props
        
        # Get selected spheres
        selected_spheres = [obj for obj in context.selected_objects 
                          if obj.name.startswith("Kugel")]
        
        if len(selected_spheres) != 2:
            self.report({'ERROR'}, "Select exactly 2 spheres!")
            return {'CANCELLED'}
        
        sphere1, sphere2 = selected_spheres
        
        # Check if already connected (simple check)
        existing = False
        for obj in bpy.data.objects:
            if "Connector" in obj.name or "Verbinder" in obj.name:
                s1 = obj.get("sphere1")
                s2 = obj.get("sphere2")
                if s1 and s2:
                    if (s1 == sphere1.name and s2 == sphere2.name) or (s1 == sphere2.name and s2 == sphere1.name):
                        existing = True
                        break
        
        if existing:
            self.report({'WARNING'}, "Spheres already connected")
            return {'CANCELLED'}
        
        # Create connector
        connector = connectors.connect_spheres(sphere1, sphere2, props.kette_use_curved, props)
        
        if connector:
            self.report({'INFO'}, f"Created {'curved' if props.kette_use_curved else 'straight'} connector")
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "Failed to create connector")
            return {'CANCELLED'}

class KETTE_OT_auto_connect(Operator):
    """Auto-connect selected spheres based on distance"""
    bl_idname = "kette.auto_connect"
    bl_label = "Auto Connect Selected"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        selected = [obj for obj in context.selected_objects 
                   if obj.name.startswith("Kugel")]
        return len(selected) >= 2
    
    def execute(self, context):
        props = context.scene.chain_construction_props
        
        # Get selected spheres
        selected_spheres = [obj for obj in context.selected_objects 
                          if obj.name.startswith("Kugel")]
        
        if len(selected_spheres) < 2:
            self.report({'ERROR'}, "Select at least 2 spheres!")
            return {'CANCELLED'}
        
        # Connect spheres in sequence
        connected = 0
        for i in range(len(selected_spheres) - 1):
            sphere1 = selected_spheres[i]
            sphere2 = selected_spheres[i + 1]
            
            # Check if already connected
            existing = False
            for obj in bpy.data.objects:
                if "Connector" in obj.name or "Verbinder" in obj.name:
                    s1 = obj.get("sphere1")
                    s2 = obj.get("sphere2")
                    if s1 and s2:
                        if (s1 == sphere1.name and s2 == sphere2.name) or (s1 == sphere2.name and s2 == sphere1.name):
                            existing = True
                            break
            
            if not existing:
                connector = connectors.connect_spheres(sphere1, sphere2, props.kette_use_curved, props)
                if connector:
                    connected += 1
        
        self.report({'INFO'}, f"Created {connected} connections")
        return {'FINISHED'}

class KETTE_OT_auto_connect_all(Operator):
    """Auto-connect all spheres in the scene"""
    bl_idname = "kette.auto_connect_all"
    bl_label = "Auto Connect All"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.chain_construction_props
        
        # Get all spheres
        all_spheres = [obj for obj in bpy.data.objects 
                      if obj.name.startswith("Kugel")]
        
        if len(all_spheres) < 2:
            self.report({'INFO'}, "Need at least 2 spheres")
            return {'CANCELLED'}
        
        # Sort by name for consistent ordering
        all_spheres.sort(key=lambda x: x.name)
        
        # Connect in sequence
        connected = 0
        for i in range(len(all_spheres) - 1):
            sphere1 = all_spheres[i]
            sphere2 = all_spheres[i + 1]
            
            # Check if already connected
            existing = False
            for obj in bpy.data.objects:
                if "Connector" in obj.name or "Verbinder" in obj.name:
                    s1 = obj.get("sphere1")
                    s2 = obj.get("sphere2")
                    if s1 and s2:
                        if (s1 == sphere1.name and s2 == sphere2.name) or (s1 == sphere2.name and s2 == sphere1.name):
                            existing = True
                            break
            
            if not existing:
                connector = connectors.connect_spheres(sphere1, sphere2, props.kette_use_curved, props)
                if connector:
                    connected += 1
        
        self.report({'INFO'}, f"Created {connected} connections")
        return {'FINISHED'}

class KETTE_OT_delete_connectors(Operator):
    """Delete selected connectors"""
    bl_idname = "kette.delete_connectors"
    bl_label = "Delete Selected Connectors"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        return any(obj for obj in context.selected_objects 
                  if "Connector" in obj.name or "Verbinder" in obj.name)
    
    def execute(self, context):
        selected_connectors = [obj for obj in context.selected_objects 
                             if "Connector" in obj.name or "Verbinder" in obj.name]
        
        count = len(selected_connectors)
        for obj in selected_connectors:
            bpy.data.objects.remove(obj, do_unlink=True)
        
        self.report({'INFO'}, f"Deleted {count} connectors")
        return {'FINISHED'}

class KETTE_OT_delete_all_connectors(Operator):
    """Delete all connectors in the scene"""
    bl_idname = "kette.delete_all_connectors"
    bl_label = "Delete All Connectors"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        all_connectors = [obj for obj in bpy.data.objects 
                         if "Connector" in obj.name or "Verbinder" in obj.name]
        
        count = len(all_connectors)
        for obj in all_connectors:
            bpy.data.objects.remove(obj, do_unlink=True)
        
        # Clear connector cache if it exists
        if hasattr(constants, '_existing_connectors'):
            constants._existing_connectors.clear()
        
        self.report({'INFO'}, f"Deleted {count} connectors")
        return {'FINISHED'}

# =========================================
# REGISTRATION
# =========================================

classes = [
    KETTE_OT_connect_selected,
    KETTE_OT_auto_connect,
    KETTE_OT_auto_connect_all,
    KETTE_OT_delete_connectors,
    KETTE_OT_delete_all_connectors,
]

def register():
    """Register connector operators"""
    for cls in classes:
        bpy.utils.register_class(cls)
    
    if constants.debug:
        constants.debug.info('OPERATORS', "Connector operators registered")

def unregister():
    """Unregister connector operators"""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
