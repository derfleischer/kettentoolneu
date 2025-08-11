# =========================================
# 4. operators/utility_ops.py - UTILITY-FUNKTIONEN
# =========================================

"""
Utility Operations - Cleanup, Clear, etc. (Funktionslogik-bezogen)
"""

import bpy
from bpy.types import Operator
from bpy.props import BoolProperty, EnumProperty

from ..creation.spheres import delete_sphere, get_spheres_by_type
from ..utils.collections import ensure_collection, cleanup_empty_collections
from ..utils.surface import clear_auras

class KETTE_OT_clear_all(Operator):
    """Löscht alle Chain-Objekte"""
    bl_idname = "kette.clear_all"
    bl_label = "Alle löschen"
    bl_description = "Löscht alle Kugeln und Verbinder"
    bl_options = {'REGISTER', 'UNDO'}

    clear_spheres: BoolProperty(
        name="Kugeln löschen",
        default=True,
        description="Löscht alle Kugeln"
    )

    clear_connectors: BoolProperty(
        name="Verbinder löschen",
        default=True,
        description="Löscht alle Verbinder"
    )

    sphere_types: EnumProperty(
        name="Kugeltypen",
        items=[
            ('ALL', 'Alle', 'Alle Kugeltypen'),
            ('START', 'Nur Start', 'Nur Start-Kugeln'),
            ('PAINTED', 'Nur Paint', 'Nur Paint-Kugeln'),
            ('MANUAL', 'Nur Manual', 'Nur Manual-Kugeln'),
            ('GENERATED', 'Nur Generated', 'Nur Auto-Kugeln'),
        ],
        default='ALL'
    )

    def execute(self, context):
        from ..core import constants
        
        removed_spheres = 0
        removed_connectors = 0
        
        try:
            # Lösche Kugeln
            if self.clear_spheres:
                if self.sphere_types == 'ALL':
                    # Alle Kugeltypen
                    for sphere_type in ['start', 'painted', 'manual', 'generated']:
                        spheres = get_spheres_by_type(sphere_type)
                        for sphere in spheres:
                            delete_sphere(sphere)
                            removed_spheres += 1
                else:
                    # Spezifischer Typ
                    type_map = {
                        'START': 'start',
                        'PAINTED': 'painted', 
                        'MANUAL': 'manual',
                        'GENERATED': 'generated'
                    }
                    sphere_type = type_map[self.sphere_types]
                    spheres = get_spheres_by_type(sphere_type)
                    for sphere in spheres:
                        delete_sphere(sphere)
                        removed_spheres += 1
            
            # Lösche Connectors
            if self.clear_connectors:
                connectors_collection = ensure_collection("connectors")
                to_remove = []
                
                for obj in connectors_collection.objects:
                    if obj.name.startswith("Connector"):
                        to_remove.append(obj)
                
                for obj in to_remove:
                    # Entferne aus Collections
                    for c in obj.users_collection:
                        c.objects.unlink(obj)
                    
                    # Lösche Daten
                    if obj.data and obj.data.users == 1:
                        if obj.type == 'MESH':
                            bpy.data.meshes.remove(obj.data)
                        elif obj.type == 'CURVE':
                            bpy.data.curves.remove(obj.data)
                    
                    bpy.data.objects.remove(obj, do_unlink=True)
                    removed_connectors += 1
            
            # Clear auras and cleanup
            clear_auras()
            cleanup_empty_collections()
            
            # Report
            msgs = []
            if removed_spheres > 0:
                msgs.append(f"{removed_spheres} Kugeln")
            if removed_connectors > 0:
                msgs.append(f"{removed_connectors} Verbinder")
            
            if msgs:
                self.report({'INFO'}, f"Gelöscht: {', '.join(msgs)}")
                if constants.debug:
                    constants.debug.info('UTILITY', f"Cleared {removed_spheres} spheres, {removed_connectors} connectors")
            else:
                self.report({'INFO'}, "Nichts zu löschen")
                
        except Exception as e:
            self.report({'ERROR'}, f"Fehler beim Löschen: {str(e)}")
            if constants.debug:
                constants.debug.error('UTILITY', f"Error clearing objects: {e}")
            return {'CANCELLED'}
        
        return {'FINISHED'}
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)
    
    def draw(self, context):
        layout = self.layout
        
        layout.prop(self, "clear_spheres")
        if self.clear_spheres:
            layout.prop(self, "sphere_types")
        
        layout.prop(self, "clear_connectors")

# Registration
classes = [
    KETTE_OT_clear_all,
]
