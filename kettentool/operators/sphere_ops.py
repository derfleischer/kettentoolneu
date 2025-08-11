"""
Sphere Operations - Kugel-Funktionslogik
Optimiert für Blender 4.4 + macOS
"""

import bpy
from bpy.types import Operator
from bpy.props import EnumProperty, BoolProperty

# Direkte Imports (statt relativ)
import kettentool.creation.spheres as sphere_creation
import kettentool.core.constants as constants

# =========================================
# SPHERE PLACEMENT OPERATORS
# =========================================

class KETTE_OT_place_sphere(Operator):
    """Platziert Kugel am 3D-Cursor mit Auto-Snap (Blender 4.4 optimiert)"""
    bl_idname = "kette.place_sphere"
    bl_label = "Kugel platzieren"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Platziert Kugel am 3D-Cursor (Auto-Snap auf Zielobjekt)"

    sphere_type: EnumProperty(
        name="Kugeltyp",
        items=[
            ('start', 'Start', 'Rote Start-Kugel', 'RADIOBUT_ON', 0),
            ('manual', 'Manual', 'Schwarze Manual-Kugel', 'RADIOBUT_OFF', 1),
            ('painted', 'Paint', 'Blaue Paint-Kugel', 'BRUSH_DATA', 2),
            ('generated', 'Generated', 'Grüne Auto-Kugel', 'AUTO', 3),
        ],
        default='start'
    )

    @classmethod
    def poll(cls, context):
        """Blender 4.4: Verbesserte Poll-Methode"""
        return (context.area and 
                context.area.type == 'VIEW_3D' and 
                context.mode == 'OBJECT')

    def execute(self, context):
        props = context.scene.chain_construction_props
        
        # macOS: Verbesserte Performance durch Context-Caching
        scene = context.scene
        view_layer = context.view_layer
        
        # Auto-Snap Status Info
        snap_info = ""
        if props.kette_target_obj:
            snap_info = f" (Auto-Snap → {props.kette_target_obj.name})"
        else:
            snap_info = " (Setze Zielobjekt für Auto-Snap)"
        
        try:
            # Blender 4.4: Optimierte Sphere-Erstellung
            sphere = sphere_creation.create_sphere_at_cursor(context, self.sphere_type)
            
            if sphere:
                # Type-Namen für User-Feedback
                type_names = {
                    'start': 'Start-Kugel',
                    'manual': 'Manual-Kugel', 
                    'painted': 'Paint-Kugel',
                    'generated': 'Auto-Kugel'
                }
                type_name = type_names.get(self.sphere_type, 'Kugel')
                
                # Blender 4.4: Verbesserte Selection-Handling
                bpy.ops.object.select_all(action='DESELECT')
                sphere.select_set(True)
                view_layer.objects.active = sphere
                
                self.report({'INFO'}, f"{type_name} erstellt: {sphere.name}{snap_info}")
                
                if constants.debug and constants.debug.enabled:
                    constants.debug.info('SPHERES', f"{self.sphere_type} sphere created: {sphere.name}")
            else:
                self.report({'ERROR'}, "Kugel konnte nicht erstellt werden")
                return {'CANCELLED'}
                
        except Exception as e:
            self.report({'ERROR'}, f"Fehler: {str(e)}")
            if constants.debug and constants.debug.enabled:
                constants.debug.error('SPHERES', f"Error creating sphere: {e}")
            return {'CANCELLED'}
        
        return {'FINISHED'}

class KETTE_OT_place_sphere_start(Operator):
    """Quick-Button: Start-Kugel"""
    bl_idname = "kette.place_sphere_start"
    bl_label = "Start-Kugel"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Platziert rote Start-Kugel"
    
    def execute(self, context):
        return bpy.ops.kette.place_sphere(sphere_type='start')

class KETTE_OT_place_sphere_manual(Operator):
    """Quick-Button: Manual-Kugel"""
    bl_idname = "kette.place_sphere_manual"
    bl_label = "Manual-Kugel"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Platziert schwarze Manual-Kugel"
    
    def execute(self, context):
        return bpy.ops.kette.place_sphere(sphere_type='manual')

# =========================================
# SPHERE UTILITIES
# =========================================

class KETTE_OT_duplicate_selected_spheres(Operator):
    """Dupliziert ausgewählte Kugeln (Blender 4.4 optimiert)"""
    bl_idname = "kette.duplicate_selected_spheres"
    bl_label = "Kugeln duplizieren"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Dupliziert ausgewählte Kugeln und registriert sie im System"

    duplicate_type: EnumProperty(
        name="Als Typ",
        items=[
            ('SAME', 'Gleicher Typ', 'Behält Original-Typ'),
            ('MANUAL', 'Als Manual', 'Wird zu Manual-Kugel'),
            ('PAINTED', 'Als Paint', 'Wird zu Paint-Kugel'),
        ],
        default='SAME'
    )

    def execute(self, context):
        # Finde ausgewählte Kugeln
        selected_spheres = [obj for obj in context.selected_objects 
                          if obj.name.startswith("Kugel")]
        
        if not selected_spheres:
            self.report({'ERROR'}, "Keine Kugeln ausgewählt!")
            return {'CANCELLED'}
        
        try:
            duplicated = 0
            
            # Blender 4.4: Optimierte Duplikation
            for sphere in selected_spheres:
                # Original-Properties lesen
                original_type = sphere.get("sphere_type", "manual")
                original_radius = sphere.get("kette_radius", 0.5)
                
                # Neuen Typ bestimmen
                if self.duplicate_type == 'SAME':
                    new_type = original_type
                elif self.duplicate_type == 'MANUAL':
                    new_type = 'manual'
                elif self.duplicate_type == 'PAINTED':
                    new_type = 'painted'
                
                # Duplizierte Position (leicht versetzt)
                new_location = sphere.location.copy()
                new_location.x += original_radius * 2.5  # Versatz
                
                # Neue Kugel erstellen
                new_sphere = sphere_creation.create_sphere(
                    location=new_location,
                    radius=original_radius,
                    sphere_type=new_type
                )
                
                duplicated += 1
            
            self.report({'INFO'}, f"{duplicated} Kugeln dupliziert")
            
        except Exception as e:
            self.report({'ERROR'}, f"Fehler beim Duplizieren: {str(e)}")
            return {'CANCELLED'}
        
        return {'FINISHED'}

class KETTE_OT_select_spheres_by_type(Operator):
    """Wählt alle Kugeln eines bestimmten Typs aus"""
    bl_idname = "kette.select_spheres_by_type"
    bl_label = "Kugeln nach Typ auswählen"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Wählt alle Kugeln eines bestimmten Typs aus"

    sphere_type: EnumProperty(
        name="Kugeltyp",
        items=[
            ('start', 'Start-Kugeln', 'Alle roten Start-Kugeln'),
            ('manual', 'Manual-Kugeln', 'Alle schwarzen Manual-Kugeln'),
            ('painted', 'Paint-Kugeln', 'Alle blauen Paint-Kugeln'),
            ('generated', 'Auto-Kugeln', 'Alle grünen Auto-Kugeln'),
            ('ALL', 'Alle Kugeln', 'Alle Kugeltypen'),
        ],
        default='start'
    )

    add_to_selection: BoolProperty(
        name="Zur Auswahl hinzufügen",
        description="Fügt zur bestehenden Auswahl hinzu statt zu ersetzen",
        default=False
    )

    def execute(self, context):
        try:
            # Aktuelle Auswahl leeren (außer wenn hinzufügen)
            if not self.add_to_selection:
                bpy.ops.object.select_all(action='DESELECT')
            
            selected_count = 0
            
            if self.sphere_type == 'ALL':
                # Alle Kugeln auswählen
                spheres = sphere_creation.get_all_spheres()
            else:
                # Spezifischer Typ
                spheres = sphere_creation.get_spheres_by_type(self.sphere_type)
            
            # Blender 4.4: Optimierte Selection
            for sphere in spheres:
                sphere.select_set(True)
                selected_count += 1
            
            if selected_count > 0:
                # Letztes Objekt als Active setzen
                context.view_layer.objects.active = spheres[-1]
                
                type_name = self.sphere_type if self.sphere_type != 'ALL' else 'alle'
                self.report({'INFO'}, f"{selected_count} {type_name} Kugeln ausgewählt")
            else:
                self.report({'INFO'}, f"Keine {self.sphere_type} Kugeln gefunden")
            
        except Exception as e:
            self.report({'ERROR'}, f"Fehler bei Auswahl: {str(e)}")
            return {'CANCELLED'}
        
        return {'FINISHED'}

# =========================================
# SPHERE MODIFICATION
# =========================================

class KETTE_OT_change_sphere_type(Operator):
    """Ändert den Typ ausgewählter Kugeln"""
    bl_idname = "kette.change_sphere_type"
    bl_label = "Kugeltyp ändern"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Ändert den Typ ausgewählter Kugeln (Farbe und Registry)"

    new_type: EnumProperty(
        name="Neuer Typ",
        items=[
            ('start', 'Start', 'Rot'),
            ('manual', 'Manual', 'Schwarz'),
            ('painted', 'Paint', 'Blau'),
            ('generated', 'Generated', 'Grün'),
        ],
        default='manual'
    )

    def execute(self, context):
        selected_spheres = [obj for obj in context.selected_objects 
                          if obj.name.startswith("Kugel")]
        
        if not selected_spheres:
            self.report({'ERROR'}, "Keine Kugeln ausgewählt!")
            return {'CANCELLED'}
        
        try:
            import kettentool.utils.materials as materials
            
            # Farb-Map
            color_map = {
                'start': (1.0, 0.0, 0.0),      # Rot
                'painted': (0.5, 0.7, 1.0),    # Blau  
                'manual': (0.02, 0.02, 0.02),  # Schwarz
                'generated': (0.0, 1.0, 0.0)   # Grün
            }
            
            changed = 0
            for sphere in selected_spheres:
                # Typ ändern
                old_type = sphere.get("sphere_type", "manual")
                sphere["sphere_type"] = self.new_type
                
                # Material ändern
                mat_name = f"Kugel_{self.new_type}"
                color = color_map.get(self.new_type, (0.02, 0.02, 0.02))
                materials.set_material(sphere, mat_name, color, metallic=0.8, roughness=0.2)
                
                # Registry aktualisieren
                sphere_creation.unregister_sphere(sphere.name)
                sphere_creation.register_sphere(sphere, self.new_type)
                
                changed += 1
            
            self.report({'INFO'}, f"{changed} Kugeln zu {self.new_type} geändert")
            
        except Exception as e:
            self.report({'ERROR'}, f"Fehler beim Typ-Wechsel: {str(e)}")
            return {'CANCELLED'}
        
        return {'FINISHED'}

# =========================================
# REGISTRATION
# =========================================

classes = [
    KETTE_OT_place_sphere,
    KETTE_OT_place_sphere_start,
    KETTE_OT_place_sphere_manual,
    KETTE_OT_duplicate_selected_spheres,
    KETTE_OT_select_spheres_by_type,
    KETTE_OT_change_sphere_type,
]

def register():
    """Registriert Sphere Operators"""
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    """Deregistriert Sphere Operators"""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
