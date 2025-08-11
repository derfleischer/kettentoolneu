"""
Surface Operations - Surface Snapping und Projektions-Funktionen
Optimiert für Blender 4.4 + macOS
"""

import bpy
from bpy.types import Operator
from bpy.props import BoolProperty, EnumProperty, FloatProperty

# Direkte Imports
import kettentool.creation.spheres as sphere_creation
import kettentool.utils.surface as surface_utils
import kettentool.core.constants as constants

# =========================================
# SURFACE SNAPPING OPERATORS
# =========================================

class KETTE_OT_snap_to_surface(Operator):
    """Snappt ausgewählte Kugeln auf Oberfläche (Blender 4.4 optimiert)"""
    bl_idname = "kette.snap_to_surface"
    bl_label = "Snap auf Oberfläche"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Platziert ausgewählte Kugeln korrekt auf die Oberfläche"

    @classmethod
    def poll(cls, context):
        """Blender 4.4: Verbesserte Poll-Validation"""
        props = context.scene.chain_construction_props
        selected_spheres = [obj for obj in context.selected_objects 
                          if obj.name.startswith("Kugel")]
        return props.kette_target_obj and len(selected_spheres) > 0

    def execute(self, context):
        props = context.scene.chain_construction_props
        
        if not props.kette_target_obj:
            self.report({'ERROR'}, "Kein Zielobjekt gewählt!")
            return {'CANCELLED'}
        
        # Finde ausgewählte Kugeln
        selected_spheres = [obj for obj in context.selected_objects 
                          if obj.name.startswith("Kugel")]
        
        if not selected_spheres:
            self.report({'ERROR'}, "Keine Kugeln ausgewählt!")
            return {'CANCELLED'}
        
        try:
            snapped = 0
            failed = 0
            total = len(selected_spheres)
            
            # macOS: Progress-Bar für viele Objekte
            if total > 10:
                context.window_manager.progress_begin(0, total)
            
            try:
                for i, sphere in enumerate(selected_spheres):
                    if sphere_creation.snap_sphere_to_surface(
                        sphere, props.kette_target_obj, props
                    ):
                        snapped += 1
                    else:
                        failed += 1
                    
                    # Progress Update
                    if total > 10:
                        context.window_manager.progress_update(i)
            
            finally:
                if total > 10:
                    context.window_manager.progress_end()
            
            if snapped > 0:
                msg = f"{snapped} Kugeln gesnapped"
                if failed > 0:
                    msg += f" ({failed} fehlgeschlagen)"
                self.report({'INFO'}, msg)
                
                if constants.debug and constants.debug.enabled:
                    constants.debug.info('SURFACE', f"Snapped {snapped} spheres, {failed} failed")
            else:
                self.report({'ERROR'}, "Keine Kugeln konnten gesnapped werden")
                return {'CANCELLED'}
                
        except Exception as e:
            self.report({'ERROR'}, f"Fehler beim Snapping: {str(e)}")
            if constants.debug and constants.debug.enabled:
                constants.debug.error('SURFACE', f"Error snapping spheres: {e}")
            return {'CANCELLED'}
        
        return {'FINISHED'}

class KETTE_OT_snap_all_spheres(Operator):
    """Snappt ALLE Kugeln auf Oberfläche"""
    bl_idname = "kette.snap_all_spheres"
    bl_label = "Alle Kugeln snappen"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Snappt alle Kugeln auf die Oberfläche"

    sphere_filter: EnumProperty(
        name="Kugelfilter",
        items=[
            ('ALL', 'Alle', 'Alle Kugeltypen'),
            ('START', 'Start', 'Nur Start-Kugeln'),
            ('MANUAL', 'Manual', 'Nur Manual-Kugeln'),
            ('PAINTED', 'Paint', 'Nur Paint-Kugeln'),
            ('GENERATED', 'Generated', 'Nur Auto-Kugeln'),
        ],
        default='ALL'
    )

    def execute(self, context):
        props = context.scene.chain_construction_props
        
        if not props.kette_target_obj:
            self.report({'ERROR'}, "Kein Zielobjekt gewählt!")
            return {'CANCELLED'}
        
        # Kugeln basierend auf Filter sammeln
        if self.sphere_filter == 'ALL':
            spheres = sphere_creation.get_all_spheres()
        else:
            type_map = {
                'START': 'start',
                'MANUAL': 'manual',
                'PAINTED': 'painted',
                'GENERATED': 'generated'
            }
            sphere_type = type_map[self.sphere_filter]
            spheres = sphere_creation.get_spheres_by_type(sphere_type)
        
        if not spheres:
            self.report({'INFO'}, f"Keine {self.sphere_filter.lower()} Kugeln vorhanden")
            return {'FINISHED'}
        
        # Alle auswählen
        bpy.ops.object.select_all(action='DESELECT')
        for sphere in spheres:
            sphere.select_set(True)
        
        # Delegate to main snap operator
        return bpy.ops.kette.snap_to_surface()

    def invoke(self, context, event):
        """Zeigt Filter-Dialog"""
        return context.window_manager.invoke_props_dialog(self)

# =========================================
# SURFACE PROJECTION TOOLS
# =========================================

class KETTE_OT_project_spheres_direction(Operator):
    """Projiziert Kugeln in bestimmte Richtung auf Oberfläche"""
    bl_idname = "kette.project_spheres_direction"
    bl_label = "Richtungsprojektion"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Projiziert Kugeln in bestimmte Richtung auf Oberfläche"

    projection_direction: EnumProperty(
        name="Projektionsrichtung",
        items=[
            ('NEGATIVE_Z', 'Nach unten (-Z)', 'Projiziert nach unten'),
            ('POSITIVE_Z', 'Nach oben (+Z)', 'Projiziert nach oben'),
            ('NEGATIVE_Y', 'Nach hinten (-Y)', 'Projiziert nach hinten'),
            ('POSITIVE_Y', 'Nach vorne (+Y)', 'Projiziert nach vorne'),
            ('NEGATIVE_X', 'Nach links (-X)', 'Projiziert nach links'),
            ('POSITIVE_X', 'Nach rechts (+X)', 'Projiziert nach rechts'),
            ('VIEW', 'Aus Sicht', 'Projiziert aus aktueller Sicht'),
        ],
        default='NEGATIVE_Z'
    )

    max_distance: FloatProperty(
        name="Max. Distanz",
        description="Maximale Projektionsdistanz",
        default=10.0,
        min=0.1,
        max=100.0,
        subtype='DISTANCE'
    )

    def execute(self, context):
        from mathutils import Vector
        import bmesh
        
        props = context.scene.chain_construction_props
        
        if not props.kette_target_obj:
            self.report({'ERROR'}, "Kein Zielobjekt gewählt!")
            return {'CANCELLED'}
        
        selected_spheres = [obj for obj in context.selected_objects 
                          if obj.name.startswith("Kugel")]
        
        if not selected_spheres:
            self.report({'ERROR'}, "Keine Kugeln ausgewählt!")
            return {'CANCELLED'}
        
        try:
            # Projektionsrichtung bestimmen
            direction_map = {
                'NEGATIVE_Z': Vector((0, 0, -1)),
                'POSITIVE_Z': Vector((0, 0, 1)),
                'NEGATIVE_Y': Vector((0, -1, 0)),
                'POSITIVE_Y': Vector((0, 1, 0)),
                'NEGATIVE_X': Vector(-1, 0, 0),
                'POSITIVE_X': Vector((1, 0, 0)),
            }
            
            if self.projection_direction == 'VIEW':
                # Aus aktueller Kamera-Sicht
                region = context.region
                rv3d = context.region_data
                direction = rv3d.view_rotation @ Vector((0, 0, -1))
            else:
                direction = direction_map[self.projection_direction]
            
            # BVH für Target-Objekt erstellen
            target_bvh = surface_utils._build_bvh(props.kette_target_obj)
            
            projected = 0
            
            for sphere in selected_spheres:
                # Ray-Cast in Projektionsrichtung
                hit_loc, hit_norm, hit_idx, hit_dist = target_bvh.ray_cast(
                    sphere.location, direction, self.max_distance
                )
                
                if hit_loc:
                    # Position mit Offset berechnen
                    offset = (props.kette_kugelradius + props.kette_abstand + 
                             props.kette_global_offset + constants.EPSILON)
                    
                    new_location = hit_loc + hit_norm.normalized() * offset
                    sphere.location = new_location
                    
                    # Registry aktualisieren
                    sphere_creation.update_sphere_registry(sphere)
                    projected += 1
            
            if projected > 0:
                direction_name = self.projection_direction.replace('_', ' ').title()
                self.report({'INFO'}, f"{projected} Kugeln projiziert ({direction_name})")
            else:
                self.report({'WARNING'}, "Keine Kugeln konnten projiziert werden")
            
        except Exception as e:
            self.report({'ERROR'}, f"Fehler bei Projektion: {str(e)}")
            return {'CANCELLED'}
        
        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

# =========================================
# SURFACE ANALYSIS TOOLS
# =========================================

class KETTE_OT_analyze_surface_coverage(Operator):
    """Analysiert Oberflächenabdeckung der Kugeln"""
    bl_idname = "kette.analyze_surface_coverage"
    bl_label = "Oberflächenabdeckung analysieren"
    bl_options = {'REGISTER'}
    bl_description = "Analysiert wie gut die Kugeln die Oberfläche abdecken"

    def execute(self, context):
        props = context.scene.chain_construction_props
        
        if not props.kette_target_obj:
            self.report({'ERROR'}, "Kein Zielobjekt gewählt!")
            return {'CANCELLED'}
        
        try:
            spheres = sphere_creation.get_all_spheres()
            
            if not spheres:
                self.report({'INFO'}, "Keine Kugeln vorhanden")
                return {'FINISHED'}
            
            # Vereinfachte Abdeckungsanalyse
            target_obj = props.kette_target_obj
            
            # Surface Area schätzen (vereinfacht)
            import bmesh
            bm = bmesh.new()
            bm.from_mesh(target_obj.data)
            bm.transform(target_obj.matrix_world)
            
            surface_area = sum(face.calc_area() for face in bm.faces)
            bm.free()
            
            # Sphere Coverage berechnen
            total_coverage = 0
            for sphere in spheres:
                radius = sphere.get("kette_radius", props.kette_kugelradius)
                coverage_area = 3.14159 * radius * radius  # Projected area
                total_coverage += coverage_area
            
            coverage_percentage = min(100.0, (total_coverage / surface_area) * 100)
            
            # Console Report
            print("\n" + "="*50)
            print("SURFACE COVERAGE ANALYSIS")
            print("="*50)
            print(f"Target Object: {target_obj.name}")
            print(f"Surface Area: {surface_area:.2f}")
            print(f"Spheres: {len(spheres)}")
            print(f"Coverage: {coverage_percentage:.1f}%")
            print("="*50 + "\n")
            
            self.report({'INFO'}, 
                       f"Abdeckung: {coverage_percentage:.1f}% ({len(spheres)} Kugeln)")
            
        except Exception as e:
            self.report({'ERROR'}, f"Fehler bei Analyse: {str(e)}")
            return {'CANCELLED'}
        
        return {'FINISHED'}

class KETTE_OT_clear_surface_caches(Operator):
    """Leert Surface-spezifische Caches"""
    bl_idname = "kette.clear_surface_caches"
    bl_label = "Surface-Caches leeren"
    bl_options = {'REGISTER'}
    bl_description = "Leert Aura- und BVH-Caches für bessere Performance"

    def execute(self, context):
        try:
            # Surface-spezifische Cache-Counts
            aura_count = len(constants._aura_cache)
            bvh_count = len(constants._bvh_cache)
            
            # Clear surface caches
            surface_utils.clear_auras()
            
            total_cleared = aura_count + bvh_count
            self.report({'INFO'}, f"Surface-Caches geleert: {total_cleared} Einträge")
            
            if constants.debug and constants.debug.enabled:
                constants.debug.info('SURFACE', 
                                   f"Cleared surface caches: {aura_count} auras, {bvh_count} BVH")
            
        except Exception as e:
            self.report({'ERROR'}, f"Fehler beim Cache-Leeren: {str(e)}")
            return {'CANCELLED'}
        
        return {'FINISHED'}

# =========================================
# REGISTRATION
# =========================================

classes = [
    KETTE_OT_snap_to_surface,
    KETTE_OT_snap_all_spheres,
    KETTE_OT_project_spheres_direction,
    KETTE_OT_analyze_surface_coverage,
    KETTE_OT_clear_surface_caches,
]

def register():
    """Registriert Surface Operators"""
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    """Deregistriert Surface Operators"""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
