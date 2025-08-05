bl_info = {
    "name": "Chain Construction Advanced (Kettenkonstruktion)",
    "author": "Michael Fleischer & Contributors",
    "version": (3, 7, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Chain Tools",
    "description": "Advanced chain hopping pattern generator with paint library support",
    "category": "Mesh",
}

import bpy
import bmesh
import math
import json
import os
import time
import random
import pathlib
import gpu
import gpu_extras
from gpu_extras.batch import batch_for_shader
from bpy_extras import view3d_utils
from math import radians, sqrt, pi, sin, cos, degrees, exp
from mathutils import Vector, Matrix, Quaternion
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree
from collections import deque, defaultdict
from typing import List, Tuple, Dict, Optional, Set, Any
from functools import wraps
from bpy.types import Panel, Operator, PropertyGroup, AddonPreferences
from bpy.props import (
    PointerProperty, EnumProperty, BoolProperty, IntProperty,
    FloatProperty, StringProperty, CollectionProperty, FloatVectorProperty
)

# =========================================
# GLOBAL VARIABLES & CONSTANTS
# =========================================

_bvh_cache: Dict[str, Tuple[BVHTree, Matrix, int]] = {}
_aura_cache: Dict[str, bpy.types.Object] = {}
_existing_connectors: Dict[Tuple[str, str], bpy.types.Object] = {}
_remembered_pairs: Set[Tuple[str, str]] = set()
_paint_stroke_buffer: List[Vector] = []  # Buffer für aufgenommene Strokes
_paint_pattern_generators: Dict[str, callable] = {}  # Pattern Generator Funktionen

EPSILON = 0.02
DEBUG_MODE = False

# =========================================
# DEBUG SYSTEM
# =========================================

class DebugLogger:
    """Verbessertes Debug-System mit Kategorien und Levels"""
    
    LEVELS = {
        'ERROR': 0,
        'WARNING': 1,
        'INFO': 2,
        'DEBUG': 3,
        'TRACE': 4
    }
    
    def __init__(self):
        self.enabled = False
        self.level = 'INFO'
        self.categories = {
            'AURA': True,
            'HOPPING': True,
            'PAINT': True,
            'CONNECTOR': True,
            'PERFORMANCE': True,
            'GENERAL': True,
            'CACHE': True
        }
    
    def log(self, level: str, category: str, message: str, **kwargs):
        if not self.enabled:
            return
            
        if self.LEVELS.get(level, 5) > self.LEVELS.get(self.level, 2):
            return
            
        if not self.categories.get(category, False):
            return
            
        prefix = f"[{level}][{category}]"
        print(f"{prefix} {message}")
        
        if kwargs:
            for key, value in kwargs.items():
                print(f"  {key}: {value}")
    
    def trace(self, category: str, message: str, **kwargs):
        self.log('TRACE', category, message, **kwargs)
        
    def debug(self, category: str, message: str, **kwargs):
        self.log('DEBUG', category, message, **kwargs)
        
    def info(self, category: str, message: str, **kwargs):
        self.log('INFO', category, message, **kwargs)
        
    def warning(self, category: str, message: str, **kwargs):
        self.log('WARNING', category, message, **kwargs)
        
    def error(self, category: str, message: str, **kwargs):
        self.log('ERROR', category, message, **kwargs)
        
    def performance(self, category: str, operation: str, duration: float):
        """Speziell für Performance-Logging"""
        if self.categories.get('PERFORMANCE', False):
            self.debug(category, f"{operation} took {duration:.3f}s")

# Globale Debug-Instanz
debug = DebugLogger()

# =========================================
# PERFORMANCE MONITOR
# =========================================

class PerformanceMonitor:
    def __init__(self):
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.counts: Dict[str, int] = defaultdict(int)
    
    def time_function(self, name: str, category: str = 'GENERAL'):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                self.counts[name] += 1
                start = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    elapsed = time.time() - start
                    self.timings[name].append(elapsed)
                    debug.performance(category, name, elapsed)
                    return result
                except Exception as e:
                    elapsed = time.time() - start
                    debug.error(category, f"{name} failed after {elapsed:.3f}s: {str(e)}")
                    raise
                    
            return wrapper
        return decorator
    
    def report(self) -> str:
        lines = ["Performance Report:"]
        lines.append("-" * 50)
        
        for name, times in sorted(self.timings.items()):
            if times:
                avg = sum(times) / len(times)
                total = sum(times)
                min_time = min(times)
                max_time = max(times)
                count = self.counts[name]
                
                lines.append(f"{name:30} | Calls: {count:4} | Avg: {avg:6.3f}s | Total: {total:6.3f}s | Min: {min_time:6.3f}s | Max: {max_time:6.3f}s")
                
        return "\n".join(lines)

performance_monitor = PerformanceMonitor()


# =========================================
# BACKUP/ERROR-HANDLING
# =========================================

class BackupState:
    def __init__(self):
        self.objects: List[Dict[str, Any]] = []
        self.collections: Dict[str, List[str]] = {}
    
    def save_object(self, obj: bpy.types.Object):
        self.objects.append({
            'name': obj.name,
            'location': obj.location.copy(),
            'rotation': obj.rotation_euler.copy(),
            'scale': obj.scale.copy(),
            'data': obj.data.name if obj.data else None
        })
    
    def save_collection(self, collection: bpy.types.Collection):
        self.collections[collection.name] = [o.name for o in collection.objects]

def safe_operation(create_backup: bool = True):
    def decorator(func):
        @wraps(func)
        def wrapper(self, context, *args, **kwargs):
            backup = None
            try:
                if create_backup and hasattr(self, 'create_backup'):
                    backup = self.create_backup(context)
                
                debug.debug('GENERAL', f"Executing {func.__name__}")
                result = func(self, context, *args, **kwargs)
                
                if result == {'CANCELLED'} and backup:
                    self.restore_backup(context, backup)
                    
                return result
                
            except Exception as e:
                debug.error('GENERAL', f"Error in {func.__name__}: {str(e)}")
                
                if hasattr(self, 'report'):
                    self.report({'ERROR'}, f"Fehler: {str(e)}")
                    
                if backup and hasattr(self, 'restore_backup'):
                    self.restore_backup(context, backup)
                    
                if debug.enabled:
                    import traceback
                    traceback.print_exc()
                    
                return {'CANCELLED'}
        return wrapper
    return decorator

# =========================================
# UTILITY FUNCTIONS
# =========================================

@performance_monitor.time_function("ensure_collection", "GENERAL")
def ensure_collection(name: str = "KonstruktionsKette") -> bpy.types.Collection:
    col = bpy.data.collections.get(name)
    if not col:
        debug.info('GENERAL', f"Creating collection: {name}")
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col

@performance_monitor.time_function("set_material", "GENERAL")
def set_material(obj: bpy.types.Object, mat_name: str, color: Tuple[float, float, float],
                 alpha: float = 1.0, metallic: float = 0.0, roughness: float = 0.5):
    mat = bpy.data.materials.get(mat_name)
    if not mat:
        debug.trace('GENERAL', f"Creating material: {mat_name}")
        mat = bpy.data.materials.new(mat_name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get('Principled BSDF')
        if bsdf:
            bsdf.inputs['Base Color'].default_value = (*color, 1)
            bsdf.inputs['Alpha'].default_value = alpha
            bsdf.inputs['Metallic'].default_value = metallic
            bsdf.inputs['Roughness'].default_value = roughness
        mat.blend_method = 'BLEND' if alpha < 1.0 else 'OPAQUE'
    if obj.data and hasattr(obj.data, 'materials'):
        if len(obj.data.materials) > 0:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)
            
# ==================================================
#  Library Funktionen
# ==================================================

def load_pattern_library() -> Dict[str, Dict[str, Any]]:
    """Lädt Chain-Patterns aus benutzerdefiniertem Ordner"""
    patterns = {}
    
    try:
        props = bpy.context.scene.chain_construction_props
        
        # Bestimme Library-Pfad
        if props.chain_library_path and pathlib.Path(props.chain_library_path).exists():
            lib_path = pathlib.Path(props.chain_library_path)
        else:
            lib_path = pathlib.Path.home() / "Documents" / "chain_patterns"
            lib_path.mkdir(parents=True, exist_ok=True)
        
        # Lade alle JSON-Dateien
        json_files = list(lib_path.rglob("*.json"))
        
        if len(json_files) == 0:
            # Erstelle Beispiel-Patterns automatisch
            create_example_patterns()
            json_files = list(lib_path.rglob("*.json"))
        
        for json_file in json_files:
            # Überspringe Metadaten-Dateien
            if json_file.name.startswith('_'):
                continue
                
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Validiere Pattern-Daten
                if "parameters" not in data:
                    continue
                
                # Pattern-ID aus Dateiname und Pfad
                relative_path = json_file.relative_to(lib_path)
                pattern_id = str(relative_path.with_suffix(''))
                
                # Korrigiere Category basierend auf Ordnerstruktur
                if relative_path.parent != pathlib.Path('.'):
                    data["category"] = relative_path.parent.name
                else:
                    data.setdefault("category", "custom")
                
                patterns[pattern_id] = data
                
            except (json.JSONDecodeError, Exception):
                continue
        
        return patterns
        
    except Exception:
        return {}
    
    

def save_pattern_to_library(pattern_data: Dict[str, Any], pattern_id: str, category: str = "custom"):
    """Speichert Chain-Pattern in Library"""
    try:
        props = bpy.context.scene.chain_construction_props
        if props.chain_library_path and pathlib.Path(props.chain_library_path).exists():
            lib_path = pathlib.Path(props.chain_library_path)
        else:
            lib_path = pathlib.Path.home() / "Documents" / "chain_patterns"
        
        category_path = lib_path / category
        category_path.mkdir(parents=True, exist_ok=True)
        
        # Ergänze Metadaten
        pattern_data.setdefault("category", category)
        pattern_data.setdefault("metadata", {})
        pattern_data["metadata"].update({
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "blender_version": bpy.app.version_string,
            "addon_version": "3.7.0"
        })
        
        file_path = category_path / f"{pattern_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(pattern_data, f, indent=2, ensure_ascii=False)
        
        return True
        
    except Exception:
        return False


# =========================================
# NEUE ZENTRALE KUGEL-ERSTELLUNG
# =========================================

def create_sphere(location, radius=None, name_prefix="Kugel", collection_name="KonstruktionsKette", 
                  material_name="KetteMaterial", material_color=(0.5, 0.7, 1.0), custom_props=None):
    """
    Zentrale Funktion zur Kugel-Erstellung für konsistente Benennung
    """
    props = bpy.context.scene.chain_construction_props
    if radius is None:
        radius = props.kette_kugelradius
        
    # Erstelle Mesh
    mesh = bpy.data.meshes.new(name=f"{name_prefix}_mesh")
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(
        bm,
        u_segments=props.kette_aufloesung,
        v_segments=props.kette_aufloesung,
        radius=radius
    )
    bm.to_mesh(mesh)
    bm.free()
    
    # Erstelle Objekt
    sphere = bpy.data.objects.new(name_prefix, mesh)
    sphere.location = location
    sphere["kette_radius"] = radius
    
    # Custom Properties
    if custom_props:
        for key, value in custom_props.items():
            sphere[key] = value
    
    # Zur Collection hinzufügen
    coll = ensure_collection(collection_name)
    coll.objects.link(sphere)
    
    # Material setzen
    set_material(sphere, material_name, material_color, metallic=0.5)
    
    return sphere

def get_aura(target_obj: bpy.types.Object, thickness: float) -> bpy.types.Object:
    if not target_obj or target_obj.name not in bpy.data.objects:
        clear_auras()
        raise ReferenceError("Target object has been removed")
    
    # SICHERHEITS-MINIMUM: Thickness nie unter 0.1
    thickness = max(thickness, 0.1)
    
    cache_key = f"{target_obj.name}_{thickness:.4f}"
    
    if cache_key in _aura_cache:
        aura = _aura_cache[cache_key]
        try:
            if aura.name in bpy.data.objects:
                return aura
        except ReferenceError:
            pass
        _aura_cache.pop(cache_key, None)
    
    # Cleanup alte Auras (unverändert)
    to_remove = []
    for key in list(_aura_cache.keys()):
        if key.startswith(f"{target_obj.name}_") and key != cache_key:
            to_remove.append(key)
    
    for key in to_remove:
        old_aura = _aura_cache.pop(key, None)
        if old_aura:
            try:
                bpy.data.objects.remove(old_aura, do_unlink=True)
            except Exception:
                pass
    
    # Aura erstellen (unverändert)
    aura = target_obj.copy()
    aura.data = target_obj.data.copy()
    aura.name = f"{target_obj.name}_CachedAura_{thickness:.4f}"
    bpy.context.collection.objects.link(aura)
    aura.hide_viewport = True
    aura.hide_render = True
    aura.hide_set(True)
    
    # Normale berechnen (unverändert)
    bm = bmesh.new()
    bm.from_mesh(aura.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(aura.data)
    bm.free()
    
    # EINZIGE VERBESSERUNG: Solidify robuster machen
    mod = aura.modifiers.new("Solidify", 'SOLIDIFY')
    mod.thickness = thickness
    mod.offset = 1.0
    mod.use_even_offset = True  # NEU: Gleichmäßigerer Offset
    
    bpy.context.view_layer.objects.active = aura
    try:
        bpy.ops.object.modifier_apply(modifier=mod.name)
    except Exception as e:
        print(f"Solidify warning for {target_obj.name}: {e}")
        # Modifier entfernen falls Apply fehlschlägt
        if mod.name in aura.modifiers:
            aura.modifiers.remove(mod)
    
    # NEU: ROBUSTE NORMAL-KORREKTUR für konkave Formen
    def fix_aura_normals(aura_obj):
        """Korrigiere Aura-Normale für konkave/konvexe Bereiche"""
        import bmesh
        from mathutils import Vector
        
        bm = bmesh.new()
        bm.from_mesh(aura_obj.data)
        
        # Berechne Mesh-Zentrum (Schwerpunkt)
        center = sum((v.co for v in bm.verts), Vector()) / len(bm.verts)
        
        # Korrigiere Face-Normale basierend auf Zentrum-Distanz
        corrected_faces = 0
        for face in bm.faces:
            face_center = face.calc_center_median()
            outward_dir = (face_center - center).normalized()
            
            # Wenn Face-Normal nicht nach außen zeigt, korrigiere
            if face.normal.dot(outward_dir) < 0:
                face.normal_flip()
                corrected_faces += 1
        
        print(f"Normal-Korrektur: {corrected_faces}/{len(bm.faces)} Faces korrigiert")
        
        # Final recalc für Konsistenz
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(aura_obj.data)
        bm.free()
        
        # Update mesh
        aura_obj.data.update()
    
    # ANWENDEN: Normal-Korrektur nach Solidify
    try:
        fix_aura_normals(aura)
        print(f"Aura {aura.name}: Normale für konkave Form korrigiert")
    except Exception as e:
        print(f"Normal-Korrektur Fehler: {e}")
    
    # Final cleanup (unverändert)
    bm2 = bmesh.new()
    bm2.from_mesh(aura.data)
    bmesh.ops.recalc_face_normals(bm2, faces=bm2.faces)
    bm2.to_mesh(aura.data)
    bm2.free()
    
    _aura_cache[cache_key] = aura
    return aura


def _build_bvh(obj: bpy.types.Object) -> BVHTree:
    if obj.name in _bvh_cache:
        cached_bvh, cached_matrix, cached_id = _bvh_cache[obj.name]
        if cached_id == id(obj.data) and cached_matrix == obj.matrix_world:
            debug.trace('CACHE', f"Using cached BVH for {obj.name}")
            return cached_bvh

    debug.trace('CACHE', f"Building new BVH for {obj.name}")
    
    # VERBESSERT: Normalen vor BVH-Erstellung korrigieren
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    
    # KRITISCH: Normale konsistent nach außen ausrichten
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    
    # Zusätzlich: Stelle sicher dass Normalen nach außen zeigen
    # Teste mit Bounding Box Center
    bbox_center = sum((v.co for v in bm.verts), Vector()) / len(bm.verts)
    
    # Prüfe erste paar Faces und korrigiere falls nötig
    outward_votes = 0
    for face in bm.faces[:min(10, len(bm.faces))]:
        face_center = face.calc_center_median()
        to_bbox = (face_center - bbox_center).normalized()
        if face.normal.dot(to_bbox) > 0:
            outward_votes += 1
        else:
            outward_votes -= 1
    
    # Wenn Mehrheit nach innen zeigt, flippe alle Normalen
    if outward_votes < 0:
        print(f"Flipping normals for {obj.name} - detected inside-out mesh")
        for face in bm.faces:
            face.normal_flip()
    
    # Transformiere und erstelle BVH
    bm.transform(obj.matrix_world)
    bvh = BVHTree.FromBMesh(bm)
    bm.free()
    
    _bvh_cache[obj.name] = (bvh, obj.matrix_world.copy(), id(obj.data))
    return bvh


def clear_auras():
    """Löscht alle gecachten Auras"""
    debug.info('AURA', "Clearing all auras")
    auras_to_remove = []

    for cache_key, aura in list(_aura_cache.items()):
        try:
            if aura.name in bpy.data.objects:
                auras_to_remove.append(aura)
        except ReferenceError:
            pass

    for aura in auras_to_remove:
        try:
            bpy.data.objects.remove(aura, do_unlink=True)
        except Exception:
            pass

    _aura_cache.clear()
    _bvh_cache.clear()

@performance_monitor.time_function("next_surface_point", "AURA")
def next_surface_point(aura_obj: bpy.types.Object, pos: Vector, ray_direction: Vector = None) -> Tuple[Vector, Vector]:
    """
    - Prüft ob Position bereits korrekt ist
    - Minimale Bewegung wenn bereits nah an Oberfläche
    - Verhindert Oszillation zwischen Volumen und Oberfläche
    """
    if not aura_obj or aura_obj.type != 'MESH':
        return pos, Vector((0, 0, 1))
    
    bvh = _build_bvh(aura_obj)
    if not bvh:
        return pos, Vector((0, 0, 1))
    
    # STABILITÄTS-CHECK: Ist die Position bereits gut?
    nearest_result = bvh.find_nearest(pos)
    if not nearest_result:
        return pos, Vector((0, 0, 1))
    
    nearest_point, nearest_normal = nearest_result[0], nearest_result[1].normalized()
    nearest_distance = (pos - nearest_point).length
    
    # STABILITÄTS-SCHWELLWERT: Wenn bereits sehr nah, nicht bewegen!
    STABILITY_THRESHOLD = 0.1  # 10cm Toleranz
    
    if nearest_distance < STABILITY_THRESHOLD:
        # Position ist bereits gut - nur Normal korrigieren
        direction_from_pos = (nearest_point - pos)
        if direction_from_pos.length > 0.001:
            direction_from_pos = direction_from_pos.normalized()
            if nearest_normal.dot(direction_from_pos) > 0:
                nearest_normal = -nearest_normal
        
        print(f"[NSP] STABIL: Bereits nah genug ({nearest_distance:.4f}), keine Bewegung")
        return pos, nearest_normal  # KEINE POSITIONSÄNDERUNG!
    
    # Wenn weiter weg, normale Logik...
    rounded_pos = pos.copy()
    
    # URSPRÜNGLICHE RAYCAST-RICHTUNGEN
    test_directions = [
        Vector((0, 0, -1)),   # Z-down (IMMER erste Priorität)
        Vector((0, 0, 1)),    # Z-up
        Vector((0, 1, 0)),    # Y+
        Vector((0, -1, 0)),   # Y-
        Vector((1, 0, 0)),    # X+
        Vector((-1, 0, 0)),   # X-
    ]
    
    # Sammle Raycast-Hits zur Validierung
    valid_raycast_hits = []
    for i, direction in enumerate(test_directions):
        hit_loc, hit_norm, hit_index, hit_dist = bvh.ray_cast(rounded_pos, direction, 6.0)
        
        if hit_loc and hit_dist > 0.001:
            ray_to_hit = (hit_loc - rounded_pos).normalized()
            normal_quality = hit_norm.dot(-ray_to_hit)
            
            # TOLERANTERER Filter für Stabilität
            if normal_quality > -0.95:
                distance_rounded = round(hit_dist, 2)
                
                # URSPRÜNGLICHE Priorität-Logik
                if i == 0:  # Z-down
                    priority = distance_rounded * 0.1
                elif hit_norm.z < -0.3:
                    priority = distance_rounded * 0.5
                else:
                    priority = 100 + i * 10 + distance_rounded
                
                valid_raycast_hits.append((hit_loc, hit_norm, hit_dist, priority, normal_quality))
    
    # ENTSCHEIDUNGSLOGIK MIT VERSTÄRKTER HYSTERESE
    if valid_raycast_hits:
        valid_raycast_hits.sort(key=lambda x: x[3])
        best_raycast = valid_raycast_hits[0]
        raycast_distance = (rounded_pos - best_raycast[0]).length
        
        # VERSTÄRKTE HYSTERESE: Größerer Schwellwert
        distance_difference = abs(raycast_distance - nearest_distance)
        
        if distance_difference < 1.0:  # VERGRÖSSERT von 0.3 auf 1.0
            # Bevorzuge find_nearest für Stabilität
            surface_location, surface_normal = nearest_point, nearest_normal
            decision = "FIND_NEAREST (Stabilität)"
        elif raycast_distance < nearest_distance * 0.5:  # Schärferer Schwellwert
            surface_location, surface_normal = best_raycast[0], best_raycast[1].normalized()
            decision = "RAYCAST (Deutlich besser)"
        else:
            surface_location, surface_normal = nearest_point, nearest_normal
            decision = "FIND_NEAREST (Standard)"
    else:
        surface_location, surface_normal = nearest_point, nearest_normal
        decision = "FIND_NEAREST (Fallback)"
    
    # URSPRÜNGLICHE Normal-Orientierung
    direction_from_pos = (surface_location - rounded_pos)
    if direction_from_pos.length > 0.001:
        direction_from_pos = direction_from_pos.normalized()
        if surface_normal.dot(direction_from_pos) > 0:
            surface_normal = -surface_normal
    
    move_distance = (pos - surface_location).length
    print(f"[NSP] BEWEGUNG: {move_distance:.4f}, Decision: {decision}")
    
    return surface_location, surface_normal

def refresh_connectors():
    debug.info('CONNECTOR', "Refreshing all connectors")
    coll = bpy.data.collections.get("KonstruktionsKette")
    if not coll:
        return 0, 0

    updated = 0
    deleted = 0

    for obj in list(coll.objects):
        k1 = obj.get("Kette_Kugel1")
        k2 = obj.get("Kette_Kugel2")
        if not (k1 and k2):
            continue

        o1 = bpy.data.objects.get(k1)
        o2 = bpy.data.objects.get(k2)
        
        if o1 and o2:
            if obj.type == 'CURVE':
                _update_curve(obj, o1, o2)
            else:
                _update_cylinder(obj, o1, o2)
            updated += 1
        else:
            for c in list(obj.users_collection):
                c.objects.unlink(obj)
            bpy.data.objects.remove(obj, do_unlink=True)
            deleted += 1

    debug.info('CONNECTOR', f"Updated: {updated}, Deleted: {deleted}")
    return updated, deleted


def update_chain():
    """Rebuild chain considering only desired connections"""
    global _existing_connectors
    props = bpy.context.scene.chain_construction_props
    coll = ensure_collection("KonstruktionsKette")
    kugeln = [o for o in coll.objects if o.name.startswith("Kugel")]
    curved = props.kette_use_curved_connectors

    desired = set()
    existing = set(_existing_connectors.keys())
    
    # Determine desired connections
    if props.kette_conn_mode == 'SEQ':
        kugeln.sort(key=lambda x: x.name)
        for i in range(len(kugeln)-1):
            key = tuple(sorted((kugeln[i].name, kugeln[i+1].name)))
            desired.add(key)
    else:
        # NEIGHBOR mode: respect max_dist and max_links
        max_dist = props.kette_max_abstand
        max_links = props.kette_max_links
        
        for k1 in kugeln:
            candidates = []
            for k2 in kugeln:
                if k1 == k2:
                    continue
                dist = (k1.location - k2.location).length
                if max_dist == 0 or dist <= max_dist:
                    candidates.append((dist, k2))
            
            candidates.sort(key=lambda x: x[0])
            for _, k2 in candidates[:max_links]:
                key = tuple(sorted((k1.name, k2.name)))
                desired.add(key)
    
    # Delete unwanted connectors
    deleted = []
    created = []
    
    for key in existing - desired:
        if key in _existing_connectors:
            try:
                obj = _existing_connectors[key]
                if obj.name in bpy.data.objects:
                    deleted.append(obj.name)
                    for c in list(obj.users_collection):
                        c.objects.unlink(obj)
                    bpy.data.objects.remove(obj, do_unlink=True)
            finally:
                if key in _existing_connectors:
                    del _existing_connectors[key]

    # Create missing connectors
    name2obj = {o.name: o for o in kugeln}
    before = set(_existing_connectors.keys())
    for key in desired:
        if key not in _existing_connectors:
            o1 = name2obj.get(key[0])
            o2 = name2obj.get(key[1])
            if o1 and o2:
                connect_kugeln(o1, o2, curved)
    after = set(_existing_connectors.keys())

    for key in after - before:
        if key in _existing_connectors:
            created.append(_existing_connectors[key].name)

    debug.info('CONNECTOR', f"Created {len(created)}, deleted {len(deleted)} connectors")
    return deleted, created


@performance_monitor.time_function("connect_kugeln", "CONNECTOR")
def connect_kugeln(o1: bpy.types.Object, o2: bpy.types.Object, use_curved: bool):
    global _remembered_pairs, _existing_connectors
    props = bpy.context.scene.chain_construction_props
    coll = ensure_collection("KonstruktionsKette")
    key = tuple(sorted((o1.name, o2.name)))
    
    debug.trace('CONNECTOR', f"Connecting {o1.name} to {o2.name}, curved={use_curved}")
    
    # Check if already connected
    if any(obj for obj in coll.objects
           if obj.get("Kette_Kugel1") == key[0] and obj.get("Kette_Kugel2") == key[1]):
        return
    
    try:
        aura = get_aura(props.kette_target_obj, props.kette_aura_dichte)
    except ReferenceError:
        clear_auras()
        aura = get_aura(props.kette_target_obj, props.kette_aura_dichte)
    
    if use_curved:
        obj = _create_curve(o1, o2)
    else:
        obj = _create_cylinder(o1, o2)
    
    coll.objects.link(obj)
    obj["Kette_Kugel1"] = o1.name
    obj["Kette_Kugel2"] = o2.name
    set_material(obj, "KetteSchwarz", (0.02, 0.02, 0.02), metallic=0.8, roughness=0.2)
    _remembered_pairs.add(key)
    _existing_connectors[key] = obj

def _create_cylinder(o1: bpy.types.Object, o2: bpy.types.Object) -> bpy.types.Object:
    props = bpy.context.scene.chain_construction_props
    vec = o2.location - o1.location
    axis = vec.normalized()
    r1 = o1.get("kette_radius", props.kette_kugelradius)
    r2 = o2.get("kette_radius", props.kette_kugelradius)
    overlap = 0.2
    
    # Berechne verlängerte Länge, aber behalte originale Mitte
    start = o1.location - axis * (r1 * overlap)
    end = o2.location + axis * (r2 * overlap)
    length = (end - start).length
    
    # WICHTIG: Mitte bleibt zwischen den Kugelzentren!
    mid = (o1.location + o2.location) * 0.5
    
    bpy.ops.mesh.primitive_cone_add(
        vertices=props.kette_zyl_segs,
        radius1=r1 * props.kette_connector_radius_factor,
        radius2=r2 * props.kette_connector_radius_factor,
        depth=length,
        location=mid
    )
    
    cyl = bpy.context.active_object
    cyl.name = "Verbinder"
    cyl.rotation_mode = 'QUATERNION'
    cyl.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(axis)
    
    return cyl

def _create_curve(o1: bpy.types.Object, o2: bpy.types.Object) -> bpy.types.Object:
    props = bpy.context.scene.chain_construction_props
    p1, p2 = o1.location, o2.location
    steps = props.kette_connector_segs
    aura = get_aura(props.kette_target_obj, props.kette_aura_dichte)
    pts: List[Vector] = []
    r1 = o1.get("kette_radius", sum(o1.dimensions) / 6.0)
    r2 = o2.get("kette_radius", sum(o2.dimensions) / 6.0)
    r = min(r1, r2)
    offs = props.kette_abstand + props.kette_global_offset + EPSILON * 2
    overlap = 0.2
    direction = (p2 - p1).normalized()
    start_extended = p1 - direction * (r1 * overlap)
    end_extended = p2 + direction * (r2 * overlap)

    for i in range(steps):
        t = i / (steps - 1)
        loc, norm = next_surface_point(aura, start_extended.lerp(end_extended, t))
        pts.append(loc + norm * (r + offs))
    
    curve = bpy.data.curves.new("ConnectorCurve", 'CURVE')
    curve.dimensions = '3D'
    sp = curve.splines.new('POLY')
    sp.points.add(len(pts) - 1)
    
    for i, co in enumerate(pts):
        sp.points[i].co = (co.x, co.y, co.z, 1)
        t = i / (steps - 1)
        sp.points[i].radius = (r1 * (1 - t) + r2 * t) * props.kette_connector_radius_factor
    
    sp.use_smooth = True
    curve.use_fill_caps = True
    curve.bevel_depth = r * props.kette_connector_radius_factor
    curve.bevel_resolution = 4
    
    obj = bpy.data.objects.new("CurvedConnector", curve)
    
    # NEU: Konvertiere zu Mesh für saubere Geometrie
    temp_collection = bpy.context.collection
    temp_collection.objects.link(obj)
    
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.convert(target='MESH')
    
    temp_collection.objects.unlink(obj)
    
    return obj


def _update_cylinder(cyl: bpy.types.Object, o1: bpy.types.Object, o2: bpy.types.Object):
    props = bpy.context.scene.chain_construction_props
    axis = (o2.location - o1.location).normalized()
    
    # Radien prüfen
    r1 = o1.get("kette_radius", props.kette_kugelradius)
    r2 = o2.get("kette_radius", props.kette_kugelradius)
    overlap = 0.2
    start = o1.location - axis * (r1 * overlap)
    end = o2.location + axis * (r2 * overlap)
    mid = (start + end) * 0.5
    length = (end - start).length
    
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = cyl
    cyl.select_set(True)
    bpy.ops.object.delete()
    
    bpy.ops.mesh.primitive_cone_add(
        vertices=props.kette_zyl_segs,
        radius1=r1 * props.kette_connector_radius_factor,
        radius2=r2 * props.kette_connector_radius_factor,
        depth=length,
        location=mid
    )
    new_cyl = bpy.context.active_object
    new_cyl.name = "Verbinder"
    new_cyl.rotation_mode = 'QUATERNION'
    new_cyl.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(axis)
    set_material(new_cyl, "KetteSchwarz", (0.02, 0.02, 0.02), metallic=0.8, roughness=0.2)
    return new_cyl

def _update_curve(obj: bpy.types.Object, o1: bpy.types.Object, o2: bpy.types.Object):
    props = bpy.context.scene.chain_construction_props
    p1, p2 = o1.location, o2.location
    sp = obj.data.splines[0]
    steps = len(sp.points)
    aura = get_aura(props.kette_target_obj, props.kette_aura_dichte)
    pts: List[Vector] = []
    r1 = o1.get("kette_radius", sum(o1.dimensions) / 6.0)
    r2 = o2.get("kette_radius", sum(o2.dimensions) / 6.0)
    r = min(r1, r2)
    offs = props.kette_abstand + props.kette_global_offset + EPSILON * 2  
    overlap = 0.2
    direction = (p2 - p1).normalized()

    # Verlängere nur für die Kurvenberechnung
    start_extended = p1 - direction * (r1 * overlap)
    end_extended = p2 + direction * (r2 * overlap)

    for i in range(steps):
        t = i / (steps - 1)
        # Interpoliere zwischen den verlängerten Punkten
        interpolated = start_extended.lerp(end_extended, t)
        loc, norm = next_surface_point(aura, interpolated)
        pts.append(loc + norm * (r + offs))
        
    for i, co in enumerate(pts):
        sp.points[i].co = (co.x, co.y, co.z, 1)
        t = i / (steps - 1)
        sp.points[i].radius = (r1 * (1 - t) + r2 * t) * props.kette_connector_radius_factor

def delaunay_triangulate_spheres(spheres):
    """Erstellt optimierte Delaunay-Triangulation der Kugeln"""
    if len(spheres) < 3:
        return []
    
    debug.info('CONNECTOR', f"Creating Delaunay triangulation for {len(spheres)} spheres")
    
    try:
        from scipy.spatial import Delaunay, Voronoi
        import numpy as np
        
        positions = np.array([s.location for s in spheres])
        
        # 1. Standard Delaunay-Triangulation
        tri = Delaunay(positions)
        
        # 2. Filtere Edges basierend auf Verbindungsqualität
        edges = set()
        edge_lengths = []
        
        for simplex in tri.simplices:
            for i in range(len(simplex)):
                for j in range(i+1, len(simplex)):
                    edge = tuple(sorted([simplex[i], simplex[j]]))
                    if edge not in edges:
                        length = np.linalg.norm(positions[edge[0]] - positions[edge[1]])
                        edges.add(edge)
                        edge_lengths.append((length, edge))
        
        # 3. Sortiere nach Länge und nimm nur die kürzesten
        edge_lengths.sort(key=lambda x: x[0])
        
        # Entferne die längsten 20% der Edges
        cutoff = int(len(edge_lengths) * 0.8)
        filtered_edges = [edge for _, edge in edge_lengths[:cutoff]]
        
        # 4. Konvertiere zu Sphere-Paaren
        pairs = []
        for edge in filtered_edges:
            pairs.append((spheres[edge[0]], spheres[edge[1]]))
        
        debug.info('CONNECTOR', f"Created {len(pairs)} Delaunay connections")
        return pairs
        
    except ImportError:
        debug.warning('CONNECTOR', "scipy not available, using simple neighbor connections")
        # Fallback ohne scipy
        pairs = []
        for i in range(len(spheres)):
            # Verbinde zu den 3 nächsten Nachbarn
            distances = []
            for j in range(len(spheres)):
                if i != j:
                    dist = (spheres[i].location - spheres[j].location).length
                    distances.append((dist, j))
            
            distances.sort(key=lambda x: x[0])
            for _, j in distances[:3]:
                pair = tuple(sorted([i, j]))
                if pair not in {tuple(sorted([spheres.index(p[0]), spheres.index(p[1])])) for p in pairs}:
                    pairs.append((spheres[i], spheres[j]))
        
        return pairs

# =========================================
# PAINT UTILITY FUNCTIONS
# =========================================

def get_paint_surface_position(context, event, target_obj) -> Optional[Vector]:
    """Get 3D position from mouse cursor"""
    
    debug.trace('PAINT', f"Mouse coords: {event.mouse_region_x}, {event.mouse_region_y}")
    
    region = context.region
    rv3d = context.region_data
    coord = (event.mouse_region_x, event.mouse_region_y)
    
    # Get view ray
    view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
    ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
    
    debug.trace('PAINT', f"Ray origin: {ray_origin}, View vector: {view_vector}")
    
    # If target object exists, use surface snapping
    if target_obj and target_obj.type == 'MESH':
        try:
            props = context.scene.chain_construction_props
            
            # Einfacher Raycast auf das Zielobjekt
            depsgraph = context.evaluated_depsgraph_get()
            obj_eval = target_obj.evaluated_get(depsgraph)
            
            # Transform ray to object space
            matrix_inv = target_obj.matrix_world.inverted()
            ray_origin_local = matrix_inv @ ray_origin
            ray_direction_local = matrix_inv.to_3x3() @ view_vector
            
            # Raycast
            success, location, normal, face_index = obj_eval.ray_cast(
                ray_origin_local, ray_direction_local
            )
            
            if success:
                # Transform back to world space
                world_location = target_obj.matrix_world @ location
                
                # NUTZE DIE ORIGINAL AURA/SURFACE LOGIK!
                aura = get_aura(target_obj, props.kette_aura_dichte)
                final_pos, final_norm = next_surface_point(aura, world_location)
                
                # Apply offset
                offset = (props.kette_kugelradius + 
                         props.kette_abstand + 
                         props.kette_global_offset + 
                         EPSILON)
                
                result = final_pos + final_norm * offset
                
                debug.trace('PAINT', f"Hit surface at: {result}")
                return result
            else:
                debug.trace('PAINT', "No surface hit")
                
        except Exception as e:
            debug.error('PAINT', f"Surface snapping failed: {e}")
    
    # Fallback: intersect with ground plane (Z=0)
    if abs(view_vector.z) > 0.001:
        t = -ray_origin.z / view_vector.z
        if t > 0:
            fallback_pos = ray_origin + view_vector * t
            debug.trace('PAINT', f"Fallback to ground plane: {fallback_pos}")
            return fallback_pos
    
    # Last resort: fixed distance from camera
    fallback_pos = ray_origin + view_vector * 5.0
    debug.trace('PAINT', f"Last resort position: {fallback_pos}")
    return fallback_pos

def draw_paint_preview(self, context):
    """Draw preview circle at cursor"""
    if not self.cursor_3d_pos:
        return
    
    try:
        props = context.scene.chain_construction_props
        
        # Setup shader
        shader = gpu.shader.from_builtin('3D_UNIFORM_COLOR')
        
        # Draw circle at cursor
        radius = props.kette_kugelradius
        segments = 24
        
        # Create circle vertices
        vertices = []
        for i in range(segments + 1):
            angle = 2.0 * math.pi * i / segments
            x = self.cursor_3d_pos.x + radius * math.cos(angle)
            y = self.cursor_3d_pos.y + radius * math.sin(angle)
            z = self.cursor_3d_pos.z
            vertices.append((x, y, z))
        
        # Draw preview circle
        gpu.state.blend_set('ALPHA')
        gpu.state.line_width_set(2.0)
        
        batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": vertices})
        shader.bind()
        shader.uniform_float("color", (1.0, 1.0, 1.0, 0.8))
        batch.draw(shader)
        
        # Draw center point
        center_vertices = [self.cursor_3d_pos]
        gpu.state.point_size_set(6.0)
        
        batch_center = batch_for_shader(shader, 'POINTS', {"pos": center_vertices})
        shader.bind()
        shader.uniform_float("color", (0.8, 0.2, 0.2, 0.9))
        batch_center.draw(shader)
        
        # Draw symmetry preview if enabled
        if props.paint_use_symmetry:
            sym_pos = self.cursor_3d_pos.copy()
            if props.paint_symmetry_axis == 'X':
                sym_pos.x = -sym_pos.x
            elif props.paint_symmetry_axis == 'Y':
                sym_pos.y = -sym_pos.y
            elif props.paint_symmetry_axis == 'Z':
                sym_pos.z = -sym_pos.z
            
            # Draw symmetry circle
            sym_vertices = []
            for i in range(segments + 1):
                angle = 2.0 * math.pi * i / segments
                x = sym_pos.x + radius * math.cos(angle)
                y = sym_pos.y + radius * math.sin(angle)
                z = sym_pos.z
                sym_vertices.append((x, y, z))
            
            batch_sym = batch_for_shader(shader, 'LINE_STRIP', {"pos": sym_vertices})
            shader.bind()
            shader.uniform_float("color", (0.5, 0.5, 1.0, 0.5))
            batch_sym.draw(shader)
        
        gpu.state.blend_set('NONE')
        gpu.state.line_width_set(1.0)
        gpu.state.point_size_set(1.0)
        
    except Exception as e:
        debug.error('PAINT', f"Draw callback error: {e}")

# =========================================
# BATCH OPERATIONS
# =========================================

class BatchOperations:
    """Optimierte Batch-Operationen für bessere Performance"""
    
    @staticmethod
    @performance_monitor.time_function("create_spheres_batch", "BATCH")
    def create_spheres_batch(positions: List[Vector], props=None, name_prefix="Kugel", 
                           use_instances=False, material_color=(0.5, 0.7, 1.0)) -> List[bpy.types.Object]:
        """
        Erstelle mehrere Kugeln effizient - NUTZT JETZT create_sphere!
        """
        if not props:
            props = bpy.context.scene.chain_construction_props
            
        debug.info('BATCH', f"Creating {len(positions)} spheres in batch")
        
        spheres = []
        
        # Viewport Update temporär deaktivieren für Performance
        bpy.context.view_layer.update()
        
        # Bei vielen Kugeln: Verwende Instanzen
        if use_instances and len(positions) > 50:
            # Erstelle Basis-Mesh einmal
            base_mesh = bpy.data.meshes.new(name=f"{name_prefix}_BaseMesh")
            
            # Generiere Kugel-Geometrie
            bm = bmesh.new()
            bmesh.ops.create_uvsphere(
                bm,
                u_segments=props.kette_aufloesung,
                v_segments=props.kette_aufloesung,
                radius=props.kette_kugelradius
            )
            
            # Smooth shading
            for face in bm.faces:
                face.smooth = True
                
            bm.to_mesh(base_mesh)
            bm.free()
            
            # Auto smooth
            base_mesh.use_auto_smooth = True
            base_mesh.auto_smooth_angle = radians(30)
            
            # Erstelle Material einmal
            mat_name = "KetteMaterial_Batch"
            mat = bpy.data.materials.get(mat_name)
            if not mat:
                mat = bpy.data.materials.new(mat_name)
                mat.use_nodes = True
                bsdf = mat.node_tree.nodes["Principled BSDF"]
                bsdf.inputs['Base Color'].default_value = (*material_color, 1.0)
                bsdf.inputs['Metallic'].default_value = 0.5
                bsdf.inputs['Roughness'].default_value = 0.5
                mat.diffuse_color = (*material_color, 1.0)
            
            base_mesh.materials.append(mat)
            
            # Collection
            coll = ensure_collection("KonstruktionsKette")
            
            # Erstelle Instanzen in Batches
            batch_size = 100
            for batch_start in range(0, len(positions), batch_size):
                batch_end = min(batch_start + batch_size, len(positions))
                
                for i in range(batch_start, batch_end):
                    pos = positions[i]
                    obj = bpy.data.objects.new(name=f"{name_prefix}_{i:03d}", object_data=base_mesh)
                    obj.location = pos
                    obj["kette_radius"] = props.kette_kugelradius
                    obj["creation_time"] = time.time()
                    obj["batch_created"] = True
                    
                    coll.objects.link(obj)
                    spheres.append(obj)
                
                # Progress update
                debug.trace('BATCH', f"Created {batch_end}/{len(positions)} sphere instances")
                
                # Gelegentliches View Update
                if batch_end % 200 == 0:
                    bpy.context.view_layer.update()
                    
        else:
            # Standard-Methode mit create_sphere für wenige Kugeln
            for i, pos in enumerate(positions):
                sphere = create_sphere(
                    location=pos,
                    radius=props.kette_kugelradius,
                    name_prefix=f"{name_prefix}_{i:03d}",
                    material_color=material_color
                )
                spheres.append(sphere)
                
                if i % 10 == 0:
                    debug.trace('BATCH', f"Created {i}/{len(positions)} spheres")
        
        # Final view update
        bpy.context.view_layer.update()
        
        debug.info('BATCH', f"Batch creation complete: {len(spheres)} spheres")
        return spheres
    
    @staticmethod
    @performance_monitor.time_function("remove_objects_batch", "BATCH")
    def remove_objects_batch(objects: List[bpy.types.Object]):
        debug.info('BATCH', f"Removing {len(objects)} objects in batch")
        
        for obj in objects:
            for c in obj.users_collection:
                c.objects.unlink(obj)
        
        meshes_to_remove: List[bpy.types.Mesh] = []
        curves_to_remove: List[bpy.types.Curve] = []
        
        for obj in objects:
            if obj.type == 'MESH' and obj.data.users == 1:
                meshes_to_remove.append(obj.data)
            elif obj.type == 'CURVE' and obj.data.users == 1:
                curves_to_remove.append(obj.data)
            bpy.data.objects.remove(obj, do_unlink=True)
        
        for mesh in meshes_to_remove:
            bpy.data.meshes.remove(mesh)
        
        for curve in curves_to_remove:
            bpy.data.curves.remove(curve)

    @staticmethod
    @performance_monitor.time_function("connect_spheres_batch", "BATCH")
    def connect_spheres_batch(pairs: List[Tuple[bpy.types.Object, bpy.types.Object]], 
                            use_curved: bool, props):
        debug.info('BATCH', f"Connecting {len(pairs)} sphere pairs in batch")
        
        for o1, o2 in pairs:
            connect_kugeln(o1, o2, use_curved)


# =========================================
# LIBRARY MANAGEMENT
# =========================================

def ensure_library_directory():
    """Erstellt und gibt Library-Pfad zurück"""
    prefs = None
    lib_path = None
    
    # Prüfe ob als Addon oder Script läuft
    try:
        if __name__ != "__main__":
            prefs = bpy.context.preferences.addons[__name__].preferences
            if prefs.library_path:
                lib_path = pathlib.Path(prefs.library_path)
    except Exception:
        pass
    
    # Fallback: Check scene property
    if not lib_path:
        try:
            scene_path = bpy.context.scene.chain_library_path
            if scene_path:
                lib_path = pathlib.Path(scene_path)
        except Exception:
            pass
    
    # Final fallback: Documents folder
    if not lib_path:
        lib_path = pathlib.Path.home() / "Documents" / "chain_patterns"
    
    lib_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    for subdir in ["orthese", "jewelry", "technical", "organic", "custom"]:
        (lib_path / subdir).mkdir(exist_ok=True)
    
    # Create README if not exists
    readme_path = lib_path / "README.txt"
    if not readme_path.exists():
        readme_text = """Chain Construction Pattern Library
==================================

This directory contains pattern files for the Chain Construction tool.
Patterns are stored as JSON files with the following structure:

{
    "name": "Pattern Name",
    "category": "orthese|jewelry|technical|organic|custom",
    "parameters": {
        // All chain construction parameters
    },
    "metadata": {
        "created": "timestamp",
        "blender_version": "version",
        "addon_version": "version"
    }
}

Subdirectories:
- orthese/    : Medical orthosis patterns
- jewelry/    : Decorative jewelry patterns
- technical/  : Technical/engineering patterns
- organic/    : Organic growth patterns
- custom/     : User-defined patterns
"""
        readme_path.write_text(readme_text)
    
    return lib_path

def save_pattern_to_library(pattern_data: Dict[str, Any], pattern_id: str):
    """Speichert Pattern in Library"""
    lib_path = ensure_library_directory()
    category = pattern_data.get("category", "custom")
    
    # Add metadata
    pattern_data["metadata"] = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "blender_version": bpy.app.version_string,
        "addon_version": "3.7.0"
    }
    
    # Save to category folder
    file_path = lib_path / category / f"{pattern_id}.json"
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(pattern_data, f, indent=2)
    
    print(f"Pattern saved to: {file_path}")

def load_pattern_library() -> Dict[str, Dict[str, Any]]:
    """Lädt alle Patterns aus Library"""
    patterns = {}
    lib_path = ensure_library_directory()
    
    # Load from all subdirectories
    for subdir in lib_path.iterdir():
        if subdir.is_dir() and not subdir.name.startswith('.'):
            for file_path in subdir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        pattern_id = f"{subdir.name}_{file_path.stem}"
                        patterns[pattern_id] = data
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    
    return patterns

def create_example_patterns():
    """Erstellt Standard-Patterns als JSON-Dateien"""
    
    examples = [
        {
            "name": "Orthese Standard",
            "category": "orthese",
            "description": "Ausgewogenes Muster für Standard-Orthesen",
            "parameters": {
                'hop_pattern': 'ORGANIC',
                'hop_distance': 2.0,
                'hop_iterations': 8,
                'hop_probability': 0.7,
                'hop_angle_variation': 15.0,
                'hop_avoid_overlap': True,
                'hop_overlap_factor': 0.8,
                'hop_max_spheres': 150,
                'kette_kugelradius': 0.6,
                'kette_abstand': 0.1,
                'kette_connector_radius_factor': 0.7,
                'kette_use_curved_connectors': True,
                'auto_connect': True
            }
        },
        {
            "name": "Orthese Verstärkt", 
            "category": "orthese",
            "description": "Dichtes Muster für maximale Stabilität",
            "parameters": {
                'hop_pattern': 'HEXAGON',
                'hop_distance': 1.2,
                'hop_iterations': 10,
                'hop_probability': 0.95,
                'hop_angle_variation': 3.0,
                'hop_avoid_overlap': True,
                'hop_overlap_factor': 0.9,
                'hop_max_spheres': 200,
                'kette_kugelradius': 0.8,
                'kette_abstand': 0.08,
                'kette_connector_radius_factor': 0.9,
                'kette_use_curved_connectors': False,
                'auto_connect': True
            }
        },
        {
            "name": "Schmuck Elegant",
            "category": "jewelry", 
            "description": "Dekoratives Muster für Schmuckstücke",
            "parameters": {
                'hop_pattern': 'MIXED',
                'hop_distance': 1.5,
                'hop_iterations': 6,
                'hop_angle_variation': 10.0,
                'hop_probability': 0.9,
                'hop_avoid_overlap': True,
                'hop_overlap_factor': 0.75,
                'hop_max_spheres': 80,
                'kette_kugelradius': 0.4,
                'kette_abstand': 0.06,
                'kette_use_curved_connectors': True,
                'kette_connector_segs': 20,
                'kette_connector_radius_factor': 0.6,
                'auto_connect': True
            }
        },
        {
            "name": "Technisch Präzise",
            "category": "technical",
            "description": "Präzises Engineering-Muster",
            "parameters": {
                'hop_pattern': 'SQUARE',
                'hop_distance': 1.8,
                'hop_iterations': 12,
                'pattern_square_diagonal': True,
                'hop_probability': 1.0,
                'hop_angle_variation': 0.0,
                'hop_avoid_overlap': True,
                'hop_overlap_factor': 0.95,
                'hop_max_spheres': 250,
                'kette_kugelradius': 0.5,
                'kette_abstand': 0.08,
                'kette_connector_radius_factor': 0.85,
                'kette_use_curved_connectors': False,
                'auto_connect': True
            }
        },
        {
            "name": "Organisch Verzweigt",
            "category": "organic",
            "description": "Natürliches Verzweigungsmuster",
            "parameters": {
                'hop_pattern': 'ORGANIC',
                'hop_distance': 1.8,
                'hop_iterations': 6,
                'hop_probability': 0.8,
                'hop_branch_probability': 0.3,
                'organic_min_neighbors': 3,
                'organic_max_neighbors': 6,
                'hop_angle_variation': 25.0,
                'hop_avoid_overlap': True,
                'hop_overlap_factor': 0.7,
                'hop_max_spheres': 120,
                'kette_kugelradius': 0.5,
                'kette_abstand': 0.1,
                'auto_connect': True
            }
        }
    ]
    
    for i, example in enumerate(examples):
        pattern_id = f"default_{example['category']}_{i:02d}"
        save_pattern_to_library(example, pattern_id, example["category"])


# =========================================
# PROPERTY GROUPS
# =========================================

class PatternLibraryItem(PropertyGroup):
    """Single pattern in library"""
    pattern_id: StringProperty()
    pattern_name: StringProperty()
    pattern_category: StringProperty()
    pattern_data: StringProperty()  # JSON string

class ChainConstructionProperties(PropertyGroup):
    """Main property group for chain construction"""
    
    # Basic settings
    kette_target_obj: PointerProperty(
        name="Zielobjekt",
        type=bpy.types.Object,
        description="Objekt auf dem die Kette platziert wird",
        poll=lambda self, obj: obj.type == 'MESH'
    )
    kette_kugelradius: FloatProperty(
        name="Kugelradius",
        default=0.5,
        min=0.01,
        max=5.0,
        description="Radius der Kugeln"
    )
    kette_aufloesung: IntProperty(
        name="Auflösung",
        default=8,
        min=3,
        max=32,
        description="Anzahl der Segmente für Kugeln"
    )
    kette_abstand: FloatProperty(
        name="Oberflächenabstand",
        default=0.1,
        min=-2.0,
        max=2.0,
        description="Abstand der Kugeln zur Oberfläche"
    )
    kette_global_offset: FloatProperty(
        name="Globaler Offset",
        default=0.0,
        min=-5.0,
        max=5.0,
        description="Zusätzlicher Offset für alle Kugeln"
    )
    kette_aura_dichte: FloatProperty(
        name="Aura Dicke",
        default=0.05,
        min=0.0,
        max=1.0,
        description="Dicke der Aura für Oberflächenprojektion"
    )
    
    # Connector settings
    kette_use_curved_connectors: BoolProperty(
        name="Gebogene Verbinder",
        default=True,
        description="Verwende gebogene statt gerade Verbinder"
    )
    kette_connector_segs: IntProperty(
        name="Kurven-Segmente",
        default=12,
        min=3,
        max=32,
        description="Anzahl Segmente für gebogene Verbinder"
    )
    kette_connector_radius_factor: FloatProperty(
        name="Verbinder-Radius",
        default=0.8,
        min=0.1,
        max=2.0,
        subtype='FACTOR',
        description="Radius-Faktor für Verbinder"
    )
    kette_zyl_segs: IntProperty(
        name="Zylinder-Segmente",
        default=8,
        min=3,
        max=32,
        description="Anzahl Segmente für Zylinder-Verbinder"
    )
    
    # Connection mode
    kette_conn_mode: EnumProperty(
        name="Verbindungsmodus",
        items=[
            ('SEQ', 'Sequentiell', 'Verbinde in Reihenfolge'),
            ('NEIGHBOR', 'Nachbarn', 'Verbinde zu nächsten Nachbarn'),
            ('DELAUNAY', 'Delaunay', 'Optimierte Triangulation'),
        ],
        default='NEIGHBOR'
    )
    kette_max_abstand: FloatProperty(
        name="Max. Abstand",
        default=0,
        min=0,
        max=10,
        description="Maximaler Verbindungsabstand (0 = unbegrenzt)"
    )
    kette_max_links: IntProperty(
        name="Max. Verbindungen",
        default=4,
        min=1,
        max=12,
        description="Maximale Anzahl Verbindungen pro Kugel"
    )
    
    # Growth settings
    kette_growth_dir: EnumProperty(
        name="Wachstumsrichtung",
        items=[
            ('X', '+X', ''), ('NX', '-X', ''),
            ('Y', '+Y', ''), ('NY', '-Y', ''),
            ('Z', '+Z', ''), ('NZ', '-Z', ''),
            ('ARROW', 'Pfeil', 'Nutze Richtungspfeil'),
        ],
        default='X'
    )
    kette_richtungs_obj: PointerProperty(
        name="Richtungspfeil",
        type=bpy.types.Object
    )
    kette_chain_abstand: FloatProperty(
        name="Kettenabstand",
        default=1.5,
        min=0.1,
        max=10.0,
        description="Abstand zwischen Kettengliedern"
    )
    kette_chain_anzahl: IntProperty(
        name="Anzahl Glieder",
        default=10,
        min=1,
        max=100,
        description="Anzahl zu erzeugender Kettenglieder"
    )
    kette_sequentiell: BoolProperty(
        name="Sequentiell",
        default=False,
        description="Ignoriere Oberfläche für geradliniges Wachstum"
    )
    
    # Hopping pattern settings
    hop_pattern: EnumProperty(
        name="Muster-Typ",
        items=[
            ('HEXAGON', 'Hexagon', 'Sechseckiges Muster'),
            ('SQUARE', 'Quadrat', 'Quadratisches Muster'),
            ('TRIANGLE', 'Dreieck', 'Dreieckiges Muster'),
            ('PENTAGON', 'Pentagon', 'Fünfeckiges Muster'),
            ('ORGANIC', 'Organisch', 'Unregelmäßiges organisches Muster'),
            ('MIXED', 'Gemischt', 'Kombination verschiedener Muster'),
            ('TRANSITION', 'Übergang', 'Übergang zwischen Mustern'),
        ],
        default='HEXAGON',
        description="Art des Hopping-Musters"
    )
    hop_distance: FloatProperty(
        name="Hop-Abstand",
        default=1.5,
        min=0.1,
        max=10.0,
        description="Basis-Abstand zwischen Hops (Faktor des Kugelradius)"
    )
    hop_iterations: IntProperty(
        name="Iterationen",
        default=5,
        min=1,
        max=50,
        description="Anzahl der Wachstums-Iterationen"
    )
    hop_angle_variation: FloatProperty(
        name="Winkel-Variation",
        default=5.0,
        min=0.0,
        max=90.0,
        subtype='ANGLE',
        description="Zufällige Variation der Hop-Winkel"
    )
    hop_distance_variation: FloatProperty(
        name="Abstands-Variation",
        default=0.1,
        min=0.0,
        max=0.5,
        subtype='FACTOR',
        description="Zufällige Variation der Hop-Abstände"
    )
    hop_probability: FloatProperty(
        name="Hop-Wahrscheinlichkeit",
        default=0.9,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
        description="Wahrscheinlichkeit dass ein Hop ausgeführt wird"
    )
    hop_max_spheres: IntProperty(
        name="Max. Kugeln",
        default=100,
        min=10,
        max=1000,
        description="Maximale Anzahl zu erzeugender Kugeln"
    )
    hop_avoid_overlap: BoolProperty(
        name="Überlappung vermeiden",
        default=True,
        description="Verhindere zu nahe beieinander liegende Kugeln"
    )
    hop_overlap_factor: FloatProperty(
        name="Überlappungs-Faktor",
        default=0.3,
        min=0.1,
        max=1.0,
        subtype='FACTOR',
        description="Minimaler Abstand als Faktor des Hop-Abstands"
    )
    pattern_square_diagonal: BoolProperty(
        name="Quadrat: Diagonale",
        default=False,
        description="Füge diagonale Nachbarn im Quadrat-Muster hinzu"
    )
    organic_min_neighbors: IntProperty(
        name="Org. Min Nachbarn",
        default=2,
        min=1,
        max=8,
        description="Minimale Anzahl Nachbarn im organischen Muster"
    )
    organic_max_neighbors: IntProperty(
        name="Org. Max Nachbarn",
        default=5,
        min=1,
        max=8,
        description="Maximale Anzahl Nachbarn im organischen Muster"
    )
    hop_pattern_start: EnumProperty(
        name="Start-Muster",
        items=[
            ('HEXAGON', 'Hexagon', ''), ('SQUARE', 'Quadrat', ''),
            ('TRIANGLE', 'Dreieck', ''), ('PENTAGON', 'Pentagon', '')
        ],
        default='HEXAGON',
        description="Muster am Anfang (für Übergang)"
    )
    hop_pattern_end: EnumProperty(
        name="End-Muster",
        items=[
            ('HEXAGON', 'Hexagon', ''), ('SQUARE', 'Quadrat', ''),
            ('TRIANGLE', 'Dreieck', ''), ('PENTAGON', 'Pentagon', '')
        ],
        default='TRIANGLE',
        description="Muster am Ende (für Übergang)"
    )
    hop_branch_probability: FloatProperty(
        name="Verzweigungs-Chance",
        default=0.1,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
        description="Wahrscheinlichkeit für Verzweigungen"
    )
    hop_generation_scale: FloatProperty(
        name="Generations-Skalierung",
        default=1.0,
        min=0.5,
        max=2.0,
        description="Skalierung des Abstands pro Generation"
    )
    hop_distance_tolerance: FloatProperty(
        name="Abstands-Toleranz",
        default=0.1,
        min=0.0,
        max=0.5,
        subtype='FACTOR',
        description="Erlaubte Abweichung vom idealen Hop-Abstand"
    )
    hop_respect_existing: BoolProperty(
        name="Vorhandene beachten",
        default=True,
        description="Berücksichtige vorhandene Kugeln beim Platzieren"
    )
    hop_ideal_distance_weight: FloatProperty(
        name="Ideal-Gewicht",
        default=0.7,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
        description="Gewichtung des idealen Abstands"
    )
    hop_cleanup_connections: BoolProperty(
        name="Verbindungen bereinigen",
        default=True,
        description="Entfernt doppelte Verbindungen"
    )
    hop_fill_missing_connections: BoolProperty(
        name="Lücken füllen",
        default=True,
        description="Fügt fehlende Verbindungen hinzu"
    )
    hop_min_connection_angle: FloatProperty(
        name="Min. Verbindungswinkel",
        default=30.0,
        min=5.0,
        max=135.5,
        subtype='ANGLE',
        description="Minimaler Winkel zwischen Verbindungen"
    )
    auto_connect: BoolProperty(
        name="Auto-Verbinden",
        default=True,
        description="Verbinde Kugeln automatisch nach Generierung"
    )
    
    # Paint Tool Properties
    paint_brush_spacing: FloatProperty(
        name="Pinsel-Abstand",
        default=2.5,
        min=0.5,
        max=10.0,
        description="Abstand zwischen gemalten Kugeln (Faktor des Kugelradius)"
    )
    paint_use_symmetry: BoolProperty(
        name="Symmetrie",
        default=False,
        description="Symmetrisches Malen aktivieren"
    )
    paint_symmetry_axis: EnumProperty(
        name="Symmetrie-Achse",
        items=[
            ('X', 'X-Achse', 'Spiegelung entlang X-Achse'),
            ('Y', 'Y-Achse', 'Spiegelung entlang Y-Achse'),
            ('Z', 'Z-Achse', 'Spiegelung entlang Z-Achse'),
        ],
        default='X'
    )
    paint_auto_connect: BoolProperty(
        name="Auto-Verbinden",
        default=True,
        description="Verbinde gemalte Kugeln automatisch beim Beenden"
    )

    # UI states
    show_hop_pattern_params: BoolProperty(default=True)
    show_hop_advanced: BoolProperty(default=False)
    show_connector_settings: BoolProperty(default=False)
    show_library_settings: BoolProperty(default=False)
    
    # Library
    pattern_library: CollectionProperty(type=PatternLibraryItem)
    pattern_library_index: IntProperty()
    library_path_override: StringProperty(
        name="Library Path Override",
        default="",
        subtype='DIR_PATH'
    )
        # Library Management
    chain_library_path: StringProperty(
        name="Chain Pattern Ordner",
        description="Ordner mit Chain Pattern JSON-Dateien",
        default="",
        subtype='DIR_PATH'
    )


# Addon Preferences
class ChainConstructionPreferences(AddonPreferences):
    bl_idname = __name__
    
    library_path: StringProperty(
        name="Pattern Library Path",
        description="Pfad zur Pattern-Bibliothek",
        default="",
        subtype='DIR_PATH'
    )
    enable_debug: BoolProperty(
        name="Debug Mode",
        description="Aktiviere Debug-Ausgabe",
        default=False
    )
    
    def draw(self, context):
        layout = self.layout
        layout.prop(self, "library_path")
        layout.prop(self, "enable_debug")

# =========================================
# OPERATORS – HOPPING PATTERN
# =========================================

class KETTE_OT_generate_hopping_pattern(Operator):
    """Hauptoperator für Hopping Pattern Generation"""
    bl_idname = "kette.generate_hopping_pattern"
    bl_label = "Hopping Muster Generieren"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Generiert ein Hopping-Pattern basierend auf den Einstellungen"
    
    @classmethod
    def poll(cls, context):
        props = context.scene.chain_construction_props
        return props.kette_target_obj is not None
    
    def execute(self, context):
        props = context.scene.chain_construction_props
        target_obj = props.kette_target_obj
        
        if not target_obj:
            self.report({'ERROR'}, "Kein Zielobjekt ausgewählt!")
            return {'CANCELLED'}
        
        # Hole Start-Kugel
        start_sphere = context.active_object
        if not (start_sphere and start_sphere.name.startswith("Kugel")):
            # Erstelle Start-Kugel wenn keine vorhanden
            cursor_loc = context.scene.cursor.location
            aura = get_aura(target_obj, props.kette_aura_dichte)
            loc, norm = next_surface_point(aura, cursor_loc)
            offset = props.kette_kugelradius + props.kette_abstand + props.kette_global_offset + EPSILON
            
            start_sphere = create_sphere(
                location=loc + norm * offset,
                name_prefix="Kugel_Start",
                material_name="KetteStart",
                material_color=(1.0, 0.0, 0.0)
            )
        
        # Sammle existierende Kugeln
        coll = ensure_collection("KonstruktionsKette")
        existing_spheres = [o for o in coll.objects if o.name.startswith("Kugel")]
        
        # Baue KDTree für existierende Kugeln
        if existing_spheres and props.hop_respect_existing:
            kdt = KDTree(len(existing_spheres))
            for i, sphere in enumerate(existing_spheres):
                kdt.insert(sphere.location, i)
            kdt.balance()
        else:
            kdt = None
        
        # Generiere Hopping Pattern
        new_positions = []
        connections = []
        
        # Queue für zu verarbeitende Kugeln (Position, Generation, Parent)
        queue = deque([(start_sphere.location, 0, None)])
        processed = set()
        position_to_index = {start_sphere.location.to_tuple(): 0}
        all_positions = [start_sphere.location]
        
        while queue and len(new_positions) < props.hop_max_spheres:
            current_pos, generation, parent_idx = queue.popleft()
            
            # Skip wenn schon verarbeitet
            pos_tuple = current_pos.to_tuple()
            if pos_tuple in processed:
                continue
            processed.add(pos_tuple)
            
            # Bestimme Anzahl der Hops basierend auf Pattern
            hop_angles = self._get_hop_angles(props.hop_pattern, generation)
            
            # Führe Hops aus
            for angle in hop_angles:
                # Prüfe Wahrscheinlichkeit
                if random.random() > props.hop_probability:
                    continue
                
                # Berechne neue Position
                new_pos = self._calculate_hop_position(
                    current_pos, angle, props, target_obj
                )
                
                if not new_pos:
                    continue
                
                # Prüfe Überlappung
                if props.hop_avoid_overlap:
                    too_close = False
                    min_dist = props.kette_kugelradius * 2 * props.hop_overlap_factor
                    
                    # Prüfe gegen neue Positionen
                    for pos in all_positions:
                        if (new_pos - pos).length < min_dist:
                            too_close = True
                            break
                    
                    # Prüfe gegen existierende Kugeln
                    if not too_close and kdt:
                        neighbors = kdt.find_range(new_pos, min_dist)
                        if neighbors:
                            too_close = True
                    
                    if too_close:
                        continue
                
                # Füge Position hinzu
                new_positions.append(new_pos)
                new_idx = len(all_positions)
                all_positions.append(new_pos)
                position_to_index[new_pos.to_tuple()] = new_idx
                
                # Füge Verbindung hinzu
                if parent_idx is not None:
                    connections.append((parent_idx, new_idx))
                
                # Füge zur Queue hinzu
                if generation + 1 < props.hop_iterations:
                    queue.append((new_pos, generation + 1, new_idx))
                
                # Verzweigung
                if random.random() < props.hop_branch_probability:
                    branch_angle = angle + random.uniform(-60, 60)
                    branch_pos = self._calculate_hop_position(
                        current_pos, branch_angle, props, target_obj
                    )
                    if branch_pos:
                        queue.append((branch_pos, generation + 1, parent_idx))
        
        # Erstelle Kugeln
        if new_positions:
            new_spheres = BatchOperations.create_spheres_batch(new_positions, props)
            
            # Auto-Verbinden wenn aktiviert
            if props.auto_connect:
                # Verbinde basierend auf den generierten Connections
                for parent_idx, child_idx in connections:
                    if parent_idx == 0:
                        parent = start_sphere
                    else:
                        parent = new_spheres[parent_idx - 1]
                    child = new_spheres[child_idx - 1]
                    connect_kugeln(parent, child, props.kette_use_curved_connectors)
                
                # Füge zusätzliche Verbindungen hinzu wenn gewünscht
                if props.hop_fill_missing_connections:
                    self._fill_missing_connections(
                        [start_sphere] + new_spheres, props
                    )
        
        self.report({'INFO'}, f"{len(new_positions)} Kugeln generiert")
        return {'FINISHED'}
    
    def _get_hop_angles(self, pattern: str, generation: int) -> List[float]:
        """Gibt Hop-Winkel basierend auf Pattern zurück"""
        if pattern == 'HEXAGON':
            return [i * 60 for i in range(6)]
        elif pattern == 'SQUARE':
            angles = [0, 90, 180, 270]
            if bpy.context.scene.chain_construction_props.pattern_square_diagonal:
                angles.extend([45, 135, 225, 315])
            return angles
        elif pattern == 'TRIANGLE':
            return [0, 120, 240]
        elif pattern == 'PENTAGON':
            return [i * 72 for i in range(5)]
        elif pattern == 'ORGANIC':
            props = bpy.context.scene.chain_construction_props
            num = random.randint(props.organic_min_neighbors, props.organic_max_neighbors)
            base_angle = 360 / num
            return [i * base_angle + random.uniform(-30, 30) for i in range(num)]
        elif pattern == 'MIXED':
            patterns = ['HEXAGON', 'SQUARE', 'TRIANGLE', 'PENTAGON']
            return self._get_hop_angles(random.choice(patterns), generation)
        elif pattern == 'TRANSITION':
            props = bpy.context.scene.chain_construction_props
            t = generation / max(props.hop_iterations - 1, 1)
            # Interpoliere zwischen Start- und End-Pattern
            start_angles = self._get_hop_angles(props.hop_pattern_start, 0)
            end_angles = self._get_hop_angles(props.hop_pattern_end, 0)
            
            # Einfache Mischung
            if random.random() < t:
                return end_angles
            else:
                return start_angles
        
        return [0, 90, 180, 270]  # Fallback
    
    def _calculate_hop_position(self, current_pos: Vector, angle: float, 
                               props, target_obj) -> Optional[Vector]:
        """Berechnet neue Position für Hop"""
        try:
            # Hole Oberflächennormale
            aura = get_aura(target_obj, props.kette_aura_dichte)
            _, normal = next_surface_point(aura, current_pos)
            
            # Berechne Hop-Richtung
            # Erstelle lokales Koordinatensystem
            if abs(normal.dot(Vector((0, 0, 1)))) < 0.99:
                tangent = normal.cross(Vector((0, 0, 1))).normalized()
            else:
                tangent = normal.cross(Vector((0, 1, 0))).normalized()
            bitangent = normal.cross(tangent).normalized()
            
            # Füge Winkel-Variation hinzu
            angle_rad = radians(angle + random.uniform(-props.hop_angle_variation, 
                                                      props.hop_angle_variation))
            
            # Berechne Richtung
            direction = tangent * cos(angle_rad) + bitangent * sin(angle_rad)
            
            # Berechne Distanz mit Variation
            distance_factor = 1.0 + random.uniform(-props.hop_distance_variation, 
                                                  props.hop_distance_variation)
            distance = props.kette_kugelradius * props.hop_distance * distance_factor
            
            # Skaliere mit Generation
            distance *= props.hop_generation_scale ** (current_pos.z / 10.0)  # Beispiel
            
            # Neue Position
            new_pos = current_pos + direction * distance
            
            # Projiziere auf Oberfläche
            surface_pos, surface_normal = next_surface_point(aura, new_pos)
            offset = props.kette_kugelradius + props.kette_abstand + props.kette_global_offset + EPSILON
            final_pos = surface_pos + surface_normal * offset
            
            return final_pos
            
        except Exception as e:
            debug.error('HOPPING', f"Error calculating hop position: {e}")
            return None
    
    def _fill_missing_connections(self, spheres: List[bpy.types.Object], props):
        """Fügt fehlende Verbindungen zwischen nahen Kugeln hinzu"""
        max_dist = props.kette_kugelradius * props.hop_distance * 1.5
        
        # Baue KDTree
        positions = [s.location for s in spheres]
        kdt = KDTree(len(positions))
        for i, pos in enumerate(positions):
            kdt.insert(pos, i)
        kdt.balance()
        
        # Finde und erstelle Verbindungen
        connections_added = 0
        for i, sphere in enumerate(spheres):
            neighbors = kdt.find_range(sphere.location, max_dist)
            
            for _, idx, dist in neighbors:
                if idx <= i:  # Vermeide doppelte Verbindungen
                    continue
                
                # Prüfe ob Verbindung schon existiert
                key = tuple(sorted([sphere.name, spheres[idx].name]))
                if key not in _existing_connectors:
                    connect_kugeln(sphere, spheres[idx], props.kette_use_curved_connectors)
                    connections_added += 1
        
        debug.info('HOPPING', f"Added {connections_added} missing connections")


class KETTE_OT_save_hopping_pattern(Operator):
    bl_idname = "kette.save_hopping_pattern"
    bl_label = "Hopping Pattern Speichern"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Speichert aktuelles Hopping-Pattern"

    pattern_name: StringProperty(
        name="Pattern-Name",
        default="Neues Hopping Pattern"
    )
    pattern_id: StringProperty(
        name="Pattern ID",
        default=""
    )
    pattern_category: EnumProperty(
        name="Kategorie",
        items=[
            ('orthese', 'Orthese', ''),
            ('jewelry', 'Schmuck', ''),
            ('technical', 'Technisch', ''),
            ('organic', 'Organisch', ''),
            ('custom', 'Benutzerdefiniert', '')
        ],
        default='custom'
    )

    def invoke(self, context, event):
        self.pattern_id = f"pattern_{int(time.time())}"
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "pattern_name")
        layout.prop(self, "pattern_category")

    def execute(self, context):
        props = context.scene.chain_construction_props
        
        # Sammle alle relevanten Properties
        pattern_data = {
            "name": self.pattern_name,
            "category": self.pattern_category,
            "parameters": {}
        }
        
        # Speichere alle hop_ und kette_ Properties
        for prop_name in dir(props):
            if (prop_name.startswith("hop_") or prop_name.startswith("kette_") or 
                prop_name in ["auto_connect", "pattern_square_diagonal", "organic_min_neighbors", "organic_max_neighbors"]):
                try:
                    value = getattr(props, prop_name)
                    # Konvertiere Blender-spezifische Typen
                    if hasattr(value, "name"):  # Object reference
                        continue
                    pattern_data["parameters"][prop_name] = value
                except Exception:
                    pass
        
        # Speichere in Library
        try:
            save_pattern_to_library(pattern_data, self.pattern_id)
            self.report({'INFO'}, f"Pattern '{self.pattern_name}' gespeichert")
            
            # Refresh library
            bpy.ops.kette.refresh_library()
            
        except Exception as e:
            self.report({'ERROR'}, f"Fehler beim Speichern: {e}")
            return {'CANCELLED'}
        
        return {'FINISHED'}

class KETTE_OT_clear_pattern(Operator):
    bl_idname = "kette.clear_pattern"
    bl_label = "Muster Löschen"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Löscht alle Kugeln außer Startkugel"
    
    keep_start: BoolProperty(
        name="Startkugel behalten",
        default=True,
        description="Behält die Startkugel"
    )
    clear_connectors: BoolProperty(
        name="Verbinder löschen",
        default=True,
        description="Löscht auch alle Verbinder"
    )
    
    def execute(self, context):
        coll = bpy.data.collections.get("KonstruktionsKette")
        if not coll:
            self.report({'INFO'}, "Keine Kette gefunden")
            return {'FINISHED'}
        
        to_remove = []
        
        for obj in coll.objects:
            if obj.name.startswith("Kugel"):
                if not (self.keep_start and obj.name == "Kugel_Start"):
                    to_remove.append(obj)
            elif self.clear_connectors and (obj.name.startswith("Verbinder") or 
                                           obj.name.startswith("CurvedConnector")):
                to_remove.append(obj)
        
        if to_remove:
            BatchOperations.remove_objects_batch(to_remove)
            _existing_connectors.clear()
            _remembered_pairs.clear()
            self.report({'INFO'}, f"{len(to_remove)} Objekte gelöscht")
        else:
            self.report({'INFO'}, "Nichts zu löschen")
        
        return {'FINISHED'}
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)
    
    def draw(self, context):
        layout = self.layout
        layout.prop(self, "keep_start")
        layout.prop(self, "clear_connectors")

# =========================================
# OPERATORS – BASIC TOOLS
# =========================================

class KETTE_OT_kugel_platzieren(Operator):
    bl_idname = "kette.kugel_platzieren"
    bl_label = "Start-Kugel platzieren"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Platziert Start-Kugel am 3D-Cursor"

    def execute(self, context):
        props = context.scene.chain_construction_props
        cursor_loc = context.scene.cursor.location
        
        if props.kette_target_obj:
            aura = get_aura(props.kette_target_obj, props.kette_aura_dichte)
            loc, norm = next_surface_point(aura, cursor_loc)
            offset = props.kette_kugelradius + props.kette_abstand + props.kette_global_offset + EPSILON
            cursor_loc = loc + norm * offset
        
        sphere = create_sphere(
            location=cursor_loc,
            name_prefix="Kugel_Start",
            material_name="KetteStart",
            material_color=(1.0, 0.0, 0.0)
        )
        
        # Selektiere
        bpy.ops.object.select_all(action='DESELECT')
        sphere.select_set(True)
        context.view_layer.objects.active = sphere
        
        return {'FINISHED'}

class KETTE_OT_snap_surface(Operator):
    bl_idname = "kette.snap_surface"
    bl_label = "Snap auf Oberfläche"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Platziert ausgewählte Kugeln korrekt auf die Oberfläche"

    def execute(self, context):
        props = context.scene.chain_construction_props
        if not props.kette_target_obj:
            self.report({'ERROR'}, "Kein Zielobjekt gewählt!")
            return {'CANCELLED'}
        
        aura = get_aura(props.kette_target_obj, props.kette_aura_dichte)
        selected_spheres = [o for o in context.selected_objects if o.name.startswith("Kugel")]
        
        if not selected_spheres:
            self.report({'ERROR'}, "Keine Kugeln ausgewählt!")
            return {'CANCELLED'}
        
        for sph in selected_spheres:
            loc, norm = next_surface_point(aura, sph.location)
            offset = (props.kette_kugelradius + props.kette_abstand + 
                     props.kette_global_offset + EPSILON)
            sph.location = loc + norm * offset
        
        self.report({'INFO'}, f"{len(selected_spheres)} Kugeln gesnapped")
        return {'FINISHED'}

class KETTE_OT_line_connector(Operator):
    bl_idname = "kette.line_connector"
    bl_label = "Gerader Verbinder"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Erstelle geraden Zylinder-Verbinder zwischen zwei ausgewählten Kugeln"

    def execute(self, context):
        sels = [o for o in context.selected_objects if o.name.startswith("Kugel")]
        if len(sels) != 2:
            self.report({'ERROR'}, "Bitte genau zwei Kugeln auswählen!")
            return {'CANCELLED'}
        
        o1, o2 = sels
        connect_kugeln(o1, o2, use_curved=False)
        
        self.report({'INFO'}, f"Gerader Verbinder zwischen {o1.name} und {o2.name} erstellt")
        return {'FINISHED'}

class KETTE_OT_curved_connector(Operator):
    bl_idname = "kette.curved_connector"
    bl_label = "Gebogener Verbinder"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Erstelle gebogenen Verbinder zwischen zwei ausgewählten Kugeln"

    def execute(self, context):
        sels = [o for o in context.selected_objects if o.name.startswith("Kugel")]
        if len(sels) != 2:
            self.report({'ERROR'}, "Bitte genau zwei Kugeln auswählen!")
            return {'CANCELLED'}
        
        o1, o2 = sels
        connect_kugeln(o1, o2, use_curved=True)
        
        self.report({'INFO'}, f"Gebogener Verbinder zwischen {o1.name} und {o2.name} erstellt")
        return {'FINISHED'}

class KETTE_OT_kette_update(Operator):
    bl_idname = "kette.kette_update"
    bl_label = "Kette aktualisieren"
    bl_description = "Aktualisiert alle Verbinder nach manuellen Änderungen"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        updated, deleted = refresh_connectors()
        msgs = []
        if updated:
            msgs.append(f"{updated} aktualisiert")
        if deleted:
            msgs.append(f"{deleted} gelöscht")
        if not msgs:
            self.report({'INFO'}, "Keine Verbinder angepasst")
        else:
            self.report({'INFO'}, ", ".join(msgs))
        return {'FINISHED'}

class KETTE_OT_automatisch_verbinden(Operator):
    bl_idname = "kette.automatisch_verbinden"
    bl_label = "Kugeln automatisch verbinden"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Verbindet Kugeln automatisch"

    def execute(self, context):
        props = context.scene.chain_construction_props
        coll = ensure_collection("KonstruktionsKette")
        kugeln = [o for o in coll.objects if o.name.startswith("Kugel")]
        
        if len(kugeln) < 2:
            self.report({'ERROR'}, "Mindestens zwei Kugeln notwendig!")
            return {'CANCELLED'}

        kugeln.sort(key=lambda x: x.name)
        mode = props.kette_conn_mode
        max_dist = props.kette_max_abstand
        max_links = props.kette_max_links
        curved = props.kette_use_curved_connectors

        if mode == 'SEQ':
            pairs = []
            for a, b in zip(kugeln, kugeln[1:]):
                pairs.append((a, b))
            BatchOperations.connect_spheres_batch(pairs, curved, props)
        elif mode == 'DELAUNAY':
            delaunay_pairs = delaunay_triangulate_spheres(kugeln)
            BatchOperations.connect_spheres_batch(delaunay_pairs, curved, props)
        else:
            # Neighbor mode with KDTree
            if len(kugeln) > 10:
                positions = [k.location for k in kugeln]
                kdt = KDTree(len(positions))
                for i, pos in enumerate(positions):
                    kdt.insert(pos, i)
                kdt.balance()
                
                pairs = []
                for i, k1 in enumerate(kugeln):
                    if max_dist > 0:
                        neighbors = kdt.find_range(k1.location, max_dist)
                        neighbors = [(d, kugeln[idx]) for _, idx, d in neighbors if idx != i]
                    else:
                        neighbors = [((k1.location - k2.location).length, k2) 
                                   for k2 in kugeln if k1 != k2]
                    
                    neighbors.sort(key=lambda x: x[0])
                    for _, k2 in neighbors[:max_links]:
                        pair = tuple(sorted([k1, k2], key=lambda x: x.name))
                        if pair not in pairs:
                            pairs.append(pair)
                
                BatchOperations.connect_spheres_batch(
                    [(p[0], p[1]) for p in pairs],
                    curved,
                    props
                )
            else:
                for k1 in kugeln:
                    neighbors = []
                    for k2 in kugeln:
                        if k1 is k2:
                            continue
                        d = (k2.location - k1.location).length
                        if max_dist == 0 or d <= max_dist:
                            neighbors.append((d, k2))
                    neighbors.sort(key=lambda x: x[0])
                    for _, k2 in neighbors[:max_links]:
                        connect_kugeln(k1, k2, curved)

        update_chain()
        return {'FINISHED'}

# =========================================
# OPERATORS – GROWTH TOOLS
# =========================================

class KETTE_OT_richtungspfeil_erstellen(Operator):
    bl_idname = "kette.richtungspfeil_erstellen"
    bl_label = "Pfeil erstellen"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Erstellt einen Richtungspfeil an markierter Kugel"

    def execute(self, context):
        scene = context.scene
        props = scene.chain_construction_props
        start = context.active_object
        if not (start and start.name.startswith("Kugel")):
            self.report({'ERROR'}, "Wähle eine Kugel als Startpunkt!")
            return {'CANCELLED'}

        bpy.ops.object.empty_add(type='SINGLE_ARROW', location=start.location)
        arrow = context.active_object
        arrow.name = "KetteRichtung"
        arrow.empty_display_size = props.kette_kugelradius * 2
        props.kette_richtungs_obj = arrow
        
        coll = ensure_collection("KonstruktionsKette")
        coll.objects.link(arrow)
        for c in list(arrow.users_collection):
            if c != coll:
                c.objects.unlink(arrow)
        
        return {'FINISHED'}

class KETTE_OT_kette_wachstum(Operator):
    bl_idname = "kette.kette_wachstum"
    bl_label = "Kette wachsen lassen"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Lässt die Kette in eine Richtung wachsen"

    def execute(self, context):
        scene = context.scene
        props = scene.chain_construction_props
        start = context.active_object
        
        if not (start and start.name.startswith("Kugel")):
            self.report({'ERROR'}, "Wähle eine Kugel als Startpunkt!")
            return {'CANCELLED'}
        
        if not props.kette_target_obj:
            self.report({'ERROR'}, "Kein Zielobjekt gewählt!")
            return {'CANCELLED'}

        # Determine direction
        dc = props.kette_growth_dir
        if dc == 'ARROW':
            arrow = props.kette_richtungs_obj
            if not arrow:
                self.report({'ERROR'}, "Erstelle zuerst den Richtungs-Pfeil!")
                return {'CANCELLED'}
            base_dir = (arrow.matrix_world.to_quaternion() @ Vector((0,0,1))).normalized()
        else:
            axis_map = {
                'X':  Vector((1,0,0)),  'NX': Vector((-1,0,0)),
                'Y':  Vector((0,1,0)),  'NY': Vector((0,-1,0)),
                'Z':  Vector((0,0,1)),  'NZ': Vector((0,0,-1)),
            }
            base_dir = axis_map.get(dc, Vector((0,0,1)))

        # Parameters
        seq = props.kette_sequentiell
        r = props.kette_kugelradius
        offs_base = props.kette_abstand + props.kette_global_offset
        step = props.kette_chain_abstand
        count = props.kette_chain_anzahl
        curved = props.kette_use_curved_connectors

        aura = get_aura(props.kette_target_obj, props.kette_aura_dichte)
        coll = ensure_collection("KonstruktionsKette")
        
        prev = start
        prev_loc = start.location.copy()
        _, norm_prev = next_surface_point(aura, prev_loc)
        norm_start = norm_prev.copy()
        prev_dir = None

        # Collect positions for batch creation
        positions = []
        
        for i in range(count):
            dot = base_dir.dot(norm_prev)
            tang = base_dir - norm_prev * dot
            direction = tang.normalized() if tang.length > 1e-6 else (prev_dir or base_dir)

            guess = prev_loc + direction * step
            if seq:
                pos, norm = guess + norm_start * (r + offs_base), norm_prev
            else:
                loc, norm = next_surface_point(aura, guess)
                pos = loc + norm * (r + offs_base + EPSILON)

            positions.append(pos)
            prev_loc = pos
            prev_dir = direction
            norm_prev = norm

        # Create all spheres at once
        new_spheres = BatchOperations.create_spheres_batch(positions, props)
        
        # Connect them
        all_spheres = [start] + new_spheres
        pairs = [(all_spheres[i], all_spheres[i+1]) for i in range(len(all_spheres)-1)]
        BatchOperations.connect_spheres_batch(pairs, curved, props)

        clear_auras()
        
        # Delete arrow if used
        if props.kette_growth_dir == 'ARROW' and props.kette_richtungs_obj:
            arrow = props.kette_richtungs_obj
            for col in list(arrow.users_collection):
                col.objects.unlink(arrow)
            bpy.data.objects.remove(arrow, do_unlink=True)
            props.kette_richtungs_obj = None

        return {'FINISHED'}

class KETTE_OT_kreis_kette(Operator):
    bl_idname = "kette.kreis_kette"
    bl_label = "Kreis-Kette erstellen"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Kugeln automatisch in Kreisform"

    def execute(self, context):
        scene = context.scene
        props = scene.chain_construction_props
        start = context.active_object
        if not (start and start.name.startswith("Kugel")):
            self.report({'ERROR'}, "Wähle eine Kugel als Mittelpunkt!")
            return {'CANCELLED'}

        # Determine axis
        axis_map = {
            'X':  Vector((1, 0, 0)),   'NX': Vector((-1, 0, 0)),
            'Y':  Vector((0, 1, 0)),   'NY': Vector((0, -1, 0)),
            'Z':  Vector((0, 0, 1)),   'NZ': Vector((0, 0, -1)),
            'ARROW': Vector((0, 0, 1))
        }
        normal = axis_map.get(props.kette_growth_dir, Vector((0, 0, 1)))
        if abs(normal.cross(Vector((0, 0, 1))).length) > 1e-6:
            u = normal.cross(Vector((0, 0, 1))).normalized()
        else:
            u = normal.cross(Vector((0, 1, 0))).normalized()
        v = normal.cross(u).normalized()

        radius = props.kette_chain_abstand
        N = props.kette_chain_anzahl
        curved = props.kette_use_curved_connectors

        # Calculate positions
        positions = []
        for i in range(N):
            angle = 2 * math.pi * i / N
            pos = start.location + radius * (u * math.cos(angle) + v * math.sin(angle))
            positions.append(pos)

        # Create spheres
        spheres = BatchOperations.create_spheres_batch(positions, props)

        # Connect in circle
        pairs = []
        for i in range(N):
            o1 = spheres[i]
            o2 = spheres[(i + 1) % N]
            pairs.append((o1, o2))
        
        BatchOperations.connect_spheres_batch(pairs, curved, props)

        return {'FINISHED'}

# =========================================
# PAINT OPERATORS
# =========================================

# Paint Vorschau

def draw_paint_preview(op, context):
    """Zeichne fetzige 3D Vorschau für Paint Mode"""
    if not op.cursor_3d_pos:
        return
        
    import gpu
    from gpu_extras.batch import batch_for_shader
    import blf
    import math
    
    props = context.scene.chain_construction_props
    
    # 3D Shader
    shader_3d = gpu.shader.from_builtin('UNIFORM_COLOR')
    
    # Erstelle Kugel-Geometrie
    def create_sphere_lines(center, radius, segments=16):
        lines = []
        
        # Horizontale Kreise
        for lat in range(-2, 3):
            angle_lat = lat * math.pi / 6
            y = radius * math.sin(angle_lat)
            r = radius * math.cos(angle_lat)
            
            for i in range(segments):
                angle1 = 2 * math.pi * i / segments
                angle2 = 2 * math.pi * (i + 1) / segments
                
                x1 = center.x + r * math.cos(angle1)
                z1 = center.z + r * math.sin(angle1)
                x2 = center.x + r * math.cos(angle2)
                z2 = center.z + r * math.sin(angle2)
                
                lines.extend([(x1, center.y + y, z1), (x2, center.y + y, z2)])
        
        # Vertikale Kreise
        for i in range(8):
            angle = 2 * math.pi * i / 8
            for j in range(segments):
                angle1 = 2 * math.pi * j / segments
                angle2 = 2 * math.pi * (j + 1) / segments
                
                x1 = center.x + radius * math.cos(angle) * math.sin(angle1)
                y1 = center.y + radius * math.cos(angle1)
                z1 = center.z + radius * math.sin(angle) * math.sin(angle1)
                
                x2 = center.x + radius * math.cos(angle) * math.sin(angle2)
                y2 = center.y + radius * math.cos(angle2)
                z2 = center.z + radius * math.sin(angle) * math.sin(angle2)
                
                lines.extend([(x1, y1, z1), (x2, y2, z2)])
        
        return lines
    
    # Erstelle animierte Kugel
    time = context.scene.frame_current * 0.1
    pulse = 1.0 + math.sin(time) * 0.1  # Pulsierender Effekt
    
    # Äußere transparente Kugel
    outer_lines = create_sphere_lines(op.cursor_3d_pos, props.kette_kugelradius * pulse * 1.2)
    outer_batch = batch_for_shader(shader_3d, 'LINES', {"pos": outer_lines})
    
    # Innere Kugel
    inner_lines = create_sphere_lines(op.cursor_3d_pos, props.kette_kugelradius * 0.8)
    inner_batch = batch_for_shader(shader_3d, 'LINES', {"pos": inner_lines})
    
    # Kreuz-Cursor
    cross_size = props.kette_kugelradius * 0.5
    cross_lines = [
        # X-Achse
        (op.cursor_3d_pos.x - cross_size, op.cursor_3d_pos.y, op.cursor_3d_pos.z),
        (op.cursor_3d_pos.x + cross_size, op.cursor_3d_pos.y, op.cursor_3d_pos.z),
        # Y-Achse
        (op.cursor_3d_pos.x, op.cursor_3d_pos.y - cross_size, op.cursor_3d_pos.z),
        (op.cursor_3d_pos.x, op.cursor_3d_pos.y + cross_size, op.cursor_3d_pos.z),
        # Z-Achse
        (op.cursor_3d_pos.x, op.cursor_3d_pos.y, op.cursor_3d_pos.z - cross_size),
        (op.cursor_3d_pos.x, op.cursor_3d_pos.y, op.cursor_3d_pos.z + cross_size),
    ]
    cross_batch = batch_for_shader(shader_3d, 'LINES', {"pos": cross_lines})
    
    # Zeichne mit Tiefentest
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.blend_set('ALPHA')
    gpu.state.line_width_set(2.0)
    
    # Äußere Kugel - transparent rot
    shader_3d.bind()
    shader_3d.uniform_float("color", (1.0, 0.2, 0.2, 0.3))
    outer_batch.draw(shader_3d)
    
    # Innere Kugel - heller
    shader_3d.uniform_float("color", (1.0, 0.5, 0.5, 0.5))
    inner_batch.draw(shader_3d)
    
    # Kreuz - weiß
    gpu.state.line_width_set(3.0)
    shader_3d.uniform_float("color", (1.0, 1.0, 1.0, 0.8))
    cross_batch.draw(shader_3d)
    
    # Verbindungslinie zur letzten Kugel
    if hasattr(op, 'painted_spheres') and op.painted_spheres:
        last_pos = op.painted_spheres[-1].location
        line_batch = batch_for_shader(shader_3d, 'LINES', {
            "pos": [last_pos, op.cursor_3d_pos]
        })
        gpu.state.line_width_set(1.0)
        shader_3d.uniform_float("color", (0.5, 1.0, 0.5, 0.5))
        line_batch.draw(shader_3d)
    
    # Reset
    gpu.state.depth_test_set('NONE')
    gpu.state.blend_set('NONE')
    gpu.state.line_width_set(1.0)
    
    # 2D Text Overlay
    if getattr(op, 'navigation_mode', False):
        text = "NAVIGATION MODE"
        color = (0.5, 1.0, 0.5, 1.0)
    else:
        text = f"PAINT MODE - Radius: {props.kette_kugelradius:.2f}"
        color = (1.0, 0.5, 0.5, 1.0)
    
    # Text position
    region = context.region
    x = region.width // 2 - 100
    y = region.height - 40
    
    blf.size(0, 20)
    blf.color(0, *color)
    blf.position(0, x, y, 0)
    blf.draw(0, text)


class KETTE_OT_paint_mode(Operator):
    """Interaktiver Paint Modus für Kugeln"""
    bl_idname = "kette.paint_mode"
    bl_label = "Paint Modus"
    bl_options = {'REGISTER', 'UNDO', 'BLOCKING'}
    bl_description = "Male Kugeln mit der Maus auf die Oberfläche"
    
    
    @classmethod
    def poll(cls, context):
        props = context.scene.chain_construction_props
        return (context.area.type == 'VIEW_3D' and 
                props.kette_target_obj is not None)
                    
    def modal(self, context, event):
        context.area.tag_redraw()
        
        props = context.scene.chain_construction_props

        # LIVE-MONITORING: Warnung bei Target-Verlust
        if not props.kette_target_obj:
            if not getattr(self, 'target_warning_shown', False):
                self.report({'WARNING'}, "⚠️ Ziel-Objekt verloren!")
                self.target_warning_shown = True
        else:
            self.target_warning_shown = False
        
                # Update cursor position NUR im Paint Modus
        if event.type == 'MOUSEMOVE' and not getattr(self, 'navigation_mode', False):
            if props.kette_target_obj:
                self.cursor_3d_pos = get_paint_surface_position(context, event, props.kette_target_obj)
            else:
                # Fallback ohne Target
                self.cursor_3d_pos = None
                
        # SPACE - Navigation Toggle
        if event.type == 'SPACE' and event.value == 'PRESS':
            self.navigation_mode = not getattr(self, 'navigation_mode', False)
            mode = "NAVIGATION" if self.navigation_mode else "PAINT"
            self.report({'INFO'}, f"Modus: {mode}")
            return {'RUNNING_MODAL'}

        # Navigation Events durchreichen
        if getattr(self, 'navigation_mode', False):
            if event.type not in {'ESC', 'SPACE'}:
                return {'PASS_THROUGH'}
            else:
                return {'RUNNING_MODAL'}
        
        # ESC - Exit (nur im Paint Modus)
        if event.type in {'ESC'}:
            return self.finish(context)
        
        # Left Mouse - Paint
        if event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                self.painting = True
                self.paint_sphere(context, event)
            elif event.value == 'RELEASE':
                self.painting = False
                self.last_position = None
        
        # Paint while dragging
        if self.painting and event.type == 'MOUSEMOVE':
            self.paint_sphere(context, event)
        
        # Right Mouse - Remove spheres
        if event.type == 'RIGHTMOUSE' and event.value == 'PRESS':
            self.remove_sphere_at_cursor(context, event)
        
        # S - Toggle symmetry
        if event.type == 'S' and event.value == 'PRESS':
            props = context.scene.chain_construction_props
            props.paint_use_symmetry = not props.paint_use_symmetry
            self.report({'INFO'}, f"Symmetrie: {'AN' if props.paint_use_symmetry else 'AUS'}")
        
        # C - Connect painted spheres
        if event.type == 'C' and event.value == 'PRESS':
            self.connect_painted_spheres(context)
        
        return {'RUNNING_MODAL'}
        
    def invoke(self, context, event):
        # Initialisierung
        self.draw_handler = None
        self.painting = False
        self.last_position = None
        self.painted_spheres = []
        self.cursor_3d_pos = None
        self.navigation_mode = False
        
        if context.space_data.type != 'VIEW_3D':
            self.report({'WARNING'}, "Muss in 3D View ausgeführt werden")
            return {'CANCELLED'}
        
        # Setup draw handler - HIER wird die fetzige Funktion aufgerufen
        args = (self, context)
        self.draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            draw_paint_preview, args, 'WINDOW', 'POST_VIEW'
        )
        
        context.window_manager.modal_handler_add(self)
        self.report({'INFO'}, "Paint Modus: LMB=Malen, RMB=Löschen, S=Symmetrie, C=Verbinden, ESC=Beenden")
        
        return {'RUNNING_MODAL'}
    
    def finish(self, context):
        """Beende Paint Modus"""
        if self.draw_handler:
            bpy.types.SpaceView3D.draw_handler_remove(self.draw_handler, 'WINDOW')
        
        # Auto-connect if enabled
        props = context.scene.chain_construction_props
        if props.paint_auto_connect and len(self.painted_spheres) > 1:
            self.connect_painted_spheres(context)
        
        self.painted_spheres.clear()
        clear_auras()  # Clean up aura cache
        
        self.report({'INFO'}, "Paint Modus beendet")
        return {'FINISHED'}
        
    def paint_sphere(self, context, event):
        """Male Kugel an Cursor-Position"""
        if not self.cursor_3d_pos:
            return
            
        props = context.scene.chain_construction_props
        
        # Check spacing
        if self.last_position:
            distance = (self.cursor_3d_pos - self.last_position).length
            min_distance = props.kette_kugelradius * props.paint_brush_spacing
            if distance < min_distance:
                return
        
        # Create sphere using the central function
        sphere = create_sphere(
            location=self.cursor_3d_pos.copy(),
            material_color=(0.5, 0.7, 1.0),
            custom_props={'painted': True}
        )
        
            
        if hasattr(self, 'painted_positions'):
            self.painted_positions.append(self.cursor_3d_pos.copy())
        else:
            self.painted_positions = [self.cursor_3d_pos.copy()]
        
        self.painted_spheres.append(sphere)
        self.last_position = self.cursor_3d_pos.copy()
        
        # Symmetry
        if props.paint_use_symmetry:
            mirrored_pos = self.cursor_3d_pos.copy()
            
            if props.paint_symmetry_axis == 'X':
                mirrored_pos.x = -mirrored_pos.x
            elif props.paint_symmetry_axis == 'Y':
                mirrored_pos.y = -mirrored_pos.y
            elif props.paint_symmetry_axis == 'Z':
                mirrored_pos.z = -mirrored_pos.z
            
            # Project mirrored position to surface
            if props.kette_target_obj:
                aura = get_aura(props.kette_target_obj, props.kette_aura_dichte)
                surf_pos, surf_norm = next_surface_point(aura, mirrored_pos)
                offset = props.kette_kugelradius + props.kette_abstand + props.kette_global_offset + EPSILON
                mirrored_pos = surf_pos + surf_norm * offset
            
            # Create mirrored sphere
            sym_sphere = create_sphere(
                location=mirrored_pos,
                material_color=(0.5, 0.7, 1.0),
                custom_props={'painted': True, 'symmetric': True}
            )
            self.painted_spheres.append(sym_sphere)
    
    def remove_sphere_at_cursor(self, context, event):
        """Lösche Kugel unter Cursor"""
        props = context.scene.chain_construction_props
        position = get_paint_surface_position(context, event, props.kette_target_obj)
        if not position:
            return
        
        # Find closest painted sphere
        coll = bpy.data.collections.get("KonstruktionsKette")
        if not coll:
            return
        
        remove_radius = props.kette_kugelradius * 1.5
        to_remove = []
        
        for obj in coll.objects:
            if (obj.name.startswith("Kugel") and 
                obj.get("painted") and 
                (obj.location - position).length < remove_radius):
                to_remove.append(obj)
        
        if to_remove:
            # Remove from painted list
            for obj in to_remove:
                if obj in self.painted_spheres:
                    self.painted_spheres.remove(obj)
            
            # Remove using batch operations
            BatchOperations.remove_objects_batch(to_remove)
            self.report({'INFO'}, f"{len(to_remove)} Kugel(n) gelöscht")
    
    def connect_painted_spheres(self, context):
        """Verbinde gemalte Kugeln"""
        if len(self.painted_spheres) < 2:
            self.report({'INFO'}, "Mindestens 2 Kugeln zum Verbinden benötigt")
            return
        
        props = context.scene.chain_construction_props
        connected = 0
        
        # CHECKBOX AUSLESEN - das ist der wichtige Teil!
        use_curved = props.kette_use_curved_connectors  # <-- Diese Zeile anpassen
        
        # Connect consecutive spheres
        for i in range(len(self.painted_spheres) - 1):
            if (self.painted_spheres[i].name in bpy.data.objects and 
                self.painted_spheres[i+1].name in bpy.data.objects):
                connect_kugeln(
                    self.painted_spheres[i], 
                    self.painted_spheres[i+1], 
                    use_curved  # <-- Checkbox Wert verwenden
                )
                connected += 1
        
        connector_type = "gebogene" if use_curved else "gerade"
        self.report({'INFO'}, f"{connected} {connector_type} Verbindungen erstellt")
        

class KETTE_OT_connect_painted_spheres(Operator):
    bl_idname = "kette.connect_painted_spheres"
    bl_label = "Gemalte Kugeln Verbinden"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Verbindet alle gemalten Kugeln"
    
    connection_mode: EnumProperty(
        name="Verbindungsmodus",
        items=[
            ('SEQUENCE', 'Sequenziell', 'Verbinde in Reihenfolge'),
            ('NEAREST', 'Nächste Nachbarn', 'Verbinde zu nächsten Nachbarn'),
        ],
        default='SEQUENCE'
    )
    
    def execute(self, context):
        props = context.scene.chain_construction_props
        coll = bpy.data.collections.get("KonstruktionsKette")
        if not coll:
            self.report({'WARNING'}, "Keine Kette gefunden")
            return {'CANCELLED'}
        
        # Sammle gemalte Kugeln
        painted = [o for o in coll.objects 
                  if o.name.startswith("Kugel") and o.get("painted")]
        
        if len(painted) < 2:
            self.report({'WARNING'}, "Mindestens 2 gemalte Kugeln benötigt")
            return {'CANCELLED'}
        
        connected = 0
        curved = props.kette_use_curved_connectors
        
        if self.connection_mode == 'SEQUENCE':
            # Sortiere nach Namen (chronologisch)
            painted.sort(key=lambda o: o.name)
            
            # Verbinde aufeinander folgende
            for i in range(len(painted) - 1):
                connect_kugeln(painted[i], painted[i+1], curved)
                connected += 1
                
        elif self.connection_mode == 'NEAREST':
            # KDTree für effiziente Nachbarsuche
            positions = [o.location for o in painted]
            kdt = KDTree(len(positions))
            for i, pos in enumerate(positions):
                kdt.insert(pos, i)
            kdt.balance()
            
            # Verbinde zu nächsten Nachbarn
            connections = set()
            for i, obj in enumerate(painted):
                # Finde 3 nächste Nachbarn
                neighbors = kdt.find_n(obj.location, 4)  # Include self
                
                for _, idx, _ in neighbors:
                    if idx != i:
                        connection = tuple(sorted([i, idx]))
                        if connection not in connections:
                            connections.add(connection)
                            connect_kugeln(painted[i], painted[idx], curved)
                            connected += 1
        
        self.report({'INFO'}, f"{connected} Verbindungen erstellt")
        return {'FINISHED'}
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)
    
    def draw(self, context):
        layout = self.layout
        layout.prop(self, "connection_mode")

class KETTE_OT_clear_painted_spheres(Operator):
    bl_idname = "kette.clear_painted_spheres"
    bl_label = "Gemalte Kugeln Löschen"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Löscht alle gemalten Kugeln"
    
    def execute(self, context):
        coll = bpy.data.collections.get("KonstruktionsKette")
        if not coll:
            self.report({'INFO'}, "Keine Kette gefunden")
            return {'FINISHED'}
        
        to_remove = []
        for obj in coll.objects:
            if obj.name.startswith("Kugel") and obj.get("painted"):
                to_remove.append(obj)
        
        if to_remove:
            BatchOperations.remove_objects_batch(to_remove)
            self.report({'INFO'}, f"{len(to_remove)} gemalte Kugeln gelöscht")
        else:
            self.report({'INFO'}, "Keine gemalten Kugeln gefunden")
        
        return {'FINISHED'}
    
        

# =========================================
# OPERATORS – FINALIZATION
# =========================================

class KETTE_OT_kette_glaetten(Operator):
    bl_idname = "kette.kette_glaetten"
    bl_label = "Glätten & Verschmelzen"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Verschmilzt alle Ketten-Objekte und glättet das Ergebnis"

    def execute(self, context):
        coll = bpy.data.collections.get("KonstruktionsKette")
        if not coll:
            self.report({'ERROR'}, "Keine Kette gefunden!")
            return {'CANCELLED'}

        # Sammle alle Objekte
        meshes_to_join = []
        for obj in coll.objects:
            if obj.type == 'MESH' and (obj.name.startswith("Kugel") or obj.name.startswith("Verbinder")):
                meshes_to_join.append(obj)
            elif obj.type == 'CURVE':
                # Konvertiere Kurven zu Mesh
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.convert(target='MESH')
                meshes_to_join.append(obj)

        if not meshes_to_join:
            self.report({'ERROR'}, "Keine Objekte zum Verschmelzen gefunden!")
            return {'CANCELLED'}

        # Wähle alle und joine
        bpy.ops.object.select_all(action='DESELECT')
        for obj in meshes_to_join:
            obj.select_set(True)
        
        context.view_layer.objects.active = meshes_to_join[0]
        bpy.ops.object.join()

        # Glätten
        merged = context.active_object
        merged.name = "KetteFinalized"
        
        # Remesh Modifier
        mod = merged.modifiers.new("Remesh", 'REMESH')
        mod.mode = 'SMOOTH'
        mod.octree_depth = 7
        mod.use_smooth_shade = True
        
        bpy.context.view_layer.objects.active = merged
        bpy.ops.object.modifier_apply(modifier=mod.name)
        
        # Smooth Modifier
        mod_smooth = merged.modifiers.new("Smooth", 'SMOOTH')
        mod_smooth.iterations = 10
        mod_smooth.factor = 0.5
        bpy.ops.object.modifier_apply(modifier=mod_smooth.name)
        
        # Shade smooth
        bpy.ops.object.shade_smooth()
        
        self.report({'INFO'}, "Kette geglättet und verschmolzen")
        clear_auras()
        return {'FINISHED'}


class KETTE_OT_generate_shell(Operator):
    bl_idname = "kette.generate_shell"
    bl_label = "Shell Erstellen"
    bl_description = "Erstellt TPU Shell mit korrekter Mathematik"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.chain_construction_props
        target_obj = props.kette_target_obj
        
        if not target_obj or target_obj.type != 'MESH':
            self.report({'ERROR'}, "Kein gültiges Zielobjekt gewählt!")
            return {'CANCELLED'}
        
        # === FINDE MAX KUGELRADIUS ===
        max_radius = props.kette_kugelradius  # Fallback
        for obj in bpy.data.objects:
            if obj.name.startswith("Kugel"):
                radius = obj.get("kette_radius")
                if radius is not None:
                    max_radius = max(max_radius, radius)
                else:
                    # Fallback: Durchmesser ÷ 2 = Radius
                    max_dim = max(obj.dimensions)
                    if max_dim > 0:
                        estimated_radius = max_dim * 0.5
                        max_radius = max(max_radius, estimated_radius)
        
        # === FINDE MAX OFFSET ===
        max_offset = max(props.kette_abstand, props.kette_global_offset)
        for obj in bpy.data.objects:
            if obj.name.startswith("Kugel"):
                # Prüfe ob Kugel eigenen Offset hat
                obj_offset = obj.get("kette_abstand")
                if obj_offset is not None:
                    max_offset = max(max_offset, obj_offset)
                obj_global_offset = obj.get("kette_global_offset") 
                if obj_global_offset is not None:
                    max_offset = max(max_offset, obj_global_offset)
        
        # === DEINE MATHEMATIK ===
        diameter = max_radius * 2
        shell_expansion = diameter + 2 * max_offset
        
        # === MATHEMATISCH KORREKTE SKALIERUNGS-BERECHNUNG ===
        # Bei Skalierung um Faktor S entsteht Wandstärke = target_size * (S-1) / 2
        # Umgestellt: S = 1 + (wandstärke * 2) / target_size
        target_dims = target_obj.dimensions
        avg_target_size = (target_dims.x + target_dims.y + target_dims.z) / 3
        scale_factor = 1.0 + (shell_expansion * 2.1) / avg_target_size  # Mit 5% Sicherheitspuffer
        
        # === ENTFERNE ALTE SHELLS ===
        for obj in list(bpy.data.objects):
            if ("shell" in obj.name.lower() or 
                obj.name.startswith("TPU_") or
                "Orthese" in obj.name):
                bpy.data.objects.remove(obj, do_unlink=True)
        
        try:
            # === ERSTELLE SHELL DURCH KOPIE ===
            shell_obj = target_obj.copy()
            shell_obj.data = target_obj.data.copy()
            shell_obj.name = "TPU_Shell_Orthese"
            context.collection.objects.link(shell_obj)
            
            # === SKALIERE SHELL (AUFBLASEN) ===
            shell_obj.scale *= scale_factor
            
            # Transform anwenden
            bpy.ops.object.select_all(action='DESELECT')
            shell_obj.select_set(True)
            context.view_layer.objects.active = shell_obj
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            
            # === TARGET + SHELL JOINEN ===
            target_obj.select_set(True)
            context.view_layer.objects.active = shell_obj
            bpy.ops.object.join()
            
            # Result
            final_shell = context.active_object
            final_shell.name = "TPU_Shell_Orthese"
            
            # === CLEANUP ===
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.remove_doubles(threshold=0.02)
            bpy.ops.mesh.normals_make_consistent(inside=False)
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.ops.object.shade_smooth()
            
            # === TPU MATERIAL ===
            mat_name = "TPU_Shell_Material"
            mat = bpy.data.materials.get(mat_name)
            if not mat:
                mat = bpy.data.materials.new(mat_name)
                mat.use_nodes = True
                bsdf = mat.node_tree.nodes["Principled BSDF"]
                bsdf.inputs['Base Color'].default_value = (0.8, 0.2, 0.8, 1.0)  # Magenta
                bsdf.inputs['Metallic'].default_value = 0.0
                bsdf.inputs['Roughness'].default_value = 0.4
                bsdf.inputs['Transmission'].default_value = 0.05
                mat.diffuse_color = (0.8, 0.2, 0.8, 1.0)
            
            if final_shell.data.materials:
                final_shell.data.materials[0] = mat
            else:
                final_shell.data.materials.append(mat)
            
            # === ERFOLG ===
            vertex_count = len(final_shell.data.vertices)
            
            self.report({'INFO'}, 
                       f"✅ Shell erstellt! Max Radius: {max_radius:.2f}, "
                       f"Max Offset: {max_offset:.2f}, "
                       f"Wandstärke: {shell_expansion:.2f} Units, "
                       f"Skalierung: {scale_factor:.3f}x")
            
        except Exception as e:
            self.report({'ERROR'}, f"Shell-Erstellung fehlgeschlagen: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}
        
        return {'FINISHED'}
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)
    
    def draw(self, context):
        layout = self.layout
        
        # Mathematik-Vorschau
        props = context.scene.chain_construction_props
        if props.kette_target_obj:
            # Berechne Preview
            max_radius = props.kette_kugelradius
            max_offset = max(props.kette_abstand, props.kette_global_offset)
            diameter = max_radius * 2
            shell_expansion = diameter + 2 * max_offset
            
            target_dims = props.kette_target_obj.dimensions
            avg_target_size = (target_dims.x + target_dims.y + target_dims.z) / 3
            scale_factor = 1.0 + (shell_expansion / avg_target_size)
            
            box = layout.box()
            box.label(text="DEINE Mathematik:", icon='PRESET')
            box.label(text=f"Max Radius: {max_radius:.2f}")
            box.label(text=f"Max Offset: {max_offset:.2f}")
            box.label(text=f"Durchmesser: {diameter:.2f}")
            box.label(text=f"Wandstärke: {shell_expansion:.2f} Units")
            box.label(text=f"Skalierung: {scale_factor:.3f}x")
        
        # Info
        info_box = layout.box()
        info_box.label(text="Deine Struktur:", icon='INFO')
        info_box.label(text="1. Target kopieren")
        info_box.label(text="2. Kopie skalieren (aufblasen)")
        info_box.label(text="3. Original + Kopie joinen")
        info_box.label(text="Formel: durchmesser + 2 × max_offset")
               


class KETTE_OT_prepare_for_export(Operator):
    bl_idname = "kette.prepare_for_export"
    bl_label = "Export Vorbereiten"
    bl_description = "Bereitet Dual-Material Orthese für 3D-Druck vor"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        # Suche Shell
        shell_obj = None
        for obj in bpy.data.objects:
            if ("shell" in obj.name.lower() or 
                obj.name.startswith("TPU_") or 
                "Orthese" in obj.name):
                shell_obj = obj
                break
        
        if not shell_obj:
            self.report({'ERROR'}, "Keine Shell gefunden! Erst 'Shell Erstellen' ausführen.")
            return {'CANCELLED'}
        
        # === KETTEN-VERARBEITUNG ===
        # Suche oder erstelle Ketten-Objekt
        chain_obj = None
        for name in ["KetteFinalized", "Kette_Final", "PETG_Chain"]:
            if name in bpy.data.objects:
                chain_obj = bpy.data.objects[name]
                break
        
        if not chain_obj:
            # Verschmelze alle Ketten-Objekte
            coll = bpy.data.collections.get("KonstruktionsKette")
            if not coll:
                self.report({'ERROR'}, "Keine KonstruktionsKette Collection gefunden!")
                return {'CANCELLED'}
            
            meshes_to_join = []
            for obj in coll.objects:
                if obj.type == 'MESH' and (obj.name.startswith("Kugel") or 
                                          obj.name.startswith("Verbinder")):
                    meshes_to_join.append(obj)
                elif obj.type == 'CURVE':
                    # Konvertiere Curves zu Mesh
                    bpy.context.view_layer.objects.active = obj
                    bpy.ops.object.convert(target='MESH')
                    meshes_to_join.append(obj)
            
            if not meshes_to_join:
                self.report({'ERROR'}, "Keine Ketten-Objekte gefunden!")
                return {'CANCELLED'}
            
            try:
                # Verschmelze alle Ketten-Teile
                bpy.ops.object.select_all(action='DESELECT')
                for obj in meshes_to_join:
                    obj.select_set(True)
                context.view_layer.objects.active = meshes_to_join[0]
                bpy.ops.object.join()
                
                chain_obj = context.active_object
                chain_obj.name = "PETG_Chain_Verstärkung"
                
                # WICHTIG: Kette glätten OHNE Position zu verändern
                # Nur Remesh für saubere Topology
                mod = chain_obj.modifiers.new("Remesh", 'REMESH')
                mod.mode = 'SMOOTH'
                mod.octree_depth = 6  # Reduziert für Performance
                mod.use_smooth_shade = True
                bpy.ops.object.modifier_apply(modifier=mod.name)
                
                # Leichte Glättung
                mod_smooth = chain_obj.modifiers.new("Smooth", 'SMOOTH')
                mod_smooth.iterations = 3  # Reduziert um Position zu erhalten
                mod_smooth.factor = 0.3    # Sanfter
                bpy.ops.object.modifier_apply(modifier=mod_smooth.name)
                
                bpy.ops.object.shade_smooth()
                
            except Exception as e:
                self.report({'ERROR'}, f"Ketten-Verschmelzung fehlgeschlagen: {e}")
                return {'CANCELLED'}
        
        # === MATERIAL-ZUWEISUNG ===
        try:
            # PETG Material für Kette (Hart, Grün)
            petg_mat = bpy.data.materials.get("PETG_Chain_Material")
            if not petg_mat:
                petg_mat = bpy.data.materials.new("PETG_Chain_Material")
                petg_mat.use_nodes = True
                bsdf = petg_mat.node_tree.nodes["Principled BSDF"]
                bsdf.inputs['Base Color'].default_value = (0.2, 0.8, 0.2, 1.0)  # Grün
                bsdf.inputs['Metallic'].default_value = 0.1
                bsdf.inputs['Roughness'].default_value = 0.3
                petg_mat.diffuse_color = (0.2, 0.8, 0.2, 1.0)
            
            # Material zu Kette
            if chain_obj.data.materials:
                chain_obj.data.materials[0] = petg_mat
            else:
                chain_obj.data.materials.append(petg_mat)
            
            # TPU Material für Shell sollte bereits vorhanden sein
            
        except Exception as e:
            self.report({'WARNING'}, f"Material-Zuweisung teilweise fehlgeschlagen: {e}")
        
        # === TRANSFORM APPLY ===
        for obj in [chain_obj, shell_obj]:
            if obj:
                bpy.ops.object.select_all(action='DESELECT')
                obj.select_set(True)
                context.view_layer.objects.active = obj
                bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        
        # === FINALE SELEKTION ===
        bpy.ops.object.select_all(action='DESELECT')
        chain_obj.select_set(True)
        shell_obj.select_set(True)
        context.view_layer.objects.active = shell_obj
        
        # === ERFOLGS-MESSAGE ===
        chain_verts = len(chain_obj.data.vertices) if chain_obj else 0
        shell_verts = len(shell_obj.data.vertices) if shell_obj else 0
        
        self.report({'INFO'}, 
                   f"✅ Dual-Material Orthese bereit! "
                   f"PETG: {chain_verts:,} Verts, TPU: {shell_verts:,} Verts")
        
        print("\n" + "="*60)
        print("🎯 DUAL-MATERIAL ORTHESE - EXPORT BEREIT")
        print("="*60)
        print(f"🟢 PETG Verstärkung: {chain_obj.name}")
        print(f"🟣 TPU Shell: {shell_obj.name}")
        print("📤 Export: File → Export → Wavefront OBJ")
        print("🔧 Drucker: Dual-Material Setup erforderlich")
        print("="*60 + "\n")
        
        return {'FINISHED'}

# =========================================
# LIBRARY OPERATORS
# =========================================

class KETTE_OT_refresh_library(Operator):
    bl_idname = "kette.refresh_library"
    bl_label = "Library Aktualisieren"
    bl_description = "Lädt die Pattern Library neu"
    
    def execute(self, context):
        props = context.scene.chain_construction_props
        
        # Clear existing
        props.pattern_library.clear()
        
        # Load patterns
        patterns = load_pattern_library()
        
        for pattern_id, pattern_data in patterns.items():
            item = props.pattern_library.add()
            item.pattern_id = pattern_id
            item.pattern_name = pattern_data.get("name", pattern_id)
            item.pattern_category = pattern_data.get("category", "custom")
            item.pattern_data = json.dumps(pattern_data)
        
        # Reset selection
        props.pattern_library_index = 0
        
        self.report({'INFO'}, f"Library geladen: {len(props.pattern_library)} Patterns")
        return {'FINISHED'}


class KETTE_OT_load_pattern(Operator):
    bl_idname = "kette.load_pattern"
    bl_label = "Pattern Laden"
    bl_description = "Lädt das ausgewählte Pattern"
    
    def execute(self, context):
        props = context.scene.chain_construction_props
        
        if props.pattern_library_index < 0 or props.pattern_library_index >= len(props.pattern_library):
            self.report({'WARNING'}, "Kein Pattern ausgewählt")
            return {'CANCELLED'}
        
        item = props.pattern_library[props.pattern_library_index]
        
        try:
            pattern_data = json.loads(item.pattern_data)
            parameters = pattern_data.get("parameters", {})
            
            # Apply parameters to properties
            applied_count = 0
            for key, value in parameters.items():
                if hasattr(props, key):
                    try:
                        setattr(props, key, value)
                        applied_count += 1
                    except Exception:
                        pass
            
            self.report({'INFO'}, f"Pattern '{item.pattern_name}' geladen ({applied_count} Parameter)")
            
        except Exception as e:
            self.report({'ERROR'}, f"Fehler beim Laden: {str(e)}")
            return {'CANCELLED'}
        
        return {'FINISHED'}
    
    
class KETTE_OT_delete_pattern(Operator):
    bl_idname = "kette.delete_pattern"
    bl_label = "Pattern Löschen"
    bl_description = "Löscht das ausgewählte Pattern aus der Library"
    
    def execute(self, context):
        props = context.scene.chain_construction_props
        
        if props.pattern_library_index < 0 or props.pattern_library_index >= len(props.pattern_library):
            self.report({'WARNING'}, "Kein Pattern ausgewählt")
            return {'CANCELLED'}
        
        item = props.pattern_library[props.pattern_library_index]
        
        # Delete file
        lib_path = ensure_library_directory()
        category = item.pattern_category
        file_path = lib_path / category / f"{item.pattern_id}.json"
        
        try:
            if file_path.exists():
                file_path.unlink()
            
            # Remove from list
            props.pattern_library.remove(props.pattern_library_index)
            props.pattern_library_index = max(0, props.pattern_library_index - 1)
            
            self.report({'INFO'}, f"Pattern '{item.pattern_name}' gelöscht")
            
        except Exception as e:
            self.report({'ERROR'}, f"Fehler beim Löschen: {e}")
            return {'CANCELLED'}
        
        return {'FINISHED'}


class KETTE_OT_refresh_library(Operator):
    bl_idname = "kette.refresh_library"
    bl_label = "Library Aktualisieren"
    bl_description = "Lädt Pattern und Paint Libraries neu"
    
    def execute(self, context):
        props = context.scene.chain_construction_props
        
        # Clear existing
        props.pattern_library.clear()
        
        # Load chain patterns
        chain_patterns = load_pattern_library()
        
        for pattern_id, pattern_data in chain_patterns.items():
            item = props.pattern_library.add()
            item.pattern_id = pattern_id
            item.pattern_name = pattern_data.get("name", pattern_id)
            item.pattern_category = pattern_data.get("category", "custom")
            item.pattern_data = json.dumps(pattern_data)
        
        
        self.report({'INFO'}, f"Library geladen: {len(chain_patterns)} Chain + {len(paint_patterns)} Paint Patterns")
        return {'FINISHED'}

class KETTE_OT_save_current_as_pattern(Operator):
    bl_idname = "kette.save_current_as_pattern"
    bl_label = "Aktuelle Einstellungen Speichern"
    bl_description = "Speichert aktuelle Einstellungen als neues Pattern"
    
    pattern_name: StringProperty(
        name="Pattern-Name",
        default="Neues Pattern"
    )
    pattern_category: EnumProperty(
        name="Kategorie",
        items=[
            ('orthese', 'Orthese', 'Medizinische Orthesen'),
            ('jewelry', 'Schmuck', 'Dekorative Schmuckstücke'),
            ('technical', 'Technisch', 'Engineering/Maschinenbau'),
            ('organic', 'Organisch', 'Natürliche/organische Muster'),
            ('custom', 'Benutzerdefiniert', 'Eigene Kreationen')
        ],
        default='custom'
    )
    
    def invoke(self, context, event):
        props = context.scene.chain_construction_props
        self.pattern_name = f"{props.hop_pattern}_{int(time.time() % 10000)}"
        return context.window_manager.invoke_props_dialog(self)
    
    def draw(self, context):
        layout = self.layout
        layout.prop(self, "pattern_name")
        layout.prop(self, "pattern_category")
    
    def execute(self, context):
        props = context.scene.chain_construction_props
        
        # Sammle alle relevanten Properties
        pattern_data = {
            "name": self.pattern_name,
            "category": self.pattern_category,
            "description": f"Erstellt am {time.strftime('%d.%m.%Y %H:%M')}",
            "parameters": {}
        }
        
        # Alle Hopping + Kette Parameter sammeln
        hopping_params = [
            "hop_pattern", "hop_distance", "hop_iterations", "hop_angle_variation",
            "hop_distance_variation", "hop_probability", "hop_max_spheres",
            "hop_avoid_overlap", "hop_overlap_factor", "hop_branch_probability",
            "hop_generation_scale", "hop_respect_existing", "hop_distance_tolerance",
            "hop_ideal_distance_weight", "hop_cleanup_connections", "hop_fill_missing_connections",
            "hop_min_connection_angle", "pattern_square_diagonal", "organic_min_neighbors",
            "organic_max_neighbors", "hop_pattern_start", "hop_pattern_end",
            "kette_kugelradius", "kette_aufloesung", "kette_abstand", "kette_global_offset",
            "kette_aura_dichte", "kette_use_curved_connectors", "kette_connector_segs",
            "kette_connector_radius_factor", "kette_zyl_segs", "kette_conn_mode",
            "kette_max_abstand", "kette_max_links", "auto_connect"
        ]
        
        for param in hopping_params:
            if hasattr(props, param):
                try:
                    value = getattr(props, param)
                    if not hasattr(value, "name"):  # Skip Object references
                        pattern_data["parameters"][param] = value
                except Exception:
                    pass
        
        # Speichere Pattern
        pattern_id = f"user_{int(time.time())}"
        success = save_pattern_to_library(pattern_data, pattern_id, self.pattern_category)
        
        if success:
            self.report({'INFO'}, f"Pattern '{self.pattern_name}' gespeichert")
            bpy.ops.kette.refresh_library()
        else:
            self.report({'ERROR'}, "Fehler beim Speichern")
            return {'CANCELLED'}
        
        return {'FINISHED'}
    

class KETTE_OT_export_pattern(Operator):
    bl_idname = "kette.export_pattern"
    bl_label = "Pattern Exportieren"
    bl_description = "Exportiert ausgewähltes Pattern als JSON-Datei"
    
    filepath: StringProperty(subtype="FILE_PATH")
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})
    
    def execute(self, context):
        props = context.scene.chain_construction_props
        
        if props.pattern_library_index < 0 or props.pattern_library_index >= len(props.pattern_library):
            self.report({'ERROR'}, "Kein Pattern ausgewählt")
            return {'CANCELLED'}
        
        try:
            item = props.pattern_library[props.pattern_library_index]
            pattern_data = json.loads(item.pattern_data)
            
            # Ergänze Export-Metadaten
            pattern_data.setdefault("metadata", {})
            pattern_data["metadata"]["exported"] = time.strftime("%Y-%m-%d %H:%M:%S")
            pattern_data["metadata"]["exported_from"] = "Kettentool v3.7.0"
            
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(pattern_data, f, indent=2, ensure_ascii=False)
            
            self.report({'INFO'}, f"Pattern '{item.pattern_name}' exportiert")
            
        except Exception as e:
            self.report({'ERROR'}, f"Export fehlgeschlagen: {e}")
            return {'CANCELLED'}
        
        return {'FINISHED'}
    
    def invoke(self, context, event):
        props = context.scene.chain_construction_props
        if props.pattern_library_index >= 0 and props.pattern_library_index < len(props.pattern_library):
            item = props.pattern_library[props.pattern_library_index]
            self.filepath = f"{item.pattern_name}.json"
        
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

class KETTE_OT_import_pattern(Operator):
    bl_idname = "kette.import_pattern"
    bl_label = "Pattern Importieren"
    bl_description = "Importiert Pattern aus JSON-Datei"
    
    filepath: StringProperty(subtype="FILE_PATH")
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})
    
    def execute(self, context):
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                pattern_data = json.load(f)
            
            # Validiere Pattern-Daten
            if "parameters" not in pattern_data:
                self.report({'ERROR'}, "Ungültiges Pattern-Format")
                return {'CANCELLED'}
            
            # Bestimme Kategorie und Name
            category = pattern_data.get("category", "custom")
            name = pattern_data.get("name", pathlib.Path(self.filepath).stem)
            
            # Speichere in Library
            pattern_id = f"imported_{int(time.time())}"
            success = save_pattern_to_library(pattern_data, pattern_id, category)
            
            if success:
                self.report({'INFO'}, f"Pattern '{name}' importiert")
                # Refresh library
                bpy.ops.kette.refresh_library()
            else:
                self.report({'ERROR'}, "Fehler beim Importieren")
                return {'CANCELLED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Import fehlgeschlagen: {e}")
            return {'CANCELLED'}
        
        return {'FINISHED'}
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
class KETTE_OT_select_chain_library_folder(Operator):
    bl_idname = "kette.select_chain_library_folder"
    bl_label = "Chain Pattern Ordner wählen"
    bl_description = "Wähle Ordner mit Chain Pattern JSON-Dateien"
    
    directory: StringProperty(
        name="Ordner",
        subtype='DIR_PATH'
    )
    
    def execute(self, context):
        props = context.scene.chain_construction_props
        props.chain_library_path = self.directory
        
        # Automatisch neu laden
        bpy.ops.kette.refresh_library()
        
        self.report({'INFO'}, f"Chain Library Ordner: {self.directory}")
        return {'FINISHED'}
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


# =========================================
# DEBUG/UTILITY OPERATORS
# =========================================

class KETTE_OT_toggle_debug(Operator):
    bl_idname = "kette.toggle_debug"
    bl_label = "Debug Toggle"
    bl_description = "Aktiviert/Deaktiviert Debug-Ausgabe"
    
    def execute(self, context):
        global debug
        debug.enabled = not debug.enabled
        
        if debug.enabled:
            debug.level = 'DEBUG'
            self.report({'INFO'}, "Debug-Modus aktiviert")
        else:
            self.report({'INFO'}, "Debug-Modus deaktiviert")
        
        return {'FINISHED'}

class KETTE_OT_performance_report(Operator):
    bl_idname = "kette.performance_report"
    bl_label = "Performance Report"
    bl_description = "Zeigt Performance-Statistiken"
    
    def execute(self, context):
        global performance_monitor
        report = performance_monitor.report()
        
        # Print to console
        print("\n" + "="*60)
        print(report)
        print("="*60 + "\n")
        
        self.report({'INFO'}, "Performance Report in Konsole ausgegeben")
        return {'FINISHED'}

class KETTE_OT_clear_caches(Operator):
    bl_idname = "kette.clear_caches"
    bl_label = "Caches Leeren"
    bl_description = "Leert alle internen Caches"
    
    def execute(self, context):
        global _bvh_cache, _aura_cache, _existing_connectors, _remembered_pairs
        
        # Clear auras first (removes objects)
        clear_auras()
        
        # Clear other caches
        _bvh_cache.clear()
        _existing_connectors.clear()
        _remembered_pairs.clear()
        
        self.report({'INFO'}, "Alle Caches geleert")
        return {'FINISHED'}

# =========================================
# UI PANELS
# =========================================

class KETTE_PT_main_panel(Panel):
    bl_label = "Chain Construction Advanced"
    bl_idname = "KETTE_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tools"
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.chain_construction_props
        
        # Target Object
        box = layout.box()
        box.label(text="Zielobjekt", icon='OBJECT_DATA')
        box.prop(props, "kette_target_obj", text="")
        
        if not props.kette_target_obj:
            box.label(text="Wähle ein Mesh-Objekt!", icon='ERROR')

class KETTE_PT_basic_settings(Panel):
    bl_label = "Basis Einstellungen"
    bl_idname = "KETTE_PT_basic_settings"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tools"
    bl_parent_id = "KETTE_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.chain_construction_props
        
        col = layout.column(align=True)
        col.prop(props, "kette_kugelradius")
        col.prop(props, "kette_aufloesung")
        col.separator()
        col.prop(props, "kette_abstand")
        col.prop(props, "kette_global_offset")
        col.prop(props, "kette_aura_dichte")

class KETTE_PT_hopping_pattern(Panel):
    bl_label = "Hopping Pattern Generator"
    bl_idname = "KETTE_PT_hopping_pattern"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tools"
    bl_parent_id = "KETTE_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        return context.scene.chain_construction_props.kette_target_obj is not None
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.chain_construction_props
        
        # Pattern Controls (OHNE apply_preset)
        row = layout.row(align=True)
        row.operator("kette.save_current_as_pattern", text="Speichern", icon='FILE_NEW')
        row.operator("kette.clear_pattern", text="Löschen", icon='TRASH')
        
        # Pattern Type
        layout.prop(props, "hop_pattern")
        
        # Basic Parameters
        box = layout.box()
        box.label(text="Parameter", icon='SETTINGS')
        if props.show_hop_pattern_params:
            col = box.column(align=True)
            col.prop(props, "hop_distance")
            col.prop(props, "hop_iterations")
            col.prop(props, "hop_max_spheres")
            col.separator()
            col.prop(props, "hop_probability")
            col.prop(props, "auto_connect")
        
        # Pattern-specific settings
        if props.hop_pattern == 'SQUARE':
            box.prop(props, "pattern_square_diagonal")
        elif props.hop_pattern == 'ORGANIC':
            col = box.column(align=True)
            col.prop(props, "organic_min_neighbors")
            col.prop(props, "organic_max_neighbors")
        elif props.hop_pattern == 'TRANSITION':
            col = box.column(align=True)
            col.prop(props, "hop_pattern_start")
            col.prop(props, "hop_pattern_end")
        
        # Advanced
        box = layout.box()
        box.prop(props, "show_hop_advanced", 
                text="Erweiterte Einstellungen",
                icon='TRIA_DOWN' if props.show_hop_advanced else 'TRIA_RIGHT')
        
        if props.show_hop_advanced:
            col = box.column(align=True)
            col.prop(props, "hop_angle_variation")
            col.prop(props, "hop_distance_variation")
            col.prop(props, "hop_branch_probability")
            col.prop(props, "hop_generation_scale")
            col.separator()
            col.prop(props, "hop_avoid_overlap")
            if props.hop_avoid_overlap:
                col.prop(props, "hop_overlap_factor")
            col.separator()
            col.prop(props, "hop_respect_existing")
            col.prop(props, "hop_distance_tolerance")
            col.prop(props, "hop_ideal_distance_weight")
            col.separator()
            col.prop(props, "hop_cleanup_connections")
            col.prop(props, "hop_fill_missing_connections")
            col.prop(props, "hop_min_connection_angle")
        
        # Generate button
        layout.separator()
        row = layout.row(align=True)
        row.scale_y = 1.5
        row.operator("kette.generate_hopping_pattern", 
                    text="Hopping Pattern Generieren", 
                    icon='PARTICLE_POINT')
        
        # Library Integration (kompakt)
        layout.separator()
        lib_box = layout.box()
        lib_box.label(text="Pattern Library", icon='PRESET')
        
        # Library Controls
        lib_row = lib_box.row(align=True)
        lib_row.operator("kette.refresh_library", text="Refresh", icon='FILE_REFRESH')
        
        # Pattern Selection (kompakt)
        if len(props.pattern_library) > 0:
            lib_box.template_list("UI_UL_list", "pattern_library_hopping",
                                 props, "pattern_library", 
                                 props, "pattern_library_index",
                                 rows=2)
            
            if props.pattern_library_index >= 0:
                selected_pattern = props.pattern_library[props.pattern_library_index]
                
                # Pattern Info (kompakt)
                info_row = lib_box.row()
                info_row.label(text=f"→ {selected_pattern.pattern_name}")
                info_row.label(text=f"({selected_pattern.pattern_category})")
                
                # Load Button
                load_row = lib_box.row()
                load_row.scale_y = 1.2
                load_row.operator("kette.load_pattern", text="Load Selected", icon='IMPORT')
        else:
            lib_box.label(text="Keine Patterns - Refresh klicken", icon='INFO')
            
    
class KETTE_PT_paint_tools(Panel):
    bl_label = "Paint Tools"
    bl_idname = "KETTE_PT_paint_tools"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tools"
    bl_parent_id = "KETTE_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.chain_construction_props
        
        # Paint Mode
        row = layout.row(align=True)
        row.scale_y = 1.5
        row.operator("kette.paint_mode", text="Paint Modus", icon='BRUSH_DATA')
        
        # Paint Settings
        settings_box = layout.box()
        settings_box.label(text="Paint Einstellungen", icon='SETTINGS')
        col = settings_box.column(align=True)
        col.prop(props, "paint_brush_spacing")
        col.prop(props, "paint_auto_connect")
        col.separator()
        col.prop(props, "paint_use_symmetry")
        if props.paint_use_symmetry:
            col.prop(props, "paint_symmetry_axis", text="")
        
        
        

class KETTE_PT_basic_tools(Panel):
    bl_label = "Basis Werkzeuge"
    bl_idname = "KETTE_PT_basic_tools"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tools"
    bl_parent_id = "KETTE_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        
        # Creation Tools
        col = layout.column(align=True)
        col.label(text="Erstellen:", icon='ADD')
        col.operator("kette.kugel_platzieren", icon='MESH_UVSPHERE')
        col.operator("kette.richtungspfeil_erstellen", icon='EMPTY_SINGLE_ARROW')
        
        # Modification Tools
        col = layout.column(align=True)
        col.label(text="Bearbeiten:", icon='MODIFIER')
        col.operator("kette.snap_surface", icon='SNAP_FACE')
        col.operator("kette.kette_update", icon='FILE_REFRESH')
        
        # Connection Tools
        col = layout.column(align=True)
        col.label(text="Verbinden:", icon='LINKED')
        row = col.row(align=True)
        row.operator("kette.line_connector", text="Gerade")
        row.operator("kette.curved_connector", text="Gebogen")
        col.operator("kette.automatisch_verbinden", icon='DRIVER_DISTANCE')

class KETTE_PT_growth_tools(Panel):
    bl_label = "Wachstums-Werkzeuge"
    bl_idname = "KETTE_PT_growth_tools"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tools"
    bl_parent_id = "KETTE_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.chain_construction_props
        
        # Growth Direction
        layout.prop(props, "kette_growth_dir", text="Richtung")
        if props.kette_growth_dir == 'ARROW':
            layout.prop(props, "kette_richtungs_obj", text="Pfeil")
        
        # Growth Parameters
        col = layout.column(align=True)
        col.prop(props, "kette_chain_abstand")
        col.prop(props, "kette_chain_anzahl")
        col.prop(props, "kette_sequentiell")
        
        # Growth Operations
        layout.separator()
        col = layout.column(align=True)
        col.operator("kette.kette_wachstum", icon='SORT_DESC')
        col.operator("kette.kreis_kette", icon='MESH_CIRCLE')

class KETTE_PT_connector_settings(Panel):
    bl_label = "Verbinder Einstellungen"
    bl_idname = "KETTE_PT_connector_settings"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tools"
    bl_parent_id = "KETTE_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.chain_construction_props
        
        # Connector Type
        layout.prop(props, "kette_use_curved_connectors")
        
        # Connector Parameters
        col = layout.column(align=True)
        if props.kette_use_curved_connectors:
            col.prop(props, "kette_connector_segs")
        else:
            col.prop(props, "kette_zyl_segs")
        col.prop(props, "kette_connector_radius_factor")
        
        # Connection Mode
        layout.separator()
        layout.prop(props, "kette_conn_mode", text="Modus")
        
        if props.kette_conn_mode == 'NEIGHBOR':
            col = layout.column(align=True)
            col.prop(props, "kette_max_abstand")
            col.prop(props, "kette_max_links")

class KETTE_PT_finalization(Panel):
    bl_label = "Finalisierung"
    bl_idname = "KETTE_PT_finalization"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tools"
    bl_parent_id = "KETTE_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        
        col = layout.column(align=True)
        col.operator("kette.kette_glaetten", icon='MOD_SMOOTH')
        col.operator("kette.export_chain", icon='EXPORT')


class KETTE_PT_dual_material_panel(bpy.types.Panel):
    bl_label = "Dual-Material Export"
    bl_idname = "KETTE_PT_dual_material"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tools"
    bl_parent_id = "KETTE_PT_main_panel"
    
    def draw(self, context):
        layout = self.layout
        
        col = layout.column(align=True)
        col.label(text="Dual-Material Export:", icon='EXPORT')
        
        col.operator("kette.generate_shell", text="1. Shell Erstellen", icon='MESH_MONKEY')
        col.operator("kette.prepare_for_export", text="2. Export Vorbereiten", icon='EXPORT')
        
        box = layout.box()
        box.label(text="Workflow:", icon='INFO')
        box.label(text="1. Kette erstellen")
        box.label(text="2. Shell Erstellen → TPU Hülle")  
        box.label(text="3. Export Vorbereiten → Glättung + Materialien")
        box.label(text="4. File → Export → Wavefront OBJ")

class KETTE_PT_debug(Panel):
    bl_label = "Debug & Utilities"
    bl_idname = "KETTE_PT_debug"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tools"
    bl_parent_id = "KETTE_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        
        col = layout.column(align=True)
        col.operator("kette.toggle_debug", text="Debug Toggle", icon='CONSOLE')
        col.operator("kette.performance_report", text="Performance Report", icon='SORTTIME')
        col.operator("kette.clear_caches", text="Caches Leeren", icon='FILE_REFRESH')

# =========================================
# REGISTRATION
# =========================================

classes = [
    # Property Groups
    PatternLibraryItem,
    ChainConstructionProperties,
    ChainConstructionPreferences,  
    
    # Hopping Pattern Operators
    KETTE_OT_generate_hopping_pattern,
    KETTE_OT_save_hopping_pattern,
    KETTE_OT_clear_pattern,
    
    # Basic Tool Operators
    KETTE_OT_kugel_platzieren,
    KETTE_OT_snap_surface,
    KETTE_OT_line_connector,
    KETTE_OT_curved_connector,
    KETTE_OT_kette_update,
    KETTE_OT_automatisch_verbinden,
    
    # Growth Tool Operators
    KETTE_OT_richtungspfeil_erstellen,
    KETTE_OT_kette_wachstum,
    KETTE_OT_kreis_kette,
    
    # Paint Tool Operators
    KETTE_OT_paint_mode,
    KETTE_OT_connect_painted_spheres,
    KETTE_OT_clear_painted_spheres,
    
    # Finalization Operators
    KETTE_OT_kette_glaetten,
    KETTE_OT_generate_shell,
    KETTE_OT_prepare_for_export,
    
    # Library Operators
    KETTE_OT_refresh_library,
    KETTE_OT_load_pattern,
    KETTE_OT_delete_pattern,
    KETTE_OT_save_current_as_pattern,
    KETTE_OT_export_pattern,
    KETTE_OT_import_pattern,
    KETTE_OT_select_chain_library_folder,
    
    # Debug/Utility Operators
    KETTE_OT_toggle_debug,
    KETTE_OT_performance_report,
    KETTE_OT_clear_caches,
    
    # UI Panels
    KETTE_PT_main_panel,
    KETTE_PT_basic_settings,
    KETTE_PT_hopping_pattern,
    KETTE_PT_paint_tools,
    KETTE_PT_basic_tools,
    KETTE_PT_growth_tools,
    KETTE_PT_connector_settings,
    KETTE_PT_finalization,
    KETTE_PT_dual_material_panel,
    KETTE_PT_debug,
]

def register():
    # Register classes
    for cls in classes:
        bpy.utils.register_class(cls)
    
    # Register properties
    bpy.types.Scene.chain_construction_props = PointerProperty(type=ChainConstructionProperties)
    bpy.types.Scene.chain_library_path = StringProperty(
        name="Library Path",
        default="",
        subtype='DIR_PATH'
    )
    
    # Initialize debug from preferences
    try:
        if __name__ != "__main__":
            prefs = bpy.context.preferences.addons[__name__].preferences
            debug.enabled = prefs.enable_debug
    except Exception:
        pass
    
    print("Chain Construction Advanced v3.7.0 registered")

def unregister():
    # Clean up any remaining auras/caches
    clear_auras()
    
    # Unregister classes
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    
    # Remove properties
    del bpy.types.Scene.chain_construction_props
    del bpy.types.Scene.chain_library_path
    
    print("Chain Construction Advanced unregistered")

if __name__ == "__main__":
    # Unregister if already registered (for script reload)
    try:
        unregister()
    except Exception:
        pass
    
    register()
    
    # Set debug mode for testing
    debug.enabled = True
    debug.level = 'INFO'



