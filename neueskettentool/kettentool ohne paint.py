bl_info = {
    "name": "Chain Construction Advanced (Kettenkonstruktion)",
    "author": "Michael Fleischer & Contributors",
    "version": (3, 6, 0),
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
# HOPPING PATTERN PRESETS
# =========================================

PATTERN_PRESETS = {
    'ORTHO_LIGHT': {
        'name': 'Orthese Leicht',
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
    },
    'ORTHO_STRONG': {
        'name': 'Orthese Stark',
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
    },
    'JEWELRY': {
        'name': 'Schmuck',
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
    },
    'TECHNICAL': {
        'name': 'Technisch',
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
}

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

def get_aura(target_obj: bpy.types.Object, thickness: float) -> bpy.types.Object:
    if not target_obj or target_obj.name not in bpy.data.objects:
        clear_auras()
        raise ReferenceError("Target object has been removed")

    # Erstelle eindeutigen Cache-Key mit Objekt UND Dicke
    cache_key = f"{target_obj.name}_{thickness:.4f}"
    
    # Prüfe ob wir schon eine Aura mit dieser Dicke haben
    if cache_key in _aura_cache:
        aura = _aura_cache[cache_key]
        try:
            if aura.name in bpy.data.objects:
                return aura
        except ReferenceError:
            pass
        _aura_cache.pop(cache_key, None)

    # Lösche alte Auras mit anderer Dicke für dieses Objekt
    to_remove = []
    for key in list(_aura_cache.keys()):
        if key.startswith(f"{target_obj.name}_") and key != cache_key:
            to_remove.append(key)
    
    for key in to_remove:
        old_aura = _aura_cache.pop(key, None)
        if old_aura:
            try:
                bpy.data.objects.remove(old_aura, do_unlink=True)
            except:
                pass

    aura = target_obj.copy()
    aura.data = target_obj.data.copy()
    aura.name = f"{target_obj.name}_CachedAura_{thickness:.4f}"
    bpy.context.collection.objects.link(aura)
    aura.hide_viewport = True
    aura.hide_render = True
    aura.hide_set(True)

    # Recalculate normals before solidify
    bm = bmesh.new()
    bm.from_mesh(aura.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(aura.data)
    bm.free()

    # Apply solidify
    mod = aura.modifiers.new("Solidify", 'SOLIDIFY')
    mod.thickness = thickness
    mod.offset = 1.0
    bpy.context.view_layer.objects.active = aura
    bpy.ops.object.modifier_apply(modifier=mod.name)

    # Recalculate normals after solidify
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
    bm = bmesh.new()
    bm.from_mesh(obj.data)
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
        except:
            pass

    _aura_cache.clear()
    _bvh_cache.clear()

@performance_monitor.time_function("next_surface_point", "AURA")
def next_surface_point(aura_obj: bpy.types.Object, pos: Vector) -> Tuple[Vector, Vector]:
    if not aura_obj or aura_obj.type != 'MESH':
        return pos, Vector((0, 0, 1))

    bvh = _build_bvh(aura_obj)
    hit = bvh.find_nearest(pos)
    if not hit:
        debug.trace('AURA', f"No hit for position {pos}")
        return pos, Vector((0, 0, 1))

    loc, norm, *_ = hit
    norm = norm.normalized() if norm.length > 1e-6 else Vector((0, 0, 1))

    # Orient normal outward
    origin = aura_obj.matrix_world.translation
    if (loc - origin).dot(norm) < 0:
        norm = -norm

    return loc, norm

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
    curve.bevel_depth = 1.0
    curve.bevel_mode = 'OBJECT'
    
    obj = bpy.data.objects.new("CurvedConnector", curve)
    
    # Set bevel object
    bevel_name = "_KetteBevelCircle"
    bevel = bpy.data.objects.get(bevel_name)
    if not bevel:
        bevel_curve = bpy.data.curves.new(bevel_name, 'CURVE')
        bevel_curve.dimensions = '2D'
        bevel_sp = bevel_curve.splines.new('BEZIER')
        bevel_sp.bezier_points.add(3)
        for i, (x, y) in enumerate([(1, 0), (0, 1), (-1, 0), (0, -1)]):
            pt = bevel_sp.bezier_points[i]
            pt.co = (x, y, 0)
            pt.handle_left_type = 'AUTO'
            pt.handle_right_type = 'AUTO'
        bevel_sp.use_cyclic_u = True
        bevel = bpy.data.objects.new(bevel_name, bevel_curve)
    
    obj.data.bevel_object = bevel
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
                world_normal = target_obj.matrix_world.to_3x3() @ normal
                world_normal.normalize()
                
                # Apply offset like everywhere else
                offset = (props.kette_kugelradius + 
                         props.kette_abstand + 
                         props.kette_global_offset + 
                         EPSILON)
                
                final_pos = world_location + world_normal * offset
                
                debug.trace('PAINT', f"Hit surface at: {final_pos}")
                return final_pos
            else:
                debug.trace('PAINT', "No surface hit")
                
        except Exception as e:
            debug.error('PAINT', f"Surface snapping failed: {e}")
    
    # Fallback: intersect with ground plane (Z=0)
    if abs(view_vector.z) > 0.001:  # Avoid division by zero
        t = -ray_origin.z / view_vector.z
        if t > 0:
            fallback_pos = ray_origin + view_vector * t
            debug.trace('PAINT', f"Fallback to ground plane: {fallback_pos}")
            return fallback_pos
    
    # Last resort: fixed distance from camera
    fallback_pos = ray_origin + view_vector * 5.0
    debug.trace('PAINT', f"Last resort position: {fallback_pos}")
    return fallback_pos

# =========================================
# BATCH OPERATIONS
# =========================================

class BatchOperations:
    @staticmethod
    @performance_monitor.time_function("create_spheres_batch", "GENERAL")
    def create_spheres_batch(positions: List[Vector], props) -> List[bpy.types.Object]:
        debug.info('GENERAL', f"Creating {len(positions)} spheres in batch")

        spheres: List[bpy.types.Object] = []
        base_mesh = bpy.data.meshes.new("BaseSphere")
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(
            bm,
            u_segments=props.kette_aufloesung,
            v_segments=props.kette_aufloesung,
            radius=props.kette_kugelradius
        )
        bm.to_mesh(base_mesh)
        bm.free()
        
        mat_name = "KetteSchwarz"
        dummy = bpy.data.objects.new("dummy", base_mesh)
        set_material(dummy, mat_name, (0.02, 0.02, 0.02), metallic=0.8, roughness=0.2)
        mat = bpy.data.materials.get(mat_name)
        bpy.data.objects.remove(dummy)
        
        coll = ensure_collection("KonstruktionsKette")
        
        for i, pos in enumerate(positions):
            mesh_copy = base_mesh.copy()
            sphere = bpy.data.objects.new(f"Kugel_{i:03d}", mesh_copy)
            sphere.location = pos
            sphere["kette_radius"] = props.kette_kugelradius
            if mesh_copy.materials:
                mesh_copy.materials[0] = mat
            else:
                mesh_copy.materials.append(mat)
            coll.objects.link(sphere)
            spheres.append(sphere)
        
        bpy.data.meshes.remove(base_mesh)
        return spheres

    @staticmethod
    @performance_monitor.time_function("remove_objects_batch", "GENERAL")
    def remove_objects_batch(objects: List[bpy.types.Object]):
        debug.info('GENERAL', f"Removing {len(objects)} objects in batch")
        
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
    @performance_monitor.time_function("connect_spheres_batch", "GENERAL")
    def connect_spheres_batch(pairs: List[Tuple[bpy.types.Object, bpy.types.Object]], 
                             use_curved: bool, props):
        debug.info('GENERAL', f"Connecting {len(pairs)} sphere pairs in batch")
        
        for o1, o2 in pairs:
            connect_kugeln(o1, o2, use_curved)

# =========================================
# PAINT PATTERN LIBRARY
# =========================================

def ensure_paint_library_directory() -> pathlib.Path:
    """Erstellt und gibt Paint Library Verzeichnis zurück"""
    lib_path = pathlib.Path.home() / "Documents" / "blender_paint_patterns"
    lib_path.mkdir(parents=True, exist_ok=True)
    
    # Erstelle README wenn nicht vorhanden
    readme_path = lib_path / "README.txt"
    if not readme_path.exists():
        readme_path.write_text(
            "Blender Paint Pattern Library\n"
            "============================\n\n"
            "Dieses Verzeichnis enthält gespeicherte Paint-Patterns für das Chain Tool.\n"
            "Jedes Pattern ist eine .json Datei mit Positions- und Verbindungsdaten.\n"
        )
    
    return lib_path

def save_paint_pattern(pattern_data: Dict[str, Any], filename: str) -> bool:
    """Speichert Paint Pattern in Library"""
    try:
        lib_path = ensure_paint_library_directory()
        file_path = lib_path / f"{filename}.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(pattern_data, f, indent=2)
        
        debug.info('PAINT', f"Saved pattern to {file_path}")
        return True
        
    except Exception as e:
        debug.error('PAINT', f"Failed to save pattern: {e}")
        return False

def load_paint_patterns() -> Dict[str, Dict[str, Any]]:
    """Lädt alle verfügbaren Paint Patterns"""
    patterns = {}
    
    try:
        lib_path = ensure_paint_library_directory()
        
        for file_path in lib_path.glob("*.json"):
            if file_path.name != "README.txt":
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        pattern_id = file_path.stem
                        patterns[pattern_id] = data
                except Exception as e:
                    debug.error('PAINT', f"Error loading {file_path.name}: {e}")
    
    except Exception as e:
        debug.error('PAINT', f"Error accessing paint library: {e}")
    
    return patterns

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
    except:
        pass
    
    # Fallback: Check scene property
    if not lib_path:
        try:
            scene_path = bpy.context.scene.chain_library_path
            if scene_path:
                lib_path = pathlib.Path(scene_path)
        except:
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
        "addon_version": "3.6.0"
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
    """Erstellt Beispiel-Patterns wenn Library leer ist"""
    examples = [
        {
            "name": "Orthese Standard",
            "category": "orthese",
            "parameters": PATTERN_PRESETS['ORTHO_LIGHT']
        },
        {
            "name": "Orthese Verstärkt",
            "category": "orthese", 
            "parameters": PATTERN_PRESETS['ORTHO_STRONG']
        },
        {
            "name": "Schmuck Elegant",
            "category": "jewelry",
            "parameters": PATTERN_PRESETS['JEWELRY']
        },
        {
            "name": "Technisch Präzise",
            "category": "technical",
            "parameters": PATTERN_PRESETS['TECHNICAL']
        }
    ]
    
    for i, example in enumerate(examples):
        pattern_id = f"example_{i:03d}"
        save_pattern_to_library(example, pattern_id)

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
    hop_pattern_name: StringProperty(
        name="Pattern-Name",
        default="Mein Hopping Pattern",
        description="Name für das zu speichernde Pattern"
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
        name="Brush Spacing",
        default=3,
        min=0.5,
        max=20.0,
        description="Spacing zwischen gemalten Kugeln (Multiplikator des Kugelradius)"
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
            
            bpy.ops.mesh.primitive_uv_sphere_add(
                segments=props.kette_aufloesung,
                ring_count=props.kette_aufloesung,
                radius=props.kette_kugelradius,
                location=loc + norm * offset
            )
            start_sphere = context.active_object
            start_sphere.name = "Kugel_Start"
            start_sphere["kette_radius"] = props.kette_kugelradius
            
            coll = ensure_collection("KonstruktionsKette")
            coll.objects.link(start_sphere)
            for c in list(start_sphere.users_collection):
                if c != coll:
                    c.objects.unlink(start_sphere)
            
            set_material(start_sphere, "KetteStart", (1.0, 0.0, 0.0), metallic=0.5, roughness=0.5)
        
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

class KETTE_OT_apply_preset(Operator):
    bl_idname = "kette.apply_preset"
    bl_label = "Preset anwenden"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Wendet vordefinierte Einstellungen an"
    
    preset_type: EnumProperty(
        name="Preset",
        items=[
            ('ORTHO_LIGHT', 'Orthese Leicht', 'Luftiges Muster für Beweglichkeit'),
            ('ORTHO_STRONG', 'Orthese Stark', 'Dichtes Muster für Stabilität'),
            ('JEWELRY', 'Schmuck', 'Dekoratives Muster'),
            ('TECHNICAL', 'Technisch', 'Präzises Engineering-Muster'),
        ],
        default='ORTHO_LIGHT'
    )
    
    def execute(self, context):
        props = context.scene.chain_construction_props
        
        preset = PATTERN_PRESETS.get(self.preset_type)
        if not preset:
            self.report({'ERROR'}, "Preset nicht gefunden")
            return {'CANCELLED'}
        
        for key, value in preset.items():
            if hasattr(props, key) and key != 'name':
                try:
                    setattr(props, key, value)
                except Exception as e:
                    debug.warning('GENERAL', f"Could not set {key}: {e}")
        
        self.report({'INFO'}, f"Preset '{preset['name']}' angewendet")
        return {'FINISHED'}
    
    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)
    
    def draw(self, context):
        layout = self.layout
        layout.prop(self, "preset_type")

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
                except:
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
            
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=props.kette_aufloesung,
            ring_count=props.kette_aufloesung,
            radius=props.kette_kugelradius,
            location=cursor_loc
        )
        sphere = context.active_object
        sphere.name = "Kugel_Start"
        sphere["kette_radius"] = props.kette_kugelradius
        
        coll = ensure_collection("KonstruktionsKette")
        coll.objects.link(sphere)
        for c in list(sphere.users_collection):
            if c != coll:
                c.objects.unlink(sphere)
        
        set_material(sphere, "KetteStart", (1.0, 0.0, 0.0), metallic=0.5, roughness=0.5)
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

class KETTE_OT_paint_mode(Operator):
    """Interaktiver Paint Modus für Kugeln"""
    bl_idname = "kette.paint_mode"
    bl_label = "Paint Modus"
    bl_options = {'REGISTER', 'UNDO', 'BLOCKING'}
    bl_description = "Interaktives Malen von Kugeln mit der Maus"
    
    def __init__(self):
        self.draw_handler = None
        self.painting = False
        self.last_position = None
        self.painted_spheres: List[bpy.types.Object] = []
        self.cursor_3d_pos = None
    
    @classmethod
    def poll(cls, context):
        return context.area.type == 'VIEW_3D'
    
    def modal(self, context, event):
        context.area.tag_redraw()
        
        # Update cursor position for preview
        if event.type == 'MOUSEMOVE':
            self.cursor_3d_pos = get_paint_surface_position(
                context, event, 
                context.scene.chain_construction_props.kette_target_obj
            )
        
        # ESC - Exit paint mode
        if event.type in {'ESC'}:
            return self.finish(context)
        
        # Left Mouse - Paint spheres
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
        if context.space_data.type != 'VIEW_3D':
            self.report({'WARNING'}, "Muss in 3D View ausgeführt werden")
            return {'CANCELLED'}
        
        # Setup draw handler
        args = (self, context)
        self.draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            self.draw_callback, args, 'WINDOW', 'POST_VIEW'
        )
        
        context.window_manager.modal_handler_add(self)
        self.report({'INFO'}, 
                   "Paint Modus: LMB=Malen, RMB=Löschen, S=Symmetrie, C=Verbinden, ESC=Beenden")
        
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
        """Male Kugel an Cursor-Position - nutzt bestehende Logik"""
        props = context.scene.chain_construction_props
        
        # Get 3D position using existing surface snapping logic
        position = get_paint_surface_position(context, event, props.kette_target_obj)
        if not position:
            return
        
        # Check spacing - nutzt kette_kugelradius
        if self.last_position:
            distance = (position - self.last_position).length
            min_distance = props.kette_kugelradius * props.paint_brush_spacing
            if distance < min_distance:
                return
        
        # Create positions (main + symmetry)
        positions = [position]
        
        if props.paint_use_symmetry:
            sym_pos = position.copy()
            if props.paint_symmetry_axis == 'X':
                sym_pos.x = -sym_pos.x
            elif props.paint_symmetry_axis == 'Y':
                sym_pos.y = -sym_pos.y
            elif props.paint_symmetry_axis == 'Z':
                sym_pos.z = -sym_pos.z
            positions.append(sym_pos)
        
        # Create spheres using existing BatchOperations - SAME as hopping tool
        new_spheres = BatchOperations.create_spheres_batch(positions, props)
        
        # Mark as painted and add to list
        for sphere in new_spheres:
            sphere["painted"] = True
            self.painted_spheres.append(sphere)
        
        self.last_position = position
    
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
            
            # Remove using existing batch logic
            BatchOperations.remove_objects_batch(to_remove)
            self.report({'INFO'}, f"{len(to_remove)} Kugel(n) gelöscht")
    
    def connect_painted_spheres(self, context):
        """Verbinde gemalte Kugeln - nutzt bestehende connect_kugeln Logik"""
        if len(self.painted_spheres) < 2:
            self.report({'INFO'}, "Mindestens 2 Kugeln zum Verbinden benötigt")
            return
        
        props = context.scene.chain_construction_props
        connected = 0
        
        # Connect consecutive spheres
        for i in range(len(self.painted_spheres) - 1):
            if (self.painted_spheres[i].name in bpy.data.objects and 
                self.painted_spheres[i+1].name in bpy.data.objects):
                # Nutze bestehende connect_kugeln Funktion
                connect_kugeln(
                    self.painted_spheres[i], 
                    self.painted_spheres[i+1], 
                    props.kette_use_curved_connectors
                )
                connected += 1
        
        self.report({'INFO'}, f"{connected} Verbindungen erstellt")
    
    def draw_callback(self, context):
        """Draw preview circle at cursor"""
        if not self.cursor_3d_pos:
            return
        
        try:
            props = context.scene.chain_construction_props
            
            # Setup shader
            shader = gpu.shader.from_builtin('3D_UNIFORM_COLOR')
            
            # Draw circle at cursor - size = kette_kugelradius
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
            
            gpu.state.blend_set('NONE')
            gpu.state.line_width_set(1.0)
            gpu.state.point_size_set(1.0)
            
        except Exception as e:
            print(f"Paint Draw callback error: {e}")

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
            
            # Verbinde aufeinander folgende - nutzt bestehende Logik
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
                neighbors = kdt.find_n(obj.location, 4)  # Include self, so get 4
                
                for _, idx, _ in neighbors:
                    if idx != i:
                        # Avoid duplicate connections
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
        merged.name = "Kette_Final"
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.remove_doubles(threshold=0.001)
        bpy.ops.mesh.normals_make_consistent()
        
        # Smoothing
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.shade_smooth()
        
        # Optional: Subdivision
        subsurf = merged.modifiers.new("Subdivision", 'SUBSURF')
        subsurf.levels = 1
        subsurf.render_levels = 2

        self.report({'INFO'}, "Kette geglättet und verschmolzen!")
        return {'FINISHED'}

class KETTE_OT_export_3d_print(Operator):
    bl_idname = "kette.export_3d_print"
    bl_label = "3D-Druck Export"
    bl_options = {'REGISTER'}
    bl_description = "Exportiert die Kette für 3D-Druck"
    
    filepath: StringProperty(
        name="File Path",
        description="Path to export STL",
        default="chain.stl",
        subtype='FILE_PATH'
    )
    
    filename_ext = ".stl"
    
    filter_glob: StringProperty(
        default="*.stl",
        options={'HIDDEN'}
    )
    
    apply_modifiers: BoolProperty(
        name="Modifier anwenden",
        default=True,
        description="Wendet alle Modifier vor Export an"
    )
    
    scale_factor: FloatProperty(
        name="Skalierungsfaktor",
        default=10.0,
        min=0.1,
        max=1000.0,
        description="Skalierung für Export (10 = cm zu mm)"
    )
    
    def execute(self, context):
        # Finde finales Objekt
        final_obj = None
        coll = bpy.data.collections.get("KonstruktionsKette")
        
        if coll:
            for obj in coll.objects:
                if obj.name == "Kette_Final":
                    final_obj = obj
                    break
        
        if not final_obj:
            # Versuche aktives Objekt
            if context.active_object and context.active_object.type == 'MESH':
                final_obj = context.active_object
            else:
                self.report({'ERROR'}, "Kein finales Mesh gefunden! Erst glätten & verschmelzen.")
                return {'CANCELLED'}
        
        # Erstelle temporäre Kopie für Export
        export_obj = final_obj.copy()
        export_obj.data = final_obj.data.copy()
        context.collection.objects.link(export_obj)
        
        # Skalieren
        export_obj.scale *= self.scale_factor
        
        # Modifier anwenden wenn gewünscht
        if self.apply_modifiers:
            context.view_layer.objects.active = export_obj
            for mod in export_obj.modifiers:
                try:
                    bpy.ops.object.modifier_apply(modifier=mod.name)
                except:
                    pass
        
        # Exportieren
        bpy.ops.object.select_all(action='DESELECT')
        export_obj.select_set(True)
        
        bpy.ops.export_mesh.stl(
            filepath=self.filepath,
            use_selection=True,
            global_scale=1.0,
            use_mesh_modifiers=False  # Already applied
        )
        
        # Cleanup
        bpy.data.objects.remove(export_obj, do_unlink=True)
        
        self.report({'INFO'}, f"Exportiert nach {self.filepath}")
        return {'FINISHED'}
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

# =========================================
# LIBRARY MANAGEMENT OPERATORS
# =========================================

class KETTE_OT_set_library_path(Operator):
    bl_idname = "kette.set_library_path"
    bl_label = "Bibliothekspfad setzen"
    bl_options = {'REGISTER'}
    
    directory: StringProperty(
        name="Directory",
        subtype='DIR_PATH'
    )
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def execute(self, context):
        props = context.scene.chain_construction_props
        props.library_path_override = self.directory
        ensure_library_directory()
        bpy.ops.kette.refresh_library()
        self.report({'INFO'}, f"Bibliothekspfad gesetzt: {self.directory}")
        return {'FINISHED'}

class KETTE_OT_refresh_library(Operator):
    bl_idname = "kette.refresh_library"
    bl_label = "Bibliothek aktualisieren"
    bl_options = {'REGISTER'}

    def execute(self, context):
        props = context.scene.chain_construction_props
        props.pattern_library.clear()
        all_patterns = load_pattern_library()
        for pid, pdata in all_patterns.items():
            item = props.pattern_library.add()
            item.pattern_id = pid
            item.pattern_name = pdata.get("name", pid)
            item.pattern_category = pdata.get("category", "")
            item.pattern_data = json.dumps(pdata)
        self.report({'INFO'}, f"{len(props.pattern_library)} Patterns geladen")
        return {'FINISHED'}

class KETTE_OT_load_pattern_from_library(Operator):
    bl_idname = "kette.load_pattern_from_library"
    bl_label = "Pattern laden"
    bl_description = "Lädt gewähltes Pattern aus Bibliothek"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.chain_construction_props
        idx = props.pattern_library_index
        if idx < 0 or idx >= len(props.pattern_library):
            self.report({'ERROR'}, "Kein Pattern ausgewählt")
            return {'CANCELLED'}
        
        item = props.pattern_library[idx]
        try:
            pdata = json.loads(item.pattern_data)
        except Exception as e:
            self.report({'ERROR'}, f"Fehler beim Parsen: {e}")
            return {'CANCELLED'}
        
        params = pdata.get("parameters", {})
        for key, val in params.items():
            if hasattr(props, key):
                try:
                    setattr(props, key, val)
                except Exception as e:
                    debug.warning('GENERAL', f"Could not set {key}: {e}")
        
        self.report({'INFO'}, f"Pattern '{item.pattern_name}' geladen")
        return {'FINISHED'}

class KETTE_OT_load_pattern_direct(Operator):
    bl_idname = "kette.load_pattern_direct"
    bl_label = "Pattern Laden"
    bl_description = "Lädt das spezifizierte Pattern"
    bl_options = {'REGISTER', 'UNDO'}
    
    pattern_index: IntProperty()
    
    def execute(self, context):
        props = context.scene.chain_construction_props
        
        if self.pattern_index < 0 or self.pattern_index >= len(props.pattern_library):
            self.report({'ERROR'}, "Ungültiger Pattern Index")
            return {'CANCELLED'}
            
        # Set index and load pattern
        props.pattern_library_index = self.pattern_index
        bpy.ops.kette.load_pattern_from_library()
        
        return {'FINISHED'}

class KETTE_OT_export_pattern(Operator):
    bl_idname = "kette.export_pattern"
    bl_label = "Muster Exportieren"
    bl_options = {'REGISTER'}
    bl_description = "Exportiert Muster als JSON"
    
    filepath: StringProperty(
        name="File Path",
        description="Path to save pattern",
        default="pattern.json",
        subtype='FILE_PATH'
    )
    
    filename_ext = ".json"
    
    filter_glob: StringProperty(
        default="*.json",
        options={'HIDDEN'}
    )
    
    export_mode: EnumProperty(
        name="Export Modus",
        items=[
            ('CURRENT', 'Aktuell', 'Exportiere aktuelle Einstellungen'),
            ('ALL', 'Alle', 'Exportiere alle Patterns aus Library'),
        ],
        default='CURRENT'
    )
    
    def execute(self, context):
        props = context.scene.chain_construction_props
        
        if self.export_mode == 'CURRENT':
            pattern_data = {
                "name": props.hop_pattern_name,
                "category": "custom",
                "parameters": {}
            }
            
            # Collect all relevant properties
            for prop_name in dir(props):
                if (prop_name.startswith("hop_") or prop_name.startswith("kette_") or 
                    prop_name in ["auto_connect", "pattern_square_diagonal"]):
                    try:
                        value = getattr(props, prop_name)
                        if not hasattr(value, "name"):  # Skip object references
                            pattern_data["parameters"][prop_name] = value
                    except:
                        pass
            
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(pattern_data, f, indent=2)
                
        else:  # Export all
            all_patterns = load_pattern_library()
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(all_patterns, f, indent=2)
        
        self.report({'INFO'}, f"Pattern exportiert nach {self.filepath}")
        return {'FINISHED'}
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

class KETTE_OT_import_pattern(Operator):
    bl_idname = "kette.import_pattern"
    bl_label = "Muster Importieren"
    bl_options = {'REGISTER'}
    bl_description = "Importiert Muster aus JSON"
    
    filepath: StringProperty(
        name="File Path",
        description="Path to load pattern",
        default="",
        subtype='FILE_PATH'
    )
    
    filename_ext = ".json"
    
    filter_glob: StringProperty(
        default="*.json",
        options={'HIDDEN'}
    )
    
    def execute(self, context):
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if single pattern or multiple
            if isinstance(data, dict) and "parameters" in data:
                # Single pattern
                pattern_id = pathlib.Path(self.filepath).stem
                save_pattern_to_library(data, pattern_id)
                self.report({'INFO'}, "Pattern importiert")
            else:
                # Multiple patterns
                count = 0
                for pid, pdata in data.items():
                    save_pattern_to_library(pdata, pid)
                    count += 1
                self.report({'INFO'}, f"{count} Patterns importiert")
            
            # Refresh library
            bpy.ops.kette.refresh_library()
            
        except Exception as e:
            self.report({'ERROR'}, f"Import fehlgeschlagen: {e}")
            return {'CANCELLED'}
        
        return {'FINISHED'}
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

# =========================================
# UI PANELS
# =========================================

class VIEW3D_PT_chain_construction(Panel):
    bl_label = "Chain Construction"
    bl_idname = "VIEW3D_PT_chain_construction"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tools"
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.chain_construction_props
        
        # Target object
        box = layout.box()
        box.label(text="Zielobjekt", icon='MESH_DATA')
        box.prop(props, "kette_target_obj", text="")
        
        if not props.kette_target_obj:
            box.label(text="Wähle ein Mesh-Objekt!", icon='ERROR')

class VIEW3D_PT_chain_basic_settings(Panel):
    bl_label = "Basis Einstellungen"
    bl_idname = "VIEW3D_PT_chain_basic_settings"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tools"
    bl_parent_id = "VIEW3D_PT_chain_construction"
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.chain_construction_props
        
        col = layout.column(align=True)
        col.prop(props, "kette_kugelradius")
        col.prop(props, "kette_aufloesung")
        col.prop(props, "kette_abstand")
        col.prop(props, "kette_global_offset")
        col.prop(props, "kette_aura_dichte")
        
        # Connector settings
        col.separator()
        col.prop(props, "kette_use_curved_connectors")
        if props.kette_use_curved_connectors:
            col.prop(props, "kette_connector_segs")
        col.prop(props, "kette_connector_radius_factor")

class VIEW3D_PT_chain_hopping(Panel):
    bl_label = "Hopping Pattern Generator"
    bl_idname = "VIEW3D_PT_chain_hopping"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tools"
    bl_parent_id = "VIEW3D_PT_chain_construction"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.chain_construction_props
        
        # Main controls
        col = layout.column()
        
        # Pattern type
        col.prop(props, "hop_pattern", text="Muster")
        
        # Basic parameters
        col.separator()
        col.prop(props, "hop_distance")
        col.prop(props, "hop_iterations")
        col.prop(props, "hop_max_spheres")
        
        # Pattern-specific settings
        if props.hop_pattern == 'SQUARE':
            col.prop(props, "pattern_square_diagonal")
        elif props.hop_pattern == 'ORGANIC':
            row = col.row(align=True)
            row.prop(props, "organic_min_neighbors", text="Min")
            row.prop(props, "organic_max_neighbors", text="Max")
        elif props.hop_pattern == 'TRANSITION':
            col.prop(props, "hop_pattern_start")
            col.prop(props, "hop_pattern_end")
        
        # Advanced settings
        col.separator()
        box = col.box()
        row = box.row()
        row.prop(props, "show_hop_advanced", 
                icon='TRIA_DOWN' if props.show_hop_advanced else 'TRIA_RIGHT',
                text="Erweiterte Einstellungen", emboss=False)
        
        if props.show_hop_advanced:
            adv_col = box.column(align=True)
            adv_col.prop(props, "hop_angle_variation")
            adv_col.prop(props, "hop_distance_variation")
            adv_col.prop(props, "hop_probability")
            adv_col.prop(props, "hop_branch_probability")
            adv_col.prop(props, "hop_generation_scale")
            
            adv_col.separator()
            adv_col.prop(props, "hop_avoid_overlap")
            if props.hop_avoid_overlap:
                adv_col.prop(props, "hop_overlap_factor")
            
            adv_col.separator()
            adv_col.prop(props, "hop_respect_existing")
            adv_col.prop(props, "auto_connect")
            if props.auto_connect:
                adv_col.prop(props, "hop_fill_missing_connections")
                adv_col.prop(props, "hop_min_connection_angle")
        
        # Presets
        col.separator()
        row = col.row(align=True)
        row.operator("kette.apply_preset", text="Preset", icon='PRESET')
        row.operator("kette.save_hopping_pattern", text="", icon='FILE_TICK')
        
        # Generate button
        col.separator()
        row = col.row()
        row.scale_y = 2.0
        row.operator("kette.generate_hopping_pattern", 
                    text="HOPPING GENERIEREN", 
                    icon='PARTICLE_DATA')
        
        # Clear button
        col.operator("kette.clear_pattern", text="Muster Löschen", icon='TRASH')

class VIEW3D_PT_chain_paint_settings(Panel):
    bl_label = "Paint Modus"
    bl_idname = "VIEW3D_PT_chain_paint_settings"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tools"
    bl_parent_id = "VIEW3D_PT_chain_construction"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.chain_construction_props
        
        # Check if target is set
        if not props.kette_target_obj:
            layout.label(text="Zielobjekt benötigt!", icon='ERROR')
            return
        
        # Paint button
        row = layout.row()
        row.scale_y = 1.5
        row.operator("kette.paint_mode", text="PAINT MODUS", icon='BRUSH_DATA')
        
        # Settings
        col = layout.column(align=True)
        col.prop(props, "paint_brush_spacing", text="Pinsel-Abstand")
        col.prop(props, "paint_auto_connect", text="Auto-Verbinden")
        
        col.separator()
        col.prop(props, "paint_use_symmetry", text="Symmetrie")
        if props.paint_use_symmetry:
            col.prop(props, "paint_symmetry_axis", text="Achse")
        
        # Painted spheres info
        layout.separator()
        coll = bpy.data.collections.get("KonstruktionsKette")
        painted_count = 0
        if coll:
            painted_count = len([o for o in coll.objects 
                               if o.name.startswith("Kugel") and o.get("painted")])
        
        if painted_count > 0:
            box = layout.box()
            box.label(text=f"Gemalte Kugeln: {painted_count}", icon='INFO')
            col = box.column(align=True)
            col.operator("kette.connect_painted_spheres", icon='LINKED')
            col.operator("kette.clear_painted_spheres", icon='TRASH')

class VIEW3D_PT_chain_advanced_tools(Panel):
    bl_label = "Erweiterte Werkzeuge"
    bl_idname = "VIEW3D_PT_chain_advanced_tools"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tools"
    bl_parent_id = "VIEW3D_PT_chain_construction"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.chain_construction_props
        
        # Basic tools
        col = layout.column(align=True)
        col.label(text="Basis Werkzeuge:", icon='TOOL_SETTINGS')
        col.operator("kette.kugel_platzieren", icon='MESH_UVSPHERE')
        col.operator("kette.snap_surface", icon='SNAP_ON')
        
        col.separator()
        row = col.row(align=True)
        row.operator("kette.line_connector", text="Gerade", icon='IPO_LINEAR')
        row.operator("kette.curved_connector", text="Gebogen", icon='IPO_BEZIER')
        
        # Connection tools
        col.separator()
        col.label(text="Verbindungen:", icon='LINKED')
        col.prop(props, "kette_conn_mode", text="")
        
        if props.kette_conn_mode == 'NEIGHBOR':
            col.prop(props, "kette_max_abstand")
            col.prop(props, "kette_max_links")
        
        col.operator("kette.automatisch_verbinden", icon='LINKED')
        col.operator("kette.kette_update", icon='FILE_REFRESH')
        
        # Growth tools
        col.separator()
        col.label(text="Gerichtetes Wachstum:", icon='ORIENTATION_CURSOR')
        col.prop(props, "kette_growth_dir", text="")
        
        if props.kette_growth_dir == 'ARROW':
            col.prop_search(props, "kette_richtungs_obj", bpy.data, "objects", text="Pfeil")
            col.operator("kette.richtungspfeil_erstellen", icon='EMPTY_SINGLE_ARROW')
        
        col.prop(props, "kette_chain_abstand")
        col.prop(props, "kette_chain_anzahl")
        col.prop(props, "kette_sequentiell")
        
        row = col.row(align=True)
        row.operator("kette.kette_wachstum", text="Linear", icon='PARTICLE_DATA')
        row.operator("kette.kreis_kette", text="Kreis", icon='MESH_CIRCLE')
        
        # Finalization
        col.separator()
        col.label(text="Finalisierung:", icon='CHECKMARK')
        col.operator("kette.kette_glaetten", icon='MOD_SMOOTH')
        col.operator("kette.export_3d_print", icon='EXPORT')
        
        # Library
        col.separator()
        col.label(text="Pattern Bibliothek:", icon='ASSET_MANAGER')
        
        row = col.row(align=True)
        row.operator("kette.set_library_path", text="Pfad", icon='FOLDER_REDIRECT')
        row.operator("kette.refresh_library", text="", icon='FILE_REFRESH')
        
        # Pattern list
        if props.pattern_library:
            col.template_list("UI_UL_list", "patterns", 
                            props, "pattern_library", 
                            props, "pattern_library_index", 
                            rows=3)
            
            if props.pattern_library_index >= 0:
                col.operator("kette.load_pattern_from_library", icon='IMPORT')
        
        # Import/Export
        row = col.row(align=True)
        row.operator("kette.export_pattern", text="Export", icon='EXPORT')
        row.operator("kette.import_pattern", text="Import", icon='IMPORT')

# =========================================
# REGISTRATION
# =========================================

classes = (
    # Properties
    PatternLibraryItem,
    ChainConstructionProperties,
    ChainConstructionPreferences,
    
    # Pattern operators - Hopping
    KETTE_OT_generate_hopping_pattern,
    KETTE_OT_apply_preset,
    KETTE_OT_save_hopping_pattern,
    KETTE_OT_clear_pattern,
    
    # Basic tools
    KETTE_OT_kugel_platzieren,
    KETTE_OT_snap_surface,
    KETTE_OT_line_connector,
    KETTE_OT_curved_connector,
    KETTE_OT_kette_update,
    KETTE_OT_automatisch_verbinden,
    
    # Paint operators
    KETTE_OT_paint_mode,
    KETTE_OT_connect_painted_spheres,
    KETTE_OT_clear_painted_spheres,
    
    # Growth tools
    KETTE_OT_richtungspfeil_erstellen,
    KETTE_OT_kette_wachstum,
    KETTE_OT_kreis_kette,
    
    # Finalization
    KETTE_OT_kette_glaetten,
    KETTE_OT_export_3d_print,
    
    # Library management
    KETTE_OT_set_library_path,
    KETTE_OT_refresh_library,
    KETTE_OT_load_pattern_from_library,
    KETTE_OT_load_pattern_direct,
    KETTE_OT_export_pattern,
    KETTE_OT_import_pattern,
    
    # UI Panels
    VIEW3D_PT_chain_construction,
    VIEW3D_PT_chain_basic_settings,
    VIEW3D_PT_chain_hopping,
    VIEW3D_PT_chain_paint_settings,
    VIEW3D_PT_chain_advanced_tools,
)

def register():
    global DEBUG_MODE
    
    # Register classes
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except ValueError:
            # Already registered
            try:
                bpy.utils.unregister_class(cls)
                bpy.utils.register_class(cls)
            except:
                pass
    
    # Scene property
    if not hasattr(bpy.types.Scene, "chain_construction_props"):
        bpy.types.Scene.chain_construction_props = PointerProperty(type=ChainConstructionProperties)
    
    # Add scene property for library path when running as script
    bpy.types.Scene.chain_library_path = StringProperty(
        name="Chain Library Path",
        default="",
        subtype='DIR_PATH'
    )
    
    # Check addon preferences
    try:
        if __name__ != "__main__":
            prefs = bpy.context.preferences.addons[__name__].preferences
            DEBUG_MODE = prefs.enable_debug
            debug.enabled = DEBUG_MODE
    except:
        pass
    
    # Initialize library
    try:
        lib_path = ensure_library_directory()
        
        # Check if library is empty
        patterns = load_pattern_library()
        if not patterns:
            create_example_patterns()
        
    except Exception as e:
        print(f"Warning: Could not initialize library: {e}")
    
    print("Chain Construction Advanced v3.6 registered successfully!")
    print("Find it in the sidebar under 'Chain Tools'")

def unregister():
    # Clean up timers
    try:
        wm = bpy.context.window_manager
        for timer in list(getattr(wm, 'timers', [])):
            if hasattr(timer, '__name__') and 'chain' in getattr(timer, '__name__', ''):
                wm.event_timer_remove(timer)
    except:
        pass
    
    # Clear caches
    clear_auras()
    _existing_connectors.clear()
    _remembered_pairs.clear()
    
    # Unregister classes
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except:
            pass
    
    # Remove properties
    if hasattr(bpy.types.Scene, "chain_construction_props"):
        del bpy.types.Scene.chain_construction_props
    
    if hasattr(bpy.types.Scene, "chain_library_path"):
        del bpy.types.Scene.chain_library_path
    
    print("Chain Construction Advanced unregistered")

if __name__ == "__main__":
    register()
    
    print("\n" + "="*60)
    print("Chain Construction Advanced v3.6 - COMPLETE VERSION")
    print("="*60)
    print("\nFeatures:")
    print("- Optimized Hopping Pattern System")
    print("- Paint Mode for Interactive Design")
    print("- Full Library Management")
    print("- 3D Print Export")
    print("- Directed Growth Tools")
    print("- Batch Operations")
    print("- Performance Monitoring & Debug System")
    print("\nUsage:")
    print("1. Select a target mesh object")
    print("2. Choose between Hopping Generator or Paint Mode")
    print("3. Configure parameters or load preset")
    print("4. Generate pattern and auto-connect")
    print("5. Finalize with smoothing and export")
    print("\nLibrary: Documents/chain_patterns/")
    print("="*60 + "\n")
