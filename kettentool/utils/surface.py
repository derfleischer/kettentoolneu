"""
Surface Projection und Aura System
"""

import bpy
import bmesh
import mathutils
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from typing import Tuple, Optional

# =========================================
# AURA SYSTEM
# =========================================

def get_aura(target_obj: bpy.types.Object, thickness: float) -> bpy.types.Object:
    """Erstellt oder holt Aura für Oberflächenprojektion"""
    from ..core import constants
    
    if not target_obj or target_obj.name not in bpy.data.objects:
        clear_auras()
        raise ReferenceError("Target object has been removed")

    # Cache-Key mit Objekt und Dicke
    cache_key = f"{target_obj.name}_{thickness:.4f}"
    
    # Prüfe Cache
    if cache_key in constants._aura_cache:
        aura = constants._aura_cache[cache_key]
        try:
            if aura.name in bpy.data.objects:
                return aura
        except ReferenceError:
            pass
        constants._aura_cache.pop(cache_key, None)

    # Lösche alte Auras mit anderer Dicke
    to_remove = []
    for key in list(constants._aura_cache.keys()):
        if key.startswith(f"{target_obj.name}_") and key != cache_key:
            to_remove.append(key)
    
    for key in to_remove:
        old_aura = constants._aura_cache.pop(key, None)
        if old_aura:
            try:
                bpy.data.objects.remove(old_aura, do_unlink=True)
            except:
                pass

    # Erstelle neue Aura
    if constants.debug:
        constants.debug.trace('SURFACE', f"Creating aura for {target_obj.name}, thickness: {thickness}")
    
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

    constants._aura_cache[cache_key] = aura
    return aura

def _build_bvh(obj: bpy.types.Object) -> BVHTree:
    """Erstellt oder holt BVH-Tree für Ray-Casting"""
    from ..core import constants
    
    if obj.name in constants._bvh_cache:
        cached_bvh, cached_matrix, cached_id = constants._bvh_cache[obj.name]
        if cached_id == id(obj.data) and cached_matrix == obj.matrix_world:
            if constants.debug:
                constants.debug.trace('SURFACE', f"Using cached BVH for {obj.name}")
            return cached_bvh

    if constants.debug:
        constants.debug.trace('SURFACE', f"Building new BVH for {obj.name}")
    
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.transform(obj.matrix_world)
    bvh = BVHTree.FromBMesh(bm)
    bm.free()
    constants._bvh_cache[obj.name] = (bvh, obj.matrix_world.copy(), id(obj.data))
    return bvh

def next_surface_point(aura_obj: bpy.types.Object, pos: Vector) -> Tuple[Vector, Vector]:
    """Findet nächsten Oberflächenpunkt"""
    from ..core import constants
    
    if constants.performance_monitor:
        constants.performance_monitor.start_timer("next_surface_point")
    
    if not aura_obj or aura_obj.type != 'MESH':
        if constants.performance_monitor:
            constants.performance_monitor.end_timer("next_surface_point")
        return pos, Vector((0, 0, 1))

    bvh = _build_bvh(aura_obj)
    hit = bvh.find_nearest(pos)
    if not hit:
        if constants.debug:
            constants.debug.trace('SURFACE', f"No hit for position {pos}")
        if constants.performance_monitor:
            constants.performance_monitor.end_timer("next_surface_point")
        return pos, Vector((0, 0, 1))

    loc, norm, *_ = hit
    norm = norm.normalized() if norm.length > 1e-6 else Vector((0, 0, 1))

    # Orient normal outward
    origin = aura_obj.matrix_world.translation
    if (loc - origin).dot(norm) < 0:
        norm = -norm

    if constants.performance_monitor:
        constants.performance_monitor.end_timer("next_surface_point")
    
    return loc, norm

def clear_auras():
    """Löscht alle gecachten Auras"""
    from ..core import constants
    
    if constants.debug:
        constants.debug.info('SURFACE', "Clearing all auras")
    
    auras_to_remove = []

    for cache_key, aura in list(constants._aura_cache.items()):
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

    constants._aura_cache.clear()
    constants._bvh_cache.clear()

# =========================================
# PAINT UTILITIES
# =========================================

def get_paint_surface_position(context, event, target_obj: bpy.types.Object) -> Optional[Vector]:
    """Ermittelt 3D-Position für Paint Mode"""
    if not target_obj:
        return None
    
    # Ray-Cast von Maus-Position
    scene = context.scene
    region = context.region
    rv3d = context.region_data
    
    coord = event.mouse_region_x, event.mouse_region_y
    view_vector = mathutils.geometry.region_2d_to_vector_3d(region, rv3d, coord)
    ray_origin = mathutils.geometry.region_2d_to_origin_3d(region, rv3d, coord)
    
    # Ray-Cast gegen Target-Objekt
    matrix_inv = target_obj.matrix_world.inverted()
    ray_origin_local = matrix_inv @ ray_origin
    ray_direction_local = matrix_inv.to_3x3() @ view_vector
    
    # BVH für Hit-Detection
    bvh = _build_bvh(target_obj)
    hit_loc, hit_norm, hit_idx, hit_dist = bvh.ray_cast(ray_origin_local, ray_direction_local)
    
    if hit_loc:
        return target_obj.matrix_world @ hit_loc
    
    return None

# =========================================
# REGISTRATION
# =========================================

def register():
    """Registriert Surface-Modul"""
    pass

def unregister():
    """Deregistriert Surface-Modul"""
    clear_auras()
