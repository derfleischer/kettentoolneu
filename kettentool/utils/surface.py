"""
Surface Projection and Aura System - HYBRID BVH/AURA METHOD
Kombiniert BVH für Performance mit Aura für bessere Surface-Projektion
"""

import bpy
import bmesh
import time
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from typing import Optional, Tuple, Any
import kettentool.core.constants as constants

# =========================================
# AURA SYSTEM (für Surface-Following)
# =========================================

class Aura:
    """Enhanced Aura wrapper für Surface-Projektion"""
    
    def __init__(self, obj: bpy.types.Object, thickness: float = 0.05):
        self.obj = obj
        self.thickness = thickness
        self.aura_obj = None
        self.bvh = None
        self._create_aura()
    
    def _create_aura(self):
        """Erstellt aufgeblähte Version des Objekts"""
        if constants.debug:
            constants.debug.trace('SURFACE', f"Creating aura for {self.obj.name}, thickness: {self.thickness}")
        
        # Kopiere Objekt
        self.aura_obj = self.obj.copy()
        self.aura_obj.data = self.obj.data.copy()
        self.aura_obj.name = f"{self.obj.name}_Aura_{self.thickness:.4f}"
        
        # Link to scene first (required before setting properties)
        bpy.context.collection.objects.link(self.aura_obj)
        
        # Now hide the Aura (after it's linked)
        self.aura_obj.hide_viewport = True
        self.aura_obj.hide_render = True
        self.aura_obj.hide_select = True
        
        # Normale korrigieren vor Solidify
        bm = bmesh.new()
        bm.from_mesh(self.aura_obj.data)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(self.aura_obj.data)
        bm.free()
        
        # Solidify Modifier für Aura-Dicke
        mod = self.aura_obj.modifiers.new("AuraSolidify", 'SOLIDIFY')
        mod.thickness = self.thickness
        mod.offset = 1.0  # Nur nach außen expandieren
        mod.use_even_offset = True
        mod.use_quality_normals = True
        
        # Apply modifier (need proper context)
        old_active = bpy.context.view_layer.objects.active
        bpy.context.view_layer.objects.active = self.aura_obj
        bpy.ops.object.select_all(action='DESELECT')
        self.aura_obj.select_set(True)
        try:
            bpy.ops.object.modifier_apply(modifier=mod.name)
            bpy.context.view_layer.objects.active = old_active
        except Exception as e:
            if constants.debug:
                constants.debug.warning('SURFACE', f"Solidify failed for {self.obj.name}: {e}")
            # Remove modifier if apply failed
            if mod.name in self.aura_obj.modifiers:
                self.aura_obj.modifiers.remove(mod)
        
        # Now hide the aura object
        self.aura_obj.hide_viewport = True
        self.aura_obj.hide_render = True
        self.aura_obj.hide_select = True
        
        # Normale nach Solidify korrigieren
        bm2 = bmesh.new()
        bm2.from_mesh(self.aura_obj.data)
        
        # Korrigiere Normale für konkave Bereiche
        mesh_center = sum((v.co for v in bm2.verts), Vector()) / len(bm2.verts)
        for face in bm2.faces:
            face_center = face.calc_center_median()
            outward_dir = (face_center - mesh_center).normalized()
            if face.normal.dot(outward_dir) < 0:
                face.normal_flip()
        
        bmesh.ops.recalc_face_normals(bm2, faces=bm2.faces)
        bm2.to_mesh(self.aura_obj.data)
        bm2.free()
        
        # Build BVH for aura
        self._build_bvh()
    
    def _build_bvh(self):
        """Build BVH for aura object"""
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = self.aura_obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.transform(self.aura_obj.matrix_world)
        
        self.bvh = BVHTree.FromBMesh(bm)
        
        bm.free()
        eval_obj.to_mesh_clear()
    
    def find_nearest(self, location: Vector) -> Tuple[Vector, Vector, float]:
        """Find nearest surface point on aura"""
        if not self.bvh:
            return location, Vector((0, 0, 1)), 0.0
        
        result = self.bvh.find_nearest(location)
        if result[0] is None:
            return location, Vector((0, 0, 1)), 0.0
        
        location, normal, index, distance = result
        
        # Normalize normal
        if normal.length > 0:
            normal = normal.normalized()
        else:
            normal = Vector((0, 0, 1))
        
        # Ensure outward normal
        obj_center = self.obj.location
        to_point = (location - obj_center).normalized()
        if normal.dot(to_point) < 0:
            normal = -normal
        
        return Vector(location), normal, distance
    
    def cleanup(self):
        """Remove aura object"""
        if self.aura_obj and self.aura_obj.name in bpy.data.objects:
            bpy.data.objects.remove(self.aura_obj, do_unlink=True)

# =========================================
# AURA CACHE MANAGEMENT
# =========================================

def get_aura(obj: bpy.types.Object, thickness: float = 0.05, force_rebuild: bool = False) -> Aura:
    """Get or create aura for object with specified thickness"""
    aura_cache = constants.get_aura_cache()
    
    # Cache key includes thickness
    cache_key = f"{obj.name}_aura_{thickness:.4f}"
    
    # Clear old auras with different thickness
    if force_rebuild or cache_key not in aura_cache:
        # Remove old auras for this object
        keys_to_remove = []
        for key in aura_cache.keys():
            if key.startswith(f"{obj.name}_aura_") and key != cache_key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            old_aura = aura_cache.pop(key, None)
            if old_aura and hasattr(old_aura, 'cleanup'):
                old_aura.cleanup()
    
    if force_rebuild and cache_key in aura_cache:
        old_aura = aura_cache.pop(cache_key)
        if hasattr(old_aura, 'cleanup'):
            old_aura.cleanup()
    
    if cache_key in aura_cache:
        return aura_cache[cache_key]
    
    # Create new aura
    if constants.debug:
        constants.debug.info('SURFACE', f"Creating new aura for {obj.name} (thickness: {thickness})")
    
    aura = Aura(obj, thickness)
    aura_cache[cache_key] = aura
    
    return aura

# =========================================
# BVH CACHE (für direkte Projektion)
# =========================================

def _build_bvh(obj: bpy.types.Object) -> Optional[BVHTree]:
    """Build or get cached BVH tree for object"""
    cache_key = f"{obj.name}_bvh"
    cache = constants.get_aura_cache()
    
    if cache_key in cache:
        cached_data = cache[cache_key]
        if cached_data.get('mesh_id') == id(obj.data):
            if constants.debug:
                constants.debug.trace('SURFACE', f"Using cached BVH for {obj.name}")
            return cached_data['bvh']
    
    if constants.debug:
        constants.debug.trace('SURFACE', f"Building new BVH for {obj.name}")
    
    start_time = time.time()
    
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.transform(obj.matrix_world)
    
    bvh = BVHTree.FromBMesh(bm)
    
    bm.free()
    eval_obj.to_mesh_clear()
    
    cache[cache_key] = {
        'bvh': bvh,
        'mesh_id': id(obj.data),
        'matrix': obj.matrix_world.copy()
    }
    
    if constants.debug:
        elapsed = time.time() - start_time
        constants.debug.info('SURFACE', f"Built BVH for {obj.name} in {elapsed:.3f}s")
    
    return bvh

# =========================================
# SURFACE PROJECTION FUNCTIONS
# =========================================

def next_surface_point(obj: Any, pos: Vector) -> Tuple[Vector, Vector]:
    """
    Find nearest surface point - works with both Aura and regular objects
    """
    # Handle Aura objects
    if hasattr(obj, 'find_nearest'):
        location, normal, _ = obj.find_nearest(pos)
        return location, normal
    
    # Handle regular objects with BVH
    if not obj or obj.type != 'MESH':
        return pos, Vector((0, 0, 1))
    
    bvh = _build_bvh(obj)
    if not bvh:
        return pos, Vector((0, 0, 1))
    
    location, normal, index, distance = bvh.find_nearest(pos)
    
    if location is None:
        if constants.debug:
            constants.debug.trace('SURFACE', f"No hit for position {pos}")
        return pos, Vector((0, 0, 1))
    
    if normal.length > 0:
        normal = normal.normalized()
    else:
        normal = Vector((0, 0, 1))
    
    obj_center = obj.location
    to_point = (location - obj_center).normalized()
    if normal.dot(to_point) < 0:
        normal = -normal
    
    return Vector(location), normal

def project_to_surface(obj: bpy.types.Object, location: Vector, 
                       offset: float = 0.0, use_aura: bool = True,
                       aura_thickness: float = 0.05) -> Tuple[Vector, Vector]:
    """
    Project location to object surface
    
    Args:
        obj: Target object
        location: Location to project
        offset: Additional offset from surface
        use_aura: Use aura for better surface following
        aura_thickness: Thickness of aura
    """
    if use_aura:
        # Use aura for better surface projection
        aura = get_aura(obj, aura_thickness)
        surface_loc, normal, _ = aura.find_nearest(location)
    else:
        # Direct BVH projection
        surface_loc, normal = next_surface_point(obj, location)
    
    # Apply offset
    if offset > 0:
        surface_loc += normal * offset
    
    return surface_loc, normal

def snap_to_surface(sphere: bpy.types.Object, target: bpy.types.Object, 
                   offset: float = 0.0, use_aura: bool = True):
    """Snap sphere to target surface"""
    props = bpy.context.scene.chain_construction_props
    aura_thickness = 0.05
    
    surface_loc, normal = project_to_surface(
        target, sphere.location, offset, 
        use_aura=use_aura, aura_thickness=aura_thickness
    )
    sphere.location = surface_loc
    
    if constants.debug:
        constants.debug.debug('SURFACE', f"Snapped {sphere.name} to surface")

# =========================================
# CACHE MANAGEMENT
# =========================================

def clear_auras():
    """Clear all aura and BVH caches"""
    aura_cache = constants.get_aura_cache()
    
    # Cleanup aura objects
    for key, value in list(aura_cache.items()):
        if '_aura_' in key and hasattr(value, 'cleanup'):
            value.cleanup()
    
    # Clear cache entries
    keys_to_remove = []
    for key in aura_cache.keys():
        if '_bvh' in key or '_aura_' in key:
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        del aura_cache[key]
    
    if constants.debug:
        constants.debug.info('SURFACE', f"Cleared {len(keys_to_remove)} surface caches")

def invalidate_aura(obj_name: str):
    """Invalidate aura cache for specific object"""
    aura_cache = constants.get_aura_cache()
    keys_to_remove = []
    
    for key in aura_cache.keys():
        if key.startswith(obj_name):
            keys_to_remove.append(key)
            # Cleanup aura objects
            if '_aura_' in key:
                aura = aura_cache.get(key)
                if hasattr(aura, 'cleanup'):
                    aura.cleanup()
    
    for key in keys_to_remove:
        del aura_cache[key]
    
    if keys_to_remove and constants.debug:
        constants.debug.info('SURFACE', f"Invalidated {len(keys_to_remove)} caches for {obj_name}")

# =========================================
# UTILITIES
# =========================================

def calculate_surface_offset(props: Any) -> float:
    """Calculate total surface offset from properties"""
    offset = props.kette_kugelradius
    offset += props.kette_abstand
    offset += props.kette_global_offset
    offset += constants.EPSILON
    return offset

def is_on_surface(location: Vector, target: bpy.types.Object, 
                  tolerance: float = 0.01, use_aura: bool = True) -> bool:
    """Check if location is on surface within tolerance"""
    if use_aura:
        aura = get_aura(target, 0.05)
        _, _, distance = aura.find_nearest(location)
    else:
        bvh = _build_bvh(target)
        if not bvh:
            return False
        _, _, _, distance = bvh.find_nearest(location)
    
    return distance <= tolerance

def get_surface_normal(target: bpy.types.Object, location: Vector,
                       use_aura: bool = True) -> Vector:
    """Get surface normal at location"""
    if use_aura:
        aura = get_aura(target, 0.05)
        _, normal, _ = aura.find_nearest(location)
    else:
        _, normal = next_surface_point(target, location)
    
    return normal

# =========================================
# RAYCASTING (for Paint Mode)
# =========================================

def raycast_to_surface(obj: bpy.types.Object, ray_origin: Vector, 
                       ray_direction: Vector, use_aura: bool = False) -> Optional[Tuple[Vector, Vector]]:
    """Raycast to surface for paint mode"""
    if use_aura:
        aura = get_aura(obj, 0.05)
        if not aura.bvh:
            return None
        bvh = aura.bvh
    else:
        bvh = _build_bvh(obj)
        if not bvh:
            return None
    
    location, normal, index, distance = bvh.ray_cast(ray_origin, ray_direction)
    
    if location is None:
        return None
    
    if normal.length > 0:
        normal = normal.normalized()
    else:
        normal = Vector((0, 0, 1))
    
    return Vector(location), normal

# =========================================
# REGISTRATION
# =========================================

def register():
    """Register surface module"""
    pass

def unregister():
    """Unregister surface module"""
    clear_auras()
