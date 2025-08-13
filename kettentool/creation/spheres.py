"""
Sphere Creation and Management
"""

import bpy
from mathutils import Vector
from typing import Optional, List, Tuple
import kettentool.core.constants as constants
import kettentool.utils.collections as collections
import kettentool.utils.materials as materials

# =========================================
# SPHERE CREATION
# =========================================

def create_sphere(location: Vector, 
                 radius: Optional[float] = None,
                 name_prefix: str = "Kugel",
                 sphere_type: str = "default") -> bpy.types.Object:
    """Create a sphere at specified location"""
    
    # Get radius from properties or use default
    if radius is None:
        if hasattr(bpy.context.scene, 'chain_construction_props'):
            radius = bpy.context.scene.chain_construction_props.kette_kugelradius
        else:
            radius = constants.DEFAULT_SPHERE_RADIUS
    
    # Generate unique name
    counter = constants.increment_sphere_counter()
    name = f"{name_prefix}_{counter:03d}"
    
    # Create mesh
    mesh = bpy.data.meshes.new(name=name)
    sphere = bpy.data.objects.new(name=name, object_data=mesh)
    
    # Create sphere geometry
    bm = __import__('bmesh').new()
    __import__('bmesh').ops.create_uvsphere(
        bm,
        u_segments=16,
        v_segments=8,
        radius=radius
    )
    bm.to_mesh(mesh)
    bm.free()
    
    # Set location
    sphere.location = location
    
    # Apply material
    material = materials.get_sphere_material(sphere_type)
    materials.apply_material(sphere, material)
    
    # Add to collection
    collection = collections.get_spheres_collection()
    collection.objects.link(sphere)
    
    # Register sphere
    register_sphere(sphere, sphere_type)
    
    if constants.debug:
        constants.debug.info('SPHERE', f"Created {name} at {location}")
    
    return sphere

def register_sphere(sphere: bpy.types.Object, sphere_type: str = "default"):
    """Register sphere in global registry"""
    sphere_registry = constants.get_sphere_registry()
    sphere_registry[sphere.name] = {
        'object': sphere,
        'type': sphere_type,
        'created': __import__('time').time()
    }

def unregister_sphere(sphere_name: str):
    """Remove sphere from registry"""
    sphere_registry = constants.get_sphere_registry()
    if sphere_name in sphere_registry:
        del sphere_registry[sphere_name]

def delete_sphere(sphere: bpy.types.Object):
    """Delete a sphere and cleanup"""
    registry = constants.get_sphere_registry()
    
    # Store name before deletion
    sphere_name = sphere.name
    
    # Remove from registry
    if sphere_name in registry:
        del registry[sphere_name]
    
    # Delete object
    bpy.data.objects.remove(sphere, do_unlink=True)
    
    if constants.debug:
        constants.debug.info('SPHERE', f"Deleted {sphere_name}")
def get_all_spheres() -> List[bpy.types.Object]:
    """Get all sphere objects"""
    spheres = []
    for obj in bpy.data.objects:
        if obj.name.startswith("Kugel"):
            spheres.append(obj)
    return spheres

def get_spheres_by_type(sphere_type: str) -> List[bpy.types.Object]:
    """Get spheres of specific type"""
    sphere_registry = constants.get_sphere_registry()
    spheres = []
    
    for name, data in sphere_registry.items():
        if data['type'] == sphere_type:
            try:
                spheres.append(data['object'])
            except ReferenceError:
                # Object no longer exists
                pass
    
    return spheres

def get_start_spheres() -> List[bpy.types.Object]:
    """Get all start spheres"""
    return get_spheres_by_type('start')

def get_nearest_sphere(location: Vector, max_distance: Optional[float] = None) -> Optional[bpy.types.Object]:
    """Find nearest sphere to location"""
    nearest = None
    min_dist = float('inf')
    
    for sphere in get_all_spheres():
        dist = (sphere.location - location).length
        if dist < min_dist:
            if max_distance is None or dist <= max_distance:
                min_dist = dist
                nearest = sphere
    
    return nearest

def get_connected_spheres(sphere: bpy.types.Object) -> List[bpy.types.Object]:
    """Get spheres connected to given sphere"""
    connected = []
    connector_registry = constants.get_connector_registry()
    
    for conn_data in connector_registry.values():
        if 'sphere1' in conn_data and 'sphere2' in conn_data:
            if conn_data['sphere1'] == sphere.name:
                try:
                    connected.append(bpy.data.objects[conn_data['sphere2']])
                except KeyError:
                    pass
            elif conn_data['sphere2'] == sphere.name:
                try:
                    connected.append(bpy.data.objects[conn_data['sphere1']])
                except KeyError:
                    pass
    
    return connected

# =========================================
# SPHERE OPERATIONS
# =========================================

def select_sphere(sphere: bpy.types.Object, deselect_others: bool = True):
    """Select sphere object"""
    if deselect_others:
        bpy.ops.object.select_all(action='DESELECT')
    
    sphere.select_set(True)
    bpy.context.view_layer.objects.active = sphere

def select_spheres(spheres: List[bpy.types.Object], deselect_others: bool = True):
    """Select multiple spheres"""
    if deselect_others:
        bpy.ops.object.select_all(action='DESELECT')
    
    for sphere in spheres:
        sphere.select_set(True)
    
    if spheres:
        bpy.context.view_layer.objects.active = spheres[0]

def delete_all_spheres():
    """Delete all spheres"""
    spheres = get_all_spheres()
    for sphere in spheres:
        delete_sphere(sphere)
    
    # Clear registry
    constants.get_sphere_registry().clear()
    constants.reset_sphere_counter()
    
    if constants.debug:
        constants.debug.info('SPHERE', f"Deleted all {len(spheres)} spheres")

def update_sphere_radius(sphere: bpy.types.Object, new_radius: float):
    """Update sphere radius"""
    # Recreate mesh with new radius
    mesh = sphere.data
    mesh.clear_geometry()
    
    bm = __import__('bmesh').new()
    __import__('bmesh').ops.create_uvsphere(
        bm,
        u_segments=16,
        v_segments=8,
        radius=new_radius
    )
    bm.to_mesh(mesh)
    bm.free()
    
    mesh.update()

# =========================================
# REGISTRATION
# =========================================

def register():
    """Register spheres module"""
    pass

def unregister():
    """Unregister spheres module"""
    constants.get_sphere_registry().clear()
    constants.reset_sphere_counter()
