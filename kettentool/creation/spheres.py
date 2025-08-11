"""
Sphere Creation System - ERWEITERT
Mit Sphere Registry, Auto-Snap und Type-Management
"""

import bpy
import bmesh
from mathutils import Vector
from typing import Dict, Optional, Tuple, List

from ..utils.collections import ensure_collection, organize_object_in_collection
from ..utils.materials import set_material
from ..utils.surface import get_aura, next_surface_point

# =========================================
# SPHERE REGISTRY MANAGEMENT
# =========================================

def register_sphere(sphere: bpy.types.Object, sphere_type: str = "manual"):
    """
    Registriert Kugel im Sphere Registry
    
    Args:
        sphere: Kugel-Objekt
        sphere_type: 'start', 'painted', 'manual', 'generated'
    """
    from ..core import constants
    
    sphere_data = {
        'name': sphere.name,
        'type': sphere_type,
        'location': sphere.location.copy(),
        'radius': sphere.get("kette_radius", 0.5),
        'created_time': bpy.context.scene.frame_current,
        'properties': dict(sphere.items())
    }
    
    # Zu Registry
    constants._sphere_registry[sphere.name] = sphere_data
    
    # Zu Type-Collection
    type_key = f"{sphere_type}_spheres"
    if type_key in constants._sphere_collections:
        if sphere.name not in constants._sphere_collections[type_key]:
            constants._sphere_collections[type_key].append(sphere.name)
    
    if constants.debug:
        constants.debug.trace('SPHERES', f"Registered {sphere_type} sphere: {sphere.name}")

def unregister_sphere(sphere_name: str):
    """Entfernt Kugel aus Registry"""
    from ..core import constants
    
    # Aus Registry entfernen
    if sphere_name in constants._sphere_registry:
        sphere_data = constants._sphere_registry.pop(sphere_name)
        sphere_type = sphere_data.get('type', 'manual')
        
        # Aus Type-Collection entfernen
        type_key = f"{sphere_type}_spheres"
        if type_key in constants._sphere_collections:
            if sphere_name in constants._sphere_collections[type_key]:
                constants._sphere_collections[type_key].remove(sphere_name)
        
        if constants.debug:
            constants.debug.trace('SPHERES', f"Unregistered sphere: {sphere_name}")

def update_sphere_registry(sphere: bpy.types.Object):
    """Aktualisiert Sphere-Daten im Registry"""
    from ..core import constants
    
    if sphere.name in constants._sphere_registry:
        sphere_data = constants._sphere_registry[sphere.name]
        sphere_data['location'] = sphere.location.copy()
        sphere_data['radius'] = sphere.get("kette_radius", sphere_data['radius'])
        sphere_data['properties'] = dict(sphere.items())
        
        if constants.debug:
            constants.debug.trace('SPHERES', f"Updated sphere registry: {sphere.name}")

# =========================================
# SPHERE CREATION
# =========================================

def create_sphere(location: Vector, 
                  radius: float = None, 
                  resolution: int = None,
                  material_color: Tuple[float, float, float] = None,
                  name_prefix: str = "Kugel", 
                  sphere_type: str = "manual",
                  custom_props: Dict = None) -> bpy.types.Object:
    """
    Erstellt einzelne Kugel
    
    Args:
        location: Position der Kugel
        radius: Radius (None = aus Properties)
        resolution: Auflösung (None = aus Properties)
        material_color: RGB-Farbe (None = nach Typ)
        name_prefix: Name-Präfix
        sphere_type: 'start', 'painted', 'manual', 'generated'
        custom_props: Custom Properties Dictionary
        
    Returns:
        Erstelltes Kugel-Objekt
    """
    from ..core import constants
    
    if constants.debug:
        constants.debug.trace('SPHERES', f"Creating {sphere_type} sphere at {location}")
    
    # Default-Werte von Properties
    props = bpy.context.scene.chain_construction_props
    if radius is None:
        radius = props.kette_kugelradius
    if resolution is None:
        resolution = props.kette_aufloesung
    
    # Material-Farbe basierend auf Typ
    if material_color is None:
        color_map = {
            'start': (1.0, 0.0, 0.0),      # Rot
            'painted': (0.5, 0.7, 1.0),    # Blau  
            'manual': (0.02, 0.02, 0.02),  # Schwarz
            'generated': (0.0, 1.0, 0.0)   # Grün
        }
        material_color = color_map.get(sphere_type, (0.02, 0.02, 0.02))
    
    # Erstelle Mesh
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(
        bm,
        u_segments=resolution,
        v_segments=resolution,
        radius=radius
    )
    
    mesh = bpy.data.meshes.new(f"{name_prefix}_mesh")
    bm.to_mesh(mesh)
    bm.free()
    
    # Eindeutiger Name
    base_name = f"{name_prefix}_{sphere_type}"
    counter = 1
    sphere_name = base_name
    while sphere_name in bpy.data.objects:
        sphere_name = f"{base_name}_{counter:03d}"
        counter += 1
    
    # Erstelle Objekt
    sphere = bpy.data.objects.new(sphere_name, mesh)
    sphere.location = location
    sphere["kette_radius"] = radius
    sphere["sphere_type"] = sphere_type
    
    # Custom Properties
    if custom_props:
        for key, value in custom_props.items():
            sphere[key] = value
    
    # Material
    mat_name = f"Kugel_{sphere_type}"
    set_material(sphere, mat_name, material_color, metallic=0.8, roughness=0.2)
    
    # Organisiere in Collection
    organize_object_in_collection(sphere, "spheres")
    
    # Registriere Kugel
    register_sphere(sphere, sphere_type)
    
    if constants.debug:
        constants.debug.info('SPHERES', f"Created {sphere_type} sphere: {sphere.name}")
    
    return sphere

def create_sphere_at_cursor(context, sphere_type: str = "start") -> Optional[bpy.types.Object]:
    """
    Erstellt Kugel am 3D-Cursor mit Auto-Snap
    
    Args:
        context: Blender Context
        sphere_type: Art der Kugel
        
    Returns:
        Erstellte Kugel oder None bei Fehler
    """
    from ..core import constants
    
    props = context.scene.chain_construction_props
    cursor_loc = context.scene.cursor.location.copy()
    
    # AUTO-SNAP: Immer versuchen wenn Zielobjekt da
    snapped = False
    snap_info = ""
    
    if props.kette_target_obj:
        try:
            aura = get_aura(props.kette_target_obj, props.kette_aura_dichte)
            loc, norm = next_surface_point(aura, cursor_loc)
            offset = (props.kette_kugelradius + props.kette_abstand + 
                     props.kette_global_offset + constants.EPSILON)
            cursor_loc = loc + norm * offset
            snapped = True
            snap_info = f" → snapped to {props.kette_target_obj.name}"
            
            if constants.debug:
                constants.debug.info('SPHERES', f"Auto-snapped to surface: {cursor_loc}")
                
        except Exception as e:
            if constants.debug:
                constants.debug.warning('SPHERES', f"Auto-snap failed: {e}, using cursor position")
            snap_info = " → snap failed, using cursor"
    else:
        snap_info = " → no target object (set for auto-snap)"
    
    # Erstelle Kugel
    sphere = create_sphere(
        location=cursor_loc,
        name_prefix="Kugel",
        sphere_type=sphere_type
    )
    
    if constants.debug:
        constants.debug.info('SPHERES', f"Sphere created{snap_info}")
    
    return sphere

def create_sphere_on_surface(target_obj: bpy.types.Object, 
                           position: Vector,
                           props,
                           sphere_type: str = "painted") -> Optional[bpy.types.Object]:
    """
    Erstellt Kugel auf Oberfläche eines Objekts
    
    Args:
        target_obj: Ziel-Objekt für Surface Snapping
        position: Ungefähre Position
        props: Chain Construction Properties
        sphere_type: Art der Kugel
        
    Returns:
        Erstellte Kugel oder None bei Fehler
    """
    from ..core import constants
    
    try:
        # Surface Snapping
        aura = get_aura(target_obj, props.kette_aura_dichte)
        loc, norm = next_surface_point(aura, position)
        offset = (props.kette_kugelradius + props.kette_abstand + 
                 props.kette_global_offset + constants.EPSILON)
        final_location = loc + norm * offset
        
        # Erstelle Kugel
        sphere = create_sphere(
            location=final_location,
            sphere_type=sphere_type,
            custom_props={'painted': True}
        )
        
        if constants.debug:
            constants.debug.trace('SPHERES', f"Created surface sphere at {final_location}")
        
        return sphere
        
    except Exception as e:
        if constants.debug:
            constants.debug.error('SPHERES', f"Failed to create surface sphere: {e}")
        return None

# =========================================
# SPHERE UTILITIES
# =========================================

def get_all_spheres() -> List[bpy.types.Object]:
    """Gibt alle Kugeln zurück"""
    spheres = []
    
    # Aus Spheres Collection
    spheres_collection = ensure_collection("spheres")
    for obj in spheres_collection.objects:
        if obj.name.startswith("Kugel"):
            spheres.append(obj)
    
    return spheres

def get_spheres_by_type(sphere_type: str) -> List[bpy.types.Object]:
    """
    Gibt alle Kugeln eines bestimmten Typs zurück
    
    Args:
        sphere_type: 'start', 'painted', 'manual', 'generated'
        
    Returns:
        Liste von Sphere-Objekten
    """
    from ..core import constants
    
    type_key = f"{sphere_type}_spheres"
    if type_key not in constants._sphere_collections:
        return []
    
    spheres = []
    sphere_names = constants._sphere_collections[type_key].copy()  # Copy to avoid modification during iteration
    
    for sphere_name in sphere_names:
        obj = bpy.data.objects.get(sphere_name)
        if obj and obj.name.startswith("Kugel"):
            spheres.append(obj)
        else:
            # Cleanup tote Referenzen
            constants._sphere_collections[type_key].remove(sphere_name)
            if constants.debug:
                constants.debug.trace('SPHERES', f"Cleaned up dead reference: {sphere_name}")
    
    return spheres

def get_sphere_count() -> int:
    """Gibt Gesamtanzahl der Kugeln zurück"""
    return len(get_all_spheres())

def get_sphere_statistics() -> Dict[str, int]:
    """Gibt Sphere-Statistiken zurück"""
    from ..core import constants
    
    stats = {}
    total = 0
    
    for sphere_type in ['start', 'painted', 'manual', 'generated']:
        spheres = get_spheres_by_type(sphere_type)
        count = len(spheres)
        stats[sphere_type] = count
        total += count
    
    stats['total'] = total
    return stats

def snap_sphere_to_surface(sphere: bpy.types.Object, target_obj: bpy.types.Object, props) -> bool:
    """
    Snappt eine existierende Kugel auf Oberfläche
    
    Args:
        sphere: Zu snappende Kugel
        target_obj: Ziel-Objekt
        props: Chain Construction Properties
        
    Returns:
        True wenn erfolgreich gesnapped
    """
    from ..core import constants
    
    try:
        aura = get_aura(target_obj, props.kette_aura_dichte)
        loc, norm = next_surface_point(aura, sphere.location)
        offset = (props.kette_kugelradius + props.kette_abstand + 
                 props.kette_global_offset + constants.EPSILON)
        sphere.location = loc + norm * offset
        
        # Update Registry
        update_sphere_registry(sphere)
        
        if constants.debug:
            constants.debug.trace('SPHERES', f"Snapped sphere {sphere.name} to surface")
        
        return True
        
    except Exception as e:
        if constants.debug:
            constants.debug.error('SPHERES', f"Failed to snap sphere: {e}")
        return False

# =========================================
# SPHERE CLEANUP
# =========================================

def delete_sphere(sphere: bpy.types.Object):
    """Löscht eine Kugel sauber"""
    from ..core import constants
    
    if constants.debug:
        constants.debug.trace('SPHERES', f"Deleting sphere: {sphere.name}")
    
    # Unregister first
    unregister_sphere(sphere.name)
    
    # Entferne aus Collections
    for c in sphere.users_collection:
        c.objects.unlink(sphere)
    
    # Lösche Mesh wenn nur von diesem Objekt verwendet
    mesh = sphere.data
    if mesh and mesh.users == 1:
        bpy.data.meshes.remove(mesh)
    
    # Lösche Objekt
    bpy.data.objects.remove(sphere, do_unlink=True)

def cleanup_sphere_registry():
    """Bereinigt Sphere Registry von toten Referenzen"""
    from ..core import constants
    
    # Prüfe Registry
    dead_spheres = []
    for sphere_name in list(constants._sphere_registry.keys()):
        if sphere_name not in bpy.data.objects:
            dead_spheres.append(sphere_name)
    
    for sphere_name in dead_spheres:
        unregister_sphere(sphere_name)
    
    if constants.debug and dead_spheres:
        constants.debug.info('SPHERES', f"Cleaned up {len(dead_spheres)} dead sphere references")

def cleanup_orphaned_spheres():
    """Entfernt verwaiste Kugeln ohne Mesh"""
    from ..core import constants
    
    removed = 0
    spheres_collection = ensure_collection("spheres")
    
    to_remove = []
    for obj in spheres_collection.objects:
        if (obj.name.startswith("Kugel") and 
            (not obj.data or obj.data.name not in bpy.data.meshes)):
            to_remove.append(obj)
    
    for obj in to_remove:
        delete_sphere(obj)
        removed += 1
    
    if constants.debug and removed > 0:
        constants.debug.info('SPHERES', f"Cleaned up {removed} orphaned spheres")
    
    return removed

def validate_sphere_integrity() -> Dict[str, Any]:
    """
    Validiert Integrität des Sphere-Systems
    
    Returns:
        Validierungs-Report
    """
    from ..core import constants
    
    report = {
        'registry_count': len(constants._sphere_registry),
        'collection_counts': {},
        'orphaned_objects': [],
        'missing_objects': [],
        'inconsistencies': []
    }
    
    # Zähle Objekte in Collections
    for sphere_type in ['start', 'painted', 'manual', 'generated']:
        type_key = f"{sphere_type}_spheres"
        if type_key in constants._sphere_collections:
            report['collection_counts'][sphere_type] = len(constants._sphere_collections[type_key])
    
    # Prüfe auf verwaiste Objekte
    spheres_collection = ensure_collection("spheres")
    for obj in spheres_collection.objects:
        if obj.name.startswith("Kugel") and obj.name not in constants._sphere_registry:
            report['orphaned_objects'].append(obj.name)
    
    # Prüfe auf fehlende Objekte
    for sphere_name in constants._sphere_registry:
        if sphere_name not in bpy.data.objects:
            report['missing_objects'].append(sphere_name)
    
    # Zähle Inkonsistenzen
    report['inconsistencies'] = len(report['orphaned_objects']) + len(report['missing_objects'])
    
    return report

# =========================================
# REGISTRATION
# =========================================

def register():
    """Registriert Spheres-Modul"""
    pass

def unregister():
    """Deregistriert Spheres-Modul"""
    cleanup_sphere_registry()
