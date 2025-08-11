"""
Connector Creation System
"""

import bpy
from mathutils import Vector, Matrix
from typing import Optional, Tuple

from ..utils.collections import ensure_collection
from ..utils.materials import set_material

# =========================================
# CONNECTOR CREATION
# =========================================

def connect_spheres(sphere1: bpy.types.Object, 
                   sphere2: bpy.types.Object, 
                   use_curved: bool = True,
                   props = None) -> Optional[bpy.types.Object]:
    """
    Verbindet zwei Kugeln mit Connector
    
    Args:
        sphere1: Erste Kugel
        sphere2: Zweite Kugel
        use_curved: True für gebogene, False für gerade Verbinder
        props: Chain Construction Properties (None = auto-detect)
        
    Returns:
        Erstellter Connector oder None bei Fehler
    """
    from ..core import constants
    
    if props is None:
        props = bpy.context.scene.chain_construction_props
    
    # Check if connection already exists
    pair_key = tuple(sorted([sphere1.name, sphere2.name]))
    if pair_key in constants._existing_connectors:
        existing = constants._existing_connectors[pair_key]
        if existing.name in bpy.data.objects:
            if constants.debug:
                constants.debug.trace('CONNECTORS', f"Connection already exists: {pair_key}")
            return existing
        else:
            constants._existing_connectors.pop(pair_key, None)
    
    if constants.debug:
        constants.debug.trace('CONNECTORS', f"Creating connector: {sphere1.name} -> {sphere2.name}")
    
    loc1 = sphere1.location
    loc2 = sphere2.location
    distance = (loc2 - loc1).length
    
    if distance < constants.EPSILON:
        if constants.debug:
            constants.debug.warning('CONNECTORS', f"Zero distance between spheres")
        return None
    
    # Berechne Connector-Radius
    radius1 = sphere1.get("kette_radius", props.kette_kugelradius)
    radius2 = sphere2.get("kette_radius", props.kette_kugelradius)
    connector_radius = min(radius1, radius2) * props.kette_connector_radius_factor
    
    # Erstelle Connector
    if use_curved:
        connector = create_curved_connector(loc1, loc2, connector_radius, props)
    else:
        connector = create_straight_connector(loc1, loc2, connector_radius, props)
    
    if connector:
        # Cache
        constants._existing_connectors[pair_key] = connector
        
        # Custom properties
        connector["sphere1"] = sphere1.name
        connector["sphere2"] = sphere2.name
        connector["connector_type"] = "curved" if use_curved else "straight"
        
        if constants.debug:
            constants.debug.info('CONNECTORS', f"Created {connector['connector_type']} connector")
        
        return connector
    
    return None

def create_straight_connector(loc1: Vector, 
                            loc2: Vector, 
                            radius: float, 
                            props) -> bpy.types.Object:
    """
    Erstellt geraden Zylinder-Connector
    
    Args:
        loc1: Start-Position
        loc2: End-Position  
        radius: Connector-Radius
        props: Chain Construction Properties
        
    Returns:
        Connector-Objekt
    """
    from ..core import constants
    
    direction = loc2 - loc1
    length = direction.length
    center = (loc1 + loc2) * 0.5
    
    # Erstelle Zylinder
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=props.kette_zyl_segs,
        radius=radius,
        depth=length,
        location=center
    )
    
    connector = bpy.context.active_object
    connector.name = "Connector_Straight"
    
    # Orientierung
    direction.normalize()
    up = Vector((0, 0, 1))
    if abs(direction.dot(up)) > 0.99:
        up = Vector((1, 0, 0))
    
    right = direction.cross(up).normalized()
    up = right.cross(direction).normalized()
    
    rotation_matrix = Matrix((right, up, direction)).transposed()
    connector.rotation_euler = rotation_matrix.to_euler()
    
    # Material und Collection
    set_material(connector, "KetteVerbinder", (0.1, 0.1, 0.1), metallic=0.9, roughness=0.3)
    coll = ensure_collection("KonstruktionsKette")
    coll.objects.link(connector)
    
    # Remove from scene collection
    for c in list(connector.users_collection):
        if c != coll:
            c.objects.unlink(connector)
    
    if constants.debug:
        constants.debug.trace('CONNECTORS', f"Created straight connector, length: {length:.3f}")
    
    return connector

def create_curved_connector(loc1: Vector, 
                          loc2: Vector, 
                          radius: float, 
                          props) -> bpy.types.Object:
    """
    Erstellt gebogenen Spline-Connector
    
    Args:
        loc1: Start-Position
        loc2: End-Position
        radius: Connector-Radius
        props: Chain Construction Properties
        
    Returns:
        Connector-Objekt
    """
    from ..core import constants
    
    # Berechne Zwischenpunkte für Kurve
    direction = (loc2 - loc1).normalized()
    distance = (loc2 - loc1).length
    
    # Control point für Biegung
    midpoint = (loc1 + loc2) * 0.5
    perpendicular = Vector((direction.y, -direction.x, 0)).normalized()
    if perpendicular.length < constants.EPSILON:
        perpendicular = Vector((0, direction.z, -direction.y)).normalized()
    
    # Leichte Biegung
    curve_height = distance * 0.1
    control_point = midpoint + perpendicular * curve_height
    
    # Erstelle Curve
    curve_data = bpy.data.curves.new('ConnectorCurve', type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.fill_mode = 'FULL'
    curve_data.bevel_depth = radius
    curve_data.bevel_resolution = 2
    curve_data.resolution_u = props.kette_connector_segs
    
    # Spline
    spline = curve_data.splines.new('BEZIER')
    spline.bezier_points.add(2)  # Total 3 points
    
    # Setze Punkte
    points = [loc1, control_point, loc2]
    for i, point in enumerate(points):
        bp = spline.bezier_points[i]
        bp.co = point
        bp.handle_left_type = 'AUTO'
        bp.handle_right_type = 'AUTO'
    
    # Erstelle Objekt
    connector = bpy.data.objects.new("Connector_Curved", curve_data)
    
    # Material und Collection
    set_material(connector, "KetteVerbinder", (0.1, 0.1, 0.1), metallic=0.9, roughness=0.3)
    coll = ensure_collection("KonstruktionsKette")
    coll.objects.link(connector)
    
    if constants.debug:
        constants.debug.trace('CONNECTORS', f"Created curved connector, distance: {distance:.3f}")
    
    return connector

# =========================================
# CONNECTOR UTILITIES
# =========================================

def refresh_connectors() -> Tuple[int, int]:
    """
    Aktualisiert alle Connectors
    
    Returns:
        Tuple[updated_count, deleted_count]
    """
    from ..core import constants
    
    if constants.debug:
        constants.debug.info('CONNECTORS', "Refreshing all connectors")
    
    coll = bpy.data.collections.get("KonstruktionsKette")
    if not coll:
        return 0, 0

    updated = 0
    deleted = 0
    
    # Sammle alle Connectors
    connectors = [obj for obj in coll.objects if obj.name.startswith("Connector")]
    
    for conn in connectors:
        sphere1_name = conn.get("sphere1")
        sphere2_name = conn.get("sphere2")
        
        if not sphere1_name or not sphere2_name:
            continue
        
        sphere1 = bpy.data.objects.get(sphere1_name)
        sphere2 = bpy.data.objects.get(sphere2_name)
        
        if not sphere1 or not sphere2:
            # Kugel wurde gelöscht - lösche Connector
            bpy.data.objects.remove(conn, do_unlink=True)
            deleted += 1
            if constants.debug:
                constants.debug.trace('CONNECTORS', f"Deleted orphaned connector: {conn.name}")
            continue
        
        # Update connector position (für straight connectors)
        if conn.get("connector_type") == "straight":
            center = (sphere1.location + sphere2.location) * 0.5
            conn.location = center
            updated += 1
    
    # Cleanup cache
    constants._existing_connectors.clear()
    
    if constants.debug:
        constants.debug.info('CONNECTORS', f"Refreshed connectors: {updated} updated, {deleted} deleted")
    
    return updated, deleted

def get_connector_count() -> int:
    """Gibt Anzahl der Connectors zurück"""
    coll = bpy.data.collections.get("KonstruktionsKette")
    if not coll:
        return 0
    
    return len([obj for obj in coll.objects if obj.name.startswith("Connector")])

def delete_connector(connector: bpy.types.Object):
    """Löscht einen Connector sauber"""
    from ..core import constants
    
    if constants.debug:
        constants.debug.trace('CONNECTORS', f"Deleting connector: {connector.name}")
    
    # Entferne aus Collections
    for c in connector.users_collection:
        c.objects.unlink(connector)
    
    # Lösche Curve/Mesh wenn nur von diesem Objekt verwendet
    data = connector.data
    if data and data.users == 1:
        if connector.type == 'CURVE':
            bpy.data.curves.remove(data)
        elif connector.type == 'MESH':
            bpy.data.meshes.remove(data)
    
    # Lösche Objekt
    bpy.data.objects.remove(connector, do_unlink=True)

# =========================================
# REGISTRATION
# =========================================

def register():
    """Registriert Connectors-Modul"""
    pass

def unregister():
    """Deregistriert Connectors-Modul"""
    pass
