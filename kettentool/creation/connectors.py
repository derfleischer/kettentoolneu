"""
Connector Creation System
SAUBERE VERSION - nutzt die tatsächlichen Variablennamen aus dem System
"""

import bpy
import bmesh
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from typing import Optional, Tuple

from ..utils.collections import ensure_collection
from ..utils.materials import set_material
from ..utils.surface import get_aura, next_surface_point
from ..core import constants

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
    if props is None:
        props = bpy.context.scene.chain_construction_props
    
    # Check if connection already exists
    pair_key = tuple(sorted([sphere1.name, sphere2.name]))
    
    # Nutze _connector_registry (der RICHTIGE Name in deinem System)
    if pair_key in constants._connector_registry:
        existing = constants._connector_registry[pair_key]
        try:
            if existing and existing.name in bpy.data.objects:
                if constants.debug:
                    constants.debug.trace('CONNECTORS', f"Connection already exists: {pair_key}")
                return existing
        except (AttributeError, ReferenceError):
            pass
        constants._connector_registry.pop(pair_key, None)
    
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
        # Cache im _connector_registry
        constants._connector_registry[pair_key] = connector
        
        # Füge zu _connector_pairs hinzu
        constants._connector_pairs.add(pair_key)
        
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
    direction = loc2 - loc1
    length = direction.length
    center = (loc1 + loc2) * 0.5
    
    # Erstelle Zylinder (kette_zyl_segs existiert möglicherweise nicht)
    vertices = 16  # Default
    if hasattr(props, 'kette_zyl_segs'):
        vertices = props.kette_zyl_segs
    
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
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
    set_material(connector, constants.MAT_CONNECTOR, constants.COLOR_CONNECTOR, metallic=0.9, roughness=0.3)
    coll = ensure_collection(constants.COLLECTION_MAIN)
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
    Erstellt gebogenen Connector der der Oberfläche folgt (geodätisch).
    
    Args:
        loc1: Start-Position
        loc2: End-Position
        radius: Connector-Radius (bereits mit factor multipliziert)
        props: Chain Construction Properties
        
    Returns:
        Connector-Objekt
    """
    # Parameter aus props (mit Defaults für fehlende Properties)
    steps = 20  # Default
    if hasattr(props, 'kette_connector_segs'):
        steps = props.kette_connector_segs
    
    # Aura-Dicke (Default wenn nicht vorhanden)
    aura_thickness = 0.05
    if hasattr(props, 'kette_aura_dichte'):
        aura_thickness = props.kette_aura_dichte
    
    offset = props.kette_abstand + props.kette_global_offset + constants.EPSILON * 2
    
    # Berechne tatsächlichen Kugelradius
    sphere_radius = radius / props.kette_connector_radius_factor
    
    # Hole Aura für Oberflächenprojektion
    aura = get_aura(props.kette_target_obj, aura_thickness)
    
    if not aura:
        if constants.debug:
            constants.debug.error('CONNECTORS', "No aura object found")
        return None
    
    # Liste für Kurvenpunkte
    curve_points = []
    
    # Richtung und Überlappung
    direction = (loc2 - loc1)
    if direction.length < constants.EPSILON:
        if constants.debug:
            constants.debug.warning('CONNECTORS', "Locations too close")
        return None
    direction.normalize()
    
    overlap = 0.2
    
    # Erweiterte Start- und Endpunkte (in die Kugeln hinein)
    start_extended = loc1 - direction * (sphere_radius * overlap)
    end_extended = loc2 + direction * (sphere_radius * overlap)
    
    # Startpunkt auf Oberfläche projizieren
    current_pos, current_normal = next_surface_point(aura, start_extended)
    curve_points.append(current_pos + current_normal * (sphere_radius + offset))
    
    # Geodätischer Pfad - folge der Oberfläche statt linear zu interpolieren
    for i in range(1, steps):
        t = i / (steps - 1)
        
        if i < steps - 1:
            # Berechne Richtung zum Ziel
            to_target = end_extended - current_pos
            distance_remaining = to_target.length
            
            if distance_remaining > constants.EPSILON:
                to_target.normalize()
                
                # Projiziere auf Tangentialebene der Oberfläche
                dot_product = to_target.dot(current_normal)
                tangent_direction = to_target - current_normal * dot_product
                
                if tangent_direction.length > constants.EPSILON:
                    tangent_direction.normalize()
                else:
                    # Fallback wenn parallel zur Normale
                    if abs(current_normal.z) < 0.99:
                        tangent_direction = Vector((0, 0, 1)).cross(current_normal).normalized()
                    else:
                        tangent_direction = Vector((1, 0, 0)).cross(current_normal).normalized()
                
                # Schrittweite basierend auf verbleibender Distanz
                step_size = distance_remaining / (steps - i)
                next_guess = current_pos + tangent_direction * step_size
            else:
                next_guess = end_extended
        else:
            # Letzter Punkt: direkt zum erweiterten Endpunkt
            next_guess = end_extended
        
        # Projiziere auf Oberfläche
        surface_pos, surface_normal = next_surface_point(aura, next_guess)
        current_pos = surface_pos
        current_normal = surface_normal
        
        # Füge Punkt mit Offset hinzu
        curve_points.append(surface_pos + surface_normal * (sphere_radius + offset))
    
    # Erstelle Kurve aus den berechneten Punkten
    curve_data = bpy.data.curves.new("ConnectorCurve", 'CURVE')
    curve_data.dimensions = '3D'
    
    # Erstelle Polyline Spline
    spline = curve_data.splines.new('POLY')
    spline.points.add(len(curve_points) - 1)
    
    # Setze Kurvenpunkte
    for i, point in enumerate(curve_points):
        spline.points[i].co = (point.x, point.y, point.z, 1)
        spline.points[i].radius = radius
    
    # Kurveneinstellungen für glatte Darstellung
    spline.use_smooth = True
    curve_data.use_fill_caps = True
    curve_data.bevel_depth = radius
    curve_data.bevel_resolution = 4
    
    # Erstelle Objekt
    connector = bpy.data.objects.new("Connector_Curved", curve_data)
    
    # Temporär zur Scene Collection für Konvertierung
    temp_collection = bpy.context.collection
    temp_collection.objects.link(connector)
    
    # Konvertiere zu Mesh für saubere Geometrie
    bpy.context.view_layer.objects.active = connector
    bpy.ops.object.convert(target='MESH')
    
    # Entferne aus temporärer Collection
    temp_collection.objects.unlink(connector)
    
    # Material setzen
    set_material(connector, constants.MAT_CONNECTOR, constants.COLOR_CONNECTOR, metallic=0.9, roughness=0.3)
    
    # Zur richtigen Collection hinzufügen
    coll = ensure_collection(constants.COLLECTION_MAIN)
    coll.objects.link(connector)
    
    if constants.debug:
        constants.debug.trace('CONNECTORS', 
                             f"Created curved connector with {len(curve_points)} points, distance: {(loc2-loc1).length:.3f}")
    
    return connector

# =========================================
# CONNECTOR UTILITIES
# =========================================

def refresh_connectors() -> Tuple[int, int]:
    """
    Aktualisiert alle Connectors bei Positionsänderungen
    
    Returns:
        Tuple[updated_count, deleted_count]
    """
    if constants.debug:
        constants.debug.info('CONNECTORS', "Refreshing all connectors")
    
    coll = bpy.data.collections.get(constants.COLLECTION_MAIN)
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
            delete_connector(conn)
            deleted += 1
        else:
            # Update Connector Position wenn nötig
            if update_connector_if_needed(conn, sphere1, sphere2):
                updated += 1
    
    # Cleanup registry
    constants._connector_registry.clear()
    
    if constants.debug:
        constants.debug.info('CONNECTORS', f"Updated: {updated}, Deleted: {deleted}")
    
    return updated, deleted

def delete_connector(connector: bpy.types.Object):
    """
    Löscht einen Connector sauber
    
    Args:
        connector: Zu löschender Connector
    """
    # Entferne aus Registry
    sphere1_name = connector.get("sphere1")
    sphere2_name = connector.get("sphere2")
    if sphere1_name and sphere2_name:
        pair_key = tuple(sorted([sphere1_name, sphere2_name]))
        constants._connector_registry.pop(pair_key, None)
        constants._connector_pairs.discard(pair_key)
    
    # Entferne aus Collections
    for coll in list(connector.users_collection):
        coll.objects.unlink(connector)
    
    # Lösche Objekt
    bpy.data.objects.remove(connector, do_unlink=True)
    
    if constants.debug:
        constants.debug.trace('CONNECTORS', f"Deleted connector: {connector.name}")

def update_connector_if_needed(connector: bpy.types.Object,
                              sphere1: bpy.types.Object,
                              sphere2: bpy.types.Object) -> bool:
    """
    Aktualisiert Connector wenn sich Kugelpositionen geändert haben
    
    Args:
        connector: Zu prüfender Connector
        sphere1: Erste Kugel
        sphere2: Zweite Kugel
        
    Returns:
        True wenn aktualisiert wurde
    """
    # Für jetzt: Keine automatischen Updates
    return False

def get_connector_count() -> int:
    """Gibt Anzahl der Connectors zurück"""
    coll = bpy.data.collections.get(constants.COLLECTION_MAIN)
    if not coll:
        return 0
    
    return len([obj for obj in coll.objects if obj.name.startswith("Connector")])

# =========================================
# REGISTRATION
# =========================================

def register():
    """Registriert Connectors-Modul"""
    pass

def unregister():
    """Deregistriert Connectors-Modul"""
    pass
