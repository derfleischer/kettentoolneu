"""
Paint Mode Utilities for Kettentool
"""

import bpy
import gpu
import blf
import math
from gpu_extras.batch import batch_for_shader
from mathutils import Vector
from typing import Optional, List
from bpy_extras.view3d_utils import (
    region_2d_to_vector_3d,
    region_2d_to_origin_3d,
    location_3d_to_region_2d
)


def get_paint_surface_position(context, event, target_obj: bpy.types.Object) -> Optional[Vector]:
    """
    Berechnet 3D-Position auf der Oberfläche unter dem Mauszeiger
    
    Args:
        context: Blender context
        event: Mouse event
        target_obj: Zielobjekt für Oberflächenprojektion
        
    Returns:
        3D Position auf der Oberfläche oder None
    """
    if not target_obj:
        return None
    
    mouse_pos = (event.mouse_region_x, event.mouse_region_y)
    region = context.region
    region_3d = context.space_data.region_3d
    
    # Get ray from mouse
    view_vector = region_2d_to_vector_3d(region, region_3d, mouse_pos)
    ray_origin = region_2d_to_origin_3d(region, region_3d, mouse_pos)
    
    # Cast ray
    result, location, normal, index = target_obj.ray_cast(ray_origin, view_vector)
    
    if result:
        # Apply offset
        props = context.scene.chain_props
        offset = props.kette_abstand + props.kette_global_offset
        return location + normal * offset
    
    return None


def check_minimum_distance(position: Vector, 
                          existing_spheres: List[bpy.types.Object], 
                          min_distance: float) -> bool:
    """
    Prüft Mindestabstand zu existierenden Kugeln
    
    Args:
        position: Neue Position
        existing_spheres: Existierende Kugeln
        min_distance: Minimaler Abstand
        
    Returns:
        True wenn Abstand ausreichend
    """
    for sphere in existing_spheres:
        if sphere and sphere.name in bpy.data.objects:
            if (sphere.location - position).length < min_distance:
                return False
    return True


def get_symmetry_positions(position: Vector, 
                          axis: str = 'X',
                          center: Vector = Vector((0, 0, 0))) -> List[Vector]:
    """
    Berechnet Symmetrie-Positionen
    
    Args:
        position: Original-Position
        axis: Symmetrie-Achse ('X', 'Y', 'Z')
        center: Symmetrie-Zentrum
        
    Returns:
        Liste mit Original und gespiegelten Positionen
    """
    positions = [position]
    mirrored = position.copy()
    
    if axis == 'X':
        mirrored.x = 2 * center.x - position.x
    elif axis == 'Y':
        mirrored.y = 2 * center.y - position.y
    elif axis == 'Z':
        mirrored.z = 2 * center.z - position.z
    
    positions.append(mirrored)
    return positions


def find_sphere_at_cursor(context, event, search_radius: float = 50.0) -> Optional[bpy.types.Object]:
    """
    Findet Kugel unter dem Mauszeiger
    
    Args:
        context: Blender context
        event: Mouse event
        search_radius: Such-Radius in Pixeln
        
    Returns:
        Gefundene Kugel oder None
    """
    mouse_pos = (event.mouse_region_x, event.mouse_region_y)
    region = context.region
    region_3d = context.space_data.region_3d
    
    spheres = [obj for obj in bpy.data.objects if obj.name.startswith("Kugel")]
    
    closest_sphere = None
    closest_dist = float('inf')
    
    for sphere in spheres:
        sphere_2d = location_3d_to_region_2d(region, region_3d, sphere.location)
        if sphere_2d:
            dist = (Vector(mouse_pos) - sphere_2d).length
            if dist < search_radius and dist < closest_dist:
                closest_sphere = sphere
                closest_dist = dist
    
    return closest_sphere


def draw_paint_preview(self, context):
    """
    Zeichnet Paint Mode Preview
    """
    if not hasattr(self, 'cursor_3d_pos'):
        return
        
    props = context.scene.chain_props
    
    # Setup GPU
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.line_width_set(2.0)
    
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    
    # Draw cursor preview
    if self.cursor_3d_pos:
        # Create circle
        segments = 32
        radius = props.kette_kugelradius
        
        vertices = []
        for i in range(segments + 1):
            angle = 2 * math.pi * i / segments
            x = self.cursor_3d_pos.x + radius * math.cos(angle)
            y = self.cursor_3d_pos.y + radius * math.sin(angle)
            z = self.cursor_3d_pos.z
            vertices.append((x, y, z))
        
        # Draw preview circle
        batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": vertices})
        shader.bind()
        
        # Color based on mode
        if getattr(self, 'navigation_mode', False):
            shader.uniform_float("color", (0.5, 0.5, 1.0, 0.5))
        else:
            shader.uniform_float("color", (0.0, 1.0, 0.0, 0.5))
        
        batch.draw(shader)
    
    # Draw connection preview
    if (hasattr(self, 'painted_spheres') and 
        self.painted_spheres and 
        self.cursor_3d_pos and 
        props.kette_auto_connect):
        
        last_sphere = self.painted_spheres[-1]
        if last_sphere and last_sphere.name in bpy.data.objects:
            line_batch = batch_for_shader(shader, 'LINES', {
                "pos": [last_sphere.location, self.cursor_3d_pos]
            })
            shader.uniform_float("color", (0.5, 1.0, 0.5, 0.3))
            line_batch.draw(shader)
    
    # Reset GPU
    gpu.state.depth_test_set('NONE')
    gpu.state.blend_set('NONE')
    gpu.state.line_width_set(1.0)
    
    # Draw HUD
    draw_paint_hud(self, context)


def draw_paint_hud(self, context):
    """
    Zeichnet 2D HUD für Paint Mode
    """
    props = context.scene.chain_props
    region = context.region
    
    # Main status
    if getattr(self, 'navigation_mode', False):
        text = "NAVIGATION MODE"
        color = (0.5, 0.5, 1.0, 1.0)
    else:
        text = f"PAINT MODE - Radius: {props.kette_kugelradius:.3f}"
        color = (0.0, 1.0, 0.0, 1.0)
    
    blf.size(0, 20)
    blf.color(0, *color)
    blf.position(0, region.width // 2 - 100, region.height - 40, 0)
    blf.draw(0, text)
    
    # Sphere count
    if hasattr(self, 'painted_spheres'):
        count_text = f"Kugeln: {len(self.painted_spheres)}"
        blf.size(0, 14)
        blf.color(0, 0.7, 0.7, 0.7, 1.0)
        blf.position(0, region.width // 2 - 50, region.height - 65, 0)
        blf.draw(0, count_text)
    
    # Help text
    help_y = 100
    blf.size(0, 12)
    blf.color(0, 0.6, 0.6, 0.6, 0.8)
    
    help_texts = [
        "LMB: Malen | RMB: Löschen | SPACE: Navigation",
        "C: Verbinden | P: Pattern | T: Triangulation",
        "S: Symmetrie | A: Auto-Connect | ESC: Beenden"
    ]
    
    for text in help_texts:
        blf.position(0, 10, help_y, 0)
        blf.draw(0, text)
        help_y -= 20
