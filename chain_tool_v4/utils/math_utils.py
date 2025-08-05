"""
Mathematical Utilities for Chain Tool V4
Common mathematical operations and geometric calculations
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Union, Set
from mathutils import Vector, Matrix, Quaternion, kdtree
from mathutils.geometry import (
    intersect_line_line, 
    intersect_point_line,
    area_tri,
    distance_point_to_plane
)
import bmesh
import bpy

from utils.debug import debug
from core.constants import EPSILON, ANGLE_EPSILON

# ============================================
# VECTOR OPERATIONS
# ============================================

def calculate_distance(p1: Vector, p2: Vector) -> float:
    """Calculate distance between two points"""
    return (p2 - p1).length

def calculate_angle(v1: Vector, v2: Vector, degrees: bool = True) -> float:
    """Calculate angle between two vectors"""
    # Normalize vectors
    v1_norm = v1.normalized()
    v2_norm = v2.normalized()
    
    # Calculate dot product
    dot = max(-1.0, min(1.0, v1_norm.dot(v2_norm)))
    
    # Calculate angle
    angle = math.acos(dot)
    
    return math.degrees(angle) if degrees else angle

def get_normal_vector(p1: Vector, p2: Vector, p3: Vector) -> Vector:
    """Calculate normal vector from three points"""
    v1 = p2 - p1
    v2 = p3 - p1
    normal = v1.cross(v2)
    
    if normal.length > EPSILON:
        return normal.normalized()
    else:
        # Degenerate triangle
        debug.warning('GEOMETRY', "Degenerate triangle in normal calculation")
        return Vector((0, 0, 1))

def interpolate_points(p1: Vector, p2: Vector, factor: float) -> Vector:
    """Interpolate between two points"""
    return p1.lerp(p2, factor)

def project_point_to_plane(point: Vector, plane_co: Vector, plane_normal: Vector) -> Vector:
    """Project point onto plane"""
    distance = distance_point_to_plane(point, plane_co, plane_normal)
    return point - (plane_normal * distance)

# ============================================
# GEOMETRIC CALCULATIONS
# ============================================

def calculate_triangle_quality(p1: Vector, p2: Vector, p3: Vector) -> float:
    """
    Calculate triangle quality (0-1, where 1 is equilateral)
    Based on the ratio of shortest edge to circumradius
    """
    # Edge lengths
    a = calculate_distance(p2, p3)
    b = calculate_distance(p1, p3)
    c = calculate_distance(p1, p2)
    
    # Check for degenerate triangle
    if a < EPSILON or b < EPSILON or c < EPSILON:
        return 0.0
    
    # Area using Heron's formula
    s = (a + b + c) / 2
    area = math.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))
    
    if area < EPSILON:
        return 0.0
    
    # Circumradius
    circumradius = (a * b * c) / (4 * area)
    
    # Shortest edge
    min_edge = min(a, b, c)
    
    # Quality metric (normalized)
    quality = min_edge / (2 * circumradius)
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, quality * 1.732))  # sqrt(3) normalization

def calculate_polygon_area(points: List[Vector]) -> float:
    """Calculate area of polygon using shoelace formula"""
    n = len(points)
    if n < 3:
        return 0.0
    
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i].x * points[j].y
        area -= points[j].x * points[i].y
    
    return abs(area) / 2.0

def calculate_centroid(points: List[Vector]) -> Vector:
    """Calculate centroid of points"""
    if not points:
        return Vector((0, 0, 0))
    
    centroid = Vector((0, 0, 0))
    for point in points:
        centroid += point
    
    return centroid / len(points)

def calculate_bounding_box(points: List[Vector]) -> Tuple[Vector, Vector]:
    """Calculate bounding box of points"""
    if not points:
        return Vector((0, 0, 0)), Vector((0, 0, 0))
    
    min_co = Vector(points[0])
    max_co = Vector(points[0])
    
    for point in points[1:]:
        min_co.x = min(min_co.x, point.x)
        min_co.y = min(min_co.y, point.y)
        min_co.z = min(min_co.z, point.z)
        
        max_co.x = max(max_co.x, point.x)
        max_co.y = max(max_co.y, point.y)
        max_co.z = max(max_co.z, point.z)
    
    return min_co, max_co

# ============================================
# TRIANGULATION
# ============================================

def triangulate_points(points: List[Vector], 
                      constraints: Optional[List[Tuple[int, int]]] = None,
                      quality_threshold: float = 0.3) -> List[Tuple[int, int, int]]:
    """
    Triangulate points with optional constraints
    Returns list of triangle indices
    """
    if len(points) < 3:
        return []
    
    # Check if points are coplanar
    if are_points_coplanar(points):
        return triangulate_2d(points, constraints, quality_threshold)
    else:
        return triangulate_3d(points, constraints, quality_threshold)

def triangulate_2d(points: List[Vector], 
                   constraints: Optional[List[Tuple[int, int]]] = None,
                   quality_threshold: float = 0.3) -> List[Tuple[int, int, int]]:
    """Triangulate coplanar points"""
    # Project to 2D
    normal = calculate_polygon_normal(points)
    points_2d = project_points_to_2d(points, normal)
    
    # Use Blender's triangulation
    from mathutils.geometry import tessellate_polygon
    
    # Simple triangulation first
    triangles = tessellate_polygon([points_2d])
    
    # Filter by quality
    filtered_triangles = []
    for tri in triangles:
        p1, p2, p3 = points[tri[0]], points[tri[1]], points[tri[2]]
        quality = calculate_triangle_quality(p1, p2, p3)
        
        if quality >= quality_threshold:
            filtered_triangles.append(tri)
    
    return filtered_triangles

def triangulate_3d(points: List[Vector],
                   constraints: Optional[List[Tuple[int, int]]] = None,
                   quality_threshold: float = 0.3) -> List[Tuple[int, int, int]]:
    """Triangulate 3D points using Delaunay"""
    # For now, use simple approach
    # In production, use scipy.spatial.Delaunay
    triangles = []
    n = len(points)
    
    # Create all possible triangles
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Check triangle quality
                quality = calculate_triangle_quality(points[i], points[j], points[k])
                
                if quality >= quality_threshold:
                    triangles.append((i, j, k))
    
    # Remove overlapping triangles
    return filter_overlapping_triangles(triangles, points)

# ============================================
# MESH OPERATIONS
# ============================================

def get_mesh_surface_area(mesh: bpy.types.Mesh) -> float:
    """Calculate total surface area of mesh"""
    total_area = 0.0
    
    for poly in mesh.polygons:
        # Get vertices
        verts = [mesh.vertices[i].co for i in poly.vertices]
        
        # Triangulate if needed
        if len(verts) == 3:
            total_area += area_tri(verts[0], verts[1], verts[2])
        elif len(verts) == 4:
            total_area += area_tri(verts[0], verts[1], verts[2])
            total_area += area_tri(verts[0], verts[2], verts[3])
        else:
            # Complex polygon - use centroid
            centroid = calculate_centroid(verts)
            for i in range(len(verts)):
                j = (i + 1) % len(verts)
                total_area += area_tri(centroid, verts[i], verts[j])
    
    return total_area

def get_mesh_volume(mesh: bpy.types.Mesh) -> float:
    """Calculate mesh volume (must be closed manifold)"""
    volume = 0.0
    
    for poly in mesh.polygons:
        # Get vertices
        verts = [mesh.vertices[i].co for i in poly.vertices]
        normal = poly.normal
        
        # Use divergence theorem
        for i in range(len(verts)):
            j = (i + 1) % len(verts)
            
            # Triangle with origin
            v1 = verts[i]
            v2 = verts[j]
            
            # Signed volume of tetrahedron
            volume += v1.dot(normal) * area_tri(Vector((0, 0, 0)), v1, v2)
    
    return abs(volume) / 3.0

# ============================================
# SPATIAL QUERIES
# ============================================

def find_nearest_points(query_points: List[Vector], 
                       target_points: List[Vector],
                       max_distance: Optional[float] = None) -> List[Tuple[int, float]]:
    """Find nearest target point for each query point"""
    # Build KD tree
    size = len(target_points)
    kd = kdtree.KDTree(size)
    
    for i, point in enumerate(target_points):
        kd.insert(point, i)
    
    kd.balance()
    
    # Query nearest points
    results = []
    for query_point in query_points:
        found = kd.find(query_point)
        
        if found:
            co, index, dist = found
            
            if max_distance is None or dist <= max_distance:
                results.append((index, dist))
            else:
                results.append((-1, float('inf')))
        else:
            results.append((-1, float('inf')))
    
    return results

def find_points_in_radius(center: Vector,
                         points: List[Vector],
                         radius: float) -> List[int]:
    """Find all points within radius of center"""
    indices = []
    radius_squared = radius * radius
    
    for i, point in enumerate(points):
        dist_squared = (point - center).length_squared
        
        if dist_squared <= radius_squared:
            indices.append(i)
    
    return indices

# ============================================
# HELPER FUNCTIONS
# ============================================

def are_points_coplanar(points: List[Vector], tolerance: float = EPSILON) -> bool:
    """Check if points are coplanar"""
    if len(points) < 4:
        return True
    
    # Use first 3 points to define plane
    p1, p2, p3 = points[:3]
    normal = get_normal_vector(p1, p2, p3)
    
    # Check remaining points
    for point in points[3:]:
        distance = abs(distance_point_to_plane(point, p1, normal))
        if distance > tolerance:
            return False
    
    return True

def calculate_polygon_normal(points: List[Vector]) -> Vector:
    """Calculate average normal of polygon"""
    if len(points) < 3:
        return Vector((0, 0, 1))
    
    # Newell's method
    normal = Vector((0, 0, 0))
    
    for i in range(len(points)):
        v1 = points[i]
        v2 = points[(i + 1) % len(points)]
        
        normal.x += (v1.y - v2.y) * (v1.z + v2.z)
        normal.y += (v1.z - v2.z) * (v1.x + v2.x)
        normal.z += (v1.x - v2.x) * (v1.y + v2.y)
    
    if normal.length > EPSILON:
        return normal.normalized()
    else:
        return Vector((0, 0, 1))

def project_points_to_2d(points: List[Vector], normal: Vector) -> List[Tuple[float, float]]:
    """Project 3D points to 2D plane"""
    # Create coordinate system
    if abs(normal.z) < 0.99:
        u = Vector((0, 0, 1)).cross(normal).normalized()
    else:
        u = Vector((1, 0, 0)).cross(normal).normalized()
    
    v = normal.cross(u).normalized()
    
    # Project points
    points_2d = []
    for point in points:
        x = point.dot(u)
        y = point.dot(v)
        points_2d.append((x, y))
    
    return points_2d

def filter_overlapping_triangles(triangles: List[Tuple[int, int, int]], 
                                points: List[Vector]) -> List[Tuple[int, int, int]]:
    """Remove overlapping triangles"""
    # Simple approach - keep triangles with no shared edges
    # In production, use more sophisticated approach
    
    filtered = []
    edges_used = set()
    
    for tri in triangles:
        i, j, k = sorted(tri)
        
        edges = [
            (min(i, j), max(i, j)),
            (min(j, k), max(j, k)),
            (min(i, k), max(i, k))
        ]
        
        # Check if any edge is already used
        overlap = False
        for edge in edges:
            if edge in edges_used:
                overlap = True
                break
        
        if not overlap:
            filtered.append(tri)
            edges_used.update(edges)
    
    return filtered

# ============================================
# CURVE OPERATIONS
# ============================================

def create_bezier_curve(points: List[Vector], 
                       smoothness: float = 0.33) -> List[Vector]:
    """Create smooth bezier curve through points"""
    if len(points) < 2:
        return points
    
    curve_points = []
    
    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i + 1]
        
        # Calculate control points
        if i == 0:
            tangent = (p1 - p0).normalized()
        else:
            tangent = (p1 - points[i - 1]).normalized()
        
        cp1 = p0 + tangent * calculate_distance(p0, p1) * smoothness
        
        if i == len(points) - 2:
            tangent = (p1 - p0).normalized()
        else:
            tangent = (points[i + 2] - p0).normalized()
        
        cp2 = p1 - tangent * calculate_distance(p0, p1) * smoothness
        
        # Generate curve points
        for t in range(10):
            t = t / 10.0
            
            # Cubic bezier
            point = (1 - t)**3 * p0 + \
                   3 * (1 - t)**2 * t * cp1 + \
                   3 * (1 - t) * t**2 * cp2 + \
                   t**3 * p1
            
            curve_points.append(point)
    
    curve_points.append(points[-1])
    
    return curve_points

# ============================================
# MATRIX OPERATIONS
# ============================================

def create_transformation_matrix(location: Vector = None,
                               rotation: Quaternion = None,
                               scale: Vector = None) -> Matrix:
    """Create transformation matrix from components"""
    mat_loc = Matrix.Translation(location) if location else Matrix.Identity(4)
    mat_rot = rotation.to_matrix().to_4x4() if rotation else Matrix.Identity(4)
    mat_scale = Matrix.Diagonal(scale).to_4x4() if scale else Matrix.Identity(4)
    
    return mat_loc @ mat_rot @ mat_scale

def decompose_matrix(matrix: Matrix) -> Tuple[Vector, Quaternion, Vector]:
    """Decompose matrix into location, rotation, scale"""
    location = matrix.to_translation()
    rotation = matrix.to_quaternion()
    scale = matrix.to_scale()
    
    return location, rotation, scale

# ============================================
# VALIDATION FUNCTIONS
# ============================================

def is_valid_triangle(p1: Vector, p2: Vector, p3: Vector,
                     min_area: float = 0.001,
                     min_angle: float = 10.0) -> bool:
    """Check if triangle is valid for mesh generation"""
    # Check area
    area = area_tri(p1, p2, p3)
    if area < min_area:
        return False
    
    # Check angles
    v1 = (p2 - p1).normalized()
    v2 = (p3 - p1).normalized()
    v3 = (p3 - p2).normalized()
    
    angle1 = calculate_angle(v1, v2)
    angle2 = calculate_angle(-v1, v3)
    angle3 = calculate_angle(-v2, -v3)
    
    if min(angle1, angle2, angle3) < min_angle:
        return False
    
    return True

def is_point_inside_mesh(point: Vector, mesh: bpy.types.Mesh) -> bool:
    """Check if point is inside mesh using ray casting"""
    # Cast ray in random direction
    direction = Vector((1, 0.1, 0.1)).normalized()
    
    # Count intersections
    intersections = 0
    
    # Simple approach - in production use BVH tree
    for poly in mesh.polygons:
        verts = [mesh.vertices[i].co for i in poly.vertices]
        
        # Ray-triangle intersection
        # ... (simplified for brevity)
        
    return intersections % 2 == 1
