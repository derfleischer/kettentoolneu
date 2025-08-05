"""
Chain Tool V4 - Math Utilities
==============================
Mathematische Hilfsfunktionen für 3D-Berechnungen
"""

import math
import numpy as np
from mathutils import Vector, Matrix, Quaternion
from typing import List, Tuple, Optional, Union

def calculate_angle(v1: Vector, v2: Vector) -> float:
    """
    Calculate angle between two vectors in radians
    
    Args:
        v1: First vector
        v2: Second vector
    
    Returns:
        Angle in radians (0 to π)
    """
    v1_normalized = v1.normalized()
    v2_normalized = v2.normalized()
    dot_product = v1_normalized.dot(v2_normalized)
    # Clamp to avoid numerical errors with acos
    dot_product = max(-1.0, min(1.0, dot_product))
    return math.acos(dot_product)

def calculate_distance(p1: Vector, p2: Vector) -> float:
    """
    Calculate Euclidean distance between two points
    
    Args:
        p1: First point
        p2: Second point
    
    Returns:
        Distance as float
    """
    return (p2 - p1).length

def calculate_distance_3d(p1: Vector, p2: Vector) -> float:
    """
    Alias for calculate_distance (for compatibility)
    """
    return calculate_distance(p1, p2)

def get_normal_vector(v1: Vector, v2: Vector) -> Vector:
    """
    Calculate normal vector from two vectors using cross product
    
    Args:
        v1: First vector
        v2: Second vector
    
    Returns:
        Normalized perpendicular vector
    """
    normal = v1.cross(v2)
    if normal.length > 0.0001:  # Avoid zero-length vectors
        return normal.normalized()
    return Vector((0, 0, 1))  # Default up vector

def interpolate_points(p1: Vector, p2: Vector, factor: float) -> Vector:
    """
    Linear interpolation between two points
    
    Args:
        p1: Start point
        p2: End point
        factor: Interpolation factor (0.0 to 1.0)
    
    Returns:
        Interpolated point
    """
    factor = max(0.0, min(1.0, factor))  # Clamp factor
    return p1.lerp(p2, factor)

def triangulate_points(points: List[Vector]) -> List[Tuple[int, int, int]]:
    """
    Simple triangulation of points using ear clipping
    
    Args:
        points: List of Vector points
    
    Returns:
        List of triangle indices as tuples
    """
    if len(points) < 3:
        return []
    
    if len(points) == 3:
        return [(0, 1, 2)]
    
    # Simple fan triangulation from first point (for convex polygons)
    triangles = []
    for i in range(1, len(points) - 1):
        triangles.append((0, i, i + 1))
    
    return triangles

def safe_normalize(vector: Vector) -> Vector:
    """
    Safely normalize a vector (returns zero vector if input is zero)
    
    Args:
        vector: Vector to normalize
    
    Returns:
        Normalized vector or zero vector
    """
    length = vector.length
    if length > 0.0001:  # Small epsilon to avoid division by zero
        return vector / length
    return Vector((0, 0, 0))

def project_point_on_plane(point: Vector, plane_origin: Vector, plane_normal: Vector) -> Vector:
    """
    Project a point onto a plane
    
    Args:
        point: Point to project
        plane_origin: A point on the plane
        plane_normal: Normal vector of the plane
    
    Returns:
        Projected point
    """
    plane_normal = plane_normal.normalized()
    distance = (point - plane_origin).dot(plane_normal)
    return point - plane_normal * distance

def calculate_barycentric_coords(point: Vector, v1: Vector, v2: Vector, v3: Vector) -> Tuple[float, float, float]:
    """
    Calculate barycentric coordinates of a point in a triangle
    
    Args:
        point: Point to calculate coordinates for
        v1, v2, v3: Triangle vertices
    
    Returns:
        Barycentric coordinates (u, v, w) where u+v+w=1
    """
    # Vectors from v1
    v0 = v3 - v1
    v1_vec = v2 - v1
    v2_vec = point - v1
    
    # Calculate dot products
    dot00 = v0.dot(v0)
    dot01 = v0.dot(v1_vec)
    dot02 = v0.dot(v2_vec)
    dot11 = v1_vec.dot(v1_vec)
    dot12 = v1_vec.dot(v2_vec)
    
    # Calculate barycentric coordinates
    inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    
    return (1 - u - v, v, u)

def rotate_vector(vector: Vector, axis: Vector, angle: float) -> Vector:
    """
    Rotate a vector around an axis by given angle
    
    Args:
        vector: Vector to rotate
        axis: Rotation axis (will be normalized)
        angle: Rotation angle in radians
    
    Returns:
        Rotated vector
    """
    rotation = Quaternion(axis.normalized(), angle)
    return rotation @ vector

def calculate_circle_points(center: Vector, radius: float, normal: Vector, segments: int = 32) -> List[Vector]:
    """
    Calculate points on a circle in 3D space
    
    Args:
        center: Circle center
        radius: Circle radius
        normal: Normal vector defining circle plane
        segments: Number of points
    
    Returns:
        List of points on the circle
    """
    points = []
    
    # Create orthonormal basis
    normal = normal.normalized()
    
    # Find a perpendicular vector
    if abs(normal.z) < 0.9:
        tangent = Vector((0, 0, 1)).cross(normal).normalized()
    else:
        tangent = Vector((1, 0, 0)).cross(normal).normalized()
    
    bitangent = normal.cross(tangent)
    
    # Generate circle points
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        point = center + radius * (math.cos(angle) * tangent + math.sin(angle) * bitangent)
        points.append(point)
    
    return points

def fit_plane_to_points(points: List[Vector]) -> Tuple[Vector, Vector]:
    """
    Fit a plane to a set of points using least squares
    
    Args:
        points: List of points
    
    Returns:
        Tuple of (plane_origin, plane_normal)
    """
    if len(points) < 3:
        return Vector((0, 0, 0)), Vector((0, 0, 1))
    
    # Calculate centroid
    centroid = sum(points, Vector()) / len(points)
    
    # Build covariance matrix
    xx = yy = zz = xy = xz = yz = 0
    for p in points:
        r = p - centroid
        xx += r.x * r.x
        yy += r.y * r.y
        zz += r.z * r.z
        xy += r.x * r.y
        xz += r.x * r.z
        yz += r.y * r.z
    
    # Solve for normal (eigenvector with smallest eigenvalue)
    det_x = yy * zz - yz * yz
    det_y = xx * zz - xz * xz
    det_z = xx * yy - xy * xy
    
    if det_x > det_y and det_x > det_z:
        normal = Vector((det_x, xz * yz - xy * zz, xy * yz - xz * yy))
    elif det_y > det_z:
        normal = Vector((xz * yz - xy * zz, det_y, xy * xz - yz * xx))
    else:
        normal = Vector((xy * yz - xz * yy, xy * xz - yz * xx, det_z))
    
    if normal.length > 0.0001:
        normal = normal.normalized()
    else:
        normal = Vector((0, 0, 1))
    
    return centroid, normal

# Export all functions
__all__ = [
    'calculate_angle',
    'calculate_distance',
    'calculate_distance_3d',
    'get_normal_vector',
    'interpolate_points',
    'triangulate_points',
    'safe_normalize',
    'project_point_on_plane',
    'calculate_barycentric_coords',
    'rotate_vector',
    'calculate_circle_points',
    'fit_plane_to_points',
]
