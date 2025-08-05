"""
Chain Tool V4 - Mathematical Utilities
======================================

Mathematical helper functions for geometry processing, vector operations,
and numerical computations used throughout Chain Tool V4.
"""

import bpy
import bmesh
import math
from mathutils import Vector, Matrix, Quaternion, Euler
from typing import List, Tuple, Optional, Union

# Constants
EPSILON = 1e-6
PI = math.pi
TAU = 2 * PI

def safe_normalize(vector: Vector, fallback: Optional[Vector] = None) -> Vector:
    """
    Safely normalize a vector, handling zero-length vectors
    
    Args:
        vector: Vector to normalize
        fallback: Fallback vector if normalization fails
        
    Returns:
        Normalized vector or fallback
    """
    try:
        if vector.length < EPSILON:
            return fallback or Vector((0, 0, 1))
        return vector.normalized()
    except:
        return fallback or Vector((0, 0, 1))

def calculate_distance_3d(point1: Vector, point2: Vector) -> float:
    """Calculate 3D distance between two points"""
    return (point2 - point1).length

def calculate_distance_2d(point1: Vector, point2: Vector) -> float:
    """Calculate 2D distance between two points (ignoring Z)"""
    dx = point2.x - point1.x
    dy = point2.y - point1.y
    return math.sqrt(dx*dx + dy*dy)

def point_in_triangle(point: Vector, tri_a: Vector, tri_b: Vector, tri_c: Vector) -> bool:
    """
    Check if point is inside triangle using barycentric coordinates
    
    Args:
        point: Point to test
        tri_a, tri_b, tri_c: Triangle vertices
        
    Returns:
        True if point is inside triangle
    """
    v0 = tri_c - tri_a
    v1 = tri_b - tri_a
    v2 = point - tri_a
    
    dot00 = v0.dot(v0)
    dot01 = v0.dot(v1)
    dot02 = v0.dot(v2)
    dot11 = v1.dot(v1)
    dot12 = v1.dot(v2)
    
    inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    
    return (u >= 0) and (v >= 0) and (u + v <= 1)

def calculate_triangle_area(a: Vector, b: Vector, c: Vector) -> float:
    """Calculate area of triangle from three vertices"""
    return 0.5 * (b - a).cross(c - a).length

def calculate_triangle_normal(a: Vector, b: Vector, c: Vector) -> Vector:
    """Calculate normal vector of triangle"""
    return safe_normalize((b - a).cross(c - a))

def calculate_triangle_quality(a: Vector, b: Vector, c: Vector) -> float:
    """
    Calculate triangle quality metric (0.0 = degenerate, 1.0 = equilateral)
    
    Uses ratio of inscribed circle radius to circumscribed circle radius
    """
    # Edge lengths
    edge_a = (b - c).length
    edge_b = (c - a).length  
    edge_c = (a - b).length
    
    if edge_a < EPSILON or edge_b < EPSILON or edge_c < EPSILON:
        return 0.0  # Degenerate triangle
    
    # Semi-perimeter
    s = (edge_a + edge_b + edge_c) * 0.5
    
    # Area using Heron's formula
    area_squared = s * (s - edge_a) * (s - edge_b) * (s - edge_c)
    if area_squared <= 0:
        return 0.0
    
    area = math.sqrt(area_squared)
    
    # Inscribed circle radius
    r_inscribed = area / s
    
    # Circumscribed circle radius
    r_circumscribed = (edge_a * edge_b * edge_c) / (4 * area)
    
    if r_circumscribed < EPSILON:
        return 0.0
    
    # Quality ratio (0 to 1, where 1 is perfect)
    quality = 2 * r_inscribed / r_circumscribed
    
    return min(1.0, max(0.0, quality))

def calculate_angle_between_vectors(v1: Vector, v2: Vector) -> float:
    """Calculate angle between two vectors in radians"""
    v1_norm = safe_normalize(v1)
    v2_norm = safe_normalize(v2)
    
    dot_product = max(-1.0, min(1.0, v1_norm.dot(v2_norm)))
    return math.acos(dot_product)

def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between two values"""
    return a + (b - a) * t

def lerp_vector(a: Vector, b: Vector, t: float) -> Vector:
    """Linear interpolation between two vectors"""
    return a + (b - a) * t

def smooth_step(edge0: float, edge1: float, x: float) -> float:
    """Smooth step interpolation function"""
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)

def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max"""
    return max(min_val, min(max_val, value))

def remap(value: float, old_min: float, old_max: float, new_min: float, new_max: float) -> float:
    """Remap value from one range to another"""
    if abs(old_max - old_min) < EPSILON:
        return new_min
    
    t = (value - old_min) / (old_max - old_min)
    return new_min + t * (new_max - new_min)

def project_point_to_plane(point: Vector, plane_origin: Vector, plane_normal: Vector) -> Vector:
    """Project point onto plane"""
    plane_normal = safe_normalize(plane_normal)
    to_point = point - plane_origin
    distance = to_point.dot(plane_normal)
    return point - plane_normal * distance

def project_point_to_line(point: Vector, line_start: Vector, line_end: Vector) -> Vector:
    """Project point onto line segment"""
    line_vec = line_end - line_start
    line_length_sq = line_vec.length_squared
    
    if line_length_sq < EPSILON:
        return line_start
    
    t = (point - line_start).dot(line_vec) / line_length_sq
    t = clamp(t, 0.0, 1.0)
    
    return line_start + line_vec * t

def closest_point_on_triangle(point: Vector, tri_a: Vector, tri_b: Vector, tri_c: Vector) -> Vector:
    """Find closest point on triangle to given point"""
    # Check if point is inside triangle
    if point_in_triangle(point, tri_a, tri_b, tri_c):
        # Project to triangle plane
        normal = calculate_triangle_normal(tri_a, tri_b, tri_c)
        return project_point_to_plane(point, tri_a, normal)
    
    # Point is outside, find closest point on edges
    closest_on_ab = project_point_to_line(point, tri_a, tri_b)
    closest_on_bc = project_point_to_line(point, tri_b, tri_c)
    closest_on_ca = project_point_to_line(point, tri_c, tri_a)
    
    dist_ab = (point - closest_on_ab).length
    dist_bc = (point - closest_on_bc).length
    dist_ca = (point - closest_on_ca).length
    
    if dist_ab <= dist_bc and dist_ab <= dist_ca:
        return closest_on_ab
    elif dist_bc <= dist_ca:
        return closest_on_bc
    else:
        return closest_on_ca

def generate_random_point_in_triangle(tri_a: Vector, tri_b: Vector, tri_c: Vector) -> Vector:
    """Generate random point inside triangle using barycentric coordinates"""
    import random
    
    # Generate random barycentric coordinates
    r1 = random.random()
    r2 = random.random()
    
    # Ensure point is inside triangle
    if r1 + r2 > 1:
        r1 = 1 - r1
        r2 = 1 - r2
    
    # Convert to cartesian coordinates
    return tri_a + r1 * (tri_b - tri_a) + r2 * (tri_c - tri_a)

def calculate_polygon_area_2d(vertices: List[Vector]) -> float:
    """Calculate area of 2D polygon using shoelace formula"""
    if len(vertices) < 3:
        return 0.0
    
    area = 0.0
    n = len(vertices)
    
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i].x * vertices[j].y
        area -= vertices[j].x * vertices[i].y
    
    return abs(area) * 0.5

def point_in_circle(point: Vector, center: Vector, radius: float) -> bool:
    """Check if point is inside circle"""
    return (point - center).length <= radius

def points_are_collinear(a: Vector, b: Vector, c: Vector, tolerance: float = EPSILON) -> bool:
    """Check if three points are collinear"""
    cross = (b - a).cross(c - a)
    return cross.length < tolerance

def calculate_bounding_box(points: List[Vector]) -> Tuple[Vector, Vector]:
    """Calculate axis-aligned bounding box of points"""
    if not points:
        return Vector((0, 0, 0)), Vector((0, 0, 0))
    
    min_point = points[0].copy()
    max_point = points[0].copy()
    
    for point in points[1:]:
        min_point.x = min(min_point.x, point.x)
        min_point.y = min(min_point.y, point.y)
        min_point.z = min(min_point.z, point.z)
        
        max_point.x = max(max_point.x, point.x)
        max_point.y = max(max_point.y, point.y)
        max_point.z = max(max_point.z, point.z)
    
    return min_point, max_point

def generate_fibonacci_sphere(num_points: int) -> List[Vector]:
    """Generate evenly distributed points on sphere using Fibonacci spiral"""
    points = []
    golden_ratio = (1 + math.sqrt(5)) / 2
    
    for i in range(num_points):
        theta = 2 * PI * i / golden_ratio
        phi = math.acos(1 - 2 * i / num_points)
        
        x = math.sin(phi) * math.cos(theta)
        y = math.sin(phi) * math.sin(theta)
        z = math.cos(phi)
        
        points.append(Vector((x, y, z)))
    
    return points

def create_rotation_matrix(axis: Vector, angle: float) -> Matrix:
    """Create rotation matrix around axis"""
    axis = safe_normalize(axis)
    return Matrix.Rotation(angle, 4, axis)

def matrix_decompose_safe(matrix: Matrix) -> Tuple[Vector, Quaternion, Vector]:
    """Safely decompose matrix into location, rotation, scale"""
    try:
        return matrix.decompose()
    except:
        # Fallback for problematic matrices
        return (
            matrix.translation,
            Quaternion(),
            Vector((1, 1, 1))
        )

def is_matrix_valid(matrix: Matrix) -> bool:
    """Check if matrix is valid (no NaN or infinite values)"""
    for row in matrix:
        for val in row:
            if not math.isfinite(val):
                return False
    return True

# Noise functions
def hash_float(x: float) -> float:
    """Simple hash function for floats"""
    import struct
    return struct.unpack('f', struct.pack('I', hash(x) & 0xffffffff))[0]

def noise_1d(x: float) -> float:
    """Simple 1D noise function"""
    return math.sin(x * 12.9898) * 43758.5453

def noise_2d(x: float, y: float) -> float:
    """Simple 2D noise function"""
    return (math.sin(x * 12.9898 + y * 78.233) * 43758.5453) % 1.0

def smooth_noise_2d(x: float, y: float) -> float:
    """Smoothed 2D noise"""
    corners = (noise_2d(x-1, y-1) + noise_2d(x+1, y-1) + 
               noise_2d(x-1, y+1) + noise_2d(x+1, y+1)) / 16
    sides = (noise_2d(x-1, y) + noise_2d(x+1, y) + 
             noise_2d(x, y-1) + noise_2d(x, y+1)) / 8
    center = noise_2d(x, y) / 4
    
    return corners + sides + center

# Utility classes
class BoundingBox:
    """Axis-aligned bounding box utility class"""
    
    def __init__(self, min_point: Vector = None, max_point: Vector = None):
        if min_point and max_point:
            self.min = min_point.copy()
            self.max = max_point.copy()
        else:
            self.min = Vector((float('inf'), float('inf'), float('inf')))
            self.max = Vector((float('-inf'), float('-inf'), float('-inf')))
    
    def add_point(self, point: Vector):
        """Add point to bounding box"""
        self.min.x = min(self.min.x, point.x)
        self.min.y = min(self.min.y, point.y)
        self.min.z = min(self.min.z, point.z)
        
        self.max.x = max(self.max.x, point.x)
        self.max.y = max(self.max.y, point.y)
        self.max.z = max(self.max.z, point.z)
    
    def add_points(self, points: List[Vector]):
        """Add multiple points to bounding box"""
        for point in points:
            self.add_point(point)
    
    @property
    def center(self) -> Vector:
        """Get center point of bounding box"""
        return (self.min + self.max) * 0.5
    
    @property
    def size(self) -> Vector:
        """Get size of bounding box"""
        return self.max - self.min
    
    @property
    def volume(self) -> float:
        """Get volume of bounding box"""
        size = self.size
        return size.x * size.y * size.z
    
    def contains_point(self, point: Vector) -> bool:
        """Check if point is inside bounding box"""
        return (self.min.x <= point.x <= self.max.x and
                self.min.y <= point.y <= self.max.y and
                self.min.z <= point.z <= self.max.z)
    
    def intersects(self, other: 'BoundingBox') -> bool:
        """Check if this bounding box intersects another"""
        return (self.min.x <= other.max.x and self.max.x >= other.min.x and
                self.min.y <= other.max.y and self.max.y >= other.min.y and
                self.min.z <= other.max.z and self.max.z >= other.min.z)
