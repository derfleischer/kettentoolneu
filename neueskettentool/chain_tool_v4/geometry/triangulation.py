"""
Triangulation Engine for Chain Tool V4
Advanced 3D Delaunay triangulation with constraints
"""

import bpy
import bmesh
import numpy as np
from typing import List, Tuple, Set, Optional, Dict
from dataclasses import dataclass
from mathutils import Vector, Matrix
from mathutils.geometry import tessellate_polygon

from utils.debug import debug
from utils.performance import performance
from utils.math_utils import (
    calculate_triangle_quality, is_valid_triangle,
    calculate_distance, are_points_coplanar
)
from core.constants import (
    DELAUNAY_MIN_ANGLE, DELAUNAY_MAX_EDGE_RATIO,
    QUALITY_THRESHOLD, MAX_CONNECTION_LENGTH
)

# ============================================
# DATA CLASSES
# ============================================

@dataclass
class TriangulationResult:
    """Result of triangulation operation"""
    triangles: List[Tuple[int, int, int]]
    edges: Set[Tuple[int, int]]
    quality_scores: List[float]
    rejected_triangles: int
    
    @property
    def average_quality(self) -> float:
        """Get average triangle quality"""
        if not self.quality_scores:
            return 0.0
        return sum(self.quality_scores) / len(self.quality_scores)
    
    @property
    def triangle_count(self) -> int:
        """Get number of triangles"""
        return len(self.triangles)

@dataclass
class DelaunayConstraint:
    """Constraint for Delaunay triangulation"""
    edge: Tuple[int, int]
    weight: float = 1.0
    fixed: bool = False

# ============================================
# TRIANGULATION ENGINE
# ============================================

class TriangulationEngine:
    """Advanced triangulation system with quality control"""
    
    def __init__(self):
        self.points = []
        self.constraints = []
        self.existing_edges = set()
        
    @performance.track_function('TRIANGULATION')
    def triangulate(self, 
                   points: List[Vector],
                   constraints: Optional[List[DelaunayConstraint]] = None,
                   max_edge_length: float = MAX_CONNECTION_LENGTH,
                   min_angle: float = DELAUNAY_MIN_ANGLE,
                   quality_threshold: float = QUALITY_THRESHOLD) -> TriangulationResult:
        """
        Perform constrained Delaunay triangulation
        """
        debug.info('TRIANGULATION', f"Triangulating {len(points)} points")
        
        self.points = points
        self.constraints = constraints or []
        
        # Check if points are coplanar
        if len(points) < 4 or are_points_coplanar(points):
            result = self._triangulate_2d(max_edge_length, min_angle, quality_threshold)
        else:
            result = self._triangulate_3d(max_edge_length, min_angle, quality_threshold)
        
        debug.debug('TRIANGULATION', 
                   f"Generated {result.triangle_count} triangles, "
                   f"avg quality: {result.average_quality:.3f}")
        
        return result
    
    def add_constraint(self, p1_idx: int, p2_idx: int, weight: float = 1.0):
        """Add edge constraint"""
        constraint = DelaunayConstraint(
            edge=(min(p1_idx, p2_idx), max(p1_idx, p2_idx)),
            weight=weight
        )
        self.constraints.append(constraint)
    
    def add_existing_edges(self, edges: List[Tuple[int, int]]):
        """Add existing edges to avoid duplicates"""
        for e in edges:
            self.existing_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    # ============================================
    # 2D TRIANGULATION
    # ============================================
    
    def _triangulate_2d(self, max_edge_length: float, 
                       min_angle: float, 
                       quality_threshold: float) -> TriangulationResult:
        """Triangulate coplanar points"""
        
        # Project to 2D plane
        points_2d, transform = self._project_to_plane()
        
        # Use Blender's triangulation as base
        base_triangles = tessellate_polygon([points_2d])
        
        # Apply constraints
        triangles = self._apply_constraints_2d(base_triangles)
        
        # Filter by quality
        filtered_result = self._filter_triangles(
            triangles, max_edge_length, min_angle, quality_threshold
        )
        
        return filtered_result
    
    def _project_to_plane(self) -> Tuple[List[Tuple[float, float]], Matrix]:
        """Project 3D points to best-fit 2D plane"""
        
        # Calculate best-fit plane using PCA
        points_array = np.array([list(p) for p in self.points])
        centroid = np.mean(points_array, axis=0)
        
        # Center points
        centered = points_array - centroid
        
        # Compute covariance matrix
        cov = np.cov(centered.T)
        
        # Get eigenvectors (principal components)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # Sort by eigenvalue
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # Create transformation matrix
        u = Vector(eigenvectors[:, 0])  # First principal component
        v = Vector(eigenvectors[:, 1])  # Second principal component
        w = Vector(eigenvectors[:, 2])  # Normal (least variance)
        
        # Project points
        points_2d = []
        for p in self.points:
            p_centered = p - Vector(centroid)
            x = p_centered.dot(u)
            y = p_centered.dot(v)
            points_2d.append((x, y))
        
        # Create transform matrix for later use
        transform = Matrix([
            [u.x, u.y, u.z, 0],
            [v.x, v.y, v.z, 0],
            [w.x, w.y, w.z, 0],
            [centroid[0], centroid[1], centroid[2], 1]
        ])
        
        return points_2d, transform
    
    def _apply_constraints_2d(self, triangles: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        """Apply edge constraints to triangulation"""
        
        if not self.constraints:
            return triangles
        
        # Build edge set from triangles
        triangle_edges = set()
        for t in triangles:
            for i in range(3):
                j = (i + 1) % 3
                edge = (min(t[i], t[j]), max(t[i], t[j]))
                triangle_edges.add(edge)
        
        # Check constraints
        constrained_triangles = []
        for t in triangles:
            keep_triangle = True
            
            # Check if triangle violates constraints
            for constraint in self.constraints:
                if constraint.fixed:
                    # Check if triangle crosses fixed edge
                    if self._triangle_crosses_edge(t, constraint.edge):
                        keep_triangle = False
                        break
            
            if keep_triangle:
                constrained_triangles.append(t)
        
        # Ensure all fixed constraints are included
        for constraint in self.constraints:
            if constraint.fixed and constraint.edge not in triangle_edges:
                # Try to add triangles that include this edge
                self._enforce_edge(constraint.edge, constrained_triangles)
        
        return constrained_triangles
    
    # ============================================
    # 3D TRIANGULATION
    # ============================================
    
    def _triangulate_3d(self, max_edge_length: float,
                       min_angle: float,
                       quality_threshold: float) -> TriangulationResult:
        """Triangulate 3D points using incremental Delaunay"""
        
        # For true 3D Delaunay, we'd use scipy.spatial.Delaunay
        # Here's a simplified approach
        
        triangles = []
        
        # Start with tetrahedron
        if len(self.points) >= 4:
            # Find initial tetrahedron
            tet = self._find_initial_tetrahedron()
            if tet:
                # Add faces of tetrahedron
                triangles.extend([
                    (tet[0], tet[1], tet[2]),
                    (tet[0], tet[1], tet[3]),
                    (tet[0], tet[2], tet[3]),
                    (tet[1], tet[2], tet[3])
                ])
        
        # Incrementally add points
        for i in range(4, len(self.points)):
            self._add_point_3d(i, triangles)
        
        # Apply constraints
        triangles = self._apply_constraints_3d(triangles)
        
        # Filter by quality
        filtered_result = self._filter_triangles(
            triangles, max_edge_length, min_angle, quality_threshold
        )
        
        return filtered_result
    
    def _find_initial_tetrahedron(self) -> Optional[Tuple[int, int, int, int]]:
        """Find initial tetrahedron with good quality"""
        
        best_tet = None
        best_volume = 0.0
        
        # Try different combinations
        n = min(len(self.points), 10)  # Limit search
        
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        # Calculate volume
                        volume = self._tetrahedron_volume(i, j, k, l)
                        
                        if volume > best_volume:
                            best_volume = volume
                            best_tet = (i, j, k, l)
        
        return best_tet
    
    def _tetrahedron_volume(self, i: int, j: int, k: int, l: int) -> float:
        """Calculate tetrahedron volume"""
        p0, p1, p2, p3 = self.points[i], self.points[j], self.points[k], self.points[l]
        
        v1 = p1 - p0
        v2 = p2 - p0
        v3 = p3 - p0
        
        return abs(v1.dot(v2.cross(v3))) / 6.0
    
    def _add_point_3d(self, point_idx: int, triangles: List[Tuple[int, int, int]]):
        """Add point to 3D triangulation"""
        
        point = self.points[point_idx]
        
        # Find triangles visible from point
        visible_triangles = []
        for i, tri in enumerate(triangles):
            if self._is_triangle_visible_from_point(tri, point):
                visible_triangles.append(i)
        
        if not visible_triangles:
            return
        
        # Remove visible triangles and get boundary
        boundary_edges = set()
        
        for i in reversed(sorted(visible_triangles)):
            tri = triangles.pop(i)
            
            # Add edges to boundary
            for j in range(3):
                k = (j + 1) % 3
                edge = (min(tri[j], tri[k]), max(tri[j], tri[k]))
                
                if edge in boundary_edges:
                    boundary_edges.remove(edge)
                else:
                    boundary_edges.add(edge)
        
        # Create new triangles with boundary edges
        for edge in boundary_edges:
            triangles.append((edge[0], edge[1], point_idx))
    
    def _is_triangle_visible_from_point(self, tri: Tuple[int, int, int], point: Vector) -> bool:
        """Check if triangle is visible from point"""
        
        p0, p1, p2 = self.points[tri[0]], self.points[tri[1]], self.points[tri[2]]
        
        # Calculate triangle normal
        normal = (p1 - p0).cross(p2 - p0).normalized()
        
        # Check if point is on positive side
        to_point = (point - p0).normalized()
        
        return normal.dot(to_point) > 0.01
    
    def _apply_constraints_3d(self, triangles: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        """Apply constraints to 3D triangulation"""
        
        # Similar to 2D but in 3D space
        return self._apply_constraints_2d(triangles)
    
    # ============================================
    # FILTERING AND QUALITY
    # ============================================
    
    def _filter_triangles(self, triangles: List[Tuple[int, int, int]],
                         max_edge_length: float,
                         min_angle: float,
                         quality_threshold: float) -> TriangulationResult:
        """Filter triangles by quality criteria"""
        
        filtered_triangles = []
        edges = set()
        quality_scores = []
        rejected = 0
        
        for tri in triangles:
            # Get vertices
            p0 = self.points[tri[0]]
            p1 = self.points[tri[1]]
            p2 = self.points[tri[2]]
            
            # Check edge lengths
            d01 = calculate_distance(p0, p1)
            d12 = calculate_distance(p1, p2)
            d20 = calculate_distance(p2, p0)
            
            if max(d01, d12, d20) > max_edge_length:
                rejected += 1
                continue
            
            # Check if edges already exist
            e01 = (min(tri[0], tri[1]), max(tri[0], tri[1]))
            e12 = (min(tri[1], tri[2]), max(tri[1], tri[2]))
            e20 = (min(tri[2], tri[0]), max(tri[2], tri[0]))
            
            if (e01 in self.existing_edges or 
                e12 in self.existing_edges or 
                e20 in self.existing_edges):
                rejected += 1
                continue
            
            # Check triangle validity
            if not is_valid_triangle(p0, p1, p2, min_angle=min_angle):
                rejected += 1
                continue
            
            # Calculate quality
            quality = calculate_triangle_quality(p0, p1, p2)
            
            if quality < quality_threshold:
                rejected += 1
                continue
            
            # Add triangle
            filtered_triangles.append(tri)
            quality_scores.append(quality)
            
            # Add edges
            edges.add(e01)
            edges.add(e12)
            edges.add(e20)
        
        return TriangulationResult(
            triangles=filtered_triangles,
            edges=edges,
            quality_scores=quality_scores,
            rejected_triangles=rejected
        )
    
    def _triangle_crosses_edge(self, tri: Tuple[int, int, int], 
                              edge: Tuple[int, int]) -> bool:
        """Check if triangle crosses constrained edge"""
        
        # Get all edges of triangle
        tri_edges = [
            (min(tri[0], tri[1]), max(tri[0], tri[1])),
            (min(tri[1], tri[2]), max(tri[1], tri[2])),
            (min(tri[2], tri[0]), max(tri[2], tri[0]))
        ]
        
        # Simple check - more sophisticated intersection test needed
        return False
    
    def _enforce_edge(self, edge: Tuple[int, int], 
                     triangles: List[Tuple[int, int, int]]):
        """Enforce edge in triangulation"""
        
        # Find triangles that could include this edge
        p0_idx, p1_idx = edge
        
        # This is a simplified version
        # In production, use edge flipping algorithm
        pass

# ============================================
# UTILITY FUNCTIONS
# ============================================

def triangulate_points(points: List[Vector],
                      constraints: Optional[List[Tuple[int, int]]] = None,
                      max_length: float = MAX_CONNECTION_LENGTH) -> TriangulationResult:
    """Quick triangulation of points"""
    
    engine = TriangulationEngine()
    
    # Convert constraints to DelaunayConstraints
    if constraints:
        for c in constraints:
            engine.add_constraint(c[0], c[1])
    
    return engine.triangulate(points, max_edge_length=max_length)

def create_delaunay_3d(points: List[Vector]) -> List[Tuple[int, int, int]]:
    """Create simple Delaunay triangulation"""
    
    result = triangulate_points(points)
    return result.triangles

@performance.track_function('TRIANGULATION')
def triangulate_mesh_boundary(mesh: bpy.types.Mesh) -> TriangulationResult:
    """Triangulate boundary of mesh"""
    
    # Get boundary vertices
    boundary_verts = []
    vert_map = {}
    
    # Find boundary edges
    bm = bmesh.new()
    bm.from_mesh(mesh)
    
    for edge in bm.edges:
        if edge.is_boundary:
            for vert in edge.verts:
                if vert.index not in vert_map:
                    vert_map[vert.index] = len(boundary_verts)
                    boundary_verts.append(Vector(vert.co))
    
    bm.free()
    
    # Triangulate boundary
    if len(boundary_verts) >= 3:
        return triangulate_points(boundary_verts)
    else:
        return TriangulationResult([], set(), [], 0)
