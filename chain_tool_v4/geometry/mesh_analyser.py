"""
Mesh Analyzer for Chain Tool V4
Comprehensive mesh analysis and BVH operations
"""

import bpy
import bmesh
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree

from utils.debug import debug
from utils.performance import performance
from utils.caching import bvh_cache
from utils.math_utils import (
    calculate_distance, get_normal_vector,
    calculate_bounding_box, get_mesh_surface_area
)

# ============================================
# DATA CLASSES
# ============================================

@dataclass
class MeshData:
    """Comprehensive mesh data container"""
    vertices: List[Vector]
    faces: List[List[int]]
    normals: List[Vector]
    edges: List[Tuple[int, int]]
    
    # Statistics
    vertex_count: int
    face_count: int
    edge_count: int
    
    # Bounds
    bbox_min: Vector
    bbox_max: Vector
    center: Vector
    dimensions: Vector
    
    # Surface properties
    surface_area: float
    volume: float
    is_manifold: bool
    is_closed: bool
    
    # Additional data
    curvature_map: Optional[Dict[int, float]] = None
    thickness_map: Optional[Dict[int, float]] = None
    stress_map: Optional[Dict[int, float]] = None

# ============================================
# MESH ANALYZER
# ============================================

class MeshAnalyzer:
    """Advanced mesh analysis system"""
    
    def __init__(self):
        self.current_mesh = None
        self.current_data = None
        self.bvh_tree = None
        
    @performance.track_function('GEOMETRY')
    def analyze(self, obj: bpy.types.Object, 
                calculate_curvature: bool = False,
                calculate_thickness: bool = False) -> MeshData:
        """Perform comprehensive mesh analysis"""
        
        if obj.type != 'MESH':
            raise ValueError(f"Object {obj.name} is not a mesh")
            
        debug.info('GEOMETRY', f"Analyzing mesh: {obj.name}")
        
        # Get or create BVH tree
        self.bvh_tree, matrix = bvh_cache.get_or_create(obj)
        
        # Extract mesh data
        mesh = obj.data
        mesh.calc_loop_triangles()
        
        # Get vertices in world space
        vertices = [matrix @ v.co for v in mesh.vertices]
        
        # Get faces
        faces = [[v for v in poly.vertices] for poly in mesh.polygons]
        
        # Get normals
        normals = [(matrix.to_3x3() @ poly.normal).normalized() 
                   for poly in mesh.polygons]
        
        # Get edges
        edges = [(e.vertices[0], e.vertices[1]) for e in mesh.edges]
        
        # Calculate bounds
        bbox_min, bbox_max = calculate_bounding_box(vertices)
        center = (bbox_min + bbox_max) / 2
        dimensions = bbox_max - bbox_min
        
        # Calculate surface properties
        surface_area = get_mesh_surface_area(mesh)
        volume = self._calculate_volume(mesh, matrix)
        
        # Check topology
        is_manifold, is_closed = self._check_topology(mesh)
        
        # Create mesh data
        self.current_data = MeshData(
            vertices=vertices,
            faces=faces,
            normals=normals,
            edges=edges,
            vertex_count=len(vertices),
            face_count=len(faces),
            edge_count=len(edges),
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            center=center,
            dimensions=dimensions,
            surface_area=surface_area,
            volume=volume,
            is_manifold=is_manifold,
            is_closed=is_closed
        )
        
        # Optional calculations
        if calculate_curvature:
            self.current_data.curvature_map = self._calculate_curvature(mesh)
            
        if calculate_thickness:
            self.current_data.thickness_map = self._calculate_thickness(mesh)
        
        debug.debug('GEOMETRY', 
                   f"Analysis complete: {len(vertices)} verts, {len(faces)} faces")
        
        return self.current_data
    
    def find_nearest_point(self, point: Vector) -> Tuple[Vector, Vector, int, float]:
        """Find nearest point on mesh surface"""
        if not self.bvh_tree:
            raise RuntimeError("No mesh analyzed")
            
        result = self.bvh_tree.find_nearest(point)
        
        if result:
            location, normal, index, distance = result
            return location, normal, index, distance
        else:
            return None, None, -1, float('inf')
    
    def ray_cast(self, origin: Vector, direction: Vector) -> Optional[Tuple[Vector, Vector, int, float]]:
        """Cast ray onto mesh"""
        if not self.bvh_tree:
            raise RuntimeError("No mesh analyzed")
            
        result = self.bvh_tree.ray_cast(origin, direction)
        
        if result:
            location, normal, index, distance = result
            return location, normal, index, distance
        else:
            return None
    
    def sample_surface(self, count: int, method: str = 'RANDOM') -> List[Vector]:
        """Sample points on mesh surface"""
        if not self.current_data:
            raise RuntimeError("No mesh analyzed")
            
        if method == 'RANDOM':
            return self._sample_random(count)
        elif method == 'UNIFORM':
            return self._sample_uniform(count)
        elif method == 'CURVATURE':
            return self._sample_by_curvature(count)
        else:
            raise ValueError(f"Unknown sampling method: {method}")
    
    def get_edge_loops(self) -> List[List[int]]:
        """Find all edge loops in mesh"""
        if not self.current_data:
            raise RuntimeError("No mesh analyzed")
            
        # Create bmesh for edge analysis
        bm = bmesh.new()
        bm.verts.extend([bm.verts.new(v) for v in self.current_data.vertices])
        
        for face_verts in self.current_data.faces:
            if len(face_verts) >= 3:
                bm.faces.new([bm.verts[i] for i in face_verts])
        
        bm.edges.ensure_lookup_table()
        
        # Find edge loops
        loops = []
        visited_edges = set()
        
        for edge in bm.edges:
            if edge.index in visited_edges or not edge.is_boundary:
                continue
                
            loop = self._trace_edge_loop(edge, bm, visited_edges)
            if len(loop) > 2:
                loops.append(loop)
        
        bm.free()
        
        return loops
    
    def calculate_geodesic_distance(self, start_idx: int, end_idx: int) -> float:
        """Calculate geodesic distance between vertices"""
        # Simplified version - in production use heat method or exact geodesics
        start = self.current_data.vertices[start_idx]
        end = self.current_data.vertices[end_idx]
        
        return calculate_distance(start, end) * 1.2  # Approximation
    
    # ============================================
    # PRIVATE METHODS
    # ============================================
    
    def _calculate_volume(self, mesh: bpy.types.Mesh, matrix: Matrix) -> float:
        """Calculate mesh volume"""
        volume = 0.0
        
        for poly in mesh.polygons:
            # Get face vertices
            verts = [matrix @ mesh.vertices[i].co for i in poly.vertices]
            
            # Triangulate if needed
            if len(verts) == 3:
                # Single triangle
                volume += self._tetrahedron_volume(Vector((0, 0, 0)), verts[0], verts[1], verts[2])
            else:
                # Triangulate from centroid
                centroid = sum(verts, Vector()) / len(verts)
                for i in range(len(verts)):
                    j = (i + 1) % len(verts)
                    volume += self._tetrahedron_volume(Vector((0, 0, 0)), centroid, verts[i], verts[j])
        
        return abs(volume)
    
    def _tetrahedron_volume(self, p0: Vector, p1: Vector, p2: Vector, p3: Vector) -> float:
        """Calculate volume of tetrahedron"""
        v1 = p1 - p0
        v2 = p2 - p0
        v3 = p3 - p0
        
        return abs(v1.dot(v2.cross(v3))) / 6.0
    
    def _check_topology(self, mesh: bpy.types.Mesh) -> Tuple[bool, bool]:
        """Check if mesh is manifold and closed"""
        # Create bmesh for topology check
        bm = bmesh.new()
        bm.from_mesh(mesh)
        
        # Check manifold
        is_manifold = True
        for edge in bm.edges:
            if len(edge.link_faces) > 2:
                is_manifold = False
                break
        
        # Check closed
        is_closed = True
        for edge in bm.edges:
            if len(edge.link_faces) == 1:
                is_closed = False
                break
        
        bm.free()
        
        return is_manifold, is_closed
    
    def _calculate_curvature(self, mesh: bpy.types.Mesh) -> Dict[int, float]:
        """Calculate vertex curvature"""
        curvature = {}
        
        # Simplified Gaussian curvature approximation
        for vert in mesh.vertices:
            # Get connected faces
            connected_faces = [poly for poly in mesh.polygons if vert.index in poly.vertices]
            
            if len(connected_faces) < 3:
                curvature[vert.index] = 0.0
                continue
            
            # Calculate angle deficit
            angle_sum = 0.0
            for face in connected_faces:
                # Find vertex position in face
                vert_pos = list(face.vertices).index(vert.index)
                
                # Get adjacent vertices
                prev_idx = face.vertices[(vert_pos - 1) % len(face.vertices)]
                next_idx = face.vertices[(vert_pos + 1) % len(face.vertices)]
                
                # Calculate angle
                v1 = (mesh.vertices[prev_idx].co - vert.co).normalized()
                v2 = (mesh.vertices[next_idx].co - vert.co).normalized()
                
                angle_sum += np.arccos(np.clip(v1.dot(v2), -1.0, 1.0))
            
            # Gaussian curvature from angle deficit
            curvature[vert.index] = 2 * np.pi - angle_sum
        
        return curvature
    
    def _calculate_thickness(self, mesh: bpy.types.Mesh) -> Dict[int, float]:
        """Calculate local thickness using ray casting"""
        thickness = {}
        
        for vert in mesh.vertices:
            # Cast ray in opposite normal direction
            origin = vert.co + vert.normal * 0.001  # Small offset
            direction = -vert.normal
            
            result = self.ray_cast(origin, direction)
            
            if result and result[0]:
                thickness[vert.index] = result[3]
            else:
                thickness[vert.index] = float('inf')
        
        return thickness
    
    def _sample_random(self, count: int) -> List[Vector]:
        """Random surface sampling"""
        import random
        
        points = []
        faces = self.current_data.faces
        vertices = self.current_data.vertices
        
        # Calculate face areas for weighted sampling
        face_areas = []
        total_area = 0.0
        
        for face in faces:
            if len(face) >= 3:
                # Calculate face area
                v0 = vertices[face[0]]
                v1 = vertices[face[1]]
                v2 = vertices[face[2]]
                
                area = 0.5 * (v1 - v0).cross(v2 - v0).length
                face_areas.append(area)
                total_area += area
            else:
                face_areas.append(0.0)
        
        # Sample points
        for _ in range(count):
            # Pick face weighted by area
            r = random.uniform(0, total_area)
            cumulative = 0.0
            face_idx = 0
            
            for i, area in enumerate(face_areas):
                cumulative += area
                if cumulative >= r:
                    face_idx = i
                    break
            
            # Sample point on face
            face = faces[face_idx]
            if len(face) >= 3:
                # Barycentric coordinates
                r1 = random.random()
                r2 = random.random()
                
                if r1 + r2 > 1:
                    r1 = 1 - r1
                    r2 = 1 - r2
                
                v0 = vertices[face[0]]
                v1 = vertices[face[1]]
                v2 = vertices[face[2]]
                
                point = v0 + r1 * (v1 - v0) + r2 * (v2 - v0)
                points.append(point)
        
        return points
    
    def _sample_uniform(self, count: int) -> List[Vector]:
        """Uniform surface sampling using Poisson disk"""
        # Simplified version - in production use proper Poisson disk sampling
        points = self._sample_random(count * 2)
        
        # Filter to maintain minimum distance
        filtered = []
        min_dist = (self.current_data.surface_area / count) ** 0.5
        
        for point in points:
            too_close = False
            for existing in filtered:
                if calculate_distance(point, existing) < min_dist:
                    too_close = True
                    break
            
            if not too_close:
                filtered.append(point)
                
            if len(filtered) >= count:
                break
        
        return filtered[:count]
    
    def _sample_by_curvature(self, count: int) -> List[Vector]:
        """Sample more points in high curvature areas"""
        if not self.current_data.curvature_map:
            return self._sample_random(count)
        
        # Weight sampling by curvature
        # Implementation depends on specific requirements
        return self._sample_random(count)
    
    def _trace_edge_loop(self, start_edge, bm, visited_edges: set) -> List[int]:
        """Trace a boundary edge loop"""
        loop = []
        current_edge = start_edge
        
        while current_edge and current_edge.index not in visited_edges:
            visited_edges.add(current_edge.index)
            
            # Add vertex to loop
            loop.append(current_edge.verts[0].index)
            
            # Find next boundary edge
            next_edge = None
            for edge in current_edge.verts[1].link_edges:
                if edge.is_boundary and edge.index not in visited_edges:
                    next_edge = edge
                    break
            
            current_edge = next_edge
            
            # Check if we completed the loop
            if current_edge == start_edge:
                break
        
        return loop

# ============================================
# UTILITY FUNCTIONS
# ============================================

def analyze_mesh(obj: bpy.types.Object, detailed: bool = False) -> MeshData:
    """Quick mesh analysis"""
    analyzer = MeshAnalyzer()
    return analyzer.analyze(obj, 
                           calculate_curvature=detailed,
                           calculate_thickness=detailed)

def get_mesh_statistics(obj: bpy.types.Object) -> Dict[str, Any]:
    """Get basic mesh statistics"""
    data = analyze_mesh(obj, detailed=False)
    
    return {
        'vertices': data.vertex_count,
        'faces': data.face_count,
        'edges': data.edge_count,
        'surface_area': data.surface_area,
        'volume': data.volume,
        'dimensions': tuple(data.dimensions),
        'is_manifold': data.is_manifold,
        'is_closed': data.is_closed
    }
