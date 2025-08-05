"""
Chain Tool V4 - Mesh Analyzer
=============================
Mesh-Analyse und BVH-Tree Verwaltung für Pattern-Generierung
"""

import bpy
import bmesh
import numpy as np
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from typing import List, Tuple, Dict, Optional, Any, Set
from dataclasses import dataclass, field

@dataclass
class MeshData:
    """Container for analyzed mesh data"""
    vertices: List[Vector] = field(default_factory=list)
    edges: List[Tuple[int, int]] = field(default_factory=list)
    faces: List[List[int]] = field(default_factory=list)
    normals: List[Vector] = field(default_factory=list)
    vertex_normals: List[Vector] = field(default_factory=list)
    
    # Additional analysis data
    bounds_min: Vector = field(default_factory=lambda: Vector((0, 0, 0)))
    bounds_max: Vector = field(default_factory=lambda: Vector((0, 0, 0)))
    center: Vector = field(default_factory=lambda: Vector((0, 0, 0)))
    surface_area: float = 0.0
    volume: float = 0.0
    
    # Edge analysis
    edge_loops: List[List[int]] = field(default_factory=list)
    boundary_edges: Set[Tuple[int, int]] = field(default_factory=set)
    sharp_edges: Set[Tuple[int, int]] = field(default_factory=set)

class MeshAnalyzer:
    """Comprehensive mesh analysis for pattern generation"""
    
    def __init__(self, obj: Optional[bpy.types.Object] = None):
        self.obj = obj
        self.bvh_tree: Optional[BVHTree] = None
        self.mesh_data = MeshData()
        self.bmesh_instance: Optional[bmesh.types.BMesh] = None
        self._cache_valid = False
    
    def set_object(self, obj: bpy.types.Object) -> None:
        """Set the object to analyze"""
        if obj and obj.type == 'MESH':
            self.obj = obj
            self._cache_valid = False
    
    def analyze(self, obj: Optional[bpy.types.Object] = None, 
                use_modifiers: bool = True) -> Optional[MeshData]:
        """
        Analyze mesh object
        
        Args:
            obj: Object to analyze (uses stored object if None)
            use_modifiers: Apply modifiers before analysis
        
        Returns:
            MeshData object or None if analysis failed
        """
        if obj:
            self.set_object(obj)
        
        if not self.obj or self.obj.type != 'MESH':
            print("MeshAnalyzer: No valid mesh object to analyze")
            return None
        
        # Clear previous data
        self._cleanup()
        
        # Create BMesh
        self.bmesh_instance = bmesh.new()
        
        if use_modifiers:
            # Get mesh with modifiers applied
            depsgraph = bpy.context.evaluated_depsgraph_get()
            eval_obj = self.obj.evaluated_get(depsgraph)
            mesh = eval_obj.to_mesh()
            self.bmesh_instance.from_mesh(mesh)
            eval_obj.to_mesh_clear()
        else:
            # Use base mesh
            self.bmesh_instance.from_mesh(self.obj.data)
        
        # Apply world transform
        self.bmesh_instance.transform(self.obj.matrix_world)
        
        # Ensure lookup tables are available
        self.bmesh_instance.verts.ensure_lookup_table()
        self.bmesh_instance.edges.ensure_lookup_table()
        self.bmesh_instance.faces.ensure_lookup_table()
        
        # Extract basic mesh data
        self._extract_vertices()
        self._extract_edges()
        self._extract_faces()
        self._extract_normals()
        
        # Calculate bounds and center
        self._calculate_bounds()
        
        # Analyze edges
        self._analyze_edges()
        
        # Calculate surface area and volume
        self._calculate_metrics()
        
        # Create BVH tree
        self.bvh_tree = BVHTree.FromBMesh(self.bmesh_instance)
        
        self._cache_valid = True
        
        return self.mesh_data
    
    def _extract_vertices(self) -> None:
        """Extract vertex data from BMesh"""
        self.mesh_data.vertices = [v.co.copy() for v in self.bmesh_instance.verts]
        self.mesh_data.vertex_normals = [v.normal.copy() for v in self.bmesh_instance.verts]
    
    def _extract_edges(self) -> None:
        """Extract edge data from BMesh"""
        self.mesh_data.edges = [
            (e.verts[0].index, e.verts[1].index) 
            for e in self.bmesh_instance.edges
        ]
    
    def _extract_faces(self) -> None:
        """Extract face data from BMesh"""
        self.mesh_data.faces = [
            [v.index for v in f.verts]
            for f in self.bmesh_instance.faces
        ]
    
    def _extract_normals(self) -> None:
        """Extract face normals from BMesh"""
        self.mesh_data.normals = [f.normal.copy() for f in self.bmesh_instance.faces]
    
    def _calculate_bounds(self) -> None:
        """Calculate bounding box and center"""
        if not self.mesh_data.vertices:
            return
        
        # Initialize with first vertex
        min_co = self.mesh_data.vertices[0].copy()
        max_co = self.mesh_data.vertices[0].copy()
        
        # Find bounds
        for v in self.mesh_data.vertices:
            for i in range(3):
                min_co[i] = min(min_co[i], v[i])
                max_co[i] = max(max_co[i], v[i])
        
        self.mesh_data.bounds_min = min_co
        self.mesh_data.bounds_max = max_co
        self.mesh_data.center = (min_co + max_co) / 2
    
    def _analyze_edges(self) -> None:
        """Analyze edge properties"""
        # Find boundary edges
        for edge in self.bmesh_instance.edges:
            if edge.is_boundary:
                self.mesh_data.boundary_edges.add(
                    (edge.verts[0].index, edge.verts[1].index)
                )
            
            # Check for sharp edges (angle threshold)
            if len(edge.link_faces) == 2:
                angle = edge.calc_face_angle()
                if angle > 0.5:  # ~30 degrees
                    self.mesh_data.sharp_edges.add(
                        (edge.verts[0].index, edge.verts[1].index)
                    )
    
    def _calculate_metrics(self) -> None:
        """Calculate surface area and volume"""
        total_area = 0.0
        total_volume = 0.0
        
        for face in self.bmesh_instance.faces:
            total_area += face.calc_area()
            
            # Volume calculation (for closed meshes)
            if len(face.verts) >= 3:
                # Use signed volume of tetrahedron formed with origin
                v0 = face.verts[0].co
                for i in range(1, len(face.verts) - 1):
                    v1 = face.verts[i].co
                    v2 = face.verts[i + 1].co
                    # Signed volume of tetrahedron
                    vol = v0.dot(v1.cross(v2)) / 6.0
                    total_volume += vol
        
        self.mesh_data.surface_area = total_area
        self.mesh_data.volume = abs(total_volume)
    
    def get_bvh_tree(self) -> Optional[BVHTree]:
        """Get BVH tree for ray casting"""
        if not self._cache_valid:
            self.analyze()
        return self.bvh_tree
    
    def ray_cast(self, origin: Vector, direction: Vector) -> Optional[Tuple[Vector, Vector, int]]:
        """
        Cast a ray against the mesh
        
        Args:
            origin: Ray origin
            direction: Ray direction
        
        Returns:
            Tuple of (hit_location, hit_normal, face_index) or None
        """
        if not self.bvh_tree:
            return None
        
        return self.bvh_tree.ray_cast(origin, direction)
    
    def find_nearest(self, point: Vector) -> Optional[Tuple[Vector, Vector, int, float]]:
        """
        Find nearest point on mesh surface
        
        Args:
            point: Query point
        
        Returns:
            Tuple of (location, normal, face_index, distance) or None
        """
        if not self.bvh_tree:
            return None
        
        return self.bvh_tree.find_nearest(point)
    
    def _cleanup(self) -> None:
        """Clean up BMesh data"""
        if self.bmesh_instance:
            self.bmesh_instance.free()
            self.bmesh_instance = None
        
        self.mesh_data = MeshData()
        self.bvh_tree = None
        self._cache_valid = False
    
    def __del__(self):
        """Cleanup on deletion"""
        self._cleanup()

def analyze_mesh(obj: bpy.types.Object, use_modifiers: bool = True) -> Optional[MeshData]:
    """
    Quick mesh analysis function
    
    Args:
        obj: Object to analyze
        use_modifiers: Apply modifiers before analysis
    
    Returns:
        MeshData object or None
    """
    analyzer = MeshAnalyzer(obj)
    return analyzer.analyze(use_modifiers=use_modifiers)

def get_mesh_statistics(obj: bpy.types.Object) -> Dict[str, Any]:
    """
    Get mesh statistics
    
    Args:
        obj: Mesh object
    
    Returns:
        Dictionary with mesh statistics
    """
    data = analyze_mesh(obj)
    
    if not data:
        return {}
    
    return {
        'vertex_count': len(data.vertices),
        'edge_count': len(data.edges),
        'face_count': len(data.faces),
        'boundary_edges': len(data.boundary_edges),
        'sharp_edges': len(data.sharp_edges),
        'surface_area': data.surface_area,
        'volume': data.volume,
        'bounds': {
            'min': data.bounds_min,
            'max': data.bounds_max,
            'center': data.center,
            'size': data.bounds_max - data.bounds_min,
        }
    }

# Export
__all__ = [
    'MeshAnalyzer',
    'MeshData',
    'analyze_mesh',
    'get_mesh_statistics',
]
