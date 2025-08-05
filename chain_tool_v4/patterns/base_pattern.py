"""
Base Pattern Class for Chain Tool V4
Abstract base for all pattern generation strategies
"""

import bpy
import bmesh
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
from mathutils import Vector
import numpy as np

from ..utils.performance import measure_time
from ..utils.caching import CacheManager
from ..utils.debug import DebugManager
from ..core.state_manager import StateManager
from ..geometry.mesh_analyzer import MeshAnalyzer
from ..geometry.edge_detector import EdgeDetector
from ..geometry.surface_sampler import SurfaceSampler

@dataclass
class PatternResult:
    """Container for pattern generation results"""
    vertices: List[Vector]  # Pattern vertex positions
    edges: List[Tuple[int, int]]  # Edge connections
    faces: List[List[int]]  # Face definitions
    attributes: Dict[str, List]  # Additional attributes (e.g., thickness, material)
    metadata: Dict  # Pattern-specific metadata
    success: bool = True
    error_message: str = ""
    
    def to_mesh(self, name: str = "Pattern") -> bpy.types.Mesh:
        """Convert pattern result to Blender mesh"""
        mesh = bpy.data.meshes.new(name)
        
        # Add vertices
        mesh.vertices.add(len(self.vertices))
        for i, vert in enumerate(self.vertices):
            mesh.vertices[i].co = vert
            
        # Add edges
        if self.edges:
            mesh.edges.add(len(self.edges))
            for i, edge in enumerate(self.edges):
                mesh.edges[i].vertices = edge
                
        # Add faces
        if self.faces:
            # Calculate total loops needed
            total_loops = sum(len(face) for face in self.faces)
            mesh.loops.add(total_loops)
            mesh.polygons.add(len(self.faces))
            
            loop_index = 0
            for poly_index, face in enumerate(self.faces):
                poly = mesh.polygons[poly_index]
                poly.loop_start = loop_index
                poly.loop_total = len(face)
                
                for vert_index in face:
                    mesh.loops[loop_index].vertex_index = vert_index
                    loop_index += 1
                    
        mesh.update()
        return mesh

class BasePattern(ABC):
    """Abstract base class for all pattern types"""
    
    def __init__(self):
        self.cache = CacheManager()
        self.debug = DebugManager()
        self.state = StateManager()
        self.mesh_analyzer = MeshAnalyzer()
        self.edge_detector = EdgeDetector()
        self.surface_sampler = SurfaceSampler()
        
        # Pattern parameters (to be set by subclasses)
        self.name = "Base Pattern"
        self.category = "Abstract"
        self.description = "Base pattern class"
        
    @abstractmethod
    def generate(self, 
                target_object: bpy.types.Object,
                **params) -> PatternResult:
        """
        Generate pattern on target object
        
        Args:
            target_object: Object to generate pattern on
            **params: Pattern-specific parameters
            
        Returns:
            PatternResult containing generated geometry
        """
        pass
        
    @abstractmethod
    def get_default_params(self) -> Dict:
        """Get default parameters for this pattern type"""
        pass
        
    @abstractmethod
    def validate_params(self, params: Dict) -> Tuple[bool, str]:
        """
        Validate parameters
        
        Returns:
            (is_valid, error_message)
        """
        pass
        
    def preview(self, 
               target_object: bpy.types.Object,
               **params) -> bpy.types.Object:
        """Generate preview of pattern"""
        result = self.generate(target_object, **params)
        
        if not result.success:
            self.debug.error(f"Pattern preview failed: {result.error_message}")
            return None
            
        # Create preview object
        mesh = result.to_mesh(f"{self.name}_Preview")
        preview_obj = bpy.data.objects.new(f"{self.name}_Preview", mesh)
        bpy.context.collection.objects.link(preview_obj)
        
        # Copy transform from target
        preview_obj.matrix_world = target_object.matrix_world.copy()
        
        # Add preview material
        self._add_preview_material(preview_obj)
        
        return preview_obj
        
    def _add_preview_material(self, obj: bpy.types.Object):
        """Add preview material to object"""
        mat = bpy.data.materials.get("Pattern_Preview")
        if not mat:
            mat = bpy.data.materials.new("Pattern_Preview")
            mat.use_nodes = True
            
            # Set up preview shader
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            
            # Clear default nodes
            nodes.clear()
            
            # Add shader nodes
            output = nodes.new('ShaderNodeOutputMaterial')
            bsdf = nodes.new('ShaderNodeBsdfPrincipled')
            
            # Configure material
            bsdf.inputs['Base Color'].default_value = (0.2, 0.5, 0.8, 1.0)
            bsdf.inputs['Metallic'].default_value = 0.8
            bsdf.inputs['Roughness'].default_value = 0.3
            
            # Link nodes
            links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
            
        obj.data.materials.append(mat)
        
    @measure_time
    def generate_with_cache(self,
                          target_object: bpy.types.Object,
                          **params) -> PatternResult:
        """Generate pattern with caching support"""
        # Create cache key
        cache_key = self._create_cache_key(target_object, params)
        
        # Check cache
        cached_result = self.cache.get(cache_key)
        if cached_result and not self._needs_regeneration(target_object):
            self.debug.log(f"Using cached pattern: {self.name}")
            return cached_result
            
        # Generate new pattern
        self.debug.log(f"Generating new pattern: {self.name}")
        result = self.generate(target_object, **params)
        
        # Cache result if successful
        if result.success:
            self.cache.set(cache_key, result)
            
        return result
        
    def _create_cache_key(self, obj: bpy.types.Object, params: Dict) -> str:
        """Create unique cache key for pattern"""
        param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
        return f"pattern_{self.name}_{obj.name}_{param_str}"
        
    def _needs_regeneration(self, obj: bpy.types.Object) -> bool:
        """Check if pattern needs regeneration"""
        return obj.is_updated_data
        
    def combine_with(self, other_pattern: 'BasePattern') -> 'BasePattern':
        """Combine this pattern with another (for hybrid patterns)"""
        from .hybrid_patterns import CombinedPattern
        return CombinedPattern(self, other_pattern)
        
    # Utility methods for subclasses
    
    def sample_surface_points(self,
                            obj: bpy.types.Object,
                            density: float = 0.02,
                            adaptive: bool = True) -> List[Vector]:
        """Sample points on object surface"""
        return self.surface_sampler.sample_surface(
            obj,
            base_radius=density,
            curvature_influence=0.5 if adaptive else 0.0,
            edge_influence=0.3 if adaptive else 0.0
        )
        
    def get_edge_loops(self, obj: bpy.types.Object) -> List[List[int]]:
        """Get edge loops from object"""
        edges = self.edge_detector.detect_edges(obj)
        loops = self.edge_detector.extract_edge_loops(obj, edges)
        return loops
        
    def project_to_surface(self,
                         point: Vector,
                         obj: bpy.types.Object) -> Optional[Vector]:
        """Project point onto object surface"""
        bvh = self.mesh_analyzer.get_bvh_tree(obj)
        if not bvh:
            return None
            
        location, normal, index, distance = bvh.find_nearest(point)
        return location if location else None
        
    def offset_from_surface(self,
                          points: List[Vector],
                          obj: bpy.types.Object,
                          offset: float) -> List[Vector]:
        """Offset points from surface along normals"""
        bvh = self.mesh_analyzer.get_bvh_tree(obj)
        if not bvh:
            return points
            
        offset_points = []
        for point in points:
            location, normal, index, distance = bvh.find_nearest(point)
            if location and normal:
                offset_points.append(location + normal * offset)
            else:
                offset_points.append(point)
                
        return offset_points
        
    def apply_thickness_variation(self,
                                vertices: List[Vector],
                                base_thickness: float,
                                variation: float = 0.3) -> Dict[int, float]:
        """Apply thickness variation to pattern elements"""
        thicknesses = {}
        
        for i, vert in enumerate(vertices):
            # Simple noise-based variation
            noise = np.random.uniform(-variation, variation)
            thicknesses[i] = base_thickness * (1.0 + noise)
            
        return thicknesses
        
    def optimize_connectivity(self,
                            vertices: List[Vector],
                            max_edge_length: float) -> List[Tuple[int, int]]:
        """Optimize connectivity between vertices"""
        edges = []
        
        # Build KD-tree for efficient neighbor search
        from mathutils import kdtree
        kd = kdtree.KDTree(len(vertices))
        for i, vert in enumerate(vertices):
            kd.insert(vert, i)
        kd.balance()
        
        # Connect nearby vertices
        for i, vert in enumerate(vertices):
            # Find neighbors within max distance
            neighbors = kd.find_range(vert, max_edge_length)
            
            for co, idx, dist in neighbors:
                if idx > i:  # Avoid duplicate edges
                    edges.append((i, idx))
                    
        return edges
        
    def remove_overlapping_elements(self,
                                  vertices: List[Vector],
                                  min_distance: float) -> List[Vector]:
        """Remove overlapping elements"""
        if not vertices:
            return []
            
        filtered = [vertices[0]]
        
        for vert in vertices[1:]:
            # Check distance to all filtered vertices
            too_close = False
            for fvert in filtered:
                if (vert - fvert).length < min_distance:
                    too_close = True
                    break
                    
            if not too_close:
                filtered.append(vert)
                
        return filtered
        
    def create_mesh_from_pattern(self,
                               vertices: List[Vector],
                               edges: List[Tuple[int, int]] = None,
                               faces: List[List[int]] = None,
                               name: str = "Pattern") -> bpy.types.Object:
        """Create mesh object from pattern data"""
        result = PatternResult(
            vertices=vertices,
            edges=edges or [],
            faces=faces or [],
            attributes={},
            metadata={}
        )
        
        mesh = result.to_mesh(name)
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(obj)
        
        return obj
