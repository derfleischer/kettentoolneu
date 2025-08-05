"""
Surface Sampler Module for Chain Tool V4
Adaptive Poisson-Disk sampling with curvature awareness
"""

import bpy
import bmesh
import numpy as np
from mathutils import Vector, kdtree
from typing import List, Tuple, Dict, Optional, Set
import random
from ..utils.performance import measure_time
from ..utils.caching import CacheManager
from ..utils.debug import DebugManager
from ..core.state_manager import StateManager
from .mesh_analyzer import MeshAnalyzer
from .edge_detector import EdgeDetector

class SurfaceSampler:
    """Adaptive Poisson-Disk sampling on mesh surfaces"""
    
    def __init__(self):
        self.cache = CacheManager()
        self.debug = DebugManager()
        self.state = StateManager()
        self.mesh_analyzer = MeshAnalyzer()
        self.edge_detector = EdgeDetector()
        
    @measure_time
    def sample_surface(self, 
                      obj: bpy.types.Object,
                      base_radius: float = 0.02,
                      min_radius: float = 0.01,
                      max_radius: float = 0.05,
                      curvature_influence: float = 0.5,
                      edge_influence: float = 0.3,
                      paint_influence: float = 0.7,
                      max_samples: int = 10000) -> List[Vector]:
        """
        Generate adaptive Poisson-disk samples on mesh surface
        
        Args:
            obj: Mesh object to sample
            base_radius: Base minimum distance between samples
            min_radius: Absolute minimum radius
            max_radius: Maximum radius in low-curvature areas
            curvature_influence: How much curvature affects density (0-1)
            edge_influence: How much edges affect density (0-1)
            paint_influence: How much paint affects density (0-1)
            max_samples: Maximum number of samples to generate
            
        Returns:
            List of sample points on surface
        """
        # Check cache
        cache_key = f"surface_samples_{obj.name}_{base_radius}_{curvature_influence}"
        cached = self.cache.get(cache_key)
        if cached and not self._needs_resampling(obj):
            return cached
            
        # Ensure mesh data
        mesh = obj.data
        if not mesh.vertices:
            return []
            
        # Get analysis data
        curvatures = self._get_curvature_map(obj)
        edge_distances = self._get_edge_distance_map(obj)
        paint_weights = self._get_paint_weights(obj)
        
        # Generate samples
        samples = self._poisson_disk_sampling(
            obj, mesh,
            base_radius, min_radius, max_radius,
            curvatures, edge_distances, paint_weights,
            curvature_influence, edge_influence, paint_influence,
            max_samples
        )
        
        # Cache results
        self.cache.set(cache_key, samples)
        
        return samples
        
    def _needs_resampling(self, obj: bpy.types.Object) -> bool:
        """Check if object needs resampling"""
        # Check if mesh was modified
        if obj.is_updated_data:
            return True
            
        # Check if paint data changed
        paint_state = self.state.get("paint_strokes", {})
        last_paint = self.cache.get(f"last_paint_state_{obj.name}")
        if paint_state != last_paint:
            self.cache.set(f"last_paint_state_{obj.name}", paint_state)
            return True
            
        return False
        
    def _get_curvature_map(self, obj: bpy.types.Object) -> Dict[int, float]:
        """Get curvature values for each vertex"""
        mesh = obj.data
        curvatures = {}
        
        # Calculate curvatures using mesh analyzer
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.verts.ensure_lookup_table()
        
        for vert in bm.verts:
            # Simple curvature estimation based on normal deviation
            if len(vert.link_faces) > 1:
                normals = [f.normal for f in vert.link_faces]
                avg_normal = sum(normals, Vector()) / len(normals)
                
                # Calculate deviation
                deviation = sum((n - avg_normal).length for n in normals) / len(normals)
                curvatures[vert.index] = min(deviation * 2.0, 1.0)  # Normalize to 0-1
            else:
                curvatures[vert.index] = 0.0
                
        bm.free()
        return curvatures
        
    def _get_edge_distance_map(self, obj: bpy.types.Object) -> Dict[int, float]:
        """Get distance to nearest edge for each vertex"""
        edges = self.edge_detector.detect_edges(obj)
        mesh = obj.data
        distances = {}
        
        if not edges:
            return {i: 1.0 for i in range(len(mesh.vertices))}
            
        # Build KD-tree of edge vertices
        edge_verts = set()
        for edge in edges:
            edge_verts.add(edge[0])
            edge_verts.add(edge[1])
            
        if not edge_verts:
            return {i: 1.0 for i in range(len(mesh.vertices))}
            
        kd = kdtree.KDTree(len(edge_verts))
        for i, vert_idx in enumerate(edge_verts):
            kd.insert(mesh.vertices[vert_idx].co, i)
        kd.balance()
        
        # Calculate distances
        max_dist = 0.0
        for vert in mesh.vertices:
            co, idx, dist = kd.find(vert.co)
            distances[vert.index] = dist
            max_dist = max(max_dist, dist)
            
        # Normalize distances
        if max_dist > 0:
            for idx in distances:
                distances[idx] = distances[idx] / max_dist
                
        return distances
        
    def _get_paint_weights(self, obj: bpy.types.Object) -> Dict[int, float]:
        """Get paint influence weights for vertices"""
        weights = {}
        mesh = obj.data
        
        # Get paint strokes from state
        paint_strokes = self.state.get("paint_strokes", {}).get(obj.name, [])
        
        if not paint_strokes:
            return {i: 0.0 for i in range(len(mesh.vertices))}
            
        # Build weight map from strokes
        for stroke in paint_strokes:
            for point in stroke.get("points", []):
                # Find nearest vertex
                min_dist = float('inf')
                nearest_vert = -1
                
                point_co = Vector(point["location"])
                for vert in mesh.vertices:
                    dist = (vert.co - point_co).length
                    if dist < min_dist:
                        min_dist = dist
                        nearest_vert = vert.index
                        
                if nearest_vert >= 0:
                    # Apply falloff
                    radius = stroke.get("radius", 0.05)
                    influence = max(0, 1.0 - (min_dist / radius))
                    weights[nearest_vert] = max(weights.get(nearest_vert, 0), influence)
                    
        return weights
        
    def _poisson_disk_sampling(self,
                             obj: bpy.types.Object,
                             mesh: bpy.types.Mesh,
                             base_radius: float,
                             min_radius: float,
                             max_radius: float,
                             curvatures: Dict[int, float],
                             edge_distances: Dict[int, float],
                             paint_weights: Dict[int, float],
                             curvature_influence: float,
                             edge_influence: float,
                             paint_influence: float,
                             max_samples: int) -> List[Vector]:
        """Core Poisson-disk sampling algorithm with adaptive radius"""
        
        samples = []
        active_list = []
        
        # Build spatial index for fast neighbor queries
        cell_size = min_radius / np.sqrt(3)  # For 3D
        grid = {}
        
        # Helper to get adaptive radius at position
        def get_adaptive_radius(co: Vector, face_idx: int = -1) -> float:
            # Find nearest vertex for attribute lookup
            min_dist = float('inf')
            nearest_vert = -1
            
            for vert in mesh.vertices:
                dist = (vert.co - co).length
                if dist < min_dist:
                    min_dist = dist
                    nearest_vert = vert.index
                    
            if nearest_vert < 0:
                return base_radius
                
            # Calculate adaptive radius based on local properties
            radius = base_radius
            
            # Curvature influence (high curvature = smaller radius = more samples)
            if curvature_influence > 0:
                curv = curvatures.get(nearest_vert, 0)
                radius *= (1.0 - curv * curvature_influence)
                
            # Edge influence (near edges = smaller radius = more samples)
            if edge_influence > 0:
                edge_dist = edge_distances.get(nearest_vert, 1.0)
                radius *= (1.0 - (1.0 - edge_dist) * edge_influence)
                
            # Paint influence (painted areas = smaller radius = more samples)
            if paint_influence > 0:
                paint_weight = paint_weights.get(nearest_vert, 0)
                radius *= (1.0 - paint_weight * paint_influence)
                
            return max(min_radius, min(max_radius, radius))
            
        # Helper to add sample to grid
        def add_to_grid(co: Vector, radius: float):
            grid_co = (
                int(co.x / cell_size),
                int(co.y / cell_size),
                int(co.z / cell_size)
            )
            if grid_co not in grid:
                grid[grid_co] = []
            grid[grid_co].append((co, radius))
            
        # Helper to check if position is valid
        def is_valid_sample(co: Vector, radius: float) -> bool:
            # Check nearby grid cells
            grid_co = (
                int(co.x / cell_size),
                int(co.y / cell_size),
                int(co.z / cell_size)
            )
            
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    for dz in range(-2, 3):
                        check_co = (grid_co[0] + dx, grid_co[1] + dy, grid_co[2] + dz)
                        if check_co in grid:
                            for sample_co, sample_radius in grid[check_co]:
                                # Use minimum of both radii for distance check
                                min_dist = min(radius, sample_radius)
                                if (co - sample_co).length < min_dist:
                                    return False
            return True
            
        # Start with random point on surface
        face = random.choice(mesh.polygons)
        start_co = self._random_point_on_face(mesh, face)
        start_radius = get_adaptive_radius(start_co, face.index)
        
        samples.append(start_co)
        active_list.append(len(samples) - 1)
        add_to_grid(start_co, start_radius)
        
        # Main sampling loop
        attempts_per_sample = 30
        
        while active_list and len(samples) < max_samples:
            # Pick random point from active list
            idx = random.randint(0, len(active_list) - 1)
            sample_idx = active_list[idx]
            sample_co = samples[sample_idx]
            sample_radius = get_adaptive_radius(sample_co)
            
            # Try to place new samples around it
            placed = False
            
            for _ in range(attempts_per_sample):
                # Generate random point in annulus
                angle1 = random.uniform(0, 2 * np.pi)
                angle2 = random.uniform(0, np.pi)
                dist = random.uniform(sample_radius, sample_radius * 2)
                
                offset = Vector((
                    dist * np.sin(angle2) * np.cos(angle1),
                    dist * np.sin(angle2) * np.sin(angle1),
                    dist * np.cos(angle2)
                ))
                
                new_co = sample_co + offset
                
                # Project onto surface
                hit, location, normal, face_idx = self._closest_point_on_mesh(obj, new_co)
                
                if hit:
                    new_radius = get_adaptive_radius(location, face_idx)
                    
                    if is_valid_sample(location, new_radius):
                        samples.append(location)
                        active_list.append(len(samples) - 1)
                        add_to_grid(location, new_radius)
                        placed = True
                        
                        if len(samples) >= max_samples:
                            break
                            
            # Remove from active list if no valid samples found
            if not placed:
                active_list.pop(idx)
                
        self.debug.log(f"Generated {len(samples)} surface samples")
        return samples
        
    def _random_point_on_face(self, mesh: bpy.types.Mesh, face: bpy.types.MeshPolygon) -> Vector:
        """Generate random point on mesh face using barycentric coordinates"""
        verts = [mesh.vertices[i].co for i in face.vertices]
        
        if len(verts) == 3:
            # Triangle - simple barycentric
            r1, r2 = random.random(), random.random()
            if r1 + r2 > 1:
                r1, r2 = 1 - r1, 1 - r2
            r3 = 1 - r1 - r2
            return verts[0] * r1 + verts[1] * r2 + verts[2] * r3
        else:
            # Quad or n-gon - triangulate first
            # Use simple fan triangulation from first vertex
            tri_idx = random.randint(0, len(verts) - 3)
            v0 = verts[0]
            v1 = verts[tri_idx + 1]
            v2 = verts[tri_idx + 2]
            
            r1, r2 = random.random(), random.random()
            if r1 + r2 > 1:
                r1, r2 = 1 - r1, 1 - r2
            r3 = 1 - r1 - r2
            return v0 * r1 + v1 * r2 + v2 * r3
            
    def _closest_point_on_mesh(self, obj: bpy.types.Object, point: Vector) -> Tuple[bool, Vector, Vector, int]:
        """Find closest point on mesh surface"""
        # Use BVH from mesh analyzer for fast lookup
        bvh = self.mesh_analyzer.get_bvh_tree(obj)
        if not bvh:
            return False, Vector(), Vector(), -1
            
        location, normal, index, distance = bvh.find_nearest(point)
        
        if location:
            return True, location, normal, index
        return False, Vector(), Vector(), -1
        
    def place_sphere_at_position(self, position: Vector, radius: float = 0.01) -> bpy.types.Object:
        """Place a sphere at the given position (using existing function)"""
        # This uses the basic sphere placement that already exists
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=8,
            ring_count=6,
            radius=radius,
            location=position
        )
        return bpy.context.active_object
        
    def visualize_samples(self, samples: List[Vector], radius: float = 0.005):
        """Visualize sample points as spheres for debugging"""
        if not self.debug.is_enabled():
            return
            
        # Create collection for samples
        col_name = "Sample_Visualization"
        if col_name in bpy.data.collections:
            col = bpy.data.collections[col_name]
            # Clear existing
            for obj in col.objects:
                bpy.data.objects.remove(obj)
        else:
            col = bpy.data.collections.new(col_name)
            bpy.context.scene.collection.children.link(col)
            
        # Create spheres at sample positions
        for i, sample in enumerate(samples):
            sphere = self.place_sphere_at_position(sample, radius)
            sphere.name = f"Sample_{i:04d}"
            
            # Move to collection
            for coll in sphere.users_collection:
                coll.objects.unlink(sphere)
            col.objects.link(sphere)
            
        self.debug.log(f"Visualized {len(samples)} samples")
