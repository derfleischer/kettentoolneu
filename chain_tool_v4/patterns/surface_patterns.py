"""
Surface Pattern Implementations for Chain Tool V4
Large area coverage patterns for structural reinforcement
"""

import bpy
import bmesh
import numpy as np
from mathutils import Vector, Matrix
from typing import List, Dict, Tuple, Set, Optional
from scipy.spatial import Voronoi, Delaunay
import math

from .base_pattern import BasePattern, PatternResult
from ..geometry.triangulation import TriangulationEngine
from ..utils.math_utils import MathUtils

class VoronoiPattern(BasePattern):
    """Voronoi cell pattern for organic-looking structures"""
    
    def __init__(self):
        super().__init__()
        self.name = "Voronoi"
        self.category = "Surface"
        self.description = "Organic cell-like pattern based on Voronoi diagram"
        self.math_utils = MathUtils()
        
    def get_default_params(self) -> Dict:
        return {
            'cell_size': 0.02,  # Average cell size
            'randomness': 0.5,  # 0-1 randomness factor
            'edge_width': 0.003,  # Width of cell edges
            'fill_ratio': 0.7,  # How much of each cell to fill (0-1)
            'adaptive_density': True,  # Adapt to surface features
            'smooth_iterations': 2,  # Smoothing passes
            'min_edge_length': 0.002,  # Minimum edge length
            'pattern_offset': 0.001  # Offset from surface
        }
        
    def validate_params(self, params: Dict) -> Tuple[bool, str]:
        if params.get('cell_size', 0) <= 0:
            return False, "Cell size must be positive"
        if not 0 <= params.get('randomness', 0.5) <= 1:
            return False, "Randomness must be between 0 and 1"
        if params.get('edge_width', 0) <= 0:
            return False, "Edge width must be positive"
        return True, ""
        
    def generate(self, target_object: bpy.types.Object, **params) -> PatternResult:
        # Merge with defaults
        full_params = self.get_default_params()
        full_params.update(params)
        
        # Validate
        valid, error = self.validate_params(full_params)
        if not valid:
            return PatternResult([], [], [], {}, {}, False, error)
            
        # Sample points on surface
        sample_points = self._generate_sample_points(
            target_object,
            full_params['cell_size'],
            full_params['randomness'],
            full_params['adaptive_density']
        )
        
        if len(sample_points) < 4:
            return PatternResult([], [], [], {}, {}, False, "Not enough sample points")
            
        # Generate Voronoi diagram
        vor = self._compute_voronoi(sample_points, target_object)
        
        # Convert to pattern geometry
        pattern_verts = []
        pattern_edges = []
        pattern_faces = []
        
        # Process each Voronoi region
        for region_idx, region in enumerate(vor.regions):
            if not region or -1 in region:
                continue
                
            # Get region vertices
            region_verts = [vor.vertices[i] for i in region]
            
            # Check if region is valid
            if len(region_verts) < 3:
                continue
                
            # Apply fill ratio (shrink cell)
            if full_params['fill_ratio'] < 1.0:
                region_verts = self._shrink_cell(
                    region_verts,
                    full_params['fill_ratio']
                )
                
            # Create cell geometry
            cell_data = self._create_cell_geometry(
                region_verts,
                full_params['edge_width'],
                target_object
            )
            
            # Add to pattern with proper indexing
            base_idx = len(pattern_verts)
            pattern_verts.extend(cell_data['vertices'])
            
            # Adjust indices for edges and faces
            for edge in cell_data['edges']:
                pattern_edges.append((edge[0] + base_idx, edge[1] + base_idx))
                
            for face in cell_data['faces']:
                pattern_faces.append([v + base_idx for v in face])
                
        # Smooth pattern
        if full_params['smooth_iterations'] > 0:
            pattern_verts = self._smooth_vertices(
                pattern_verts,
                pattern_edges,
                full_params['smooth_iterations'],
                target_object
            )
            
        # Project and offset from surface
        pattern_verts = self._project_and_offset(
            pattern_verts,
            target_object,
            full_params['pattern_offset']
        )
        
        # Remove too-small edges
        pattern_verts, pattern_edges, pattern_faces = self._clean_geometry(
            pattern_verts,
            pattern_edges,
            pattern_faces,
            full_params['min_edge_length']
        )
        
        # Create attributes
        attributes = {
            'thickness': [full_params['edge_width']] * len(pattern_verts),
            'material_index': [1] * len(pattern_faces)  # PETG material
        }
        
        metadata = {
            'pattern_type': 'voronoi',
            'cell_count': len(vor.regions),
            'parameters': full_params
        }
        
        return PatternResult(
            vertices=pattern_verts,
            edges=pattern_edges,
            faces=pattern_faces,
            attributes=attributes,
            metadata=metadata
        )
        
    def _generate_sample_points(self,
                              obj: bpy.types.Object,
                              cell_size: float,
                              randomness: float,
                              adaptive: bool) -> List[Vector]:
        """Generate sample points for Voronoi cells"""
        if adaptive:
            # Use surface sampler for adaptive sampling
            points = self.sample_surface_points(
                obj,
                density=cell_size,
                adaptive=True
            )
        else:
            # Regular grid sampling
            points = self._grid_sample_surface(obj, cell_size)
            
        # Add randomness
        if randomness > 0:
            points = self._randomize_points(points, cell_size * randomness)
            
        return points
        
    def _grid_sample_surface(self, obj: bpy.types.Object, spacing: float) -> List[Vector]:
        """Sample surface on regular grid"""
        points = []
        
        # Get bounds
        bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        min_co = Vector((min(v.x for v in bbox), 
                        min(v.y for v in bbox),
                        min(v.z for v in bbox)))
        max_co = Vector((max(v.x for v in bbox),
                        max(v.y for v in bbox),
                        max(v.z for v in bbox)))
        
        # Generate grid
        x_steps = int((max_co.x - min_co.x) / spacing) + 1
        y_steps = int((max_co.y - min_co.y) / spacing) + 1
        z_steps = int((max_co.z - min_co.z) / spacing) + 1
        
        bvh = self.mesh_analyzer.get_bvh_tree(obj)
        
        for x in range(x_steps):
            for y in range(y_steps):
                for z in range(z_steps):
                    grid_point = Vector((
                        min_co.x + x * spacing,
                        min_co.y + y * spacing,
                        min_co.z + z * spacing
                    ))
                    
                    # Project to surface
                    location, normal, index, distance = bvh.find_nearest(grid_point)
                    if location and distance < spacing:
                        points.append(location)
                        
        return points
        
    def _randomize_points(self, points: List[Vector], max_offset: float) -> List[Vector]:
        """Add random offset to points"""
        randomized = []
        for point in points:
            offset = Vector((
                np.random.uniform(-max_offset, max_offset),
                np.random.uniform(-max_offset, max_offset),
                np.random.uniform(-max_offset, max_offset)
            ))
            randomized.append(point + offset)
        return randomized
        
    def _compute_voronoi(self, points: List[Vector], obj: bpy.types.Object) -> Voronoi:
        """Compute 2D Voronoi diagram on surface"""
        # For simplicity, project to best-fit plane
        # In production, use proper surface parameterization
        
        # Find principal plane
        points_array = np.array([list(p) for p in points])
        center = np.mean(points_array, axis=0)
        
        # PCA for plane fitting
        centered = points_array - center
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # Sort by eigenvalue
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # Use first two eigenvectors as plane basis
        u_axis = Vector(eigenvectors[:, 0])
        v_axis = Vector(eigenvectors[:, 1])
        normal = u_axis.cross(v_axis)
        
        # Project points to 2D
        points_2d = []
        for p in points:
            p_centered = p - Vector(center)
            u = p_centered.dot(u_axis)
            v = p_centered.dot(v_axis)
            points_2d.append([u, v])
            
        # Compute Voronoi
        vor = Voronoi(np.array(points_2d))
        
        # Transform vertices back to 3D
        vertices_3d = []
        for v2d in vor.vertices:
            p3d = Vector(center) + u_axis * v2d[0] + v_axis * v2d[1]
            # Project onto surface
            location = self.project_to_surface(p3d, obj)
            if location:
                vertices_3d.append(location)
            else:
                vertices_3d.append(p3d)
                
        vor.vertices = np.array([list(v) for v in vertices_3d])
        
        return vor
        
    def _shrink_cell(self, vertices: List[Vector], ratio: float) -> List[Vector]:
        """Shrink cell toward center"""
        if not vertices:
            return vertices
            
        center = sum(vertices, Vector()) / len(vertices)
        shrunk = []
        
        for vert in vertices:
            direction = vert - center
            shrunk.append(center + direction * ratio)
            
        return shrunk
        
    def _create_cell_geometry(self,
                            cell_verts: List[Vector],
                            edge_width: float,
                            obj: bpy.types.Object) -> Dict:
        """Create geometry for a single Voronoi cell"""
        geometry = {
            'vertices': [],
            'edges': [],
            'faces': []
        }
        
        # Create frame from cell outline
        n = len(cell_verts)
        
        # Inner and outer vertices
        center = sum(cell_verts, Vector()) / n
        
        for i, vert in enumerate(cell_verts):
            # Outer vertex
            geometry['vertices'].append(vert)
            
            # Inner vertex (offset toward center)
            direction = (center - vert).normalized()
            inner = vert + direction * edge_width
            geometry['vertices'].append(inner)
            
        # Create edges and faces
        for i in range(n):
            next_i = (i + 1) % n
            
            # Indices
            outer_current = i * 2
            inner_current = i * 2 + 1
            outer_next = next_i * 2
            inner_next = next_i * 2 + 1
            
            # Edge connections
            geometry['edges'].extend([
                (outer_current, outer_next),
                (inner_current, inner_next),
                (outer_current, inner_current),
                (outer_next, inner_next)
            ])
            
            # Face (quad)
            geometry['faces'].append([
                outer_current, outer_next,
                inner_next, inner_current
            ])
            
        return geometry
        
    def _smooth_vertices(self,
                        vertices: List[Vector],
                        edges: List[Tuple[int, int]],
                        iterations: int,
                        obj: bpy.types.Object) -> List[Vector]:
        """Apply Laplacian smoothing"""
        if not edges or iterations <= 0:
            return vertices
            
        # Build adjacency
        adjacency = {i: set() for i in range(len(vertices))}
        for e in edges:
            adjacency[e[0]].add(e[1])
            adjacency[e[1]].add(e[0])
            
        # Smooth
        smoothed = vertices.copy()
        for _ in range(iterations):
            new_positions = []
            
            for i, vert in enumerate(smoothed):
                if adjacency[i]:
                    # Average of neighbors
                    avg = sum((smoothed[j] for j in adjacency[i]), Vector()) / len(adjacency[i])
                    # Blend with original
                    new_pos = vert * 0.5 + avg * 0.5
                    new_positions.append(new_pos)
                else:
                    new_positions.append(vert)
                    
            smoothed = new_positions
            
        return smoothed
        
    def _project_and_offset(self,
                          vertices: List[Vector],
                          obj: bpy.types.Object,
                          offset: float) -> List[Vector]:
        """Project vertices to surface and apply offset"""
        return self.offset_from_surface(vertices, obj, offset)
        
    def _clean_geometry(self,
                       vertices: List[Vector],
                       edges: List[Tuple[int, int]],
                       faces: List[List[int]],
                       min_edge_length: float) -> Tuple[List[Vector], List[Tuple[int, int]], List[List[int]]]:
        """Remove too-small edges and degenerate faces"""
        # This is simplified - in production, use proper mesh cleaning
        return vertices, edges, faces


class HexagonalGrid(BasePattern):
    """Hexagonal grid pattern for maximum strength-to-weight ratio"""
    
    def __init__(self):
        super().__init__()
        self.name = "Hexagonal Grid"
        self.category = "Surface"
        self.description = "Regular hexagonal grid pattern"
        
    def get_default_params(self) -> Dict:
        return {
            'hex_size': 0.015,  # Hexagon radius
            'wall_thickness': 0.003,  # Wall thickness
            'pattern_height': 0.002,  # Pattern extrusion height
            'orientation': 0.0,  # Grid rotation angle
            'adaptive_sizing': False,  # Vary hex size by stress
            'pattern_offset': 0.001  # Offset from surface
        }
        
    def validate_params(self, params: Dict) -> Tuple[bool, str]:
        if params.get('hex_size', 0) <= 0:
            return False, "Hex size must be positive"
        if params.get('wall_thickness', 0) <= 0:
            return False, "Wall thickness must be positive"
        return True, ""
        
    def generate(self, target_object: bpy.types.Object, **params) -> PatternResult:
        # Merge with defaults
        full_params = self.get_default_params()
        full_params.update(params)
        
        # Validate
        valid, error = self.validate_params(full_params)
        if not valid:
            return PatternResult([], [], [], {}, {}, False, error)
            
        # Generate hex centers on surface
        hex_centers = self._generate_hex_centers(
            target_object,
            full_params['hex_size'],
            full_params['orientation']
        )
        
        # Create hex geometry
        pattern_verts = []
        pattern_edges = []
        pattern_faces = []
        
        for center in hex_centers:
            hex_data = self._create_hexagon(
                center,
                full_params['hex_size'],
                full_params['wall_thickness'],
                full_params['orientation'],
                target_object
            )
            
            # Add to pattern
            base_idx = len(pattern_verts)
            pattern_verts.extend(hex_data['vertices'])
            
            for edge in hex_data['edges']:
                pattern_edges.append((edge[0] + base_idx, edge[1] + base_idx))
                
            for face in hex_data['faces']:
                pattern_faces.append([v + base_idx for v in face])
                
        # Project and offset
        pattern_verts = self._project_and_offset(
            pattern_verts,
            target_object,
            full_params['pattern_offset']
        )
        
        # Create attributes
        attributes = {
            'thickness': [full_params['pattern_height']] * len(pattern_verts),
            'material_index': [1] * len(pattern_faces)
        }
        
        metadata = {
            'pattern_type': 'hexagonal',
            'hex_count': len(hex_centers),
            'parameters': full_params
        }
        
        return PatternResult(
            vertices=pattern_verts,
            edges=pattern_edges,
            faces=pattern_faces,
            attributes=attributes,
            metadata=metadata
        )
        
    def _generate_hex_centers(self,
                            obj: bpy.types.Object,
                            hex_size: float,
                            orientation: float) -> List[Vector]:
        """Generate hexagon center points on surface"""
        centers = []
        
        # Calculate spacing
        h_spacing = hex_size * 3 / 2  # Horizontal spacing
        v_spacing = hex_size * math.sqrt(3)  # Vertical spacing
        
        # Get bounds
        bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        min_co = Vector((min(v.x for v in bbox), 
                        min(v.y for v in bbox),
                        min(v.z for v in bbox)))
        max_co = Vector((max(v.x for v in bbox),
                        max(v.y for v in bbox),
                        max(v.z for v in bbox)))
        
        # Generate grid
        rows = int((max_co.y - min_co.y) / v_spacing) + 2
        cols = int((max_co.x - min_co.x) / h_spacing) + 2
        
        bvh = self.mesh_analyzer.get_bvh_tree(obj)
        
        for row in range(rows):
            for col in range(cols):
                # Calculate position
                x = min_co.x + col * h_spacing
                y = min_co.y + row * v_spacing
                
                # Offset every other row
                if row % 2 == 1:
                    x += h_spacing / 2
                    
                # Apply rotation
                if orientation != 0:
                    # Rotate around center
                    center_x = (min_co.x + max_co.x) / 2
                    center_y = (min_co.y + max_co.y) / 2
                    
                    dx = x - center_x
                    dy = y - center_y
                    
                    cos_a = math.cos(orientation)
                    sin_a = math.sin(orientation)
                    
                    x = center_x + dx * cos_a - dy * sin_a
                    y = center_y + dx * sin_a + dy * cos_a
                    
                # Project to surface
                test_point = Vector((x, y, (min_co.z + max_co.z) / 2))
                location, normal, index, distance = bvh.find_nearest(test_point)
                
                if location:
                    centers.append(location)
                    
        return centers
        
    def _create_hexagon(self,
                       center: Vector,
                       size: float,
                       wall_thickness: float,
                       orientation: float,
                       obj: bpy.types.Object) -> Dict:
        """Create a single hexagon"""
        geometry = {
            'vertices': [],
            'edges': [],
            'faces': []
        }
        
        # Generate hex vertices
        for i in range(6):
            angle = orientation + i * math.pi / 3
            
            # Outer vertex
            outer = center + Vector((
                size * math.cos(angle),
                size * math.sin(angle),
                0
            ))
            geometry['vertices'].append(outer)
            
            # Inner vertex
            inner = center + Vector((
                (size - wall_thickness) * math.cos(angle),
                (size - wall_thickness) * math.sin(angle),
                0
            ))
            geometry['vertices'].append(inner)
            
        # Create edges and faces
        for i in range(6):
            next_i = (i + 1) % 6
            
            # Indices
            outer_current = i * 2
            inner_current = i * 2 + 1
            outer_next = next_i * 2
            inner_next = next_i * 2 + 1
            
            # Edges
            geometry['edges'].extend([
                (outer_current, outer_next),
                (inner_current, inner_next)
            ])
            
            # Face
            geometry['faces'].append([
                outer_current, outer_next,
                inner_next, inner_current
            ])
            
        return geometry


class TriangularMesh(BasePattern):
    """Triangular mesh pattern using Delaunay triangulation"""
    
    def __init__(self):
        super().__init__()
        self.name = "Triangular Mesh"
        self.category = "Surface"
        self.description = "Triangulated mesh pattern"
        self.triangulation = TriangulationEngine()
        
    def get_default_params(self) -> Dict:
        return {
            'point_density': 0.02,  # Average spacing between points
            'edge_thickness': 0.003,  # Thickness of triangle edges
            'fill_type': 'frame',  # 'frame', 'solid', or 'gradient'
            'min_angle': 20.0,  # Minimum triangle angle (degrees)
            'max_edge_length': 0.05,  # Maximum edge length
            'pattern_offset': 0.001,  # Offset from surface
            'adaptive': True  # Adaptive point distribution
        }
        
    def validate_params(self, params: Dict) -> Tuple[bool, str]:
        if params.get('point_density', 0) <= 0:
            return False, "Point density must be positive"
        if params.get('min_angle', 0) <= 0 or params.get('min_angle', 0) >= 60:
            return False, "Minimum angle must be between 0 and 60 degrees"
        return True, ""
        
    def generate(self, target_object: bpy.types.Object, **params) -> PatternResult:
        # Merge with defaults
        full_params = self.get_default_params()
        full_params.update(params)
        
        # Validate
        valid, error = self.validate_params(full_params)
        if not valid:
            return PatternResult([], [], [], {}, {}, False, error)
            
        # Sample points
        sample_points = self.sample_surface_points(
            target_object,
            density=full_params['point_density'],
            adaptive=full_params['adaptive']
        )
        
        # Triangulate
        triangulation = self.triangulation.triangulate_surface(
            sample_points,
            quality_threshold=full_params['min_angle'],
            max_edge_length=full_params['max_edge_length']
        )
        
        # Generate pattern based on fill type
        if full_params['fill_type'] == 'frame':
            result = self._generate_frame_pattern(
                triangulation,
                full_params['edge_thickness'],
                target_object
            )
        elif full_params['fill_type'] == 'solid':
            result = self._generate_solid_pattern(triangulation, target_object)
        else:  # gradient
            result = self._generate_gradient_pattern(
                triangulation,
                full_params['edge_thickness'],
                target_object
            )
            
        # Apply offset
        result.vertices = self.offset_from_surface(
            result.vertices,
            target_object,
            full_params['pattern_offset']
        )
        
        # Add metadata
        result.metadata = {
            'pattern_type': 'triangular',
            'triangle_count': len(triangulation['faces']),
            'parameters': full_params
        }
        
        return result
        
    def _generate_frame_pattern(self,
                              triangulation: Dict,
                              thickness: float,
                              obj: bpy.types.Object) -> PatternResult:
        """Generate frame pattern from triangulation"""
        pattern_verts = []
        pattern_edges = []
        pattern_faces = []
        
        # Process each edge
        processed_edges = set()
        
        for face in triangulation['faces']:
            # Get face edges
            edges = [
                (face[0], face[1]),
                (face[1], face[2]),
                (face[2], face[0])
            ]
            
            for edge in edges:
                # Normalize edge (smaller index first)
                edge_key = tuple(sorted(edge))
                
                if edge_key not in processed_edges:
                    processed_edges.add(edge_key)
                    
                    # Create beam geometry for edge
                    beam = self._create_edge_beam(
                        triangulation['vertices'][edge[0]],
                        triangulation['vertices'][edge[1]],
                        thickness
                    )
                    
                    # Add to pattern
                    base_idx = len(pattern_verts)
                    pattern_verts.extend(beam['vertices'])
                    
                    for e in beam['edges']:
                        pattern_edges.append((e[0] + base_idx, e[1] + base_idx))
                        
                    for f in beam['faces']:
                        pattern_faces.append([v + base_idx for v in f])
                        
        attributes = {
            'thickness': [thickness] * len(pattern_verts),
            'material_index': [1] * len(pattern_faces)
        }
        
        return PatternResult(
            vertices=pattern_verts,
            edges=pattern_edges,
            faces=pattern_faces,
            attributes=attributes,
            metadata={}
        )
        
    def _generate_solid_pattern(self,
                              triangulation: Dict,
                              obj: bpy.types.Object) -> PatternResult:
        """Generate solid triangular pattern"""
        # Simply use triangulation as-is
        attributes = {
            'thickness': [0.002] * len(triangulation['vertices']),
            'material_index': [1] * len(triangulation['faces'])
        }
        
        return PatternResult(
            vertices=triangulation['vertices'],
            edges=triangulation['edges'],
            faces=triangulation['faces'],
            attributes=attributes,
            metadata={}
        )
        
    def _generate_gradient_pattern(self,
                                 triangulation: Dict,
                                 thickness: float,
                                 obj: bpy.types.Object) -> PatternResult:
        """Generate gradient density pattern"""
        # This would implement variable density based on stress analysis
        # For now, fallback to frame pattern
        return self._generate_frame_pattern(triangulation, thickness, obj)
        
    def _create_edge_beam(self,
                         v1: Vector,
                         v2: Vector,
                         thickness: float) -> Dict:
        """Create beam geometry for an edge"""
        geometry = {
            'vertices': [],
            'edges': [],
            'faces': []
        }
        
        # Calculate edge direction and perpendiculars
        edge_dir = (v2 - v1).normalized()
        
        # Find perpendicular directions
        up = Vector((0, 0, 1))
        if abs(edge_dir.dot(up)) > 0.99:
            up = Vector((1, 0, 0))
            
        perp1 = edge_dir.cross(up).normalized()
        perp2 = edge_dir.cross(perp1).normalized()
        
        # Create box vertices
        half_thick = thickness / 2
        
        offsets = [
            perp1 * half_thick + perp2 * half_thick,
            perp1 * half_thick - perp2 * half_thick,
            -perp1 * half_thick - perp2 * half_thick,
            -perp1 * half_thick + perp2 * half_thick
        ]
        
        # Add vertices
        for offset in offsets:
            geometry['vertices'].append(v1 + offset)
        for offset in offsets:
            geometry['vertices'].append(v2 + offset)
            
        # Create faces (box)
        # Bottom
        geometry['faces'].append([0, 3, 2, 1])
        # Top
        geometry['faces'].append([4, 5, 6, 7])
        # Sides
        geometry['faces'].append([0, 1, 5, 4])
        geometry['faces'].append([1, 2, 6, 5])
        geometry['faces'].append([2, 3, 7, 6])
        geometry['faces'].append([3, 0, 4, 7])
        
        return geometry


class AdaptiveIslands(BasePattern):
    """Adaptive island pattern that concentrates material where needed"""
    
    def __init__(self):
        super().__init__()
        self.name = "Adaptive Islands"
        self.category = "Surface"
        self.description = "Islands of reinforcement adapted to stress patterns"
        
    def get_default_params(self) -> Dict:
        return {
            'island_size': 0.03,  # Average island size
            'min_island_size': 0.01,  # Minimum island size
            'max_island_size': 0.06,  # Maximum island size
            'coverage': 0.4,  # Surface coverage ratio (0-1)
            'edge_priority': 0.7,  # Priority for edge areas (0-1)
            'curvature_priority': 0.6,  # Priority for high curvature (0-1)
            'merge_distance': 0.005,  # Distance to merge islands
            'smoothness': 0.5,  # Island edge smoothness
            'pattern_offset': 0.001  # Offset from surface
        }
        
    def validate_params(self, params: Dict) -> Tuple[bool, str]:
        if params.get('min_island_size', 0) <= 0:
            return False, "Minimum island size must be positive"
        if params.get('max_island_size', 0) <= params.get('min_island_size', 0):
            return False, "Maximum island size must be larger than minimum"
        if not 0 <= params.get('coverage', 0.4) <= 1:
            return False, "Coverage must be between 0 and 1"
        return True, ""
        
    def generate(self, target_object: bpy.types.Object, **params) -> PatternResult:
        # Merge with defaults
        full_params = self.get_default_params()
        full_params.update(params)
        
        # Validate
        valid, error = self.validate_params(full_params)
        if not valid:
            return PatternResult([], [], [], {}, {}, False, error)
            
        # Analyze surface for island placement
        analysis = self._analyze_surface_priority(
            target_object,
            full_params['edge_priority'],
            full_params['curvature_priority']
        )
        
        # Generate island seeds
        seeds = self._generate_island_seeds(
            target_object,
            analysis,
            full_params['coverage'],
            full_params['island_size']
        )
        
        # Grow islands
        islands = self._grow_islands(
            seeds,
            analysis,
            full_params['min_island_size'],
            full_params['max_island_size'],
            target_object
        )
        
        # Merge nearby islands
        islands = self._merge_islands(islands, full_params['merge_distance'])
        
        # Generate geometry
        pattern_verts = []
        pattern_edges = []
        pattern_faces = []
        
        for island in islands:
            island_geom = self._create_island_geometry(
                island,
                full_params['smoothness'],
                target_object
            )
            
            # Add to pattern
            base_idx = len(pattern_verts)
            pattern_verts.extend(island_geom['vertices'])
            
            for edge in island_geom['edges']:
                pattern_edges.append((edge[0] + base_idx, edge[1] + base_idx))
                
            for face in island_geom['faces']:
                pattern_faces.append([v + base_idx for v in face])
                
        # Apply offset
        pattern_verts = self.offset_from_surface(
            pattern_verts,
            target_object,
            full_params['pattern_offset']
        )
        
        # Create attributes
        attributes = {
            'thickness': [0.003] * len(pattern_verts),
            'material_index': [1] * len(pattern_faces)
        }
        
        metadata = {
            'pattern_type': 'adaptive_islands',
            'island_count': len(islands),
            'coverage_ratio': len(islands) * full_params['island_size']**2 / self._get_surface_area(target_object),
            'parameters': full_params
        }
        
        return PatternResult(
            vertices=pattern_verts,
            edges=pattern_edges,
            faces=pattern_faces,
            attributes=attributes,
            metadata=metadata
        )
        
    def _analyze_surface_priority(self,
                                obj: bpy.types.Object,
                                edge_priority: float,
                                curvature_priority: float) -> Dict[Vector, float]:
        """Analyze surface to determine priority areas"""
        priority_map = {}
        
        mesh = obj.data
        
        # Get edge information
        edges = self.edge_detector.detect_edges(obj)
        edge_verts = set()
        for edge in edges:
            edge_verts.add(edge[0])
            edge_verts.add(edge[1])
            
        # Analyze each vertex
        for vert in mesh.vertices:
            priority = 0.3  # Base priority
            
            # Edge influence
            if vert.index in edge_verts:
                priority += edge_priority * 0.7
                
            # Curvature influence (simplified)
            if len(vert.link_faces) > 1:
                normals = [mesh.polygons[f].normal for f in vert.link_faces]
                avg_normal = sum(normals, Vector()) / len(normals)
                deviation = sum((n - avg_normal).length for n in normals) / len(normals)
                priority += curvature_priority * min(deviation * 2, 1.0)
                
            priority_map[vert.co.copy()] = min(priority, 1.0)
            
        return priority_map
        
    def _generate_island_seeds(self,
                             obj: bpy.types.Object,
                             priority_map: Dict[Vector, float],
                             coverage: float,
                             island_size: float) -> List[Dict]:
        """Generate seed points for islands based on priority"""
        seeds = []
        
        # Calculate number of seeds needed
        surface_area = self._get_surface_area(obj)
        island_area = math.pi * (island_size / 2) ** 2
        num_seeds = int(surface_area * coverage / island_area)
        
        # Sample points weighted by priority
        all_points = list(priority_map.keys())
        priorities = list(priority_map.values())
        
        # Normalize priorities for probability
        total_priority = sum(priorities)
        if total_priority > 0:
            probabilities = [p / total_priority for p in priorities]
        else:
            probabilities = [1.0 / len(priorities)] * len(priorities)
            
        # Sample seeds
        selected_indices = np.random.choice(
            len(all_points),
            size=min(num_seeds, len(all_points)),
            replace=False,
            p=probabilities
        )
        
        for idx in selected_indices:
            seeds.append({
                'position': all_points[idx],
                'priority': priorities[idx],
                'size': island_size
            })
            
        return seeds
        
    def _grow_islands(self,
                     seeds: List[Dict],
                     priority_map: Dict[Vector, float],
                     min_size: float,
                     max_size: float,
                     obj: bpy.types.Object) -> List[Dict]:
        """Grow islands from seeds based on local priority"""
        islands = []
        
        for seed in seeds:
            # Adjust size based on priority
            size = seed['size'] * (0.5 + seed['priority'])
            size = max(min_size, min(max_size, size))
            
            island = {
                'center': seed['position'],
                'size': size,
                'priority': seed['priority'],
                'points': []
            }
            
            # Collect points within island radius
            for point, priority in priority_map.items():
                if (point - seed['position']).length < size / 2:
                    island['points'].append(point)
                    
            if island['points']:
                islands.append(island)
                
        return islands
        
    def _merge_islands(self, islands: List[Dict], merge_distance: float) -> List[Dict]:
        """Merge nearby islands"""
        if not islands:
            return islands
            
        merged = []
        used = set()
        
        for i, island1 in enumerate(islands):
            if i in used:
                continue
                
            # Start new merged island
            merged_island = island1.copy()
            merged_island['points'] = island1['points'].copy()
            used.add(i)
            
            # Check for nearby islands to merge
            for j, island2 in enumerate(islands[i+1:], i+1):
                if j in used:
                    continue
                    
                # Check distance between centers
                if (island1['center'] - island2['center']).length < merge_distance + (island1['size'] + island2['size']) / 2:
                    # Merge
                    merged_island['points'].extend(island2['points'])
                    merged_island['size'] = max(merged_island['size'], island2['size'])
                    used.add(j)
                    
            merged.append(merged_island)
            
        return merged
        
    def _create_island_geometry(self,
                              island: Dict,
                              smoothness: float,
                              obj: bpy.types.Object) -> Dict:
        """Create geometry for a single island"""
        if not island['points']:
            return {'vertices': [], 'edges': [], 'faces': []}
            
        # Create convex hull of points
        points_2d = self._project_to_2d(island['points'], island['center'])
        
        if len(points_2d) < 3:
            return {'vertices': [], 'edges': [], 'faces': []}
            
        # Use scipy's ConvexHull
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(points_2d)
        except:
            return {'vertices': [], 'edges': [], 'faces': []}
            
        # Create offset polygon for frame
        hull_points = [island['points'][i] for i in hull.vertices]
        
        # Smooth if needed
        if smoothness > 0:
            hull_points = self._smooth_polygon(hull_points, smoothness)
            
        # Create frame geometry
        return self._create_polygon_frame(hull_points, 0.003)
        
    def _project_to_2d(self, points: List[Vector], center: Vector) -> np.ndarray:
        """Project 3D points to 2D for hull calculation"""
        if not points:
            return np.array([])
            
        # Simple projection - in production use proper parameterization
        points_2d = []
        for p in points:
            offset = p - center
            points_2d.append([offset.x, offset.y])
            
        return np.array(points_2d)
        
    def _smooth_polygon(self, points: List[Vector], smoothness: float) -> List[Vector]:
        """Smooth polygon using subdivision"""
        if len(points) < 3 or smoothness <= 0:
            return points
            
        smooth_points = []
        n = len(points)
        
        for i in range(n):
            prev_idx = (i - 1) % n
            next_idx = (i + 1) % n
            
            # Get neighboring points
            prev_point = points[prev_idx]
            curr_point = points[i]
            next_point = points[next_idx]
            
            # Smooth position
            smooth_pos = curr_point * (1 - smoothness) + (prev_point + next_point) * (smoothness / 2)
            smooth_points.append(smooth_pos)
            
        return smooth_points
        
    def _create_polygon_frame(self, points: List[Vector], thickness: float) -> Dict:
        """Create frame geometry from polygon"""
        geometry = {
            'vertices': [],
            'edges': [],
            'faces': []
        }
        
        if len(points) < 3:
            return geometry
            
        # Calculate polygon center
        center = sum(points, Vector()) / len(points)
        
        # Create inner and outer vertices
        for point in points:
            # Outer vertex
            geometry['vertices'].append(point)
            
            # Inner vertex
            direction = (center - point).normalized()
            inner = point + direction * thickness
            geometry['vertices'].append(inner)
            
        # Create faces
        n = len(points)
        for i in range(n):
            next_i = (i + 1) % n
            
            outer_curr = i * 2
            inner_curr = i * 2 + 1
            outer_next = next_i * 2
            inner_next = next_i * 2 + 1
            
            # Quad face
            geometry['faces'].append([
                outer_curr, outer_next,
                inner_next, inner_curr
            ])
            
        return geometry
        
    def _get_surface_area(self, obj: bpy.types.Object) -> float:
        """Calculate approximate surface area"""
        mesh = obj.data
        area = 0.0
        
        for poly in mesh.polygons:
            area += poly.area
            
        return area
