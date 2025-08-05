"""
Hybrid Pattern Implementations for Chain Tool V4
Combined pattern strategies for optimized reinforcement
"""

import bpy
import bmesh
import numpy as np
from mathutils import Vector, Matrix
from typing import List, Dict, Tuple, Set, Optional, Union
import math

from .base_pattern import BasePattern, PatternResult
from .surface_patterns import VoronoiPattern, HexagonalGrid, TriangularMesh
from .edge_patterns import RimReinforcement, ContourBanding
from ..utils.math_utils import MathUtils

class CoreShellPattern(BasePattern):
    """Core-shell pattern with dense core and lighter shell"""
    
    def __init__(self):
        super().__init__()
        self.name = "Core Shell"
        self.category = "Hybrid"
        self.description = "Dense core pattern transitioning to lighter shell"
        self.math_utils = MathUtils()
        
    def get_default_params(self) -> Dict:
        return {
            'core_pattern': 'hexagonal',  # Pattern type for core
            'shell_pattern': 'voronoi',  # Pattern type for shell
            'core_ratio': 0.4,  # Size of core region (0-1)
            'transition_width': 0.02,  # Width of transition zone
            'core_density': 0.01,  # Pattern density in core
            'shell_density': 0.03,  # Pattern density in shell
            'blend_mode': 'smooth',  # 'smooth', 'stepped', 'gradient'
            'center_mode': 'centroid',  # 'centroid', 'stress', 'manual'
            'pattern_offset': 0.001  # Offset from surface
        }
        
    def validate_params(self, params: Dict) -> Tuple[bool, str]:
        if not 0 <= params.get('core_ratio', 0.4) <= 1:
            return False, "Core ratio must be between 0 and 1"
        if params.get('transition_width', 0) < 0:
            return False, "Transition width must be non-negative"
        if params.get('core_density', 0) <= 0 or params.get('shell_density', 0) <= 0:
            return False, "Densities must be positive"
        return True, ""
        
    def generate(self, target_object: bpy.types.Object, **params) -> PatternResult:
        # Merge with defaults
        full_params = self.get_default_params()
        full_params.update(params)
        
        # Validate
        valid, error = self.validate_params(full_params)
        if not valid:
            return PatternResult([], [], [], {}, {}, False, error)
            
        # Determine core center
        core_center = self._determine_core_center(
            target_object,
            full_params['center_mode']
        )
        
        # Calculate core and shell regions
        regions = self._calculate_regions(
            target_object,
            core_center,
            full_params['core_ratio'],
            full_params['transition_width']
        )
        
        # Generate patterns for each region
        core_result = self._generate_region_pattern(
            target_object,
            regions['core'],
            full_params['core_pattern'],
            full_params['core_density']
        )
        
        shell_result = self._generate_region_pattern(
            target_object,
            regions['shell'],
            full_params['shell_pattern'],
            full_params['shell_density']
        )
        
        # Generate transition if needed
        if full_params['transition_width'] > 0 and full_params['blend_mode'] != 'stepped':
            transition_result = self._generate_transition_pattern(
                target_object,
                regions['transition'],
                core_result,
                shell_result,
                full_params['blend_mode']
            )
        else:
            transition_result = None
            
        # Combine results
        combined = self._combine_pattern_results(
            [core_result, shell_result, transition_result],
            full_params['pattern_offset'],
            target_object
        )
        
        # Add metadata
        combined.metadata = {
            'pattern_type': 'core_shell',
            'core_center': list(core_center),
            'core_vertices': len(core_result.vertices),
            'shell_vertices': len(shell_result.vertices),
            'parameters': full_params
        }
        
        return combined
        
    def _determine_core_center(self, obj: bpy.types.Object, mode: str) -> Vector:
        """Determine center point for core region"""
        if mode == 'centroid':
            # Use object centroid
            mesh = obj.data
            center = sum((v.co for v in mesh.vertices), Vector()) / len(mesh.vertices)
            return obj.matrix_world @ center
            
        elif mode == 'stress':
            # Find highest stress concentration
            # Simplified - use lowest point (highest load)
            mesh = obj.data
            lowest_vert = min(mesh.vertices, key=lambda v: (obj.matrix_world @ v.co).z)
            return obj.matrix_world @ lowest_vert.co
            
        else:  # manual
            # Use object origin
            return obj.location.copy()
            
    def _calculate_regions(self,
                         obj: bpy.types.Object,
                         center: Vector,
                         core_ratio: float,
                         transition_width: float) -> Dict[str, List[Vector]]:
        """Calculate core, shell, and transition regions"""
        regions = {
            'core': [],
            'shell': [],
            'transition': []
        }
        
        # Calculate max distance from center
        mesh = obj.data
        max_dist = 0.0
        
        for vert in mesh.vertices:
            world_co = obj.matrix_world @ vert.co
            dist = (world_co - center).length
            max_dist = max(max_dist, dist)
            
        # Classify vertices
        core_radius = max_dist * core_ratio
        transition_outer = core_radius + transition_width
        
        for vert in mesh.vertices:
            world_co = obj.matrix_world @ vert.co
            dist = (world_co - center).length
            
            if dist <= core_radius:
                regions['core'].append(vert.co.copy())
            elif dist <= transition_outer:
                regions['transition'].append(vert.co.copy())
            else:
                regions['shell'].append(vert.co.copy())
                
        return regions
        
    def _generate_region_pattern(self,
                               obj: bpy.types.Object,
                               region_points: List[Vector],
                               pattern_type: str,
                               density: float) -> PatternResult:
        """Generate pattern for a specific region"""
        if not region_points:
            return PatternResult([], [], [], {}, {})
            
        # Create temporary mesh for region
        region_mesh = self._create_region_mesh(obj, region_points)
        
        # Generate pattern based on type
        if pattern_type == 'hexagonal':
            pattern = HexagonalGrid()
            result = pattern.generate(region_mesh, hex_size=density)
        elif pattern_type == 'voronoi':
            pattern = VoronoiPattern()
            result = pattern.generate(region_mesh, cell_size=density)
        elif pattern_type == 'triangular':
            pattern = TriangularMesh()
            result = pattern.generate(region_mesh, point_density=density)
        else:
            result = PatternResult([], [], [], {}, {}, False, f"Unknown pattern type: {pattern_type}")
            
        # Clean up temporary mesh
        bpy.data.objects.remove(region_mesh)
        bpy.data.meshes.remove(region_mesh.data)
        
        return result
        
    def _create_region_mesh(self, obj: bpy.types.Object, region_points: List[Vector]) -> bpy.types.Object:
        """Create temporary mesh for region"""
        # Create new mesh
        mesh = bpy.data.meshes.new("TempRegion")
        region_obj = bpy.data.objects.new("TempRegion", mesh)
        
        # Add vertices
        mesh.vertices.add(len(region_points))
        for i, point in enumerate(region_points):
            mesh.vertices[i].co = point
            
        # Copy relevant faces from original
        original_mesh = obj.data
        faces_to_copy = []
        
        region_indices = set()
        for i, vert in enumerate(original_mesh.vertices):
            if vert.co in region_points:
                region_indices.add(i)
                
        for face in original_mesh.polygons:
            if all(v in region_indices for v in face.vertices):
                faces_to_copy.append([region_points.index(original_mesh.vertices[v].co) 
                                    for v in face.vertices])
                
        # Add faces
        if faces_to_copy:
            mesh.loops.add(sum(len(f) for f in faces_to_copy))
            mesh.polygons.add(len(faces_to_copy))
            
            loop_idx = 0
            for poly_idx, face_verts in enumerate(faces_to_copy):
                poly = mesh.polygons[poly_idx]
                poly.loop_start = loop_idx
                poly.loop_total = len(face_verts)
                
                for vert_idx in face_verts:
                    mesh.loops[loop_idx].vertex_index = vert_idx
                    loop_idx += 1
                    
        mesh.update()
        region_obj.matrix_world = obj.matrix_world.copy()
        
        return region_obj
        
    def _generate_transition_pattern(self,
                                   obj: bpy.types.Object,
                                   transition_points: List[Vector],
                                   core_result: PatternResult,
                                   shell_result: PatternResult,
                                   blend_mode: str) -> PatternResult:
        """Generate blended pattern for transition zone"""
        if not transition_points:
            return PatternResult([], [], [], {}, {})
            
        if blend_mode == 'smooth':
            # Gradual blend between patterns
            # Sample from both patterns with distance-based weighting
            transition_verts = []
            
            # This is simplified - in production, properly blend geometries
            # For now, just use averaged density
            avg_density = (core_result.metadata.get('density', 0.02) + 
                          shell_result.metadata.get('density', 0.02)) / 2
                          
            # Generate intermediate pattern
            pattern = VoronoiPattern()  # Use Voronoi for smooth transitions
            temp_mesh = self._create_region_mesh(obj, transition_points)
            result = pattern.generate(temp_mesh, cell_size=avg_density)
            
            # Clean up
            bpy.data.objects.remove(temp_mesh)
            bpy.data.meshes.remove(temp_mesh.data)
            
            return result
            
        else:  # gradient
            # Create gradient of densities
            return self._generate_gradient_transition(
                obj,
                transition_points,
                core_result,
                shell_result
            )
            
    def _generate_gradient_transition(self,
                                    obj: bpy.types.Object,
                                    points: List[Vector],
                                    core: PatternResult,
                                    shell: PatternResult) -> PatternResult:
        """Generate gradient density transition"""
        # This would implement a smooth density gradient
        # For now, return empty pattern
        return PatternResult([], [], [], {}, {})
        
    def _combine_pattern_results(self,
                               results: List[Optional[PatternResult]],
                               offset: float,
                               obj: bpy.types.Object) -> PatternResult:
        """Combine multiple pattern results"""
        combined_verts = []
        combined_edges = []
        combined_faces = []
        combined_attributes = {}
        
        for result in results:
            if not result or not result.success:
                continue
                
            # Add vertices with offset
            base_idx = len(combined_verts)
            
            for vert in result.vertices:
                combined_verts.append(vert)
                
            # Add edges with adjusted indices
            for edge in result.edges:
                combined_edges.append((edge[0] + base_idx, edge[1] + base_idx))
                
            # Add faces with adjusted indices
            for face in result.faces:
                combined_faces.append([v + base_idx for v in face])
                
            # Merge attributes
            for key, values in result.attributes.items():
                if key not in combined_attributes:
                    combined_attributes[key] = []
                combined_attributes[key].extend(values)
                
        # Apply surface offset
        combined_verts = self.offset_from_surface(combined_verts, obj, offset)
        
        return PatternResult(
            vertices=combined_verts,
            edges=combined_edges,
            faces=combined_faces,
            attributes=combined_attributes,
            metadata={},
            success=True
        )


class GradientDensity(BasePattern):
    """Pattern with gradient density based on various factors"""
    
    def __init__(self):
        super().__init__()
        self.name = "Gradient Density"
        self.category = "Hybrid"
        self.description = "Variable density pattern based on gradients"
        
    def get_default_params(self) -> Dict:
        return {
            'base_pattern': 'voronoi',  # Base pattern type
            'gradient_type': 'radial',  # 'radial', 'linear', 'stress', 'curvature'
            'min_density': 0.005,  # Minimum pattern density
            'max_density': 0.03,  # Maximum pattern density
            'gradient_power': 2.0,  # Power curve for gradient (1=linear)
            'gradient_direction': (0, 0, 1),  # Direction for linear gradient
            'gradient_center': None,  # Center for radial gradient
            'invert_gradient': False,  # Invert density distribution
            'pattern_offset': 0.001  # Offset from surface
        }
        
    def validate_params(self, params: Dict) -> Tuple[bool, str]:
        if params.get('min_density', 0) <= 0:
            return False, "Minimum density must be positive"
        if params.get('max_density', 0) <= params.get('min_density', 0):
            return False, "Maximum density must be greater than minimum"
        if params.get('gradient_power', 1) <= 0:
            return False, "Gradient power must be positive"
        return True, ""
        
    def generate(self, target_object: bpy.types.Object, **params) -> PatternResult:
        # Merge with defaults
        full_params = self.get_default_params()
        full_params.update(params)
        
        # Validate
        valid, error = self.validate_params(full_params)
        if not valid:
            return PatternResult([], [], [], {}, {}, False, error)
            
        # Calculate gradient field
        gradient_field = self._calculate_gradient_field(
            target_object,
            full_params['gradient_type'],
            full_params['gradient_direction'],
            full_params['gradient_center'],
            full_params['gradient_power'],
            full_params['invert_gradient']
        )
        
        # Generate adaptive samples based on gradient
        sample_points = self._generate_gradient_samples(
            target_object,
            gradient_field,
            full_params['min_density'],
            full_params['max_density']
        )
        
        # Generate pattern
        pattern_result = self._generate_base_pattern(
            sample_points,
            target_object,
            full_params['base_pattern']
        )
        
        # Apply gradient-based attributes
        pattern_result = self._apply_gradient_attributes(
            pattern_result,
            gradient_field,
            full_params
        )
        
        # Apply offset
        pattern_result.vertices = self.offset_from_surface(
            pattern_result.vertices,
            target_object,
            full_params['pattern_offset']
        )
        
        # Add metadata
        pattern_result.metadata = {
            'pattern_type': 'gradient_density',
            'gradient_type': full_params['gradient_type'],
            'sample_count': len(sample_points),
            'parameters': full_params
        }
        
        return pattern_result
        
    def _calculate_gradient_field(self,
                                obj: bpy.types.Object,
                                gradient_type: str,
                                direction: Tuple[float, float, float],
                                center: Optional[Vector],
                                power: float,
                                invert: bool) -> Dict[Vector, float]:
        """Calculate gradient values across surface"""
        gradient_field = {}
        mesh = obj.data
        
        # Get bounds for normalization
        world_verts = [obj.matrix_world @ v.co for v in mesh.vertices]
        
        if gradient_type == 'radial':
            # Radial gradient from center
            if center is None:
                center = sum(world_verts, Vector()) / len(world_verts)
            else:
                center = Vector(center)
                
            # Calculate max distance
            max_dist = max((v - center).length for v in world_verts)
            
            for vert in mesh.vertices:
                world_co = obj.matrix_world @ vert.co
                dist = (world_co - center).length
                value = (dist / max_dist) ** power
                
                if invert:
                    value = 1.0 - value
                    
                gradient_field[vert.co.copy()] = value
                
        elif gradient_type == 'linear':
            # Linear gradient along direction
            direction = Vector(direction).normalized()
            
            # Project vertices onto direction
            projections = [(v.dot(direction), v) for v in world_verts]
            min_proj = min(p[0] for p in projections)
            max_proj = max(p[0] for p in projections)
            proj_range = max_proj - min_proj
            
            for i, vert in enumerate(mesh.vertices):
                world_co = obj.matrix_world @ vert.co
                projection = world_co.dot(direction)
                value = ((projection - min_proj) / proj_range) ** power
                
                if invert:
                    value = 1.0 - value
                    
                gradient_field[vert.co.copy()] = value
                
        elif gradient_type == 'stress':
            # Stress-based gradient
            stress_map = self._calculate_stress_gradient(obj)
            
            for vert_co, stress in stress_map.items():
                value = stress ** power
                
                if invert:
                    value = 1.0 - value
                    
                gradient_field[vert_co] = value
                
        elif gradient_type == 'curvature':
            # Curvature-based gradient
            curvature_map = self._calculate_curvature_gradient(obj)
            
            for vert_co, curvature in curvature_map.items():
                value = curvature ** power
                
                if invert:
                    value = 1.0 - value
                    
                gradient_field[vert_co] = value
                
        return gradient_field
        
    def _calculate_stress_gradient(self, obj: bpy.types.Object) -> Dict[Vector, float]:
        """Calculate stress-based gradient"""
        stress_map = {}
        mesh = obj.data
        
        # Simplified stress calculation
        # Higher stress at: edges, support points, high curvature
        
        edges = self.edge_detector.detect_edges(obj)
        edge_verts = set()
        for edge in edges:
            edge_verts.add(edge[0])
            edge_verts.add(edge[1])
            
        for vert in mesh.vertices:
            stress = 0.3  # Base stress
            
            # Edge proximity
            if vert.index in edge_verts:
                stress += 0.4
                
            # Bottom vertices (support)
            z_normalized = (vert.co.z - min(v.co.z for v in mesh.vertices)) / \
                          (max(v.co.z for v in mesh.vertices) - min(v.co.z for v in mesh.vertices))
            if z_normalized < 0.2:
                stress += 0.3 * (1 - z_normalized * 5)
                
            stress_map[vert.co.copy()] = min(stress, 1.0)
            
        return stress_map
        
    def _calculate_curvature_gradient(self, obj: bpy.types.Object) -> Dict[Vector, float]:
        """Calculate curvature-based gradient"""
        curvature_map = {}
        mesh = obj.data
        
        for vert in mesh.vertices:
            if len(vert.link_faces) > 1:
                # Calculate normal deviation
                normals = [mesh.polygons[f].normal for f in vert.link_faces]
                avg_normal = sum(normals, Vector()) / len(normals)
                deviation = sum((n - avg_normal).length for n in normals) / len(normals)
                curvature = min(deviation * 2.0, 1.0)
            else:
                curvature = 0.0
                
            curvature_map[vert.co.copy()] = curvature
            
        return curvature_map
        
    def _generate_gradient_samples(self,
                                 obj: bpy.types.Object,
                                 gradient_field: Dict[Vector, float],
                                 min_density: float,
                                 max_density: float) -> List[Vector]:
        """Generate sample points with gradient-based density"""
        samples = []
        
        # Use rejection sampling based on gradient
        mesh = obj.data
        surface_area = sum(p.area for p in mesh.polygons)
        
        # Estimate required samples
        avg_density = (min_density + max_density) / 2
        estimated_samples = int(surface_area / (avg_density ** 2))
        
        # Generate samples
        attempts = 0
        max_attempts = estimated_samples * 10
        
        while len(samples) < estimated_samples and attempts < max_attempts:
            attempts += 1
            
            # Random face weighted by area
            face_idx = np.random.choice(
                len(mesh.polygons),
                p=[f.area / surface_area for f in mesh.polygons]
            )
            face = mesh.polygons[face_idx]
            
            # Random point on face
            verts = [mesh.vertices[i].co for i in face.vertices]
            if len(verts) == 3:
                # Barycentric coordinates
                r1, r2 = np.random.random(2)
                if r1 + r2 > 1:
                    r1, r2 = 1 - r1, 1 - r2
                point = verts[0] * r1 + verts[1] * r2 + verts[2] * (1 - r1 - r2)
            else:
                # Simple average for quads
                point = sum(verts, Vector()) / len(verts)
                
            # Get gradient value at point
            # Find nearest vertex gradient
            min_dist = float('inf')
            gradient_value = 0.5
            
            for vert_co, grad in gradient_field.items():
                dist = (vert_co - point).length
                if dist < min_dist:
                    min_dist = dist
                    gradient_value = grad
                    
            # Calculate local density
            local_density = min_density + (max_density - min_density) * gradient_value
            
            # Rejection test
            if len(samples) == 0:
                samples.append(point)
            else:
                # Check minimum distance to existing samples
                min_dist_to_samples = min((point - s).length for s in samples)
                
                if min_dist_to_samples >= local_density:
                    samples.append(point)
                    
        return samples
        
    def _generate_base_pattern(self,
                             sample_points: List[Vector],
                             obj: bpy.types.Object,
                             pattern_type: str) -> PatternResult:
        """Generate pattern from sample points"""
        if pattern_type == 'voronoi':
            # Direct Voronoi from samples
            pattern = VoronoiPattern()
            
            # Create temporary object with samples
            temp_mesh = bpy.data.meshes.new("TempSamples")
            temp_obj = bpy.data.objects.new("TempSamples", temp_mesh)
            
            temp_mesh.vertices.add(len(sample_points))
            for i, point in enumerate(sample_points):
                temp_mesh.vertices[i].co = point
                
            temp_mesh.update()
            temp_obj.matrix_world = obj.matrix_world.copy()
            
            # Generate pattern
            result = pattern.generate(temp_obj)
            
            # Clean up
            bpy.data.objects.remove(temp_obj)
            bpy.data.meshes.remove(temp_mesh)
            
            return result
            
        else:
            # Use triangulation for other patterns
            tri_engine = TriangulationEngine()
            triangulation = tri_engine.triangulate_surface(sample_points)
            
            return PatternResult(
                vertices=triangulation['vertices'],
                edges=triangulation['edges'],
                faces=triangulation['faces'],
                attributes={},
                metadata={}
            )
            
    def _apply_gradient_attributes(self,
                                 result: PatternResult,
                                 gradient_field: Dict[Vector, float],
                                 params: Dict) -> PatternResult:
        """Apply gradient-based attributes to pattern"""
        # Calculate thickness based on gradient
        thicknesses = []
        
        for vert in result.vertices:
            # Find gradient value
            min_dist = float('inf')
            gradient_value = 0.5
            
            for vert_co, grad in gradient_field.items():
                dist = (vert_co - vert).length
                if dist < min_dist:
                    min_dist = dist
                    gradient_value = grad
                    
            # Variable thickness
            thickness = 0.002 + gradient_value * 0.003
            thicknesses.append(thickness)
            
        result.attributes['thickness'] = thicknesses
        result.attributes['gradient_value'] = [
            self._get_nearest_gradient(v, gradient_field) for v in result.vertices
        ]
        
        return result
        
    def _get_nearest_gradient(self, point: Vector, gradient_field: Dict[Vector, float]) -> float:
        """Get gradient value for a point"""
        min_dist = float('inf')
        value = 0.5
        
        for vert_co, grad in gradient_field.items():
            dist = (vert_co - point).length
            if dist < min_dist:
                min_dist = dist
                value = grad
                
        return value


class ZonalReinforcement(BasePattern):
    """Zone-based reinforcement with different patterns per zone"""
    
    def __init__(self):
        super().__init__()
        self.name = "Zonal Reinforcement"
        self.category = "Hybrid"
        self.description = "Different reinforcement patterns for different zones"
        
    def get_default_params(self) -> Dict:
        return {
            'zones': [
                {
                    'name': 'high_stress',
                    'pattern': 'triangular',
                    'density': 0.01,
                    'selection': 'stress',  # 'stress', 'curvature', 'height', 'paint'
                    'threshold': 0.7
                },
                {
                    'name': 'edges',
                    'pattern': 'rim',
                    'density': 0.015,
                    'selection': 'edges',
                    'threshold': 0.5
                },
                {
                    'name': 'general',
                    'pattern': 'voronoi',
                    'density': 0.025,
                    'selection': 'remaining',
                    'threshold': 0.0
                }
            ],
            'zone_overlap': 0.1,  # Overlap between zones (0-1)
            'blend_zones': True,  # Blend patterns at zone boundaries
            'pattern_offset': 0.001  # Offset from surface
        }
        
    def validate_params(self, params: Dict) -> Tuple[bool, str]:
        zones = params.get('zones', [])
        if not zones:
            return False, "At least one zone must be defined"
            
        for zone in zones:
            if 'pattern' not in zone or 'density' not in zone:
                return False, "Each zone must have pattern and density"
            if zone.get('density', 0) <= 0:
                return False, f"Zone {zone.get('name', 'unnamed')} density must be positive"
                
        return True, ""
        
    def generate(self, target_object: bpy.types.Object, **params) -> PatternResult:
        # Merge with defaults
        full_params = self.get_default_params()
        full_params.update(params)
        
        # Validate
        valid, error = self.validate_params(full_params)
        if not valid:
            return PatternResult([], [], [], {}, {}, False, error)
            
        # Analyze object for zone assignment
        analysis = self._analyze_object_zones(target_object)
        
        # Assign vertices to zones
        zone_assignments = self._assign_zones(
            target_object,
            full_params['zones'],
            analysis,
            full_params['zone_overlap']
        )
        
        # Generate patterns for each zone
        zone_results = []
        
        for zone_def in full_params['zones']:
            zone_name = zone_def['name']
            
            if zone_name not in zone_assignments or not zone_assignments[zone_name]:
                continue
                
            # Generate zone pattern
            zone_result = self._generate_zone_pattern(
                target_object,
                zone_assignments[zone_name],
                zone_def
            )
            
            if zone_result.success:
                zone_results.append((zone_name, zone_result))
                
        # Blend zones if enabled
        if full_params['blend_zones'] and len(zone_results) > 1:
            blended_results = self._blend_zone_boundaries(
                zone_results,
                zone_assignments,
                target_object
            )
        else:
            blended_results = zone_results
            
        # Combine all zone patterns
        combined = self._combine_zone_results(
            blended_results,
            full_params['pattern_offset'],
            target_object
        )
        
        # Add metadata
        combined.metadata = {
            'pattern_type': 'zonal_reinforcement',
            'zone_count': len(zone_results),
            'zones': [z[0] for z in zone_results],
            'parameters': full_params
        }
        
        return combined
        
    def _analyze_object_zones(self, obj: bpy.types.Object) -> Dict:
        """Analyze object for zone characteristics"""
        analysis = {}
        mesh = obj.data
        
        # Stress analysis
        stress_values = {}
        edge_verts = set()
        edges = self.edge_detector.detect_edges(obj)
        for edge in edges:
            edge_verts.add(edge[0])
            edge_verts.add(edge[1])
            
        for vert in mesh.vertices:
            stress = 0.3
            if vert.index in edge_verts:
                stress += 0.4
            # Add height-based stress (lower = higher stress)
            z_norm = (vert.co.z - min(v.co.z for v in mesh.vertices)) / \
                    (max(v.co.z for v in mesh.vertices) - min(v.co.z for v in mesh.vertices))
            stress += (1 - z_norm) * 0.3
            stress_values[vert.index] = min(stress, 1.0)
            
        analysis['stress'] = stress_values
        
        # Curvature analysis
        curvature_values = {}
        for vert in mesh.vertices:
            if len(vert.link_faces) > 1:
                normals = [mesh.polygons[f].normal for f in vert.link_faces]
                avg_normal = sum(normals, Vector()) / len(normals)
                deviation = sum((n - avg_normal).length for n in normals) / len(normals)
                curvature_values[vert.index] = min(deviation * 2.0, 1.0)
            else:
                curvature_values[vert.index] = 0.0
                
        analysis['curvature'] = curvature_values
        
        # Height analysis
        height_values = {}
        z_values = [v.co.z for v in mesh.vertices]
        z_min, z_max = min(z_values), max(z_values)
        
        for vert in mesh.vertices:
            height_values[vert.index] = (vert.co.z - z_min) / (z_max - z_min)
            
        analysis['height'] = height_values
        
        # Edge detection
        analysis['edges'] = edge_verts
        
        # Paint analysis (if available)
        paint_weights = {}
        paint_data = self.state.get("paint_strokes", {}).get(obj.name, [])
        
        if paint_data:
            for stroke in paint_data:
                for point in stroke.get("points", []):
                    point_co = Vector(point["location"])
                    
                    # Find affected vertices
                    for vert in mesh.vertices:
                        dist = (vert.co - point_co).length
                        radius = stroke.get("radius", 0.05)
                        
                        if dist < radius:
                            influence = 1.0 - (dist / radius)
                            paint_weights[vert.index] = max(
                                paint_weights.get(vert.index, 0),
                                influence
                            )
                            
        analysis['paint'] = paint_weights
        
        return analysis
        
    def _assign_zones(self,
                     obj: bpy.types.Object,
                     zone_definitions: List[Dict],
                     analysis: Dict,
                     overlap: float) -> Dict[str, List[int]]:
        """Assign vertices to zones based on criteria"""
        zone_assignments = {zone['name']: [] for zone in zone_definitions}
        mesh = obj.data
        used_vertices = set()
        
        # Process zones in order (priority)
        for zone_def in zone_definitions:
            zone_name = zone_def['name']
            selection = zone_def.get('selection', 'remaining')
            threshold = zone_def.get('threshold', 0.5)
            
            if selection == 'stress':
                values = analysis['stress']
                for vert_idx, value in values.items():
                    if value >= threshold and (overlap > 0 or vert_idx not in used_vertices):
                        zone_assignments[zone_name].append(vert_idx)
                        used_vertices.add(vert_idx)
                        
            elif selection == 'curvature':
                values = analysis['curvature']
                for vert_idx, value in values.items():
                    if value >= threshold and (overlap > 0 or vert_idx not in used_vertices):
                        zone_assignments[zone_name].append(vert_idx)
                        used_vertices.add(vert_idx)
                        
            elif selection == 'height':
                values = analysis['height']
                for vert_idx, value in values.items():
                    if value >= threshold and (overlap > 0 or vert_idx not in used_vertices):
                        zone_assignments[zone_name].append(vert_idx)
                        used_vertices.add(vert_idx)
                        
            elif selection == 'edges':
                edge_verts = analysis['edges']
                for vert_idx in edge_verts:
                    if overlap > 0 or vert_idx not in used_vertices:
                        zone_assignments[zone_name].append(vert_idx)
                        used_vertices.add(vert_idx)
                        
            elif selection == 'paint':
                values = analysis['paint']
                for vert_idx, value in values.items():
                    if value >= threshold and (overlap > 0 or vert_idx not in used_vertices):
                        zone_assignments[zone_name].append(vert_idx)
                        used_vertices.add(vert_idx)
                        
            elif selection == 'remaining':
                # Assign all unused vertices
                for i in range(len(mesh.vertices)):
                    if i not in used_vertices:
                        zone_assignments[zone_name].append(i)
                        
        return zone_assignments
        
    def _generate_zone_pattern(self,
                             obj: bpy.types.Object,
                             vertex_indices: List[int],
                             zone_def: Dict) -> PatternResult:
        """Generate pattern for a specific zone"""
        if not vertex_indices:
            return PatternResult([], [], [], {}, {})
            
        # Get zone vertices
        mesh = obj.data
        zone_verts = [mesh.vertices[i].co.copy() for i in vertex_indices]
        
        # Create temporary mesh for zone
        zone_mesh = self._create_zone_mesh(obj, vertex_indices)
        
        # Generate pattern based on type
        pattern_type = zone_def['pattern']
        density = zone_def['density']
        
        if pattern_type == 'triangular':
            pattern = TriangularMesh()
            result = pattern.generate(zone_mesh, point_density=density)
        elif pattern_type == 'hexagonal':
            pattern = HexagonalGrid()
            result = pattern.generate(zone_mesh, hex_size=density)
        elif pattern_type == 'voronoi':
            pattern = VoronoiPattern()
            result = pattern.generate(zone_mesh, cell_size=density)
        elif pattern_type == 'rim':
            pattern = RimReinforcement()
            result = pattern.generate(zone_mesh, rim_width=density)
        else:
            result = PatternResult([], [], [], {}, {}, False, f"Unknown pattern: {pattern_type}")
            
        # Add zone identifier to attributes
        if result.success:
            result.attributes['zone'] = [zone_def['name']] * len(result.vertices)
            
        # Clean up
        bpy.data.objects.remove(zone_mesh)
        bpy.data.meshes.remove(zone_mesh.data)
        
        return result
        
    def _create_zone_mesh(self, obj: bpy.types.Object, vertex_indices: List[int]) -> bpy.types.Object:
        """Create temporary mesh for zone"""
        mesh = bpy.data.meshes.new("TempZone")
        zone_obj = bpy.data.objects.new("TempZone", mesh)
        
        # Get original mesh
        orig_mesh = obj.data
        
        # Add zone vertices
        zone_verts = [orig_mesh.vertices[i].co for i in vertex_indices]
        mesh.vertices.add(len(zone_verts))
        
        # Create vertex index mapping
        index_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(vertex_indices)}
        
        for i, vert_co in enumerate(zone_verts):
            mesh.vertices[i].co = vert_co
            
        # Add faces that belong entirely to zone
        faces_to_add = []
        vertex_set = set(vertex_indices)
        
        for face in orig_mesh.polygons:
            if all(v in vertex_set for v in face.vertices):
                # Map to new indices
                new_face = [index_map[v] for v in face.vertices]
                faces_to_add.append(new_face)
                
        # Create faces
        if faces_to_add:
            total_loops = sum(len(f) for f in faces_to_add)
            mesh.loops.add(total_loops)
            mesh.polygons.add(len(faces_to_add))
            
            loop_idx = 0
            for poly_idx, face_verts in enumerate(faces_to_add):
                poly = mesh.polygons[poly_idx]
                poly.loop_start = loop_idx
                poly.loop_total = len(face_verts)
                
                for vert_idx in face_verts:
                    mesh.loops[loop_idx].vertex_index = vert_idx
                    loop_idx += 1
                    
        mesh.update()
        zone_obj.matrix_world = obj.matrix_world.copy()
        
        return zone_obj
        
    def _blend_zone_boundaries(self,
                             zone_results: List[Tuple[str, PatternResult]],
                             zone_assignments: Dict[str, List[int]],
                             obj: bpy.types.Object) -> List[Tuple[str, PatternResult]]:
        """Blend patterns at zone boundaries"""
        # This would implement smooth transitions between zones
        # For now, return as-is
        return zone_results
        
    def _combine_zone_results(self,
                            zone_results: List[Tuple[str, PatternResult]],
                            offset: float,
                            obj: bpy.types.Object) -> PatternResult:
        """Combine all zone patterns into single result"""
        combined_verts = []
        combined_edges = []
        combined_faces = []
        combined_attributes = {}
        
        zone_info = []
        
        for zone_name, result in zone_results:
            if not result.success:
                continue
                
            # Add vertices
            base_idx = len(combined_verts)
            combined_verts.extend(result.vertices)
            
            # Add edges with adjusted indices
            for edge in result.edges:
                combined_edges.append((edge[0] + base_idx, edge[1] + base_idx))
                
            # Add faces with adjusted indices
            for face in result.faces:
                combined_faces.append([v + base_idx for v in face])
                
            # Merge attributes
            for key, values in result.attributes.items():
                if key not in combined_attributes:
                    combined_attributes[key] = []
                combined_attributes[key].extend(values)
                
            zone_info.append({
                'name': zone_name,
                'vertex_count': len(result.vertices),
                'face_count': len(result.faces)
            })
            
        # Apply surface offset
        combined_verts = self.offset_from_surface(combined_verts, obj, offset)
        
        return PatternResult(
            vertices=combined_verts,
            edges=combined_edges,
            faces=combined_faces,
            attributes=combined_attributes,
            metadata={'zones': zone_info},
            success=True
        )


class CombinedPattern(BasePattern):
    """Generic pattern combiner for custom combinations"""
    
    def __init__(self, pattern1: BasePattern, pattern2: BasePattern):
        super().__init__()
        self.pattern1 = pattern1
        self.pattern2 = pattern2
        self.name = f"Combined ({pattern1.name} + {pattern2.name})"
        self.category = "Hybrid"
        self.description = f"Combination of {pattern1.name} and {pattern2.name}"
        
    def get_default_params(self) -> Dict:
        # Merge parameters from both patterns
        params = {}
        params.update({f"p1_{k}": v for k, v in self.pattern1.get_default_params().items()})
        params.update({f"p2_{k}": v for k, v in self.pattern2.get_default_params().items()})
        
        # Add combination parameters
        params.update({
            'combination_mode': 'overlay',  # 'overlay', 'intersect', 'subtract'
            'blend_factor': 0.5,  # Weight between patterns (0=pattern1, 1=pattern2)
            'pattern_offset': 0.001
        })
        
        return params
        
    def validate_params(self, params: Dict) -> Tuple[bool, str]:
        # Validate pattern 1 params
        p1_params = {k[3:]: v for k, v in params.items() if k.startswith('p1_')}
        valid, error = self.pattern1.validate_params(p1_params)
        if not valid:
            return False, f"Pattern 1: {error}"
            
        # Validate pattern 2 params
        p2_params = {k[3:]: v for k, v in params.items() if k.startswith('p2_')}
        valid, error = self.pattern2.validate_params(p2_params)
        if not valid:
            return False, f"Pattern 2: {error}"
            
        return True, ""
        
    def generate(self, target_object: bpy.types.Object, **params) -> PatternResult:
        # Extract parameters
        p1_params = {k[3:]: v for k, v in params.items() if k.startswith('p1_')}
        p2_params = {k[3:]: v for k, v in params.items() if k.startswith('p2_')}
        
        # Generate both patterns
        result1 = self.pattern1.generate(target_object, **p1_params)
        result2 = self.pattern2.generate(target_object, **p2_params)
        
        if not result1.success:
            return result1
        if not result2.success:
            return result2
            
        # Combine based on mode
        mode = params.get('combination_mode', 'overlay')
        blend = params.get('blend_factor', 0.5)
        
        if mode == 'overlay':
            combined = self._overlay_patterns(result1, result2, blend)
        elif mode == 'intersect':
            combined = self._intersect_patterns(result1, result2)
        elif mode == 'subtract':
            combined = self._subtract_patterns(result1, result2)
        else:
            combined = result1  # Fallback
            
        # Apply offset
        combined.vertices = self.offset_from_surface(
            combined.vertices,
            target_object,
            params.get('pattern_offset', 0.001)
        )
        
        return combined
        
    def _overlay_patterns(self,
                         result1: PatternResult,
                         result2: PatternResult,
                         blend: float) -> PatternResult:
        """Overlay two patterns"""
        # Simple combination - merge both
        combined_verts = result1.vertices.copy()
        combined_edges = result1.edges.copy()
        combined_faces = result1.faces.copy()
        
        # Add second pattern with offset
        base_idx = len(combined_verts)
        combined_verts.extend(result2.vertices)
        
        for edge in result2.edges:
            combined_edges.append((edge[0] + base_idx, edge[1] + base_idx))
            
        for face in result2.faces:
            combined_faces.append([v + base_idx for v in face])
            
        return PatternResult(
            vertices=combined_verts,
            edges=combined_edges,
            faces=combined_faces,
            attributes={},
            metadata={'combination_mode': 'overlay', 'blend': blend}
        )
        
    def _intersect_patterns(self,
                          result1: PatternResult,
                          result2: PatternResult) -> PatternResult:
        """Keep only intersecting areas"""
        # This would require geometric intersection
        # For now, return first pattern
        return result1
        
    def _subtract_patterns(self,
                         result1: PatternResult,
                         result2: PatternResult) -> PatternResult:
        """Subtract second pattern from first"""
        # This would require geometric boolean operations
        # For now, return first pattern
        return result1
