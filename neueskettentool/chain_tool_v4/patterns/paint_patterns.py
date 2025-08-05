"""
Paint-Based Pattern Implementations for Chain Tool V4
Patterns generated from user-painted strokes
"""

import bpy
import bmesh
import numpy as np
from mathutils import Vector, Matrix, kdtree
from typing import List, Dict, Tuple, Set, Optional
import math
from collections import defaultdict

from .base_pattern import BasePattern, PatternResult
from ..geometry.triangulation import TriangulationEngine
from ..utils.math_utils import MathUtils

class PaintPattern(BasePattern):
    """Generate pattern from painted strokes"""
    
    def __init__(self):
        super().__init__()
        self.name = "Paint Pattern"
        self.category = "Paint"
        self.description = "Pattern generated from painted areas"
        self.math_utils = MathUtils()
        self.triangulation = TriangulationEngine()
        
    def get_default_params(self) -> Dict:
        return {
            'pattern_type': 'voronoi',  # 'voronoi', 'hexagonal', 'triangular', 'organic'
            'density_mode': 'stroke_based',  # 'stroke_based', 'uniform', 'adaptive'
            'base_density': 0.015,  # Base pattern density
            'stroke_influence': 0.8,  # How much strokes affect density (0-1)
            'connection_radius': 0.02,  # Radius for connecting pattern elements
            'fill_gaps': True,  # Fill gaps between strokes
            'smooth_boundaries': True,  # Smooth pattern boundaries
            'pattern_offset': 0.001,  # Offset from surface
            'min_stroke_points': 3  # Minimum points required in stroke
        }
        
    def validate_params(self, params: Dict) -> Tuple[bool, str]:
        if params.get('base_density', 0) <= 0:
            return False, "Base density must be positive"
        if not 0 <= params.get('stroke_influence', 0.8) <= 1:
            return False, "Stroke influence must be between 0 and 1"
        if params.get('connection_radius', 0) <= 0:
            return False, "Connection radius must be positive"
        return True, ""
        
    def generate(self, target_object: bpy.types.Object, **params) -> PatternResult:
        # Merge with defaults
        full_params = self.get_default_params()
        full_params.update(params)
        
        # Validate
        valid, error = self.validate_params(full_params)
        if not valid:
            return PatternResult([], [], [], {}, {}, False, error)
            
        # Get paint strokes
        strokes = self._get_paint_strokes(target_object)
        
        if not strokes:
            return PatternResult([], [], [], {}, {}, False, "No paint strokes found")
            
        # Process strokes
        processed_strokes = self._process_strokes(
            strokes,
            target_object,
            full_params['min_stroke_points']
        )
        
        if not processed_strokes:
            return PatternResult([], [], [], {}, {}, False, "No valid strokes after processing")
            
        # Generate influence map from strokes
        influence_map = self._generate_influence_map(
            processed_strokes,
            target_object,
            full_params['stroke_influence']
        )
        
        # Generate sample points based on strokes and density
        sample_points = self._generate_paint_samples(
            processed_strokes,
            influence_map,
            target_object,
            full_params['density_mode'],
            full_params['base_density']
        )
        
        # Generate pattern from samples
        pattern_result = self._generate_pattern_from_samples(
            sample_points,
            full_params['pattern_type'],
            target_object
        )
        
        # Connect pattern elements if needed
        if full_params['connection_radius'] > 0:
            pattern_result = self._connect_pattern_elements(
                pattern_result,
                full_params['connection_radius']
            )
            
        # Fill gaps between strokes
        if full_params['fill_gaps']:
            pattern_result = self._fill_stroke_gaps(
                pattern_result,
                processed_strokes,
                target_object,
                full_params['base_density']
            )
            
        # Smooth boundaries
        if full_params['smooth_boundaries']:
            pattern_result = self._smooth_pattern_boundaries(
                pattern_result,
                influence_map
            )
            
        # Apply offset
        pattern_result.vertices = self.offset_from_surface(
            pattern_result.vertices,
            target_object,
            full_params['pattern_offset']
        )
        
        # Add metadata
        pattern_result.metadata = {
            'pattern_type': 'paint',
            'stroke_count': len(processed_strokes),
            'sample_count': len(sample_points),
            'parameters': full_params
        }
        
        return pattern_result
        
    def _get_paint_strokes(self, obj: bpy.types.Object) -> List[Dict]:
        """Get paint strokes for object from state"""
        paint_data = self.state.get("paint_strokes", {})
        return paint_data.get(obj.name, [])
        
    def _process_strokes(self,
                        strokes: List[Dict],
                        obj: bpy.types.Object,
                        min_points: int) -> List[Dict]:
        """Process and validate strokes"""
        processed = []
        
        for stroke in strokes:
            points = stroke.get("points", [])
            
            if len(points) < min_points:
                continue
                
            # Convert to Vector and project to surface
            processed_points = []
            bvh = self.mesh_analyzer.get_bvh_tree(obj)
            
            for point_data in points:
                point_co = Vector(point_data["location"])
                
                # Project to surface
                location, normal, index, distance = bvh.find_nearest(point_co)
                
                if location:
                    processed_points.append({
                        'position': location,
                        'normal': normal,
                        'pressure': point_data.get('pressure', 1.0),
                        'original': point_co
                    })
                    
            if len(processed_points) >= min_points:
                processed.append({
                    'points': processed_points,
                    'radius': stroke.get('radius', 0.05),
                    'strength': stroke.get('strength', 1.0),
                    'timestamp': stroke.get('timestamp', 0)
                })
                
        return processed
        
    def _generate_influence_map(self,
                              strokes: List[Dict],
                              obj: bpy.types.Object,
                              influence_strength: float) -> Dict[Vector, float]:
        """Generate influence map from paint strokes"""
        influence_map = {}
        mesh = obj.data
        
        # Build KD-tree of stroke points for fast lookup
        all_stroke_points = []
        point_data = []
        
        for stroke in strokes:
            for point in stroke['points']:
                all_stroke_points.append(point['position'])
                point_data.append({
                    'radius': stroke['radius'],
                    'strength': stroke['strength'] * point['pressure']
                })
                
        if not all_stroke_points:
            return influence_map
            
        kd = kdtree.KDTree(len(all_stroke_points))
        for i, point in enumerate(all_stroke_points):
            kd.insert(point, i)
        kd.balance()
        
        # Calculate influence for each vertex
        for vert in mesh.vertices:
            vert_co = obj.matrix_world @ vert.co
            
            # Find nearby stroke points
            influence = 0.0
            
            for co, idx, dist in kd.find_range(vert_co, max(d['radius'] for d in point_data)):
                data = point_data[idx]
                
                # Calculate influence with falloff
                if dist < data['radius']:
                    falloff = 1.0 - (dist / data['radius'])
                    influence = max(influence, falloff * data['strength'])
                    
            influence_map[vert.co.copy()] = influence * influence_strength
            
        return influence_map
        
    def _generate_paint_samples(self,
                              strokes: List[Dict],
                              influence_map: Dict[Vector, float],
                              obj: bpy.types.Object,
                              density_mode: str,
                              base_density: float) -> List[Vector]:
        """Generate sample points from paint strokes"""
        samples = []
        
        if density_mode == 'stroke_based':
            # Generate samples along strokes with adaptive density
            for stroke in strokes:
                stroke_samples = self._sample_along_stroke(
                    stroke,
                    base_density * (1.0 / stroke['radius']),  # Denser for smaller brushes
                    obj
                )
                samples.extend(stroke_samples)
                
        elif density_mode == 'uniform':
            # Uniform sampling in painted areas
            samples = self._uniform_paint_sampling(
                influence_map,
                base_density,
                obj
            )
            
        else:  # adaptive
            # Adaptive sampling based on influence strength
            samples = self._adaptive_paint_sampling(
                influence_map,
                base_density,
                obj
            )
            
        # Remove duplicates with minimum distance
        samples = self.remove_overlapping_elements(samples, base_density * 0.8)
        
        return samples
        
    def _sample_along_stroke(self,
                           stroke: Dict,
                           density: float,
                           obj: bpy.types.Object) -> List[Vector]:
        """Sample points along a paint stroke"""
        samples = []
        points = stroke['points']
        
        if len(points) < 2:
            return [p['position'] for p in points]
            
        # Calculate stroke length
        total_length = 0.0
        segments = []
        
        for i in range(len(points) - 1):
            segment_length = (points[i+1]['position'] - points[i]['position']).length
            segments.append(segment_length)
            total_length += segment_length
            
        # Sample along stroke
        if total_length > 0:
            num_samples = int(total_length / density) + 1
            
            for i in range(num_samples):
                t = i / (num_samples - 1) if num_samples > 1 else 0
                
                # Find position along stroke
                target_length = t * total_length
                current_length = 0.0
                
                for j, seg_length in enumerate(segments):
                    if current_length + seg_length >= target_length:
                        # Interpolate within segment
                        local_t = (target_length - current_length) / seg_length if seg_length > 0 else 0
                        
                        pos = points[j]['position'].lerp(points[j+1]['position'], local_t)
                        
                        # Add perpendicular offset for width
                        if j < len(points) - 2:
                            tangent = (points[j+1]['position'] - points[j]['position']).normalized()
                            normal = points[j]['normal']
                            perp = tangent.cross(normal).normalized()
                            
                            # Add samples across stroke width
                            width = stroke['radius']
                            for offset in [-0.5, 0, 0.5]:
                                sample_pos = pos + perp * (offset * width)
                                
                                # Project to surface
                                projected = self.project_to_surface(sample_pos, obj)
                                if projected:
                                    samples.append(projected)
                        else:
                            samples.append(pos)
                            
                        break
                        
                    current_length += seg_length
                    
        return samples
        
    def _uniform_paint_sampling(self,
                              influence_map: Dict[Vector, float],
                              density: float,
                              obj: bpy.types.Object) -> List[Vector]:
        """Uniform sampling in painted areas"""
        samples = []
        
        # Get painted vertices (influence > 0)
        painted_verts = [v for v, inf in influence_map.items() if inf > 0.1]
        
        if not painted_verts:
            return samples
            
        # Calculate bounds
        min_co = Vector((
            min(v.x for v in painted_verts),
            min(v.y for v in painted_verts),
            min(v.z for v in painted_verts)
        ))
        max_co = Vector((
            max(v.x for v in painted_verts),
            max(v.y for v in painted_verts),
            max(v.z for v in painted_verts)
        ))
        
        # Grid sampling
        steps = [int((max_co[i] - min_co[i]) / density) + 1 for i in range(3)]
        
        for x in range(steps[0]):
            for y in range(steps[1]):
                for z in range(steps[2]):
                    sample_pos = Vector((
                        min_co.x + x * density,
                        min_co.y + y * density,
                        min_co.z + z * density
                    ))
                    
                    # Check if in painted area
                    min_dist = float('inf')
                    nearest_influence = 0.0
                    
                    for vert, inf in influence_map.items():
                        dist = (vert - sample_pos).length
                        if dist < min_dist:
                            min_dist = dist
                            nearest_influence = inf
                            
                    if nearest_influence > 0.1 and min_dist < density:
                        # Project to surface
                        projected = self.project_to_surface(sample_pos, obj)
                        if projected:
                            samples.append(projected)
                            
        return samples
        
    def _adaptive_paint_sampling(self,
                               influence_map: Dict[Vector, float],
                               base_density: float,
                               obj: bpy.types.Object) -> List[Vector]:
        """Adaptive sampling based on paint influence"""
        samples = []
        
        # Use surface sampler with paint influence
        all_samples = self.surface_sampler.sample_surface(
            obj,
            base_radius=base_density,
            paint_influence=0.8
        )
        
        # Filter by paint influence
        for sample in all_samples:
            # Find influence at sample
            min_dist = float('inf')
            influence = 0.0
            
            for vert, inf in influence_map.items():
                dist = (vert - sample).length
                if dist < min_dist:
                    min_dist = dist
                    influence = inf
                    
            if influence > 0.1:
                samples.append(sample)
                
        return samples
        
    def _generate_pattern_from_samples(self,
                                     samples: List[Vector],
                                     pattern_type: str,
                                     obj: bpy.types.Object) -> PatternResult:
        """Generate specific pattern type from sample points"""
        if not samples:
            return PatternResult([], [], [], {}, {}, False, "No samples to generate pattern")
            
        if pattern_type == 'voronoi':
            return self._generate_voronoi_from_samples(samples, obj)
        elif pattern_type == 'triangular':
            return self._generate_triangular_from_samples(samples)
        elif pattern_type == 'hexagonal':
            return self._generate_hexagonal_from_samples(samples, obj)
        elif pattern_type == 'organic':
            return self._generate_organic_from_samples(samples, obj)
        else:
            # Default to triangulation
            return self._generate_triangular_from_samples(samples)
            
    def _generate_voronoi_from_samples(self,
                                     samples: List[Vector],
                                     obj: bpy.types.Object) -> PatternResult:
        """Generate Voronoi pattern from samples"""
        from .surface_patterns import VoronoiPattern
        
        # Create temporary mesh with samples
        temp_mesh = bpy.data.meshes.new("TempSamples")
        temp_obj = bpy.data.objects.new("TempSamples", temp_mesh)
        
        temp_mesh.vertices.add(len(samples))
        for i, sample in enumerate(samples):
            temp_mesh.vertices[i].co = sample
            
        temp_mesh.update()
        temp_obj.matrix_world = obj.matrix_world.copy()
        
        # Generate Voronoi
        voronoi = VoronoiPattern()
        result = voronoi.generate(temp_obj, cell_size=0.02, edge_width=0.003)
        
        # Clean up
        bpy.data.objects.remove(temp_obj)
        bpy.data.meshes.remove(temp_mesh)
        
        return result
        
    def _generate_triangular_from_samples(self, samples: List[Vector]) -> PatternResult:
        """Generate triangular mesh from samples"""
        triangulation = self.triangulation.triangulate_surface(samples)
        
        # Create frame pattern from triangulation
        pattern_verts = []
        pattern_edges = []
        pattern_faces = []
        
        # Process each edge to create beams
        edge_set = set()
        
        for face in triangulation['faces']:
            edges = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]
            
            for edge in edges:
                edge_key = tuple(sorted(edge))
                if edge_key not in edge_set:
                    edge_set.add(edge_key)
                    
                    # Create beam geometry
                    v1 = triangulation['vertices'][edge[0]]
                    v2 = triangulation['vertices'][edge[1]]
                    
                    beam = self._create_beam(v1, v2, 0.003)
                    
                    base_idx = len(pattern_verts)
                    pattern_verts.extend(beam['vertices'])
                    
                    for e in beam['edges']:
                        pattern_edges.append((e[0] + base_idx, e[1] + base_idx))
                        
                    for f in beam['faces']:
                        pattern_faces.append([v + base_idx for v in f])
                        
        return PatternResult(
            vertices=pattern_verts,
            edges=pattern_edges,
            faces=pattern_faces,
            attributes={'material_index': [1] * len(pattern_faces)},
            metadata={}
        )
        
    def _generate_hexagonal_from_samples(self,
                                       samples: List[Vector],
                                       obj: bpy.types.Object) -> PatternResult:
        """Generate hexagonal pattern from samples"""
        # Use samples as hex centers
        pattern_verts = []
        pattern_edges = []
        pattern_faces = []
        
        hex_radius = 0.01
        
        for sample in samples:
            # Create hexagon at sample
            hex_verts = []
            
            for i in range(6):
                angle = i * math.pi / 3
                offset = Vector((
                    hex_radius * math.cos(angle),
                    hex_radius * math.sin(angle),
                    0
                ))
                
                # Project offset to surface tangent plane
                vert_pos = sample + offset
                projected = self.project_to_surface(vert_pos, obj)
                
                if projected:
                    hex_verts.append(projected)
                else:
                    hex_verts.append(vert_pos)
                    
            if len(hex_verts) == 6:
                # Create hex frame
                base_idx = len(pattern_verts)
                
                # Add outer and inner vertices
                for vert in hex_verts:
                    pattern_verts.append(vert)  # Outer
                    
                center = sum(hex_verts, Vector()) / 6
                for vert in hex_verts:
                    inner = center + (vert - center) * 0.7
                    pattern_verts.append(inner)  # Inner
                    
                # Create faces
                for i in range(6):
                    next_i = (i + 1) % 6
                    
                    outer_curr = base_idx + i * 2
                    inner_curr = base_idx + i * 2 + 1
                    outer_next = base_idx + next_i * 2
                    inner_next = base_idx + next_i * 2 + 1
                    
                    pattern_faces.append([
                        outer_curr, outer_next,
                        inner_next, inner_curr
                    ])
                    
        return PatternResult(
            vertices=pattern_verts,
            edges=pattern_edges,
            faces=pattern_faces,
            attributes={'material_index': [1] * len(pattern_faces)},
            metadata={}
        )
        
    def _generate_organic_from_samples(self,
                                     samples: List[Vector],
                                     obj: bpy.types.Object) -> PatternResult:
        """Generate organic curved pattern from samples"""
        # Create smooth curves through samples
        pattern_verts = []
        pattern_edges = []
        pattern_faces = []
        
        # Cluster nearby samples into groups
        clusters = self._cluster_samples(samples, 0.03)
        
        for cluster in clusters:
            if len(cluster) < 3:
                continue
                
            # Create smooth curve through cluster
            curve_points = self._smooth_curve_through_points(cluster)
            
            # Create tube geometry along curve
            for i in range(len(curve_points) - 1):
                tube = self._create_tube_segment(
                    curve_points[i],
                    curve_points[i + 1],
                    0.003,
                    6  # Resolution
                )
                
                base_idx = len(pattern_verts)
                pattern_verts.extend(tube['vertices'])
                
                for e in tube['edges']:
                    pattern_edges.append((e[0] + base_idx, e[1] + base_idx))
                    
                for f in tube['faces']:
                    pattern_faces.append([v + base_idx for v in f])
                    
        return PatternResult(
            vertices=pattern_verts,
            edges=pattern_edges,
            faces=pattern_faces,
            attributes={'material_index': [1] * len(pattern_faces)},
            metadata={}
        )
        
    def _cluster_samples(self, samples: List[Vector], max_dist: float) -> List[List[Vector]]:
        """Cluster nearby samples"""
        clusters = []
        used = set()
        
        for i, sample in enumerate(samples):
            if i in used:
                continue
                
            cluster = [sample]
            used.add(i)
            
            # Find nearby samples
            for j, other in enumerate(samples):
                if j not in used and (sample - other).length < max_dist:
                    cluster.append(other)
                    used.add(j)
                    
            if len(cluster) >= 3:
                clusters.append(cluster)
                
        return clusters
        
    def _smooth_curve_through_points(self, points: List[Vector]) -> List[Vector]:
        """Create smooth curve through points"""
        if len(points) < 3:
            return points
            
        # Simple Catmull-Rom spline
        curve_points = []
        
        for i in range(len(points)):
            p0 = points[max(0, i-1)]
            p1 = points[i]
            p2 = points[min(len(points)-1, i+1)]
            p3 = points[min(len(points)-1, i+2)]
            
            # Generate intermediate points
            for t in [0, 0.33, 0.67]:
                point = self._catmull_rom_point(p0, p1, p2, p3, t)
                curve_points.append(point)
                
        return curve_points
        
    def _catmull_rom_point(self, p0: Vector, p1: Vector, p2: Vector, p3: Vector, t: float) -> Vector:
        """Calculate point on Catmull-Rom spline"""
        t2 = t * t
        t3 = t2 * t
        
        return (
            0.5 * ((2 * p1) +
                  (-p0 + p2) * t +
                  (2*p0 - 5*p1 + 4*p2 - p3) * t2 +
                  (-p0 + 3*p1 - 3*p2 + p3) * t3)
        )
        
    def _create_beam(self, v1: Vector, v2: Vector, width: float) -> Dict:
        """Create beam geometry between two points"""
        geometry = {'vertices': [], 'edges': [], 'faces': []}
        
        # Calculate beam direction
        direction = (v2 - v1).normalized()
        length = (v2 - v1).length
        
        # Find perpendicular vectors
        up = Vector((0, 0, 1))
        if abs(direction.dot(up)) > 0.99:
            up = Vector((1, 0, 0))
            
        right = direction.cross(up).normalized()
        up = right.cross(direction).normalized()
        
        # Create box vertices
        half_width = width / 2
        
        # Bottom vertices
        geometry['vertices'].extend([
            v1 - right * half_width - up * half_width,
            v1 + right * half_width - up * half_width,
            v1 + right * half_width + up * half_width,
            v1 - right * half_width + up * half_width
        ])
        
        # Top vertices
        geometry['vertices'].extend([
            v2 - right * half_width - up * half_width,
            v2 + right * half_width - up * half_width,
            v2 + right * half_width + up * half_width,
            v2 - right * half_width + up * half_width
        ])
        
        # Create faces
        geometry['faces'] = [
            [0, 1, 2, 3],  # Bottom
            [4, 7, 6, 5],  # Top
            [0, 4, 5, 1],  # Front
            [2, 6, 7, 3],  # Back
            [0, 3, 7, 4],  # Left
            [1, 5, 6, 2]   # Right
        ]
        
        return geometry
        
    def _create_tube_segment(self, v1: Vector, v2: Vector, radius: float, resolution: int) -> Dict:
        """Create tube segment between two points"""
        geometry = {'vertices': [], 'edges': [], 'faces': []}
        
        # Calculate tube direction
        direction = (v2 - v1).normalized()
        
        # Find perpendicular vectors
        up = Vector((0, 0, 1))
        if abs(direction.dot(up)) > 0.99:
            up = Vector((1, 0, 0))
            
        right = direction.cross(up).normalized()
        up = right.cross(direction).normalized()
        
        # Create vertices
        for i in range(2):  # Start and end
            center = v1 if i == 0 else v2
            
            for j in range(resolution):
                angle = j * 2 * math.pi / resolution
                offset = right * (radius * math.cos(angle)) + up * (radius * math.sin(angle))
                geometry['vertices'].append(center + offset)
                
        # Create faces
        for i in range(resolution):
            next_i = (i + 1) % resolution
            
            # Quad face
            geometry['faces'].append([
                i,                          # Bottom current
                next_i,                     # Bottom next
                next_i + resolution,        # Top next
                i + resolution              # Top current
            ])
            
        return geometry
        
    def _connect_pattern_elements(self,
                                result: PatternResult,
                                radius: float) -> PatternResult:
        """Connect nearby pattern elements"""
        # Build KD-tree of vertices
        if not result.vertices:
            return result
            
        kd = kdtree.KDTree(len(result.vertices))
        for i, vert in enumerate(result.vertices):
            kd.insert(vert, i)
        kd.balance()
        
        # Find connections
        new_edges = []
        connected_pairs = set()
        
        for i, vert in enumerate(result.vertices):
            # Find nearby vertices
            for co, idx, dist in kd.find_range(vert, radius):
                if idx > i and (i, idx) not in connected_pairs:
                    # Check if not already connected
                    edge_exists = any(
                        (e[0] == i and e[1] == idx) or (e[0] == idx and e[1] == i)
                        for e in result.edges
                    )
                    
                    if not edge_exists:
                        new_edges.append((i, idx))
                        connected_pairs.add((i, idx))
                        
        # Add new edges
        result.edges.extend(new_edges)
        
        return result
        
    def _fill_stroke_gaps(self,
                        result: PatternResult,
                        strokes: List[Dict],
                        obj: bpy.types.Object,
                        density: float) -> PatternResult:
        """Fill gaps between paint strokes"""
        # Find gap regions between strokes
        gap_samples = []
        
        # Simple approach: find areas between stroke endpoints
        endpoints = []
        for stroke in strokes:
            if stroke['points']:
                endpoints.append(stroke['points'][0]['position'])
                endpoints.append(stroke['points'][-1]['position'])
                
        # Connect nearby endpoints
        for i, p1 in enumerate(endpoints):
            for j, p2 in enumerate(endpoints[i+1:], i+1):
                dist = (p2 - p1).length
                
                if 0.02 < dist < 0.1:  # Gap range
                    # Sample points along gap
                    num_samples = int(dist / density) + 1
                    
                    for k in range(1, num_samples):
                        t = k / num_samples
                        gap_point = p1.lerp(p2, t)
                        
                        # Project to surface
                        projected = self.project_to_surface(gap_point, obj)
                        if projected:
                            gap_samples.append(projected)
                            
        if gap_samples:
            # Generate pattern for gap regions
            gap_pattern = self._generate_triangular_from_samples(gap_samples)
            
            # Merge with existing pattern
            base_idx = len(result.vertices)
            result.vertices.extend(gap_pattern.vertices)
            
            for edge in gap_pattern.edges:
                result.edges.append((edge[0] + base_idx, edge[1] + base_idx))
                
            for face in gap_pattern.faces:
                result.faces.append([v + base_idx for v in face])
                
        return result
        
    def _smooth_pattern_boundaries(self,
                                 result: PatternResult,
                                 influence_map: Dict[Vector, float]) -> PatternResult:
        """Smooth pattern boundaries based on influence"""
        if not result.vertices:
            return result
            
        # Identify boundary vertices
        vertex_faces = defaultdict(list)
        for i, face in enumerate(result.faces):
            for v in face:
                vertex_faces[v].append(i)
                
        boundary_verts = []
        for v_idx, faces in vertex_faces.items():
            if len(faces) < 4:  # Likely boundary
                boundary_verts.append(v_idx)
                
        # Smooth boundary vertices
        smoothed_positions = {}
        
        for v_idx in boundary_verts:
            if v_idx >= len(result.vertices):
                continue
                
            vert = result.vertices[v_idx]
            
            # Find influence at vertex
            min_dist = float('inf')
            influence = 0.0
            
            for inf_vert, inf in influence_map.items():
                dist = (inf_vert - vert).length
                if dist < min_dist:
                    min_dist = dist
                    influence = inf
                    
            # Smooth based on influence
            if influence < 0.5:
                # Find neighbor average
                neighbors = []
                
                for face in result.faces:
                    if v_idx in face:
                        for other_v in face:
                            if other_v != v_idx and other_v not in boundary_verts:
                                neighbors.append(result.vertices[other_v])
                                
                if neighbors:
                    avg_pos = sum(neighbors, Vector()) / len(neighbors)
                    # Blend towards average
                    blend_factor = 0.3 * (1.0 - influence)
                    smoothed_positions[v_idx] = vert.lerp(avg_pos, blend_factor)
                    
        # Apply smoothed positions
        for v_idx, new_pos in smoothed_positions.items():
            result.vertices[v_idx] = new_pos
            
        return result


class StrokeConnector(BasePattern):
    """Connect paint strokes with optimized paths"""
    
    def __init__(self):
        super().__init__()
        self.name = "Stroke Connector"
        self.category = "Paint"
        self.description = "Optimized connections between paint strokes"
        
    def get_default_params(self) -> Dict:
        return {
            'connection_type': 'shortest',  # 'shortest', 'smooth', 'stress_based'
            'max_connection_length': 0.1,  # Maximum connection distance
            'connection_width': 0.004,  # Width of connections
            'branch_angle': 45.0,  # Maximum branching angle (degrees)
            'optimize_junctions': True,  # Optimize junction points
            'pattern_offset': 0.001  # Offset from surface
        }
        
    def validate_params(self, params: Dict) -> Tuple[bool, str]:
        if params.get('max_connection_length', 0) <= 0:
            return False, "Maximum connection length must be positive"
        if params.get('connection_width', 0) <= 0:
            return False, "Connection width must be positive"
        return True, ""
        
    def generate(self, target_object: bpy.types.Object, **params) -> PatternResult:
        # Merge with defaults
        full_params = self.get_default_params()
        full_params.update(params)
        
        # Validate
        valid, error = self.validate_params(full_params)
        if not valid:
            return PatternResult([], [], [], {}, {}, False, error)
            
        # Get and process strokes
        strokes = self._get_paint_strokes(target_object)
        if not strokes:
            return PatternResult([], [], [], {}, {}, False, "No paint strokes found")
            
        processed_strokes = self._process_strokes(strokes, target_object, 2)
        
        # Extract stroke endpoints and key points
        key_points = self._extract_key_points(processed_strokes)
        
        # Generate connection graph
        connections = self._generate_connections(
            key_points,
            full_params['connection_type'],
            full_params['max_connection_length'],
            full_params['branch_angle'],
            target_object
        )
        
        # Optimize junctions if enabled
        if full_params['optimize_junctions']:
            connections = self._optimize_junctions(connections)
            
        # Generate geometry for connections
        pattern_verts = []
        pattern_edges = []
        pattern_faces = []
        
        for conn in connections:
            conn_geom = self._create_connection_geometry(
                conn,
                full_params['connection_width'],
                target_object
            )
            
            base_idx = len(pattern_verts)
            pattern_verts.extend(conn_geom['vertices'])
            
            for edge in conn_geom['edges']:
                pattern_edges.append((edge[0] + base_idx, edge[1] + base_idx))
                
            for face in conn_geom['faces']:
                pattern_faces.append([v + base_idx for v in face])
                
        # Apply offset
        pattern_verts = self.offset_from_surface(
            pattern_verts,
            target_object,
            full_params['pattern_offset']
        )
        
        return PatternResult(
            vertices=pattern_verts,
            edges=pattern_edges,
            faces=pattern_faces,
            attributes={'material_index': [1] * len(pattern_faces)},
            metadata={
                'pattern_type': 'stroke_connector',
                'connection_count': len(connections),
                'parameters': full_params
            }
        )
        
    def _extract_key_points(self, strokes: List[Dict]) -> List[Dict]:
        """Extract endpoints and junction points from strokes"""
        key_points = []
        
        for i, stroke in enumerate(strokes):
            points = stroke['points']
            
            if not points:
                continue
                
            # Add endpoints
            key_points.append({
                'position': points[0]['position'],
                'type': 'endpoint',
                'stroke_id': i,
                'point_index': 0
            })
            
            key_points.append({
                'position': points[-1]['position'],
                'type': 'endpoint',
                'stroke_id': i,
                'point_index': len(points) - 1
            })
            
            # Add sharp turns as key points
            for j in range(1, len(points) - 1):
                prev_dir = (points[j]['position'] - points[j-1]['position']).normalized()
                next_dir = (points[j+1]['position'] - points[j]['position']).normalized()
                
                if prev_dir.length > 0 and next_dir.length > 0:
                    angle = math.acos(max(-1, min(1, prev_dir.dot(next_dir))))
                    
                    if angle > math.radians(30):  # Sharp turn
                        key_points.append({
                            'position': points[j]['position'],
                            'type': 'junction',
                            'stroke_id': i,
                            'point_index': j
                        })
                        
        return key_points
        
    def _generate_connections(self,
                            key_points: List[Dict],
                            connection_type: str,
                            max_length: float,
                            max_angle: float,
                            obj: bpy.types.Object) -> List[Dict]:
        """Generate connections between key points"""
        connections = []
        
        if connection_type == 'shortest':
            connections = self._shortest_path_connections(key_points, max_length)
        elif connection_type == 'smooth':
            connections = self._smooth_connections(key_points, max_length, max_angle)
        elif connection_type == 'stress_based':
            connections = self._stress_based_connections(key_points, max_length, obj)
            
        return connections
        
    def _shortest_path_connections(self,
                                 key_points: List[Dict],
                                 max_length: float) -> List[Dict]:
        """Generate shortest path connections"""
        connections = []
        
        # Build distance matrix
        n = len(key_points)
        distances = np.full((n, n), np.inf)
        
        for i in range(n):
            for j in range(i + 1, n):
                # Don't connect points on same stroke
                if key_points[i]['stroke_id'] == key_points[j]['stroke_id']:
                    continue
                    
                dist = (key_points[i]['position'] - key_points[j]['position']).length
                
                if dist <= max_length:
                    distances[i, j] = dist
                    distances[j, i] = dist
                    
        # Find minimum spanning tree
        visited = [False] * n
        visited[0] = True
        
        while not all(visited):
            min_dist = np.inf
            min_i, min_j = -1, -1
            
            for i in range(n):
                if visited[i]:
                    for j in range(n):
                        if not visited[j] and distances[i, j] < min_dist:
                            min_dist = distances[i, j]
                            min_i, min_j = i, j
                            
            if min_i >= 0 and min_j >= 0:
                visited[min_j] = True
                connections.append({
                    'start': key_points[min_i]['position'],
                    'end': key_points[min_j]['position'],
                    'type': 'direct',
                    'weight': min_dist
                })
            else:
                break
                
        return connections
        
    def _smooth_connections(self,
                          key_points: List[Dict],
                          max_length: float,
                          max_angle: float) -> List[Dict]:
        """Generate smooth curved connections"""
        connections = []
        max_angle_rad = math.radians(max_angle)
        
        # Group points by proximity
        groups = []
        used = set()
        
        for i, point in enumerate(key_points):
            if i in used:
                continue
                
            group = [i]
            used.add(i)
            
            # Find nearby points
            for j, other in enumerate(key_points):
                if j not in used:
                    dist = (point['position'] - other['position']).length
                    
                    if dist < max_length * 0.5:
                        group.append(j)
                        used.add(j)
                        
            if len(group) > 1:
                groups.append(group)
                
        # Connect groups with smooth curves
        for group in groups:
            if len(group) < 2:
                continue
                
            # Sort by position for smooth curve
            sorted_indices = sorted(group, key=lambda i: key_points[i]['position'].x)
            
            for i in range(len(sorted_indices) - 1):
                idx1 = sorted_indices[i]
                idx2 = sorted_indices[i + 1]
                
                # Create smooth curve
                start = key_points[idx1]['position']
                end = key_points[idx2]['position']
                
                # Add control points for bezier curve
                mid = (start + end) * 0.5
                offset = (end - start).normalized().cross(Vector((0, 0, 1))) * 0.02
                
                connections.append({
                    'start': start,
                    'end': end,
                    'type': 'bezier',
                    'control1': mid - offset,
                    'control2': mid + offset,
                    'weight': (end - start).length
                })
                
        return connections
        
    def _stress_based_connections(self,
                                key_points: List[Dict],
                                max_length: float,
                                obj: bpy.types.Object) -> List[Dict]:
        """Generate connections following stress lines"""
        connections = []
        
        # Simplified stress field
        # In production, use proper FEA or stress analysis
        
        for i, point1 in enumerate(key_points):
            for j, point2 in enumerate(key_points[i+1:], i+1):
                if point1['stroke_id'] == point2['stroke_id']:
                    continue
                    
                dist = (point1['position'] - point2['position']).length
                
                if dist <= max_length:
                    # Check if connection follows stress direction
                    # Simplified: prefer vertical connections (gravity load)
                    direction = (point2['position'] - point1['position']).normalized()
                    vertical_alignment = abs(direction.dot(Vector((0, 0, 1))))
                    
                    if vertical_alignment > 0.5:  # Mostly vertical
                        connections.append({
                            'start': point1['position'],
                            'end': point2['position'],
                            'type': 'direct',
                            'weight': dist * (1.0 - vertical_alignment)  # Prefer vertical
                        })
                        
        return connections
        
    def _optimize_junctions(self, connections: List[Dict]) -> List[Dict]:
        """Optimize junction points for better connectivity"""
        # Find junction points (where multiple connections meet)
        junction_map = defaultdict(list)
        
        for i, conn in enumerate(connections):
            junction_map[tuple(conn['start'])].append(i)
            junction_map[tuple(conn['end'])].append(i)
            
        # Optimize junctions with 3+ connections
        optimized = connections.copy()
        
        for junction_pos, conn_indices in junction_map.items():
            if len(conn_indices) >= 3:
                # Calculate optimal junction position
                connected_points = []
                
                for idx in conn_indices:
                    conn = connections[idx]
                    if tuple(conn['start']) == junction_pos:
                        connected_points.append(conn['end'])
                    else:
                        connected_points.append(conn['start'])
                        
                # New junction position (centroid)
                if connected_points:
                    new_pos = sum(connected_points, Vector()) / len(connected_points)
                    
                    # Update connections
                    for idx in conn_indices:
                        if tuple(optimized[idx]['start']) == junction_pos:
                            optimized[idx]['start'] = new_pos
                        elif tuple(optimized[idx]['end']) == junction_pos:
                            optimized[idx]['end'] = new_pos
                            
        return optimized
        
    def _create_connection_geometry(self,
                                  connection: Dict,
                                  width: float,
                                  obj: bpy.types.Object) -> Dict:
        """Create geometry for a connection"""
        if connection['type'] == 'direct':
            return self._create_beam(connection['start'], connection['end'], width)
        elif connection['type'] == 'bezier':
            # Create bezier curve geometry
            curve_points = self._evaluate_bezier(
                connection['start'],
                connection['control1'],
                connection['control2'],
                connection['end'],
                10  # Resolution
            )
            
            geometry = {'vertices': [], 'edges': [], 'faces': []}
            
            # Create tube along curve
            for i in range(len(curve_points) - 1):
                segment = self._create_beam(curve_points[i], curve_points[i+1], width)
                
                base_idx = len(geometry['vertices'])
                geometry['vertices'].extend(segment['vertices'])
                
                for edge in segment['edges']:
                    geometry['edges'].append((edge[0] + base_idx, edge[1] + base_idx))
                    
                for face in segment['faces']:
                    geometry['faces'].append([v + base_idx for v in face])
                    
            return geometry
        else:
            return self._create_beam(connection['start'], connection['end'], width)
            
    def _evaluate_bezier(self,
                        p0: Vector,
                        p1: Vector,
                        p2: Vector,
                        p3: Vector,
                        resolution: int) -> List[Vector]:
        """Evaluate bezier curve at given resolution"""
        points = []
        
        for i in range(resolution + 1):
            t = i / resolution
            t2 = t * t
            t3 = t2 * t
            
            mt = 1 - t
            mt2 = mt * mt
            mt3 = mt2 * mt
            
            point = mt3 * p0 + 3 * mt2 * t * p1 + 3 * mt * t2 * p2 + t3 * p3
            points.append(point)
            
        return points


class PaintDensityMap(BasePattern):
    """Variable density pattern based on paint intensity"""
    
    def __init__(self):
        super().__init__()
        self.name = "Paint Density Map"
        self.category = "Paint"
        self.description = "Pattern density controlled by paint stroke intensity"
        
    def get_default_params(self) -> Dict:
        return {
            'base_pattern': 'voronoi',  # Underlying pattern type
            'min_density': 0.03,  # Minimum pattern density
            'max_density': 0.008,  # Maximum pattern density
            'intensity_power': 1.5,  # Power curve for intensity mapping
            'smooth_transitions': True,  # Smooth density transitions
            'pattern_offset': 0.001  # Offset from surface
        }
        
    def validate_params(self, params: Dict) -> Tuple[bool, str]:
        if params.get('min_density', 0) <= 0:
            return False, "Minimum density must be positive"
        if params.get('max_density', 0) <= 0:
            return False, "Maximum density must be positive"
        if params.get('max_density') >= params.get('min_density'):
            return False, "Maximum density must be less than minimum (smaller value = denser)"
        return True, ""
        
    def generate(self, target_object: bpy.types.Object, **params) -> PatternResult:
        # Merge with defaults
        full_params = self.get_default_params()
        full_params.update(params)
        
        # Validate
        valid, error = self.validate_params(full_params)
        if not valid:
            return PatternResult([], [], [], {}, {}, False, error)
            
        # Get paint strokes
        strokes = self._get_paint_strokes(target_object)
        if not strokes:
            return PatternResult([], [], [], {}, {}, False, "No paint strokes found")
            
        # Generate intensity map from paint strokes
        intensity_map = self._generate_paint_intensity_map(
            strokes,
            target_object,
            full_params['intensity_power']
        )
        
        # Generate variable density samples
        samples = self._generate_density_mapped_samples(
            intensity_map,
            target_object,
            full_params['min_density'],
            full_params['max_density'],
            full_params['smooth_transitions']
        )
        
        # Generate pattern
        if full_params['base_pattern'] == 'voronoi':
            from .surface_patterns import VoronoiPattern
            pattern = VoronoiPattern()
            
            # Create temp object with samples
            temp_obj = self._create_sample_object(samples, target_object)
            result = pattern.generate(temp_obj)
            
            # Clean up
            bpy.data.objects.remove(temp_obj)
            bpy.data.meshes.remove(temp_obj.data)
            
        else:
            # Default triangulation
            triangulation = self.triangulation.triangulate_surface(samples)
            result = PatternResult(
                vertices=triangulation['vertices'],
                edges=triangulation['edges'],
                faces=triangulation['faces'],
                attributes={},
                metadata={}
            )
            
        # Apply variable thickness based on intensity
        result = self._apply_intensity_based_thickness(result, intensity_map)
        
        # Apply offset
        result.vertices = self.offset_from_surface(
            result.vertices,
            target_object,
            full_params['pattern_offset']
        )
        
        # Add metadata
        result.metadata = {
            'pattern_type': 'paint_density_map',
            'sample_count': len(samples),
            'intensity_range': self._get_intensity_range(intensity_map),
            'parameters': full_params
        }
        
        return result
        
    def _generate_paint_intensity_map(self,
                                    strokes: List[Dict],
                                    obj: bpy.types.Object,
                                    power: float) -> Dict[Vector, float]:
        """Generate intensity map from paint strokes"""
        intensity_map = {}
        mesh = obj.data
        
        # Process each vertex
        for vert in mesh.vertices:
            world_pos = obj.matrix_world @ vert.co
            max_intensity = 0.0
            
            # Check intensity from all strokes
            for stroke in strokes:
                stroke_intensity = self._calculate_stroke_intensity(
                    world_pos,
                    stroke,
                    power
                )
                max_intensity = max(max_intensity, stroke_intensity)
                
            intensity_map[vert.co.copy()] = max_intensity
            
        return intensity_map
        
    def _calculate_stroke_intensity(self,
                                  point: Vector,
                                  stroke: Dict,
                                  power: float) -> float:
        """Calculate intensity at point from stroke"""
        if not stroke.get('points'):
            return 0.0
            
        max_intensity = 0.0
        radius = stroke.get('radius', 0.05)
        strength = stroke.get('strength', 1.0)
        
        for stroke_point in stroke['points']:
            pos = Vector(stroke_point.get('location', stroke_point.get('position', (0,0,0))))
            pressure = stroke_point.get('pressure', 1.0)
            
            dist = (point - pos).length
            
            if dist < radius:
                # Falloff function
                falloff = 1.0 - (dist / radius)
                intensity = falloff * strength * pressure
                
                # Apply power curve
                intensity = intensity ** power
                
                max_intensity = max(max_intensity, intensity)
                
        return max_intensity
        
    def _generate_density_mapped_samples(self,
                                       intensity_map: Dict[Vector, float],
                                       obj: bpy.types.Object,
                                       min_density: float,
                                       max_density: float,
                                       smooth: bool) -> List[Vector]:
        """Generate samples with density based on intensity map"""
        samples = []
        
        # Get mesh data
        mesh = obj.data
        
        # Calculate surface area per face
        face_areas = [f.area for f in mesh.polygons]
        total_area = sum(face_areas)
        
        # Sample each face based on local intensity
        for face_idx, face in enumerate(mesh.polygons):
            # Get average intensity for face
            face_intensity = 0.0
            for vert_idx in face.vertices:
                vert_co = mesh.vertices[vert_idx].co
                face_intensity += intensity_map.get(vert_co, 0.0)
                
            face_intensity /= len(face.vertices)
            
            if face_intensity > 0.01:  # Only sample painted areas
                # Calculate local density
                # Higher intensity = smaller density value = more samples
                local_density = min_density + (max_density - min_density) * (1.0 - face_intensity)
                
                # Number of samples for this face
                face_area = face_areas[face_idx]
                num_samples = int(face_area / (local_density ** 2))
                
                # Generate samples on face
                face_verts = [mesh.vertices[i].co for i in face.vertices]
                
                for _ in range(num_samples):
                    # Random point on face
                    if len(face_verts) == 3:
                        # Barycentric for triangle
                        r1, r2 = np.random.random(2)
                        if r1 + r2 > 1:
                            r1, r2 = 1 - r1, 1 - r2
                        sample = face_verts[0] * r1 + face_verts[1] * r2 + face_verts[2] * (1 - r1 - r2)
                    else:
                        # Simple average for quads
                        weights = np.random.dirichlet(np.ones(len(face_verts)))
                        sample = sum(w * v for w, v in zip(weights, face_verts))
                        
                    samples.append(sample)
                    
        # Apply smoothing if enabled
        if smooth and samples:
            samples = self._smooth_sample_distribution(samples, min_density)
            
        return samples
        
    def _smooth_sample_distribution(self,
                                  samples: List[Vector],
                                  min_spacing: float) -> List[Vector]:
        """Smooth sample distribution using Lloyd's algorithm"""
        if len(samples) < 4:
            return samples
            
        # Perform a few iterations of Lloyd's relaxation
        for _ in range(3):
            # Build Voronoi diagram
            points_2d = []
            
            # Project to 2D (simplified)
            for s in samples:
                points_2d.append([s.x, s.y])
                
            from scipy.spatial import Voronoi
            
            try:
                vor = Voronoi(np.array(points_2d))
                
                # Move points to centroids
                new_samples = []
                
                for i, region in enumerate(vor.regions):
                    if -1 not in region and len(region) > 0:
                        # Calculate centroid
                        region_verts = [vor.vertices[j] for j in region]
                        if region_verts:
                            centroid_2d = np.mean(region_verts, axis=0)
                            
                            # Preserve Z coordinate
                            if i < len(samples):
                                new_pos = Vector((centroid_2d[0], centroid_2d[1], samples[i].z))
                                new_samples.append(new_pos)
                                
                samples = new_samples if new_samples else samples
                
            except:
                # Voronoi failed, keep original
                pass
                
        return samples
        
    def _create_sample_object(self, samples: List[Vector], ref_obj: bpy.types.Object) -> bpy.types.Object:
        """Create temporary object from samples"""
        mesh = bpy.data.meshes.new("TempSamples")
        obj = bpy.data.objects.new("TempSamples", mesh)
        
        mesh.vertices.add(len(samples))
        for i, sample in enumerate(samples):
            mesh.vertices[i].co = sample
            
        mesh.update()
        obj.matrix_world = ref_obj.matrix_world.copy()
        
        return obj
        
    def _apply_intensity_based_thickness(self,
                                       result: PatternResult,
                                       intensity_map: Dict[Vector, float]) -> PatternResult:
        """Apply variable thickness based on paint intensity"""
        thicknesses = []
        
        for vert in result.vertices:
            # Find nearest intensity
            min_dist = float('inf')
            intensity = 0.5
            
            for map_vert, map_intensity in intensity_map.items():
                dist = (vert - map_vert).length
                if dist < min_dist:
                    min_dist = dist
                    intensity = map_intensity
                    
            # Map intensity to thickness
            # Higher intensity = thicker pattern
            thickness = 0.002 + intensity * 0.003
            thicknesses.append(thickness)
            
        result.attributes['thickness'] = thicknesses
        result.attributes['intensity'] = [self._get_nearest_intensity(v, intensity_map) 
                                         for v in result.vertices]
        
        return result
        
    def _get_nearest_intensity(self, point: Vector, intensity_map: Dict[Vector, float]) -> float:
        """Get intensity value for a point"""
        min_dist = float('inf')
        intensity = 0.0
        
        for vert, intens in intensity_map.items():
            dist = (vert - point).length
            if dist < min_dist:
                min_dist = dist
                intensity = intens
                
        return intensity
        
    def _get_intensity_range(self, intensity_map: Dict[Vector, float]) -> Tuple[float, float]:
        """Get min and max intensity values"""
        if not intensity_map:
            return 0.0, 0.0
            
        values = list(intensity_map.values())
        return min(values), max(values)
