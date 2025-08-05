"""
Edge Pattern Implementations for Chain Tool V4
Specialized patterns for edge and rim reinforcement
"""

import bpy
import bmesh
import numpy as np
from mathutils import Vector, Matrix
from typing import List, Dict, Tuple, Set, Optional
import math

from .base_pattern import BasePattern, PatternResult
from ..utils.math_utils import MathUtils

class RimReinforcement(BasePattern):
    """Continuous rim reinforcement along edges"""
    
    def __init__(self):
        super().__init__()
        self.name = "Rim Reinforcement"
        self.category = "Edge"
        self.description = "Continuous reinforcement band along edges"
        self.math_utils = MathUtils()
        
    def get_default_params(self) -> Dict:
        return {
            'rim_width': 0.01,  # Width of rim band
            'rim_height': 0.003,  # Height/thickness of rim
            'inner_pattern': 'wave',  # 'straight', 'wave', 'zigzag'
            'wave_amplitude': 0.002,  # For wave pattern
            'wave_frequency': 10,  # Waves per unit length
            'corner_radius': 0.005,  # Radius for corner smoothing
            'edge_selection': 'auto',  # 'auto', 'manual', 'all'
            'min_edge_angle': 45.0,  # Minimum angle to consider as edge
            'pattern_offset': 0.001  # Offset from surface
        }
        
    def validate_params(self, params: Dict) -> Tuple[bool, str]:
        if params.get('rim_width', 0) <= 0:
            return False, "Rim width must be positive"
        if params.get('rim_height', 0) <= 0:
            return False, "Rim height must be positive"
        if params.get('inner_pattern') not in ['straight', 'wave', 'zigzag']:
            return False, "Invalid inner pattern type"
        return True, ""
        
    def generate(self, target_object: bpy.types.Object, **params) -> PatternResult:
        # Merge with defaults
        full_params = self.get_default_params()
        full_params.update(params)
        
        # Validate
        valid, error = self.validate_params(full_params)
        if not valid:
            return PatternResult([], [], [], {}, {}, False, error)
            
        # Get edge loops
        edge_loops = self._get_edge_loops_for_rim(
            target_object,
            full_params['edge_selection'],
            full_params['min_edge_angle']
        )
        
        if not edge_loops:
            return PatternResult([], [], [], {}, {}, False, "No suitable edges found")
            
        # Generate rim geometry for each loop
        pattern_verts = []
        pattern_edges = []
        pattern_faces = []
        
        for loop in edge_loops:
            rim_geom = self._create_rim_geometry(
                loop,
                target_object,
                full_params
            )
            
            # Add to pattern
            base_idx = len(pattern_verts)
            pattern_verts.extend(rim_geom['vertices'])
            
            for edge in rim_geom['edges']:
                pattern_edges.append((edge[0] + base_idx, edge[1] + base_idx))
                
            for face in rim_geom['faces']:
                pattern_faces.append([v + base_idx for v in face])
                
        # Apply offset
        pattern_verts = self.offset_from_surface(
            pattern_verts,
            target_object,
            full_params['pattern_offset']
        )
        
        # Create attributes
        attributes = {
            'thickness': [full_params['rim_height']] * len(pattern_verts),
            'material_index': [1] * len(pattern_faces),
            'is_edge': [True] * len(pattern_verts)  # Mark as edge pattern
        }
        
        metadata = {
            'pattern_type': 'rim_reinforcement',
            'edge_loop_count': len(edge_loops),
            'total_length': self._calculate_total_length(edge_loops, target_object),
            'parameters': full_params
        }
        
        return PatternResult(
            vertices=pattern_verts,
            edges=pattern_edges,
            faces=pattern_faces,
            attributes=attributes,
            metadata=metadata
        )
        
    def _get_edge_loops_for_rim(self,
                              obj: bpy.types.Object,
                              selection: str,
                              min_angle: float) -> List[List[int]]:
        """Get edge loops suitable for rim reinforcement"""
        if selection == 'manual':
            # Use selected edges in edit mode
            return self._get_selected_edge_loops(obj)
        elif selection == 'all':
            # Use all boundary edges
            return self._get_all_boundary_loops(obj)
        else:  # auto
            # Detect sharp edges
            edges = self.edge_detector.detect_edges(obj, angle_threshold=min_angle)
            return self.edge_detector.extract_edge_loops(obj, edges)
            
    def _get_selected_edge_loops(self, obj: bpy.types.Object) -> List[List[int]]:
        """Get manually selected edge loops"""
        loops = []
        
        # This would interface with Blender's selection
        # For now, return empty
        return loops
        
    def _get_all_boundary_loops(self, obj: bpy.types.Object) -> List[List[int]]:
        """Get all boundary edge loops"""
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.edges.ensure_lookup_table()
        
        # Find boundary edges
        boundary_edges = [e for e in bm.edges if len(e.link_faces) == 1]
        
        # Extract loops
        loops = []
        used_edges = set()
        
        for edge in boundary_edges:
            if edge in used_edges:
                continue
                
            # Trace loop
            loop = []
            current_edge = edge
            current_vert = edge.verts[0]
            
            while current_edge and current_edge not in used_edges:
                used_edges.add(current_edge)
                loop.append(current_vert.index)
                
                # Find next edge
                next_edge = None
                for e in current_vert.link_edges:
                    if e != current_edge and e in boundary_edges and e not in used_edges:
                        next_edge = e
                        break
                        
                if next_edge:
                    # Move to next vertex
                    current_vert = next_edge.other_vert(current_vert)
                    current_edge = next_edge
                else:
                    break
                    
            if len(loop) > 2:
                loops.append(loop)
                
        bm.free()
        return loops
        
    def _create_rim_geometry(self,
                           loop: List[int],
                           obj: bpy.types.Object,
                           params: Dict) -> Dict:
        """Create rim geometry for a single edge loop"""
        geometry = {
            'vertices': [],
            'edges': [],
            'faces': []
        }
        
        mesh = obj.data
        loop_verts = [mesh.vertices[i].co for i in loop]
        
        # Smooth loop if needed
        if params['corner_radius'] > 0:
            loop_verts = self._smooth_loop_corners(
                loop_verts,
                params['corner_radius']
            )
            
        # Generate rim profile
        for i, vert_co in enumerate(loop_verts):
            # Calculate local coordinate system
            prev_idx = (i - 1) % len(loop_verts)
            next_idx = (i + 1) % len(loop_verts)
            
            tangent = (loop_verts[next_idx] - loop_verts[prev_idx]).normalized()
            
            # Get normal from mesh
            vert_normal = mesh.vertices[loop[i % len(loop)]].normal
            
            # Calculate rim direction (inward)
            rim_dir = tangent.cross(vert_normal).normalized()
            
            # Generate rim profile points
            profile_points = self._generate_rim_profile(
                vert_co,
                rim_dir,
                vert_normal,
                tangent,
                params,
                i / len(loop_verts)  # Parameter along loop
            )
            
            # Add vertices
            for point in profile_points:
                geometry['vertices'].append(point)
                
        # Create faces along rim
        profile_size = self._get_profile_point_count(params['inner_pattern'])
        
        for i in range(len(loop_verts)):
            next_i = (i + 1) % len(loop_verts)
            
            for j in range(profile_size - 1):
                # Current profile indices
                curr_base = i * profile_size
                next_base = next_i * profile_size
                
                # Quad face
                geometry['faces'].append([
                    curr_base + j,
                    curr_base + j + 1,
                    next_base + j + 1,
                    next_base + j
                ])
                
        return geometry
        
    def _smooth_loop_corners(self,
                           verts: List[Vector],
                           radius: float) -> List[Vector]:
        """Smooth sharp corners in loop"""
        if len(verts) < 3:
            return verts
            
        smoothed = []
        
        for i in range(len(verts)):
            prev_idx = (i - 1) % len(verts)
            next_idx = (i + 1) % len(verts)
            
            # Calculate angle
            v1 = verts[i] - verts[prev_idx]
            v2 = verts[next_idx] - verts[i]
            
            if v1.length > 0 and v2.length > 0:
                v1.normalize()
                v2.normalize()
                angle = math.acos(max(-1, min(1, v1.dot(v2))))
                
                # Smooth if sharp corner
                if angle < math.pi * 0.75:  # Less than 135 degrees
                    # Simple corner smoothing
                    blend = min(radius / v1.length, 0.3)
                    smooth_pos = verts[i] * (1 - blend) + (verts[prev_idx] + verts[next_idx]) * (blend / 2)
                    smoothed.append(smooth_pos)
                else:
                    smoothed.append(verts[i])
            else:
                smoothed.append(verts[i])
                
        return smoothed
        
    def _generate_rim_profile(self,
                            base_pos: Vector,
                            rim_dir: Vector,
                            normal: Vector,
                            tangent: Vector,
                            params: Dict,
                            t: float) -> List[Vector]:
        """Generate rim cross-section profile"""
        profile_points = []
        
        width = params['rim_width']
        height = params['rim_height']
        
        if params['inner_pattern'] == 'straight':
            # Simple rectangular profile
            profile_points.extend([
                base_pos,
                base_pos + rim_dir * width,
                base_pos + rim_dir * width + normal * height,
                base_pos + normal * height
            ])
            
        elif params['inner_pattern'] == 'wave':
            # Wavy inner edge
            wave_amp = params['wave_amplitude']
            wave_freq = params['wave_frequency']
            
            # Sample points along wave
            n_points = 6
            for i in range(n_points):
                u = i / (n_points - 1)
                
                # Base position along width
                pos = base_pos + rim_dir * (width * u)
                
                # Add wave displacement
                if 0.2 < u < 0.8:  # Only wave the middle section
                    wave_offset = math.sin(t * wave_freq * 2 * math.pi) * wave_amp
                    pos = pos + tangent * wave_offset
                    
                # Add height variations
                if i == 0 or i == n_points - 1:
                    profile_points.append(pos)
                    profile_points.append(pos + normal * height)
                else:
                    h = height * (0.8 + 0.2 * math.sin(u * math.pi))
                    profile_points.append(pos + normal * h)
                    
        elif params['inner_pattern'] == 'zigzag':
            # Zigzag pattern
            n_segments = 4
            for i in range(n_segments + 1):
                u = i / n_segments
                
                # Zigzag offset
                if i % 2 == 0:
                    offset = 0
                else:
                    offset = params['wave_amplitude']
                    
                pos = base_pos + rim_dir * (width * u) + tangent * offset
                
                profile_points.append(pos)
                profile_points.append(pos + normal * height)
                
        return profile_points
        
    def _get_profile_point_count(self, pattern: str) -> int:
        """Get number of points in rim profile"""
        if pattern == 'straight':
            return 4
        elif pattern == 'wave':
            return 12
        elif pattern == 'zigzag':
            return 10
        return 4
        
    def _calculate_total_length(self, loops: List[List[int]], obj: bpy.types.Object) -> float:
        """Calculate total length of edge loops"""
        total = 0.0
        mesh = obj.data
        
        for loop in loops:
            for i in range(len(loop)):
                next_i = (i + 1) % len(loop)
                v1 = mesh.vertices[loop[i]].co
                v2 = mesh.vertices[loop[next_i]].co
                total += (v2 - v1).length
                
        return total


class ContourBanding(BasePattern):
    """Contour bands following surface curvature"""
    
    def __init__(self):
        super().__init__()
        self.name = "Contour Banding"
        self.category = "Edge"
        self.description = "Reinforcement bands following surface contours"
        
    def get_default_params(self) -> Dict:
        return {
            'band_spacing': 0.02,  # Spacing between bands
            'band_width': 0.005,  # Width of each band
            'band_height': 0.002,  # Height of bands
            'follow_curvature': True,  # Follow surface curvature
            'adaptive_spacing': True,  # Vary spacing by curvature
            'min_spacing': 0.01,  # Minimum band spacing
            'max_spacing': 0.04,  # Maximum band spacing
            'band_profile': 'rectangular',  # 'rectangular', 'rounded', 'triangular'
            'pattern_offset': 0.001  # Offset from surface
        }
        
    def validate_params(self, params: Dict) -> Tuple[bool, str]:
        if params.get('band_spacing', 0) <= 0:
            return False, "Band spacing must be positive"
        if params.get('band_width', 0) <= 0:
            return False, "Band width must be positive"
        return True, ""
        
    def generate(self, target_object: bpy.types.Object, **params) -> PatternResult:
        # Merge with defaults
        full_params = self.get_default_params()
        full_params.update(params)
        
        # Validate
        valid, error = self.validate_params(full_params)
        if not valid:
            return PatternResult([], [], [], {}, {}, False, error)
            
        # Generate contour lines
        contours = self._generate_contours(
            target_object,
            full_params['band_spacing'],
            full_params['adaptive_spacing'],
            full_params['min_spacing'],
            full_params['max_spacing'],
            full_params['follow_curvature']
        )
        
        # Create band geometry
        pattern_verts = []
        pattern_edges = []
        pattern_faces = []
        
        for contour in contours:
            band_geom = self._create_band_geometry(
                contour,
                full_params['band_width'],
                full_params['band_height'],
                full_params['band_profile'],
                target_object
            )
            
            # Add to pattern
            base_idx = len(pattern_verts)
            pattern_verts.extend(band_geom['vertices'])
            
            for edge in band_geom['edges']:
                pattern_edges.append((edge[0] + base_idx, edge[1] + base_idx))
                
            for face in band_geom['faces']:
                pattern_faces.append([v + base_idx for v in face])
                
        # Apply offset
        pattern_verts = self.offset_from_surface(
            pattern_verts,
            target_object,
            full_params['pattern_offset']
        )
        
        # Create attributes
        attributes = {
            'thickness': [full_params['band_height']] * len(pattern_verts),
            'material_index': [1] * len(pattern_faces)
        }
        
        metadata = {
            'pattern_type': 'contour_banding',
            'band_count': len(contours),
            'parameters': full_params
        }
        
        return PatternResult(
            vertices=pattern_verts,
            edges=pattern_edges,
            faces=pattern_faces,
            attributes=attributes,
            metadata=metadata
        )
        
    def _generate_contours(self,
                         obj: bpy.types.Object,
                         spacing: float,
                         adaptive: bool,
                         min_spacing: float,
                         max_spacing: float,
                         follow_curvature: bool) -> List[List[Vector]]:
        """Generate contour lines on surface"""
        contours = []
        
        # Get bounds
        bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        min_z = min(v.z for v in bbox)
        max_z = max(v.z for v in bbox)
        
        # Generate horizontal slicing planes
        current_z = min_z + spacing
        
        while current_z < max_z:
            # Create slicing plane
            plane_co = Vector((0, 0, current_z))
            plane_no = Vector((0, 0, 1))
            
            # Find intersection with mesh
            contour = self._intersect_mesh_plane(obj, plane_co, plane_no)
            
            if contour:
                # Process contour
                if follow_curvature:
                    contour = self._adapt_to_curvature(contour, obj)
                    
                contours.append(contour)
                
            # Adaptive spacing
            if adaptive:
                # Adjust spacing based on local curvature
                local_curv = self._estimate_local_curvature(obj, current_z)
                adjusted_spacing = spacing * (1 + local_curv)
                adjusted_spacing = max(min_spacing, min(max_spacing, adjusted_spacing))
                current_z += adjusted_spacing
            else:
                current_z += spacing
                
        return contours
        
    def _intersect_mesh_plane(self,
                            obj: bpy.types.Object,
                            plane_co: Vector,
                            plane_no: Vector) -> List[Vector]:
        """Find intersection of mesh with plane"""
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.transform(obj.matrix_world)
        
        # Use bmesh bisect
        geom = bmesh.ops.bisect_plane(
            bm,
            geom=bm.faces[:] + bm.edges[:] + bm.verts[:],
            plane_co=plane_co,
            plane_no=plane_no,
            clear_outer=False,
            clear_inner=False
        )
        
        # Extract intersection edges
        intersection_points = []
        for edge in geom['geom_cut']:
            if isinstance(edge, bmesh.types.BMEdge):
                intersection_points.append(edge.verts[0].co.copy())
                intersection_points.append(edge.verts[1].co.copy())
                
        bm.free()
        
        # Order points into continuous contour
        if intersection_points:
            return self._order_contour_points(intersection_points)
        return []
        
    def _order_contour_points(self, points: List[Vector]) -> List[Vector]:
        """Order scattered points into continuous contour"""
        if len(points) < 3:
            return points
            
        # Simple nearest-neighbor ordering
        ordered = [points[0]]
        remaining = set(range(1, len(points)))
        
        while remaining:
            last_point = ordered[-1]
            
            # Find nearest remaining point
            min_dist = float('inf')
            nearest_idx = -1
            
            for idx in remaining:
                dist = (points[idx] - last_point).length
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = idx
                    
            if nearest_idx >= 0 and min_dist < 0.01:  # 10mm threshold
                ordered.append(points[nearest_idx])
                remaining.remove(nearest_idx)
            else:
                break
                
        return ordered
        
    def _adapt_to_curvature(self,
                          contour: List[Vector],
                          obj: bpy.types.Object) -> List[Vector]:
        """Adapt contour to follow surface curvature"""
        if not contour:
            return contour
            
        adapted = []
        bvh = self.mesh_analyzer.get_bvh_tree(obj)
        
        for point in contour:
            # Project to surface
            location, normal, index, distance = bvh.find_nearest(point)
            if location:
                adapted.append(location)
            else:
                adapted.append(point)
                
        return adapted
        
    def _estimate_local_curvature(self, obj: bpy.types.Object, z_level: float) -> float:
        """Estimate curvature at given Z level"""
        # Simplified estimation
        # In production, use proper curvature analysis
        return 0.5
        
    def _create_band_geometry(self,
                            contour: List[Vector],
                            width: float,
                            height: float,
                            profile: str,
                            obj: bpy.types.Object) -> Dict:
        """Create geometry for a contour band"""
        geometry = {
            'vertices': [],
            'edges': [],
            'faces': []
        }
        
        if len(contour) < 2:
            return geometry
            
        # Generate band profile along contour
        for i, point in enumerate(contour):
            # Calculate local frame
            prev_idx = (i - 1) % len(contour)
            next_idx = (i + 1) % len(contour)
            
            tangent = (contour[next_idx] - contour[prev_idx]).normalized()
            
            # Get surface normal at point
            bvh = self.mesh_analyzer.get_bvh_tree(obj)
            location, normal, index, distance = bvh.find_nearest(point)
            
            if not normal:
                normal = Vector((0, 0, 1))
                
            # Calculate band width direction
            width_dir = tangent.cross(normal).normalized()
            
            # Generate profile points
            if profile == 'rectangular':
                profile_points = [
                    point - width_dir * (width / 2),
                    point + width_dir * (width / 2),
                    point + width_dir * (width / 2) + normal * height,
                    point - width_dir * (width / 2) + normal * height
                ]
            elif profile == 'rounded':
                # Rounded profile with more points
                n_points = 6
                profile_points = []
                for j in range(n_points):
                    angle = j * math.pi / (n_points - 1)
                    offset = Vector((
                        -math.cos(angle) * width / 2,
                        0,
                        math.sin(angle) * height
                    ))
                    # Transform to local frame
                    local_offset = width_dir * offset.x + normal * offset.z
                    profile_points.append(point + local_offset)
            else:  # triangular
                profile_points = [
                    point - width_dir * (width / 2),
                    point + width_dir * (width / 2),
                    point + normal * height
                ]
                
            geometry['vertices'].extend(profile_points)
            
        # Create faces
        points_per_profile = len(geometry['vertices']) // len(contour)
        
        for i in range(len(contour)):
            next_i = (i + 1) % len(contour)
            base_idx = i * points_per_profile
            next_base = next_i * points_per_profile
            
            # Connect profiles
            for j in range(points_per_profile - 1):
                if profile == 'triangular' and j == 1:
                    # Triangle face
                    geometry['faces'].append([
                        base_idx + j,
                        next_base + j,
                        base_idx + j + 1
                    ])
                else:
                    # Quad face
                    geometry['faces'].append([
                        base_idx + j,
                        next_base + j,
                        next_base + j + 1,
                        base_idx + j + 1
                    ])
                    
        return geometry


class StressEdgeLoop(BasePattern):
    """Edge loops positioned at high-stress areas"""
    
    def __init__(self):
        super().__init__()
        self.name = "Stress Edge Loop"
        self.category = "Edge"
        self.description = "Reinforcement loops at stress concentration points"
        
    def get_default_params(self) -> Dict:
        return {
            'loop_width': 0.008,  # Width of stress loops
            'loop_height': 0.003,  # Height of loops
            'stress_threshold': 0.6,  # Minimum stress level (0-1)
            'loop_density': 'adaptive',  # 'uniform', 'adaptive'
            'min_loop_spacing': 0.015,  # Minimum spacing between loops
            'cross_pattern': True,  # Add crossing patterns at intersections
            'pattern_offset': 0.001  # Offset from surface
        }
        
    def validate_params(self, params: Dict) -> Tuple[bool, str]:
        if params.get('loop_width', 0) <= 0:
            return False, "Loop width must be positive"
        if not 0 <= params.get('stress_threshold', 0.6) <= 1:
            return False, "Stress threshold must be between 0 and 1"
        return True, ""
        
    def generate(self, target_object: bpy.types.Object, **params) -> PatternResult:
        # Merge with defaults
        full_params = self.get_default_params()
        full_params.update(params)
        
        # Validate
        valid, error = self.validate_params(full_params)
        if not valid:
            return PatternResult([], [], [], {}, {}, False, error)
            
        # Analyze stress distribution
        stress_map = self._analyze_stress_distribution(target_object)
        
        # Generate stress-based edge loops
        stress_loops = self._generate_stress_loops(
            target_object,
            stress_map,
            full_params['stress_threshold'],
            full_params['loop_density'],
            full_params['min_loop_spacing']
        )
        
        # Create loop geometry
        pattern_verts = []
        pattern_edges = []
        pattern_faces = []
        
        for loop in stress_loops:
            loop_geom = self._create_stress_loop_geometry(
                loop,
                full_params['loop_width'],
                full_params['loop_height'],
                target_object
            )
            
            # Add to pattern
            base_idx = len(pattern_verts)
            pattern_verts.extend(loop_geom['vertices'])
            
            for edge in loop_geom['edges']:
                pattern_edges.append((edge[0] + base_idx, edge[1] + base_idx))
                
            for face in loop_geom['faces']:
                pattern_faces.append([v + base_idx for v in face])
                
        # Add crossing patterns if enabled
        if full_params['cross_pattern']:
            crossings = self._find_loop_intersections(stress_loops)
            for crossing in crossings:
                cross_geom = self._create_crossing_pattern(
                    crossing,
                    full_params['loop_width'],
                    full_params['loop_height'],
                    target_object
                )
                
                base_idx = len(pattern_verts)
                pattern_verts.extend(cross_geom['vertices'])
                
                for edge in cross_geom['edges']:
                    pattern_edges.append((edge[0] + base_idx, edge[1] + base_idx))
                    
                for face in cross_geom['faces']:
                    pattern_faces.append([v + base_idx for v in face])
                    
        # Apply offset
        pattern_verts = self.offset_from_surface(
            pattern_verts,
            target_object,
            full_params['pattern_offset']
        )
        
        # Create attributes
        attributes = {
            'thickness': [full_params['loop_height']] * len(pattern_verts),
            'material_index': [1] * len(pattern_faces),
            'stress_level': self._calculate_vertex_stress(pattern_verts, stress_map)
        }
        
        metadata = {
            'pattern_type': 'stress_edge_loop',
            'loop_count': len(stress_loops),
            'crossing_count': len(crossings) if full_params['cross_pattern'] else 0,
            'parameters': full_params
        }
        
        return PatternResult(
            vertices=pattern_verts,
            edges=pattern_edges,
            faces=pattern_faces,
            attributes=attributes,
            metadata=metadata
        )
        
    def _analyze_stress_distribution(self, obj: bpy.types.Object) -> Dict[Vector, float]:
        """Analyze stress distribution on object"""
        stress_map = {}
        mesh = obj.data
        
        # Simplified stress analysis based on:
        # 1. Distance from edges
        # 2. Local curvature
        # 3. Support points (bottom vertices)
        
        # Get edge information
        edges = self.edge_detector.detect_edges(obj)
        edge_verts = set()
        for edge in edges:
            edge_verts.add(edge[0])
            edge_verts.add(edge[1])
            
        # Find support vertices (bottom 20%)
        z_values = [v.co.z for v in mesh.vertices]
        z_threshold = min(z_values) + (max(z_values) - min(z_values)) * 0.2
        support_verts = set(i for i, v in enumerate(mesh.vertices) if v.co.z < z_threshold)
        
        # Calculate stress for each vertex
        for vert in mesh.vertices:
            stress = 0.3  # Base stress
            
            # Edge proximity increases stress
            if vert.index in edge_verts:
                stress += 0.4
                
            # Support areas have high stress
            if vert.index in support_verts:
                stress += 0.3
                
            # Curvature increases stress
            if len(vert.link_faces) > 1:
                normals = [mesh.polygons[f].normal for f in vert.link_faces]
                avg_normal = sum(normals, Vector()) / len(normals)
                deviation = sum((n - avg_normal).length for n in normals) / len(normals)
                stress += min(deviation, 0.3)
                
            stress_map[vert.co.copy()] = min(stress, 1.0)
            
        return stress_map
        
    def _generate_stress_loops(self,
                             obj: bpy.types.Object,
                             stress_map: Dict[Vector, float],
                             threshold: float,
                             density: str,
                             min_spacing: float) -> List[List[Vector]]:
        """Generate loops following stress contours"""
        loops = []
        
        # Extract high-stress regions
        high_stress_points = [
            point for point, stress in stress_map.items()
            if stress >= threshold
        ]
        
        if not high_stress_points:
            return loops
            
        # Cluster high-stress points
        clusters = self._cluster_points(high_stress_points, min_spacing * 2)
        
        # Generate loops around clusters
        for cluster in clusters:
            if len(cluster) < 3:
                continue
                
            # Create convex hull
            loop = self._create_convex_loop(cluster)
            
            # Smooth and project to surface
            loop = self._smooth_and_project_loop(loop, obj)
            
            loops.append(loop)
            
        # Add density-based subdivisions
        if density == 'adaptive':
            loops = self._subdivide_loops_by_stress(loops, stress_map, min_spacing)
            
        return loops
        
    def _cluster_points(self,
                       points: List[Vector],
                       cluster_distance: float) -> List[List[Vector]]:
        """Cluster nearby points"""
        clusters = []
        used = set()
        
        for i, point in enumerate(points):
            if i in used:
                continue
                
            cluster = [point]
            used.add(i)
            
            # Find nearby points
            for j, other in enumerate(points[i+1:], i+1):
                if j not in used and (point - other).length < cluster_distance:
                    cluster.append(other)
                    used.add(j)
                    
            clusters.append(cluster)
            
        return clusters
        
    def _create_convex_loop(self, points: List[Vector]) -> List[Vector]:
        """Create convex hull loop from points"""
        if len(points) < 3:
            return points
            
        # Project to 2D for hull calculation
        center = sum(points, Vector()) / len(points)
        
        # Simple 2D projection
        points_2d = []
        for p in points:
            offset = p - center
            points_2d.append([offset.x, offset.y])
            
        # Calculate hull
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(np.array(points_2d))
            hull_indices = hull.vertices
        except:
            return points
            
        # Return hull points in order
        return [points[i] for i in hull_indices]
        
    def _smooth_and_project_loop(self,
                                loop: List[Vector],
                                obj: bpy.types.Object) -> List[Vector]:
        """Smooth loop and project to surface"""
        if len(loop) < 3:
            return loop
            
        # Smooth
        smoothed = []
        for i in range(len(loop)):
            prev_idx = (i - 1) % len(loop)
            next_idx = (i + 1) % len(loop)
            
            smooth_pos = (loop[prev_idx] + loop[i] * 2 + loop[next_idx]) / 4
            smoothed.append(smooth_pos)
            
        # Project to surface
        projected = []
        bvh = self.mesh_analyzer.get_bvh_tree(obj)
        
        for point in smoothed:
            location, normal, index, distance = bvh.find_nearest(point)
            if location:
                projected.append(location)
            else:
                projected.append(point)
                
        return projected
        
    def _subdivide_loops_by_stress(self,
                                 loops: List[List[Vector]],
                                 stress_map: Dict[Vector, float],
                                 min_spacing: float) -> List[List[Vector]]:
        """Subdivide loops based on local stress"""
        subdivided = []
        
        for loop in loops:
            new_loop = []
            
            for i in range(len(loop)):
                next_i = (i + 1) % len(loop)
                
                new_loop.append(loop[i])
                
                # Check if subdivision needed
                edge_length = (loop[next_i] - loop[i]).length
                
                if edge_length > min_spacing * 1.5:
                    # Get stress at midpoint
                    midpoint = (loop[i] + loop[next_i]) / 2
                    
                    # Find nearest stress value
                    min_dist = float('inf')
                    mid_stress = 0.5
                    
                    for point, stress in stress_map.items():
                        dist = (point - midpoint).length
                        if dist < min_dist:
                            min_dist = dist
                            mid_stress = stress
                            
                    # Subdivide if high stress
                    if mid_stress > 0.7:
                        new_loop.append(midpoint)
                        
            subdivided.append(new_loop)
            
        return subdivided
        
    def _find_loop_intersections(self,
                               loops: List[List[Vector]]) -> List[Dict]:
        """Find intersection points between loops"""
        intersections = []
        
        # This is simplified - in production use proper intersection algorithm
        for i, loop1 in enumerate(loops):
            for j, loop2 in enumerate(loops[i+1:], i+1):
                # Check for proximity between loops
                for p1 in loop1:
                    for p2 in loop2:
                        if (p1 - p2).length < 0.01:  # 10mm threshold
                            intersections.append({
                                'position': (p1 + p2) / 2,
                                'loops': [i, j]
                            })
                            
        return intersections
        
    def _create_stress_loop_geometry(self,
                                   loop: List[Vector],
                                   width: float,
                                   height: float,
                                   obj: bpy.types.Object) -> Dict:
        """Create geometry for stress loop"""
        geometry = {
            'vertices': [],
            'edges': [],
            'faces': []
        }
        
        if len(loop) < 3:
            return geometry
            
        # Calculate loop center
        center = sum(loop, Vector()) / len(loop)
        
        # Generate inner and outer loops
        for point in loop:
            # Outer point
            geometry['vertices'].append(point)
            
            # Inner point
            direction = (center - point).normalized()
            inner = point + direction * width
            geometry['vertices'].append(inner)
            
            # Top vertices
            bvh = self.mesh_analyzer.get_bvh_tree(obj)
            location, normal, index, distance = bvh.find_nearest(point)
            
            if normal:
                geometry['vertices'].append(point + normal * height)
                geometry['vertices'].append(inner + normal * height)
            else:
                geometry['vertices'].append(point + Vector((0, 0, height)))
                geometry['vertices'].append(inner + Vector((0, 0, height)))
                
        # Create faces
        n = len(loop)
        for i in range(n):
            next_i = (i + 1) % n
            
            # Vertex indices
            outer_bottom = i * 4
            inner_bottom = i * 4 + 1
            outer_top = i * 4 + 2
            inner_top = i * 4 + 3
            
            next_outer_bottom = next_i * 4
            next_inner_bottom = next_i * 4 + 1
            next_outer_top = next_i * 4 + 2
            next_inner_top = next_i * 4 + 3
            
            # Side faces
            geometry['faces'].extend([
                # Outer side
                [outer_bottom, next_outer_bottom, next_outer_top, outer_top],
                # Inner side
                [inner_bottom, inner_top, next_inner_top, next_inner_bottom],
                # Top
                [outer_top, next_outer_top, next_inner_top, inner_top],
                # Bottom
                [outer_bottom, inner_bottom, next_inner_bottom, next_outer_bottom]
            ])
            
        return geometry
        
    def _create_crossing_pattern(self,
                               crossing: Dict,
                               width: float,
                               height: float,
                               obj: bpy.types.Object) -> Dict:
        """Create reinforcement pattern at loop intersection"""
        geometry = {
            'vertices': [],
            'edges': [],
            'faces': []
        }
        
        center = crossing['position']
        
        # Create star pattern
        n_points = 8
        radius = width * 2
        
        for i in range(n_points):
            angle = i * 2 * math.pi / n_points
            
            # Alternate between inner and outer radius
            if i % 2 == 0:
                r = radius
            else:
                r = radius * 0.5
                
            point = center + Vector((
                r * math.cos(angle),
                r * math.sin(angle),
                0
            ))
            
            # Project to surface and add height
            bvh = self.mesh_analyzer.get_bvh_tree(obj)
            location, normal, index, distance = bvh.find_nearest(point)
            
            if location:
                geometry['vertices'].append(location)
                geometry['vertices'].append(location + normal * height)
            else:
                geometry['vertices'].append(point)
                geometry['vertices'].append(point + Vector((0, 0, height)))
                
        # Create faces
        for i in range(n_points):
            next_i = (i + 1) % n_points
            
            # Skip every other face for star pattern
            if i % 2 == 0:
                geometry['faces'].append([
                    i * 2, next_i * 2,
                    next_i * 2 + 1, i * 2 + 1
                ])
                
        return geometry
        
    def _calculate_vertex_stress(self,
                               vertices: List[Vector],
                               stress_map: Dict[Vector, float]) -> List[float]:
        """Calculate stress value for each vertex"""
        stress_values = []
        
        for vert in vertices:
            # Find nearest stress value
            min_dist = float('inf')
            stress = 0.5
            
            for point, s in stress_map.items():
                dist = (point - vert).length
                if dist < min_dist:
                    min_dist = dist
                    stress = s
                    
            stress_values.append(stress)
            
        return stress_values
