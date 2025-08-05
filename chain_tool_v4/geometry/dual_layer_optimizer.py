"""
Dual Layer Optimizer for Chain Tool V4
Optimizes separation and connection between TPU shell and PETG reinforcement
"""

import bpy
import bmesh
import numpy as np
from mathutils import Vector, Matrix
from typing import List, Dict, Tuple, Set, Optional
from ..utils.performance import measure_time
from ..utils.caching import CacheManager
from ..utils.debug import DebugManager
from ..utils.math_utils import MathUtils
from ..core.state_manager import StateManager
from ..core.constants import LayerType, MaterialType
from .mesh_analyzer import MeshAnalyzer
from .edge_detector import EdgeDetector

class DualLayerOptimizer:
    """Optimizes dual-material layer separation and connections"""
    
    def __init__(self):
        self.cache = CacheManager()
        self.debug = DebugManager()
        self.state = StateManager()
        self.mesh_analyzer = MeshAnalyzer()
        self.edge_detector = EdgeDetector()
        self.math = MathUtils()
        
    @measure_time
    def optimize_layers(self,
                       base_mesh: bpy.types.Object,
                       pattern_mesh: bpy.types.Object,
                       shell_thickness: float = 0.003,  # 3mm TPU
                       pattern_offset: float = 0.0005,  # 0.5mm gap
                       connection_radius: float = 0.002,  # 2mm connections
                       min_connection_spacing: float = 0.02) -> Dict:
        """
        Optimize dual-layer structure for 3D printing
        
        Args:
            base_mesh: TPU shell mesh
            pattern_mesh: PETG pattern mesh
            shell_thickness: Thickness of TPU shell
            pattern_offset: Offset between shell and pattern
            connection_radius: Radius of connection points
            min_connection_spacing: Minimum distance between connections
            
        Returns:
            Dictionary with optimization results
        """
        results = {
            'success': False,
            'shell_mesh': None,
            'pattern_mesh': None,
            'connections': [],
            'overlap_areas': [],
            'gap_areas': []
        }
        
        # Validate inputs
        if not self._validate_meshes(base_mesh, pattern_mesh):
            self.debug.error("Invalid mesh inputs for dual layer optimization")
            return results
            
        # Step 1: Generate inner shell surface
        inner_shell = self._generate_inner_shell(base_mesh, shell_thickness)
        
        # Step 2: Offset pattern from inner shell
        offset_pattern = self._offset_pattern_mesh(
            pattern_mesh, 
            inner_shell, 
            pattern_offset
        )
        
        # Step 3: Detect and resolve intersections
        intersections = self._detect_intersections(inner_shell, offset_pattern)
        if intersections:
            offset_pattern = self._resolve_intersections(
                offset_pattern, 
                inner_shell, 
                intersections
            )
            
        # Step 4: Identify connection points
        connections = self._find_optimal_connections(
            inner_shell,
            offset_pattern,
            connection_radius,
            min_connection_spacing
        )
        
        # Step 5: Generate connection geometry
        connection_geometry = self._generate_connections(
            connections,
            connection_radius
        )
        
        # Step 6: Analyze gaps and coverage
        analysis = self._analyze_layer_separation(
            base_mesh,
            inner_shell,
            offset_pattern
        )
        
        # Store results
        results['success'] = True
        results['shell_mesh'] = inner_shell
        results['pattern_mesh'] = offset_pattern
        results['connections'] = connection_geometry
        results['overlap_areas'] = analysis['overlaps']
        results['gap_areas'] = analysis['gaps']
        
        # Cache results
        cache_key = f"dual_layer_{base_mesh.name}_{pattern_mesh.name}"
        self.cache.set(cache_key, results)
        
        return results
        
    def _validate_meshes(self, base: bpy.types.Object, pattern: bpy.types.Object) -> bool:
        """Validate input meshes"""
        if not base or not pattern:
            return False
            
        if base.type != 'MESH' or pattern.type != 'MESH':
            return False
            
        # Check for valid geometry
        if not base.data.vertices or not pattern.data.vertices:
            return False
            
        return True
        
    def _generate_inner_shell(self, base_mesh: bpy.types.Object, thickness: float) -> bpy.types.Object:
        """Generate inner shell surface by offsetting inward"""
        # Duplicate mesh
        inner_shell = base_mesh.copy()
        inner_shell.data = base_mesh.data.copy()
        inner_shell.name = f"{base_mesh.name}_InnerShell"
        bpy.context.collection.objects.link(inner_shell)
        
        # Apply solidify modifier in reverse
        solidify = inner_shell.modifiers.new("InnerShell", 'SOLIDIFY')
        solidify.thickness = -thickness  # Negative for inward
        solidify.offset = 0  # From outer surface inward
        solidify.use_quality_normals = True
        solidify.use_even_offset = True
        
        # Apply modifier
        bpy.context.view_layer.objects.active = inner_shell
        bpy.ops.object.modifier_apply(modifier=solidify.name)
        
        # Remove outer faces (keep only inner shell)
        self._remove_outer_faces(inner_shell, base_mesh)
        
        return inner_shell
        
    def _remove_outer_faces(self, shell_obj: bpy.types.Object, original_obj: bpy.types.Object):
        """Remove outer faces, keeping only inner shell"""
        bm = bmesh.new()
        bm.from_mesh(shell_obj.data)
        bm.faces.ensure_lookup_table()
        
        # Get BVH of original mesh
        orig_bvh = self.mesh_analyzer.get_bvh_tree(original_obj)
        
        # Mark faces that are on the original surface
        faces_to_remove = []
        for face in bm.faces:
            # Check if face center is on original surface
            center = face.calc_center_median()
            location, normal, index, distance = orig_bvh.find_nearest(center)
            
            # If very close to original surface, it's an outer face
            if distance < 0.0001:  # 0.1mm threshold
                faces_to_remove.append(face)
                
        # Remove outer faces
        bmesh.ops.delete(bm, geom=faces_to_remove, context='FACES')
        
        # Update mesh
        bm.to_mesh(shell_obj.data)
        bm.free()
        
    def _offset_pattern_mesh(self, 
                           pattern: bpy.types.Object, 
                           reference: bpy.types.Object, 
                           offset: float) -> bpy.types.Object:
        """Offset pattern mesh from reference surface"""
        # Duplicate pattern
        offset_pattern = pattern.copy()
        offset_pattern.data = pattern.data.copy()
        offset_pattern.name = f"{pattern.name}_Offset"
        bpy.context.collection.objects.link(offset_pattern)
        
        # Get reference BVH
        ref_bvh = self.mesh_analyzer.get_bvh_tree(reference)
        
        # Offset vertices toward reference surface normal
        mesh = offset_pattern.data
        for vert in mesh.vertices:
            # Find closest point on reference
            location, normal, index, distance = ref_bvh.find_nearest(vert.co)
            
            if location:
                # Calculate offset direction (away from reference)
                direction = (vert.co - location).normalized()
                if direction.length == 0:
                    direction = normal
                    
                # Apply offset
                vert.co = location + direction * (distance + offset)
                
        mesh.update()
        return offset_pattern
        
    def _detect_intersections(self, 
                            mesh1: bpy.types.Object, 
                            mesh2: bpy.types.Object) -> List[Dict]:
        """Detect intersections between two meshes"""
        intersections = []
        
        # Use BMesh boolean operation to find intersections
        bm1 = bmesh.new()
        bm1.from_mesh(mesh1.data)
        
        bm2 = bmesh.new() 
        bm2.from_mesh(mesh2.data)
        
        # Get BVH trees
        bvh1 = self.mesh_analyzer.get_bvh_tree(mesh1)
        bvh2 = self.mesh_analyzer.get_bvh_tree(mesh2)
        
        # Check each face in mesh2 against mesh1
        for face in bm2.faces:
            center = face.calc_center_median()
            
            # Ray cast from face center along normal
            hit_location, hit_normal, hit_index, hit_distance = bvh1.ray_cast(
                center, 
                face.normal
            )
            
            if hit_location:
                # Check if it's a real intersection
                back_hit = bvh1.ray_cast(center, -face.normal)
                if back_hit[0]:  # Hit from both sides = inside
                    intersections.append({
                        'face_index': face.index,
                        'location': center,
                        'normal': face.normal,
                        'depth': hit_distance
                    })
                    
        bm1.free()
        bm2.free()
        
        return intersections
        
    def _resolve_intersections(self,
                             pattern: bpy.types.Object,
                             shell: bpy.types.Object,
                             intersections: List[Dict]) -> bpy.types.Object:
        """Resolve intersections by pushing pattern vertices outward"""
        if not intersections:
            return pattern
            
        mesh = pattern.data
        shell_bvh = self.mesh_analyzer.get_bvh_tree(shell)
        
        # Group intersections by proximity
        affected_verts = set()
        for intersection in intersections:
            # Find vertices near intersection
            for vert in mesh.vertices:
                if (vert.co - intersection['location']).length < 0.01:  # 10mm radius
                    affected_verts.add(vert.index)
                    
        # Push affected vertices outward
        for vert_idx in affected_verts:
            vert = mesh.vertices[vert_idx]
            
            # Find closest point on shell
            location, normal, index, distance = shell_bvh.find_nearest(vert.co)
            
            if location:
                # Push outward by safe margin
                direction = (vert.co - location).normalized()
                if direction.length == 0:
                    direction = normal
                    
                min_distance = 0.001  # 1mm minimum clearance
                if distance < min_distance:
                    vert.co = location + direction * min_distance
                    
        mesh.update()
        return pattern
        
    def _find_optimal_connections(self,
                                shell: bpy.types.Object,
                                pattern: bpy.types.Object,
                                radius: float,
                                min_spacing: float) -> List[Dict]:
        """Find optimal connection points between layers"""
        connections = []
        
        # Get high-stress areas from edge detection
        edges = self.edge_detector.detect_edges(shell)
        edge_verts = set()
        for edge in edges:
            edge_verts.add(edge[0])
            edge_verts.add(edge[1])
            
        # Get pattern BVH
        pattern_bvh = self.mesh_analyzer.get_bvh_tree(pattern)
        
        # Sample potential connection points
        shell_mesh = shell.data
        candidates = []
        
        # Priority 1: Edge vertices (high stress areas)
        for vert_idx in edge_verts:
            vert = shell_mesh.vertices[vert_idx]
            
            # Find closest point on pattern
            p_location, p_normal, p_index, p_distance = pattern_bvh.find_nearest(vert.co)
            
            if p_location and p_distance < 0.01:  # Within 10mm
                candidates.append({
                    'shell_pos': vert.co.copy(),
                    'pattern_pos': p_location,
                    'distance': p_distance,
                    'priority': 1.0,  # High priority for edges
                    'normal': p_normal
                })
                
        # Priority 2: Regular grid sampling
        grid_size = min_spacing * 2
        bounds_min, bounds_max = self._get_bounds(shell)
        
        x_steps = int((bounds_max.x - bounds_min.x) / grid_size) + 1
        y_steps = int((bounds_max.y - bounds_min.y) / grid_size) + 1
        z_steps = int((bounds_max.z - bounds_min.z) / grid_size) + 1
        
        shell_bvh = self.mesh_analyzer.get_bvh_tree(shell)
        
        for x in range(x_steps):
            for y in range(y_steps):
                for z in range(z_steps):
                    sample_point = Vector((
                        bounds_min.x + x * grid_size,
                        bounds_min.y + y * grid_size,
                        bounds_min.z + z * grid_size
                    ))
                    
                    # Project onto shell
                    s_location, s_normal, s_index, s_distance = shell_bvh.find_nearest(sample_point)
                    
                    if s_location:
                        # Find corresponding point on pattern
                        p_location, p_normal, p_index, p_distance = pattern_bvh.find_nearest(s_location)
                        
                        if p_location and p_distance < 0.01:
                            candidates.append({
                                'shell_pos': s_location,
                                'pattern_pos': p_location,
                                'distance': p_distance,
                                'priority': 0.5,  # Medium priority for grid
                                'normal': s_normal
                            })
                            
        # Select connections with minimum spacing
        candidates.sort(key=lambda x: -x['priority'])  # Sort by priority
        
        for candidate in candidates:
            # Check minimum spacing
            too_close = False
            for conn in connections:
                if (conn['shell_pos'] - candidate['shell_pos']).length < min_spacing:
                    too_close = True
                    break
                    
            if not too_close:
                connections.append(candidate)
                
        self.debug.log(f"Found {len(connections)} optimal connection points")
        return connections
        
    def _generate_connections(self, 
                            connections: List[Dict], 
                            radius: float) -> List[bpy.types.Object]:
        """Generate cylindrical connection geometry"""
        connection_objects = []
        
        for i, conn in enumerate(connections):
            # Calculate connection vector
            start = conn['shell_pos']
            end = conn['pattern_pos']
            direction = end - start
            length = direction.length
            
            if length < 0.0001:  # Too short
                continue
                
            # Create cylinder
            bpy.ops.mesh.primitive_cylinder_add(
                radius=radius,
                depth=length,
                location=(start + end) * 0.5
            )
            
            cylinder = bpy.context.active_object
            cylinder.name = f"Connection_{i:03d}"
            
            # Orient cylinder
            z_axis = direction.normalized()
            up = Vector((0, 0, 1))
            if abs(z_axis.dot(up)) > 0.99:
                up = Vector((1, 0, 0))
                
            x_axis = up.cross(z_axis).normalized()
            y_axis = z_axis.cross(x_axis).normalized()
            
            # Create rotation matrix
            mat_rot = Matrix((x_axis, y_axis, z_axis)).transposed().to_4x4()
            cylinder.matrix_world = mat_rot @ cylinder.matrix_world
            
            # Add taper for better printing
            self._add_connection_taper(cylinder)
            
            connection_objects.append(cylinder)
            
        return connection_objects
        
    def _add_connection_taper(self, cylinder: bpy.types.Object):
        """Add taper to connection for better 3D printing"""
        bm = bmesh.new()
        bm.from_mesh(cylinder.data)
        bm.verts.ensure_lookup_table()
        
        # Find top and bottom vertices
        top_verts = [v for v in bm.verts if v.co.z > 0.01]
        bottom_verts = [v for v in bm.verts if v.co.z < -0.01]
        
        # Scale top vertices slightly
        for vert in top_verts:
            vert.co.x *= 1.2
            vert.co.y *= 1.2
            
        # Scale bottom vertices slightly  
        for vert in bottom_verts:
            vert.co.x *= 1.2
            vert.co.y *= 1.2
            
        bm.to_mesh(cylinder.data)
        bm.free()
        
    def _analyze_layer_separation(self,
                                base: bpy.types.Object,
                                shell: bpy.types.Object,
                                pattern: bpy.types.Object) -> Dict:
        """Analyze gaps and overlaps between layers"""
        analysis = {
            'overlaps': [],
            'gaps': [],
            'min_clearance': float('inf'),
            'max_clearance': 0,
            'avg_clearance': 0
        }
        
        # Sample points on pattern surface
        pattern_mesh = pattern.data
        shell_bvh = self.mesh_analyzer.get_bvh_tree(shell)
        
        clearances = []
        
        for vert in pattern_mesh.vertices:
            # Find distance to shell
            location, normal, index, distance = shell_bvh.find_nearest(vert.co)
            
            if location:
                clearances.append(distance)
                
                # Check for problems
                if distance < 0.0001:  # Too close (< 0.1mm)
                    analysis['overlaps'].append({
                        'position': vert.co.copy(),
                        'distance': distance
                    })
                elif distance > 0.005:  # Too far (> 5mm)
                    analysis['gaps'].append({
                        'position': vert.co.copy(),
                        'distance': distance
                    })
                    
        if clearances:
            analysis['min_clearance'] = min(clearances)
            analysis['max_clearance'] = max(clearances)
            analysis['avg_clearance'] = sum(clearances) / len(clearances)
            
        return analysis
        
    def _get_bounds(self, obj: bpy.types.Object) -> Tuple[Vector, Vector]:
        """Get bounding box of object"""
        mesh = obj.data
        if not mesh.vertices:
            return Vector(), Vector()
            
        min_co = Vector(mesh.vertices[0].co)
        max_co = Vector(mesh.vertices[0].co)
        
        for vert in mesh.vertices:
            min_co.x = min(min_co.x, vert.co.x)
            min_co.y = min(min_co.y, vert.co.y)
            min_co.z = min(min_co.z, vert.co.z)
            
            max_co.x = max(max_co.x, vert.co.x)
            max_co.y = max(max_co.y, vert.co.y)
            max_co.z = max(max_co.z, vert.co.z)
            
        return min_co, max_co
        
    def merge_for_export(self,
                        shell: bpy.types.Object,
                        pattern: bpy.types.Object,
                        connections: List[bpy.types.Object],
                        merge_connections: bool = True) -> Dict[str, bpy.types.Object]:
        """Merge geometries for dual-material export"""
        export_objects = {}
        
        # TPU Shell remains separate
        export_objects['tpu_shell'] = shell
        
        # PETG Pattern + Connections
        if merge_connections and connections:
            # Duplicate pattern
            petg_merged = pattern.copy()
            petg_merged.data = pattern.data.copy()
            petg_merged.name = "PETG_Combined"
            bpy.context.collection.objects.link(petg_merged)
            
            # Join connections
            bpy.context.view_layer.objects.active = petg_merged
            for conn in connections:
                conn.select_set(True)
            petg_merged.select_set(True)
            
            bpy.ops.object.join()
            
            export_objects['petg_pattern'] = petg_merged
        else:
            export_objects['petg_pattern'] = pattern
            export_objects['connections'] = connections
            
        return export_objects
