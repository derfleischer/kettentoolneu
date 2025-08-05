"""
Triangulation Operators for Chain Tool V4
Operators for triangulation and overlap management
"""

import bpy
import bmesh
from bpy.types import Operator
from bpy.props import FloatProperty, BoolProperty, IntProperty, EnumProperty
from mathutils import Vector
import math
from typing import List, Dict, Tuple, Set

from ..geometry.triangulation import TriangulationEngine
from ..algorithms.overlap_prevention import OverlapPrevention
from ..algorithms.gap_filling import GapFiller
from ..core.state_manager import StateManager
from ..utils.debug import DebugManager
from ..utils.performance import measure_time

class CHAIN_OT_triangulate_pattern(Operator):
    """Triangulate pattern mesh with quality constraints"""
    bl_idname = "chain.triangulate_pattern"
    bl_label = "Triangulate Pattern"
    bl_description = "Create quality triangulation of pattern"
    bl_options = {'REGISTER', 'UNDO'}
    
    method: EnumProperty(
        name="Method",
        description="Triangulation method",
        items=[
            ('DELAUNAY', "Delaunay", "Delaunay triangulation for quality triangles"),
            ('CONSTRAINED', "Constrained Delaunay", "Preserve existing edges"),
            ('ADAPTIVE', "Adaptive", "Adaptive triangulation based on curvature"),
            ('REGULAR', "Regular", "Regular triangulation pattern")
        ],
        default='DELAUNAY'
    )
    
    min_angle: FloatProperty(
        name="Minimum Angle",
        description="Minimum angle for triangles",
        default=20.0,
        min=0.0,
        max=60.0,
        subtype='ANGLE'
    )
    
    max_edge_length: FloatProperty(
        name="Max Edge Length",
        description="Maximum edge length for triangles",
        default=0.05,
        min=0.01,
        max=0.2,
        subtype='DISTANCE'
    )
    
    preserve_edges: BoolProperty(
        name="Preserve Edges",
        description="Preserve existing edges in triangulation",
        default=True
    )
    
    optimize_triangles: BoolProperty(
        name="Optimize Triangles",
        description="Optimize triangle quality after triangulation",
        default=True
    )
    
    def __init__(self):
        self.triangulation = TriangulationEngine()
        self.debug = DebugManager()
        
    @classmethod
    def poll(cls, context):
        return (context.active_object is not None and 
                context.active_object.type == 'MESH' and
                context.mode == 'OBJECT')
        
    @measure_time
    def execute(self, context):
        obj = context.active_object
        
        try:
            # Get vertices from mesh
            mesh = obj.data
            vertices = [obj.matrix_world @ v.co for v in mesh.vertices]
            
            if len(vertices) < 3:
                self.report({'ERROR'}, "Need at least 3 vertices for triangulation")
                return {'CANCELLED'}
                
            # Get constraints if preserving edges
            constraints = []
            if self.preserve_edges and mesh.edges:
                constraints = [(e.vertices[0], e.vertices[1]) for e in mesh.edges]
                
            # Perform triangulation based on method
            if self.method == 'DELAUNAY':
                result = self.triangulation.triangulate_surface(
                    vertices,
                    quality_threshold=self.min_angle,
                    max_edge_length=self.max_edge_length
                )
                
            elif self.method == 'CONSTRAINED':
                result = self.triangulation.constrained_triangulation(
                    vertices,
                    constraints,
                    quality_threshold=self.min_angle
                )
                
            elif self.method == 'ADAPTIVE':
                result = self._adaptive_triangulation(
                    obj,
                    vertices,
                    self.max_edge_length
                )
                
            else:  # REGULAR
                result = self._regular_triangulation(
                    vertices,
                    self.max_edge_length
                )
                
            if not result or not result.get('faces'):
                self.report({'ERROR'}, "Triangulation failed")
                return {'CANCELLED'}
                
            # Optimize if requested
            if self.optimize_triangles:
                result = self.triangulation.optimize_triangulation(
                    result['vertices'],
                    result['faces'],
                    min_angle=math.radians(self.min_angle)
                )
                
            # Create new mesh
            new_mesh = bpy.data.meshes.new(f"{obj.name}_Triangulated")
            
            # Add vertices
            new_mesh.vertices.add(len(result['vertices']))
            for i, vert in enumerate(result['vertices']):
                new_mesh.vertices[i].co = vert
                
            # Add faces
            new_mesh.loops.add(len(result['faces']) * 3)
            new_mesh.polygons.add(len(result['faces']))
            
            for i, face in enumerate(result['faces']):
                poly = new_mesh.polygons[i]
                poly.loop_start = i * 3
                poly.loop_total = 3
                
                for j, vert_idx in enumerate(face):
                    new_mesh.loops[i * 3 + j].vertex_index = vert_idx
                    
            new_mesh.update()
            
            # Create new object or update existing
            if "_Triangulated" in obj.name:
                # Update existing
                obj.data = new_mesh
                self.report({'INFO'}, f"Updated triangulation: {len(result['faces'])} triangles")
            else:
                # Create new
                new_obj = bpy.data.objects.new(f"{obj.name}_Triangulated", new_mesh)
                context.collection.objects.link(new_obj)
                new_obj.matrix_world = obj.matrix_world.copy()
                
                # Select new object
                bpy.ops.object.select_all(action='DESELECT')
                new_obj.select_set(True)
                context.view_layer.objects.active = new_obj
                
                self.report({'INFO'}, f"Created triangulation: {len(result['faces'])} triangles")
                
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Triangulation error: {str(e)}")
            self.debug.error(f"Triangulation error: {str(e)}")
            return {'CANCELLED'}
            
    def _adaptive_triangulation(self, obj, vertices, max_edge_length):
        """Adaptive triangulation based on surface properties"""
        # Analyze surface curvature
        mesh = obj.data
        curvatures = []
        
        for vert in mesh.vertices:
            # Simple curvature estimation
            if len(vert.link_faces) > 1:
                normals = [mesh.polygons[f].normal for f in vert.link_faces]
                avg_normal = sum(normals, Vector()) / len(normals)
                deviation = sum((n - avg_normal).length for n in normals) / len(normals)
                curvatures.append(deviation)
            else:
                curvatures.append(0.0)
                
        # Adaptive sampling based on curvature
        adaptive_points = []
        
        for i, (vert, curv) in enumerate(zip(vertices, curvatures)):
            adaptive_points.append(vert)
            
            # Add extra points in high curvature areas
            if curv > 0.1:  # Threshold
                # Add nearby points
                for j, other_vert in enumerate(vertices):
                    if i != j and (vert - other_vert).length < max_edge_length * 0.5:
                        mid_point = (vert + other_vert) / 2
                        if mid_point not in adaptive_points:
                            adaptive_points.append(mid_point)
                            
        # Triangulate adaptive points
        return self.triangulation.triangulate_surface(
            adaptive_points,
            max_edge_length=max_edge_length
        )
        
    def _regular_triangulation(self, vertices, spacing):
        """Create regular triangulation pattern"""
        # Find bounds
        min_co = Vector((
            min(v.x for v in vertices),
            min(v.y for v in vertices),
            min(v.z for v in vertices)
        ))
        max_co = Vector((
            max(v.x for v in vertices),
            max(v.y for v in vertices),
            max(v.z for v in vertices)
        ))
        
        # Generate regular grid
        grid_points = []
        
        x_steps = int((max_co.x - min_co.x) / spacing) + 1
        y_steps = int((max_co.y - min_co.y) / spacing) + 1
        
        for x in range(x_steps):
            for y in range(y_steps):
                point = Vector((
                    min_co.x + x * spacing,
                    min_co.y + y * spacing,
                    (min_co.z + max_co.z) / 2  # Average Z
                ))
                
                # Find closest original vertex for Z coordinate
                closest_z = min(vertices, key=lambda v: (v.x - point.x)**2 + (v.y - point.y)**2).z
                point.z = closest_z
                
                grid_points.append(point)
                
        # Triangulate grid
        return self.triangulation.triangulate_surface(grid_points)


class CHAIN_OT_optimize_triangulation(Operator):
    """Optimize existing triangulation for quality"""
    bl_idname = "chain.optimize_triangulation"
    bl_label = "Optimize Triangulation"
    bl_description = "Improve triangle quality in mesh"
    bl_options = {'REGISTER', 'UNDO'}
    
    optimization_type: EnumProperty(
        name="Optimization Type",
        description="Type of optimization",
        items=[
            ('ANGLES', "Improve Angles", "Improve triangle angles"),
            ('EDGE_LENGTH', "Equalize Edges", "Make edge lengths more uniform"),
            ('ASPECT_RATIO', "Aspect Ratio", "Improve triangle aspect ratios"),
            ('ALL', "All", "Apply all optimizations")
        ],
        default='ALL'
    )
    
    iterations: IntProperty(
        name="Iterations",
        description="Number of optimization passes",
        default=5,
        min=1,
        max=20
    )
    
    target_angle: FloatProperty(
        name="Target Angle",
        description="Target angle for triangles",
        default=60.0,
        min=30.0,
        max=90.0,
        subtype='ANGLE'
    )
    
    def __init__(self):
        self.triangulation = TriangulationEngine()
        
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            return False
            
        # Check if mesh has triangles
        return any(len(p.vertices) == 3 for p in obj.data.polygons)
        
    def execute(self, context):
        obj = context.active_object
        mesh = obj.data
        
        # Convert to bmesh
        bm = bmesh.new()
        bm.from_mesh(mesh)
        
        # Ensure triangulated
        bmesh.ops.triangulate(bm, faces=bm.faces[:])
        
        improved_count = 0
        
        for iteration in range(self.iterations):
            # Apply optimizations
            if self.optimization_type in ['ANGLES', 'ALL']:
                improved = self._optimize_angles(bm, math.radians(self.target_angle))
                improved_count += improved
                
            if self.optimization_type in ['EDGE_LENGTH', 'ALL']:
                improved = self._optimize_edge_lengths(bm)
                improved_count += improved
                
            if self.optimization_type in ['ASPECT_RATIO', 'ALL']:
                improved = self._optimize_aspect_ratios(bm)
                improved_count += improved
                
        # Update mesh
        bm.to_mesh(mesh)
        bm.free()
        
        self.report({'INFO'}, f"Optimization complete: {improved_count} improvements made")
        
        return {'FINISHED'}
        
    def _optimize_angles(self, bm, target_angle):
        """Optimize triangle angles by edge flipping"""
        improved = 0
        
        for edge in list(bm.edges):
            if len(edge.link_faces) != 2:
                continue
                
            # Get the two triangles
            face1, face2 = edge.link_faces
            
            if len(face1.verts) != 3 or len(face2.verts) != 3:
                continue
                
            # Check if flipping would improve angles
            if self._should_flip_edge(edge, target_angle):
                # Flip edge
                try:
                    bmesh.ops.edge_rotate(bm, edges=[edge])
                    improved += 1
                except:
                    pass  # Skip if flip would create invalid geometry
                    
        return improved
        
    def _should_flip_edge(self, edge, target_angle):
        """Check if flipping edge would improve triangle quality"""
        face1, face2 = edge.link_faces
        
        # Get vertices of quadrilateral
        verts = list(set(face1.verts + face2.verts))
        
        if len(verts) != 4:
            return False
            
        # Calculate current min angle
        current_min_angle = float('inf')
        
        for face in [face1, face2]:
            for i in range(3):
                v0 = face.verts[i].co
                v1 = face.verts[(i + 1) % 3].co
                v2 = face.verts[(i + 2) % 3].co
                
                edge1 = (v1 - v0).normalized()
                edge2 = (v2 - v1).normalized()
                
                angle = math.acos(max(-1, min(1, -edge1.dot(edge2))))
                current_min_angle = min(current_min_angle, angle)
                
        # Calculate min angle after flip
        # Find opposite vertices
        opp_verts = [v for v in verts if v not in edge.verts]
        
        if len(opp_verts) != 2:
            return False
            
        # Check new triangles
        new_min_angle = float('inf')
        
        # New triangle 1: edge.verts[0], opp_verts[0], opp_verts[1]
        # New triangle 2: edge.verts[1], opp_verts[0], opp_verts[1]
        
        for tri_verts in [
            [edge.verts[0], opp_verts[0], opp_verts[1]],
            [edge.verts[1], opp_verts[0], opp_verts[1]]
        ]:
            for i in range(3):
                v0 = tri_verts[i].co
                v1 = tri_verts[(i + 1) % 3].co
                v2 = tri_verts[(i + 2) % 3].co
                
                edge1 = (v1 - v0).normalized()
                edge2 = (v2 - v1).normalized()
                
                angle = math.acos(max(-1, min(1, -edge1.dot(edge2))))
                new_min_angle = min(new_min_angle, angle)
                
        # Flip if new configuration is better
        return new_min_angle > current_min_angle
        
    def _optimize_edge_lengths(self, bm):
        """Optimize edge length uniformity"""
        improved = 0
        
        # Calculate average edge length
        avg_length = sum(e.calc_length() for e in bm.edges) / len(bm.edges)
        
        # Smooth vertex positions
        for vert in bm.verts:
            if len(vert.link_edges) < 3:
                continue
                
            # Calculate average neighbor position
            neighbor_sum = Vector()
            for edge in vert.link_edges:
                other_vert = edge.other_vert(vert)
                neighbor_sum += other_vert.co
                
            avg_pos = neighbor_sum / len(vert.link_edges)
            
            # Move vertex toward average
            move_vec = avg_pos - vert.co
            
            # Check if movement improves edge length uniformity
            current_deviation = sum(abs(e.calc_length() - avg_length) for e in vert.link_edges)
            
            # Test new position
            old_co = vert.co.copy()
            vert.co = vert.co + move_vec * 0.5
            
            new_deviation = sum(abs(e.calc_length() - avg_length) for e in vert.link_edges)
            
            if new_deviation < current_deviation:
                improved += 1
            else:
                vert.co = old_co  # Restore
                
        return improved
        
    def _optimize_aspect_ratios(self, bm):
        """Optimize triangle aspect ratios"""
        improved = 0
        
        for face in bm.faces:
            if len(face.verts) != 3:
                continue
                
            # Calculate aspect ratio
            edges = [face.edges[i].calc_length() for i in range(3)]
            max_edge = max(edges)
            min_edge = min(edges)
            
            aspect_ratio = max_edge / min_edge if min_edge > 0 else float('inf')
            
            # Try to improve if aspect ratio is bad
            if aspect_ratio > 2.0:
                # Find longest edge
                longest_edge = max(face.edges, key=lambda e: e.calc_length())
                
                # Try to split or flip
                if len(longest_edge.link_faces) == 2:
                    # Try flipping
                    if self._should_flip_edge(longest_edge, math.radians(60)):
                        try:
                            bmesh.ops.edge_rotate(bm, edges=[longest_edge])
                            improved += 1
                        except:
                            pass
                            
        return improved


class CHAIN_OT_check_overlaps(Operator):
    """Check for overlapping elements in pattern"""
    bl_idname = "chain.check_overlaps"
    bl_label = "Check Overlaps"
    bl_description = "Detect overlapping elements"
    bl_options = {'REGISTER'}
    
    check_type: EnumProperty(
        name="Check Type",
        description="Type of overlap to check",
        items=[
            ('FACES', "Face Overlaps", "Check for overlapping faces"),
            ('EDGES', "Edge Intersections", "Check for intersecting edges"),
            ('CONNECTIONS', "Connection Overlaps", "Check connection overlaps"),
            ('ALL', "All", "Check all types")
        ],
        default='ALL'
    )
    
    tolerance: FloatProperty(
        name="Tolerance",
        description="Overlap detection tolerance",
        default=0.001,
        min=0.0,
        max=0.01,
        subtype='DISTANCE'
    )
    
    mark_overlaps: BoolProperty(
        name="Mark Overlaps",
        description="Mark overlapping elements",
        default=True
    )
    
    def __init__(self):
        self.overlap_prevention = OverlapPrevention()
        
    @classmethod
    def poll(cls, context):
        return (context.active_object is not None and 
                context.active_object.type == 'MESH')
        
    def execute(self, context):
        obj = context.active_object
        
        overlaps = {
            'faces': [],
            'edges': [],
            'connections': []
        }
        
        # Check face overlaps
        if self.check_type in ['FACES', 'ALL']:
            overlaps['faces'] = self._check_face_overlaps(obj, self.tolerance)
            
        # Check edge intersections
        if self.check_type in ['EDGES', 'ALL']:
            overlaps['edges'] = self._check_edge_intersections(obj, self.tolerance)
            
        # Check connection overlaps
        if self.check_type in ['CONNECTIONS', 'ALL']:
            overlaps['connections'] = self._check_connection_overlaps(obj, self.tolerance)
            
        # Report results
        total_overlaps = (len(overlaps['faces']) + 
                         len(overlaps['edges']) + 
                         len(overlaps['connections']))
                         
        if total_overlaps == 0:
            self.report({'INFO'}, "No overlaps detected")
        else:
            self.report({'WARNING'}, 
                       f"Found {total_overlaps} overlaps: "
                       f"{len(overlaps['faces'])} faces, "
                       f"{len(overlaps['edges'])} edges, "
                       f"{len(overlaps['connections'])} connections")
                       
            # Mark overlaps if requested
            if self.mark_overlaps:
                self._mark_overlapping_elements(obj, overlaps)
                
        # Store overlap data
        obj["overlap_check"] = overlaps
        
        return {'FINISHED'}
        
    def _check_face_overlaps(self, obj, tolerance):
        """Check for overlapping faces"""
        mesh = obj.data
        overlapping_faces = []
        
        # Build BVH tree
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.transform(obj.matrix_world)
        
        tree = self.overlap_prevention._build_bvh_tree(
            [v.co for v in bm.verts],
            [[v.index for v in f.verts] for f in bm.faces]
        )
        
        # Check each face against others
        for i, face1 in enumerate(bm.faces):
            # Get face bounds
            min_co = Vector((
                min(v.co.x for v in face1.verts),
                min(v.co.y for v in face1.verts),
                min(v.co.z for v in face1.verts)
            ))
            max_co = Vector((
                max(v.co.x for v in face1.verts),
                max(v.co.y for v in face1.verts),
                max(v.co.z for v in face1.verts)
            ))
            
            # Find potential overlaps
            overlaps = tree.find_range(face1.calc_center_median(), face1.calc_area())
            
            for location, index, dist in overlaps:
                if index != i and index not in [f[1] for f in overlapping_faces if f[0] == i]:
                    # Check if faces actually overlap
                    face2 = bm.faces[index]
                    
                    if self._faces_overlap(face1, face2, tolerance):
                        overlapping_faces.append((i, index))
                        
        bm.free()
        return overlapping_faces
        
    def _check_edge_intersections(self, obj, tolerance):
        """Check for intersecting edges"""
        mesh = obj.data
        intersecting_edges = []
        
        # Check each edge pair
        for i, edge1 in enumerate(mesh.edges):
            v1_start = obj.matrix_world @ mesh.vertices[edge1.vertices[0]].co
            v1_end = obj.matrix_world @ mesh.vertices[edge1.vertices[1]].co
            
            for j, edge2 in enumerate(mesh.edges[i + 1:], i + 1):
                v2_start = obj.matrix_world @ mesh.vertices[edge2.vertices[0]].co
                v2_end = obj.matrix_world @ mesh.vertices[edge2.vertices[1]].co
                
                # Check if edges share a vertex
                if (edge1.vertices[0] in edge2.vertices or 
                    edge1.vertices[1] in edge2.vertices):
                    continue
                    
                # Check intersection
                if self._edges_intersect(v1_start, v1_end, v2_start, v2_end, tolerance):
                    intersecting_edges.append((i, j))
                    
        return intersecting_edges
        
    def _check_connection_overlaps(self, obj, tolerance):
        """Check for overlapping connections"""
        # Look for connection objects
        connections = []
        
        for child in obj.children:
            if "connection" in child.name.lower():
                connections.append(child)
                
        overlapping_connections = []
        
        # Check each connection pair
        for i, conn1 in enumerate(connections):
            for j, conn2 in enumerate(connections[i + 1:], i + 1):
                if self._objects_overlap(conn1, conn2, tolerance):
                    overlapping_connections.append((conn1.name, conn2.name))
                    
        return overlapping_connections
        
    def _faces_overlap(self, face1, face2, tolerance):
        """Check if two faces overlap"""
        # Simple check: if face centers are very close
        dist = (face1.calc_center_median() - face2.calc_center_median()).length
        
        if dist < tolerance:
            return True
            
        # Check if faces are coplanar and overlapping
        # This is simplified - in production use proper geometric tests
        normal1 = face1.normal
        normal2 = face2.normal
        
        # Check if parallel
        if abs(normal1.dot(normal2)) > 0.99:
            # Check if coplanar
            plane_dist = abs((face1.verts[0].co - face2.verts[0].co).dot(normal1))
            
            if plane_dist < tolerance:
                # Faces are coplanar, check for overlap
                # Simplified: check if any vertex of one face is inside the other
                for vert in face1.verts:
                    if self._point_in_face(vert.co, face2):
                        return True
                        
        return False
        
    def _edges_intersect(self, p1, p2, p3, p4, tolerance):
        """Check if two line segments intersect"""
        # Based on orientation test
        d1 = (p4 - p3).cross(p1 - p3)
        d2 = (p4 - p3).cross(p2 - p3)
        d3 = (p2 - p1).cross(p3 - p1)
        d4 = (p2 - p1).cross(p4 - p1)
        
        # Check if segments intersect
        if d1.dot(d2) < 0 and d3.dot(d4) < 0:
            # Calculate intersection point
            t = d3.length / (d3.length + d4.length)
            intersection = p1 + (p2 - p1) * t
            
            # Check distance to both segments
            dist1 = self._point_to_segment_distance(intersection, p1, p2)
            dist2 = self._point_to_segment_distance(intersection, p3, p4)
            
            return dist1 < tolerance and dist2 < tolerance
            
        return False
        
    def _point_to_segment_distance(self, point, seg_start, seg_end):
        """Calculate distance from point to line segment"""
        segment = seg_end - seg_start
        t = max(0, min(1, (point - seg_start).dot(segment) / segment.length_squared))
        projection = seg_start + segment * t
        return (point - projection).length
        
    def _point_in_face(self, point, face):
        """Check if point is inside face (simplified)"""
        # Project to 2D and use winding number test
        # This is simplified - in production use proper point-in-polygon test
        center = face.calc_center_median()
        return (point - center).length < face.calc_area() ** 0.5
        
    def _objects_overlap(self, obj1, obj2, tolerance):
        """Check if two objects overlap"""
        # Simple bounding box check
        bbox1_min = Vector((min(v[0] for v in obj1.bound_box),
                           min(v[1] for v in obj1.bound_box),
                           min(v[2] for v in obj1.bound_box)))
        bbox1_max = Vector((max(v[0] for v in obj1.bound_box),
                           max(v[1] for v in obj1.bound_box),
                           max(v[2] for v in obj1.bound_box)))
                           
        bbox2_min = Vector((min(v[0] for v in obj2.bound_box),
                           min(v[1] for v in obj2.bound_box),
                           min(v[2] for v in obj2.bound_box)))
        bbox2_max = Vector((max(v[0] for v in obj2.bound_box),
                           max(v[1] for v in obj2.bound_box),
                           max(v[2] for v in obj2.bound_box)))
                           
        # Transform to world space
        bbox1_min = obj1.matrix_world @ bbox1_min
        bbox1_max = obj1.matrix_world @ bbox1_max
        bbox2_min = obj2.matrix_world @ bbox2_min
        bbox2_max = obj2.matrix_world @ bbox2_max
        
        # Check overlap
        return (bbox1_max.x >= bbox2_min.x - tolerance and 
                bbox2_max.x >= bbox1_min.x - tolerance and
                bbox1_max.y >= bbox2_min.y - tolerance and 
                bbox2_max.y >= bbox1_min.y - tolerance and
                bbox1_max.z >= bbox2_min.z - tolerance and 
                bbox2_max.z >= bbox1_min.z - tolerance)
                
    def _mark_overlapping_elements(self, obj, overlaps):
        """Mark overlapping elements for visualization"""
        mesh = obj.data
        
        # Create vertex groups for marking
        if "Overlapping_Faces" not in obj.vertex_groups:
            obj.vertex_groups.new(name="Overlapping_Faces")
        if "Overlapping_Edges" not in obj.vertex_groups:
            obj.vertex_groups.new(name="Overlapping_Edges")
            
        face_group = obj.vertex_groups["Overlapping_Faces"]
        edge_group = obj.vertex_groups["Overlapping_Edges"]
        
        # Mark overlapping faces
        for face_pair in overlaps['faces']:
            for face_idx in face_pair:
                if face_idx < len(mesh.polygons):
                    face = mesh.polygons[face_idx]
                    for vert_idx in face.vertices:
                        face_group.add([vert_idx], 1.0, 'REPLACE')
                        
        # Mark intersecting edges
        for edge_pair in overlaps['edges']:
            for edge_idx in edge_pair:
                if edge_idx < len(mesh.edges):
                    edge = mesh.edges[edge_idx]
                    for vert_idx in edge.vertices:
                        edge_group.add([vert_idx], 1.0, 'REPLACE')


class CHAIN_OT_fix_overlaps(Operator):
    """Fix detected overlaps in pattern"""
    bl_idname = "chain.fix_overlaps"
    bl_label = "Fix Overlaps"
    bl_description = "Attempt to fix overlapping elements"
    bl_options = {'REGISTER', 'UNDO'}
    
    fix_method: EnumProperty(
        name="Fix Method",
        description="Method to fix overlaps",
        items=[
            ('REMOVE', "Remove Overlapping", "Remove overlapping elements"),
            ('MERGE', "Merge Close", "Merge very close elements"),
            ('OFFSET', "Offset Apart", "Move overlapping elements apart"),
            ('REBUILD', "Rebuild Area", "Rebuild overlapping areas")
        ],
        default='MERGE'
    )
    
    merge_distance: FloatProperty(
        name="Merge Distance",
        description="Distance for merging close elements",
        default=0.002,
        min=0.0001,
        max=0.01,
        subtype='DISTANCE'
    )
    
    offset_distance: FloatProperty(
        name="Offset Distance",
        description="Distance to offset overlapping elements",
        default=0.005,
        min=0.001,
        max=0.02,
        subtype='DISTANCE'
    )
    
    def __init__(self):
        self.overlap_prevention = OverlapPrevention()
        self.gap_filler = GapFiller()
        
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (obj and obj.type == 'MESH' and 
                obj.get("overlap_check"))
                
    def execute(self, context):
        obj = context.active_object
        overlaps = obj.get("overlap_check", {})
        
        if not any(overlaps.values()):
            self.report({'INFO'}, "No overlaps to fix")
            return {'FINISHED'}
            
        fixed_count = 0
        
        # Fix based on method
        if self.fix_method == 'REMOVE':
            fixed_count = self._remove_overlapping(obj, overlaps)
            
        elif self.fix_method == 'MERGE':
            fixed_count = self._merge_close_elements(obj, overlaps, self.merge_distance)
            
        elif self.fix_method == 'OFFSET':
            fixed_count = self._offset_overlapping(obj, overlaps, self.offset_distance)
            
        elif self.fix_method == 'REBUILD':
            fixed_count = self._rebuild_overlapping_areas(obj, overlaps)
            
        # Clear overlap data
        del obj["overlap_check"]
        
        self.report({'INFO'}, f"Fixed {fixed_count} overlapping elements")
        
        return {'FINISHED'}
        
    def _remove_overlapping(self, obj, overlaps):
        """Remove overlapping elements"""
        mesh = obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        
        fixed = 0
        
        # Remove overlapping faces
        faces_to_remove = set()
        for face_pair in overlaps.get('faces', []):
            # Remove the smaller face
            if face_pair[0] < len(bm.faces) and face_pair[1] < len(bm.faces):
                face1 = bm.faces[face_pair[0]]
                face2 = bm.faces[face_pair[1]]
                
                if face1.calc_area() < face2.calc_area():
                    faces_to_remove.add(face1)
                else:
                    faces_to_remove.add(face2)
                    
        bmesh.ops.delete(bm, geom=list(faces_to_remove), context='FACES')
        fixed += len(faces_to_remove)
        
        # Remove intersecting edges
        edges_to_remove = set()
        for edge_pair in overlaps.get('edges', []):
            # Remove the shorter edge
            if edge_pair[0] < len(bm.edges) and edge_pair[1] < len(bm.edges):
                edge1 = bm.edges[edge_pair[0]]
                edge2 = bm.edges[edge_pair[1]]
                
                if edge1.calc_length() < edge2.calc_length():
                    edges_to_remove.add(edge1)
                else:
                    edges_to_remove.add(edge2)
                    
        bmesh.ops.delete(bm, geom=list(edges_to_remove), context='EDGES')
        fixed += len(edges_to_remove)
        
        bm.to_mesh(mesh)
        bm.free()
        
        return fixed
        
    def _merge_close_elements(self, obj, overlaps, merge_distance):
        """Merge very close elements"""
        mesh = obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        
        # Merge close vertices
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=merge_distance)
        
        # Count merged elements
        fixed = len(mesh.vertices) - len(bm.verts)
        
        bm.to_mesh(mesh)
        bm.free()
        
        return fixed
        
    def _offset_overlapping(self, obj, overlaps, offset_distance):
        """Move overlapping elements apart"""
        mesh = obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        
        fixed = 0
        
        # Offset overlapping faces
        for face_pair in overlaps.get('faces', []):
            if face_pair[0] < len(bm.faces) and face_pair[1] < len(bm.faces):
                face1 = bm.faces[face_pair[0]]
                face2 = bm.faces[face_pair[1]]
                
                # Calculate offset direction
                offset_dir = (face1.calc_center_median() - face2.calc_center_median()).normalized()
                
                # Offset face1 vertices
                for vert in face1.verts:
                    vert.co += offset_dir * offset_distance
                    
                fixed += 1
                
        bm.to_mesh(mesh)
        bm.free()
        
        return fixed
        
    def _rebuild_overlapping_areas(self, obj, overlaps):
        """Rebuild areas with overlaps"""
        mesh = obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        
        # Remove overlapping faces
        faces_to_remove = set()
        affected_verts = set()
        
        for face_pair in overlaps.get('faces', []):
            for face_idx in face_pair:
                if face_idx < len(bm.faces):
                    face = bm.faces[face_idx]
                    faces_to_remove.add(face)
                    affected_verts.update(face.verts)
                    
        bmesh.ops.delete(bm, geom=list(faces_to_remove), context='FACES_ONLY')
        
        # Re-triangulate affected areas
        if affected_verts:
            # Get boundary edges
            boundary_edges = [e for e in bm.edges 
                            if len(e.link_faces) == 1 and 
                            any(v in affected_verts for v in e.verts)]
                            
            # Fill holes
            bmesh.ops.triangle_fill(bm, use_beauty=True, use_dissolve=True,
                                   edges=boundary_edges)
                                   
        fixed = len(faces_to_remove)
        
        bm.to_mesh(mesh)
        bm.free()
        
        return fixed


# Register all operators
classes = [
    CHAIN_OT_triangulate_pattern,
    CHAIN_OT_optimize_triangulation,
    CHAIN_OT_check_overlaps,
    CHAIN_OT_fix_overlaps
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
        
def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
