"""
Connection Tools Operators for Chain Tool V4
Tools for creating and optimizing connections between pattern elements
"""

import bpy
import bmesh
from bpy.types import Operator
from bpy.props import FloatProperty, BoolProperty, IntProperty, EnumProperty
from mathutils import Vector, kdtree
import math
from typing import List, Dict, Tuple, Set

from ..algorithms.connection_optimizer import ConnectionOptimizer
from ..algorithms.overlap_prevention import OverlapPrevention
from ..core.state_manager import StateManager
from ..utils.debug import DebugManager
from ..utils.performance import measure_time

class CHAIN_OT_auto_connect(Operator):
    """Automatically connect pattern elements"""
    bl_idname = "chain.auto_connect"
    bl_label = "Auto Connect"
    bl_description = "Automatically create connections between pattern elements"
    bl_options = {'REGISTER', 'UNDO'}
    
    connection_distance: FloatProperty(
        name="Connection Distance",
        description="Maximum distance for connections",
        default=0.03,
        min=0.005,
        max=0.1,
        subtype='DISTANCE'
    )
    
    connection_radius: FloatProperty(
        name="Connection Radius",
        description="Radius of connection beams",
        default=0.002,
        min=0.001,
        max=0.01,
        subtype='DISTANCE'
    )
    
    min_angle: FloatProperty(
        name="Minimum Angle",
        description="Minimum angle between connections",
        default=30.0,
        min=0.0,
        max=90.0,
        subtype='ANGLE'
    )
    
    max_connections: IntProperty(
        name="Max Connections",
        description="Maximum connections per element",
        default=6,
        min=1,
        max=12
    )
    
    connect_mode: EnumProperty(
        name="Connect Mode",
        description="Connection strategy",
        items=[
            ('NEAREST', "Nearest Neighbors", "Connect to nearest elements"),
            ('DELAUNAY', "Delaunay", "Use Delaunay triangulation"),
            ('MST', "Minimum Spanning Tree", "Create minimal connections"),
            ('STRESS', "Stress-Based", "Connect based on stress patterns")
        ],
        default='NEAREST'
    )
    
    prevent_overlaps: BoolProperty(
        name="Prevent Overlaps",
        description="Prevent connection overlaps",
        default=True
    )
    
    optimize_junctions: BoolProperty(
        name="Optimize Junctions",
        description="Optimize connection junction points",
        default=True
    )
    
    def __init__(self):
        self.conn_optimizer = ConnectionOptimizer()
        self.overlap_prevention = OverlapPrevention()
        self.state = StateManager()
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
            # Get connection points from mesh
            connection_points = self._extract_connection_points(obj)
            
            if len(connection_points) < 2:
                self.report({'WARNING'}, "Not enough points for connections")
                return {'CANCELLED'}
                
            self.debug.log(f"Found {len(connection_points)} connection points")
            
            # Generate connections based on mode
            if self.connect_mode == 'NEAREST':
                connections = self._nearest_neighbor_connections(
                    connection_points,
                    self.connection_distance,
                    self.max_connections,
                    math.radians(self.min_angle)
                )
                
            elif self.connect_mode == 'DELAUNAY':
                connections = self._delaunay_connections(
                    connection_points,
                    self.connection_distance
                )
                
            elif self.connect_mode == 'MST':
                connections = self._mst_connections(
                    connection_points,
                    self.connection_distance
                )
                
            elif self.connect_mode == 'STRESS':
                connections = self._stress_based_connections(
                    connection_points,
                    obj,
                    self.connection_distance
                )
                
            else:
                connections = []
                
            if not connections:
                self.report({'WARNING'}, "No valid connections found")
                return {'CANCELLED'}
                
            self.debug.log(f"Generated {len(connections)} initial connections")
            
            # Prevent overlaps if requested
            if self.prevent_overlaps:
                connections = self.overlap_prevention.remove_overlapping_connections(
                    connections,
                    self.connection_radius * 2
                )
                self.debug.log(f"After overlap removal: {len(connections)} connections")
                
            # Optimize junctions if requested
            if self.optimize_junctions:
                connections = self.conn_optimizer.optimize_junction_positions(connections)
                
            # Create connection geometry
            connection_mesh = self._create_connection_mesh(
                connections,
                self.connection_radius
            )
            
            if not connection_mesh:
                self.report({'ERROR'}, "Failed to create connection geometry")
                return {'CANCELLED'}
                
            # Create connection object
            conn_obj = bpy.data.objects.new(f"{obj.name}_Connections", connection_mesh)
            context.collection.objects.link(conn_obj)
            
            # Copy transform
            conn_obj.matrix_world = obj.matrix_world.copy()
            
            # Store reference
            obj["connections"] = conn_obj.name
            
            # Select connection object
            bpy.ops.object.select_all(action='DESELECT')
            conn_obj.select_set(True)
            context.view_layer.objects.active = conn_obj
            
            self.report({'INFO'}, f"Created {len(connections)} connections")
            
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Error creating connections: {str(e)}")
            self.debug.error(f"Connection error: {str(e)}")
            return {'CANCELLED'}
            
    def _extract_connection_points(self, obj: bpy.types.Object) -> List[Dict]:
        """Extract potential connection points from mesh"""
        points = []
        mesh = obj.data
        
        # Use vertex positions as connection points
        # In production, could use face centers, edge midpoints, etc.
        for vert in mesh.vertices:
            world_pos = obj.matrix_world @ vert.co
            
            points.append({
                'position': world_pos,
                'normal': (obj.matrix_world.to_3x3() @ vert.normal).normalized(),
                'index': vert.index,
                'connections': []  # Track connections for each point
            })
            
        return points
        
    def _nearest_neighbor_connections(self,
                                    points: List[Dict],
                                    max_dist: float,
                                    max_conn: int,
                                    min_angle: float) -> List[Tuple[Vector, Vector]]:
        """Create connections to nearest neighbors"""
        connections = []
        
        # Build KD tree
        kd = kdtree.KDTree(len(points))
        for i, point in enumerate(points):
            kd.insert(point['position'], i)
        kd.balance()
        
        # Find connections for each point
        for i, point in enumerate(points):
            # Find nearby points
            neighbors = []
            for co, idx, dist in kd.find_range(point['position'], max_dist):
                if idx != i and dist > 0:
                    neighbors.append((idx, dist))
                    
            # Sort by distance
            neighbors.sort(key=lambda x: x[1])
            
            # Add connections up to max_conn
            conn_count = 0
            
            for neighbor_idx, dist in neighbors:
                if conn_count >= max_conn:
                    break
                    
                neighbor = points[neighbor_idx]
                
                # Check if connection already exists (from other point)
                conn_key = tuple(sorted([i, neighbor_idx]))
                exists = any(
                    tuple(sorted([points.index(p) for p in [
                        next(p for p in points if p['position'] == c[0]),
                        next(p for p in points if p['position'] == c[1])
                    ]])) == conn_key
                    for c in connections
                )
                
                if not exists:
                    # Check angle constraints
                    valid_angle = True
                    
                    if point['connections']:
                        # Check angles with existing connections
                        new_dir = (neighbor['position'] - point['position']).normalized()
                        
                        for existing_conn in point['connections']:
                            existing_dir = (existing_conn - point['position']).normalized()
                            angle = math.acos(max(-1, min(1, new_dir.dot(existing_dir))))
                            
                            if angle < min_angle:
                                valid_angle = False
                                break
                                
                    if valid_angle:
                        connections.append((point['position'], neighbor['position']))
                        point['connections'].append(neighbor['position'])
                        neighbor['connections'].append(point['position'])
                        conn_count += 1
                        
        return connections
        
    def _delaunay_connections(self,
                            points: List[Dict],
                            max_dist: float) -> List[Tuple[Vector, Vector]]:
        """Create connections using Delaunay triangulation"""
        from scipy.spatial import Delaunay
        import numpy as np
        
        # Convert to numpy array
        positions = np.array([list(p['position']) for p in points])
        
        # Compute Delaunay triangulation
        tri = Delaunay(positions)
        
        connections = []
        connection_set = set()
        
        # Extract edges from triangulation
        for simplex in tri.simplices:
            # Each simplex is a tetrahedron (4 vertices)
            for i in range(4):
                for j in range(i + 1, 4):
                    v1, v2 = simplex[i], simplex[j]
                    
                    # Check distance
                    dist = (points[v1]['position'] - points[v2]['position']).length
                    
                    if dist <= max_dist:
                        edge_key = tuple(sorted([v1, v2]))
                        
                        if edge_key not in connection_set:
                            connection_set.add(edge_key)
                            connections.append((
                                points[v1]['position'],
                                points[v2]['position']
                            ))
                            
        return connections
        
    def _mst_connections(self,
                       points: List[Dict],
                       max_dist: float) -> List[Tuple[Vector, Vector]]:
        """Create minimum spanning tree connections"""
        import numpy as np
        
        n = len(points)
        if n < 2:
            return []
            
        # Build distance matrix
        distances = np.full((n, n), np.inf)
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = (points[i]['position'] - points[j]['position']).length
                if dist <= max_dist:
                    distances[i, j] = dist
                    distances[j, i] = dist
                    
        # Prim's algorithm for MST
        connections = []
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
                connections.append((
                    points[min_i]['position'],
                    points[min_j]['position']
                ))
            else:
                # No more connections possible
                break
                
        return connections
        
    def _stress_based_connections(self,
                                points: List[Dict],
                                obj: bpy.types.Object,
                                max_dist: float) -> List[Tuple[Vector, Vector]]:
        """Create connections based on stress patterns"""
        connections = []
        
        # Simple stress heuristic: prefer vertical connections (gravity)
        # In production, use actual FEA or stress analysis
        
        for i, point1 in enumerate(points):
            for j, point2 in enumerate(points[i + 1:], i + 1):
                dist = (point1['position'] - point2['position']).length
                
                if dist <= max_dist:
                    # Calculate stress factor based on direction
                    direction = (point2['position'] - point1['position']).normalized()
                    
                    # Vertical alignment factor (1 = vertical, 0 = horizontal)
                    vertical_factor = abs(direction.dot(Vector((0, 0, 1))))
                    
                    # Height factor (lower points have higher stress)
                    z_avg = (point1['position'].z + point2['position'].z) / 2
                    height_factor = 1.0 - (z_avg - obj.location.z) / (obj.dimensions.z + 0.001)
                    
                    # Combined stress factor
                    stress_factor = vertical_factor * height_factor
                    
                    # Add connection if stress factor is high enough
                    if stress_factor > 0.3:
                        connections.append((point1['position'], point2['position']))
                        
        return connections
        
    def _create_connection_mesh(self,
                              connections: List[Tuple[Vector, Vector]],
                              radius: float) -> bpy.types.Mesh:
        """Create mesh from connections"""
        if not connections:
            return None
            
        # Create mesh
        mesh = bpy.data.meshes.new("Connections")
        
        vertices = []
        edges = []
        faces = []
        
        # Create cylindrical beams for each connection
        for conn_idx, (start, end) in enumerate(connections):
            # Calculate beam orientation
            direction = (end - start).normalized()
            length = (end - start).length
            
            # Find perpendicular vectors
            up = Vector((0, 0, 1))
            if abs(direction.dot(up)) > 0.99:
                up = Vector((1, 0, 0))
                
            right = direction.cross(up).normalized()
            up = right.cross(direction).normalized()
            
            # Create cylinder vertices (8 per connection)
            base_idx = len(vertices)
            segments = 8
            
            for i in range(segments):
                angle = i * 2 * math.pi / segments
                offset = right * (radius * math.cos(angle)) + up * (radius * math.sin(angle))
                
                # Start cap
                vertices.append(start + offset)
                # End cap
                vertices.append(end + offset)
                
            # Create faces
            for i in range(segments):
                next_i = (i + 1) % segments
                
                # Side face
                v0 = base_idx + i * 2
                v1 = base_idx + i * 2 + 1
                v2 = base_idx + next_i * 2 + 1
                v3 = base_idx + next_i * 2
                
                faces.append([v0, v3, v2, v1])
                
            # End caps
            cap_start = []
            cap_end = []
            for i in range(segments):
                cap_start.append(base_idx + i * 2)
                cap_end.append(base_idx + i * 2 + 1)
                
            faces.append(cap_start)
            faces.append(list(reversed(cap_end)))
            
        # Create mesh
        mesh.from_pydata(vertices, edges, faces)
        mesh.update()
        
        return mesh


class CHAIN_OT_connect_selected(Operator):
    """Connect selected elements"""
    bl_idname = "chain.connect_selected"
    bl_label = "Connect Selected"
    bl_description = "Create connections between selected elements"
    bl_options = {'REGISTER', 'UNDO'}
    
    connection_radius: FloatProperty(
        name="Connection Radius",
        description="Radius of connection beams",
        default=0.002,
        min=0.001,
        max=0.01,
        subtype='DISTANCE'
    )
    
    @classmethod
    def poll(cls, context):
        return (context.active_object is not None and 
                context.active_object.type == 'MESH' and
                context.mode == 'EDIT_MESH')
        
    def execute(self, context):
        obj = context.active_object
        
        # Get selected vertices
        bm = bmesh.from_edit_mesh(obj.data)
        selected_verts = [v for v in bm.verts if v.select]
        
        if len(selected_verts) < 2:
            self.report({'WARNING'}, "Select at least 2 vertices")
            return {'CANCELLED'}
            
        # Create connections between all selected vertices
        connections = []
        
        for i, v1 in enumerate(selected_verts):
            for v2 in selected_verts[i + 1:]:
                world_v1 = obj.matrix_world @ v1.co
                world_v2 = obj.matrix_world @ v2.co
                connections.append((world_v1, world_v2))
                
        # Switch to object mode
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # Create connection mesh
        conn_mesh = self._create_connection_mesh(connections, self.connection_radius)
        
        if conn_mesh:
            # Create object
            conn_obj = bpy.data.objects.new(f"{obj.name}_SelectedConnections", conn_mesh)
            context.collection.objects.link(conn_obj)
            
            # Select new object
            bpy.ops.object.select_all(action='DESELECT')
            conn_obj.select_set(True)
            context.view_layer.objects.active = conn_obj
            
            self.report({'INFO'}, f"Created {len(connections)} connections")
        else:
            self.report({'ERROR'}, "Failed to create connections")
            return {'CANCELLED'}
            
        return {'FINISHED'}
        
    def _create_connection_mesh(self, connections, radius):
        """Reuse connection mesh creation from auto connect"""
        auto_conn = CHAIN_OT_auto_connect()
        return auto_conn._create_connection_mesh(connections, radius)


class CHAIN_OT_optimize_connections(Operator):
    """Optimize existing connections"""
    bl_idname = "chain.optimize_connections"
    bl_label = "Optimize Connections"
    bl_description = "Optimize connection paths and junctions"
    bl_options = {'REGISTER', 'UNDO'}
    
    optimization_type: EnumProperty(
        name="Optimization Type",
        description="Type of optimization",
        items=[
            ('LENGTH', "Minimize Length", "Minimize total connection length"),
            ('STRESS', "Stress Distribution", "Optimize for stress distribution"),
            ('JUNCTIONS', "Junction Points", "Optimize junction positions"),
            ('ALL', "All", "Apply all optimizations")
        ],
        default='ALL'
    )
    
    iterations: IntProperty(
        name="Iterations",
        description="Number of optimization iterations",
        default=10,
        min=1,
        max=100
    )
    
    def __init__(self):
        self.conn_optimizer = ConnectionOptimizer()
        
    @classmethod
    def poll(cls, context):
        return (context.active_object is not None and 
                context.active_object.type == 'MESH')
        
    def execute(self, context):
        obj = context.active_object
        
        # Extract connections from mesh
        connections = self._extract_connections_from_mesh(obj)
        
        if not connections:
            self.report({'WARNING'}, "No connections found in mesh")
            return {'CANCELLED'}
            
        original_count = len(connections)
        
        # Apply optimizations
        if self.optimization_type in ['LENGTH', 'ALL']:
            connections = self.conn_optimizer.minimize_total_length(
                connections,
                iterations=self.iterations
            )
            
        if self.optimization_type in ['STRESS', 'ALL']:
            connections = self.conn_optimizer.optimize_for_stress(
                connections,
                stress_direction=Vector((0, 0, -1)),  # Gravity
                iterations=self.iterations
            )
            
        if self.optimization_type in ['JUNCTIONS', 'ALL']:
            connections = self.conn_optimizer.optimize_junction_positions(
                connections,
                iterations=self.iterations
            )
            
        # Rebuild mesh with optimized connections
        new_mesh = CHAIN_OT_auto_connect()._create_connection_mesh(
            connections,
            0.002  # Default radius
        )
        
        if new_mesh:
            # Replace mesh data
            obj.data = new_mesh
            
            self.report({'INFO'}, 
                       f"Optimized {original_count} connections "
                       f"({len(connections)} after optimization)")
        else:
            self.report({'ERROR'}, "Failed to create optimized mesh")
            return {'CANCELLED'}
            
        return {'FINISHED'}
        
    def _extract_connections_from_mesh(self, obj: bpy.types.Object) -> List[Tuple[Vector, Vector]]:
        """Extract connection pairs from mesh geometry"""
        connections = []
        mesh = obj.data
        
        # Simple approach: use edges as connections
        # In production, analyze cylindrical geometry
        for edge in mesh.edges:
            v1 = mesh.vertices[edge.vertices[0]]
            v2 = mesh.vertices[edge.vertices[1]]
            
            world_v1 = obj.matrix_world @ v1.co
            world_v2 = obj.matrix_world @ v2.co
            
            connections.append((world_v1, world_v2))
            
        return connections


class CHAIN_OT_remove_connections(Operator):
    """Remove connections from pattern"""
    bl_idname = "chain.remove_connections"
    bl_label = "Remove Connections"
    bl_description = "Remove connection objects"
    bl_options = {'REGISTER', 'UNDO'}
    
    remove_all: BoolProperty(
        name="Remove All",
        description="Remove all connection objects",
        default=False
    )
    
    @classmethod
    def poll(cls, context):
        return context.active_object is not None
        
    def execute(self, context):
        removed_count = 0
        
        if self.remove_all:
            # Remove all objects with "Connection" in name
            for obj in list(context.scene.objects):
                if "Connection" in obj.name or "connections" in obj.name.lower():
                    bpy.data.objects.remove(obj, do_unlink=True)
                    removed_count += 1
                    
        else:
            # Remove connections referenced by active object
            obj = context.active_object
            
            # Check custom property
            conn_name = obj.get("connections")
            if conn_name and conn_name in bpy.data.objects:
                bpy.data.objects.remove(bpy.data.objects[conn_name], do_unlink=True)
                del obj["connections"]
                removed_count += 1
                
            # Check if active object is a connection object
            elif "Connection" in obj.name or "connections" in obj.name.lower():
                bpy.data.objects.remove(obj, do_unlink=True)
                removed_count += 1
                
        if removed_count > 0:
            self.report({'INFO'}, f"Removed {removed_count} connection object(s)")
        else:
            self.report({'INFO'}, "No connections to remove")
            
        return {'FINISHED'}


# Register all operators
classes = [
    CHAIN_OT_auto_connect,
    CHAIN_OT_connect_selected,
    CHAIN_OT_optimize_connections,
    CHAIN_OT_remove_connections
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
        
def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
