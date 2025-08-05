"""
Edge Detector for Chain Tool V4
Advanced edge detection and loop finding
"""

import bpy
import bmesh
import numpy as np
from typing import List, Set, Tuple, Optional, Dict
from dataclasses import dataclass
from mathutils import Vector

from utils.debug import debug
from utils.performance import performance
from utils.math_utils import calculate_angle, calculate_distance

# ============================================
# DATA CLASSES
# ============================================

@dataclass
class EdgeLoop:
    """Represents a continuous edge loop"""
    vertices: List[int]
    edges: List[Tuple[int, int]]
    is_closed: bool
    length: float
    
    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

@dataclass
class EdgeFeature:
    """Edge feature information"""
    edge: Tuple[int, int]
    angle: float  # Angle between adjacent faces
    length: float
    is_sharp: bool
    is_boundary: bool
    is_seam: bool
    crease_weight: float

# ============================================
# EDGE DETECTOR
# ============================================

class EdgeDetector:
    """Advanced edge detection and analysis"""
    
    def __init__(self):
        self.mesh = None
        self.bm = None
        self.edge_features = {}
        
    @performance.track_function('GEOMETRY')
    def analyze_edges(self, obj: bpy.types.Object,
                     sharp_angle_threshold: float = 30.0) -> Dict[Tuple[int, int], EdgeFeature]:
        """Analyze all edges in mesh"""
        
        if obj.type != 'MESH':
            raise ValueError(f"Object {obj.name} is not a mesh")
            
        debug.info('GEOMETRY', f"Analyzing edges for {obj.name}")
        
        self.mesh = obj.data
        self.bm = bmesh.new()
        self.bm.from_mesh(self.mesh)
        self.bm.edges.ensure_lookup_table()
        self.bm.faces.ensure_lookup_table()
        
        # Analyze each edge
        self.edge_features.clear()
        
        for edge in self.bm.edges:
            feature = self._analyze_edge(edge, sharp_angle_threshold)
            edge_key = (edge.verts[0].index, edge.verts[1].index)
            self.edge_features[edge_key] = feature
        
        debug.debug('GEOMETRY', f"Analyzed {len(self.edge_features)} edges")
        
        return self.edge_features
    
    def find_edge_loops(self, 
                       boundary_only: bool = True,
                       min_length: int = 3) -> List[EdgeLoop]:
        """Find all edge loops in mesh"""
        
        if not self.bm:
            raise RuntimeError("No mesh analyzed")
            
        loops = []
        visited_edges = set()
        
        for edge in self.bm.edges:
            if edge in visited_edges:
                continue
                
            if boundary_only and not edge.is_boundary:
                continue
                
            # Try to trace loop from this edge
            loop = self._trace_edge_loop(edge, visited_edges, boundary_only)
            
            if loop and len(loop.vertices) >= min_length:
                loops.append(loop)
        
        debug.info('GEOMETRY', f"Found {len(loops)} edge loops")
        
        return loops
    
    def find_sharp_edges(self, angle_threshold: float = 30.0) -> List[Tuple[int, int]]:
        """Find all sharp edges"""
        
        sharp_edges = []
        
        for edge_key, feature in self.edge_features.items():
            if feature.is_sharp or feature.angle > angle_threshold:
                sharp_edges.append(edge_key)
        
        return sharp_edges
    
    def find_boundary_edges(self) -> List[Tuple[int, int]]:
        """Find all boundary edges"""
        
        boundary_edges = []
        
        for edge_key, feature in self.edge_features.items():
            if feature.is_boundary:
                boundary_edges.append(edge_key)
        
        return boundary_edges
    
    def get_edge_chains(self, 
                       edge_type: str = 'boundary',
                       min_chain_length: int = 2) -> List[List[Tuple[int, int]]]:
        """Get chains of connected edges of specified type"""
        
        # Select edges based on type
        if edge_type == 'boundary':
            target_edges = self.find_boundary_edges()
        elif edge_type == 'sharp':
            target_edges = self.find_sharp_edges()
        else:
            target_edges = list(self.edge_features.keys())
        
        # Build adjacency
        adjacency = self._build_edge_adjacency(target_edges)
        
        # Find chains
        chains = []
        visited = set()
        
        for edge in target_edges:
            if edge in visited:
                continue
                
            chain = self._trace_edge_chain(edge, adjacency, visited)
            
            if len(chain) >= min_chain_length:
                chains.append(chain)
        
        return chains
    
    def simplify_edge_loops(self, 
                           loops: List[EdgeLoop],
                           tolerance: float = 0.1) -> List[EdgeLoop]:
        """Simplify edge loops using Douglas-Peucker algorithm"""
        
        simplified_loops = []
        
        for loop in loops:
            if len(loop.vertices) <= 3:
                simplified_loops.append(loop)
                continue
                
            # Get vertex positions
            positions = [Vector(self.bm.verts[i].co) for i in loop.vertices]
            
            # Simplify
            simplified_indices = self._douglas_peucker(positions, tolerance)
            
            # Create simplified loop
            simplified_vertices = [loop.vertices[i] for i in simplified_indices]
            simplified_edges = []
            
            for i in range(len(simplified_vertices) - 1):
                simplified_edges.append((simplified_vertices[i], simplified_vertices[i + 1]))
                
            if loop.is_closed and len(simplified_vertices) > 2:
                simplified_edges.append((simplified_vertices[-1], simplified_vertices[0]))
            
            # Calculate length
            length = sum(self.edge_features.get((e[0], e[1]), 
                        self.edge_features.get((e[1], e[0]))).length 
                        for e in simplified_edges if (e[0], e[1]) in self.edge_features 
                        or (e[1], e[0]) in self.edge_features)
            
            simplified_loop = EdgeLoop(
                vertices=simplified_vertices,
                edges=simplified_edges,
                is_closed=loop.is_closed,
                length=length
            )
            
            simplified_loops.append(simplified_loop)
        
        return simplified_loops
    
    # ============================================
    # PRIVATE METHODS
    # ============================================
    
    def _analyze_edge(self, edge: bmesh.types.BMEdge, 
                     sharp_angle_threshold: float) -> EdgeFeature:
        """Analyze single edge"""
        
        # Basic properties
        v0, v1 = edge.verts
        length = edge.calc_length()
        
        # Check if boundary
        is_boundary = edge.is_boundary
        
        # Calculate angle between faces
        angle = 0.0
        if len(edge.link_faces) == 2:
            face1, face2 = edge.link_faces
            angle = calculate_angle(face1.normal, face2.normal, degrees=True)
        
        # Check if sharp
        is_sharp = edge.smooth is False or angle > sharp_angle_threshold
        
        # Check if seam
        is_seam = edge.seam
        
        # Get crease weight
        crease_weight = edge.crease
        
        return EdgeFeature(
            edge=(v0.index, v1.index),
            angle=angle,
            length=length,
            is_sharp=is_sharp,
            is_boundary=is_boundary,
            is_seam=is_seam,
            crease_weight=crease_weight
        )
    
    def _trace_edge_loop(self, start_edge: bmesh.types.BMEdge,
                        visited_edges: Set,
                        boundary_only: bool) -> Optional[EdgeLoop]:
        """Trace a continuous edge loop"""
        
        vertices = []
        edges = []
        total_length = 0.0
        
        current_edge = start_edge
        current_vert = start_edge.verts[0]
        
        while current_edge:
            # Mark as visited
            visited_edges.add(current_edge)
            
            # Add to loop
            vertices.append(current_vert.index)
            edge_tuple = (current_edge.verts[0].index, current_edge.verts[1].index)
            edges.append(edge_tuple)
            total_length += current_edge.calc_length()
            
            # Get next vertex
            next_vert = current_edge.other_vert(current_vert)
            
            # Find next edge
            next_edge = None
            for edge in next_vert.link_edges:
                if edge in visited_edges:
                    continue
                    
                if boundary_only and not edge.is_boundary:
                    continue
                    
                # Check if edge continues the loop smoothly
                if self._is_continuous_edge(current_edge, edge, next_vert):
                    next_edge = edge
                    break
            
            # Check if we completed the loop
            if not next_edge and next_vert == start_edge.verts[0]:
                # Closed loop
                return EdgeLoop(
                    vertices=vertices,
                    edges=edges,
                    is_closed=True,
                    length=total_length
                )
            elif not next_edge:
                # Open loop - trace in opposite direction
                if len(vertices) == 1:
                    # Try other direction
                    vertices.clear()
                    edges.clear()
                    total_length = 0.0
                    current_edge = start_edge
                    current_vert = start_edge.verts[1]
                    continue
                else:
                    # Open loop complete
                    vertices.append(next_vert.index)
                    return EdgeLoop(
                        vertices=vertices,
                        edges=edges,
                        is_closed=False,
                        length=total_length
                    )
            
            current_edge = next_edge
            current_vert = next_vert
        
        return None
    
    def _is_continuous_edge(self, edge1: bmesh.types.BMEdge,
                           edge2: bmesh.types.BMEdge,
                           shared_vert: bmesh.types.BMVert) -> bool:
        """Check if two edges form a continuous path"""
        
        # Get the other vertices
        v1 = edge1.other_vert(shared_vert)
        v2 = edge2.other_vert(shared_vert)
        
        # Calculate angle
        dir1 = (shared_vert.co - v1.co).normalized()
        dir2 = (v2.co - shared_vert.co).normalized()
        
        angle = calculate_angle(dir1, dir2, degrees=True)
        
        # Check if reasonably straight
        return angle > 90.0  # Allow up to 90 degree turns
    
    def _build_edge_adjacency(self, edges: List[Tuple[int, int]]) -> Dict[int, Set[Tuple[int, int]]]:
        """Build adjacency map for edges"""
        
        adjacency = {}
        
        for edge in edges:
            v0, v1 = edge
            
            if v0 not in adjacency:
                adjacency[v0] = set()
            if v1 not in adjacency:
                adjacency[v1] = set()
                
            adjacency[v0].add(edge)
            adjacency[v1].add(edge)
        
        return adjacency
    
    def _trace_edge_chain(self, start_edge: Tuple[int, int],
                         adjacency: Dict[int, Set[Tuple[int, int]]],
                         visited: Set) -> List[Tuple[int, int]]:
        """Trace a chain of connected edges"""
        
        chain = [start_edge]
        visited.add(start_edge)
        
        # Trace forward
        current_vert = start_edge[1]
        while current_vert in adjacency:
            found_next = False
            
            for edge in adjacency[current_vert]:
                if edge not in visited:
                    chain.append(edge)
                    visited.add(edge)
                    
                    # Get next vertex
                    current_vert = edge[1] if edge[0] == current_vert else edge[0]
                    found_next = True
                    break
            
            if not found_next:
                break
        
        # Trace backward
        current_vert = start_edge[0]
        while current_vert in adjacency:
            found_next = False
            
            for edge in adjacency[current_vert]:
                if edge not in visited:
                    chain.insert(0, edge)
                    visited.add(edge)
                    
                    # Get next vertex
                    current_vert = edge[1] if edge[0] == current_vert else edge[0]
                    found_next = True
                    break
            
            if not found_next:
                break
        
        return chain
    
    def _douglas_peucker(self, points: List[Vector], tolerance: float) -> List[int]:
        """Douglas-Peucker line simplification algorithm"""
        
        if len(points) <= 2:
            return list(range(len(points)))
        
        # Find point with maximum distance
        max_dist = 0.0
        max_idx = 0
        
        for i in range(1, len(points) - 1):
            dist = self._point_line_distance(points[i], points[0], points[-1])
            
            if dist > max_dist:
                max_dist = dist
                max_idx = i
        
        # If max distance is greater than tolerance, recursively simplify
        if max_dist > tolerance:
            # Recursive simplification
            left = self._douglas_peucker(points[:max_idx + 1], tolerance)
            right = self._douglas_peucker(points[max_idx:], tolerance)
            
            # Combine results
            result = left[:-1] + [i + max_idx for i in right]
            return result
        else:
            # Return endpoints
            return [0, len(points) - 1]
    
    def _point_line_distance(self, point: Vector, line_start: Vector, line_end: Vector) -> float:
        """Calculate distance from point to line segment"""
        
        line_vec = line_end - line_start
        point_vec = point - line_start
        
        line_length = line_vec.length
        if line_length == 0:
            return point_vec.length
        
        # Project point onto line
        t = max(0.0, min(1.0, point_vec.dot(line_vec) / (line_length * line_length)))
        projection = line_start + line_vec * t
        
        return (point - projection).length
    
    def cleanup(self):
        """Clean up resources"""
        if self.bm:
            self.bm.free()
            self.bm = None
        self.mesh = None
        self.edge_features.clear()

# ============================================
# UTILITY FUNCTIONS
# ============================================

def detect_edges(obj: bpy.types.Object, sharp_angle: float = 30.0) -> Dict[Tuple[int, int], EdgeFeature]:
    """Quick edge detection"""
    detector = EdgeDetector()
    return detector.analyze_edges(obj, sharp_angle)

def find_boundary_loops(obj: bpy.types.Object) -> List[EdgeLoop]:
    """Find boundary loops in mesh"""
    detector = EdgeDetector()
    detector.analyze_edges(obj)
    loops = detector.find_edge_loops(boundary_only=True)
    detector.cleanup()
    return loops

def get_mesh_outline(obj: bpy.types.Object, simplify: float = 0.1) -> List[List[Vector]]:
    """Get simplified mesh outline"""
    detector = EdgeDetector()
    detector.analyze_edges(obj)
    
    # Find boundary loops
    loops = detector.find_edge_loops(boundary_only=True)
    
    # Simplify if requested
    if simplify > 0:
        loops = detector.simplify_edge_loops(loops, simplify)
    
    # Convert to positions
    outlines = []
    for loop in loops:
        positions = [Vector(detector.bm.verts[i].co) for i in loop.vertices]
        outlines.append(positions)
    
    detector.cleanup()
    return outlines
