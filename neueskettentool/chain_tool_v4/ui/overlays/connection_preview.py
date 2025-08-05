"""
Chain Tool V4 - Connection Preview System
========================================

Basic connection visualization overlay for showing potential connections
between pattern points and optimized connection paths.

This system provides real-time feedback during connection optimization
and dual-material path planning.
"""

import bpy
import gpu
import gpu_extras
from gpu_extras.batch import batch_for_shader
from mathutils import Vector, Matrix
from typing import List, Dict, Tuple, Optional, Any, Set
import time
import math

class ConnectionPreviewSystem:
    """
    Basic connection visualization system for pattern connections.
    
    Features:
    - Connection path visualization
    - Connection strength/quality indication
    - Real-time optimization preview
    - Dual-material connection paths
    """
    
    def __init__(self):
        self.is_active = False
        self.connections = []  # List of (start_point, end_point, properties)
        self.connection_paths = []  # Optimized connection paths
        
        # Visualization modes
        self.display_mode = 'CONNECTIONS'  # 'CONNECTIONS', 'PATHS', 'BOTH'
        self.show_quality_indicators = True
        self.show_material_types = True
        
        # Visual parameters
        self.connection_colors = {
            'default': (0.4, 0.7, 0.9, 0.8),      # Blue
            'strong': (0.2, 0.9, 0.2, 0.9),       # Green  
            'weak': (0.9, 0.9, 0.2, 0.7),         # Yellow
            'problematic': (0.9, 0.2, 0.2, 0.8),  # Red
            'tpu_material': (0.9, 0.4, 0.9, 0.8), # Magenta (flexible)
            'petg_material': (0.2, 0.8, 0.8, 0.8) # Cyan (rigid)
        }
        
        self.line_widths = {
            'thin': 1.0,
            'normal': 2.0,
            'thick': 3.0,
            'extra_thick': 4.0
        }
        
        # Performance settings
        self.max_connections = 1000
        self.update_interval = 0.05  # 20 FPS
        self.last_update_time = 0.0
        
        # GPU resources
        self._connection_batches = {}
        self._path_batches = {}
        self._shader_uniform_color = None
        
        # Connection analysis cache
        self._analysis_cache = {}
        self._cache_timeout = 2.0  # seconds
        
        self._init_gpu_resources()
    
    def _init_gpu_resources(self):
        """Initialize GPU shaders."""
        try:
            self._shader_uniform_color = gpu.shader.from_builtin('UNIFORM_COLOR')
        except Exception as e:
            print(f"Connection Preview: GPU initialization failed - {e}")
    
    def set_active(self, active: bool):
        """Activate or deactivate the connection preview."""
        if active and not self.is_active:
            self._start_preview()
        elif not active and self.is_active:
            self._stop_preview()
    
    def _start_preview(self):
        """Start the connection preview system."""
        self.is_active = True
        
        # Register draw handler
        self._draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_connections,
            (),
            'WINDOW',
            'POST_VIEW'
        )
        
        print("Connection Preview: Started")
    
    def _stop_preview(self):
        """Stop the connection preview system."""
        self.is_active = False
        
        # Remove draw handler
        if hasattr(self, '_draw_handler'):
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handler, 'WINDOW')
        
        # Cleanup GPU resources
        self._cleanup_batches()
        
        print("Connection Preview: Stopped")
    
    def update_connections(self, points: List[Vector], connection_params: Dict[str, Any] = None):
        """
        Update connection visualization with new points.
        
        Args:
            points: List of 3D points to connect
            connection_params: Connection generation parameters
        """
        
        if not self.is_active:
            return
        
        # Performance throttling
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return
        
        if not points or len(points) < 2:
            self.connections.clear()
            self.connection_paths.clear()
            self._update_gpu_batches()
            return
        
        try:
            # Generate connections
            self.connections = self._generate_connections(points, connection_params or {})
            
            # Generate optimized paths if requested
            if self.display_mode in ['PATHS', 'BOTH']:
                self.connection_paths = self._generate_connection_paths(self.connections)
            
            # Update GPU visualization
            self._update_gpu_batches()
            
            self.last_update_time = current_time
            
            # Force redraw
            self._force_redraw()
            
        except Exception as e:
            print(f"Connection Preview: Update failed - {e}")
    
    def _generate_connections(self, points: List[Vector], params: Dict) -> List[Tuple[Vector, Vector, Dict]]:
        """
        Generate connections between points with quality analysis.
        
        Returns:
            List of (start_point, end_point, properties) tuples
        """
        
        connections = []
        
        max_distance = params.get('max_connection_distance', 0.15)
        min_distance = params.get('min_connection_distance', 0.02)
        connection_algorithm = params.get('algorithm', 'nearest_neighbor')
        
        if connection_algorithm == 'nearest_neighbor':
            connections = self._generate_nearest_neighbor_connections(
                points, max_distance, min_distance
            )
        elif connection_algorithm == 'delaunay':
            connections = self._generate_delaunay_connections(points, max_distance)
        elif connection_algorithm == 'minimum_spanning_tree':
            connections = self._generate_mst_connections(points, max_distance)
        else:
            # Default: simple distance-based connections
            connections = self._generate_distance_based_connections(
                points, max_distance, min_distance
            )
        
        # Limit connections for performance
        if len(connections) > self.max_connections:
            connections = connections[:self.max_connections]
        
        # Analyze connection quality
        connections = self._analyze_connection_quality(connections, params)
        
        return connections
    
    def _generate_nearest_neighbor_connections(self, points: List[Vector], max_dist: float, min_dist: float) -> List[Tuple[Vector, Vector, Dict]]:
        """Generate nearest neighbor connections."""
        
        connections = []
        neighbors_per_point = 3  # Connect each point to 3 nearest neighbors
        
        for i, point_a in enumerate(points):
            # Find nearest neighbors
            distances = []
            for j, point_b in enumerate(points):
                if i != j:
                    distance = (point_a - point_b).length
                    if min_dist <= distance <= max_dist:
                        distances.append((distance, j, point_b))
            
            # Sort by distance and take nearest neighbors
            distances.sort(key=lambda x: x[0])
            for distance, j, point_b in distances[:neighbors_per_point]:
                properties = {
                    'distance': distance,
                    'strength': 1.0 - (distance / max_dist),  # Closer = stronger
                    'type': 'nearest_neighbor'
                }
                connections.append((point_a, point_b, properties))
        
        return connections
    
    def _generate_delaunay_connections(self, points: List[Vector], max_dist: float) -> List[Tuple[Vector, Vector, Dict]]:
        """Generate connections based on simplified Delaunay triangulation."""
        
        connections = []
        
        # For performance, use simplified 2D Delaunay approach
        # Project points to XY plane
        points_2d = [(p.x, p.y) for p in points]
        
        try:
            from scipy.spatial import Delaunay
            tri = Delaunay(points_2d)
            
            # Extract edges from triangulation
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    edge = tuple(sorted([simplex[i], simplex[(i+1)%3]]))
                    edges.add(edge)
            
            # Convert edges to connections
            for i, j in edges:
                point_a, point_b = points[i], points[j]
                distance = (point_a - point_b).length
                
                if distance <= max_dist:
                    properties = {
                        'distance': distance,
                        'strength': 0.8,  # Delaunay connections are generally good
                        'type': 'delaunay'
                    }
                    connections.append((point_a, point_b, properties))
        
        except ImportError:
            # Fallback to nearest neighbor if scipy not available
            print("Connection Preview: Scipy not available, using nearest neighbor")
            connections = self._generate_nearest_neighbor_connections(points, max_dist, 0.01)
        
        except Exception as e:
            print(f"Connection Preview: Delaunay failed - {e}")
            connections = self._generate_nearest_neighbor_connections(points, max_dist, 0.01)
        
        return connections
    
    def _generate_mst_connections(self, points: List[Vector], max_dist: float) -> List[Tuple[Vector, Vector, Dict]]:
        """Generate minimum spanning tree connections."""
        
        connections = []
        n_points = len(points)
        
        if n_points < 2:
            return connections
        
        # Build distance matrix
        distances = {}
        for i in range(n_points):
            for j in range(i + 1, n_points):
                dist = (points[i] - points[j]).length
                if dist <= max_dist:
                    distances[(i, j)] = dist
        
        # Simple MST using Kruskal's algorithm
        edges = [(dist, i, j) for (i, j), dist in distances.items()]
        edges.sort()  # Sort by distance
        
        # Union-Find structure
        parent = list(range(n_points))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                return True
            return False
        
        # Build MST
        mst_edges = []
        for dist, i, j in edges:
            if union(i, j):
                mst_edges.append((i, j, dist))
                if len(mst_edges) == n_points - 1:
                    break
        
        # Convert to connections
        for i, j, dist in mst_edges:
            properties = {
                'distance': dist,
                'strength': 0.9,  # MST connections are strong
                'type': 'mst'
            }
            connections.append((points[i], points[j], properties))
        
        return connections
    
    def _generate_distance_based_connections(self, points: List[Vector], max_dist: float, min_dist: float) -> List[Tuple[Vector, Vector, Dict]]:
        """Generate simple distance-based connections."""
        
        connections = []
        
        for i, point_a in enumerate(points):
            for j, point_b in enumerate(points[i+1:], i+1):
                distance = (point_a - point_b).length
                
                if min_dist <= distance <= max_dist:
                    properties = {
                        'distance': distance,
                        'strength': 1.0 - (distance / max_dist),
                        'type': 'distance_based'
                    }
                    connections.append((point_a, point_b, properties))
        
        return connections
    
    def _analyze_connection_quality(self, connections: List[Tuple], params: Dict) -> List[Tuple]:
        """Analyze and classify connection quality."""
        
        analyzed_connections = []
        
        for start, end, props in connections:
            # Calculate quality metrics
            distance = props['distance']
            strength = props.get('strength', 0.5)
            
            # Classify connection quality
            if strength > 0.8:
                quality = 'strong'
            elif strength > 0.5:
                quality = 'normal'
            elif strength > 0.3:
                quality = 'weak'
            else:
                quality = 'problematic'
            
            # Determine material type (simplified logic)
            material_type = 'default'
            if params.get('dual_material_mode', False):
                # Flexible connections for shorter distances
                if distance < params.get('flexible_threshold', 0.08):
                    material_type = 'tpu_material'
                else:
                    material_type = 'petg_material'
            
            # Update properties
            props.update({
                'quality': quality,
                'material_type': material_type,
                'color': self.connection_colors.get(material_type, self.connection_colors['default']),
                'line_width': self._get_line_width_for_quality(quality)
            })
            
            analyzed_connections.append((start, end, props))
        
        return analyzed_connections
    
    def _get_line_width_for_quality(self, quality: str) -> float:
        """Get line width based on connection quality."""
        width_map = {
            'strong': self.line_widths['thick'],
            'normal': self.line_widths['normal'],
            'weak': self.line_widths['thin'],
            'problematic': self.line_widths['thin']
        }
        return width_map.get(quality, self.line_widths['normal'])
    
    def _generate_connection_paths(self, connections: List[Tuple]) -> List[List[Vector]]:
        """Generate optimized connection paths."""
        
        paths = []
        
        for start, end, props in connections:
            # Simple straight line path (can be enhanced with curve optimization)
            path = [start, end]
            
            # Add curve for longer connections
            if props['distance'] > 0.1:
                mid_point = (start + end) / 2
                # Add slight arc
                offset_direction = Vector((0, 0, 0.02))  # Small upward arc
                mid_point += offset_direction
                path = [start, mid_point, end]
            
            paths.append(path)
        
        return paths
    
    def _update_gpu_batches(self):
        """Update GPU batches for rendering."""
        
        try:
            # Clear old batches
            self._cleanup_batches()
            
            if not self._shader_uniform_color:
                return
            
            # Group connections by visual properties
            connection_groups = {}
            
            for start, end, props in self.connections:
                key = (props.get('material_type', 'default'), props.get('quality', 'normal'))
                if key not in connection_groups:
                    connection_groups[key] = []
                connection_groups[key].extend([
                    (start.x, start.y, start.z),
                    (end.x, end.y, end.z)
                ])
            
            # Create batches for each group
            for (material_type, quality), coords in connection_groups.items():
                if coords:
                    batch = batch_for_shader(
                        self._shader_uniform_color,
                        'LINES',
                        {"pos": coords}
                    )
                    self._connection_batches[(material_type, quality)] = {
                        'batch': batch,
                        'color': self.connection_colors.get(material_type, self.connection_colors['default']),
                        'width': self._get_line_width_for_quality(quality)
                    }
        
        except Exception as e:
            print(f"Connection Preview: GPU batch update failed - {e}")
    
    def _cleanup_batches(self):
        """Cleanup GPU batch resources."""
        self._connection_batches.clear()
        self._path_batches.clear()
    
    def _draw_connections(self):
        """Draw the connection visualization."""
        
        if not self.is_active or not self._shader_uniform_color:
            return
        
        try:
            # Enable blending and depth testing
            gpu.state.blend_set('ALPHA')
            gpu.state.depth_test_set('LESS_EQUAL')
            
            # Draw connection batches
            for batch_info in self._connection_batches.values():
                gpu.state.line_width_set(batch_info['width'])
                self._shader_uniform_color.bind()
                self._shader_uniform_color.uniform_float("color", batch_info['color'])
                batch_info['batch'].draw(self._shader_uniform_color)
            
            # Restore GPU state
            gpu.state.blend_set('NONE')
            gpu.state.depth_test_set('NONE')
            gpu.state.line_width_set(1.0)
        
        except Exception as e:
            print(f"Connection Preview: Draw failed - {e}")
    
    def _force_redraw(self):
        """Force viewport redraw."""
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
    
    def update_parameters(self, **kwargs):
        """Update display parameters."""
        if 'display_mode' in kwargs:
            self.display_mode = kwargs['display_mode']
        
        if 'show_quality_indicators' in kwargs:
            self.show_quality_indicators = kwargs['show_quality_indicators']
        
        if 'show_material_types' in kwargs:
            self.show_material_types = kwargs['show_material_types']
        
        # Update colors if provided
        if 'connection_colors' in kwargs:
            self.connection_colors.update(kwargs['connection_colors'])
    
    def clear_connections(self):
        """Clear all connections."""
        self.connections.clear()
        self.connection_paths.clear()
        self._cleanup_batches()
        if self.is_active:
            self._force_redraw()
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get statistics about current connections."""
        
        if not self.connections:
            return {'total': 0}
        
        stats = {
            'total': len(self.connections),
            'by_quality': {},
            'by_material': {},
            'avg_distance': 0.0,
            'min_distance': float('inf'),
            'max_distance': 0.0
        }
        
        total_distance = 0.0
        
        for _, _, props in self.connections:
            # Quality stats
            quality = props.get('quality', 'unknown')
            stats['by_quality'][quality] = stats['by_quality'].get(quality, 0) + 1
            
            # Material stats  
            material = props.get('material_type', 'default')
            stats['by_material'][material] = stats['by_material'].get(material, 0) + 1
            
            # Distance stats
            distance = props.get('distance', 0.0)
            total_distance += distance
            stats['min_distance'] = min(stats['min_distance'], distance)
            stats['max_distance'] = max(stats['max_distance'], distance)
        
        stats['avg_distance'] = total_distance / len(self.connections)
        
        return stats
    
    def cleanup(self):
        """Cleanup all resources."""
        if self.is_active:
            self._stop_preview()
        
        self._cleanup_batches()
        self.connections.clear()
        self.connection_paths.clear()
        self._analysis_cache.clear()
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            'active': self.is_active,
            'connection_count': len(self.connections),
            'path_count': len(self.connection_paths),
            'display_mode': self.display_mode,
            'batch_count': len(self._connection_batches)
        }
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information."""
        stats = self.get_connection_stats()
        return {
            'gpu_batches': len(self._connection_batches),
            'cache_size': len(self._analysis_cache),
            'quality_distribution': str(stats.get('by_quality', {})),
            'avg_distance': f"{stats.get('avg_distance', 0):.3f}"
        }
