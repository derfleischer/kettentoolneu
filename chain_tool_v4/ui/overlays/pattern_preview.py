"""
Chain Tool V4 - Pattern Preview System
=====================================

Simplified pattern preview overlay system that shows pattern generation
results in real-time without complex GPU shaders.

This is a lightweight preview system compared to the full GPU-accelerated
paint overlay system in paint_overlay.py.
"""

import bpy
import gpu
import gpu_extras
from gpu_extras.batch import batch_for_shader
from mathutils import Vector, Matrix
from typing import List, Dict, Tuple, Optional, Any
import time

class PatternPreviewSystem:
    """
    Lightweight pattern preview system for real-time pattern visualization.
    
    Features:
    - Real-time pattern point preview
    - Basic connection visualization
    - Configurable display modes
    - Performance-optimized for interactive use
    """
    
    def __init__(self):
        self.is_active = False
        self.preview_data = []
        self.connection_data = []
        self.display_mode = 'POINTS'  # 'POINTS', 'CONNECTIONS', 'BOTH'
        
        # Visualization parameters
        self.point_size = 4.0
        self.point_color = (0.2, 0.8, 0.4, 0.8)  # Green
        self.connection_color = (0.8, 0.2, 0.4, 0.6)  # Red
        self.connection_width = 2.0
        
        # Performance tracking
        self.last_update_time = 0.0
        self.update_interval = 0.1  # 10 FPS update rate
        
        # GPU resources
        self._point_batch = None
        self._connection_batch = None
        self._shader_2d_uniform_color = None
        
        # Pattern cache
        self._cached_pattern_hash = None
        self._cached_preview_data = None
        
        self._init_gpu_resources()
    
    def _init_gpu_resources(self):
        """Initialize GPU shaders and resources."""
        try:
            self._shader_2d_uniform_color = gpu.shader.from_builtin('UNIFORM_COLOR')
        except Exception as e:
            print(f"Pattern Preview: GPU initialization failed - {e}")
    
    def set_active(self, active: bool):
        """Activate or deactivate the preview system."""
        if active and not self.is_active:
            self._start_preview()
        elif not active and self.is_active:
            self._stop_preview()
    
    def _start_preview(self):
        """Start the preview system."""
        self.is_active = True
        
        # Register draw handler
        self._draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_preview,
            (),
            'WINDOW',
            'POST_VIEW'
        )
        
        print("Pattern Preview: Started")
    
    def _stop_preview(self):
        """Stop the preview system."""
        self.is_active = False
        
        # Remove draw handler
        if hasattr(self, '_draw_handler'):
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handler, 'WINDOW')
        
        # Cleanup GPU resources
        self._cleanup_batches()
        
        print("Pattern Preview: Stopped")
    
    def update_pattern_preview(self, pattern_type: str, parameters: Dict[str, Any], target_object=None):
        """
        Update pattern preview with new parameters.
        
        Args:
            pattern_type: Type of pattern ('voronoi', 'hexagonal', etc.)
            parameters: Pattern-specific parameters
            target_object: Target mesh object
        """
        
        if not self.is_active:
            return
        
        # Check if update is needed (performance optimization)
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return
        
        # Generate pattern hash for caching
        pattern_hash = self._generate_pattern_hash(pattern_type, parameters, target_object)
        if pattern_hash == self._cached_pattern_hash:
            return  # Use cached data
        
        try:
            # Generate preview pattern
            preview_points = self._generate_pattern_points(pattern_type, parameters, target_object)
            connections = self._generate_pattern_connections(preview_points, parameters)
            
            # Update preview data
            self.preview_data = preview_points
            self.connection_data = connections
            
            # Update GPU batches
            self._update_gpu_batches()
            
            # Cache results
            self._cached_pattern_hash = pattern_hash
            self._cached_preview_data = (preview_points, connections)
            
            self.last_update_time = current_time
            
            # Force viewport redraw
            self._force_redraw()
            
        except Exception as e:
            print(f"Pattern Preview: Update failed - {e}")
    
    def _generate_pattern_points(self, pattern_type: str, parameters: Dict, target_object) -> List[Vector]:
        """
        Generate pattern points for preview.
        Simplified version of actual pattern generation.
        """
        
        points = []
        
        if not target_object or target_object.type != 'MESH':
            return points
        
        try:
            # Get object data
            mesh = target_object.data
            world_matrix = target_object.matrix_world
            
            if pattern_type == 'voronoi':
                points = self._generate_voronoi_preview(mesh, world_matrix, parameters)
            elif pattern_type == 'hexagonal':
                points = self._generate_hexagonal_preview(mesh, world_matrix, parameters)
            elif pattern_type == 'random':
                points = self._generate_random_preview(mesh, world_matrix, parameters)
            else:
                # Default: surface sampling
                points = self._generate_surface_sampling_preview(mesh, world_matrix, parameters)
        
        except Exception as e:
            print(f"Pattern Preview: Point generation failed - {e}")
        
        return points
    
    def _generate_voronoi_preview(self, mesh, world_matrix: Matrix, parameters: Dict) -> List[Vector]:
        """Generate simplified Voronoi pattern preview."""
        import random
        
        points = []
        point_count = min(parameters.get('point_count', 50), 200)  # Limit for performance
        
        # Get mesh bounds
        bbox_min = Vector((float('inf'),) * 3)
        bbox_max = Vector((float('-inf'),) * 3)
        
        for vertex in mesh.vertices:
            world_pos = world_matrix @ vertex.co
            bbox_min = Vector(min(bbox_min[i], world_pos[i]) for i in range(3))
            bbox_max = Vector(max(bbox_max[i], world_pos[i]) for i in range(3))
        
        # Generate random points within bounds
        for _ in range(point_count):
            x = random.uniform(bbox_min.x, bbox_max.x)
            y = random.uniform(bbox_min.y, bbox_max.y)
            z = random.uniform(bbox_min.z, bbox_max.z)
            points.append(Vector((x, y, z)))
        
        return points
    
    def _generate_hexagonal_preview(self, mesh, world_matrix: Matrix, parameters: Dict) -> List[Vector]:
        """Generate simplified hexagonal pattern preview."""
        import math
        
        points = []
        spacing = parameters.get('spacing', 0.1)
        
        # Get mesh bounds
        bbox_min = Vector((float('inf'),) * 3)
        bbox_max = Vector((float('-inf'),) * 3)
        
        for vertex in mesh.vertices:
            world_pos = world_matrix @ vertex.co
            bbox_min = Vector(min(bbox_min[i], world_pos[i]) for i in range(3))
            bbox_max = Vector(max(bbox_max[i], world_pos[i]) for i in range(3))
        
        # Generate hexagonal grid in XY plane
        hex_height = spacing * math.sqrt(3) / 2
        
        y = bbox_min.y
        row = 0
        while y <= bbox_max.y:
            x_offset = (spacing / 2) if row % 2 else 0
            x = bbox_min.x + x_offset
            
            while x <= bbox_max.x:
                z = (bbox_min.z + bbox_max.z) / 2  # Simplified Z placement
                points.append(Vector((x, y, z)))
                x += spacing
            
            y += hex_height
            row += 1
            
            # Limit points for performance
            if len(points) > 300:
                break
        
        return points
    
    def _generate_random_preview(self, mesh, world_matrix: Matrix, parameters: Dict) -> List[Vector]:
        """Generate random distribution preview."""
        import random
        
        points = []
        point_count = min(parameters.get('point_count', 30), 150)
        
        # Use existing vertices as base with some randomization
        vertices = [world_matrix @ v.co for v in mesh.vertices]
        
        for _ in range(point_count):
            if vertices:
                base_vertex = random.choice(vertices)
                offset_strength = parameters.get('randomization', 0.05)
                
                offset = Vector((
                    random.uniform(-offset_strength, offset_strength),
                    random.uniform(-offset_strength, offset_strength),
                    random.uniform(-offset_strength, offset_strength)
                ))
                
                points.append(base_vertex + offset)
        
        return points
    
    def _generate_surface_sampling_preview(self, mesh, world_matrix: Matrix, parameters: Dict) -> List[Vector]:
        """Generate surface sampling preview."""
        import random
        
        points = []
        sample_count = min(parameters.get('sample_count', 40), 100)
        
        # Simple vertex-based sampling
        vertices = [world_matrix @ v.co for v in mesh.vertices]
        
        if len(vertices) > sample_count:
            # Random sampling
            points = random.sample(vertices, sample_count)
        else:
            points = vertices
        
        return points
    
    def _generate_pattern_connections(self, points: List[Vector], parameters: Dict) -> List[Tuple[Vector, Vector]]:
        """Generate connections between pattern points."""
        
        connections = []
        
        if len(points) < 2:
            return connections
        
        max_distance = parameters.get('connection_distance', 0.2)
        max_connections = min(len(points) * 3, 500)  # Performance limit
        
        # Simple nearest neighbor connections
        for i, point_a in enumerate(points):
            if len(connections) >= max_connections:
                break
                
            distances = []
            for j, point_b in enumerate(points):
                if i != j:
                    distance = (point_a - point_b).length
                    if distance <= max_distance:
                        distances.append((distance, point_b))
            
            # Connect to 2-3 nearest neighbors
            distances.sort(key=lambda x: x[0])
            for distance, point_b in distances[:3]:
                connections.append((point_a, point_b))
        
        return connections
    
    def _generate_pattern_hash(self, pattern_type: str, parameters: Dict, target_object) -> str:
        """Generate hash for pattern caching."""
        import hashlib
        
        # Create hash from parameters
        hash_data = f"{pattern_type}_{parameters}_{target_object.name if target_object else 'None'}"
        return hashlib.md5(hash_data.encode()).hexdigest()
    
    def _update_gpu_batches(self):
        """Update GPU batches with current preview data."""
        
        try:
            # Cleanup old batches
            self._cleanup_batches()
            
            # Create point batch
            if self.preview_data and self.display_mode in ['POINTS', 'BOTH']:
                coords = [(p.x, p.y, p.z) for p in self.preview_data]
                self._point_batch = batch_for_shader(
                    self._shader_2d_uniform_color,
                    'POINTS',
                    {"pos": coords}
                )
            
            # Create connection batch
            if self.connection_data and self.display_mode in ['CONNECTIONS', 'BOTH']:
                line_coords = []
                for start, end in self.connection_data:
                    line_coords.extend([(start.x, start.y, start.z), (end.x, end.y, end.z)])
                
                if line_coords:
                    self._connection_batch = batch_for_shader(
                        self._shader_2d_uniform_color,
                        'LINES',
                        {"pos": line_coords}
                    )
        
        except Exception as e:
            print(f"Pattern Preview: GPU batch update failed - {e}")
    
    def _cleanup_batches(self):
        """Cleanup GPU batch resources."""
        self._point_batch = None
        self._connection_batch = None
    
    def _draw_preview(self):
        """Draw the pattern preview."""
        
        if not self.is_active or not self._shader_2d_uniform_color:
            return
        
        try:
            # Enable blending for transparency
            gpu.state.blend_set('ALPHA')
            gpu.state.depth_test_set('LESS_EQUAL')
            
            # Draw points
            if self._point_batch and self.display_mode in ['POINTS', 'BOTH']:
                gpu.state.point_size_set(self.point_size)
                self._shader_2d_uniform_color.bind()
                self._shader_2d_uniform_color.uniform_float("color", self.point_color)
                self._point_batch.draw(self._shader_2d_uniform_color)
            
            # Draw connections
            if self._connection_batch and self.display_mode in ['CONNECTIONS', 'BOTH']:
                gpu.state.line_width_set(self.connection_width)
                self._shader_2d_uniform_color.bind()
                self._shader_2d_uniform_color.uniform_float("color", self.connection_color)
                self._connection_batch.draw(self._shader_2d_uniform_color)
            
            # Restore GPU state
            gpu.state.blend_set('NONE')
            gpu.state.depth_test_set('NONE')
            gpu.state.point_size_set(1.0)
            gpu.state.line_width_set(1.0)
        
        except Exception as e:
            print(f"Pattern Preview: Draw failed - {e}")
    
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
        
        if 'point_size' in kwargs:
            self.point_size = kwargs['point_size']
        
        if 'point_color' in kwargs:
            self.point_color = kwargs['point_color']
        
        if 'connection_color' in kwargs:
            self.connection_color = kwargs['connection_color']
        
        if 'connection_width' in kwargs:
            self.connection_width = kwargs['connection_width']
    
    def cleanup(self):
        """Cleanup resources."""
        if self.is_active:
            self._stop_preview()
        
        self._cleanup_batches()
        self.preview_data.clear()
        self.connection_data.clear()
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            'active': self.is_active,
            'point_count': len(self.preview_data),
            'connection_count': len(self.connection_data),
            'display_mode': self.display_mode,
            'last_update': self.last_update_time
        }
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information."""
        return {
            'gpu_batches': f"Points: {'✓' if self._point_batch else '✗'}, Connections: {'✓' if self._connection_batch else '✗'}",
            'cached_hash': self._cached_pattern_hash[:8] if self._cached_pattern_hash else 'None',
            'update_interval': f"{self.update_interval:.2f}s"
        }
