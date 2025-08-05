"""
Chain Tool V4 - Paint Overlay System (KRITISCHSTE DATEI!)
GPU-Shader basierte Real-time Paint-Visualisierung

FEATURES:
- GPU Shader für Brush Preview (60fps+)
- Stroke Buffer für flüssige Darstellung
- Auto-Connect Visualisierung beim Malen
- Symmetrie-Linien Anzeige
- Performance-optimierte Rendering-Pipeline
- Memory-Management für 1000+ Spheres
"""

import bpy
import gpu
import gpu_extras
import bmesh
import mathutils
from gpu_extras.batch import batch_for_shader
from mathutils import Vector, Matrix
from bpy_extras import view3d_utils
import numpy as np
from typing import List, Dict, Tuple, Optional
import time

from ...core.properties import ChainToolProperties
from ...core.state_manager import StateManager
from ...utils.performance import PerformanceMonitor
from ...utils.math_utils import MathUtils


class PaintOverlaySystem:
    """
    GPU-basiertes Paint Overlay System
    Verwaltet alle visuellen Aspekte des Paint-Modes
    """
    
    def __init__(self):
        self.is_active = False
        self.is_initialized = False
        
        # GPU Shader und Batches
        self.brush_shader = None
        self.stroke_shader = None
        self.connection_shader = None
        self.symmetry_shader = None
        
        # Rendering Batches
        self.brush_batch = None
        self.stroke_batch = None
        self.connection_batch = None
        self.symmetry_batch = None
        
        # Stroke Buffer System
        self.stroke_buffer = StrokeBuffer()
        self.connection_buffer = ConnectionBuffer()
        
        # Performance Monitoring
        self.perf_monitor = PerformanceMonitor()
        self.last_frame_time = 0
        self.fps_counter = FPSCounter()
        
        # Event Handler
        self.draw_handler = None
        self.mouse_pos = Vector((0, 0))
        self.brush_size = 1.0
        
        # Auto-Connect System
        self.auto_connect_visualizer = AutoConnectVisualizer()
        
        # Symmetry System
        self.symmetry_visualizer = SymmetryVisualizer()
    
    def initialize(self):
        """Initialisiert GPU-Shader und Ressourcen"""
        if self.is_initialized:
            return True
        
        try:
            # GPU Shader laden
            self._create_shaders()
            
            # Event Handler registrieren
            self._register_draw_handler()
            
            # Performance-Monitoring starten
            self.perf_monitor.start_paint_monitoring()
            
            self.is_initialized = True
            print("✓ Paint Overlay System initialisiert")
            return True
            
        except Exception as e:
            print(f"❌ Paint Overlay Initialisierung fehlgeschlagen: {e}")
            return False
    
    def cleanup(self):
        """Räumt GPU-Ressourcen auf"""
        if self.draw_handler:
            bpy.types.SpaceView3D.draw_handler_remove(self.draw_handler, 'WINDOW')
            self.draw_handler = None
        
        # GPU-Ressourcen freigeben
        self._cleanup_shaders()
        self._cleanup_batches()
        
        # Monitoring stoppen
        self.perf_monitor.stop_paint_monitoring()
        
        self.is_initialized = False
        self.is_active = False
        print("✓ Paint Overlay System bereinigt")
    
    def activate(self, context):
        """Aktiviert Paint Overlay"""
        if not self.is_initialized:
            if not self.initialize():
                return False
        
        self.is_active = True
        self.fps_counter.reset()
        
        # Context-abhängige Initialisierung
        self._update_viewport_settings(context)
        
        return True
    
    def deactivate(self):
        """Deaktiviert Paint Overlay"""
        self.is_active = False
        self.stroke_buffer.clear()
        self.connection_buffer.clear()
    
    def _create_shaders(self):
        """Erstellt alle GPU-Shader"""
        
        # BRUSH PREVIEW SHADER
        brush_vertex_shader = """
        in vec3 pos;
        in vec2 uv;
        
        uniform mat4 ModelViewProjectionMatrix;
        uniform vec3 brush_center;
        uniform float brush_size;
        uniform float brush_falloff;
        
        out vec2 v_uv;
        out float v_distance;
        
        void main() {
            vec3 world_pos = pos * brush_size + brush_center;
            gl_Position = ModelViewProjectionMatrix * vec4(world_pos, 1.0);
            
            v_uv = uv;
            v_distance = length(uv - vec2(0.5)) * 2.0;  // 0.0 = center, 1.0 = edge
        }
        """
        
        brush_fragment_shader = """
        in vec2 v_uv;
        in float v_distance;
        
        uniform float brush_opacity;
        uniform vec4 brush_color;
        uniform float brush_hardness;
        
        out vec4 fragColor;
        
        void main() {
            // Brush-Falloff berechnen
            float falloff = 1.0 - smoothstep(0.0, 1.0, v_distance);
            falloff = pow(falloff, brush_hardness);
            
            // Brush-Ring Effekt für bessere Sichtbarkeit
            float ring = abs(v_distance - 0.8);
            float ring_alpha = 1.0 - smoothstep(0.0, 0.1, ring);
            
            float final_alpha = max(falloff * brush_opacity, ring_alpha * 0.3);
            fragColor = vec4(brush_color.rgb, final_alpha);
        }
        """
        
        self.brush_shader = gpu.types.GPUShader(brush_vertex_shader, brush_fragment_shader)
        
        # STROKE RENDERING SHADER
        stroke_vertex_shader = """
        in vec3 pos;
        in float point_size;
        in vec4 point_color;
        
        uniform mat4 ModelViewProjectionMatrix;
        uniform float global_scale;
        
        out vec4 v_color;
        
        void main() {
            gl_Position = ModelViewProjectionMatrix * vec4(pos, 1.0);
            gl_PointSize = point_size * global_scale;
            v_color = point_color;
        }
        """
        
        stroke_fragment_shader = """
        in vec4 v_color;
        out vec4 fragColor;
        
        void main() {
            // Kreisförmige Punkte
            vec2 coord = gl_PointCoord - vec2(0.5);
            float dist = length(coord);
            
            if (dist > 0.5) {
                discard;
            }
            
            // Soft-Edge Falloff
            float alpha = 1.0 - smoothstep(0.3, 0.5, dist);
            fragColor = vec4(v_color.rgb, v_color.a * alpha);
        }
        """
        
        self.stroke_shader = gpu.types.GPUShader(stroke_vertex_shader, stroke_fragment_shader)
        
        # CONNECTION PREVIEW SHADER
        connection_vertex_shader = """
        in vec3 pos;
        in vec4 line_color;
        
        uniform mat4 ModelViewProjectionMatrix;
        
        out vec4 v_color;
        
        void main() {
            gl_Position = ModelViewProjectionMatrix * vec4(pos, 1.0);
            v_color = line_color;
        }
        """
        
        connection_fragment_shader = """
        in vec4 v_color;
        out vec4 fragColor;
        
        void main() {
            fragColor = v_color;
        }
        """
        
        self.connection_shader = gpu.types.GPUShader(connection_vertex_shader, connection_fragment_shader)
        
        # SYMMETRY LINES SHADER
        symmetry_vertex_shader = """
        in vec3 pos;
        
        uniform mat4 ModelViewProjectionMatrix;
        uniform vec4 symmetry_color;
        
        out vec4 v_color;
        
        void main() {
            gl_Position = ModelViewProjectionMatrix * vec4(pos, 1.0);
            v_color = symmetry_color;
        }
        """
        
        symmetry_fragment_shader = """
        in vec4 v_color;
        out vec4 fragColor;
        
        void main() {
            fragColor = v_color;
        }
        """
        
        self.symmetry_shader = gpu.types.GPUShader(symmetry_vertex_shader, symmetry_fragment_shader)
    
    def _cleanup_shaders(self):
        """Bereinigt GPU-Shader"""
        shaders = [self.brush_shader, self.stroke_shader, 
                  self.connection_shader, self.symmetry_shader]
        
        for shader in shaders:
            if shader:
                del shader
        
        self.brush_shader = None
        self.stroke_shader = None
        self.connection_shader = None
        self.symmetry_shader = None
    
    def _cleanup_batches(self):
        """Bereinigt GPU-Batches"""
        batches = [self.brush_batch, self.stroke_batch, 
                  self.connection_batch, self.symmetry_batch]
        
        for batch in batches:
            if batch:
                del batch
        
        self.brush_batch = None
        self.stroke_batch = None
        self.connection_batch = None
        self.symmetry_batch = None
    
    def _register_draw_handler(self):
        """Registriert Draw-Handler für GPU-Rendering"""
        self.draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_callback, 
            (), 
            'WINDOW', 
            'POST_VIEW'
        )
    
    def _draw_callback(self):
        """Haupt Draw-Callback für GPU-Rendering"""
        if not self.is_active:
            return
        
        # Performance-Messung starten
        frame_start = time.time()
        
        try:
            # GPU-State für Blending
            gpu.state.blend_set('ALPHA')
            gpu.state.depth_test_set('LESS_EQUAL')
            
            context = bpy.context
            
            # Symmetrie-Linien rendern (falls aktiv)
            if self._should_draw_symmetry(context):
                self._draw_symmetry_lines(context)
            
            # Stroke Buffer rendern 
            if self.stroke_buffer.has_data():
                self._draw_stroke_buffer(context)
            
            # Auto-Connect Linien rendern
            if self._should_draw_connections(context):
                self._draw_connection_preview(context)
            
            # Brush Preview rendern (falls Maus in Viewport)
            if self._should_draw_brush(context):
                self._draw_brush_preview(context)
            
            # GPU-State zurücksetzen
            gpu.state.blend_set('NONE')
            gpu.state.depth_test_set('NONE')
            
        except Exception as e:
            print(f"❌ Paint Overlay Draw Error: {e}")
        
        # Performance-Messung beenden
        frame_time = time.time() - frame_start
        self.fps_counter.update(frame_time)
        
        # Performance-Warning bei niedrigen FPS
        if self.fps_counter.get_fps() < 30:
            self.perf_monitor.register_performance_warning("paint_overlay_low_fps")
    
    def _draw_brush_preview(self, context):
        """Rendert Brush-Preview"""
        if not self.brush_shader:
            return
        
        props = context.scene.chain_tool_properties
        region = context.region
        rv3d = context.region_data
        
        # Brush-Position aus Maus-Koordinaten berechnen
        brush_center = self._get_brush_world_position(context)
        if not brush_center:
            return
        
        # Brush-Geometrie erstellen (falls nötig)
        if not self.brush_batch:
            self._create_brush_batch()
        
        # Shader-Uniforms setzen
        self.brush_shader.bind()
        self.brush_shader.uniform_float("brush_center", brush_center)
        self.brush_shader.uniform_float("brush_size", props.paint_brush_size)
        self.brush_shader.uniform_float("brush_opacity", 0.3)
        self.brush_shader.uniform_vector_float("brush_color", (1.0, 0.5, 0.0, 1.0))
        self.brush_shader.uniform_float("brush_hardness", 2.0)
        
        # Matrix setzen
        self.brush_shader.uniform_float("ModelViewProjectionMatrix", 
                                       rv3d.perspective_matrix)
        
        # Rendern
        self.brush_batch.draw(self.brush_shader)
    
    def _draw_stroke_buffer(self, context):
        """Rendert Stroke Buffer"""
        if not self.stroke_shader or not self.stroke_buffer.has_data():
            return
        
        # Stroke-Daten vom Buffer holen
        stroke_data = self.stroke_buffer.get_render_data()
        
        # Batch erstellen (oder aktualisieren)
        if stroke_data:
            positions = [point['position'] for point in stroke_data]
            sizes = [point['size'] for point in stroke_data]
            colors = [point['color'] for point in stroke_data]
            
            stroke_batch = batch_for_shader(
                self.stroke_shader,
                'POINTS',
                {
                    "pos": positions,
                    "point_size": sizes,
                    "point_color": colors
                }
            )
            
            # Shader-Uniforms setzen
            self.stroke_shader.bind()
            self.stroke_shader.uniform_float("ModelViewProjectionMatrix", 
                                            context.region_data.perspective_matrix)
            self.stroke_shader.uniform_float("global_scale", 1.0)
            
            # Rendern
            stroke_batch.draw(self.stroke_shader)
    
    def _draw_connection_preview(self, context):
        """Rendert Auto-Connect Vorschau"""
        if not self.connection_shader:
            return
        
        props = context.scene.chain_tool_properties
        if not props.paint_auto_connect:
            return
        
        # Connection-Daten vom Buffer holen
        connection_data = self.connection_buffer.get_render_data()
        
        if connection_data:
            positions = []
            colors = []
            
            for connection in connection_data:
                # Linie als zwei Punkte
                positions.extend([connection['start'], connection['end']])
                colors.extend([connection['color'], connection['color']])
            
            if positions:
                connection_batch = batch_for_shader(
                    self.connection_shader,
                    'LINES',
                    {
                        "pos": positions,
                        "line_color": colors
                    }
                )
                
                # Shader-Uniforms setzen
                self.connection_shader.bind()
                self.connection_shader.uniform_float("ModelViewProjectionMatrix", 
                                                   context.region_data.perspective_matrix)
                
                # Rendern
                connection_batch.draw(self.connection_shader)
    
    def _draw_symmetry_lines(self, context):
        """Rendert Symmetrie-Linien"""
        if not self.symmetry_shader:
            return
        
        props = context.scene.chain_tool_properties
        if not props.paint_use_symmetry:
            return
        
        # Symmetrie-Geometrie erstellen
        symmetry_lines = self.symmetry_visualizer.get_symmetry_lines(context, props)
        
        if symmetry_lines:
            symmetry_batch = batch_for_shader(
                self.symmetry_shader,
                'LINES',
                {"pos": symmetry_lines}
            )
            
            # Shader-Uniforms setzen
            self.symmetry_shader.bind()
            self.symmetry_shader.uniform_float("ModelViewProjectionMatrix", 
                                             context.region_data.perspective_matrix)
            self.symmetry_shader.uniform_vector_float("symmetry_color", (0.0, 1.0, 1.0, 0.5))
            
            # Rendern
            symmetry_batch.draw(self.symmetry_shader)
    
    def _create_brush_batch(self):
        """Erstellt Brush-Geometrie Batch"""
        # Kreisförmige Brush-Geometrie
        segments = 32
        vertices = []
        uvs = []
        indices = []
        
        # Center vertex
        vertices.append(Vector((0, 0, 0)))
        uvs.append(Vector((0.5, 0.5)))
        
        # Ring vertices
        for i in range(segments):
            angle = (i / segments) * 2.0 * 3.14159
            x = np.cos(angle)
            y = np.sin(angle)
            
            vertices.append(Vector((x, y, 0)))
            uvs.append(Vector((x * 0.5 + 0.5, y * 0.5 + 0.5)))
            
            # Triangles
            next_i = (i + 1) % segments
            indices.extend([0, i + 1, next_i + 1])
        
        self.brush_batch = batch_for_shader(
            self.brush_shader,
            'TRIS',
            {
                "pos": vertices,
                "uv": uvs
            },
            indices=indices
        )
    
    def _get_brush_world_position(self, context):
        """Berechnet Brush-Position in Weltkoordinaten"""
        region = context.region
        rv3d = context.region_data
        
        # Surface Snapping (falls aktiviert)
        props = context.scene.chain_tool_properties
        if props.paint_snap_mode != 'NONE':
            return self._get_snapped_position(context)
        
        # Fallback: Position auf Z=0 Ebene
        mouse_coord = self.mouse_pos
        if mouse_coord:
            # Mouse zu 3D-Koordinaten
            origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse_coord)
            direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse_coord)
            
            # Intersection mit Z=0 Ebene
            if direction.z != 0:
                t = -origin.z / direction.z
                return origin + direction * t
        
        return None
    
    def _get_snapped_position(self, context):
        """Berechnet Surface-Snapped Position"""
        props = context.scene.chain_tool_properties
        target_obj = props.target_object
        
        if not target_obj:
            return None
        
        # Raycast zum Target-Object
        region = context.region
        rv3d = context.region_data
        mouse_coord = self.mouse_pos
        
        if mouse_coord:
            origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse_coord)
            direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse_coord)
            
            # Matrix für Object-Space
            mat_inv = target_obj.matrix_world.inverted()
            origin_local = mat_inv @ origin
            direction_local = mat_inv.to_3x3() @ direction
            
            # Raycast
            depsgraph = context.evaluated_depsgraph_get()
            obj_eval = target_obj.evaluated_get(depsgraph)
            
            success, location, normal, face_index = obj_eval.ray_cast(
                origin_local, direction_local
            )
            
            if success:
                # Zurück zu Weltkoordinaten
                world_location = target_obj.matrix_world @ location
                
                # Normal-Offset anwenden
                if props.paint_use_normal_offset:
                    world_normal = target_obj.matrix_world.to_3x3() @ normal
                    world_location += world_normal * props.paint_normal_offset_distance
                
                return world_location
        
        return None
    
    def _should_draw_brush(self, context):
        """Prüft ob Brush Preview gezeichnet werden soll"""
        return (self.mouse_pos and 
                context.region and 
                context.region_data and
                context.area.type == 'VIEW_3D')
    
    def _should_draw_connections(self, context):
        """Prüft ob Connection Preview gezeichnet werden soll"""
        props = context.scene.chain_tool_properties
        return (props.paint_auto_connect and 
                self.connection_buffer.has_data())
    
    def _should_draw_symmetry(self, context):
        """Prüft ob Symmetrie-Linien gezeichnet werden sollen"""
        props = context.scene.chain_tool_properties
        return props.paint_use_symmetry
    
    def _update_viewport_settings(self, context):
        """Aktualisiert Viewport-spezifische Einstellungen"""
        # Viewport-Settings für bessere Performance
        if context.space_data:
            # Temporär Viewport-Shading für Paint-Mode optimieren
            pass
    
    def update_mouse_position(self, mouse_x, mouse_y):
        """Aktualisiert Maus-Position für Brush Preview"""
        self.mouse_pos = Vector((mouse_x, mouse_y))
    
    def update_brush_size(self, size):
        """Aktualisiert Brush-Größe"""
        self.brush_size = max(0.1, size)
    
    def add_stroke_point(self, position, size=None, color=None):
        """Fügt Punkt zum Stroke Buffer hinzu"""
        if size is None:
            size = self.brush_size
        if color is None:
            color = (1.0, 0.5, 0.0, 0.8)  # Orange
        
        self.stroke_buffer.add_point(position, size, color)
    
    def add_connection_preview(self, start_pos, end_pos, connection_type='POTENTIAL'):
        """Fügt Connection-Vorschau hinzu"""
        color_map = {
            'POTENTIAL': (0.0, 1.0, 0.0, 0.5),  # Grün
            'ACTIVE': (1.0, 1.0, 0.0, 0.8),     # Gelb
            'BLOCKED': (1.0, 0.0, 0.0, 0.5)     # Rot
        }
        
        color = color_map.get(connection_type, (0.5, 0.5, 0.5, 0.5))
        self.connection_buffer.add_connection(start_pos, end_pos, color)
    
    def clear_stroke_buffer(self):
        """Leert Stroke Buffer"""
        self.stroke_buffer.clear()
    
    def clear_connection_buffer(self):
        """Leert Connection Buffer"""
        self.connection_buffer.clear()
    
    def get_performance_stats(self):
        """Gibt Performance-Statistiken zurück"""
        return {
            'fps': self.fps_counter.get_fps(),
            'frame_time_ms': self.fps_counter.get_avg_frame_time() * 1000,
            'stroke_points': self.stroke_buffer.get_point_count(),
            'connections': self.connection_buffer.get_connection_count(),
            'memory_usage_mb': self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self):
        """Schätzt Speicherverbrauch in MB"""
        point_size = 32  # bytes per point (position + size + color)
        connection_size = 40  # bytes per connection
        
        total_bytes = (self.stroke_buffer.get_point_count() * point_size + 
                      self.connection_buffer.get_connection_count() * connection_size)
        
        return total_bytes / (1024 * 1024)


class StrokeBuffer:
    """Optimierter Buffer für Stroke-Daten"""
    
    def __init__(self, max_points=10000):
        self.max_points = max_points
        self.points = []
        self.needs_update = True
    
    def add_point(self, position, size, color):
        """Fügt Punkt hinzu"""
        point_data = {
            'position': Vector(position),
            'size': size,
            'color': color,
            'timestamp': time.time()
        }
        
        self.points.append(point_data)
        
        # Memory-Management
        if len(self.points) > self.max_points:
            self.points.pop(0)  # Ältesten Punkt entfernen
        
        self.needs_update = True
    
    def get_render_data(self):
        """Gibt Render-Daten zurück"""
        if self.needs_update:
            self.needs_update = False
        
        return self.points
    
    def clear(self):
        """Leert Buffer"""
        self.points.clear()
        self.needs_update = True
    
    def has_data(self):
        """Prüft ob Daten vorhanden"""
        return len(self.points) > 0
    
    def get_point_count(self):
        """Gibt Anzahl Punkte zurück"""
        return len(self.points)


class ConnectionBuffer:
    """Buffer für Connection-Preview Daten"""
    
    def __init__(self, max_connections=1000):
        self.max_connections = max_connections
        self.connections = []
        self.needs_update = True
    
    def add_connection(self, start_pos, end_pos, color):
        """Fügt Connection hinzu"""
        connection_data = {
            'start': Vector(start_pos),
            'end': Vector(end_pos),
            'color': color,
            'timestamp': time.time()
        }
        
        self.connections.append(connection_data)
        
        # Memory-Management
        if len(self.connections) > self.max_connections:
            self.connections.pop(0)
        
        self.needs_update = True
    
    def get_render_data(self):
        """Gibt Render-Daten zurück"""
        return self.connections
    
    def clear(self):
        """Leert Buffer"""
        self.connections.clear()
        self.needs_update = True
    
    def has_data(self):
        """Prüft ob Daten vorhanden"""
        return len(self.connections) > 0
    
    def get_connection_count(self):
        """Gibt Anzahl Connections zurück"""
        return len(self.connections)


class FPSCounter:
    """FPS-Counter für Performance-Monitoring"""
    
    def __init__(self, sample_count=60):
        self.sample_count = sample_count
        self.frame_times = []
        self.last_time = time.time()
    
    def update(self, frame_time):
        """Aktualisiert FPS-Messung"""
        self.frame_times.append(frame_time)
        
        if len(self.frame_times) > self.sample_count:
            self.frame_times.pop(0)
    
    def get_fps(self):
        """Berechnet aktuelle FPS"""
        if not self.frame_times:
            return 0
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / max(avg_frame_time, 0.001)  # Avoid division by zero
    
    def get_avg_frame_time(self):
        """Gibt durchschnittliche Frame-Zeit zurück"""
        if not self.frame_times:
            return 0
        
        return sum(self.frame_times) / len(self.frame_times)
    
    def reset(self):
        """Setzt Counter zurück"""
        self.frame_times.clear()


class AutoConnectVisualizer:
    """Visualisierung für Auto-Connect System"""
    
    def __init__(self):
        self.active_connections = []
    
    def update_connections(self, stroke_points, connection_distance):
        """Aktualisiert Connection-Visualisierung"""
        # Implementation würde hier Auto-Connect Algorithmus aufrufen
        # und potentielle Verbindungen berechnen
        pass


class SymmetryVisualizer:
    """Visualisierung für Symmetrie-System"""
    
    def get_symmetry_lines(self, context, props):
        """Berechnet Symmetrie-Linien für Anzeige"""
        if not props.paint_use_symmetry:
            return []
        
        target_obj = props.target_object
        if not target_obj:
            return []
        
        # Bounding Box des Objekts holen
        bbox = [target_obj.matrix_world @ Vector(corner) for corner in target_obj.bound_box]
        
        min_co = Vector((min(v.x for v in bbox), min(v.y for v in bbox), min(v.z for v in bbox)))
        max_co = Vector((max(v.x for v in bbox), max(v.y for v in bbox), max(v.z for v in bbox)))
        center = (min_co + max_co) / 2
        
        lines = []
        
        # Symmetrie-Linie basierend auf Achse
        if props.paint_symmetry_axis == 'X':
            # YZ-Ebene
            lines.extend([
                Vector((center.x, min_co.y, min_co.z)),
                Vector((center.x, max_co.y, min_co.z)),
                Vector((center.x, max_co.y, max_co.z)),
                Vector((center.x, min_co.y, max_co.z)),
                Vector((center.x, min_co.y, min_co.z))
            ])
        elif props.paint_symmetry_axis == 'Y':
            # XZ-Ebene
            lines.extend([
                Vector((min_co.x, center.y, min_co.z)),
                Vector((max_co.x, center.y, min_co.z)),
                Vector((max_co.x, center.y, max_co.z)),
                Vector((min_co.x, center.y, max_co.z)),
                Vector((min_co.x, center.y, min_co.z))
            ])
        elif props.paint_symmetry_axis == 'Z':
            # XY-Ebene
            lines.extend([
                Vector((min_co.x, min_co.y, center.z)),
                Vector((max_co.x, min_co.y, center.z)),
                Vector((max_co.x, max_co.y, center.z)),
                Vector((min_co.x, max_co.y, center.z)),
                Vector((min_co.x, min_co.y, center.z))
            ])
        
        return lines


# Globale Paint Overlay Instanz
_paint_overlay_system = None

def get_paint_overlay_system():
    """Gibt globale Paint Overlay System Instanz zurück"""
    global _paint_overlay_system
    if _paint_overlay_system is None:
        _paint_overlay_system = PaintOverlaySystem()
    return _paint_overlay_system

def cleanup_paint_overlay_system():
    """Bereinigt globale Paint Overlay System Instanz"""
    global _paint_overlay_system
    if _paint_overlay_system:
        _paint_overlay_system.cleanup()
        _paint_overlay_system = None
