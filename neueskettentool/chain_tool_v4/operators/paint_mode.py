"""
Chain Tool V4 - Paint Mode Operators
====================================

Paint-basierte Pattern-Generierung für Chain Tool V4.
Ermöglicht manuelles "Malen" von Pattern-Bereichen auf Mesh-Oberflächen.

Features:
- Interactive Paint Mode mit Modal Operator
- 3D Surface Painting mit Ray Casting
- Brush Size und Strength Controls
- Stroke Recording und Undo System
- Real-time Visual Feedback
- Integration mit Pattern Generation System

Author: Chain Tool V4 Development Team
Version: 4.0.0
"""

import bpy
import bmesh
import gpu
import bgl
from gpu_extras.batch import batch_for_shader
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from bpy.types import Operator
from bpy.props import FloatProperty, BoolProperty, IntProperty
import bpy_extras.view3d_utils as view3d_utils
import time
from typing import List, Dict, Tuple, Optional, Any

# Import system modules
from ..core.state_manager import StateManager, StateCategory
from ..core.properties import get_chain_tool_properties
from ..utils.debug import DebugManager
from ..utils.performance import measure_time

# Initialize systems
debug = DebugManager()

class CHAIN_OT_enter_paint_mode(Operator):
    """Enter paint mode for manual pattern placement"""
    bl_idname = "chain.enter_paint_mode"
    bl_label = "Enter Paint Mode"
    bl_description = "Enter interactive paint mode for pattern placement"
    bl_options = {'REGISTER'}
    
    @classmethod
    def poll(cls, context):
        return (
            context.active_object is not None and 
            context.active_object.type == 'MESH' and
            context.mode == 'OBJECT'
        )
    
    def execute(self, context):
        try:
            # Get properties
            props = get_chain_tool_properties(context)
            if not props:
                self.report({'ERROR'}, "Chain Tool properties not found")
                return {'CANCELLED'}
            
            # Initialize paint system
            paint_system = PaintSystem()
            if not paint_system.initialize(context):
                self.report({'ERROR'}, "Failed to initialize paint system")
                return {'CANCELLED'}
            
            # Set paint mode active
            props.paint_mode_active = True
            
            # Update state manager
            state_manager = StateManager()
            state_manager.is_paint_mode = True
            state_manager.set(StateCategory.PAINT, "target_object", context.active_object.name)
            
            # Start modal paint operator
            bpy.ops.chain.paint_stroke('INVOKE_DEFAULT')
            
            self.report({'INFO'}, "Paint mode activated. Left-click and drag to paint, ESC to exit.")
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Failed to enter paint mode: {e}")
            debug.log_error(f"Paint mode activation failed: {e}")
            return {'CANCELLED'}


class CHAIN_OT_exit_paint_mode(Operator):
    """Exit paint mode"""
    bl_idname = "chain.exit_paint_mode"
    bl_label = "Exit Paint Mode"
    bl_description = "Exit paint mode and return to normal operation"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        try:
            # Get properties
            props = get_chain_tool_properties(context)
            if props:
                props.paint_mode_active = False
            
            # Update state manager
            state_manager = StateManager()
            state_manager.is_paint_mode = False
            
            # Cleanup paint system
            paint_system = PaintSystem()
            paint_system.cleanup()
            
            self.report({'INFO'}, "Paint mode deactivated")
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Failed to exit paint mode: {e}")
            return {'CANCELLED'}


class CHAIN_OT_paint_stroke(Operator):
    """Modal operator for interactive painting"""
    bl_idname = "chain.paint_stroke"
    bl_label = "Paint Stroke"
    bl_description = "Interactive painting on mesh surface"
    bl_options = {'REGISTER'}
    
    def __init__(self):
        self.state_manager = StateManager()
        self.paint_system = PaintSystem()
        self.is_painting = False
        self.current_stroke = []
        self.last_mouse_pos = Vector((0, 0))
        self.stroke_start_time = 0
        
        # Drawing handles
        self._draw_handler = None
        self._draw_event = None
    
    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'
    
    def invoke(self, context, event):
        """Start modal paint operation"""
        if context.area.type == 'VIEW_3D':
            # Initialize paint system
            if not self.paint_system.initialize(context):
                self.report({'ERROR'}, "Failed to initialize paint system")
                return {'CANCELLED'}
            
            # Add drawing handler
            self._setup_drawing(context)
            
            # Add modal handler
            context.window_manager.modal_handler_add(self)
            
            # Update cursor
            context.window.cursor_modal_set('PAINT_BRUSH')
            
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "Paint mode requires 3D Viewport")
            return {'CANCELLED'}
    
    def modal(self, context, event):
        """Handle modal paint events"""
        context.area.tag_redraw()
        
        try:
            # Handle ESC - Exit paint mode
            if event.type == 'ESC':
                return self._finish_paint_mode(context)
            
            # Handle mouse movement
            if event.type == 'MOUSEMOVE':
                self.last_mouse_pos = Vector((event.mouse_region_x, event.mouse_region_y))
                
                # Continue stroke if painting
                if self.is_painting:
                    self._add_stroke_point(context, event)
            
            # Handle left mouse button
            elif event.type == 'LEFTMOUSE':
                if event.value == 'PRESS':
                    # Start new stroke
                    self._start_stroke(context, event)
                elif event.value == 'RELEASE':
                    # End current stroke
                    self._end_stroke(context)
            
            # Handle right mouse button - Cancel current stroke
            elif event.type == 'RIGHTMOUSE' and event.value == 'PRESS':
                if self.is_painting:
                    self._cancel_stroke(context)
            
            # Handle scroll wheel - Adjust brush size
            elif event.type == 'WHEELUPMOUSE':
                self._adjust_brush_size(context, 1.1)
            elif event.type == 'WHEELDOWNMOUSE':
                self._adjust_brush_size(context, 0.9)
            
            # Handle keyboard shortcuts
            elif event.type == 'Z' and event.value == 'PRESS':
                # Undo last stroke
                bpy.ops.chain.undo_last_stroke()
            elif event.type == 'X' and event.value == 'PRESS':
                # Clear all strokes
                bpy.ops.chain.clear_strokes()
            
            return {'RUNNING_MODAL'}
            
        except Exception as e:
            debug.log_error(f"Paint modal error: {e}")
            return self._finish_paint_mode(context)
    
    def _setup_drawing(self, context):
        """Setup OpenGL drawing for paint feedback"""
        try:
            # Add drawing handler for paint visualization
            args = (context,)
            self._draw_handler = bpy.types.SpaceView3D.draw_handler_add(
                self._draw_paint_overlay, args, 'WINDOW', 'POST_VIEW'
            )
            
        except Exception as e:
            debug.log_error(f"Drawing setup failed: {e}")
    
    def _draw_paint_overlay(self, context):
        """Draw paint overlay in 3D viewport"""
        try:
            # Get properties
            props = get_chain_tool_properties(context)
            if not props:
                return
            
            # Draw brush cursor
            self._draw_brush_cursor(context, props)
            
            # Draw active strokes
            self._draw_paint_strokes(context, props)
            
            # Draw current stroke
            if self.is_painting and self.current_stroke:
                self._draw_current_stroke(context, props)
                
        except Exception as e:
            debug.log_error(f"Paint overlay drawing failed: {e}")
    
    def _draw_brush_cursor(self, context, props):
        """Draw brush cursor at mouse position"""
        try:
            # Get 3D coordinates from mouse position
            region = context.region
            rv3d = context.region_data
            
            # Create circle points for brush cursor
            import math
            segments = 32
            radius = props.paint_brush_size
            
            vertices = []
            for i in range(segments):
                angle = 2.0 * math.pi * i / segments
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                vertices.append((x, y, 0))
            
            # Draw circle using GPU module
            shader = gpu.shader.from_builtin('3D_UNIFORM_COLOR')
            batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": vertices})
            
            shader.bind()
            shader.uniform_float("color", (1.0, 1.0, 1.0, 0.5))  # White, semi-transparent
            
            # Set OpenGL state
            gpu.state.blend_set('ALPHA')
            gpu.state.line_width_set(2.0)
            
            batch.draw(shader)
            
            # Reset OpenGL state
            gpu.state.blend_set('NONE')
            gpu.state.line_width_set(1.0)
            
        except Exception as e:
            debug.log_error(f"Brush cursor drawing failed: {e}")
    
    def _draw_paint_strokes(self, context, props):
        """Draw existing paint strokes"""
        try:
            # Get paint strokes from state manager
            target_obj = context.active_object
            if not target_obj:
                return
            
            paint_strokes = self.state_manager.get(StateCategory.PAINT, "paint_strokes", {})
            
            if target_obj.name not in paint_strokes:
                return
            
            strokes = paint_strokes[target_obj.name]
            
            # Draw each stroke
            for stroke in strokes:
                points = stroke.get('points', [])
                if len(points) < 2:
                    continue
                
                # Extract positions
                positions = [point['location'] for point in points if 'location' in point]
                
                if len(positions) < 2:
                    continue
                
                # Create line batch
                shader = gpu.shader.from_builtin('3D_UNIFORM_COLOR')
                batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": positions})
                
                shader.bind()
                shader.uniform_float("color", (0.0, 0.8, 1.0, 0.8))  # Blue
                
                gpu.state.blend_set('ALPHA')
                gpu.state.line_width_set(3.0)
                
                batch.draw(shader)
                
                gpu.state.blend_set('NONE')
                gpu.state.line_width_set(1.0)
                
        except Exception as e:
            debug.log_error(f"Stroke drawing failed: {e}")
    
    def _draw_current_stroke(self, context, props):
        """Draw current stroke being painted"""
        try:
            if not self.current_stroke or len(self.current_stroke) < 2:
                return
            
            # Extract positions from current stroke
            positions = [point['location'] for point in self.current_stroke if 'location' in point]
            
            if len(positions) < 2:
                return
            
            # Draw current stroke
            shader = gpu.shader.from_builtin('3D_UNIFORM_COLOR')
            batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": positions})
            
            shader.bind()
            shader.uniform_float("color", (1.0, 0.0, 0.0, 1.0))  # Red for current stroke
            
            gpu.state.blend_set('ALPHA')
            gpu.state.line_width_set(4.0)
            
            batch.draw(shader)
            
            gpu.state.blend_set('NONE')
            gpu.state.line_width_set(1.0)
            
        except Exception as e:
            debug.log_error(f"Current stroke drawing failed: {e}")
    
    def _start_stroke(self, context, event):
        """Start a new paint stroke"""
        try:
            # Get 3D position from mouse
            hit_location = self._get_surface_hit(context, event)
            
            if hit_location:
                self.is_painting = True
                self.stroke_start_time = time.time()
                self.current_stroke = []
                
                # Add first point
                self._add_stroke_point(context, event, hit_location)
                
                debug.log_info("Started new paint stroke")
            
        except Exception as e:
            debug.log_error(f"Stroke start failed: {e}")
    
    def _add_stroke_point(self, context, event, location=None):
        """Add point to current stroke"""
        try:
            if not self.is_painting:
                return
            
            # Get 3D position
            if location is None:
                location = self._get_surface_hit(context, event)
            
            if location:
                # Get properties for point data
                props = get_chain_tool_properties(context)
                
                # Create stroke point
                point_data = {
                    'location': location,
                    'timestamp': time.time() - self.stroke_start_time,
                    'strength': props.paint_strength if props else 1.0,
                    'radius': props.paint_brush_size if props else 2.0,
                    'mouse_pos': (event.mouse_region_x, event.mouse_region_y)
                }
                
                self.current_stroke.append(point_data)
                
                # Limit stroke length for performance
                if len(self.current_stroke) > 1000:
                    self.current_stroke = self.current_stroke[-800:]  # Keep last 800 points
            
        except Exception as e:
            debug.log_error(f"Add stroke point failed: {e}")
    
    def _end_stroke(self, context):
        """End current stroke and save it"""
        try:
            if not self.is_painting or not self.current_stroke:
                return
            
            self.is_painting = False
            
            # Save stroke to state manager
            target_obj = context.active_object
            if target_obj:
                paint_strokes = self.state_manager.get(StateCategory.PAINT, "paint_strokes", {})
                
                if target_obj.name not in paint_strokes:
                    paint_strokes[target_obj.name] = []
                
                # Create stroke data
                stroke_data = {
                    'points': self.current_stroke.copy(),
                    'created': time.time(),
                    'object_name': target_obj.name,
                    'stroke_id': len(paint_strokes[target_obj.name])
                }
                
                paint_strokes[target_obj.name].append(stroke_data)
                self.state_manager.set(StateCategory.PAINT, "paint_strokes", paint_strokes)
                
                debug.log_info(f"Saved stroke with {len(self.current_stroke)} points")
            
            # Clear current stroke
            self.current_stroke = []
            
        except Exception as e:
            debug.log_error(f"End stroke failed: {e}")
    
    def _cancel_stroke(self, context):
        """Cancel current stroke without saving"""
        if self.is_painting:
            self.is_painting = False
            self.current_stroke = []
            debug.log_info("Cancelled current stroke")
    
    def _adjust_brush_size(self, context, factor):
        """Adjust brush size by factor"""
        try:
            props = get_chain_tool_properties(context)
            if props:
                new_size = props.paint_brush_size * factor
                props.paint_brush_size = max(0.1, min(20.0, new_size))
                
        except Exception as e:
            debug.log_error(f"Brush size adjustment failed: {e}")
    
    def _get_surface_hit(self, context, event) -> Optional[Vector]:
        """Get 3D surface hit position from mouse coordinates"""
        try:
            # Get viewport and region data
            region = context.region
            rv3d = context.region_data
            
            # Convert mouse coordinates to 3D ray
            mouse_pos = (event.mouse_region_x, event.mouse_region_y)
            ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse_pos)
            ray_direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse_pos)
            
            # Get target object
            target_obj = context.active_object
            if not target_obj or target_obj.type != 'MESH':
                return None
            
            # Create BVH tree for ray casting
            mesh = target_obj.data
            bvh = BVHTree.FromMesh(mesh, target_obj.matrix_world)
            
            # Perform ray cast
            hit_location, hit_normal, hit_index, hit_distance = bvh.ray_cast(ray_origin, ray_direction)
            
            return hit_location
            
        except Exception as e:
            debug.log_error(f"Surface hit detection failed: {e}")
            return None
    
    def _finish_paint_mode(self, context):
        """Finish paint mode and cleanup"""
        try:
            # Remove drawing handler
            if self._draw_handler:
                bpy.types.SpaceView3D.draw_handler_remove(self._draw_handler, 'WINDOW')
                self._draw_handler = None
            
            # Reset cursor
            context.window.cursor_modal_restore()
            
            # Exit paint mode
            bpy.ops.chain.exit_paint_mode()
            
            return {'FINISHED'}
            
        except Exception as e:
            debug.log_error(f"Paint mode finish failed: {e}")
            return {'CANCELLED'}


class CHAIN_OT_clear_strokes(Operator):
    """Clear all paint strokes"""
    bl_idname = "chain.clear_strokes"
    bl_label = "Clear Paint Strokes"
    bl_description = "Clear all paint strokes on active object"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        return context.active_object is not None
    
    def execute(self, context):
        try:
            target_obj = context.active_object
            state_manager = StateManager()
            
            paint_strokes = state_manager.get(StateCategory.PAINT, "paint_strokes", {})
            
            if target_obj.name in paint_strokes:
                stroke_count = len(paint_strokes[target_obj.name])
                paint_strokes[target_obj.name] = []
                state_manager.set(StateCategory.PAINT, "paint_strokes", paint_strokes)
                
                self.report({'INFO'}, f"Cleared {stroke_count} paint strokes")
            else:
                self.report({'INFO'}, "No strokes to clear")
            
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Failed to clear strokes: {e}")
            return {'CANCELLED'}


class CHAIN_OT_undo_last_stroke(Operator):
    """Undo the last paint stroke"""
    bl_idname = "chain.undo_last_stroke"
    bl_label = "Undo Last Stroke"
    bl_description = "Remove the last painted stroke"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        if not context.active_object:
            return False
        
        state_manager = StateManager()
        paint_strokes = state_manager.get(StateCategory.PAINT, "paint_strokes", {})
        
        if context.active_object.name in paint_strokes:
            return len(paint_strokes[context.active_object.name]) > 0
        
        return False
    
    def execute(self, context):
        try:
            target_obj = context.active_object
            state_manager = StateManager()
            
            paint_strokes = state_manager.get(StateCategory.PAINT, "paint_strokes", {})
            
            if target_obj.name in paint_strokes and paint_strokes[target_obj.name]:
                removed_stroke = paint_strokes[target_obj.name].pop()
                state_manager.set(StateCategory.PAINT, "paint_strokes", paint_strokes)
                
                point_count = len(removed_stroke.get('points', []))
                self.report({'INFO'}, f"Removed stroke with {point_count} points")
            else:
                self.report({'INFO'}, "No strokes to undo")
            
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Failed to undo stroke: {e}")
            return {'CANCELLED'}


class PaintSystem:
    """
    Paint system manager for handling paint mode initialization,
    cleanup, and state management.
    """
    
    def __init__(self):
        self.is_initialized = False
        self.target_object = None
        self.bvh_tree = None
    
    def initialize(self, context) -> bool:
        """Initialize paint system for context"""
        try:
            # Get target object
            self.target_object = context.active_object
            if not self.target_object or self.target_object.type != 'MESH':
                return False
            
            # Create BVH tree for ray casting
            mesh = self.target_object.data
            self.bvh_tree = BVHTree.FromMesh(mesh, self.target_object.matrix_world)
            
            self.is_initialized = True
            debug.log_info("Paint system initialized")
            return True
            
        except Exception as e:
            debug.log_error(f"Paint system initialization failed: {e}")
            return False
    
    def cleanup(self):
        """Cleanup paint system"""
        try:
            self.target_object = None
            self.bvh_tree = None
            self.is_initialized = False
            debug.log_info("Paint system cleaned up")
            
        except Exception as e:
            debug.log_error(f"Paint system cleanup failed: {e}")


# Collection of operator classes for registration
CLASSES = [
    CHAIN_OT_enter_paint_mode,
    CHAIN_OT_exit_paint_mode,
    CHAIN_OT_paint_stroke,
    CHAIN_OT_clear_strokes,
    CHAIN_OT_undo_last_stroke,
]

def register():
    """Register paint mode operators"""
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    print("[Paint Mode] Operators registered")

def unregister():
    """Unregister paint mode operators"""
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
    print("[Paint Mode] Operators unregistered")
