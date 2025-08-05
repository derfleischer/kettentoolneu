"""
Paint Mode Operators for Chain Tool V4
Interactive painting system for pattern generation
"""

import bpy
import bmesh
from bpy.types import Operator
from bpy.props import FloatProperty, BoolProperty, IntProperty, EnumProperty
from mathutils import Vector, Matrix
import time
from typing import List, Dict, Optional

from ..core.state_manager import StateManager
from ..utils.debug import DebugManager
from ..ui.overlays.paint_overlay import PaintOverlay

class CHAIN_OT_enter_paint_mode(Operator):
    """Enter paint mode for pattern creation"""
    bl_idname = "chain.enter_paint_mode"
    bl_label = "Enter Paint Mode"
    bl_description = "Enter interactive paint mode"
    bl_options = {'REGISTER'}
    
    brush_size: FloatProperty(
        name="Brush Size",
        description="Initial brush size",
        default=0.05,
        min=0.01,
        max=0.2,
        subtype='DISTANCE'
    )
    
    brush_strength: FloatProperty(
        name="Brush Strength",
        description="Brush strength/opacity",
        default=1.0,
        min=0.1,
        max=1.0,
        subtype='FACTOR'
    )
    
    def __init__(self):
        self.state = StateManager()
        self.debug = DebugManager()
        self._handle = None
        self._timer = None
        
    @classmethod
    def poll(cls, context):
        return (context.active_object is not None and 
                context.active_object.type == 'MESH' and
                context.mode == 'OBJECT')
        
    def modal(self, context, event):
        # Handle paint mode events
        if event.type == 'TIMER':
            # Update overlay
            if hasattr(self, '_overlay'):
                self._overlay.update()
                
        elif event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                # Start stroke
                self._start_stroke(context, event)
                return {'RUNNING_MODAL'}
                
            elif event.value == 'RELEASE':
                # End stroke
                self._end_stroke(context)
                return {'RUNNING_MODAL'}
                
        elif event.type == 'MOUSEMOVE':
            # Update brush position
            self._update_brush(context, event)
            
            # Add point if painting
            if self._is_painting:
                self._add_stroke_point(context, event)
                
            return {'RUNNING_MODAL'}
            
        elif event.type == 'WHEELUPMOUSE':
            # Increase brush size
            self.brush_size = min(self.brush_size * 1.1, 0.2)
            self._update_overlay_settings()
            return {'RUNNING_MODAL'}
            
        elif event.type == 'WHEELDOWNMOUSE':
            # Decrease brush size
            self.brush_size = max(self.brush_size * 0.9, 0.01)
            self._update_overlay_settings()
            return {'RUNNING_MODAL'}
            
        elif event.type in {'X', 'SHIFT'} and event.value == 'PRESS':
            # Toggle symmetry
            if event.type == 'X':
                self._toggle_symmetry('X')
            return {'RUNNING_MODAL'}
            
        elif event.type == 'Z' and event.value == 'PRESS' and event.ctrl:
            # Undo last stroke
            bpy.ops.chain.undo_last_stroke()
            return {'RUNNING_MODAL'}
            
        elif event.type in {'RET', 'SPACE'} and event.value == 'PRESS':
            # Apply and exit
            self._exit_paint_mode(context)
            return {'FINISHED'}
            
        elif event.type == 'ESC':
            # Cancel and exit
            self._exit_paint_mode(context, cancel=True)
            return {'CANCELLED'}
            
        return {'RUNNING_MODAL'}
        
    def invoke(self, context, event):
        if context.area.type != 'VIEW_3D':
            self.report({'ERROR'}, "Must be in 3D Viewport")
            return {'CANCELLED'}
            
        # Initialize paint mode
        self._is_painting = False
        self._current_stroke = None
        self._symmetry_x = False
        self._symmetry_y = False
        self._symmetry_z = False
        
        # Store active object
        self._target_object = context.active_object
        self.state.set("paint_target", self._target_object.name)
        
        # Initialize paint data
        if "paint_strokes" not in self.state.data:
            self.state.set("paint_strokes", {})
            
        paint_data = self.state.get("paint_strokes")
        if self._target_object.name not in paint_data:
            paint_data[self._target_object.name] = []
            
        # Create overlay
        self._overlay = PaintOverlay()
        self._overlay.enable(context)
        self._overlay.set_brush_size(self.brush_size)
        self._overlay.set_brush_strength(self.brush_strength)
        
        # Add timer
        self._timer = context.window_manager.event_timer_add(0.01, window=context.window)
        
        # Add modal handler
        context.window_manager.modal_handler_add(self)
        
        self.report({'INFO'}, "Paint Mode: LMB to paint, Scroll to resize, ESC to exit")
        
        return {'RUNNING_MODAL'}
        
    def _start_stroke(self, context, event):
        """Start a new paint stroke"""
        self._is_painting = True
        
        # Get 3D position
        mouse_pos = (event.mouse_region_x, event.mouse_region_y)
        hit_pos = self._get_surface_position(context, mouse_pos)
        
        if hit_pos:
            # Create new stroke
            self._current_stroke = {
                'points': [],
                'radius': self.brush_size,
                'strength': self.brush_strength,
                'timestamp': time.time()
            }
            
            # Add first point
            self._add_stroke_point(context, event)
            
            # Update overlay
            self._overlay.start_stroke()
            
    def _end_stroke(self, context):
        """End current paint stroke"""
        if not self._is_painting or not self._current_stroke:
            return
            
        self._is_painting = False
        
        # Validate stroke
        if len(self._current_stroke['points']) >= 2:
            # Add to paint data
            paint_data = self.state.get("paint_strokes")
            strokes = paint_data.get(self._target_object.name, [])
            
            # Add main stroke
            strokes.append(self._current_stroke)
            
            # Add symmetry strokes if enabled
            if self._symmetry_x or self._symmetry_y or self._symmetry_z:
                sym_strokes = self._create_symmetry_strokes(self._current_stroke)
                strokes.extend(sym_strokes)
                
            paint_data[self._target_object.name] = strokes
            self.state.set("paint_strokes", paint_data)
            
            self.debug.log(f"Stroke added with {len(self._current_stroke['points'])} points")
            
        # Clear current stroke
        self._current_stroke = None
        
        # Update overlay
        self._overlay.end_stroke()
        
    def _add_stroke_point(self, context, event):
        """Add point to current stroke"""
        if not self._current_stroke:
            return
            
        mouse_pos = (event.mouse_region_x, event.mouse_region_y)
        hit_pos = self._get_surface_position(context, mouse_pos)
        
        if hit_pos:
            # Check minimum distance from last point
            min_distance = self.brush_size * 0.2
            
            if self._current_stroke['points']:
                last_pos = Vector(self._current_stroke['points'][-1]['location'])
                if (hit_pos - last_pos).length < min_distance:
                    return
                    
            # Add point
            point_data = {
                'location': hit_pos.copy(),
                'normal': self._get_surface_normal(context, hit_pos),
                'pressure': self._get_pressure(event),
                'time': time.time()
            }
            
            self._current_stroke['points'].append(point_data)
            
            # Update overlay
            self._overlay.add_stroke_point(hit_pos)
            
    def _get_surface_position(self, context, mouse_pos) -> Optional[Vector]:
        """Get 3D position on surface from mouse coordinates"""
        region = context.region
        rv3d = context.region_data
        
        # Get ray from mouse
        view_vector = bpy_extras.view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse_pos)
        ray_origin = bpy_extras.view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse_pos)
        
        # Cast ray
        depsgraph = context.evaluated_depsgraph_get()
        
        # Use evaluated object
        obj_eval = self._target_object.evaluated_get(depsgraph)
        
        success, location, normal, index = obj_eval.ray_cast(
            ray_origin,
            view_vector,
            distance=1000.0
        )
        
        if success:
            # Transform to world space
            world_loc = self._target_object.matrix_world @ location
            return world_loc
            
        return None
        
    def _get_surface_normal(self, context, position: Vector) -> Vector:
        """Get surface normal at position"""
        # Find closest point on mesh
        obj = self._target_object
        
        # Convert to local space
        local_pos = obj.matrix_world.inverted() @ position
        
        # Find closest polygon
        mesh = obj.data
        min_dist = float('inf')
        closest_normal = Vector((0, 0, 1))
        
        for poly in mesh.polygons:
            center = Vector()
            for vert_idx in poly.vertices:
                center += mesh.vertices[vert_idx].co
            center /= len(poly.vertices)
            
            dist = (center - local_pos).length
            if dist < min_dist:
                min_dist = dist
                closest_normal = poly.normal
                
        # Transform to world space
        world_normal = obj.matrix_world.to_3x3() @ closest_normal
        return world_normal.normalized()
        
    def _get_pressure(self, event) -> float:
        """Get pen pressure or simulate it"""
        # Check if using tablet
        if hasattr(event, 'pressure'):
            return event.pressure
            
        # Simulate pressure based on speed
        # In production, track mouse speed
        return 1.0
        
    def _update_brush(self, context, event):
        """Update brush position and preview"""
        mouse_pos = (event.mouse_region_x, event.mouse_region_y)
        hit_pos = self._get_surface_position(context, mouse_pos)
        
        if hit_pos:
            self._overlay.set_brush_position(hit_pos)
            
    def _toggle_symmetry(self, axis: str):
        """Toggle symmetry for axis"""
        if axis == 'X':
            self._symmetry_x = not self._symmetry_x
            self._overlay.set_symmetry_x(self._symmetry_x)
        elif axis == 'Y':
            self._symmetry_y = not self._symmetry_y
            self._overlay.set_symmetry_y(self._symmetry_y)
        elif axis == 'Z':
            self._symmetry_z = not self._symmetry_z
            self._overlay.set_symmetry_z(self._symmetry_z)
            
    def _create_symmetry_strokes(self, stroke: Dict) -> List[Dict]:
        """Create mirrored strokes for symmetry"""
        sym_strokes = []
        
        # X symmetry
        if self._symmetry_x:
            x_stroke = {
                'points': [],
                'radius': stroke['radius'],
                'strength': stroke['strength'],
                'timestamp': stroke['timestamp']
            }
            
            for point in stroke['points']:
                loc = Vector(point['location'])
                loc.x = -loc.x
                
                x_point = point.copy()
                x_point['location'] = loc
                x_stroke['points'].append(x_point)
                
            sym_strokes.append(x_stroke)
            
        # Y symmetry
        if self._symmetry_y:
            y_stroke = {
                'points': [],
                'radius': stroke['radius'],
                'strength': stroke['strength'],
                'timestamp': stroke['timestamp']
            }
            
            for point in stroke['points']:
                loc = Vector(point['location'])
                loc.y = -loc.y
                
                y_point = point.copy()
                y_point['location'] = loc
                y_stroke['points'].append(y_point)
                
            sym_strokes.append(y_stroke)
            
        # XY symmetry
        if self._symmetry_x and self._symmetry_y:
            xy_stroke = {
                'points': [],
                'radius': stroke['radius'],
                'strength': stroke['strength'],
                'timestamp': stroke['timestamp']
            }
            
            for point in stroke['points']:
                loc = Vector(point['location'])
                loc.x = -loc.x
                loc.y = -loc.y
                
                xy_point = point.copy()
                xy_point['location'] = loc
                xy_stroke['points'].append(xy_point)
                
            sym_strokes.append(xy_stroke)
            
        return sym_strokes
        
    def _update_overlay_settings(self):
        """Update overlay with current settings"""
        if hasattr(self, '_overlay'):
            self._overlay.set_brush_size(self.brush_size)
            self._overlay.set_brush_strength(self.brush_strength)
            
    def _exit_paint_mode(self, context, cancel=False):
        """Exit paint mode"""
        # Remove timer
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
            
        # Disable overlay
        if hasattr(self, '_overlay'):
            self._overlay.disable()
            
        # Clear state
        self.state.set("paint_target", None)
        
        if cancel:
            self.report({'INFO'}, "Paint mode cancelled")
        else:
            # Count strokes
            paint_data = self.state.get("paint_strokes", {})
            strokes = paint_data.get(self._target_object.name, [])
            self.report({'INFO'}, f"Paint mode ended. {len(strokes)} strokes painted.")
            
    def cancel(self, context):
        """Handle operator cancel"""
        self._exit_paint_mode(context, cancel=True)


class CHAIN_OT_exit_paint_mode(Operator):
    """Exit paint mode"""
    bl_idname = "chain.exit_paint_mode"
    bl_label = "Exit Paint Mode"
    bl_description = "Exit paint mode and apply strokes"
    bl_options = {'REGISTER'}
    
    @classmethod
    def poll(cls, context):
        state = StateManager()
        return state.get("paint_target") is not None
        
    def execute(self, context):
        # This is handled by the modal operator
        self.report({'INFO'}, "Use ESC or ENTER to exit paint mode")
        return {'FINISHED'}


class CHAIN_OT_paint_stroke(Operator):
    """Paint a single stroke (for testing)"""
    bl_idname = "chain.paint_stroke"
    bl_label = "Paint Stroke"
    bl_description = "Paint a test stroke"
    bl_options = {'REGISTER', 'UNDO'}
    
    def __init__(self):
        self.state = StateManager()
        
    @classmethod
    def poll(cls, context):
        return (context.active_object is not None and 
                context.active_object.type == 'MESH')
        
    def execute(self, context):
        obj = context.active_object
        
        # Initialize paint data
        if "paint_strokes" not in self.state.data:
            self.state.set("paint_strokes", {})
            
        paint_data = self.state.get("paint_strokes")
        if obj.name not in paint_data:
            paint_data[obj.name] = []
            
        # Create test stroke
        stroke = {
            'points': [],
            'radius': 0.05,
            'strength': 1.0,
            'timestamp': time.time()
        }
        
        # Add some test points
        import random
        for i in range(10):
            # Random point on surface
            vert = random.choice(obj.data.vertices)
            point = {
                'location': (obj.matrix_world @ vert.co).copy(),
                'normal': vert.normal.copy(),
                'pressure': random.uniform(0.5, 1.0),
                'time': time.time()
            }
            stroke['points'].append(point)
            
        # Add stroke
        paint_data[obj.name].append(stroke)
        self.state.set("paint_strokes", paint_data)
        
        self.report({'INFO'}, "Test stroke added")
        return {'FINISHED'}


class CHAIN_OT_clear_strokes(Operator):
    """Clear all paint strokes"""
    bl_idname = "chain.clear_strokes"
    bl_label = "Clear Strokes"
    bl_description = "Clear all paint strokes from object"
    bl_options = {'REGISTER', 'UNDO'}
    
    clear_all: BoolProperty(
        name="Clear All Objects",
        description="Clear strokes from all objects",
        default=False
    )
    
    def __init__(self):
        self.state = StateManager()
        
    @classmethod
    def poll(cls, context):
        state = StateManager()
        paint_data = state.get("paint_strokes", {})
        
        if context.active_object and context.active_object.type == 'MESH':
            return context.active_object.name in paint_data
        return len(paint_data) > 0
        
    def execute(self, context):
        paint_data = self.state.get("paint_strokes", {})
        
        if self.clear_all:
            # Clear all
            cleared = len(paint_data)
            self.state.set("paint_strokes", {})
            self.report({'INFO'}, f"Cleared strokes from {cleared} objects")
            
        else:
            # Clear current object
            obj = context.active_object
            if obj and obj.name in paint_data:
                stroke_count = len(paint_data[obj.name])
                del paint_data[obj.name]
                self.state.set("paint_strokes", paint_data)
                self.report({'INFO'}, f"Cleared {stroke_count} strokes")
            else:
                self.report({'INFO'}, "No strokes to clear")
                
        return {'FINISHED'}


class CHAIN_OT_undo_last_stroke(Operator):
    """Undo the last paint stroke"""
    bl_idname = "chain.undo_last_stroke"
    bl_label = "Undo Last Stroke"
    bl_description = "Remove the last painted stroke"
    bl_options = {'REGISTER', 'UNDO'}
    
    def __init__(self):
        self.state = StateManager()
        
    @classmethod
    def poll(cls, context):
        state = StateManager()
        paint_data = state.get("paint_strokes", {})
        
        if context.active_object and context.active_object.type == 'MESH':
            strokes = paint_data.get(context.active_object.name, [])
            return len(strokes) > 0
        return False
        
    def execute(self, context):
        obj = context.active_object
        paint_data = self.state.get("paint_strokes", {})
        
        if obj.name in paint_data:
            strokes = paint_data[obj.name]
            if strokes:
                # Remove last stroke
                removed = strokes.pop()
                paint_data[obj.name] = strokes
                self.state.set("paint_strokes", paint_data)
                
                self.report({'INFO'}, f"Removed stroke with {len(removed['points'])} points")
            else:
                self.report({'INFO'}, "No strokes to undo")
        else:
            self.report({'INFO'}, "No strokes to undo")
            
        return {'FINISHED'}


# Import bpy_extras for ray casting
import bpy_extras

# Register all operators
classes = [
    CHAIN_OT_enter_paint_mode,
    CHAIN_OT_exit_paint_mode,
    CHAIN_OT_paint_stroke,
    CHAIN_OT_clear_strokes,
    CHAIN_OT_undo_last_stroke
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
        
def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
