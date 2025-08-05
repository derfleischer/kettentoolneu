"""
Edge Tools Operators for Chain Tool V4
Edge detection and manipulation tools
"""

import bpy
import bmesh
from bpy.types import Operator
from bpy.props import FloatProperty, BoolProperty, EnumProperty, IntProperty
from mathutils import Vector
import math

from ..geometry.edge_detector import EdgeDetector
from ..patterns.edge_patterns import RimReinforcement, ContourBanding, StressEdgeLoop
from ..core.state_manager import StateManager
from ..utils.debug import DebugManager

class CHAIN_OT_detect_edges(Operator):
    """Detect and mark edges based on angle threshold"""
    bl_idname = "chain.detect_edges"
    bl_label = "Detect Edges"
    bl_description = "Detect sharp edges and boundaries"
    bl_options = {'REGISTER', 'UNDO'}
    
    angle_threshold: FloatProperty(
        name="Angle Threshold",
        description="Minimum angle to consider as sharp edge",
        default=30.0,
        min=0.0,
        max=180.0,
        subtype='ANGLE'
    )
    
    detect_boundaries: BoolProperty(
        name="Detect Boundaries",
        description="Also detect boundary edges",
        default=True
    )
    
    mark_sharp: BoolProperty(
        name="Mark Sharp",
        description="Mark detected edges as sharp",
        default=True
    )
    
    mark_seam: BoolProperty(
        name="Mark Seam",
        description="Mark detected edges as UV seams",
        default=False
    )
    
    mark_crease: BoolProperty(
        name="Mark Crease",
        description="Mark detected edges with crease",
        default=False
    )
    
    crease_weight: FloatProperty(
        name="Crease Weight",
        description="Weight for edge crease",
        default=1.0,
        min=0.0,
        max=1.0,
        subtype='FACTOR'
    )
    
    def __init__(self):
        self.edge_detector = EdgeDetector()
        self.debug = DebugManager()
        
    @classmethod
    def poll(cls, context):
        return (context.active_object is not None and 
                context.active_object.type == 'MESH' and
                context.mode in {'OBJECT', 'EDIT_MESH'})
        
    def execute(self, context):
        obj = context.active_object
        
        # Switch to object mode to access mesh data
        current_mode = context.mode
        if current_mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
            
        try:
            # Detect edges
            detected_edges = self.edge_detector.detect_edges(
                obj,
                angle_threshold=math.radians(self.angle_threshold),
                include_boundaries=self.detect_boundaries
            )
            
            if not detected_edges:
                self.report({'INFO'}, "No edges detected with current threshold")
                return {'FINISHED'}
                
            # Create bmesh for edge operations
            bm = bmesh.new()
            bm.from_mesh(obj.data)
            bm.edges.ensure_lookup_table()
            
            # Mark detected edges
            marked_count = 0
            
            for edge_indices in detected_edges:
                # Find edge in bmesh
                v1, v2 = edge_indices
                
                for edge in bm.edges:
                    if ((edge.verts[0].index == v1 and edge.verts[1].index == v2) or
                        (edge.verts[0].index == v2 and edge.verts[1].index == v1)):
                        
                        # Mark as requested
                        if self.mark_sharp:
                            edge.smooth = False
                            
                        if self.mark_seam:
                            edge.seam = True
                            
                        if self.mark_crease:
                            edge.crease = self.crease_weight
                            
                        marked_count += 1
                        break
                        
            # Update mesh
            bm.to_mesh(obj.data)
            bm.free()
            
            # Restore mode
            if current_mode != 'OBJECT':
                bpy.ops.object.mode_set(mode=current_mode.replace('_MESH', ''))
                
            self.report({'INFO'}, f"Detected and marked {marked_count} edges")
            
            # Store detected edges in object for later use
            obj["detected_edges"] = detected_edges
            
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Error detecting edges: {str(e)}")
            
            # Restore mode on error
            if context.mode != current_mode:
                bpy.ops.object.mode_set(mode=current_mode.replace('_MESH', ''))
                
            return {'CANCELLED'}


class CHAIN_OT_select_edge_loops(Operator):
    """Select edge loops from detected edges"""
    bl_idname = "chain.select_edge_loops"
    bl_label = "Select Edge Loops"
    bl_description = "Select continuous edge loops"
    bl_options = {'REGISTER', 'UNDO'}
    
    loop_type: EnumProperty(
        name="Loop Type",
        description="Type of loops to select",
        items=[
            ('DETECTED', "Detected Edges", "Use previously detected edges"),
            ('SHARP', "Sharp Edges", "Select sharp edge loops"),
            ('BOUNDARY', "Boundaries", "Select boundary loops"),
            ('SEAM', "UV Seams", "Select UV seam loops")
        ],
        default='DETECTED'
    )
    
    min_loop_length: IntProperty(
        name="Minimum Length",
        description="Minimum number of edges in loop",
        default=3,
        min=2,
        max=100
    )
    
    def __init__(self):
        self.edge_detector = EdgeDetector()
        
    @classmethod
    def poll(cls, context):
        return (context.active_object is not None and 
                context.active_object.type == 'MESH')
        
    def execute(self, context):
        obj = context.active_object
        
        # Switch to edit mode
        if context.mode != 'EDIT_MESH':
            bpy.ops.object.mode_set(mode='EDIT')
            
        # Deselect all
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.mesh.select_mode(type='EDGE')
        
        # Get edges based on type
        bm = bmesh.from_edit_mesh(obj.data)
        edges_to_process = []
        
        if self.loop_type == 'DETECTED':
            # Use stored detected edges
            detected = obj.get("detected_edges", [])
            if not detected:
                self.report({'WARNING'}, "No detected edges found. Run edge detection first.")
                return {'CANCELLED'}
                
            edges_to_process = detected
            
        elif self.loop_type == 'SHARP':
            # Find sharp edges
            for edge in bm.edges:
                if not edge.smooth:
                    edges_to_process.append((edge.verts[0].index, edge.verts[1].index))
                    
        elif self.loop_type == 'BOUNDARY':
            # Find boundary edges
            for edge in bm.edges:
                if len(edge.link_faces) == 1:
                    edges_to_process.append((edge.verts[0].index, edge.verts[1].index))
                    
        elif self.loop_type == 'SEAM':
            # Find UV seam edges
            for edge in bm.edges:
                if edge.seam:
                    edges_to_process.append((edge.verts[0].index, edge.verts[1].index))
                    
        if not edges_to_process:
            self.report({'INFO'}, "No edges found of specified type")
            return {'FINISHED'}
            
        # Extract loops
        loops = self.edge_detector.extract_edge_loops(obj, edges_to_process)
        
        # Select loops that meet minimum length
        selected_count = 0
        
        for loop in loops:
            if len(loop) >= self.min_loop_length:
                # Select edges in loop
                for i in range(len(loop)):
                    v1 = loop[i]
                    v2 = loop[(i + 1) % len(loop)]
                    
                    # Find and select edge
                    for edge in bm.edges:
                        if ((edge.verts[0].index == v1 and edge.verts[1].index == v2) or
                            (edge.verts[0].index == v2 and edge.verts[1].index == v1)):
                            edge.select = True
                            selected_count += 1
                            break
                            
        bmesh.update_edit_mesh(obj.data)
        
        self.report({'INFO'}, f"Selected {selected_count} edges in {len(loops)} loops")
        
        return {'FINISHED'}


class CHAIN_OT_mark_edges_sharp(Operator):
    """Mark selected edges as sharp"""
    bl_idname = "chain.mark_edges_sharp"
    bl_label = "Mark Sharp"
    bl_description = "Mark selected edges as sharp"
    bl_options = {'REGISTER', 'UNDO'}
    
    clear: BoolProperty(
        name="Clear Sharp",
        description="Clear sharp marking instead of setting",
        default=False
    )
    
    @classmethod
    def poll(cls, context):
        return (context.active_object is not None and 
                context.active_object.type == 'MESH' and
                context.mode == 'EDIT_MESH')
        
    def execute(self, context):
        obj = context.active_object
        bm = bmesh.from_edit_mesh(obj.data)
        
        marked_count = 0
        
        for edge in bm.edges:
            if edge.select:
                edge.smooth = self.clear
                marked_count += 1
                
        bmesh.update_edit_mesh(obj.data)
        
        action = "cleared" if self.clear else "marked sharp"
        self.report({'INFO'}, f"{marked_count} edges {action}")
        
        return {'FINISHED'}


class CHAIN_OT_generate_edge_pattern(Operator):
    """Generate pattern along edges"""
    bl_idname = "chain.generate_edge_pattern"
    bl_label = "Generate Edge Pattern"
    bl_description = "Generate reinforcement pattern along edges"
    bl_options = {'REGISTER', 'UNDO'}
    
    pattern_type: EnumProperty(
        name="Pattern Type",
        description="Type of edge pattern",
        items=[
            ('RIM', "Rim Reinforcement", "Continuous rim along edges"),
            ('CONTOUR', "Contour Bands", "Contour following bands"),
            ('STRESS', "Stress Loops", "Loops at stress points")
        ],
        default='RIM'
    )
    
    # Rim parameters
    rim_width: FloatProperty(
        name="Rim Width",
        description="Width of rim reinforcement",
        default=0.01,
        min=0.002,
        max=0.05,
        subtype='DISTANCE'
    )
    
    rim_height: FloatProperty(
        name="Rim Height",
        description="Height/thickness of rim",
        default=0.003,
        min=0.001,
        max=0.01,
        subtype='DISTANCE'
    )
    
    # Contour parameters
    band_spacing: FloatProperty(
        name="Band Spacing",
        description="Spacing between contour bands",
        default=0.02,
        min=0.005,
        max=0.1,
        subtype='DISTANCE'
    )
    
    band_width: FloatProperty(
        name="Band Width",
        description="Width of each band",
        default=0.005,
        min=0.002,
        max=0.02,
        subtype='DISTANCE'
    )
    
    # Stress parameters
    stress_threshold: FloatProperty(
        name="Stress Threshold",
        description="Minimum stress level for loops",
        default=0.6,
        min=0.0,
        max=1.0,
        subtype='FACTOR'
    )
    
    def __init__(self):
        self.state = StateManager()
        self.debug = DebugManager()
        
    @classmethod
    def poll(cls, context):
        return (context.active_object is not None and 
                context.active_object.type == 'MESH' and
                context.mode == 'OBJECT')
        
    def execute(self, context):
        obj = context.active_object
        
        try:
            # Create pattern based on type
            if self.pattern_type == 'RIM':
                pattern = RimReinforcement()
                params = {
                    'rim_width': self.rim_width,
                    'rim_height': self.rim_height,
                    'edge_selection': 'auto'
                }
                
            elif self.pattern_type == 'CONTOUR':
                pattern = ContourBanding()
                params = {
                    'band_spacing': self.band_spacing,
                    'band_width': self.band_width,
                    'follow_curvature': True
                }
                
            elif self.pattern_type == 'STRESS':
                pattern = StressEdgeLoop()
                params = {
                    'stress_threshold': self.stress_threshold,
                    'loop_width': 0.008,
                    'optimize_junctions': True
                }
                
            else:
                self.report({'ERROR'}, f"Unknown pattern type: {self.pattern_type}")
                return {'CANCELLED'}
                
            # Generate pattern
            self.debug.log(f"Generating {self.pattern_type} edge pattern...")
            
            result = pattern.generate(obj, **params)
            
            if result.success:
                # Create pattern object
                pattern_mesh = result.to_mesh(f"{obj.name}_EdgePattern")
                pattern_obj = bpy.data.objects.new(f"{obj.name}_EdgePattern", pattern_mesh)
                context.collection.objects.link(pattern_obj)
                
                # Copy transform
                pattern_obj.matrix_world = obj.matrix_world.copy()
                
                # Store reference
                obj["edge_pattern"] = pattern_obj.name
                
                # Select pattern
                bpy.ops.object.select_all(action='DESELECT')
                pattern_obj.select_set(True)
                context.view_layer.objects.active = pattern_obj
                
                self.report({'INFO'}, 
                           f"Edge pattern generated: {len(result.vertices)} vertices, "
                           f"{len(result.faces)} faces")
                           
            else:
                self.report({'ERROR'}, f"Pattern generation failed: {result.error_message}")
                return {'CANCELLED'}
                
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Error generating edge pattern: {str(e)}")
            self.debug.error(f"Edge pattern error: {str(e)}")
            return {'CANCELLED'}


# Register all operators
classes = [
    CHAIN_OT_detect_edges,
    CHAIN_OT_select_edge_loops,
    CHAIN_OT_mark_edges_sharp,
    CHAIN_OT_generate_edge_pattern
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
        
def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
