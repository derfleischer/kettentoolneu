"""
Pattern Generation Operators for Chain Tool V4
Main operators for creating and managing patterns
"""

import bpy
from bpy.types import Operator
from bpy.props import StringProperty, FloatProperty, BoolProperty, EnumProperty
import time

from ..patterns import (
    get_pattern_class, 
    get_available_patterns,
    PATTERN_CATEGORIES
)
from ..core.state_manager import StateManager
from ..utils.debug import DebugManager
from ..utils.performance import measure_time
from ..geometry.dual_layer_optimizer import DualLayerOptimizer

class CHAIN_OT_generate_pattern(Operator):
    """Generate reinforcement pattern on selected object"""
    bl_idname = "chain.generate_pattern"
    bl_label = "Generate Pattern"
    bl_description = "Generate reinforcement pattern on the selected object"
    bl_options = {'REGISTER', 'UNDO'}
    
    # Pattern type selection
    pattern_type: EnumProperty(
        name="Pattern Type",
        description="Type of pattern to generate",
        items=lambda self, context: self.get_pattern_items(context),
        default=0
    )
    
    # Common parameters
    pattern_density: FloatProperty(
        name="Density",
        description="Base pattern density",
        default=0.02,
        min=0.005,
        max=0.1,
        subtype='DISTANCE'
    )
    
    pattern_offset: FloatProperty(
        name="Surface Offset",
        description="Offset pattern from surface",
        default=0.001,
        min=0.0,
        max=0.01,
        subtype='DISTANCE'
    )
    
    use_preview: BoolProperty(
        name="Preview Only",
        description="Generate preview without applying",
        default=True
    )
    
    def __init__(self):
        self.state = StateManager()
        self.debug = DebugManager()
        
    @classmethod
    def poll(cls, context):
        return (context.active_object is not None and 
                context.active_object.type == 'MESH' and
                context.mode == 'OBJECT')
        
    def get_pattern_items(self, context):
        """Get pattern items for enum"""
        items = []
        patterns = get_available_patterns()
        
        for i, (category, pattern_list) in enumerate(patterns.items()):
            for pattern in pattern_list:
                items.append((
                    pattern,
                    pattern.replace('_', ' ').title(),
                    f"{category} pattern",
                    i * 10 + len(items)
                ))
                
        return items if items else [('NONE', 'None', 'No patterns available', 0)]
        
    def draw(self, context):
        layout = self.layout
        
        # Pattern selection
        layout.prop(self, "pattern_type")
        
        # Common parameters
        layout.separator()
        layout.prop(self, "pattern_density")
        layout.prop(self, "pattern_offset")
        
        # Pattern-specific parameters
        pattern_class = get_pattern_class(self.pattern_type)
        if pattern_class:
            # Get default params and create UI
            default_params = pattern_class().get_default_params()
            
            box = layout.box()
            box.label(text="Pattern Parameters:", icon='SETTINGS')
            
            # This is simplified - in production, create dynamic properties
            for param, value in default_params.items():
                if param in ['pattern_offset']:  # Skip common params
                    continue
                    
                row = box.row()
                if isinstance(value, bool):
                    row.label(text=param.replace('_', ' ').title())
                    row.label(text=str(value))
                elif isinstance(value, (int, float)):
                    row.label(text=param.replace('_', ' ').title())
                    row.label(text=f"{value:.3f}")
                else:
                    row.label(text=f"{param}: {value}")
                    
        # Preview option
        layout.separator()
        layout.prop(self, "use_preview")
        
    @measure_time
    def execute(self, context):
        obj = context.active_object
        
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Please select a mesh object")
            return {'CANCELLED'}
            
        try:
            # Get pattern class
            pattern_class = get_pattern_class(self.pattern_type)
            if not pattern_class:
                self.report({'ERROR'}, f"Unknown pattern type: {self.pattern_type}")
                return {'CANCELLED'}
                
            # Create pattern instance
            pattern = pattern_class()
            
            # Prepare parameters
            params = {
                'pattern_offset': self.pattern_offset
            }
            
            # Add pattern-specific parameters
            # In production, get these from dynamic properties
            default_params = pattern.get_default_params()
            
            # Override with density if applicable
            if 'cell_size' in default_params:
                params['cell_size'] = self.pattern_density
            elif 'hex_size' in default_params:
                params['hex_size'] = self.pattern_density
            elif 'point_density' in default_params:
                params['point_density'] = self.pattern_density
            elif 'base_density' in default_params:
                params['base_density'] = self.pattern_density
                
            # Generate pattern
            self.debug.log(f"Generating {self.pattern_type} pattern...")
            start_time = time.time()
            
            if self.use_preview:
                # Generate preview
                preview_obj = pattern.preview(obj, **params)
                
                if preview_obj:
                    # Store preview reference
                    self.state.set("preview_object", preview_obj.name)
                    
                    # Select preview
                    bpy.ops.object.select_all(action='DESELECT')
                    preview_obj.select_set(True)
                    context.view_layer.objects.active = preview_obj
                    
                    self.report({'INFO'}, f"Pattern preview generated in {time.time() - start_time:.2f}s")
                else:
                    self.report({'ERROR'}, "Failed to generate pattern preview")
                    return {'CANCELLED'}
                    
            else:
                # Generate and apply pattern
                result = pattern.generate_with_cache(obj, **params)
                
                if result.success:
                    # Create pattern object
                    pattern_mesh = result.to_mesh(f"{obj.name}_Pattern")
                    pattern_obj = bpy.data.objects.new(f"{obj.name}_Pattern", pattern_mesh)
                    context.collection.objects.link(pattern_obj)
                    
                    # Copy transform
                    pattern_obj.matrix_world = obj.matrix_world.copy()
                    
                    # Store reference
                    self.state.set("pattern_object", pattern_obj.name)
                    obj["chain_pattern"] = pattern_obj.name
                    
                    # Select pattern
                    bpy.ops.object.select_all(action='DESELECT')
                    pattern_obj.select_set(True)
                    context.view_layer.objects.active = pattern_obj
                    
                    self.report({'INFO'}, 
                               f"Pattern generated: {len(result.vertices)} vertices, "
                               f"{len(result.faces)} faces in {time.time() - start_time:.2f}s")
                else:
                    self.report({'ERROR'}, f"Pattern generation failed: {result.error_message}")
                    return {'CANCELLED'}
                    
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Error generating pattern: {str(e)}")
            self.debug.error(f"Pattern generation error: {str(e)}")
            return {'CANCELLED'}


class CHAIN_OT_preview_pattern(Operator):
    """Preview pattern with current settings"""
    bl_idname = "chain.preview_pattern"
    bl_label = "Preview Pattern"
    bl_description = "Preview the pattern before applying"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        return (context.active_object is not None and 
                context.active_object.type == 'MESH')
        
    def execute(self, context):
        # Use generate operator in preview mode
        bpy.ops.chain.generate_pattern(use_preview=True)
        return {'FINISHED'}


class CHAIN_OT_apply_pattern(Operator):
    """Apply previewed pattern to object"""
    bl_idname = "chain.apply_pattern"
    bl_label = "Apply Pattern"
    bl_description = "Apply the previewed pattern"
    bl_options = {'REGISTER', 'UNDO'}
    
    optimize_for_printing: BoolProperty(
        name="Optimize for 3D Printing",
        description="Optimize pattern for dual-material 3D printing",
        default=True
    )
    
    def __init__(self):
        self.state = StateManager()
        self.debug = DebugManager()
        
    @classmethod
    def poll(cls, context):
        state = StateManager()
        preview_name = state.get("preview_object")
        return preview_name and preview_name in bpy.data.objects
        
    def execute(self, context):
        preview_name = self.state.get("preview_object")
        
        if not preview_name or preview_name not in bpy.data.objects:
            self.report({'ERROR'}, "No preview to apply")
            return {'CANCELLED'}
            
        preview_obj = bpy.data.objects[preview_name]
        base_obj = context.active_object
        
        if not base_obj or base_obj == preview_obj:
            # Find base object
            for obj in context.scene.objects:
                if obj.type == 'MESH' and obj != preview_obj:
                    base_obj = obj
                    break
                    
        if not base_obj:
            self.report({'ERROR'}, "No base object found")
            return {'CANCELLED'}
            
        try:
            # Rename preview to final pattern
            pattern_obj = preview_obj
            pattern_obj.name = f"{base_obj.name}_Pattern"
            
            # Clear preview state
            self.state.set("preview_object", None)
            
            # Store as applied pattern
            self.state.set("pattern_object", pattern_obj.name)
            base_obj["chain_pattern"] = pattern_obj.name
            
            # Optimize if requested
            if self.optimize_for_printing:
                optimizer = DualLayerOptimizer()
                
                # Run optimization
                result = optimizer.optimize_layers(
                    base_obj,
                    pattern_obj,
                    shell_thickness=0.003,
                    pattern_offset=0.0005,
                    connection_radius=0.002
                )
                
                if result['success']:
                    # Create optimized objects
                    if result['connections']:
                        # Join connections with pattern
                        bpy.context.view_layer.objects.active = pattern_obj
                        for conn in result['connections']:
                            conn.select_set(True)
                        pattern_obj.select_set(True)
                        bpy.ops.object.join()
                        
                    self.report({'INFO'}, "Pattern applied and optimized for printing")
                else:
                    self.report({'WARNING'}, "Pattern applied but optimization failed")
            else:
                self.report({'INFO'}, "Pattern applied successfully")
                
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Error applying pattern: {str(e)}")
            return {'CANCELLED'}


class CHAIN_OT_clear_pattern(Operator):
    """Clear generated pattern from object"""
    bl_idname = "chain.clear_pattern"
    bl_label = "Clear Pattern"
    bl_description = "Remove generated pattern"
    bl_options = {'REGISTER', 'UNDO'}
    
    clear_preview: BoolProperty(
        name="Clear Preview",
        description="Also clear preview if exists",
        default=True
    )
    
    def __init__(self):
        self.state = StateManager()
        
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if not obj:
            return False
            
        state = StateManager()
        return (obj.get("chain_pattern") or 
                state.get("pattern_object") or
                state.get("preview_object"))
        
    def execute(self, context):
        obj = context.active_object
        removed_count = 0
        
        # Clear applied pattern
        pattern_name = obj.get("chain_pattern")
        if not pattern_name:
            pattern_name = self.state.get("pattern_object")
            
        if pattern_name and pattern_name in bpy.data.objects:
            pattern_obj = bpy.data.objects[pattern_name]
            
            # Remove object
            bpy.data.objects.remove(pattern_obj, do_unlink=True)
            removed_count += 1
            
            # Clear references
            if "chain_pattern" in obj:
                del obj["chain_pattern"]
            self.state.set("pattern_object", None)
            
        # Clear preview if requested
        if self.clear_preview:
            preview_name = self.state.get("preview_object")
            
            if preview_name and preview_name in bpy.data.objects:
                preview_obj = bpy.data.objects[preview_name]
                bpy.data.objects.remove(preview_obj, do_unlink=True)
                removed_count += 1
                
                self.state.set("preview_object", None)
                
        if removed_count > 0:
            self.report({'INFO'}, f"Cleared {removed_count} pattern object(s)")
        else:
            self.report({'INFO'}, "No patterns to clear")
            
        return {'FINISHED'}


# Register all operators
classes = [
    CHAIN_OT_generate_pattern,
    CHAIN_OT_preview_pattern,
    CHAIN_OT_apply_pattern,
    CHAIN_OT_clear_pattern
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
        
def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
