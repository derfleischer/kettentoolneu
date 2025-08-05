"""
Export Tools Operators for Chain Tool V4
Tools for exporting dual-material patterns and 3D print preparation
"""

import bpy
import bmesh
from bpy.types import Operator
from bpy.props import StringProperty, BoolProperty, FloatProperty, EnumProperty
from bpy_extras.io_utils import ExportHelper
import os
from mathutils import Vector
import math

from ..geometry.dual_layer_optimizer import DualLayerOptimizer
from ..core.state_manager import StateManager
from ..utils.debug import DebugManager
from ..utils.performance import measure_time

class CHAIN_OT_export_dual_material(Operator, ExportHelper):
    """Export pattern for dual-material 3D printing"""
    bl_idname = "chain.export_dual_material"
    bl_label = "Export Dual Material"
    bl_description = "Export TPU shell and PETG pattern separately"
    bl_options = {'PRESET'}
    
    # File settings
    filename_ext = ".stl"
    
    filter_glob: StringProperty(
        default="*.stl;*.obj",
        options={'HIDDEN'}
    )
    
    file_format: EnumProperty(
        name="Format",
        description="Export file format",
        items=[
            ('STL', "STL", "Stereolithography format"),
            ('OBJ', "OBJ", "Wavefront OBJ format"),
            ('3MF', "3MF", "3D Manufacturing Format")
        ],
        default='STL'
    )
    
    # Export options
    export_mode: EnumProperty(
        name="Export Mode",
        description="How to export the dual materials",
        items=[
            ('SEPARATE', "Separate Files", "Export TPU and PETG as separate files"),
            ('COMBINED', "Combined with Materials", "Single file with material assignments"),
            ('ASSEMBLY', "Assembly", "Export positioned for dual-extrusion printing")
        ],
        default='SEPARATE'
    )
    
    # Material settings
    shell_material: StringProperty(
        name="Shell Material",
        description="Name for shell material (TPU)",
        default="TPU_Shell"
    )
    
    pattern_material: StringProperty(
        name="Pattern Material", 
        description="Name for pattern material (PETG)",
        default="PETG_Pattern"
    )
    
    # Optimization settings
    optimize_for_printing: BoolProperty(
        name="Optimize for Printing",
        description="Apply optimizations for 3D printing",
        default=True
    )
    
    shell_thickness: FloatProperty(
        name="Shell Thickness",
        description="Thickness of TPU shell",
        default=3.0,
        min=1.0,
        max=10.0,
        subtype='DISTANCE',
        unit='LENGTH'
    )
    
    connection_diameter: FloatProperty(
        name="Connection Diameter",
        description="Diameter of connections between layers",
        default=4.0,
        min=2.0,
        max=10.0,
        subtype='DISTANCE',
        unit='LENGTH'
    )
    
    # Positioning
    separate_distance: FloatProperty(
        name="Separation Distance",
        description="Distance between parts for assembly export",
        default=10.0,
        min=0.0,
        max=50.0,
        subtype='DISTANCE',
        unit='LENGTH'
    )
    
    def __init__(self):
        self.state = StateManager()
        self.debug = DebugManager()
        self.optimizer = DualLayerOptimizer()
        
    @classmethod
    def poll(cls, context):
        # Need at least one mesh object
        return any(obj.type == 'MESH' for obj in context.selected_objects)
        
    def draw(self, context):
        layout = self.layout
        
        # File format
        layout.prop(self, "file_format")
        
        # Export mode
        layout.separator()
        layout.label(text="Export Settings:")
        layout.prop(self, "export_mode")
        
        # Material names
        box = layout.box()
        box.label(text="Material Names:")
        box.prop(self, "shell_material")
        box.prop(self, "pattern_material")
        
        # Optimization
        layout.separator()
        layout.prop(self, "optimize_for_printing")
        
        if self.optimize_for_printing:
            box = layout.box()
            box.label(text="Print Optimization:")
            box.prop(self, "shell_thickness")
            box.prop(self, "connection_diameter")
            
        # Assembly settings
        if self.export_mode == 'ASSEMBLY':
            layout.prop(self, "separate_distance")
            
    @measure_time
    def execute(self, context):
        # Get base path without extension
        base_path = os.path.splitext(self.filepath)[0]
        
        # Find shell and pattern objects
        shell_obj, pattern_obj = self._identify_objects(context)
        
        if not shell_obj:
            self.report({'ERROR'}, "No shell object found")
            return {'CANCELLED'}
            
        if not pattern_obj:
            self.report({'WARNING'}, "No pattern object found, exporting shell only")
            
        try:
            # Optimize if requested
            if self.optimize_for_printing and pattern_obj:
                self.debug.log("Optimizing for dual-material printing...")
                
                optimization_result = self.optimizer.optimize_layers(
                    shell_obj,
                    pattern_obj,
                    shell_thickness=self.shell_thickness * 0.001,  # Convert mm to m
                    pattern_offset=0.0005,
                    connection_radius=self.connection_diameter * 0.0005  # Radius from diameter
                )
                
                if optimization_result['success']:
                    # Use optimized meshes
                    if optimization_result['shell_mesh']:
                        shell_obj = optimization_result['shell_mesh']
                    if optimization_result['pattern_mesh']:
                        pattern_obj = optimization_result['pattern_mesh']
                        
                    # Handle connections
                    if optimization_result['connections']:
                        # Merge connections with pattern
                        pattern_obj = self._merge_connections(
                            pattern_obj,
                            optimization_result['connections']
                        )
                        
            # Export based on mode
            if self.export_mode == 'SEPARATE':
                exported_files = self._export_separate(
                    base_path,
                    shell_obj,
                    pattern_obj
                )
                
            elif self.export_mode == 'COMBINED':
                exported_files = self._export_combined(
                    self.filepath,
                    shell_obj,
                    pattern_obj
                )
                
            elif self.export_mode == 'ASSEMBLY':
                exported_files = self._export_assembly(
                    self.filepath,
                    shell_obj,
                    pattern_obj
                )
                
            # Report success
            if exported_files:
                self.report({'INFO'}, 
                           f"Exported {len(exported_files)} file(s): "
                           f"{', '.join(os.path.basename(f) for f in exported_files)}")
            else:
                self.report({'ERROR'}, "Export failed")
                return {'CANCELLED'}
                
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Export error: {str(e)}")
            self.debug.error(f"Export error: {str(e)}")
            return {'CANCELLED'}
            
    def _identify_objects(self, context):
        """Identify shell and pattern objects"""
        shell_obj = None
        pattern_obj = None
        
        # Check selected objects
        for obj in context.selected_objects:
            if obj.type != 'MESH':
                continue
                
            name_lower = obj.name.lower()
            
            if any(term in name_lower for term in ['shell', 'base', 'tpu']):
                shell_obj = obj
            elif any(term in name_lower for term in ['pattern', 'chain', 'petg']):
                pattern_obj = obj
                
        # If not found by name, use active object as shell
        if not shell_obj and context.active_object and context.active_object.type == 'MESH':
            shell_obj = context.active_object
            
            # Look for pattern reference
            pattern_name = shell_obj.get("chain_pattern")
            if pattern_name and pattern_name in bpy.data.objects:
                pattern_obj = bpy.data.objects[pattern_name]
                
        return shell_obj, pattern_obj
        
    def _merge_connections(self, pattern_obj, connections):
        """Merge connection objects with pattern"""
        if not connections:
            return pattern_obj
            
        # Duplicate pattern
        merged = pattern_obj.copy()
        merged.data = pattern_obj.data.copy()
        merged.name = f"{pattern_obj.name}_Merged"
        bpy.context.collection.objects.link(merged)
        
        # Select objects for joining
        bpy.ops.object.select_all(action='DESELECT')
        merged.select_set(True)
        bpy.context.view_layer.objects.active = merged
        
        for conn in connections:
            conn.select_set(True)
            
        # Join
        bpy.ops.object.join()
        
        return merged
        
    def _export_separate(self, base_path, shell_obj, pattern_obj):
        """Export as separate files"""
        exported = []
        
        # Determine extension
        ext = self._get_extension()
        
        # Export shell
        if shell_obj:
            shell_path = f"{base_path}_shell{ext}"
            if self._export_object(shell_obj, shell_path):
                exported.append(shell_path)
                
        # Export pattern
        if pattern_obj:
            pattern_path = f"{base_path}_pattern{ext}"
            if self._export_object(pattern_obj, pattern_path):
                exported.append(pattern_path)
                
        return exported
        
    def _export_combined(self, filepath, shell_obj, pattern_obj):
        """Export as single file with materials"""
        # Duplicate objects
        objects_to_export = []
        
        if shell_obj:
            shell_copy = shell_obj.copy()
            shell_copy.data = shell_obj.data.copy()
            bpy.context.collection.objects.link(shell_copy)
            
            # Assign material
            self._assign_material(shell_copy, self.shell_material)
            objects_to_export.append(shell_copy)
            
        if pattern_obj:
            pattern_copy = pattern_obj.copy()
            pattern_copy.data = pattern_obj.data.copy()
            bpy.context.collection.objects.link(pattern_copy)
            
            # Assign material
            self._assign_material(pattern_copy, self.pattern_material)
            objects_to_export.append(pattern_copy)
            
        # Select objects
        bpy.ops.object.select_all(action='DESELECT')
        for obj in objects_to_export:
            obj.select_set(True)
            
        # Export
        success = self._export_selected(filepath)
        
        # Clean up
        for obj in objects_to_export:
            bpy.data.objects.remove(obj, do_unlink=True)
            
        return [filepath] if success else []
        
    def _export_assembly(self, filepath, shell_obj, pattern_obj):
        """Export positioned for dual-extrusion"""
        # Duplicate and position objects
        objects_to_export = []
        
        if shell_obj:
            shell_copy = shell_obj.copy()
            shell_copy.data = shell_obj.data.copy()
            bpy.context.collection.objects.link(shell_copy)
            objects_to_export.append(shell_copy)
            
        if pattern_obj:
            pattern_copy = pattern_obj.copy()
            pattern_copy.data = pattern_obj.data.copy()
            bpy.context.collection.objects.link(pattern_copy)
            
            # Position pattern next to shell
            pattern_copy.location.x += self.separate_distance * 0.001  # Convert mm to m
            
            objects_to_export.append(pattern_copy)
            
        # Select objects
        bpy.ops.object.select_all(action='DESELECT')
        for obj in objects_to_export:
            obj.select_set(True)
            
        # Export
        success = self._export_selected(filepath)
        
        # Clean up
        for obj in objects_to_export:
            bpy.data.objects.remove(obj, do_unlink=True)
            
        return [filepath] if success else []
        
    def _get_extension(self):
        """Get file extension based on format"""
        if self.file_format == 'STL':
            return '.stl'
        elif self.file_format == 'OBJ':
            return '.obj'
        elif self.file_format == '3MF':
            return '.3mf'
        return '.stl'
        
    def _export_object(self, obj, filepath):
        """Export single object"""
        # Select only this object
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        
        return self._export_selected(filepath)
        
    def _export_selected(self, filepath):
        """Export selected objects based on format"""
        try:
            if self.file_format == 'STL':
                bpy.ops.export_mesh.stl(
                    filepath=filepath,
                    use_selection=True,
                    use_mesh_modifiers=True,
                    ascii=False
                )
                
            elif self.file_format == 'OBJ':
                bpy.ops.export_scene.obj(
                    filepath=filepath,
                    use_selection=True,
                    use_mesh_modifiers=True,
                    use_materials=True
                )
                
            elif self.file_format == '3MF':
                # 3MF export if available
                try:
                    bpy.ops.export_mesh.threemf(
                        filepath=filepath,
                        use_selection=True
                    )
                except:
                    # Fallback to STL
                    self.debug.warning("3MF export not available, using STL")
                    filepath = filepath.replace('.3mf', '.stl')
                    bpy.ops.export_mesh.stl(
                        filepath=filepath,
                        use_selection=True,
                        use_mesh_modifiers=True
                    )
                    
            return True
            
        except Exception as e:
            self.debug.error(f"Export failed: {str(e)}")
            return False
            
    def _assign_material(self, obj, material_name):
        """Assign material to object"""
        # Create or get material
        mat = bpy.data.materials.get(material_name)
        if not mat:
            mat = bpy.data.materials.new(material_name)
            
        # Clear existing materials
        obj.data.materials.clear()
        
        # Assign material
        obj.data.materials.append(mat)


class CHAIN_OT_separate_by_material(Operator):
    """Separate mesh by material for multi-material printing"""
    bl_idname = "chain.separate_by_material"
    bl_label = "Separate by Material"
    bl_description = "Split object into separate objects by material"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (obj and obj.type == 'MESH' and 
                len(obj.data.materials) > 1)
                
    def execute(self, context):
        obj = context.active_object
        
        # Duplicate object for each material
        separated_objects = []
        
        for mat_idx, mat in enumerate(obj.data.materials):
            if not mat:
                continue
                
            # Duplicate object
            new_obj = obj.copy()
            new_obj.data = obj.data.copy()
            new_obj.name = f"{obj.name}_{mat.name}"
            context.collection.objects.link(new_obj)
            
            # Remove faces with different materials
            bm = bmesh.new()
            bm.from_mesh(new_obj.data)
            
            faces_to_remove = [f for f in bm.faces if f.material_index != mat_idx]
            bmesh.ops.delete(bm, geom=faces_to_remove, context='FACES')
            
            bm.to_mesh(new_obj.data)
            bm.free()
            
            # Clean up materials
            new_obj.data.materials.clear()
            new_obj.data.materials.append(mat)
            
            separated_objects.append(new_obj)
            
        # Select separated objects
        bpy.ops.object.select_all(action='DESELECT')
        for new_obj in separated_objects:
            new_obj.select_set(True)
            
        self.report({'INFO'}, f"Separated into {len(separated_objects)} objects")
        
        return {'FINISHED'}


class CHAIN_OT_prepare_for_print(Operator):
    """Prepare model for 3D printing"""
    bl_idname = "chain.prepare_for_print"
    bl_label = "Prepare for Print"
    bl_description = "Apply common preparations for 3D printing"
    bl_options = {'REGISTER', 'UNDO'}
    
    apply_modifiers: BoolProperty(
        name="Apply Modifiers",
        description="Apply all modifiers",
        default=True
    )
    
    remove_doubles: BoolProperty(
        name="Remove Doubles",
        description="Merge duplicate vertices",
        default=True
    )
    
    merge_distance: FloatProperty(
        name="Merge Distance",
        description="Maximum distance for merging vertices",
        default=0.0001,
        min=0.0,
        max=0.01,
        subtype='DISTANCE'
    )
    
    fix_normals: BoolProperty(
        name="Fix Normals",
        description="Recalculate normals facing outside",
        default=True
    )
    
    make_manifold: BoolProperty(
        name="Make Manifold",
        description="Ensure mesh is manifold (watertight)",
        default=True
    )
    
    triangulate: BoolProperty(
        name="Triangulate",
        description="Convert all faces to triangles",
        default=False
    )
    
    scale_to_size: BoolProperty(
        name="Scale to Size",
        description="Scale object to specific size",
        default=False
    )
    
    target_size: FloatProperty(
        name="Target Size",
        description="Target size in mm",
        default=100.0,
        min=1.0,
        max=1000.0,
        subtype='DISTANCE',
        unit='LENGTH'
    )
    
    @classmethod
    def poll(cls, context):
        return (context.active_object is not None and 
                context.active_object.type == 'MESH')
                
    def execute(self, context):
        obj = context.active_object
        
        # Apply modifiers
        if self.apply_modifiers:
            self._apply_all_modifiers(obj)
            
        # Enter edit mode for mesh operations
        bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(obj.data)
        
        # Remove doubles
        if self.remove_doubles:
            before = len(bm.verts)
            bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=self.merge_distance)
            removed = before - len(bm.verts)
            if removed > 0:
                self.report({'INFO'}, f"Removed {removed} duplicate vertices")
                
        # Fix normals
        if self.fix_normals:
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            
        # Make manifold
        if self.make_manifold:
            # Remove loose vertices and edges
            loose_verts = [v for v in bm.verts if not v.link_edges]
            loose_edges = [e for e in bm.edges if not e.link_faces]
            
            if loose_verts or loose_edges:
                bmesh.ops.delete(bm, geom=loose_verts + loose_edges)
                self.report({'INFO'}, 
                           f"Removed {len(loose_verts)} loose vertices "
                           f"and {len(loose_edges)} loose edges")
                           
            # Fill holes
            holes_filled = self._fill_holes(bm)
            if holes_filled:
                self.report({'INFO'}, f"Filled {holes_filled} holes")
                
        # Triangulate
        if self.triangulate:
            bmesh.ops.triangulate(bm, faces=bm.faces[:])
            
        # Update mesh
        bmesh.update_edit_mesh(obj.data)
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # Scale to size
        if self.scale_to_size:
            self._scale_to_target_size(obj, self.target_size)
            
        self.report({'INFO'}, "Print preparation complete")
        
        return {'FINISHED'}
        
    def _apply_all_modifiers(self, obj):
        """Apply all modifiers to object"""
        # Store current active
        current_active = bpy.context.view_layer.objects.active
        bpy.context.view_layer.objects.active = obj
        
        # Apply modifiers in order
        for modifier in obj.modifiers[:]:
            try:
                bpy.ops.object.modifier_apply(modifier=modifier.name)
            except:
                self.report({'WARNING'}, f"Could not apply modifier: {modifier.name}")
                
        # Restore active
        bpy.context.view_layer.objects.active = current_active
        
    def _fill_holes(self, bm):
        """Fill holes in mesh"""
        # Find boundary edges
        boundary_edges = [e for e in bm.edges if len(e.link_faces) == 1]
        
        if not boundary_edges:
            return 0
            
        # Group into loops
        edge_loops = []
        used_edges = set()
        
        for start_edge in boundary_edges:
            if start_edge in used_edges:
                continue
                
            # Trace loop
            loop_edges = [start_edge]
            used_edges.add(start_edge)
            
            current_vert = start_edge.verts[1]
            
            while True:
                # Find next boundary edge
                next_edge = None
                for edge in current_vert.link_edges:
                    if edge in boundary_edges and edge not in used_edges:
                        next_edge = edge
                        break
                        
                if not next_edge:
                    break
                    
                loop_edges.append(next_edge)
                used_edges.add(next_edge)
                current_vert = next_edge.other_vert(current_vert)
                
                # Check if loop is closed
                if current_vert == start_edge.verts[0]:
                    break
                    
            if len(loop_edges) > 2:
                edge_loops.append(loop_edges)
                
        # Fill each hole
        holes_filled = 0
        for loop in edge_loops:
            try:
                bmesh.ops.triangle_fill(bm, use_beauty=True, edges=loop)
                holes_filled += 1
            except:
                pass
                
        return holes_filled
        
    def _scale_to_target_size(self, obj, target_size_mm):
        """Scale object to target size"""
        # Get current dimensions
        dimensions = obj.dimensions
        max_dim = max(dimensions)
        
        if max_dim > 0:
            # Calculate scale factor
            # Convert mm to Blender units (default 1 unit = 1m)
            target_size_m = target_size_mm * 0.001
            scale_factor = target_size_m / max_dim
            
            # Apply scale
            obj.scale *= scale_factor
            
            # Apply scale to mesh
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            
            self.report({'INFO'}, f"Scaled to {target_size_mm}mm")


class CHAIN_OT_check_printability(Operator):
    """Check model for 3D printing issues"""
    bl_idname = "chain.check_printability"
    bl_label = "Check Printability"
    bl_description = "Check for common 3D printing issues"
    bl_options = {'REGISTER'}
    
    check_manifold: BoolProperty(
        name="Check Manifold",
        description="Check if mesh is watertight",
        default=True
    )
    
    check_normals: BoolProperty(
        name="Check Normals",
        description="Check for inverted normals",
        default=True
    )
    
    check_thin_walls: BoolProperty(
        name="Check Thin Walls",
        description="Check for walls too thin to print",
        default=True
    )
    
    min_wall_thickness: FloatProperty(
        name="Min Wall Thickness",
        description="Minimum printable wall thickness in mm",
        default=0.8,
        min=0.1,
        max=5.0,
        subtype='DISTANCE',
        unit='LENGTH'
    )
    
    check_overhangs: BoolProperty(
        name="Check Overhangs",
        description="Check for steep overhangs",
        default=True
    )
    
    max_overhang_angle: FloatProperty(
        name="Max Overhang Angle",
        description="Maximum printable overhang angle",
        default=45.0,
        min=0.0,
        max=90.0,
        subtype='ANGLE'
    )
    
    @classmethod
    def poll(cls, context):
        return (context.active_object is not None and 
                context.active_object.type == 'MESH')
                
    def execute(self, context):
        obj = context.active_object
        mesh = obj.data
        
        issues = []
        
        # Check manifold
        if self.check_manifold:
            manifold_issues = self._check_manifold(mesh)
            if manifold_issues:
                issues.extend(manifold_issues)
                
        # Check normals
        if self.check_normals:
            normal_issues = self._check_normals(mesh)
            if normal_issues:
                issues.extend(normal_issues)
                
        # Check thin walls
        if self.check_thin_walls:
            thin_wall_issues = self._check_thin_walls(obj, self.min_wall_thickness * 0.001)
            if thin_wall_issues:
                issues.extend(thin_wall_issues)
                
        # Check overhangs
        if self.check_overhangs:
            overhang_issues = self._check_overhangs(obj, math.radians(self.max_overhang_angle))
            if overhang_issues:
                issues.extend(overhang_issues)
                
        # Report results
        if not issues:
            self.report({'INFO'}, "No printability issues found!")
        else:
            self.report({'WARNING'}, f"Found {len(issues)} issues:")
            for issue in issues[:10]:  # Limit to first 10
                self.report({'WARNING'}, f"  - {issue}")
                
            if len(issues) > 10:
                self.report({'WARNING'}, f"  ... and {len(issues) - 10} more")
                
        # Store issues for visualization
        obj["printability_issues"] = issues
        
        return {'FINISHED'}
        
    def _check_manifold(self, mesh):
        """Check if mesh is manifold"""
        issues = []
        
        # Check for non-manifold edges
        bm = bmesh.new()
        bm.from_mesh(mesh)
        
        non_manifold_edges = [e for e in bm.edges if len(e.link_faces) != 2]
        if non_manifold_edges:
            issues.append(f"{len(non_manifold_edges)} non-manifold edges")
            
        # Check for loose vertices
        loose_verts = [v for v in bm.verts if not v.link_edges]
        if loose_verts:
            issues.append(f"{len(loose_verts)} loose vertices")
            
        # Check for duplicate vertices
        for v1 in bm.verts:
            for v2 in bm.verts:
                if v1 != v2 and (v1.co - v2.co).length < 0.0001:
                    issues.append("Duplicate vertices found")
                    break
                    
        bm.free()
        return issues
        
    def _check_normals(self, mesh):
        """Check for inverted normals"""
        issues = []
        
        # Simple check: calculate total signed volume
        total_volume = 0.0
        
        for face in mesh.polygons:
            if len(face.vertices) == 3:
                # Triangle
                v0 = mesh.vertices[face.vertices[0]].co
                v1 = mesh.vertices[face.vertices[1]].co
                v2 = mesh.vertices[face.vertices[2]].co
                
                # Signed volume of tetrahedron with origin
                volume = v0.dot(v1.cross(v2)) / 6.0
                total_volume += volume
                
        if total_volume < 0:
            issues.append("Mesh appears to be inside-out (negative volume)")
            
        return issues
        
    def _check_thin_walls(self, obj, min_thickness):
        """Check for walls thinner than minimum"""
        issues = []
        
        # This is a simplified check
        # In production, use ray casting or distance fields
        
        mesh = obj.data
        thin_edges = 0
        
        for edge in mesh.edges:
            v1 = mesh.vertices[edge.vertices[0]]
            v2 = mesh.vertices[edge.vertices[1]]
            
            edge_length = (v1.co - v2.co).length
            
            if edge_length < min_thickness:
                thin_edges += 1
                
        if thin_edges > 0:
            issues.append(f"{thin_edges} edges shorter than minimum thickness")
            
        return issues
        
    def _check_overhangs(self, obj, max_angle):
        """Check for steep overhangs"""
        issues = []
        
        mesh = obj.data
        overhang_faces = 0
        
        # Check face normals
        for face in mesh.polygons:
            # Transform normal to world space
            world_normal = obj.matrix_world.to_3x3() @ face.normal
            
            # Check angle with build direction (Z-up)
            z_angle = math.acos(max(-1, min(1, world_normal.dot(Vector((0, 0, 1))))))
            
            # Overhang if facing too far down
            if z_angle > math.pi/2 + max_angle:
                overhang_faces += 1
                
        if overhang_faces > 0:
            issues.append(f"{overhang_faces} faces with steep overhangs")
            
        return issues


# Register all operators
classes = [
    CHAIN_OT_export_dual_material,
    CHAIN_OT_separate_by_material,
    CHAIN_OT_prepare_for_print,
    CHAIN_OT_check_printability
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
        
def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
