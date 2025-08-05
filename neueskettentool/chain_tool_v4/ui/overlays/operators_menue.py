"""
Chain Tool V4 - Context Menu Integration
=======================================

Provides right-click context menu integration for Chain Tool operators.
Adds convenient access to Chain Tool functions directly from the viewport.
"""

import bpy
from bpy.types import Menu

class CHAINTOOL_MT_context_menu(Menu):
    """
    Main Chain Tool context menu that appears on right-click.
    Provides quick access to primary Chain Tool operations.
    """
    bl_label = "Chain Tool"
    bl_idname = "CHAINTOOL_MT_context_menu"
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        # Check if we have properties available
        if not hasattr(scene, 'chain_tool_properties'):
            layout.label(text="Chain Tool Properties not found", icon='ERROR')
            return
        
        props = scene.chain_tool_properties
        
        # Header
        layout.label(text="Chain Tool V4", icon='MESH_ICOSPHERE')
        layout.separator()
        
        # Pattern Generation Section
        layout.label(text="Pattern Generation:", icon='MODIFIER_ON')
        
        # Surface pattern operator
        op = layout.operator(
            "chaintool.generate_surface_pattern",
            text="Generate Surface Pattern",
            icon='SURFACE_NSPHERE'
        )
        
        # Edge detection operator
        layout.operator(
            "chaintool.detect_edges",
            text="Detect Edges",
            icon='EDGESEL'
        )
        
        # Rim reinforcement operator
        layout.operator(
            "chaintool.reinforce_rim",
            text="Reinforce Rim",
            icon='MOD_SOLIDIFY'
        )
        
        layout.separator()
        
        # Interactive Tools Section
        layout.label(text="Interactive Tools:", icon='BRUSH_DATA')
        
        # Paint mode operator
        paint_op = layout.operator(
            "chaintool.paint_mode",
            text="Paint Mode (GPU)" if not props.paint_mode_active else "Exit Paint Mode",
            icon='BRUSH_SCULPT_DRAW' if not props.paint_mode_active else 'CANCEL'
        )
        
        # Triangulation operator
        layout.operator(
            "chaintool.triangulate_surface",
            text="Triangulate Surface",
            icon='MOD_TRIANGULATE'
        )
        
        layout.separator()
        
        # Export Section
        layout.label(text="Export:", icon='EXPORT')
        
        # Dual material export
        export_op = layout.operator(
            "chaintool.export_dual_material",
            text="Export Dual Material",
            icon='MATERIAL'
        )
        
        # Quick export with current settings
        layout.operator(
            "chaintool.quick_export",
            text="Quick Export",
            icon='FILE_3D'
        )
        
        layout.separator()
        
        # Utilities Section
        layout.label(text="Utilities:", icon='TOOL_SETTINGS')
        
        # Connection optimization
        layout.operator(
            "chaintool.optimize_connections",
            text="Optimize Connections",
            icon='FORCE_MAGNETIC'
        )
        
        # Gap filling
        layout.operator(
            "chaintool.fill_gaps",
            text="Fill Gaps",
            icon='MOD_MESHDEFORM'
        )


class CHAINTOOL_MT_pattern_submenu(Menu):
    """
    Submenu for specific pattern operations.
    """
    bl_label = "Pattern Types"
    bl_idname = "CHAINTOOL_MT_pattern_submenu"
    
    def draw(self, context):
        layout = self.layout
        
        # Voronoi pattern
        op = layout.operator("chaintool.generate_surface_pattern", text="Voronoi Pattern")
        op.pattern_type = 'VORONOI'
        
        # Hexagonal pattern
        op = layout.operator("chaintool.generate_surface_pattern", text="Hexagonal Grid")
        op.pattern_type = 'HEXAGONAL'
        
        # Random distribution
        op = layout.operator("chaintool.generate_surface_pattern", text="Random Distribution")
        op.pattern_type = 'RANDOM'
        
        # Core-shell pattern
        op = layout.operator("chaintool.generate_surface_pattern", text="Core-Shell Pattern")
        op.pattern_type = 'CORE_SHELL'
        
        # Gradient density
        op = layout.operator("chaintool.generate_surface_pattern", text="Gradient Density")
        op.pattern_type = 'GRADIENT_DENSITY'


class CHAINTOOL_MT_edge_submenu(Menu):
    """
    Submenu for edge-related operations.
    """
    bl_label = "Edge Tools"
    bl_idname = "CHAINTOOL_MT_edge_submenu"
    
    def draw(self, context):
        layout = self.layout
        
        # Edge detection with different algorithms
        layout.label(text="Detection Algorithms:")
        
        op = layout.operator("chaintool.detect_edges", text="Angle-Based")
        op.detection_method = 'ANGLE'
        
        op = layout.operator("chaintool.detect_edges", text="Curvature-Based")
        op.detection_method = 'CURVATURE'
        
        op = layout.operator("chaintool.detect_edges", text="Feature-Based")
        op.detection_method = 'FEATURE'
        
        layout.separator()
        
        # Rim reinforcement options
        layout.label(text="Reinforcement:")
        
        op = layout.operator("chaintool.reinforce_rim", text="Standard Rim")
        op.reinforcement_type = 'STANDARD'
        
        op = layout.operator("chaintool.reinforce_rim", text="Heavy Duty")
        op.reinforcement_type = 'HEAVY_DUTY'
        
        op = layout.operator("chaintool.reinforce_rim", text="Flexible Edge")
        op.reinforcement_type = 'FLEXIBLE'


class CHAINTOOL_MT_export_submenu(Menu):
    """
    Submenu for export operations with different presets.
    """
    bl_label = "Export Options"
    bl_idname = "CHAINTOOL_MT_export_submenu"
    
    def draw(self, context):
        layout = self.layout
        
        # Material presets
        layout.label(text="Material Presets:")
        
        op = layout.operator("chaintool.export_dual_material", text="TPU + PETG Standard")
        op.tpu_preset = 'STANDARD'
        op.petg_preset = 'STANDARD'
        
        op = layout.operator("chaintool.export_dual_material", text="Flexible + Rigid")
        op.tpu_preset = 'FLEXIBLE'
        op.petg_preset = 'RIGID'
        
        op = layout.operator("chaintool.export_dual_material", text="Medical Grade")
        op.tpu_preset = 'MEDICAL'
        op.petg_preset = 'MEDICAL'
        
        layout.separator()
        
        # Export formats
        layout.label(text="Export Formats:")
        
        op = layout.operator("chaintool.export_dual_material", text="STL (Dual Files)")
        op.export_format = 'STL_DUAL'
        
        op = layout.operator("chaintool.export_dual_material", text="3MF (Single File)")
        op.export_format = '3MF_COMBINED'
        
        op = layout.operator("chaintool.export_dual_material", text="G-Code Ready")
        op.export_format = 'GCODE_READY'


def draw_chaintool_context_menu(self, context):
    """
    Function to add Chain Tool menu to existing context menus.
    This gets appended to Blender's built-in context menus.
    """
    
    # Only show in object mode with mesh objects
    if (context.mode == 'OBJECT' and 
        context.active_object and 
        context.active_object.type == 'MESH'):
        
        self.layout.separator()
        self.layout.menu("CHAINTOOL_MT_context_menu")


def draw_chaintool_edit_context_menu(self, context):
    """
    Function to add simplified Chain Tool menu in edit mode.
    """
    
    if context.mode == 'EDIT_MESH':
        self.layout.separator()
        
        # Simplified menu for edit mode
        layout = self.layout
        layout.label(text="Chain Tool:", icon='MESH_ICOSPHERE')
        
        # Edge operations are useful in edit mode
        layout.operator("chaintool.detect_edges", text="Detect Edges", icon='EDGESEL')
        layout.operator("chaintool.reinforce_rim", text="Reinforce Selected", icon='MOD_SOLIDIFY')


def draw_chaintool_paint_context_menu(self, context):
    """
    Function to add paint-specific Chain Tool menu in sculpt/paint modes.
    """
    
    if context.mode in ['SCULPT', 'PAINT_WEIGHT', 'PAINT_VERTEX']:
        self.layout.separator()
        
        layout = self.layout
        layout.label(text="Chain Tool Paint:", icon='BRUSH_DATA')
        
        # Paint mode toggle
        scene = context.scene
        if hasattr(scene, 'chain_tool_properties'):
            props = scene.chain_tool_properties
            
            paint_op = layout.operator(
                "chaintool.paint_mode",
                text="Paint Mode (GPU)" if not props.paint_mode_active else "Exit Paint Mode",
                icon='BRUSH_SCULPT_DRAW' if not props.paint_mode_active else 'CANCEL'
            )


# Menu registration list
menu_classes = [
    CHAINTOOL_MT_context_menu,
    CHAINTOOL_MT_pattern_submenu,
    CHAINTOOL_MT_edge_submenu,
    CHAINTOOL_MT_export_submenu,
]


def register():
    """Register all menu classes and context menu handlers."""
    
    # Register menu classes
    for cls in menu_classes:
        bpy.utils.register_class(cls)
    
    # Add context menu handlers to existing Blender menus
    bpy.types.VIEW3D_MT_object_context_menu.append(draw_chaintool_context_menu)
    bpy.types.VIEW3D_MT_edit_mesh_context_menu.append(draw_chaintool_edit_context_menu)
    bpy.types.VIEW3D_MT_sculpt_context_menu.append(draw_chaintool_paint_context_menu)
    
    print("Chain Tool V4: Context menus registered")


def unregister():
    """Unregister all menu classes and context menu handlers."""
    
    # Remove context menu handlers
    try:
        bpy.types.VIEW3D_MT_sculpt_context_menu.remove(draw_chaintool_paint_context_menu)
        bpy.types.VIEW3D_MT_edit_mesh_context_menu.remove(draw_chaintool_edit_context_menu)
        bpy.types.VIEW3D_MT_object_context_menu.remove(draw_chaintool_context_menu)
    except ValueError:
        # Handler wasn't registered, ignore
        pass
    
    # Unregister menu classes
    for cls in reversed(menu_classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            # Class wasn't registered, ignore
            pass
    
    print("Chain Tool V4: Context menus unregistered")


# Development utilities for testing menus
def test_menu_registration():
    """Test if menus are properly registered."""
    
    results = {}
    
    # Test main menu
    try:
        menu = bpy.types.CHAINTOOL_MT_context_menu
        results['main_menu'] = f"✅ {menu.bl_label}"
    except AttributeError:
        results['main_menu'] = "❌ Main menu not found"
    
    # Test submenus
    submenu_ids = [
        'CHAINTOOL_MT_pattern_submenu',
        'CHAINTOOL_MT_edge_submenu', 
        'CHAINTOOL_MT_export_submenu'
    ]
    
    for menu_id in submenu_ids:
        try:
            menu = getattr(bpy.types, menu_id)
            results[menu_id] = f"✅ {menu.bl_label}"
        except AttributeError:
            results[menu_id] = f"❌ {menu_id} not found"
    
    return results


def debug_context_menu_availability():
    """Debug which context menus are available and have Chain Tool integration."""
    
    context_menus = [
        'VIEW3D_MT_object_context_menu',
        'VIEW3D_MT_edit_mesh_context_menu', 
        'VIEW3D_MT_sculpt_context_menu'
    ]
    
    for menu_name in context_menus:
        try:
            menu = getattr(bpy.types, menu_name)
            has_chaintool = any('chaintool' in str(handler).lower() for handler in menu._dyn_ui_initialize())
            status = "✅" if has_chaintool else "⚪"
            print(f"{status} {menu_name}: Chain Tool integration {'active' if has_chaintool else 'inactive'}")
        except Exception as e:
            print(f"❌ {menu_name}: Error checking - {e}")
