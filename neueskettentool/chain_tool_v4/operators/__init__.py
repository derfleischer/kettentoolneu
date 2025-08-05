"""
Operators Module for Chain Tool V4
All Blender operators for the addon
"""

# Import all operators
from .pattern_generation import (
    CHAIN_OT_generate_pattern,
    CHAIN_OT_preview_pattern,
    CHAIN_OT_apply_pattern,
    CHAIN_OT_clear_pattern
)

from .paint_mode import (
    CHAIN_OT_enter_paint_mode,
    CHAIN_OT_exit_paint_mode,
    CHAIN_OT_paint_stroke,
    CHAIN_OT_clear_strokes,
    CHAIN_OT_undo_last_stroke
)

from .edge_tools import (
    CHAIN_OT_detect_edges,
    CHAIN_OT_select_edge_loops,
    CHAIN_OT_mark_edges_sharp,
    CHAIN_OT_generate_edge_pattern
)

from .connection_tools import (
    CHAIN_OT_auto_connect,
    CHAIN_OT_connect_selected,
    CHAIN_OT_optimize_connections,
    CHAIN_OT_remove_connections
)

from .triangulation_ops import (
    CHAIN_OT_triangulate_pattern,
    CHAIN_OT_optimize_triangulation,
    CHAIN_OT_check_overlaps,
    CHAIN_OT_fix_overlaps
)

from .export_tools import (
    CHAIN_OT_export_dual_material,
    CHAIN_OT_separate_by_material,
    CHAIN_OT_prepare_for_print,
    CHAIN_OT_check_printability
)

# Operator classes list for registration
operator_classes = [
    # Pattern Generation
    CHAIN_OT_generate_pattern,
    CHAIN_OT_preview_pattern,
    CHAIN_OT_apply_pattern,
    CHAIN_OT_clear_pattern,
    
    # Paint Mode
    CHAIN_OT_enter_paint_mode,
    CHAIN_OT_exit_paint_mode,
    CHAIN_OT_paint_stroke,
    CHAIN_OT_clear_strokes,
    CHAIN_OT_undo_last_stroke,
    
    # Edge Tools
    CHAIN_OT_detect_edges,
    CHAIN_OT_select_edge_loops,
    CHAIN_OT_mark_edges_sharp,
    CHAIN_OT_generate_edge_pattern,
    
    # Connection Tools
    CHAIN_OT_auto_connect,
    CHAIN_OT_connect_selected,
    CHAIN_OT_optimize_connections,
    CHAIN_OT_remove_connections,
    
    # Triangulation
    CHAIN_OT_triangulate_pattern,
    CHAIN_OT_optimize_triangulation,
    CHAIN_OT_check_overlaps,
    CHAIN_OT_fix_overlaps,
    
    # Export
    CHAIN_OT_export_dual_material,
    CHAIN_OT_separate_by_material,
    CHAIN_OT_prepare_for_print,
    CHAIN_OT_check_printability
]

def register():
    """Register all operators"""
    for cls in operator_classes:
        bpy.utils.register_class(cls)
    
def unregister():
    """Unregister all operators"""
    for cls in reversed(operator_classes):
        bpy.utils.unregister_class(cls)

# Version info
__version__ = "4.0.0"
__all__ = [cls.__name__ for cls in operator_classes]
