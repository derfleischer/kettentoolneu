"""
Chain Tool V4 - Properties Definition
=====================================
Central property definitions for Chain Tool
"""

import bpy
from bpy.types import PropertyGroup
from bpy.props import (
    FloatProperty,
    IntProperty,
    BoolProperty,
    EnumProperty,
    PointerProperty,
    StringProperty,
    FloatVectorProperty,
    CollectionProperty
)

# Import constants with fallback
try:
    from .constants import (
        DEFAULT_SPHERE_SIZE, DEFAULT_CONNECTION_THICKNESS, DEFAULT_PATTERN_DENSITY,
        DEFAULT_SHELL_THICKNESS, DEFAULT_EDGE_ANGLE, DEFAULT_EDGE_WIDTH, DEFAULT_BRUSH_SIZE,
        MIN_SPHERE_SIZE, MAX_SPHERE_SIZE, MIN_CONNECTION_THICKNESS, MAX_CONNECTION_THICKNESS,
        MIN_PATTERN_DENSITY, MAX_PATTERN_DENSITY, MIN_SHELL_THICKNESS, MAX_SHELL_THICKNESS
    )
except ImportError:
    # Fallback defaults if constants can't be imported
    DEFAULT_SPHERE_SIZE = 2.0
    DEFAULT_CONNECTION_THICKNESS = 0.8
    DEFAULT_PATTERN_DENSITY = 5.0
    DEFAULT_SHELL_THICKNESS = 1.2
    DEFAULT_EDGE_ANGLE = 30.0
    DEFAULT_EDGE_WIDTH = 5.0
    DEFAULT_BRUSH_SIZE = 10.0
    MIN_SPHERE_SIZE = 0.1
    MAX_SPHERE_SIZE = 10.0
    MIN_CONNECTION_THICKNESS = 0.1
    MAX_CONNECTION_THICKNESS = 5.0
    MIN_PATTERN_DENSITY = 0.1
    MAX_PATTERN_DENSITY = 20.0
    MIN_SHELL_THICKNESS = 0.1
    MAX_SHELL_THICKNESS = 5.0

class ChainToolProperties(PropertyGroup):
    """Main property group for Chain Tool V4"""
    
    # Target Object
    target_object: PointerProperty(
        name="Target Object",
        description="Object to apply pattern to",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'MESH' if obj else False
    )
    
    # Pattern Type Selection
    pattern_type: EnumProperty(
        name="Pattern Type",
        description="Type of pattern to generate",
        items=[
            ('SURFACE', "Surface Pattern", "Patterns across surface", 'MESH_UVSPHERE', 0),
            ('EDGE', "Edge Pattern", "Pattern along edges", 'EDGESEL', 1),
            ('HYBRID', "Hybrid Pattern", "Combined surface and edge", 'MOD_LATTICE', 2),
            ('PAINT', "Paint Pattern", "Paint custom patterns", 'BRUSH_DATA', 3),
        ],
        default='SURFACE'
    )
    
    # Surface Pattern Sub-Type
    surface_pattern_subtype: EnumProperty(
        name="Surface Pattern",
        description="Type of surface pattern",
        items=[
            ('VORONOI', "Voronoi", "Voronoi cell pattern", 'MESH_ICOSPHERE', 0),
            ('HEXAGONAL', "Hexagonal", "Hexagon grid pattern", 'MESH_CIRCLE', 1),
            ('TRIANGULAR', "Triangular", "Triangle mesh pattern", 'MESH_CONE', 2),
            ('RANDOM', "Random", "Random distribution", 'PARTICLE_DATA', 3),
            ('ADAPTIVE', "Adaptive", "Adaptive island pattern", 'MOD_ADAPTIVE_SUBDIVISION', 4),
        ],
        default='VORONOI'
    )
    
    # Edge Pattern Sub-Type
    edge_pattern_subtype: EnumProperty(
        name="Edge Pattern",
        description="Type of edge pattern",
        items=[
            ('RIM', "Rim Reinforcement", "Reinforce edges", 'LINCURVE', 0),
            ('CONTOUR', "Contour Bands", "Contour-following bands", 'CURVE_PATH', 1),
            ('STRESS', "Stress Edges", "Stress-based reinforcement", 'FORCE_LENNARDJONES', 2),
            ('CORNER', "Corner Enhancement", "Enhance corners", 'TRIA_RIGHT', 3),
        ],
        default='RIM'
    )
    
    # Hybrid Pattern Sub-Type
    hybrid_pattern_subtype: EnumProperty(
        name="Hybrid Pattern",
        description="Type of hybrid pattern",
        items=[
            ('CORE_SHELL', "Core-Shell", "Core with shell pattern", 'SHADING_BBOX', 0),
            ('GRADIENT', "Gradient Density", "Density gradient pattern", 'BRUSH_GRADIENT', 1),
            ('ZONAL', "Zonal Reinforcement", "Zone-based reinforcement", 'IMGDISPLAY', 2),
        ],
        default='CORE_SHELL'
    )
    
    # Connection Type
    connection_type: EnumProperty(
        name="Connection Type",
        description="Type of connections between spheres",
        items=[
            ('STRAIGHT', "Straight", "Direct connections", 'LINCURVE', 0),
            ('CURVED', "Curved", "Curved connections", 'CURVE_BEZCURVE', 1),
            ('ADAPTIVE', "Adaptive", "Surface-following connections", 'MODIFIER', 2),
        ],
        default='STRAIGHT'
    )
    
    # Basic Parameters
    pattern_density: FloatProperty(
        name="Density",
        description="Pattern density (points per unit)",
        default=DEFAULT_PATTERN_DENSITY,
        min=MIN_PATTERN_DENSITY,
        max=MAX_PATTERN_DENSITY,
        subtype='FACTOR'
    )
    
    sphere_size: FloatProperty(
        name="Sphere Size",
        description="Size of connection spheres",
        default=DEFAULT_SPHERE_SIZE,
        min=MIN_SPHERE_SIZE,
        max=MAX_SPHERE_SIZE,
        subtype='DISTANCE',
        unit='LENGTH'
    )
    
    connection_thickness: FloatProperty(
        name="Connection Thickness",
        description="Thickness of connections between spheres",
        default=DEFAULT_CONNECTION_THICKNESS,
        min=MIN_CONNECTION_THICKNESS,
        max=MAX_CONNECTION_THICKNESS,
        subtype='DISTANCE',
        unit='LENGTH'
    )
    
    # Material Settings
    use_dual_material: BoolProperty(
        name="Dual Material",
        description="Use dual material setup (TPU + PETG)",
        default=True
    )
    
    shell_thickness: FloatProperty(
        name="Shell Thickness",
        description="Thickness of TPU shell",
        default=DEFAULT_SHELL_THICKNESS,
        min=MIN_SHELL_THICKNESS,
        max=MAX_SHELL_THICKNESS,
        subtype='DISTANCE',
        unit='LENGTH'
    )
    
    material_tpu: EnumProperty(
        name="TPU Type",
        description="Type of TPU material",
        items=[
            ('TPU_95A', "TPU 95A", "Shore 95A hardness"),
            ('TPU_85A', "TPU 85A", "Shore 85A hardness"),
            ('TPU_75A', "TPU 75A", "Shore 75A hardness"),
            ('TPU_FLEX', "TPU Flex", "Extra flexible TPU"),
        ],
        default='TPU_95A'
    )
    
    material_petg: EnumProperty(
        name="PETG Type",
        description="Type of PETG material",
        items=[
            ('PETG_STANDARD', "PETG Standard", "Standard PETG"),
            ('PETG_TOUGH', "PETG Tough", "Toughened PETG"),
            ('PETG_CLEAR', "PETG Clear", "Transparent PETG"),
            ('PETG_CF', "PETG CF", "Carbon fiber reinforced"),
        ],
        default='PETG_STANDARD'
    )
    
    # Paint Mode
    paint_mode_active: BoolProperty(
        name="Paint Mode Active",
        description="Paint mode is currently active",
        default=False
    )
    
    brush_size: FloatProperty(
        name="Brush Size",
        description="Size of paint brush",
        default=DEFAULT_BRUSH_SIZE,
        min=1.0,
        max=100.0,
        subtype='DISTANCE'
    )
    
    brush_strength: FloatProperty(
        name="Brush Strength",
        description="Strength of brush effect",
        default=1.0,
        min=0.0,
        max=1.0,
        subtype='FACTOR'
    )
    
    use_symmetry: BoolProperty(
        name="Use Symmetry",
        description="Enable symmetrical painting",
        default=False
    )
    
    symmetry_axis: EnumProperty(
        name="Symmetry Axis",
        description="Axis for symmetrical painting",
        items=[
            ('X', "X", "Mirror along X axis"),
            ('Y', "Y", "Mirror along Y axis"),
            ('Z', "Z", "Mirror along Z axis"),
        ],
        default='X'
    )
    
    # Edge Detection
    edge_detection_angle: FloatProperty(
        name="Edge Angle",
        description="Minimum angle for edge detection",
        default=DEFAULT_EDGE_ANGLE,
        min=0.0,
        max=90.0,
        subtype='ANGLE'
    )
    
    edge_reinforcement_width: FloatProperty(
        name="Reinforcement Width",
        description="Width of edge reinforcement",
        default=DEFAULT_EDGE_WIDTH,
        min=1.0,
        max=20.0,
        subtype='DISTANCE'
    )
    
    detect_sharp_edges: BoolProperty(
        name="Detect Sharp Edges",
        description="Automatically detect sharp edges",
        default=True
    )
    
    detect_boundary_edges: BoolProperty(
        name="Detect Boundaries",
        description="Automatically detect boundary edges",
        default=True
    )
    
    # Advanced Settings
    pattern_seed: IntProperty(
        name="Seed",
        description="Random seed for pattern generation",
        default=0,
        min=0,
        max=10000
    )
    
    use_adaptive_density: BoolProperty(
        name="Adaptive Density",
        description="Adjust density based on surface curvature",
        default=False
    )
    
    adaptive_min_density: FloatProperty(
        name="Min Density",
        description="Minimum density for adaptive mode",
        default=2.0,
        min=0.1,
        max=10.0
    )
    
    adaptive_max_density: FloatProperty(
        name="Max Density",
        description="Maximum density for adaptive mode",
        default=10.0,
        min=1.0,
        max=20.0
    )
    
    # Performance Settings
    use_preview: BoolProperty(
        name="Preview Mode",
        description="Generate preview with reduced quality",
        default=False
    )
    
    preview_quality: FloatProperty(
        name="Preview Quality",
        description="Quality of preview (0.1 = fast, 1.0 = full)",
        default=0.3,
        min=0.1,
        max=1.0,
        subtype='FACTOR'
    )
    
    # Live Preview
    pattern_live_preview: BoolProperty(
        name="Live Preview",
        description="Show live preview of pattern",
        default=False
    )
    
    # Export Settings
    export_format: EnumProperty(
        name="Export Format",
        description="File format for export",
        items=[
            ('STL', "STL", "STL format"),
            ('OBJ', "OBJ", "Wavefront OBJ"),
            ('PLY', "PLY", "Stanford PLY"),
            ('FBX', "FBX", "Autodesk FBX"),
        ],
        default='STL'
    )
    
    export_separate_materials: BoolProperty(
        name="Separate Materials",
        description="Export TPU and PETG as separate files",
        default=True
    )
    
    export_apply_modifiers: BoolProperty(
        name="Apply Modifiers",
        description="Apply modifiers before export",
        default=True
    )
    
    # Quality Settings
    quality_preset: EnumProperty(
        name="Quality Preset",
        description="Quality preset for generation",
        items=[
            ('DRAFT', "Draft", "Fast preview quality"),
            ('NORMAL', "Normal", "Standard quality"),
            ('HIGH', "High", "High quality"),
            ('ULTRA', "Ultra", "Maximum quality"),
        ],
        default='NORMAL'
    )

def get_chain_tool_properties(context):
    """Get Chain Tool properties from context (NOT content!)"""
    if context and hasattr(context, 'scene') and hasattr(context.scene, 'chain_tool_properties'):
        return context.scene.chain_tool_properties
    return None

def register_properties():
    """Register properties"""
    try:
        bpy.utils.register_class(ChainToolProperties)
        bpy.types.Scene.chain_tool_properties = PointerProperty(type=ChainToolProperties)
        print("Chain Tool V4: Properties registered")
    except Exception as e:
        print(f"Chain Tool V4: Could not register properties: {e}")

def unregister_properties():
    """Unregister properties"""
    try:
        if hasattr(bpy.types.Scene, 'chain_tool_properties'):
            del bpy.types.Scene.chain_tool_properties
        bpy.utils.unregister_class(ChainToolProperties)
        print("Chain Tool V4: Properties unregistered")
    except Exception as e:
        print(f"Chain Tool V4: Could not unregister properties: {e}")

__all__ = ['ChainToolProperties', 'register_properties', 'unregister_properties', 'get_chain_tool_properties']
