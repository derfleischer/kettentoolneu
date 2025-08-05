"""
Properties for Chain Tool V4
All Blender properties used by the addon
"""

import bpy
from bpy.types import PropertyGroup, Scene
from bpy.props import (
    BoolProperty, IntProperty, FloatProperty, StringProperty,
    EnumProperty, PointerProperty, CollectionProperty,
    FloatVectorProperty
)

from core.constants import (
    PATTERN_TYPE_ITEMS, CONNECTION_MODE_ITEMS, 
    MATERIAL_PRESET_ITEMS, DEFAULT_SPHERE_RADIUS,
    DEFAULT_CONNECTOR_RADIUS, MIN_PATTERN_SPACING,
    MAX_PATTERN_SPACING
)

# ============================================
# PATTERN PROPERTIES
# ============================================

class PatternProperties(PropertyGroup):
    """Properties for pattern generation"""
    
    # Pattern type selection
    pattern_type: EnumProperty(
        name="Pattern Type",
        description="Type of pattern to generate",
        items=PATTERN_TYPE_ITEMS,
        default='VORONOI'
    )
    
    # Surface pattern settings
    cell_size: FloatProperty(
        name="Cell Size",
        description="Average size of pattern cells",
        default=12.0,
        min=3.0,
        max=50.0,
        unit='LENGTH',
        subtype='DISTANCE'
    )
    
    edge_thickness: FloatProperty(
        name="Edge Thickness", 
        description="Thickness of pattern edges",
        default=2.5,
        min=0.5,
        max=10.0,
        unit='LENGTH',
        subtype='DISTANCE'
    )
    
    fill_ratio: FloatProperty(
        name="Fill Ratio",
        description="Pattern density (0=empty, 1=solid)",
        default=0.7,
        min=0.1,
        max=0.95,
        subtype='FACTOR'
    )
    
    # Edge pattern settings
    edge_bands: IntProperty(
        name="Edge Bands",
        description="Number of reinforcement bands",
        default=3,
        min=1,
        max=5
    )
    
    band_spacing: FloatProperty(
        name="Band Spacing",
        description="Distance between edge bands",
        default=4.0,
        min=1.0,
        max=20.0,
        unit='LENGTH',
        subtype='DISTANCE'
    )
    
    # Advanced settings
    randomize: FloatProperty(
        name="Randomize",
        description="Random variation in pattern",
        default=0.0,
        min=0.0,
        max=1.0,
        subtype='FACTOR'
    )
    
    gradient_influence: FloatProperty(
        name="Gradient Influence",
        description="How much to vary pattern by stress/curvature",
        default=0.5,
        min=0.0,
        max=1.0,
        subtype='FACTOR'
    )
    
    smooth_iterations: IntProperty(
        name="Smooth Iterations",
        description="Smoothing passes for pattern",
        default=2,
        min=0,
        max=10
    )

# ============================================
# CONNECTION PROPERTIES
# ============================================

class ConnectionProperties(PropertyGroup):
    """Properties for connections between points"""
    
    connection_mode: EnumProperty(
        name="Connection Mode",
        description="How to connect pattern points",
        items=CONNECTION_MODE_ITEMS,
        default='AUTO'
    )
    
    max_connection_length: FloatProperty(
        name="Max Length",
        description="Maximum connection length",
        default=25.0,
        min=1.0,
        max=100.0,
        unit='LENGTH',
        subtype='DISTANCE'
    )
    
    min_angle: FloatProperty(
        name="Min Angle",
        description="Minimum angle for triangulation",
        default=20.0,
        min=10.0,
        max=60.0,
        subtype='ANGLE'
    )
    
    connector_radius: FloatProperty(
        name="Connector Radius",
        description="Radius of connector cylinders",
        default=DEFAULT_CONNECTOR_RADIUS,
        min=0.1,
        max=5.0,
        unit='LENGTH',
        subtype='DISTANCE'
    )
    
    sphere_radius: FloatProperty(
        name="Sphere Radius",
        description="Radius of connection spheres",
        default=DEFAULT_SPHERE_RADIUS,
        min=0.1,
        max=10.0,
        unit='LENGTH',
        subtype='DISTANCE'
    )
    
    curved_connectors: BoolProperty(
        name="Curved Connectors",
        description="Use curved instead of straight connectors",
        default=False
    )
    
    curve_resolution: IntProperty(
        name="Curve Resolution",
        description="Segments for curved connectors",
        default=8,
        min=3,
        max=32
    )

# ============================================
# PAINT MODE PROPERTIES
# ============================================

class PaintProperties(PropertyGroup):
    """Properties for paint mode"""
    
    brush_size: FloatProperty(
        name="Brush Size",
        description="Size of paint brush",
        default=8.0,
        min=2.0,
        max=50.0,
        unit='LENGTH',
        subtype='DISTANCE'
    )
    
    brush_strength: FloatProperty(
        name="Brush Strength",
        description="Strength of brush effect",
        default=1.0,
        min=0.1,
        max=1.0,
        subtype='FACTOR'
    )
    
    paint_mode: EnumProperty(
        name="Paint Mode",
        items=[
            ('ADD', "Add", "Add pattern points"),
            ('REMOVE', "Remove", "Remove pattern points"),
            ('CONNECT', "Connect", "Paint connections"),
        ],
        default='ADD'
    )
    
    snap_to_surface: BoolProperty(
        name="Snap to Surface",
        description="Snap painted points to surface",
        default=True
    )
    
    auto_connect: BoolProperty(
        name="Auto Connect",
        description="Automatically connect painted points",
        default=True
    )
    
    stroke_smooth: IntProperty(
        name="Stroke Smooth",
        description="Smoothing iterations for strokes",
        default=2,
        min=0,
        max=5
    )

# ============================================
# EXPORT PROPERTIES
# ============================================

class ExportProperties(PropertyGroup):
    """Properties for export operations"""
    
    material_preset: EnumProperty(
        name="Material Preset",
        description="Predefined material combinations",
        items=MATERIAL_PRESET_ITEMS,
        default='TPU_PETG'
    )
    
    shell_thickness: FloatProperty(
        name="Shell Thickness",
        description="Thickness of outer shell",
        default=1.2,
        min=0.4,
        max=3.0,
        unit='LENGTH',
        subtype='DISTANCE'
    )
    
    pattern_thickness: FloatProperty(
        name="Pattern Thickness",
        description="Thickness of pattern structure",
        default=0.8,
        min=0.4,
        max=2.0,
        unit='LENGTH',
        subtype='DISTANCE'
    )
    
    export_scale: FloatProperty(
        name="Export Scale",
        description="Scale factor for export (10 = cm to mm)",
        default=10.0,
        min=0.1,
        max=1000.0
    )
    
    separate_materials: BoolProperty(
        name="Separate Materials",
        description="Export as separate objects per material",
        default=True
    )
    
    apply_modifiers: BoolProperty(
        name="Apply Modifiers",
        description="Apply modifiers before export",
        default=True
    )
    
    export_path: StringProperty(
        name="Export Path",
        description="Path for export files",
        default="//",
        subtype='DIR_PATH'
    )

# ============================================
# MAIN PROPERTIES
# ============================================

class ChainToolProperties(PropertyGroup):
    """Main property group for Chain Tool"""
    
    # Target object
    target_object: PointerProperty(
        name="Target Object",
        description="Object to create pattern on",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'MESH'
    )
    
    # Sub-property groups
    pattern: PointerProperty(type=PatternProperties)
    connection: PointerProperty(type=ConnectionProperties)
    paint: PointerProperty(type=PaintProperties)
    export: PointerProperty(type=ExportProperties)
    
    # Global settings
    live_preview: BoolProperty(
        name="Live Preview",
        description="Show real-time preview",
        default=True
    )
    
    keep_construction: BoolProperty(
        name="Keep Construction",
        description="Keep construction objects after generation",
        default=False
    )
    
    use_collections: BoolProperty(
        name="Use Collections",
        description="Organize objects in collections",
        default=True
    )
    
    # State tracking
    is_generating: BoolProperty(
        name="Is Generating",
        description="Pattern generation in progress",
        default=False
    )
    
    paint_mode_active: BoolProperty(
        name="Paint Mode Active",
        description="Paint mode is currently active",
        default=False
    )

# ============================================
# REGISTRATION
# ============================================

CLASSES = [
    PatternProperties,
    ConnectionProperties,
    PaintProperties,
    ExportProperties,
    ChainToolProperties,
]

def register_properties():
    """Register all properties"""
    # Register property on Scene
    Scene.chain_tool = PointerProperty(type=ChainToolProperties)
    
    print("[Chain Tool V4] Properties registered")

def unregister_properties():
    """Unregister all properties"""
    # Remove property from Scene
    if hasattr(Scene, 'chain_tool'):
        del Scene.chain_tool
    
    print("[Chain Tool V4] Properties unregistered")
