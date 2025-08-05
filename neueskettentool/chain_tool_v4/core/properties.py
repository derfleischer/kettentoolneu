"""
Chain Tool V4 - Properties System
FIXED VERSION - Property-Namen korrigiert für UI-Kompatibilität

Hauptänderung: Scene.chain_tool_properties (statt chain_tool)
"""

import bpy
from bpy.types import PropertyGroup, Scene
from bpy.props import (
    EnumProperty, PointerProperty, BoolProperty, 
    IntProperty, FloatProperty, StringProperty,
    FloatVectorProperty
)

# ============================================
# ENUMS FÜR PROPERTY-AUSWAHL
# ============================================

def get_pattern_types(self, context):
    """Dynamische Pattern-Type Liste"""
    return [
        ('SURFACE', 'Surface Pattern', 'Large area surface patterns', 'MESH_UVSPHERE', 0),
        ('EDGE', 'Edge Pattern', 'Edge reinforcement patterns', 'EDGESEL', 1),
        ('HYBRID', 'Hybrid Pattern', 'Combined pattern strategies', 'MOD_LATTICE', 2),
        ('PAINT', 'Paint Mode', 'Interactive paint-based patterns', 'BRUSH_DATA', 3),
    ]

def get_surface_pattern_subtypes(self, context):
    """Surface Pattern Subtypen"""
    return [
        ('VORONOI', 'Voronoi', 'Voronoi cell pattern', 'MESH_ICOSPHERE', 0),
        ('HEXAGONAL', 'Hexagonal', 'Hexagonal grid pattern', 'MESH_CYLINDER', 1),
        ('TRIANGULAR', 'Triangular', 'Triangular mesh pattern', 'MESH_CONE', 2),
        ('RANDOM', 'Random Distribution', 'Random point distribution', 'PARTICLES', 3),
    ]

def get_connection_types(self, context):
    """Verbindungstypen"""
    return [
        ('STRAIGHT', 'Straight', 'Direct straight connections', 'IPO_LINEAR', 0),
        ('CURVED', 'Curved', 'Curved organic connections', 'IPO_BEZIER', 1),
        ('ADAPTIVE', 'Adaptive', 'Surface-adaptive connections', 'IPO_EASE_IN_OUT', 2),
    ]

# ============================================
# PATTERN PROPERTIES
# ============================================

class PatternProperties(PropertyGroup):
    """Pattern-spezifische Eigenschaften"""
    
    # Surface Pattern Settings
    surface_subtype: EnumProperty(
        name="Surface Type",
        description="Type of surface pattern",
        items=get_surface_pattern_subtypes,
        default='VORONOI'
    )
    
    # Voronoi Settings
    voronoi_sites: IntProperty(
        name="Voronoi Sites",
        description="Number of Voronoi sites",
        default=50,
        min=10,
        max=500
    )
    
    voronoi_relaxation: IntProperty(
        name="Relaxation Steps",
        description="Lloyd relaxation iterations",
        default=3,
        min=0,
        max=10
    )
    
    # Hexagonal Settings
    hex_spacing: FloatProperty(
        name="Hex Spacing",
        description="Distance between hexagon centers",
        default=0.1,
        min=0.01,
        max=1.0,
        subtype='DISTANCE'
    )
    
    hex_scale_variation: FloatProperty(
        name="Scale Variation",
        description="Random scale variation",
        default=0.1,
        min=0.0,
        max=0.5,
        subtype='FACTOR'
    )
    
    # General Pattern Settings
    density: FloatProperty(
        name="Density",
        description="Pattern density factor",
        default=1.0,
        min=0.1,
        max=5.0,
        subtype='FACTOR'
    )
    
    size_variation: FloatProperty(
        name="Size Variation",
        description="Random size variation",
        default=0.2,
        min=0.0,
        max=1.0,
        subtype='FACTOR'
    )
    
    use_surface_projection: BoolProperty(
        name="Surface Projection",
        description="Project pattern onto surface",
        default=True
    )

# ============================================
# CONNECTION PROPERTIES
# ============================================

class ConnectionProperties(PropertyGroup):
    """Verbindungs-Eigenschaften"""
    
    connection_type: EnumProperty(
        name="Connection Type",
        description="Type of connections between elements",
        items=get_connection_types,
        default='ADAPTIVE'
    )
    
    max_distance: FloatProperty(
        name="Max Distance",
        description="Maximum connection distance",
        default=0.15,
        min=0.01,
        max=1.0,
        subtype='DISTANCE'
    )
    
    min_distance: FloatProperty(
        name="Min Distance",
        description="Minimum connection distance",
        default=0.02,
        min=0.001,
        max=0.1,
        subtype='DISTANCE'
    )
    
    thickness: FloatProperty(
        name="Connection Thickness",
        description="Thickness of connections",
        default=0.02,
        min=0.001,
        max=0.1,
        subtype='DISTANCE'
    )
    
    smoothing: IntProperty(
        name="Smoothing",
        description="Connection smoothing iterations",
        default=2,
        min=0,
        max=10
    )

# ============================================
# PAINT PROPERTIES
# ============================================

class PaintProperties(PropertyGroup):
    """Paint-Mode Eigenschaften"""
    
    brush_size: FloatProperty(
        name="Brush Size",
        description="Paint brush size",
        default=0.05,
        min=0.01,
        max=0.5,
        subtype='DISTANCE'
    )
    
    brush_strength: FloatProperty(
        name="Brush Strength",
        description="Paint brush strength",
        default=1.0,
        min=0.1,
        max=2.0,
        subtype='FACTOR'
    )
    
    brush_falloff: EnumProperty(
        name="Brush Falloff",
        description="Brush falloff type",
        items=[
            ('SMOOTH', 'Smooth', 'Smooth falloff'),
            ('SHARP', 'Sharp', 'Sharp falloff'),
            ('LINEAR', 'Linear', 'Linear falloff'),
        ],
        default='SMOOTH'
    )
    
    symmetry_x: BoolProperty(
        name="X Symmetry",
        description="Mirror painting on X axis",
        default=False
    )
    
    symmetry_y: BoolProperty(
        name="Y Symmetry", 
        description="Mirror painting on Y axis",
        default=False
    )
    
    symmetry_z: BoolProperty(
        name="Z Symmetry",
        description="Mirror painting on Z axis", 
        default=False
    )
    
    auto_connect: BoolProperty(
        name="Auto Connect",
        description="Automatically connect painted elements",
        default=True
    )
    
    surface_snapping: BoolProperty(
        name="Surface Snapping",
        description="Snap paint to surface",
        default=True
    )

# ============================================
# EXPORT PROPERTIES
# ============================================

class ExportProperties(PropertyGroup):
    """Export-Eigenschaften für 3D-Druck"""
    
    export_format: EnumProperty(
        name="Export Format",
        description="Export file format",
        items=[
            ('STL', 'STL', 'STL format for 3D printing'),
            ('OBJ', 'OBJ', 'Wavefront OBJ format'),
            ('PLY', 'PLY', 'Stanford PLY format'),
            ('3MF', '3MF', 'Microsoft 3MF format'),
        ],
        default='STL'
    )
    
    dual_material: BoolProperty(
        name="Dual Material Export",
        description="Export as separate TPU/PETG files",
        default=True
    )
    
    tpu_material: StringProperty(
        name="TPU Material",
        description="Flexible material preset",
        default="flexible_standard"
    )
    
    petg_material: StringProperty(
        name="PETG Material", 
        description="Rigid material preset",
        default="rigid_standard"
    )
    
    wall_thickness: FloatProperty(
        name="Wall Thickness",
        description="Minimum wall thickness",
        default=0.8,
        min=0.4,
        max=3.0,
        subtype='DISTANCE'
    )
    
    infill_density: IntProperty(
        name="Infill Density",
        description="Infill density percentage",
        default=20,
        min=0,
        max=100,
        subtype='PERCENTAGE'
    )

# ============================================
# MAIN PROPERTIES CLASS
# ============================================

class ChainToolProperties(PropertyGroup):
    """Haupt-Properties für Chain Tool V4"""
    
    # Pattern Selection
    pattern_type: EnumProperty(
        name="Pattern Type",
        description="Type of pattern to generate",
        items=get_pattern_types,
        default='SURFACE'
    )
    
    # Target Object
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
    
    # Performance settings
    use_bvh_acceleration: BoolProperty(
        name="BVH Acceleration",
        description="Use BVH trees for performance",
        default=True
    )
    
    cache_results: BoolProperty(
        name="Cache Results",
        description="Cache computation results",
        default=True
    )
    
    # State tracking (read-only for UI)
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
    # Register classes first
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    
    # FIXED: Korrekte Property-Namen für UI-Kompatibilität
    Scene.chain_tool_properties = PointerProperty(type=ChainToolProperties)
    
    print("Chain Tool V4: Properties registered (FIXED VERSION)")

def unregister_properties():
    """Unregister all properties"""
    # Remove property from Scene
    if hasattr(Scene, 'chain_tool_properties'):
        del Scene.chain_tool_properties
    
    # Unregister classes
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
    
    print("Chain Tool V4: Properties unregistered")

def register():
    """Register function for module"""
    register_properties()

def unregister():
    """Unregister function for module"""
    unregister_properties()

# Development test
if __name__ == "__main__":
    register()
    print("Chain Tool V4 Properties: Test registration successful")
