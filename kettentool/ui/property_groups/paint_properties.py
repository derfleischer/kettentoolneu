"""
Paint Mode Properties for Chain Construction Tool
Settings for interactive painting of chain spheres
"""

import bpy
from bpy.props import (
    BoolProperty,
    FloatProperty,
    IntProperty,
    EnumProperty,
    StringProperty,
    FloatVectorProperty
)
from bpy.types import PropertyGroup


class PaintProperties(PropertyGroup):
    """Properties for paint mode"""
    
    # Paint Mode State
    mode_active: BoolProperty(
        name="Paint Mode Active",
        description="Whether paint mode is currently active",
        default=False
    )
    
    # Brush Settings
    sphere_radius: FloatProperty(
        name="Sphere Size",
        description="Size of spheres when painting",
        default=0.05,
        min=0.001,
        max=1.0,
        precision=3,
        step=0.01,
        subtype='DISTANCE'
    )
    
    spacing: FloatProperty(
        name="Spacing",
        description="Minimum spacing between painted spheres",
        default=0.1,
        min=0.01,
        max=1.0,
        precision=3,
        subtype='DISTANCE'
    )
    
    smooth_factor: FloatProperty(
        name="Smoothing",
        description="Path smoothing factor",
        default=0.5,
        min=0.0,
        max=1.0
    )
    
    # Brush Behavior
    follow_surface: BoolProperty(
        name="Follow Surface",
        description="Make brush follow surface contours",
        default=True
    )
    
    surface_offset: FloatProperty(
        name="Surface Offset",
        description="Offset from surface when painting",
        default=0.01,
        min=-0.5,
        max=0.5,
        precision=3,
        subtype='DISTANCE'
    )
    
    continuous: BoolProperty(
        name="Continuous Stroke",
        description="Create continuous stroke while painting",
        default=True
    )
    
    stroke_smooth: FloatProperty(
        name="Stroke Smoothing",
        description="Smoothing for continuous strokes",
        default=0.3,
        min=0.0,
        max=1.0
    )
    
    # Auto-Connect in Paint Mode
    auto_connect: BoolProperty(
        name="Auto-Connect",
        description="Automatically connect painted spheres",
        default=True
    )
    
    connect_distance: FloatProperty(
        name="Connect Distance",
        description="Maximum distance for auto-connection while painting",
        default=0.15,
        min=0.01,
        max=1.0,
        precision=3,
        subtype='DISTANCE'
    )
    
    connect_chain: BoolProperty(
        name="Chain Mode",
        description="Connect spheres in chain sequence",
        default=True
    )
    
    chain_max: IntProperty(
        name="Max Chain Links",
        description="Maximum chain links per stroke",
        default=50,
        min=2,
        max=500
    )
    
    # Paint Patterns
    pattern: EnumProperty(
        name="Paint Pattern",
        description="Pattern to use while painting",
        items=[
            ('NONE', "None", "No pattern"),
            ('DOTS', "Dots", "Dotted pattern"),
            ('LINE', "Line", "Continuous line"),
            ('WAVE', "Wave", "Wave pattern"),
            ('SPIRAL', "Spiral", "Spiral pattern")
        ],
        default='NONE'
    )
    
    dot_spacing: FloatProperty(
        name="Dot Spacing",
        description="Spacing for dot pattern",
        default=0.05,
        min=0.01,
        max=0.5,
        subtype='DISTANCE'
    )
    
    line_segments: IntProperty(
        name="Line Segments",
        description="Number of segments in line pattern",
        default=10,
        min=2,
        max=100
    )
    
    wave_amplitude: FloatProperty(
        name="Wave Amplitude",
        description="Amplitude of wave pattern",
        default=0.1,
        min=0.0,
        max=1.0,
        subtype='DISTANCE'
    )
    
    wave_frequency: FloatProperty(
        name="Wave Frequency",
        description="Frequency of wave pattern",
        default=2.0,
        min=0.1,
        max=10.0
    )
    
    # Symmetry
    use_symmetry: BoolProperty(
        name="Use Symmetry",
        description="Paint with symmetry",
        default=False
    )
    
    symmetry_axis: EnumProperty(
        name="Symmetry Axis",
        description="Axis for symmetry",
        items=[
            ('X', "X", "X axis symmetry"),
            ('Y', "Y", "Y axis symmetry"),
            ('Z', "Z", "Z axis symmetry")
        ],
        default='X'
    )
    
    # Brush Display
    show_brush: BoolProperty(
        name="Show Brush",
        description="Show brush preview",
        default=True
    )
    
    brush_color: FloatVectorProperty(
        name="Brush Color",
        description="Color for brush preview",
        default=(1.0, 0.0, 0.0, 0.3),
        size=4,
        min=0.0,
        max=1.0,
        subtype='COLOR_GAMMA'
    )
    
    # Stroke Settings
    min_distance: FloatProperty(
        name="Min Distance",
        description="Minimum distance to register new point",
        default=0.01,
        min=0.001,
        max=0.1,
        precision=3,
        subtype='DISTANCE'
    )
    
    adaptive_spacing: BoolProperty(
        name="Adaptive Spacing",
        description="Adapt spacing based on stroke speed",
        default=False
    )
    
    # Legacy compatibility
    paint_mode_active: BoolProperty(
        name="Paint Mode (Legacy)",
        default=False,
        get=lambda self: self.mode_active,
        set=lambda self, value: setattr(self, 'mode_active', value)
    )
    
    paint_sphere_radius: FloatProperty(
        name="Sphere Radius (Legacy)",
        default=0.05,
        get=lambda self: self.sphere_radius,
        set=lambda self, value: setattr(self, 'sphere_radius', value)
    )
    
    paint_spacing: FloatProperty(
        name="Spacing (Legacy)",
        default=0.1,
        get=lambda self: self.spacing,
        set=lambda self, value: setattr(self, 'spacing', value)
    )
