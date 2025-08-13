"""
Pattern Properties for Chain Construction Tool
Settings for pattern-based connector generation
"""

import bpy
from bpy.props import (
    BoolProperty,
    FloatProperty,
    IntProperty,
    EnumProperty,
    StringProperty
)
from bpy.types import PropertyGroup


class PatternProperties(PropertyGroup):
    """Properties for pattern connector"""
    
    # Pattern Type
    type: EnumProperty(
        name="Pattern Type",
        description="Type of connection pattern",
        items=[
            ('LINEAR', "Linear", "Connect in linear sequence"),
            ('ZIGZAG', "Zigzag", "Zigzag pattern"),
            ('SPIRAL', "Spiral", "Spiral pattern"),
            ('RADIAL', "Radial", "Radial/star pattern"),
            ('RANDOM', "Random", "Random connections"),
            ('CUSTOM', "Custom", "Custom pattern rules")
        ],
        default='LINEAR'
    )
    
    # Linear Pattern
    skip: IntProperty(
        name="Skip Every N",
        description="Skip every N-th sphere in linear pattern",
        default=0,
        min=0,
        max=10
    )
    
    reverse: BoolProperty(
        name="Reverse Order",
        description="Reverse connection order",
        default=False
    )
    
    closed_loop: BoolProperty(
        name="Close Loop",
        description="Connect last to first sphere",
        default=False
    )
    
    # Zigzag Pattern
    zigzag_width: FloatProperty(
        name="Width",
        description="Width of zigzag pattern",
        default=0.2,
        min=0.01,
        max=2.0,
        subtype='DISTANCE'
    )
    
    zigzag_count: IntProperty(
        name="Segments",
        description="Number of zigzag segments",
        default=5,
        min=2,
        max=50
    )
    
    zigzag_sharp: BoolProperty(
        name="Sharp Corners",
        description="Use sharp corners instead of smooth",
        default=False
    )
    
    zigzag_smooth: FloatProperty(
        name="Smoothness",
        description="Smoothness of zigzag curves",
        default=0.5,
        min=0.0,
        max=1.0
    )
    
    # Spiral Pattern
    spiral_turns: FloatProperty(
        name="Turns",
        description="Number of spiral turns",
        default=3.0,
        min=0.5,
        max=10.0
    )
    
    spiral_height: FloatProperty(
        name="Height Factor",
        description="Height of spiral",
        default=1.0,
        min=0.0,
        max=5.0
    )
    
    spiral_tightness: FloatProperty(
        name="Tightness",
        description="How tight the spiral is",
        default=1.0,
        min=0.1,
        max=3.0
    )
    
    spiral_direction: EnumProperty(
        name="Direction",
        description="Spiral direction",
        items=[
            ('CW', "Clockwise", "Clockwise spiral"),
            ('CCW', "Counter-Clockwise", "Counter-clockwise spiral")
        ],
        default='CW'
    )
    
    # Radial Pattern
    radial_rings: IntProperty(
        name="Ring Groups",
        description="Number of ring groups",
        default=3,
        min=1,
        max=10
    )
    
    radial_center: EnumProperty(
        name="Center Point",
        description="Center point for radial pattern",
        items=[
            ('ORIGIN', "Origin", "World origin"),
            ('CURSOR', "3D Cursor", "3D cursor position"),
            ('ACTIVE', "Active", "Active object"),
            ('MEDIAN', "Median", "Median point of selection")
        ],
        default='MEDIAN'
    )
    
    radial_connect_rings: BoolProperty(
        name="Connect Rings",
        description="Connect between rings",
        default=True
    )
    
    radial_ring_offset: FloatProperty(
        name="Ring Offset",
        description="Offset between rings",
        default=0.0,
        min=-1.0,
        max=1.0
    )
    
    # Random Pattern
    random_seed: IntProperty(
        name="Seed",
        description="Random seed for pattern",
        default=0,
        min=0,
        max=10000
    )
    
    random_probability: FloatProperty(
        name="Connection Probability",
        description="Probability of connection",
        default=0.3,
        min=0.0,
        max=1.0,
        subtype='PERCENTAGE'
    )
    
    random_min_distance: FloatProperty(
        name="Min Distance",
        description="Minimum distance for random connections",
        default=0.05,
        min=0.01,
        max=1.0,
        subtype='DISTANCE'
    )
    
    # Custom Pattern
    custom_rule: StringProperty(
        name="Custom Rule",
        description="Custom pattern rule (e.g., 'nearest', 'every:3', 'star')",
        default="nearest"
    )
    
    # Common Settings
    max_distance: FloatProperty(
        name="Max Distance",
        description="Maximum connection distance",
        default=0.5,
        min=0.01,
        max=10.0,
        subtype='DISTANCE'
    )
    
    use_distance: BoolProperty(
        name="Use Distance Limit",
        description="Limit connections by distance",
        default=True
    )
    
    smooth_connections: BoolProperty(
        name="Smooth Paths",
        description="Create smooth connection paths",
        default=False
    )
    
    smooth_iterations: IntProperty(
        name="Smooth Iterations",
        description="Number of smoothing iterations",
        default=2,
        min=0,
        max=10
    )
    
    # Advanced Options
    avoid_existing: BoolProperty(
        name="Avoid Duplicates",
        description="Avoid creating duplicate connections",
        default=True
    )
    
    bidirectional: BoolProperty(
        name="Bidirectional",
        description="Create bidirectional connections",
        default=False
    )
    
    use_selected: BoolProperty(
        name="Selected Only",
        description="Apply pattern only to selected spheres",
        default=False
    )
    
    # Preview
    preview: BoolProperty(
        name="Preview Pattern",
        description="Show pattern preview before applying",
        default=False
    )
    
    # Legacy compatibility
    pattern_type: EnumProperty(
        name="Pattern Type (Legacy)",
        items=[
            ('LINEAR', "Linear", "Linear"),
            ('ZIGZAG', "Zigzag", "Zigzag"),
            ('SPIRAL', "Spiral", "Spiral"),
            ('RADIAL', "Radial", "Radial"),
            ('RANDOM', "Random", "Random"),
            ('CUSTOM', "Custom", "Custom")
        ],
        default='LINEAR',
        get=lambda self: self.type,
        set=lambda self, value: setattr(self, 'type', value)
    )
    
    pattern_max_distance: FloatProperty(
        name="Max Distance (Legacy)",
        default=0.5,
        get=lambda self: self.max_distance,
        set=lambda self, value: setattr(self, 'max_distance', value)
    )
