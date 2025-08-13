"""
Connector Properties for Chain Construction Tool
Settings for connectors between chain spheres
"""

import bpy
from bpy.props import (
    BoolProperty,
    FloatProperty,
    IntProperty,
    EnumProperty,
    PointerProperty,
    FloatVectorProperty
)
from bpy.types import PropertyGroup


class ConnectorProperties(PropertyGroup):
    """Properties for chain connectors"""
    
    # Basic Geometry
    radius: FloatProperty(
        name="Radius",
        description="Radius of connector cylinders",
        default=0.02,
        min=0.001,
        max=1.0,
        precision=3,
        step=0.001,
        subtype='DISTANCE'
    )
    
    radius_factor: FloatProperty(
        name="Radius Factor",
        description="Connector radius as factor of sphere radius",
        default=0.4,
        min=0.1,
        max=1.0,
        precision=2
    )
    
    segments: IntProperty(
        name="Segments",
        description="Number of segments for connector cylinder",
        default=8,
        min=3,
        max=32
    )
    
    # Connector Type
    connector_type: EnumProperty(
        name="Connector Type",
        description="Type of connector to create",
        items=[
            ('CYLINDER', "Cylinder", "Simple cylinder connector"),
            ('CURVED', "Curved", "Curved connector following path"),
            ('CHAIN', "Chain Link", "Chain link style connector"),
            ('CUSTOM', "Custom", "Use custom mesh object")
        ],
        default='CYLINDER'
    )
    
    custom_object: PointerProperty(
        name="Custom Object",
        description="Custom object to use as connector",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'MESH' if obj else True
    )
    
    # Curve Settings
    use_curved: BoolProperty(
        name="Use Curved",
        description="Create curved connectors instead of straight",
        default=False
    )
    
    curve_factor: FloatProperty(
        name="Curve Factor",
        description="Amount of curvature for connectors",
        default=0.3,
        min=0.0,
        max=1.0
    )
    
    curve_resolution: IntProperty(
        name="Curve Resolution",
        description="Resolution of curved connectors",
        default=12,
        min=2,
        max=64
    )
    
    curve_type: EnumProperty(
        name="Curve Type",
        description="Type of curve for connectors",
        items=[
            ('BEZIER', "Bezier", "Bezier curve"),
            ('NURBS', "NURBS", "NURBS curve"),
            ('POLY', "Poly", "Poly curve")
        ],
        default='BEZIER'
    )
    
    # Auto-Connect Settings
    auto_connect_enabled: BoolProperty(
        name="Auto-Connect",
        description="Automatically connect nearby spheres",
        default=False
    )
    
    connect_distance: FloatProperty(
        name="Connect Distance",
        description="Maximum distance for auto-connection",
        default=0.15,
        min=0.01,
        max=10.0,
        precision=3,
        subtype='DISTANCE'
    )
    
    connect_angle: FloatProperty(
        name="Connect Angle",
        description="Maximum angle for auto-connection",
        default=60.0,
        min=0.0,
        max=180.0,
        subtype='ANGLE'
    )
    
    avoid_crossing: BoolProperty(
        name="Avoid Crossing",
        description="Try to avoid crossing connectors",
        default=True
    )
    
    prefer_shortest: BoolProperty(
        name="Prefer Shortest",
        description="Prefer shortest path connections",
        default=True
    )
    
    # Material Settings
    use_same_material: BoolProperty(
        name="Use Same Material",
        description="Use same material as spheres",
        default=True
    )
    
    connector_material: PointerProperty(
        name="Connector Material",
        description="Material for connectors",
        type=bpy.types.Material
    )
    
    # Advanced Settings
    merge_distance: FloatProperty(
        name="Merge Distance",
        description="Distance threshold for merging connectors",
        default=0.001,
        min=0.0,
        max=0.1,
        precision=4,
        subtype='DISTANCE'
    )
    
    smooth_shading: BoolProperty(
        name="Smooth Shading",
        description="Use smooth shading for connectors",
        default=True
    )
    
    # Preview Settings
    preview_connections: BoolProperty(
        name="Preview Connections",
        description="Show preview of connections before creating",
        default=False
    )
    
    preview_color: FloatVectorProperty(
        name="Preview Color",
        description="Color for connection preview",
        default=(1.0, 0.5, 0.0, 0.5),
        size=4,
        min=0.0,
        max=1.0,
        subtype='COLOR_GAMMA'
    )
    
    # Legacy compatibility
    kette_connector_segs: IntProperty(
        name="Connector Segments (Legacy)",
        default=8,
        get=lambda self: self.segments,
        set=lambda self, value: setattr(self, 'segments', value)
    )
    
    kette_connector_radius_factor: FloatProperty(
        name="Radius Factor (Legacy)",
        default=0.4,
        get=lambda self: self.radius_factor,
        set=lambda self, value: setattr(self, 'radius_factor', value)
    )
