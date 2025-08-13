"""
Main Properties for Chain Construction Tool
Target object and surface projection settings
"""

import bpy
from bpy.props import (
    BoolProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    EnumProperty
)
from bpy.types import PropertyGroup


class MainProperties(PropertyGroup):
    """Main settings for chain construction"""
    
    # Target Object
    target_object: PointerProperty(
        name="Target Object",
        description="Target mesh object for chain construction",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'MESH' if obj else True
    )
    
    # Surface Projection
    surface_projection: BoolProperty(
        name="Surface Projection",
        description="Project chain elements onto target surface",
        default=False
    )
    
    surface_offset: FloatProperty(
        name="Surface Offset",
        description="Distance offset from surface",
        default=0.01,
        min=-1.0,
        max=1.0,
        precision=3,
        step=0.1,
        subtype='DISTANCE'
    )
    
    projection_smoothing: IntProperty(
        name="Smoothing",
        description="Smoothing iterations for surface projection",
        default=2,
        min=0,
        max=10
    )
    
    projection_method: EnumProperty(
        name="Projection Method",
        description="Method for projecting onto surface",
        items=[
            ('NEAREST', "Nearest", "Project to nearest point on surface"),
            ('NORMAL', "Normal", "Project along vertex normals"),
            ('VIEW', "View", "Project from view direction"),
            ('AXIS', "Axis", "Project along specified axis")
        ],
        default='NEAREST'
    )
    
    # Aura Settings
    aura_distance: FloatProperty(
        name="Aura Distance",
        description="Distance for aura effect around surface",
        default=0.1,
        min=0.0,
        max=2.0,
        precision=3,
        step=0.01,
        subtype='DISTANCE'
    )
    
    aura_falloff: EnumProperty(
        name="Aura Falloff",
        description="Falloff type for aura effect",
        items=[
            ('LINEAR', "Linear", "Linear falloff"),
            ('SMOOTH', "Smooth", "Smooth falloff"),
            ('SHARP', "Sharp", "Sharp falloff"),
            ('CONSTANT', "Constant", "No falloff")
        ],
        default='SMOOTH'
    )
    
    # Construction Mode
    construction_mode: EnumProperty(
        name="Construction Mode",
        description="Mode for chain construction",
        items=[
            ('MANUAL', "Manual", "Place spheres manually"),
            ('PAINT', "Paint", "Paint spheres on surface"),
            ('PATH', "Path", "Follow path or curve"),
            ('PATTERN', "Pattern", "Use pattern generator")
        ],
        default='MANUAL'
    )
    
    # Global Settings
    use_collections: BoolProperty(
        name="Use Collections",
        description="Organize chain elements in collections",
        default=True
    )
    
    collection_name: StringProperty(
        name="Collection Name",
        description="Name for chain collection",
        default="Chain_"
    )
    
    # Update callback
    def update_surface_projection(self, context):
        """Called when surface projection changes"""
        if self.surface_projection and not self.target_object:
            self.surface_projection = False
            # Could show a message here
    
    # Legacy compatibility properties
    # (These map to the new property names for backward compatibility)
    kette_target_obj: PointerProperty(
        name="Target Object (Legacy)",
        description="Legacy property - use target_object instead",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'MESH' if obj else True,
        get=lambda self: self.target_object,
        set=lambda self, value: setattr(self, 'target_object', value)
    )
