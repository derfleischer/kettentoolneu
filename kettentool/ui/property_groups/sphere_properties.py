"""
Sphere Properties for Chain Construction Tool
Settings for chain sphere creation and appearance
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


class SphereProperties(PropertyGroup):
    """Properties for chain spheres"""
    
    # Basic Geometry
    radius: FloatProperty(
        name="Radius",
        description="Default radius for chain spheres",
        default=0.05,
        min=0.001,
        max=10.0,
        precision=3,
        step=0.01,
        subtype='DISTANCE'
    )
    
    subdivisions: IntProperty(
        name="Subdivisions",
        description="Subdivision level for spheres",
        default=2,
        min=0,
        max=6
    )
    
    segments: IntProperty(
        name="Segments",
        description="Number of segments for UV sphere",
        default=16,
        min=3,
        max=64
    )
    
    rings: IntProperty(
        name="Rings",
        description="Number of rings for UV sphere",
        default=8,
        min=2,
        max=32
    )
    
    # Sphere Type
    sphere_type: EnumProperty(
        name="Sphere Type",
        description="Type of sphere to create",
        items=[
            ('UV', "UV Sphere", "Standard UV sphere"),
            ('ICO', "Ico Sphere", "Icosphere (subdivided icosahedron)"),
            ('CUSTOM', "Custom", "Use custom mesh object")
        ],
        default='ICO'
    )
    
    custom_object: PointerProperty(
        name="Custom Object",
        description="Custom object to use as sphere",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'MESH' if obj else True
    )
    
    # Material Settings
    use_material: BoolProperty(
        name="Use Material",
        description="Apply material to spheres",
        default=True
    )
    
    material_type: EnumProperty(
        name="Material Type",
        description="Type of material to apply",
        items=[
            ('SIMPLE', "Simple", "Simple principled material"),
            ('METAL', "Metal", "Metallic material"),
            ('GLASS', "Glass", "Glass/transparent material"),
            ('CUSTOM', "Custom", "Use custom material"),
            ('RANDOM', "Random", "Random color per sphere")
        ],
        default='SIMPLE'
    )
    
    custom_material: PointerProperty(
        name="Custom Material",
        description="Custom material to use",
        type=bpy.types.Material
    )
    
    base_color: FloatVectorProperty(
        name="Base Color",
        description="Base color for spheres",
        default=(0.5, 0.5, 0.5, 1.0),
        size=4,
        min=0.0,
        max=1.0,
        subtype='COLOR_GAMMA'
    )
    
    metallic: FloatProperty(
        name="Metallic",
        description="Metallic value for material",
        default=0.0,
        min=0.0,
        max=1.0
    )
    
    roughness: FloatProperty(
        name="Roughness",
        description="Roughness value for material",
        default=0.5,
        min=0.0,
        max=1.0
    )
    
    # Instance Settings
    use_instances: BoolProperty(
        name="Use Instances",
        description="Use instancing for better performance",
        default=False
    )
    
    instance_collection: PointerProperty(
        name="Instance Collection",
        description="Collection to instance spheres from",
        type=bpy.types.Collection
    )
    
    # Variation Settings
    enable_variation: BoolProperty(
        name="Enable Variation",
        description="Add random variation to spheres",
        default=False
    )
    
    size_variation: FloatProperty(
        name="Size Variation",
        description="Random size variation percentage",
        default=0.1,
        min=0.0,
        max=1.0,
        subtype='PERCENTAGE'
    )
    
    rotation_variation: FloatProperty(
        name="Rotation Variation",
        description="Random rotation variation",
        default=0.0,
        min=0.0,
        max=180.0,
        subtype='ANGLE'
    )
    
    # Legacy compatibility
    kette_kugelradius: FloatProperty(
        name="Kugel Radius (Legacy)",
        description="Legacy property - use radius instead",
        default=0.05,
        min=0.001,
        max=10.0,
        get=lambda self: self.radius,
        set=lambda self, value: setattr(self, 'radius', value)
    )
