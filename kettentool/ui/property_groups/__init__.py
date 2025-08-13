"""
Property Groups Module for Kettentool
Central registration of all property groups - CLEAN VERSION
"""

import bpy
from bpy.props import (
    PointerProperty,
    FloatProperty,
    BoolProperty,
    EnumProperty,
    IntProperty,
    StringProperty,
)
from bpy.types import PropertyGroup

# Import all property modules
from . import main_properties
from . import sphere_properties
from . import connector_properties
from . import paint_properties
from . import pattern_properties
from . import settings_properties
from . import debug_properties


class ChainConstructionProperties(PropertyGroup):
    """
    Main property group with ALL properties FLAT
    Operators access properties directly: props.kette_target_obj etc.
    """
    
    # ============================================
    # DIREKTE PROPERTIES FÜR OPERATORS
    # ============================================
    
    # Von main_properties
    kette_target_obj: PointerProperty(
        name="Target Object",
        description="Target object for chain construction",
        type=bpy.types.Object
    )
    
    # Von sphere_properties  
    kette_kugelradius: FloatProperty(
        name="Sphere Radius",
        description="Radius of chain spheres",
        default=0.1,
        min=0.01,
        max=10.0,
        step=0.01,
        precision=3
    )
    
    # Von connector_properties
    kette_use_curved: BoolProperty(
        name="Use Curved Connectors",
        description="Use curved connectors instead of straight ones",
        default=False
    )
    
    kette_connector_radius_factor: FloatProperty(
        name="Connector Radius Factor",
        description="Factor for connector radius relative to sphere radius",
        default=0.5,
        min=0.1,
        max=2.0,
        step=0.1
    )
    
    kette_auto_connect: BoolProperty(
        name="Auto Connect",
        description="Automatically connect spheres when painting",
        default=True
    )
    
    # Von paint_properties
    paint_brush_spacing: FloatProperty(
        name="Brush Spacing",
        description="Spacing between painted spheres",
        default=2.0,
        min=1.1,
        max=10.0,
        step=0.1
    )
    
    paint_use_symmetry: BoolProperty(
        name="Use Symmetry",
        description="Use symmetry when painting",
        default=False
    )
    
    paint_symmetry_axis: EnumProperty(
        name="Symmetry Axis",
        description="Axis for symmetry",
        items=[
            ('X', "X", "Mirror along X axis"),
            ('Y', "Y", "Mirror along Y axis"),
            ('Z', "Z", "Mirror along Z axis"),
        ],
        default='X'
    )
    
    # ============================================
    # ZUSÄTZLICHE UI PROPERTIES
    # ============================================
    
    # UI Properties von main_properties
    chain_type: EnumProperty(
        name="Chain Type",
        description="Type of chain to create",
        items=[
            ('NORMAL', "Normal", "Standard chain"),
            ('TWISTED', "Twisted", "Twisted chain pattern"),
            ('DOUBLE', "Double", "Double linked chain"),
        ],
        default='NORMAL'
    )
    
    link_count: IntProperty(
        name="Link Count",
        description="Number of chain links",
        default=10,
        min=1,
        max=1000
    )
    
    custom_link: PointerProperty(
        name="Custom Link",
        description="Custom link object for chain",
        type=bpy.types.Object
    )
    
    surface_offset: FloatProperty(
        name="Surface Offset",
        description="Offset from surface when snapping",
        default=0.0,
        min=-1.0,
        max=1.0,
        step=0.01
    )
    
    show_preview: BoolProperty(
        name="Show Preview",
        description="Show preview in viewport",
        default=True
    )
    
    auto_update: BoolProperty(
        name="Auto Update",
        description="Automatically update chain on parameter change",
        default=True
    )
    
    use_modifiers: BoolProperty(
        name="Use Modifiers",
        description="Apply modifiers to chain",
        default=False
    )
    
    # Material settings
    material_preset: EnumProperty(
        name="Material Preset",
        description="Preset material for chains",
        items=[
            ('NONE', "None", "No material"),
            ('METAL', "Metal", "Metallic material"),
            ('GOLD', "Gold", "Gold material"),
            ('SILVER', "Silver", "Silver material"),
        ],
        default='METAL'
    )
    
    # Pattern settings
    pattern_type: EnumProperty(
        name="Pattern Type",
        description="Type of connection pattern",
        items=[
            ('LINEAR', "Linear", "Linear connection"),
            ('CURVED', "Curved", "Curved connection"),
            ('SPIRAL', "Spiral", "Spiral pattern"),
        ],
        default='LINEAR'
    )
    
    subdivision_level: IntProperty(
        name="Subdivisions",
        description="Number of subdivisions in pattern",
        default=2,
        min=0,
        max=10
    )
    
    # Export settings
    export_path: StringProperty(
        name="Export Path",
        description="Path for exporting chains",
        default="//chains/",
        subtype='DIR_PATH'
    )
    
    # Debug
    show_debug: BoolProperty(
        name="Show Debug Info",
        description="Show debug information",
        default=False
    )


# List of all property modules
property_modules = [
    main_properties,
    sphere_properties,
    connector_properties,
    paint_properties,
    pattern_properties,
    settings_properties,
    debug_properties,
]


def register():
    """Register all property groups"""
    print("[PROPERTY_GROUPS] Starting registration...")
    
    # Register all sub-modules first
    for module in property_modules:
        if hasattr(module, 'register'):
            module.register()
            print(f"[PROPERTY_GROUPS] ✅ Registered {module.__name__}")
    
    # Register main property group
    bpy.utils.register_class(ChainConstructionProperties)
    
    # Add to Scene with EXACT name operators expect
    bpy.types.Scene.chain_construction_props = PointerProperty(
        type=ChainConstructionProperties
    )
    
    # Also add shorter name for UI panels
    bpy.types.Scene.chain_props = bpy.types.Scene.chain_construction_props
    
    print("[PROPERTY_GROUPS] ✅ Registration complete")


def unregister():
    """Unregister all property groups"""
    print("[PROPERTY_GROUPS] Starting unregistration...")
    
    # Remove from Scene
    if hasattr(bpy.types.Scene, 'chain_props'):
        del bpy.types.Scene.chain_props
    
    if hasattr(bpy.types.Scene, 'chain_construction_props'):
        del bpy.types.Scene.chain_construction_props
    
    # Unregister main property group
    bpy.utils.unregister_class(ChainConstructionProperties)
    
    # Unregister all sub-modules
    for module in reversed(property_modules):
        if hasattr(module, 'unregister'):
            module.unregister()
            print(f"[PROPERTY_GROUPS] ✅ Unregistered {module.__name__}")
    
    print("[PROPERTY_GROUPS] ✅ Unregistration complete")
