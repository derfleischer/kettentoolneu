"""
Settings Properties for Chain Construction Tool
General settings and preferences
"""

import bpy
from bpy.props import (
    BoolProperty,
    FloatProperty,
    IntProperty,
    EnumProperty,
    StringProperty,
    PointerProperty,
    FloatVectorProperty
)
from bpy.types import PropertyGroup


class SettingsProperties(PropertyGroup):
    """General settings for Chain Construction Tool"""
    
    # Display Settings
    auto_update: BoolProperty(
        name="Auto Update",
        description="Automatically update chain when settings change",
        default=True
    )
    
    update_delay: IntProperty(
        name="Update Delay",
        description="Delay in milliseconds before auto-update",
        default=100,
        min=0,
        max=1000
    )
    
    show_overlays: BoolProperty(
        name="Show Overlays",
        description="Show visual overlays in viewport",
        default=True
    )
    
    overlay_color: FloatVectorProperty(
        name="Overlay Color",
        description="Color for overlay displays",
        default=(1.0, 0.5, 0.0, 0.5),
        size=4,
        min=0.0,
        max=1.0,
        subtype='COLOR_GAMMA'
    )
    
    overlay_alpha: FloatProperty(
        name="Overlay Alpha",
        description="Transparency of overlays",
        default=0.5,
        min=0.0,
        max=1.0
    )
    
    # Performance Settings
    viewport_display_mode: EnumProperty(
        name="Viewport Display",
        description="Display mode for viewport",
        items=[
            ('FULL', "Full Quality", "Show full quality in viewport"),
            ('ADAPTIVE', "Adaptive", "Adapt quality based on performance"),
            ('FAST', "Fast", "Optimize for speed"),
            ('BBOX', "Bounding Box", "Show only bounding boxes")
        ],
        default='ADAPTIVE'
    )
    
    adaptive_threshold: IntProperty(
        name="Adaptive Threshold",
        description="Object count threshold for adaptive display",
        default=100,
        min=10,
        max=1000
    )
    
    use_instancing: BoolProperty(
        name="Use Instancing",
        description="Use instancing for better performance",
        default=False
    )
    
    instance_collection: PointerProperty(
        name="Instance Collection",
        description="Collection for instanced objects",
        type=bpy.types.Collection
    )
    
    max_undo_steps: IntProperty(
        name="Max Undo Steps",
        description="Maximum undo steps for chain operations",
        default=32,
        min=0,
        max=256
    )
    
    # Default Values
    reset_on_new: BoolProperty(
        name="Reset on New",
        description="Reset settings when creating new chain",
        default=False
    )
    
    save_defaults: BoolProperty(
        name="Save as Defaults",
        description="Save current settings as defaults",
        default=False
    )
    
    # Export/Import Settings
    export_format: EnumProperty(
        name="Export Format",
        description="Format for exporting chains",
        items=[
            ('BLEND', "Blend", "Blender native format"),
            ('FBX', "FBX", "FBX format"),
            ('OBJ', "OBJ", "Wavefront OBJ format"),
            ('GLTF', "glTF", "glTF 2.0 format"),
            ('STL', "STL", "STL format for 3D printing")
        ],
        default='BLEND'
    )
    
    export_path: StringProperty(
        name="Export Path",
        description="Default export path",
        default="//chains/",
        subtype='DIR_PATH'
    )
    
    auto_export: BoolProperty(
        name="Auto Export",
        description="Automatically export after creation",
        default=False
    )
    
    # Units and Precision
    use_metric: BoolProperty(
        name="Use Metric",
        description="Use metric units",
        default=True
    )
    
    precision: IntProperty(
        name="Precision",
        description="Decimal precision for values",
        default=3,
        min=1,
        max=6
    )
    
    # UI Settings
    compact_ui: BoolProperty(
        name="Compact UI",
        description="Use compact UI layout",
        default=False
    )
    
    show_tooltips: BoolProperty(
        name="Show Tooltips",
        description="Show extended tooltips",
        default=True
    )
    
    panel_width: IntProperty(
        name="Panel Width",
        description="Width of UI panels",
        default=300,
        min=200,
        max=500
    )
    
    # Color Theme
    theme: EnumProperty(
        name="Color Theme",
        description="Color theme for chain elements",
        items=[
            ('DEFAULT', "Default", "Default Blender colors"),
            ('MEDICAL', "Medical", "Medical/surgical theme"),
            ('TECHNICAL', "Technical", "Technical/engineering theme"),
            ('CUSTOM', "Custom", "Custom color theme")
        ],
        default='DEFAULT'
    )
    
    custom_theme_primary: FloatVectorProperty(
        name="Primary Color",
        description="Primary color for custom theme",
        default=(0.5, 0.5, 1.0, 1.0),
        size=4,
        min=0.0,
        max=1.0,
        subtype='COLOR_GAMMA'
    )
    
    custom_theme_secondary: FloatVectorProperty(
        name="Secondary Color",
        description="Secondary color for custom theme",
        default=(1.0, 0.5, 0.5, 1.0),
        size=4,
        min=0.0,
        max=1.0,
        subtype='COLOR_GAMMA'
    )
    
    # Advanced
    developer_mode: BoolProperty(
        name="Developer Mode",
        description="Enable developer features",
        default=False
    )
    
    experimental_features: BoolProperty(
        name="Experimental Features",
        description="Enable experimental features",
        default=False
    )
