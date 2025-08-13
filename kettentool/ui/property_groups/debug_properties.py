"""
Debug Properties for Chain Construction Tool
Debug and performance monitoring settings
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


class DebugProperties(PropertyGroup):
    """Debug and performance properties"""
    
    # Debug Display
    show_info: BoolProperty(
        name="Show Debug Info",
        description="Display debug information in viewport",
        default=False
    )
    
    debug_level: EnumProperty(
        name="Debug Level",
        description="Level of debug information",
        items=[
            ('ERROR', "Errors Only", "Show only errors"),
            ('WARNING', "Warnings", "Show warnings and errors"),
            ('INFO', "Info", "Show info, warnings and errors"),
            ('DEBUG', "Debug", "Show all debug information"),
            ('VERBOSE', "Verbose", "Show verbose debug output")
        ],
        default='WARNING'
    )
    
    # Visualization Options
    show_normals: BoolProperty(
        name="Show Normals",
        description="Display surface normals",
        default=False
    )
    
    normal_size: FloatProperty(
        name="Normal Size",
        description="Display size of normals",
        default=0.1,
        min=0.01,
        max=1.0,
        subtype='DISTANCE'
    )
    
    normal_color: FloatVectorProperty(
        name="Normal Color",
        description="Color for normal display",
        default=(0.0, 0.0, 1.0, 1.0),
        size=4,
        min=0.0,
        max=1.0,
        subtype='COLOR_GAMMA'
    )
    
    show_weights: BoolProperty(
        name="Show Weights",
        description="Display vertex weights",
        default=False
    )
    
    weight_gradient: EnumProperty(
        name="Weight Gradient",
        description="Gradient for weight display",
        items=[
            ('RG', "Red-Green", "Red to green gradient"),
            ('BW', "Black-White", "Black to white gradient"),
            ('RAINBOW', "Rainbow", "Rainbow gradient"),
            ('HEAT', "Heat Map", "Heat map gradient")
        ],
        default='RG'
    )
    
    show_connections: BoolProperty(
        name="Show Connections",
        description="Display connection lines",
        default=False
    )
    
    connection_color: FloatVectorProperty(
        name="Connection Color",
        description="Color for connection lines",
        default=(1.0, 1.0, 0.0, 0.5),
        size=4,
        min=0.0,
        max=1.0,
        subtype='COLOR_GAMMA'
    )
    
    connection_width: FloatProperty(
        name="Connection Width",
        description="Width of connection lines",
        default=2.0,
        min=1.0,
        max=10.0
    )
    
    show_indices: BoolProperty(
        name="Show Indices",
        description="Display object indices",
        default=False
    )
    
    show_bounding_box: BoolProperty(
        name="Show Bounding Boxes",
        description="Display bounding boxes",
        default=False
    )
    
    # Performance Monitoring
    profile_operations: BoolProperty(
        name="Profile Operations",
        description="Profile performance of operations",
        default=False
    )
    
    show_statistics: BoolProperty(
        name="Show Statistics",
        description="Display statistics in viewport",
        default=False
    )
    
    fps_target: FloatProperty(
        name="Target FPS",
        description="Target frames per second",
        default=30.0,
        min=1.0,
        max=120.0
    )
    
    # Memory Management
    show_memory: BoolProperty(
        name="Show Memory Usage",
        description="Display memory usage information",
        default=False
    )
    
    cache_size: FloatProperty(
        name="Cache Size",
        description="Current cache size in MB",
        default=0.0,
        min=0.0
    )
    
    cache_objects: IntProperty(
        name="Cached Objects",
        description="Number of cached objects",
        default=0,
        min=0
    )
    
    auto_clear_cache: BoolProperty(
        name="Auto Clear Cache",
        description="Automatically clear cache when needed",
        default=True
    )
    
    cache_limit: FloatProperty(
        name="Cache Limit",
        description="Maximum cache size in MB",
        default=100.0,
        min=1.0,
        max=1000.0
    )
    
    # Validation
    auto_validate: BoolProperty(
        name="Auto Validate",
        description="Automatically validate chain integrity",
        default=False
    )
    
    last_validation_result: EnumProperty(
        name="Last Validation",
        description="Result of last validation",
        items=[
            ('NONE', "Not Validated", "Not validated yet"),
            ('VALID', "Valid", "Chain is valid"),
            ('WARNING', "Warning", "Chain has warnings"),
            ('ERROR', "Error", "Chain has errors")
        ],
        default='NONE'
    )
    
    # Logging
    enable_logging: BoolProperty(
        name="Enable Logging",
        description="Enable operation logging",
        default=False
    )
    
    log_file: StringProperty(
        name="Log File",
        description="Path to log file",
        default="//kettentool.log",
        subtype='FILE_PATH'
    )
    
    log_level: EnumProperty(
        name="Log Level",
        description="Logging level",
        items=[
            ('ERROR', "Error", "Log only errors"),
            ('WARNING', "Warning", "Log warnings and errors"),
            ('INFO', "Info", "Log info, warnings and errors"),
            ('DEBUG', "Debug", "Log everything")
        ],
        default='INFO'
    )
    
    # Test Mode
    test_mode: BoolProperty(
        name="Test Mode",
        description="Enable test mode features",
        default=False
    )
    
    # Timing
    last_update_time: FloatProperty(
        name="Last Update Time",
        description="Time of last update in seconds",
        default=0.0,
        min=0.0
    )
    
    fps_current: FloatProperty(
        name="Current FPS",
        description="Current frames per second",
        default=0.0,
        min=0.0
    )
    
    # Legacy compatibility
    show_debug_info: BoolProperty(
        name="Debug Info (Legacy)",
        default=False,
        get=lambda self: self.show_info,
        set=lambda self, value: setattr(self, 'show_info', value)
    )
    
    debug_show_normals: BoolProperty(
        name="Show Normals (Legacy)",
        default=False,
        get=lambda self: self.show_normals,
        set=lambda self, value: setattr(self, 'show_normals', value)
    )
