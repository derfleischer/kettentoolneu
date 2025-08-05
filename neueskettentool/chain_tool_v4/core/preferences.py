"""
Addon Preferences for Chain Tool V4
User preferences and addon settings
"""

import bpy
from bpy.types import AddonPreferences
from bpy.props import (
    StringProperty, BoolProperty, EnumProperty,
    IntProperty, FloatProperty
)
import os
from pathlib import Path

# ============================================
# ADDON PREFERENCES
# ============================================

class ChainToolPreferences(AddonPreferences):
    bl_idname = "chain_tool_v4"
    
    # Library paths
    pattern_library_path: StringProperty(
        name="Pattern Library Path",
        description="Path to pattern library folder",
        default=str(Path.home() / "Documents" / "ChainTool_Patterns"),
        subtype='DIR_PATH'
    )
    
    export_path: StringProperty(
        name="Default Export Path",
        description="Default path for exports",
        default="//exports/",
        subtype='DIR_PATH'
    )
    
    # Debug settings
    enable_debug: BoolProperty(
        name="Enable Debug Mode",
        description="Enable debug logging and visualization",
        default=False,
        update=lambda self, context: self.update_debug_mode(context)
    )
    
    debug_level: EnumProperty(
        name="Debug Level",
        description="Debug verbosity level",
        items=[
            ('ERROR', "Errors Only", "Show only errors"),
            ('WARNING', "Warnings", "Show warnings and errors"),
            ('INFO', "Info", "Show info, warnings and errors"),
            ('DEBUG', "Debug", "Show debug messages"),
            ('TRACE', "Trace", "Show all messages including traces"),
        ],
        default='INFO',
        update=lambda self, context: self.update_debug_level(context)
    )
    
    # Performance settings
    enable_caching: BoolProperty(
        name="Enable Caching",
        description="Enable caching for better performance",
        default=True
    )
    
    cache_size_mb: IntProperty(
        name="Cache Size (MB)",
        description="Maximum cache size in megabytes",
        default=100,
        min=10,
        max=1000
    )
    
    background_threshold: IntProperty(
        name="Background Processing Threshold",
        description="Vertex count threshold for background processing",
        default=50000,
        min=1000,
        max=1000000
    )
    
    # UI settings
    use_custom_icons: BoolProperty(
        name="Use Custom Icons",
        description="Use custom icons in UI",
        default=True
    )
    
    panel_scale: FloatProperty(
        name="UI Scale",
        description="Scale factor for UI elements",
        default=1.0,
        min=0.5,
        max=2.0
    )
    
    show_tooltips: BoolProperty(
        name="Show Extended Tooltips",
        description="Show detailed tooltips",
        default=True
    )
    
    # Pattern defaults
    default_pattern_type: EnumProperty(
        name="Default Pattern",
        description="Default pattern type for new operations",
        items=[
            ('VORONOI', "Voronoi", "Organic cell pattern"),
            ('HEXAGONAL', "Hexagonal", "Honeycomb pattern"),
            ('TRIANGULAR', "Triangular", "Triangle mesh pattern"),
        ],
        default='VORONOI'
    )
    
    default_cell_size: FloatProperty(
        name="Default Cell Size",
        description="Default size for pattern cells",
        default=12.0,
        min=1.0,
        max=50.0,
        unit='LENGTH'
    )
    
    # Material presets
    use_material_presets: BoolProperty(
        name="Use Material Presets",
        description="Automatically apply material presets",
        default=True
    )
    
    shell_material_name: StringProperty(
        name="Shell Material",
        description="Name for shell material",
        default="TPU_Shell"
    )
    
    pattern_material_name: StringProperty(
        name="Pattern Material",
        description="Name for pattern material",
        default="PETG_Pattern"
    )
    
    # Export settings
    auto_export_backup: BoolProperty(
        name="Auto Export Backup",
        description="Automatically backup before export",
        default=True
    )
    
    export_scale: FloatProperty(
        name="Export Scale",
        description="Default export scale factor",
        default=10.0,
        min=0.001,
        max=1000.0
    )
    
    # Advanced settings
    show_advanced: BoolProperty(
        name="Show Advanced Settings",
        description="Show advanced settings in UI",
        default=False
    )
    
    experimental_features: BoolProperty(
        name="Enable Experimental Features",
        description="Enable experimental features (may be unstable)",
        default=False
    )
    
    def draw(self, context):
        layout = self.layout
        
        # Main settings
        box = layout.box()
        box.label(text="Library Settings", icon='FILE_FOLDER')
        box.prop(self, "pattern_library_path")
        box.prop(self, "export_path")
        
        # Debug settings
        box = layout.box()
        box.label(text="Debug Settings", icon='CONSOLE')
        box.prop(self, "enable_debug")
        if self.enable_debug:
            box.prop(self, "debug_level")
        
        # Performance settings
        box = layout.box()
        box.label(text="Performance", icon='TIME')
        box.prop(self, "enable_caching")
        if self.enable_caching:
            box.prop(self, "cache_size_mb")
        box.prop(self, "background_threshold")
        
        # UI settings
        box = layout.box()
        box.label(text="User Interface", icon='PREFERENCES')
        box.prop(self, "use_custom_icons")
        box.prop(self, "panel_scale")
        box.prop(self, "show_tooltips")
        
        # Pattern defaults
        box = layout.box()
        box.label(text="Pattern Defaults", icon='MESH_GRID')
        box.prop(self, "default_pattern_type")
        box.prop(self, "default_cell_size")
        
        # Material settings
        box = layout.box()
        box.label(text="Materials", icon='MATERIAL')
        box.prop(self, "use_material_presets")
        if self.use_material_presets:
            box.prop(self, "shell_material_name")
            box.prop(self, "pattern_material_name")
        
        # Export settings
        box = layout.box()
        box.label(text="Export", icon='EXPORT')
        box.prop(self, "auto_export_backup")
        box.prop(self, "export_scale")
        
        # Advanced settings
        box = layout.box()
        box.prop(self, "show_advanced", 
                text="Advanced Settings",
                icon='TRIA_DOWN' if self.show_advanced else 'TRIA_RIGHT')
        
        if self.show_advanced:
            box.prop(self, "experimental_features")
            
            # Buttons
            row = box.row(align=True)
            row.operator("chain_tool.reset_preferences", icon='FILE_REFRESH')
            row.operator("chain_tool.open_config_folder", icon='FILE_FOLDER')
    
    def update_debug_mode(self, context):
        """Update debug mode globally"""
        from utils.debug import debug
        debug.enabled = self.enable_debug
        
        if self.enable_debug:
            print(f"[Chain Tool V4] Debug mode enabled")
        else:
            print(f"[Chain Tool V4] Debug mode disabled")
    
    def update_debug_level(self, context):
        """Update debug level globally"""
        from utils.debug import debug
        debug.set_level(self.debug_level)

# ============================================
# PREFERENCE OPERATORS
# ============================================

class CHAIN_TOOL_OT_reset_preferences(bpy.types.Operator):
    """Reset all preferences to defaults"""
    bl_idname = "chain_tool.reset_preferences"
    bl_label = "Reset to Defaults"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        prefs = context.preferences.addons["chain_tool_v4"].preferences
        
        # Reset all properties to defaults
        for prop in prefs.bl_rna.properties:
            if not prop.is_readonly and prop.identifier != "bl_idname":
                try:
                    setattr(prefs, prop.identifier, prop.default)
                except:
                    pass
        
        self.report({'INFO'}, "Preferences reset to defaults")
        return {'FINISHED'}
    
    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

class CHAIN_TOOL_OT_open_config_folder(bpy.types.Operator):
    """Open addon configuration folder"""
    bl_idname = "chain_tool.open_config_folder"
    bl_label = "Open Config Folder"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        import subprocess
        import sys
        from config import ADDON_PATH
        
        # Open folder in system file browser
        if sys.platform == "win32":
            subprocess.Popen(["explorer", str(ADDON_PATH)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(ADDON_PATH)])
        else:  # Linux
            subprocess.Popen(["xdg-open", str(ADDON_PATH)])
        
        self.report({'INFO'}, f"Opened: {ADDON_PATH}")
        return {'FINISHED'}

# ============================================
# UTILITY FUNCTIONS
# ============================================

def get_preferences() -> ChainToolPreferences:
    """Get addon preferences"""
    return bpy.context.preferences.addons["chain_tool_v4"].preferences

def get_pattern_library_path() -> Path:
    """Get pattern library path from preferences"""
    prefs = get_preferences()
    path = Path(prefs.pattern_library_path)
    
    # Ensure path exists
    path.mkdir(parents=True, exist_ok=True)
    
    return path

def get_export_path() -> Path:
    """Get export path from preferences"""
    prefs = get_preferences()
    
    # Handle relative paths
    if prefs.export_path.startswith("//"):
        if bpy.data.filepath:
            base_path = Path(bpy.data.filepath).parent
            rel_path = prefs.export_path[2:]
            path = base_path / rel_path
        else:
            # No file saved, use temp
            path = Path(bpy.app.tempdir) / "chain_tool_exports"
    else:
        path = Path(prefs.export_path)
    
    # Ensure path exists
    path.mkdir(parents=True, exist_ok=True)
    
    return path

# ============================================
# REGISTRATION
# ============================================

CLASSES = [
    ChainToolPreferences,
    CHAIN_TOOL_OT_reset_preferences,
    CHAIN_TOOL_OT_open_config_folder,
]
