"""
Chain Tool V4 - Presets Module
==============================

Material presets and orthotic templates for Chain Tool V4.
Includes TPU/PETG material settings and printer profiles.
"""

import bpy

# Import preset modules
try:
    from . import material_presets
    from . import orthotic_presets
    
    # Make key presets available at package level
    from .material_presets import (
        TPU_PRESETS,
        PETG_PRESETS, 
        PRINTER_PROFILES,
        QUALITY_PROFILES,
        get_material_preset,
        get_printer_profile,
        create_dual_material_profile
    )
    
    from .orthotic_presets import (
        ORTHOTIC_TEMPLATES,
        get_orthotic_template,
        create_orthotic_preset
    )
    
    _available_modules = ['material_presets', 'orthotic_presets']
    
except ImportError as e:
    print(f"Chain Tool V4 Presets: Import warning - {e}")
    _available_modules = []

__all__ = [
    'material_presets',
    'orthotic_presets',
    'TPU_PRESETS',
    'PETG_PRESETS',
    'PRINTER_PROFILES', 
    'QUALITY_PROFILES',
    'ORTHOTIC_TEMPLATES',
    'get_material_preset',
    'get_printer_profile',
    'create_dual_material_profile',
    'get_orthotic_template',
    'create_orthotic_preset'
]

def register():
    """Register preset modules"""
    try:
        # Register material presets
        if 'material_presets' in _available_modules:
            material_presets.register()
        
        # Register orthotic presets  
        if 'orthotic_presets' in _available_modules:
            orthotic_presets.register()
        
        print(f"Chain Tool V4 Presets: {len(_available_modules)} modules registered")
        
    except Exception as e:
        print(f"Chain Tool V4 Presets: Registration warning - {e}")

def unregister():
    """Unregister preset modules"""
    try:
        # Unregister in reverse order
        if 'orthotic_presets' in _available_modules:
            orthotic_presets.unregister()
        
        if 'material_presets' in _available_modules:
            material_presets.unregister()
        
        print("Chain Tool V4 Presets: Unregistered")
        
    except Exception as e:
        print(f"Chain Tool V4 Presets: Unregistration warning - {e}")

# Module info
__version__ = "4.0.0"
