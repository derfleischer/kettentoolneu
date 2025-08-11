"""
KETTENTOOL - Main Registration
Blender Addon für Ketten-Konstruktion mit modularer Architektur
"""

bl_info = {
    "name": "Chain Construction Tool",
    "author": "Your Name",
    "version": (3, 6, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Chain Tools",
    "description": "Advanced chain construction tool for medical and technical applications",
    "category": "Mesh",
}

# =========================================
# IMPORTS
# =========================================

import bpy
from . import auto_load

# Import core modules
from .core import debug, constants, cache_manager
from .utils import collections, materials, surface
from .creation import spheres, connectors
from .ui import panels, properties
from .operators import basic_ops

# =========================================
# INITIALIZATION
# =========================================

def initialize_globals():
    """Initialisiert globale Instanzen"""
    # Debug System
    constants.debug = debug.DebugLogger()
    constants.performance_monitor = debug.PerformanceMonitor()
    
    # Aktiviere Debug standardmäßig
    constants.debug.enabled = True
    constants.debug.level = 'INFO'
    
    constants.debug.info('INIT', "Kettentool globals initialized")

# =========================================
# REGISTRATION
# =========================================

# Auto-Load System für automatische Registrierung aller Module
auto_load.init()

def register():
    """Registriert das Addon"""
    try:
        # Initialisiere globale Systeme
        initialize_globals()
        
        # Auto-registriere alle Module
        auto_load.register()
        
        # Registriere Property Groups an Scene
        bpy.types.Scene.chain_construction_props = bpy.props.PointerProperty(
            type=properties.ChainConstructionProperties
        )
        
        constants.debug.info('INIT', "Kettentool addon registered successfully")
        
    except Exception as e:
        print(f"Error registering Kettentool: {e}")
        import traceback
        traceback.print_exc()

def unregister():
    """Deregistriert das Addon"""
    try:
        # Cleanup
        cache_manager.cleanup_all_caches()
        
        # Auto-deregistriere alle Module  
        auto_load.unregister()
        
        # Entferne Property Groups
        if hasattr(bpy.types.Scene, 'chain_construction_props'):
            del bpy.types.Scene.chain_construction_props
        
        if constants.debug:
            constants.debug.info('INIT', "Kettentool addon unregistered")
            
    except Exception as e:
        print(f"Error unregistering Kettentool: {e}")

if __name__ == "__main__":
    register()
