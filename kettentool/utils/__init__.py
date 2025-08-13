"""
Kettentool Utils Module
Utility functions for collections, materials, surface projection, and painting
"""

# Import utility modules
from . import collections
from . import materials
from . import surface
from . import paintmode

def register():
    """Initialize utility systems"""
    print("[UTILS] Initializing utilities...")
    
    # Initialize materials system
    if hasattr(materials, 'init'):
        materials.init()
        print("[UTILS] ✅ Materials system initialized")
    
    # Initialize surface system
    if hasattr(surface, 'init'):
        surface.init()
        print("[UTILS] ✅ Surface system initialized")
    
    # Initialize paintmode utilities
    if hasattr(paintmode, 'init'):
        paintmode.init()
        print("[UTILS] ✅ Paint mode utilities initialized")
    
    # Initialize collections manager
    if hasattr(collections, 'init'):
        collections.init()
        print("[UTILS] ✅ Collections manager initialized")
    
    print("[UTILS] ✅ All utilities ready")

def unregister():
    """Cleanup utility systems"""
    print("[UTILS] Cleaning up utilities...")
    
    # Cleanup in reverse order
    if hasattr(paintmode, 'cleanup'):
        paintmode.cleanup()
    
    if hasattr(surface, 'cleanup'):
        surface.cleanup()
    
    if hasattr(materials, 'cleanup'):
        materials.cleanup()
    
    if hasattr(collections, 'cleanup'):
        collections.cleanup()
    
    print("[UTILS] ✅ Utilities cleaned up")
