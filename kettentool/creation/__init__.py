"""
Kettentool Creation Module
Object creation systems for spheres, connectors, and patterns
"""

# Import creation modules
from . import spheres
from . import connectors
from . import pattern_geometry

def register():
    """Initialize creation systems"""
    print("[CREATION] Initializing creation systems...")
    
    # Initialize sphere creation system
    if hasattr(spheres, 'init'):
        spheres.init()
        print("[CREATION] ✅ Sphere system initialized")
    elif hasattr(spheres, 'register'):
        spheres.register()
        print("[CREATION] ✅ Sphere system registered")
    
    # Initialize connector creation system
    if hasattr(connectors, 'init'):
        connectors.init()
        print("[CREATION] ✅ Connector system initialized")
    elif hasattr(connectors, 'register'):
        connectors.register()
        print("[CREATION] ✅ Connector system registered")
    
    # Initialize pattern geometry system
    if hasattr(pattern_geometry, 'init'):
        pattern_geometry.init()
        print("[CREATION] ✅ Pattern geometry system initialized")
    elif hasattr(pattern_geometry, 'register'):
        pattern_geometry.register()
        print("[CREATION] ✅ Pattern geometry system registered")
    
    print("[CREATION] ✅ All creation systems ready")

def unregister():
    """Cleanup creation systems"""
    print("[CREATION] Cleaning up creation systems...")
    
    # Cleanup in reverse order
    if hasattr(pattern_geometry, 'cleanup'):
        pattern_geometry.cleanup()
    elif hasattr(pattern_geometry, 'unregister'):
        pattern_geometry.unregister()
    
    if hasattr(connectors, 'cleanup'):
        connectors.cleanup()
    elif hasattr(connectors, 'unregister'):
        connectors.unregister()
    
    if hasattr(spheres, 'cleanup'):
        spheres.cleanup()
    elif hasattr(spheres, 'unregister'):
        spheres.unregister()
    
    print("[CREATION] ✅ Creation systems cleaned up")
