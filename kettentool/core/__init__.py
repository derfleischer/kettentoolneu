"""
Kettentool Core Module
Core functionality, constants, and cache management
"""

# Import core modules
from . import constants
from . import cache_manager
from . import debug

def register():
    """Initialize core systems"""
    print("[CORE] Initializing core systems...")
    
    # Initialize constants
    if hasattr(constants, 'init'):
        constants.init()
        print("[CORE] ✅ Constants initialized")
    
    # Initialize cache manager
    if hasattr(cache_manager, 'init'):
        cache_manager.init()
        print("[CORE] ✅ Cache manager initialized")
    
    # Initialize debug system
    if hasattr(debug, 'init'):
        debug.init()
        print("[CORE] ✅ Debug system initialized")
    
    print("[CORE] ✅ Core systems ready")

def unregister():
    """Cleanup core systems"""
    print("[CORE] Cleaning up core systems...")
    
    # Cleanup in reverse order
    if hasattr(debug, 'cleanup'):
        debug.cleanup()
    
    if hasattr(cache_manager, 'cleanup'):
        cache_manager.cleanup()
        print("[CORE] ✅ Cache cleared")
    
    if hasattr(constants, 'cleanup'):
        constants.cleanup()
    
    print("[CORE] ✅ Core systems cleaned up")
