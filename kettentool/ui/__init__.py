"""
Kettentool UI Module
Handles Property Groups and User Interface
"""

import bpy

# Import submodules - FIXED: using correct module names
from . import property_groups
from . import panels

def register():
    """Register UI components"""
    print("[UI] Starting UI registration...")
    
    # Register property groups first (panels depend on them)
    property_groups.register()
    
    # Then register panels
    panels.register()
    
    print("[UI] ✅ UI registration complete")

def unregister():
    """Unregister UI components"""
    print("[UI] Starting UI unregistration...")
    
    # Unregister in reverse order
    panels.unregister()
    property_groups.unregister()
    
    print("[UI] ✅ UI unregistration complete")
