"""
Panel Module for Kettentool
Manages all UI panels
"""

import bpy

# Import all panel modules
from . import main_panel
from . import tool_panels
from . import pattern_panel
from . import paint_panel
from . import settings_panels

# List of all panel modules
panel_modules = [
    main_panel,
    tool_panels,
    pattern_panel,
    paint_panel,
    settings_panels,
]

# List to track registered classes
registered_classes = []

def register():
    """Register all panels"""
    print("[PANELS] Starting panel registration...")
    
    # Register each panel module
    for module in panel_modules:
        # Check if module has its own register function
        if hasattr(module, 'register'):
            try:
                module.register()
                print(f"[PANELS] ✅ Registered module: {module.__name__}")
            except Exception as e:
                print(f"[PANELS] ⚠️ Failed to register {module.__name__}: {e}")
        
        # Also register individual panel classes
        for attr_name in dir(module):
            if attr_name.startswith('_'):
                continue
                
            attr = getattr(module, attr_name)
            
            # Check if it's a Panel class
            if isinstance(attr, type) and issubclass(attr, bpy.types.Panel):
                try:
                    # Check if not already registered
                    if not hasattr(bpy.types, attr.__name__):
                        bpy.utils.register_class(attr)
                        registered_classes.append(attr)
                        print(f"[PANELS] ✅ Registered panel: {attr.__name__}")
                except Exception as e:
                    if "already registered" not in str(e):
                        print(f"[PANELS] ⚠️ Failed to register {attr.__name__}: {e}")
    
    print(f"[PANELS] ✅ Panel registration complete - {len(registered_classes)} panels registered")

def unregister():
    """Unregister all panels"""
    print("[PANELS] Starting panel unregistration...")
    
    # Unregister in reverse order
    for cls in reversed(registered_classes):
        try:
            bpy.utils.unregister_class(cls)
            print(f"[PANELS] ✅ Unregistered panel: {cls.__name__}")
        except Exception as e:
            if "not registered" not in str(e):
                print(f"[PANELS] ⚠️ Failed to unregister {cls.__name__}: {e}")
    
    registered_classes.clear()
    
    # Unregister each module
    for module in reversed(panel_modules):
        if hasattr(module, 'unregister'):
            try:
                module.unregister()
                print(f"[PANELS] ✅ Unregistered module: {module.__name__}")
            except Exception as e:
                print(f"[PANELS] ⚠️ Failed to unregister {module.__name__}: {e}")
    
    print("[PANELS] ✅ Panel unregistration complete")
