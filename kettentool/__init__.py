"""
Kettentool - Chain Construction Addon for Blender
Main initialization file
"""

bl_info = {
    "name": "Kettentool - Chain Construction",
    "author": "Your Name",
    "version": (1, 0, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Chain Tools",
    "description": "Advanced chain construction and painting tools",
    "warning": "",
    "doc_url": "",
    "category": "Add Mesh",
}

import bpy
import sys
import importlib
from pathlib import Path

# Add addon directory to path
addon_dir = Path(__file__).parent
if str(addon_dir) not in sys.path:
    sys.path.append(str(addon_dir))

# Import autoload system
from . import autoload

# Global variables for chain management
_chain_data = {}
_preview_handlers = []

def init_globals():
    """Initialize global variables"""
    global _chain_data, _preview_handlers
    _chain_data = {}
    _preview_handlers = []
    print("[INIT] Global variables initialized")

def cleanup_globals():
    """Clean up global variables"""
    global _chain_data, _preview_handlers
    
    # Clean up preview handlers
    for handler in _preview_handlers:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(handler, 'WINDOW')
        except:
            pass
    
    _preview_handlers.clear()
    _chain_data.clear()
    print("[INIT] Global variables cleaned up")

def clear_caches():
    """Clear Python caches"""
    # Clear module caches
    keys_to_remove = []
    for key in sys.modules.keys():
        if 'kettentool' in key:
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        try:
            del sys.modules[key]
        except:
            pass
    
    print(f"[INIT] Cleared {len(keys_to_remove)} cached modules")

def verify_registration():
    """Verify that all components are registered properly"""
    print("\n[VERIFICATION] Checking registration...")
    
    # Check for property groups
    if hasattr(bpy.types.Scene, 'chain_props'):
        print("[VERIFICATION] ✅ Property groups registered")
    else:
        print("[VERIFICATION] ❌ Property groups NOT registered")
    
    # Check for panels
    panel_count = 0
    for cls_name in dir(bpy.types):
        if cls_name.startswith('KETTE_PT_'):
            panel_count += 1
    
    if panel_count > 0:
        print(f"[VERIFICATION] ✅ {panel_count} panels registered")
    else:
        print("[VERIFICATION] ❌ No panels registered")
    
    # Check for operators
    operator_count = 0
    for cls_name in dir(bpy.types):
        if cls_name.startswith('KETTE_OT_'):
            operator_count += 1
    
    if operator_count > 0:
        print(f"[VERIFICATION] ✅ {operator_count} operators registered")
    else:
        print("[VERIFICATION] ⚠️ No operators registered")
    
    print("[VERIFICATION] Complete\n")

def register():
    """Register the addon"""
    print("\n" + "=" * 60)
    print("KETTENTOOL REGISTRATION STARTING")
    print("=" * 60)
    
    try:
        # Initialize globals
        init_globals()
        
        # Initialize autoload system
        autoload.init()
        
        # Register all modules
        autoload.register()
        
        # Verify registration
        verify_registration()
        
        print("[INIT] ✅ Kettentool addon registered successfully")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"[INIT] ❌ Registration failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to clean up
        try:
            unregister()
        except:
            pass
        
        raise

def unregister():
    """Unregister the addon"""
    print("\n" + "=" * 60)
    print("KETTENTOOL UNREGISTRATION")
    print("=" * 60)
    
    try:
        # Unregister all modules
        autoload.unregister()
        
        # Clean up globals
        cleanup_globals()
        
        # Clear caches
        clear_caches()
        
        print("[INIT] ✅ Kettentool addon unregistered")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"[INIT] ⚠️ Unregistration warning: {e}")

# For development: reload modules
if "bpy" in locals():
    print("[INIT] Reloading modules...")
    importlib.reload(autoload)
