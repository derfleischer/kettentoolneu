"""
Chain Tool V4 - Professional Orthotic Construction Tool
Main initialization file
"""

bl_info = {
    "name": "Chain Tool V4 - Orthotic Edition",
    "author": "Original: M.Fleischer, Refactored: Community",
    "version": (4, 0, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Chain Tool",
    "description": "Advanced dual-layer orthotic construction with intelligent patterns",
    "warning": "",
    "doc_url": "",
    "category": "Mesh",
}

import bpy
import importlib
import sys
from pathlib import Path

# Add addon path to sys.path for clean imports
addon_path = Path(__file__).parent
if str(addon_path) not in sys.path:
    sys.path.append(str(addon_path))

# Module imports - Order matters!
module_names = [
    # Core modules first
    "config",
    "core.constants", 
    "core.properties",
    "core.preferences",
    
    # Utilities
    "utils.debug",
    "utils.performance",
    "utils.caching",
    "utils.math_utils",
    
    # Geometry processing
    "geometry.mesh_analyzer",
    "geometry.triangulation",
    "geometry.edge_detector",
    
    # Pattern systems
    "patterns.base_pattern",
    "patterns.surface_patterns",
    "patterns.edge_patterns",
    
    # Operators
    "operators.pattern_generation",
    "operators.paint_mode",
    "operators.connection_tools",
    "operators.export_tools",
    
    # UI
    "ui.panels.main_panel",
    "ui.panels.pattern_panel",
    "ui.panels.paint_panel",
    "ui.panels.export_panel",
]

# Import handling
modules = []
for module_name in module_names:
    try:
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
        else:
            module = importlib.import_module(module_name)
            modules.append(module)
    except ImportError as e:
        print(f"[Chain Tool V4] Warning: Could not import {module_name}: {e}")

# Collect all classes to register
classes = []

def collect_classes():
    """Collect all registrable classes from modules"""
    global classes
    classes = []
    
    # Get classes from each module
    for module_name in module_names:
        if module_name in sys.modules:
            module = sys.modules[module_name]
            
            # Look for CLASSES list in module
            if hasattr(module, 'CLASSES'):
                classes.extend(module.CLASSES)
            
            # Also check for register_classes function
            elif hasattr(module, 'register_classes'):
                module_classes = module.register_classes()
                if module_classes:
                    classes.extend(module_classes)

def register():
    """Register all classes and setup handlers"""
    print("[Chain Tool V4] Starting registration...")
    
    # Collect classes
    collect_classes()
    
    # Register classes
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
            print(f"[Chain Tool V4] Registered: {cls.__name__}")
        except Exception as e:
            print(f"[Chain Tool V4] Failed to register {cls.__name__}: {e}")
    
    # Register properties
    from core.properties import register_properties
    register_properties()
    
    # Setup handlers if needed
    try:
        from ui.handlers import register_handlers
        register_handlers()
    except ImportError:
        pass
    
    print(f"[Chain Tool V4] Registration complete. {len(classes)} classes registered.")

def unregister():
    """Unregister all classes and cleanup"""
    print("[Chain Tool V4] Starting unregistration...")
    
    # Unregister handlers
    try:
        from ui.handlers import unregister_handlers
        unregister_handlers()
    except ImportError:
        pass
    
    # Unregister properties
    try:
        from core.properties import unregister_properties
        unregister_properties()
    except ImportError:
        pass
    
    # Unregister classes in reverse order
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception as e:
            print(f"[Chain Tool V4] Failed to unregister {cls.__name__}: {e}")
    
    # Clean up sys.path
    if str(addon_path) in sys.path:
        sys.path.remove(str(addon_path))
    
    print("[Chain Tool V4] Unregistration complete.")

# For development: Force reload of all modules
if "bpy" in locals():
    print("[Chain Tool V4] Reloading modules...")
    for module_name in module_names:
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
