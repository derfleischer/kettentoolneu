"""
Chain Tool V4 - Advanced Pattern Generator for Orthotic 3D Printing
==================================================================

Main registration file for the complete Chain Tool V4 Blender addon.
Coordinates all modules and ensures proper initialization order.

Version: 4.0.0
Author: Chain Tool Development Team
License: GPL v3
"""

bl_info = {
    "name": "Chain Tool V4 - Advanced Pattern Generator",
    "author": "Chain Tool Development Team",
    "version": (4, 0, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Chain Tool",
    "description": "Advanced dual-material pattern generation for orthotic 3D printing",
    "category": "Mesh",
    "doc_url": "https://github.com/chain-tool/chain-tool-v4",
    "tracker_url": "https://github.com/chain-tool/chain-tool-v4/issues",
    "support": "COMMUNITY",
}

import bpy
import sys
from pathlib import Path

# Add addon directory to Python path for relative imports
addon_dir = Path(__file__).parent
if str(addon_dir) not in sys.path:
    sys.path.append(str(addon_dir))

# Import all module packages in dependency order
try:
    from . import core
    from . import utils
    from . import geometry
    from . import algorithms
    from . import patterns
    from . import operators
    from . import ui
    from . import presets
    
    # Store module references for debugging
    _modules = [core, utils, geometry, algorithms, patterns, operators, ui, presets]
    
except ImportError as e:
    print(f"Chain Tool V4: Import error - {e}")
    _modules = []


def register():
    """
    Register all Chain Tool V4 modules in correct dependency order.
    
    Order is critical:
    1. core - Properties, State Manager, Constants
    2. utils - Caching, Performance, Debug, Math utilities
    3. geometry - Mesh analysis, triangulation, edge detection
    4. algorithms - Delaunay, optimization, stress analysis
    5. patterns - Pattern generation classes
    6. operators - All Blender operators
    7. ui - Panels, overlays, menus
    8. presets - Material and printer presets
    """
    
    print("Chain Tool V4: Starting registration...")
    
    try:
        # Register core module (Properties, State Manager)
        core.register()
        print("  ✅ Core module registered")
        
        # Register utility modules
        utils.register()
        print("  ✅ Utils modules registered")
        
        # Register geometry analysis
        geometry.register()
        print("  ✅ Geometry modules registered")
        
        # Register algorithms
        algorithms.register()
        print("  ✅ Algorithm modules registered")
        
        # Register pattern generators
        patterns.register()
        print("  ✅ Pattern modules registered")
        
        # Register operators
        operators.register()
        print("  ✅ Operator modules registered")
        
        # Register UI system
        ui.register()
        print("  ✅ UI modules registered")
        
        # Register presets
        presets.register()
        print("  ✅ Preset modules registered")
        
        print("Chain Tool V4: Registration complete! 🎉")
        
        # Initialize state manager
        from .core.state_manager import StateManager
        state_manager = StateManager()
        state_manager.initialize()
        
        # Verify critical components
        if hasattr(bpy.context.scene, 'chain_tool_properties'):
            print("  ✅ Properties system active")
        else:
            print("  ⚠️  Properties system not found")
            
    except Exception as e:
        print(f"Chain Tool V4: Registration failed - {e}")
        import traceback
        traceback.print_exc()


def unregister():
    """
    Unregister all Chain Tool V4 modules in reverse order.
    Ensures clean cleanup of all resources.
    """
    
    print("Chain Tool V4: Starting unregistration...")
    
    try:
        # Cleanup state manager first
        from .core.state_manager import StateManager
        state_manager = StateManager()
        state_manager.cleanup()
        
        # Unregister in reverse order
        presets.unregister()
        print("  ✅ Preset modules unregistered")
        
        ui.unregister()
        print("  ✅ UI modules unregistered")
        
        operators.unregister()
        print("  ✅ Operator modules unregistered") 
        
        patterns.unregister()
        print("  ✅ Pattern modules unregistered")
        
        algorithms.unregister()
        print("  ✅ Algorithm modules unregistered")
        
        geometry.unregister()
        print("  ✅ Geometry modules unregistered")
        
        utils.unregister()
        print("  ✅ Utils modules unregistered")
        
        core.unregister()
        print("  ✅ Core module unregistered")
        
        print("Chain Tool V4: Unregistration complete! 👋")
        
    except Exception as e:
        print(f"Chain Tool V4: Unregistration error - {e}")
        import traceback
        traceback.print_exc()


# For development and testing
if __name__ == "__main__":
    register()
    
    # Development test
    try:
        import bpy
        if hasattr(bpy.context.scene, 'chain_tool_properties'):
            props = bpy.context.scene.chain_tool_properties
            print(f"Chain Tool V4: Development test successful")
            print(f"  Pattern types available: {len(props.bl_rna.properties['pattern_type'].enum_items)}")
        else:
            print("Chain Tool V4: Development test failed - no properties")
    except Exception as e:
        print(f"Chain Tool V4: Development test error - {e}")
