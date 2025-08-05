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
import sys
import traceback
from pathlib import Path

# Add addon path to sys.path for clean imports
addon_path = Path(__file__).parent
if str(addon_path) not in sys.path:
    sys.path.append(str(addon_path))

# Global list for registered classes
classes = []

def register():
    """Register the addon"""
    print("\n" + "="*60)
    print("CHAIN TOOL V4 - REGISTRATION")
    print("="*60)
    
    global classes
    classes = []
    
    # Import and register core module FIRST (contains properties)
    try:
        from . import core
        if hasattr(core, 'register'):
            core.register()
        if hasattr(core, 'classes'):
            classes.extend(core.classes)
        print("✓ Core module registered")
    except Exception as e:
        print(f"✗ Core module failed: {e}")
        traceback.print_exc()
    
    # Import and register utils
    try:
        from . import utils
        if hasattr(utils, 'register'):
            utils.register()
        if hasattr(utils, 'classes'):
            classes.extend(utils.classes)
        print("✓ Utils module registered")
    except Exception as e:
        print(f"✗ Utils module failed: {e}")
    
    # Import and register geometry
    try:
        from . import geometry
        if hasattr(geometry, 'register'):
            geometry.register()
        if hasattr(geometry, 'classes'):
            classes.extend(geometry.classes)
        print("✓ Geometry module registered")
    except Exception as e:
        print(f"✗ Geometry module failed: {e}")
    
    # Import and register algorithms
    try:
        from . import algorithms
        if hasattr(algorithms, 'register'):
            algorithms.register()
        if hasattr(algorithms, 'classes'):
            classes.extend(algorithms.classes)
        print("✓ Algorithms module registered")
    except Exception as e:
        print(f"✗ Algorithms module failed: {e}")
    
    # Import and register patterns
    try:
        from . import patterns
        if hasattr(patterns, 'register'):
            patterns.register()
        if hasattr(patterns, 'classes'):
            classes.extend(patterns.classes)
        print("✓ Patterns module registered")
    except Exception as e:
        print(f"✗ Patterns module failed: {e}")
    
    # Import and register operators
    try:
        from . import operators
        if hasattr(operators, 'register'):
            operators.register()
        if hasattr(operators, 'classes'):
            classes.extend(operators.classes)
        print("✓ Operators module registered")
    except Exception as e:
        print(f"✗ Operators module failed: {e}")
    
    # Import and register UI
    try:
        from . import ui
        if hasattr(ui, 'register'):
            ui.register()
        if hasattr(ui, 'classes'):
            classes.extend(ui.classes)
        print("✓ UI module registered")
    except Exception as e:
        print(f"✗ UI module failed: {e}")
    
    # Import and register presets
    try:
        from . import presets
        if hasattr(presets, 'register'):
            presets.register()
        if hasattr(presets, 'classes'):
            classes.extend(presets.classes)
        print("✓ Presets module registered")
    except Exception as e:
        print(f"✗ Presets module failed: {e}")
    
    # Register all collected classes
    registered = 0
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
            registered += 1
        except ValueError:
            # Class already registered
            pass
        except Exception as e:
            print(f"✗ Failed to register {cls.__name__}: {e}")
    
    print(f"\n✓ Registration complete: {registered}/{len(classes)} classes registered")
    
    # State Manager initialization - DELAYED
    def init_state_manager():
        """Initialize state manager when context is available"""
        try:
            from .core.state_manager import StateManager
            state = StateManager()
            state.initialize()
            print("✓ State Manager initialized")
        except Exception as e:
            print(f"✗ State Manager initialization failed: {e}")
        
        # Check properties
        try:
            if hasattr(bpy.types.Scene, 'chain_tool_properties'):
                print("✓ Properties system active")
                # Test if we can access it
                for scene in bpy.data.scenes:
                    if hasattr(scene, 'chain_tool_properties'):
                        print(f"  ✓ Properties accessible in scene: {scene.name}")
                        break
            else:
                print("⚠ Properties system not found")
        except Exception as e:
            print(f"⚠ Could not check properties: {e}")
        
        return None  # Timer entfernen
    
    # Verzögere State Manager Init mit Timer
    if bpy.app.timers.is_registered(init_state_manager):
        bpy.app.timers.unregister(init_state_manager)
    bpy.app.timers.register(init_state_manager, first_interval=0.5)
    
    print("="*60 + "\n")

def unregister():
    """Unregister the addon"""
    print("\nChain Tool V4: Unregistering...")
    
    global classes
    
    # Remove any timers
    def init_state_manager():
        pass
    
    if bpy.app.timers.is_registered(init_state_manager):
        bpy.app.timers.unregister(init_state_manager)
    
    # Unregister UI first
    try:
        from . import ui
        if hasattr(ui, 'unregister'):
            ui.unregister()
    except:
        pass
    
    # Unregister operators
    try:
        from . import operators
        if hasattr(operators, 'unregister'):
            operators.unregister()
    except:
        pass
    
    # Unregister patterns
    try:
        from . import patterns
        if hasattr(patterns, 'unregister'):
            patterns.unregister()
    except:
        pass
    
    # Unregister algorithms
    try:
        from . import algorithms
        if hasattr(algorithms, 'unregister'):
            algorithms.unregister()
    except:
        pass
    
    # Unregister geometry
    try:
        from . import geometry
        if hasattr(geometry, 'unregister'):
            geometry.unregister()
    except:
        pass
    
    # Unregister utils
    try:
        from . import utils
        if hasattr(utils, 'unregister'):
            utils.unregister()
    except:
        pass
    
    # Unregister presets
    try:
        from . import presets
        if hasattr(presets, 'unregister'):
            presets.unregister()
    except:
        pass
    
    # Unregister core last
    try:
        from . import core
        if hasattr(core, 'unregister'):
            core.unregister()
    except:
        pass
    
    # Unregister all classes
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except:
            pass
    
    # Clean up sys.path
    if str(addon_path) in sys.path:
        sys.path.remove(str(addon_path))
    
    print("Chain Tool V4: Unregistration complete\n")

if __name__ == "__main__":
    register()
