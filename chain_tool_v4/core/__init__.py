"""
Chain Tool V4 - Core Module
==========================
Core functionality and properties
"""

import bpy

# Import all core components
classes = []

# Import with fallbacks
try:
    from .constants import *
except ImportError as e:
    print(f"Chain Tool V4 - Core: Could not import constants: {e}")

try:
    from .properties import ChainToolProperties, get_chain_tool_properties
except ImportError as e:
    print(f"Chain Tool V4 - Core: Could not import properties: {e}")
    ChainToolProperties = None
    get_chain_tool_properties = None

try:
    from .state_manager import StateManager, PatternState, PaintState, ExportState
except ImportError as e:
    print(f"Chain Tool V4 - Core: Could not import state_manager: {e}")
    StateManager = None
    PatternState = None
    PaintState = None
    ExportState = None

try:
    from .preferences import ChainToolPreferences
    classes.append(ChainToolPreferences)
except ImportError:
    # Preferences might not exist yet
    ChainToolPreferences = None

# Collect classes
if ChainToolProperties:
    classes.append(ChainToolProperties)

# Export
__all__ = ['classes']

# Add successfully imported items to __all__
if ChainToolProperties:
    __all__.extend(['ChainToolProperties', 'get_chain_tool_properties'])

if StateManager:
    __all__.extend(['StateManager', 'PatternState', 'PaintState', 'ExportState'])

if ChainToolPreferences:
    __all__.append('ChainToolPreferences')

def register():
    """Register core module"""
    print("Chain Tool V4: Registering core module...")
    
    # Register properties class
    if ChainToolProperties:
        try:
            bpy.utils.register_class(ChainToolProperties)
            print("  ✓ ChainToolProperties registered")
        except ValueError:
            # Klasse bereits registriert
            print("  ⚠ ChainToolProperties already registered")
        except Exception as e:
            print(f"  ✗ Could not register ChainToolProperties: {e}")
    
    # Add properties to Scene
    if ChainToolProperties:
        try:
            if not hasattr(bpy.types.Scene, 'chain_tool_properties'):
                bpy.types.Scene.chain_tool_properties = bpy.props.PointerProperty(
                    type=ChainToolProperties
                )
                print("  ✓ Properties added to Scene")
            else:
                print("  ⚠ Properties already in Scene")
        except Exception as e:
            print(f"  ✗ Could not add properties to Scene: {e}")
    
    # Register preferences if available
    if ChainToolPreferences:
        try:
            bpy.utils.register_class(ChainToolPreferences)
            print("  ✓ ChainToolPreferences registered")
        except ValueError:
            print("  ⚠ ChainToolPreferences already registered")
        except Exception as e:
            print(f"  ✗ Could not register ChainToolPreferences: {e}")
    
    # State manager initialization happens later via timer in main __init__.py
    print("Chain Tool V4: Core module registered")

def unregister():
    """Unregister core module"""
    print("Chain Tool V4: Unregistering core module...")
    
    # Remove properties from Scene
    if hasattr(bpy.types.Scene, 'chain_tool_properties'):
        try:
            del bpy.types.Scene.chain_tool_properties
            print("  ✓ Properties removed from Scene")
        except:
            pass
    
    # Unregister classes
    if ChainToolProperties:
        try:
            bpy.utils.unregister_class(ChainToolProperties)
            print("  ✓ ChainToolProperties unregistered")
        except:
            pass
    
    if ChainToolPreferences:
        try:
            bpy.utils.unregister_class(ChainToolPreferences)
            print("  ✓ ChainToolPreferences unregistered")
        except:
            pass
    
    print("Chain Tool V4: Core module unregistered")
