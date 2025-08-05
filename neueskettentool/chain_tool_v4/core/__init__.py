"""
Chain Tool V4 - Core Module
==========================
Core functionality and properties
"""

# Import all core components
classes = []

# Import with fallbacks
try:
    from .constants import *
except ImportError as e:
    print(f"Chain Tool V4 - Core: Could not import constants: {e}")

try:
    from .properties import ChainToolProperties
    classes.append(ChainToolProperties)
except ImportError as e:
    print(f"Chain Tool V4 - Core: Could not import properties: {e}")
    ChainToolProperties = None

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

# Export
__all__ = ['classes']

# Add successfully imported items to __all__
if ChainToolProperties:
    __all__.append('ChainToolProperties')

if StateManager:
    __all__.extend(['StateManager', 'PatternState', 'PaintState', 'ExportState'])

if ChainToolPreferences:
    __all__.append('ChainToolPreferences')

# Also export all constants
__all__.extend([
    'PatternType', 'SurfacePatternType', 'EdgePatternType', 'HybridPatternType',
    'MaterialType', 'ExportFormat', 'QualityLevel', 'PrinterProfile',
    'AlgorithmType', 'LoadCase', 'StressType', 'CollisionType',
    'ResolutionStrategy', 'GapType', 'FillStrategy', 'ConnectionType',
    'TriangulationMethod', 'DEFAULT_SPHERE_SIZE', 'DEFAULT_CONNECTION_THICKNESS',
    'DEFAULT_PATTERN_DENSITY', 'DEFAULT_SHELL_THICKNESS'
])

def register():
    """Register core module"""
    import bpy
    
    # Register properties
    if ChainToolProperties:
        try:
            bpy.utils.register_class(ChainToolProperties)
            bpy.types.Scene.chain_tool_properties = bpy.props.PointerProperty(
                type=ChainToolProperties
            )
            print("Chain Tool V4: Core properties registered")
        except Exception as e:
            print(f"Chain Tool V4: Could not register properties: {e}")
    
    # Register preferences
    if ChainToolPreferences:
        try:
            bpy.utils.register_class(ChainToolPreferences)
            print("Chain Tool V4: Preferences registered")
        except:
            pass
    
    # Initialize state manager
    if StateManager:
        state = StateManager()
        state.initialize()
        print("Chain Tool V4: State manager initialized")

def unregister():
    """Unregister core module"""
    import bpy
    
    # Unregister properties
    if hasattr(bpy.types.Scene, 'chain_tool_properties'):
        del bpy.types.Scene.chain_tool_properties
    
    if ChainToolProperties:
        try:
            bpy.utils.unregister_class(ChainToolProperties)
        except:
            pass
    
    if ChainToolPreferences:
        try:
            bpy.utils.unregister_class(ChainToolPreferences)
        except:
            pass
    
    print("Chain Tool V4: Core module unregistered")
