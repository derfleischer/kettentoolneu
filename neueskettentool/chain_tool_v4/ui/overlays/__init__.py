"""
Chain Tool V4 - Overlay System Registration
==========================================

Manages all GPU-accelerated overlay systems:
- Paint overlay (primary GPU shader system - already implemented)
- Pattern preview overlay (simplified version)
- Connection preview overlay (basic visualization)

The paint_overlay.py is already fully implemented as the most critical component.
"""

import bpy
from typing import Dict, Any, List

# Import existing paint overlay system (already fully implemented)
from .paint_overlay import (
    get_paint_overlay_system,
    cleanup_paint_overlay_system
)

# Import simplified overlay systems
from .pattern_preview import PatternPreviewSystem
from .connection_preview import ConnectionPreviewSystem

# Global overlay system registry
_overlay_systems: Dict[str, Any] = {}
_active_overlays: List[str] = []


def register():
    """
    Register all overlay systems.
    
    Note: paint_overlay.py is already fully implemented and registered
    via get_paint_overlay_system(). This function coordinates additional overlays.
    """
    
    global _overlay_systems
    
    try:
        # Paint overlay system is already implemented - just get reference
        paint_system = get_paint_overlay_system()
        _overlay_systems['paint'] = paint_system
        print("Chain Tool V4 Overlays: Paint system active (GPU shader)")
        
        # Initialize pattern preview system
        pattern_preview = PatternPreviewSystem()
        _overlay_systems['pattern_preview'] = pattern_preview
        print("Chain Tool V4 Overlays: Pattern preview system initialized")
        
        # Initialize connection preview system
        connection_preview = ConnectionPreviewSystem()
        _overlay_systems['connection_preview'] = connection_preview
        print("Chain Tool V4 Overlays: Connection preview system initialized")
        
        print(f"Chain Tool V4 Overlays: {len(_overlay_systems)} systems registered")
        
    except Exception as e:
        print(f"Chain Tool V4 Overlays: Registration failed - {e}")
        import traceback
        traceback.print_exc()


def unregister():
    """
    Unregister and cleanup all overlay systems.
    """
    
    global _overlay_systems, _active_overlays
    
    try:
        # Cleanup connection preview
        if 'connection_preview' in _overlay_systems:
            _overlay_systems['connection_preview'].cleanup()
            print("Chain Tool V4 Overlays: Connection preview cleaned up")
        
        # Cleanup pattern preview
        if 'pattern_preview' in _overlay_systems:
            _overlay_systems['pattern_preview'].cleanup()
            print("Chain Tool V4 Overlays: Pattern preview cleaned up")
        
        # Cleanup paint overlay system (fully implemented system)
        cleanup_paint_overlay_system()
        print("Chain Tool V4 Overlays: Paint system cleaned up")
        
        _overlay_systems.clear()
        _active_overlays.clear()
        
    except Exception as e:
        print(f"Chain Tool V4 Overlays: Cleanup error - {e}")


def get_overlay_system(name: str):
    """
    Get specific overlay system by name.
    
    Args:
        name: 'paint', 'pattern_preview', or 'connection_preview'
    """
    return _overlay_systems.get(name)


def get_overlay_systems():
    """
    Get all registered overlay systems.
    Returns dict of {name: system} pairs.
    """
    return _overlay_systems


def activate_overlay(name: str, **kwargs):
    """
    Activate specific overlay system.
    
    Args:
        name: Overlay system name
        **kwargs: System-specific parameters
    """
    
    if name not in _overlay_systems:
        print(f"Chain Tool V4 Overlays: Unknown system '{name}'")
        return False
    
    try:
        system = _overlay_systems[name]
        
        if name == 'paint':
            # Paint system uses specific activation method
            system.activate(**kwargs)
        else:
            # Generic activation for other systems
            system.set_active(True)
            system.update_parameters(**kwargs)
        
        if name not in _active_overlays:
            _active_overlays.append(name)
        
        print(f"Chain Tool V4 Overlays: '{name}' activated")
        return True
        
    except Exception as e:
        print(f"Chain Tool V4 Overlays: Failed to activate '{name}' - {e}")
        return False


def deactivate_overlay(name: str):
    """
    Deactivate specific overlay system.
    """
    
    if name not in _overlay_systems:
        return False
    
    try:
        system = _overlay_systems[name]
        
        if name == 'paint':
            system.deactivate()
        else:
            system.set_active(False)
        
        if name in _active_overlays:
            _active_overlays.remove(name)
        
        print(f"Chain Tool V4 Overlays: '{name}' deactivated")
        return True
        
    except Exception as e:
        print(f"Chain Tool V4 Overlays: Failed to deactivate '{name}' - {e}")
        return False


def deactivate_all_overlays():
    """
    Deactivate all overlay systems.
    """
    
    for name in list(_active_overlays):
        deactivate_overlay(name)


def get_active_overlays():
    """
    Get list of currently active overlay systems.
    """
    return _active_overlays.copy()


def force_overlay_update():
    """
    Force update of all active overlay systems.
    """
    
    for name in _active_overlays:
        try:
            system = _overlay_systems[name]
            if hasattr(system, 'force_update'):
                system.force_update()
        except Exception as e:
            print(f"Chain Tool V4 Overlays: Update failed for '{name}' - {e}")


def get_overlay_status():
    """
    Get status information about all overlay systems.
    For debugging and UI display.
    """
    
    status = {
        'total_systems': len(_overlay_systems),
        'active_systems': len(_active_overlays),
        'systems': {}
    }
    
    for name, system in _overlay_systems.items():
        system_status = {
            'active': name in _active_overlays,
            'type': type(system).__name__,
            'gpu_accelerated': name == 'paint'
        }
        
        # Get system-specific status
        if hasattr(system, 'get_status'):
            system_status.update(system.get_status())
        elif hasattr(system, 'is_active'):
            system_status['ready'] = system.is_active
        
        status['systems'][name] = system_status
    
    return status


# Development utilities
def debug_overlay_systems():
    """Print debug information about overlay systems."""
    print("Chain Tool V4 Overlays - System Status:")
    
    for name, system in _overlay_systems.items():
        active = "🟢" if name in _active_overlays else "⚪"
        gpu = "⚡" if name == 'paint' else "🔧"
        print(f"  {active} {gpu} {name}: {type(system).__name__}")
        
        if hasattr(system, 'get_debug_info'):
            debug_info = system.get_debug_info()
            for key, value in debug_info.items():
                print(f"      {key}: {value}")
