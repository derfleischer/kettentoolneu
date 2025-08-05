"""
Chain Tool V4 - UI Module
========================
User Interface Components
"""

# Import panels submodule
from . import panels

# Re-export classes for convenience
classes = []

# Get classes from panels
if hasattr(panels, 'classes'):
    classes = panels.classes
elif hasattr(panels, 'CLASSES'):
    classes = panels.CLASSES

# Export
__all__ = ['panels', 'classes']

def register():
    """Register UI components"""
    print("Chain Tool V4: Registering UI module...")
    
    # Register panels
    if hasattr(panels, 'register'):
        panels.register()
    
    print(f"Chain Tool V4: UI module registered with {len(classes)} panels")

def unregister():
    """Unregister UI components"""
    print("Chain Tool V4: Unregistering UI module...")
    
    # Unregister panels
    if hasattr(panels, 'unregister'):
        panels.unregister()
    
    print("Chain Tool V4: UI module unregistered")
