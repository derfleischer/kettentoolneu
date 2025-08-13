"""
Kettentool Operators Module
Central registration for all operators
"""

import bpy

# Import all operator modules
from . import sphere_ops
from . import connector_ops
from . import surface_ops
from . import system_ops
from . import utility_ops
from . import pattern_connector
from . import paint_operator

# Collect all operator classes
classes = []

# Get classes from each module
def get_classes_from_module(module):
    """Extract operator classes from a module"""
    module_classes = []
    
    # Check if module has a classes list
    if hasattr(module, 'classes'):
        module_classes.extend(module.classes)
    else:
        # Try to find classes manually
        import inspect
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, bpy.types.Operator):
                # Check if it's from this module (not imported)
                if obj.__module__ == module.__name__:
                    module_classes.append(obj)
    
    return module_classes

# Collect classes from all modules
modules = [
    sphere_ops,
    connector_ops,
    surface_ops,
    system_ops,
    utility_ops,
    pattern_connector,
    paint_operator
]

for module in modules:
    module_classes = get_classes_from_module(module)
    classes.extend(module_classes)
    
    # Print info for debugging
    if module_classes:
        print(f"[OPERATORS] Found {len(module_classes)} operators in {module.__name__.split('.')[-1]}")

def register():
    """Register all operator classes"""
    print("[OPERATORS] Starting operator registration...")
    
    registered_count = 0
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
            registered_count += 1
            print(f"[OPERATORS] ✅ Registered: {cls.__name__}")
        except Exception as e:
            # Class might already be registered
            if "already registered" in str(e):
                print(f"[OPERATORS] ⚠️ Already registered: {cls.__name__}")
            else:
                print(f"[OPERATORS] ❌ Failed to register {cls.__name__}: {e}")
    
    print(f"[OPERATORS] Registered {registered_count} operators")
    
    # Verify key operators exist
    verify_operators()

def unregister():
    """Unregister all operator classes"""
    print("[OPERATORS] Unregistering operators...")
    
    unregistered_count = 0
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
            unregistered_count += 1
        except Exception as e:
            # Class might not be registered
            if "not registered" not in str(e):
                print(f"[OPERATORS] ⚠️ Failed to unregister {cls.__name__}: {e}")
    
    print(f"[OPERATORS] Unregistered {unregistered_count} operators")

def verify_operators():
    """Verify that key operators are accessible"""
    key_operators = [
        ('kette.create_sphere', 'Create Sphere'),
        ('kette.create_connector', 'Create Connector'),
        ('kette.toggle_paint_mode', 'Toggle Paint Mode'),
        ('kette.apply_pattern', 'Apply Pattern'),
        ('kette.clear_all', 'Clear All'),
        ('kette.auto_connect_selected', 'Auto Connect'),
    ]
    
    all_good = True
    for op_id, op_name in key_operators:
        try:
            # Try to access the operator
            op_module, op_func = op_id.split('.')
            if hasattr(bpy.ops, op_module):
                module = getattr(bpy.ops, op_module)
                if hasattr(module, op_func):
                    print(f"[OPERATORS] ✅ {op_name} ({op_id})")
                else:
                    print(f"[OPERATORS] ❌ {op_name} ({op_id}) - function not found")
                    all_good = False
            else:
                print(f"[OPERATORS] ❌ {op_name} ({op_id}) - module not found")
                all_good = False
        except Exception as e:
            print(f"[OPERATORS] ❌ {op_name} ({op_id}): {e}")
            all_good = False
    
    if all_good:
        print("[OPERATORS] ✅ All key operators verified")
    else:
        print("[OPERATORS] ⚠️ Some operators missing - check individual modules")

# List all registered operators (for debugging)
def list_operators():
    """List all registered operators"""
    print("\n" + "=" * 50)
    print("REGISTERED OPERATORS:")
    print("=" * 50)
    
    for i, cls in enumerate(classes, 1):
        bl_idname = getattr(cls, 'bl_idname', 'NO_ID')
        bl_label = getattr(cls, 'bl_label', 'NO_LABEL')
        print(f"{i:3}. {bl_idname:30} - {bl_label}")
    
    print("=" * 50)
