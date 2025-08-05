"""
Chain Tool V4 - Advanced Debug Initialization
Behält deine modulare Struktur bei und debuggt sie Schritt für Schritt
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
import os
import traceback
from pathlib import Path

# Add addon path to sys.path
addon_path = Path(__file__).parent
if str(addon_path) not in sys.path:
    sys.path.append(str(addon_path))

# Global storage for successfully loaded components
_loaded_modules = {}
_registered_classes = []

def test_import(module_name, required_items=None):
    """Testet ob ein Modul importiert werden kann"""
    try:
        # Versuche das Modul zu importieren
        if '.' in module_name:
            # Sub-module import
            parts = module_name.split('.')
            exec(f"from {'.'.join(parts[:-1])} import {parts[-1]}")
            module = eval(parts[-1])
        else:
            # Top-level module import
            exec(f"from . import {module_name}")
            module = eval(module_name)
        
        _loaded_modules[module_name] = module
        
        # Prüfe ob required_items vorhanden sind
        if required_items:
            missing = []
            for item in required_items:
                if not hasattr(module, item):
                    missing.append(item)
            if missing:
                print(f"  ⚠ {module_name}: Fehlende Items: {missing}")
                return False
        
        print(f"  ✓ {module_name} erfolgreich geladen")
        return True
        
    except ImportError as e:
        print(f"  ✗ {module_name}: ImportError - {e}")
        return False
    except Exception as e:
        print(f"  ✗ {module_name}: Unerwarteter Fehler - {e}")
        traceback.print_exc()
        return False

def create_minimal_panels():
    """Erstellt minimale Panels falls Module fehlen"""
    
    class CHAINTOOL_PT_main_panel(bpy.types.Panel):
        """Minimal Main Panel"""
        bl_label = "Chain Tool V4"
        bl_idname = "CHAINTOOL_PT_main_panel"
        bl_space_type = 'VIEW_3D'
        bl_region_type = 'UI'
        bl_category = "Chain Tool"
        
        def draw(self, context):
            layout = self.layout
            
            # Status Box
            box = layout.box()
            box.label(text="Module Status:", icon='INFO')
            
            for module_name, module in _loaded_modules.items():
                row = box.row()
                row.label(text=f"✓ {module_name}", icon='CHECKMARK')
            
            # Fehlende Module
            expected_modules = ['core', 'utils', 'geometry', 'algorithms', 
                              'patterns', 'operators', 'ui', 'presets']
            missing = [m for m in expected_modules if m not in _loaded_modules]
            
            if missing:
                box.separator()
                for module_name in missing:
                    row = box.row()
                    row.label(text=f"✗ {module_name}", icon='ERROR')
            
            # Properties Test
            box = layout.box()
            box.label(text="Properties:", icon='PROPERTIES')
            
            if hasattr(context.scene, 'chain_tool_properties'):
                props = context.scene.chain_tool_properties
                box.prop(props, 'pattern_type')
                box.prop(props, 'pattern_density')
                box.prop(props, 'sphere_size')
            else:
                box.label(text="Properties nicht gefunden!", icon='ERROR')
            
            # Debug Actions
            box = layout.box()
            box.label(text="Debug:", icon='CONSOLE')
            box.operator("chaintool.reload_modules", text="Module Neu Laden")
            box.operator("chaintool.test_operator", text="Test Operator")
    
    return [CHAINTOOL_PT_main_panel]

def create_test_operator():
    """Erstellt einen Test-Operator"""
    
    class CHAINTOOL_OT_test_operator(bpy.types.Operator):
        bl_idname = "chaintool.test_operator"
        bl_label = "Test Operator"
        bl_options = {'REGISTER'}
        
        def execute(self, context):
            self.report({'INFO'}, "Test Operator funktioniert!")
            print("\n" + "="*50)
            print("TEST OPERATOR AUSGEFÜHRT")
            print("Geladene Module:", list(_loaded_modules.keys()))
            print("Registrierte Klassen:", len(_registered_classes))
            print("="*50 + "\n")
            return {'FINISHED'}
    
    class CHAINTOOL_OT_reload_modules(bpy.types.Operator):
        bl_idname = "chaintool.reload_modules"
        bl_label = "Reload Modules"
        bl_options = {'REGISTER'}
        
        def execute(self, context):
            # Reload all loaded modules
            import importlib
            reloaded = 0
            for module_name, module in _loaded_modules.items():
                try:
                    importlib.reload(module)
                    reloaded += 1
                except:
                    pass
            
            self.report({'INFO'}, f"{reloaded} Module neu geladen")
            return {'FINISHED'}
    
    return [CHAINTOOL_OT_test_operator, CHAINTOOL_OT_reload_modules]

def create_minimal_properties():
    """Erstellt minimale Properties falls core.properties fehlt"""
    
    class ChainToolProperties(bpy.types.PropertyGroup):
        pattern_type: bpy.props.EnumProperty(
            name="Pattern Type",
            items=[
                ('SURFACE', "Surface", "Surface pattern"),
                ('EDGE', "Edge", "Edge pattern"),
                ('HYBRID', "Hybrid", "Hybrid pattern"),
                ('PAINT', "Paint", "Paint pattern"),
            ],
            default='SURFACE'
        )
        
        pattern_density: bpy.props.FloatProperty(
            name="Density",
            default=5.0,
            min=0.1,
            max=20.0
        )
        
        sphere_size: bpy.props.FloatProperty(
            name="Sphere Size",
            default=2.0,
            min=0.1,
            max=10.0
        )
    
    return ChainToolProperties

def register():
    """Intelligente Registrierung mit Fallbacks"""
    global _registered_classes
    
    print("\n" + "="*60)
    print("CHAIN TOOL V4 - MODULARE REGISTRIERUNG")
    print("="*60)
    
    # 1. Versuche Module zu laden
    print("\n1. Lade Module:")
    
    # Core Module (wichtigste zuerst)
    core_loaded = test_import('core')
    
    # Wenn core fehlt, versuche sub-modules direkt
    if not core_loaded:
        print("  Versuche core sub-modules direkt zu laden...")
        test_import('core.properties', ['ChainToolProperties'])
        test_import('core.state_manager', ['StateManager'])
        test_import('core.constants')
    
    # Utils
    utils_loaded = test_import('utils')
    if not utils_loaded:
        test_import('utils.debug')
        test_import('utils.performance')
        test_import('utils.caching')
    
    # Geometry
    test_import('geometry')
    
    # Algorithms  
    test_import('algorithms')
    
    # Patterns
    test_import('patterns')
    
    # Operators
    test_import('operators')
    
    # UI - Besonders wichtig!
    ui_loaded = test_import('ui')
    if not ui_loaded:
        print("  Versuche ui.panels direkt zu laden...")
        test_import('ui.panels', ['classes'])
    
    # Presets
    test_import('presets')
    
    # 2. Registriere Properties
    print("\n2. Registriere Properties:")
    properties_registered = False
    
    # Versuche Properties aus core.properties
    if 'core.properties' in _loaded_modules or 'core' in _loaded_modules:
        try:
            if 'core' in _loaded_modules and hasattr(_loaded_modules['core'], 'properties'):
                props_module = _loaded_modules['core'].properties
            else:
                props_module = _loaded_modules.get('core.properties')
            
            if props_module and hasattr(props_module, 'ChainToolProperties'):
                bpy.utils.register_class(props_module.ChainToolProperties)
                bpy.types.Scene.chain_tool_properties = bpy.props.PointerProperty(
                    type=props_module.ChainToolProperties
                )
                _registered_classes.append(props_module.ChainToolProperties)
                properties_registered = True
                print("  ✓ Properties aus core.properties registriert")
        except Exception as e:
            print(f"  ✗ Fehler bei core.properties: {e}")
    
    # Fallback: Minimale Properties
    if not properties_registered:
        print("  ⚠ Verwende minimale Fallback-Properties")
        MinimalProps = create_minimal_properties()
        bpy.utils.register_class(MinimalProps)
        bpy.types.Scene.chain_tool_properties = bpy.props.PointerProperty(type=MinimalProps)
        _registered_classes.append(MinimalProps)
    
    # 3. Registriere UI Panels
    print("\n3. Registriere UI Panels:")
    panels_registered = 0
    
    # Versuche Panels aus ui.panels
    if 'ui.panels' in _loaded_modules or 'ui' in _loaded_modules:
        try:
            if 'ui' in _loaded_modules and hasattr(_loaded_modules['ui'], 'panels'):
                panels_module = _loaded_modules['ui'].panels
            else:
                panels_module = _loaded_modules.get('ui.panels')
            
            if panels_module and hasattr(panels_module, 'classes'):
                for cls in panels_module.classes:
                    try:
                        bpy.utils.register_class(cls)
                        _registered_classes.append(cls)
                        panels_registered += 1
                        print(f"  ✓ {cls.bl_idname}")
                    except Exception as e:
                        print(f"  ✗ {cls.bl_idname}: {e}")
        except Exception as e:
            print(f"  ✗ Fehler beim Laden der Panels: {e}")
    
    # Fallback: Minimale Panels
    if panels_registered == 0:
        print("  ⚠ Verwende minimale Fallback-Panels")
        for cls in create_minimal_panels():
            bpy.utils.register_class(cls)
            _registered_classes.append(cls)
            panels_registered += 1
    
    # 4. Registriere Test-Operators
    print("\n4. Registriere Test-Operators:")
    for cls in create_test_operator():
        bpy.utils.register_class(cls)
        _registered_classes.append(cls)
        print(f"  ✓ {cls.bl_idname}")
    
    # 5. Zusammenfassung
    print("\n" + "="*60)
    print("REGISTRIERUNG ABGESCHLOSSEN")
    print(f"✓ Module geladen: {len(_loaded_modules)}")
    print(f"✓ Klassen registriert: {len(_registered_classes)}")
    print(f"✓ Panels registriert: {panels_registered}")
    
    if len(_loaded_modules) < 3:
        print("\n⚠ WARNUNG: Wenige Module geladen!")
        print("Prüfe ob alle __init__.py Dateien in den Modulordnern existieren")
    
    print("="*60 + "\n")

def unregister():
    """Deregistrierung"""
    global _registered_classes
    
    print("\nChain Tool V4: Deregistrierung...")
    
    # Deregistriere alle Klassen
    for cls in reversed(_registered_classes):
        try:
            bpy.utils.unregister_class(cls)
        except:
            pass
    
    # Entferne Properties
    if hasattr(bpy.types.Scene, 'chain_tool_properties'):
        del bpy.types.Scene.chain_tool_properties
    
    # Cleanup
    _registered_classes.clear()
    _loaded_modules.clear()
    
    # Entferne aus sys.path
    if str(addon_path) in sys.path:
        sys.path.remove(str(addon_path))
    
    print("Chain Tool V4: Deregistrierung abgeschlossen\n")

if __name__ == "__main__":
    register()
