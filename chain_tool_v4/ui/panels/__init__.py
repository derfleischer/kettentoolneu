"""
Chain Tool V4 - UI Panels Module
Zentrale Registrierung aller UI-Panels

Registriert:
- main_panel.py - Haupt-Interface
- paint_panel.py - Paint-Mode Controls  
- pattern_panel.py - Pattern-Konfiguration
- edge_panel.py - Edge-Tools
- export_panel.py - Export-Controls
"""

import bpy
from bpy.utils import register_class, unregister_class

# Import aller Panel-Module
from .main_panel import (
    CHAINTOOL_PT_main_panel,
    CHAINTOOL_PT_quick_actions,
)

from .paint_panel import (
    CHAINTOOL_PT_paint_panel,
    CHAINTOOL_PT_paint_tools,
)

from .pattern_panel import (
    CHAINTOOL_PT_pattern_panel,
    CHAINTOOL_PT_pattern_presets,
    CHAINTOOL_PT_pattern_analysis,
)

from .edge_panel import (
    CHAINTOOL_PT_edge_panel,
    CHAINTOOL_PT_edge_presets,
    CHAINTOOL_PT_edge_performance,
)

from .export_panel import (
    CHAINTOOL_PT_export_panel,
    CHAINTOOL_PT_export_quality,
    CHAINTOOL_PT_export_advanced,
)


# Zentrale Panel-Klassen Liste
# Reihenfolge bestimmt Panel-Hierarchie in Blender UI
classes = [
    # Main Panel (Basis-Interface)
    CHAINTOOL_PT_main_panel,
    CHAINTOOL_PT_quick_actions,
    
    # Paint Panel (nur bei Pattern-Type PAINT)
    CHAINTOOL_PT_paint_panel,
    CHAINTOOL_PT_paint_tools,
    
    # Pattern Panel (Pattern-Konfiguration)
    CHAINTOOL_PT_pattern_panel,
    CHAINTOOL_PT_pattern_presets,
    CHAINTOOL_PT_pattern_analysis,
    
    # Edge Panel (Edge-Tools)
    CHAINTOOL_PT_edge_panel,
    CHAINTOOL_PT_edge_presets,
    CHAINTOOL_PT_edge_performance,
    
    # Export Panel (3D-Druck Export)
    CHAINTOOL_PT_export_panel,
    CHAINTOOL_PT_export_quality,
    CHAINTOOL_PT_export_advanced,
]


def register():
    """Registriert alle UI-Panels"""
    print("Chain Tool V4: Registriere UI-Panels...")
    
    for cls in classes:
        try:
            register_class(cls)
            print(f"  ✓ Panel registriert: {cls.bl_idname}")
        except Exception as e:
            print(f"  ❌ Fehler bei Panel {cls.bl_idname}: {e}")
    
    print(f"Chain Tool V4: {len(classes)} UI-Panels registriert")


def unregister():
    """Deregistriert alle UI-Panels"""
    print("Chain Tool V4: Deregistriere UI-Panels...")
    
    # Rückwärts deregistrieren (umgekehrte Reihenfolge)
    for cls in reversed(classes):
        try:
            unregister_class(cls)
            print(f"  ✓ Panel deregistriert: {cls.bl_idname}")
        except Exception as e:
            print(f"  ❌ Fehler bei Panel {cls.bl_idname}: {e}")
    
    print("Chain Tool V4: UI-Panels deregistriert")


# Module-Informationen für Debug-Zwecke
def get_panel_info():
    """Gibt Informationen über alle registrierten Panels zurück"""
    panel_info = {
        'total_panels': len(classes),
        'panels': []
    }
    
    for cls in classes:
        info = {
            'name': cls.bl_label,
            'idname': cls.bl_idname,
            'category': getattr(cls, 'bl_category', 'Unknown'),
            'parent': getattr(cls, 'bl_parent_id', None),
            'poll_function': hasattr(cls, 'poll'),
            'options': getattr(cls, 'bl_options', set()),
        }
        panel_info['panels'].append(info)
    
    return panel_info


def validate_panel_hierarchy():
    """Validiert die Panel-Hierarchie auf Konsistenz"""
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    panel_ids = {cls.bl_idname for cls in classes}
    
    for cls in classes:
        # Prüfe Parent-Child Beziehungen
        if hasattr(cls, 'bl_parent_id') and cls.bl_parent_id:
            if cls.bl_parent_id not in panel_ids:
                validation_results['errors'].append(
                    f"Panel {cls.bl_idname} hat unbekannten Parent: {cls.bl_parent_id}"
                )
                validation_results['is_valid'] = False
        
        # Prüfe auf doppelte IDs
        id_count = sum(1 for c in classes if c.bl_idname == cls.bl_idname)
        if id_count > 1:
            validation_results['errors'].append(
                f"Doppelte Panel-ID gefunden: {cls.bl_idname}"
            )
            validation_results['is_valid'] = False
        
        # Warnungen für fehlende Poll-Funktionen bei Child-Panels
        if hasattr(cls, 'bl_parent_id') and cls.bl_parent_id and not hasattr(cls, 'poll'):
            validation_results['warnings'].append(
                f"Child-Panel {cls.bl_idname} hat keine poll() Funktion"
            )
    
    return validation_results


# Debug-Operator für Panel-Informationen
class CHAINTOOL_OT_debug_panels(bpy.types.Operator):
    """Debug-Operator für Panel-Informationen"""
    bl_idname = "chaintool.debug_panels"
    bl_label = "Debug Panel Info"
    bl_description = "Zeigt Debug-Informationen über Chain Tool Panels"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        panel_info = get_panel_info()
        validation = validate_panel_hierarchy()
        
        print("\n" + "="*50)
        print("CHAIN TOOL V4 - PANEL DEBUG INFO")
        print("="*50)
        
        print(f"\nRegistrierte Panels: {panel_info['total_panels']}")
        print("-" * 30)
        
        for panel in panel_info['panels']:
            print(f"• {panel['name']} ({panel['idname']})")
            print(f"  Kategorie: {panel['category']}")
            if panel['parent']:
                print(f"  Parent: {panel['parent']}")
            if panel['poll_function']:
                print(f"  Hat Poll-Funktion: ✓")
            if panel['options']:
                print(f"  Optionen: {', '.join(panel['options'])}")
            print()
        
        print("VALIDIERUNG:")
        print("-" * 20)
        if validation['is_valid']:
            print("✓ Panel-Hierarchie ist valid")
        else:
            print("❌ Panel-Hierarchie hat Fehler:")
            for error in validation['errors']:
                print(f"  • {error}")
        
        if validation['warnings']:
            print("\n⚠ Warnungen:")
            for warning in validation['warnings']:
                print(f"  • {warning}")
        
        print("="*50 + "\n")
        
        self.report({'INFO'}, f"Panel-Debug Info ausgegeben ({panel_info['total_panels']} Panels)")
        return {'FINISHED'}


# Zusätzliche Debug-Klassen registrieren
debug_classes = [
    CHAINTOOL_OT_debug_panels,
]


def register_debug():
    """Registriert Debug-Operatoren (nur in Debug-Mode)"""
    for cls in debug_classes:
        try:
            register_class(cls)
        except:
            pass  # Ignoriere Fehler bei Debug-Operatoren


def unregister_debug():
    """Deregistriert Debug-Operatoren"""
    for cls in reversed(debug_classes):
        try:
            unregister_class(cls)
        except:
            pass


# Vollständige Registrierung mit Debug-Support
def register_all():
    """Registriert Panels und Debug-Tools"""
    register()
    register_debug()


def unregister_all():
    """Deregistriert Panels und Debug-Tools"""
    unregister_debug()
    unregister()


# Module-Export für übergeordnete __init__.py
__all__ = [
    'classes',
    'register',
    'unregister', 
    'register_all',
    'unregister_all',
    'get_panel_info',
    'validate_panel_hierarchy'
]
