# =========================================
# 5. operators/system_ops.py - SYSTEM-MANAGEMENT
# =========================================

"""
System Operations - Debug, Performance, Cache Management, Statistics
KEINE Funktionslogik - nur System-interne Verwaltung
"""

import bpy
from bpy.types import Operator

class KETTE_OT_sphere_statistics(Operator):
    """Zeigt detaillierte Sphere-Statistiken"""
    bl_idname = "kette.sphere_statistics"
    bl_label = "Sphere Statistiken"
    bl_description = "Zeigt detaillierte Sphere-Statistiken"

    def execute(self, context):
        from ..core import constants
        from ..creation.spheres import get_sphere_statistics, cleanup_sphere_registry, validate_sphere_integrity
        
        try:
            # Cleanup erst
            cleanup_sphere_registry()
            
            # Stats holen
            stats = get_sphere_statistics()
            validation = validate_sphere_integrity()
            
            # Report für UI
            msg_parts = []
            for sphere_type, count in stats.items():
                if sphere_type != 'total' and count > 0:
                    msg_parts.append(f"{sphere_type}: {count}")
            
            if msg_parts:
                msg = f"Spheres - {', '.join(msg_parts)} (Total: {stats['total']})"
            else:
                msg = "Keine Spheres vorhanden"
            
            self.report({'INFO'}, msg)
            
            # Detaillierte Console-Ausgabe
            print("\n" + "="*50)
            print("SPHERE STATISTICS")
            print("="*50)
            
            print("\nBy Type:")
            for sphere_type, count in stats.items():
                print(f"  {sphere_type:12}: {count}")
            
            print(f"\nRegistry: {validation['registry_count']} entries")
            
            if validation['inconsistencies'] > 0:
                print(f"\n⚠️  Inconsistencies: {validation['inconsistencies']}")
                if validation['orphaned_objects']:
                    print(f"  Orphaned: {len(validation['orphaned_objects'])}")
                if validation['missing_objects']:
                    print(f"  Missing: {len(validation['missing_objects'])}")
            else:
                print("\n✅ System integrity OK")
            
            print("="*50 + "\n")
            
            if constants.debug:
                constants.debug.info('SYSTEM', f"Sphere statistics: {stats}")
                
        except Exception as e:
            self.report({'ERROR'}, f"Fehler bei Statistiken: {str(e)}")
            if constants.debug:
                constants.debug.error('SYSTEM', f"Error getting statistics: {e}")
            return {'CANCELLED'}
        
        return {'FINISHED'}

class KETTE_OT_system_status(Operator):
    """Zeigt detaillierten System-Status"""
    bl_idname = "kette.system_status"
    bl_label = "System Status"
    bl_description = "Zeigt detaillierten System-Status"

    def execute(self, context):
        from ..core import constants
        from ..core.cache_manager import validate_system_health
        
        try:
            health = validate_system_health()
            
            # Console Report
            print("\n" + "="*60)
            print("KETTENTOOL SYSTEM STATUS")
            print("="*60)
            
            print(f"\n🎯 OVERALL STATUS: {health['overall_status'].upper()}")
            
            # Cache Health
            cache_health = health['cache_health']
            print(f"\n💾 CACHE HEALTH:")
            print(f"  Errors: {len(cache_health['validation_errors'])}")
            print(f"  Warnings: {len(cache_health['warnings'])}")
            
            # Cache Statistics
            cache_stats = health['cache_statistics']['cache_counts']
            print(f"\n📊 CACHE STATISTICS:")
            for cache_type, count in cache_stats.items():
                print(f"  {cache_type:20}: {count}")
            
            # Collections
            print(f"\n📁 COLLECTIONS:")
            for coll_name, coll_data in health['collections'].items():
                status = "✅" if coll_data['exists'] else "❌"
                print(f"  {status} {coll_name:25}: {coll_data['object_count']} objects")
            
            print("="*60 + "\n")
            
            self.report({'INFO'}, f"System Status: {health['overall_status']}")
            
        except Exception as e:
            self.report({'ERROR'}, f"Fehler bei System Status: {str(e)}")
            if constants.debug:
                constants.debug.error('SYSTEM', f"Error getting system status: {e}")
            return {'CANCELLED'}
        
        return {'FINISHED'}

class KETTE_OT_toggle_debug(Operator):
    """Aktiviert/Deaktiviert Debug-Modus"""
    bl_idname = "kette.toggle_debug"
    bl_label = "Debug umschalten"
    bl_description = "Aktiviert/Deaktiviert Debug-Ausgabe"

    def execute(self, context):
        from ..core import constants
        
        if constants.debug:
            constants.debug.enabled = not constants.debug.enabled
            
            if constants.debug.enabled:
                constants.debug.level = 'DEBUG'
                self.report({'INFO'}, "Debug-Modus aktiviert")
            else:
                self.report({'INFO'}, "Debug-Modus deaktiviert")
        else:
            self.report({'WARNING'}, "Debug-System nicht verfügbar")
        
        return {'FINISHED'}

class KETTE_OT_performance_report(Operator):
    """Zeigt Performance-Statistiken"""
    bl_idname = "kette.performance_report"
    bl_label = "Performance Report"
    bl_description = "Zeigt Performance-Statistiken in der Konsole"

    def execute(self, context):
        from ..core import constants
        
        if constants.performance_monitor:
            report = constants.performance_monitor.report()
            
            # Print to console
            print("\n" + "="*60)
            print(report)
            print("="*60 + "\n")
            
            self.report({'INFO'}, "Performance Report in Konsole ausgegeben")
        else:
            self.report({'WARNING'}, "Performance Monitor nicht verfügbar")
        
        return {'FINISHED'}

class KETTE_OT_clear_caches(Operator):
    """Leert alle internen Caches"""
    bl_idname = "kette.clear_caches"
    bl_label = "Caches leeren"
    bl_description = "Leert alle internen Caches"

    def execute(self, context):
        from ..core import constants
        from ..core.cache_manager import clear_all_caches
        
        try:
            cache_counts = clear_all_caches()
            
            # Report
            total_cleared = sum(cache_counts.values())
            self.report({'INFO'}, f"Caches geleert: {total_cleared} Einträge")
            
            if constants.debug:
                constants.debug.info('SYSTEM', f"Cleared caches: {cache_counts}")
                
        except Exception as e:
            self.report({'ERROR'}, f"Fehler beim Cache-Leeren: {str(e)}")
            if constants.debug:
                constants.debug.error('SYSTEM', f"Error clearing caches: {e}")
            return {'CANCELLED'}
        
        return {'FINISHED'}

class KETTE_OT_cache_maintenance(Operator):
    """Führt vollständige Cache-Wartung durch"""
    bl_idname = "kette.cache_maintenance"
    bl_label = "Cache Wartung"
    bl_description = "Führt vollständige Cache-Wartung durch"

    def execute(self, context):
        from ..core import constants
        from ..core.cache_manager import perform_cache_maintenance
        
        try:
            maintenance_report = perform_cache_maintenance()
            
            cleaned_total = sum(maintenance_report['cleanup_performed'].values())
            recommendations = len(maintenance_report['recommendations'])
            
            msg = f"Cache-Wartung: {cleaned_total} Einträge bereinigt"
            if recommendations > 0:
                msg += f", {recommendations} Empfehlungen"
            
            self.report({'INFO'}, msg)
            
            # Detailed console output
            print("\n" + "="*50)
            print("CACHE MAINTENANCE REPORT")
            print("="*50)
            
            print("\nCleanup performed:")
            for cache_type, count in maintenance_report['cleanup_performed'].items():
                if count > 0:
                    print(f"  {cache_type:20}: {count} cleaned")
            
            if maintenance_report['recommendations']:
                print("\nRecommendations:")
                for rec in maintenance_report['recommendations']:
                    print(f"  • {rec}")
            
            print("="*50 + "\n")
            
            if constants.debug:
                constants.debug.info('SYSTEM', f"Cache maintenance completed")
                
        except Exception as e:
            self.report({'ERROR'}, f"Fehler bei Cache-Wartung: {str(e)}")
            if constants.debug:
                constants.debug.error('SYSTEM', f"Error in cache maintenance: {e}")
            return {'CANCELLED'}
        
        return {'FINISHED'}

# Registration
classes = [
    KETTE_OT_sphere_statistics,
    KETTE_OT_system_status,
    KETTE_OT_toggle_debug,
    KETTE_OT_performance_report,
    KETTE_OT_clear_caches,
    KETTE_OT_cache_maintenance,
]
