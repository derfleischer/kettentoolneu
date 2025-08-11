"""
Cache Management System - NEU
Zentrale Verwaltung aller Caches mit Cleanup und Validation
"""

import bpy
from typing import Dict, Any, List
from .constants import (
    _aura_cache, _bvh_cache, _existing_connectors, 
    _remembered_pairs, _sphere_registry, _sphere_collections
)

# =========================================
# CACHE CLEANUP
# =========================================

def clear_auras():
    """Löscht alle gecachten Auras"""
    from . import constants
    
    if constants.debug:
        constants.debug.info('CACHE', "Clearing all auras")
    
    auras_to_remove = []

    for cache_key, aura in list(_aura_cache.items()):
        try:
            if aura.name in bpy.data.objects:
                auras_to_remove.append(aura)
        except ReferenceError:
            pass

    for aura in auras_to_remove:
        try:
            bpy.data.objects.remove(aura, do_unlink=True)
        except:
            pass

    _aura_cache.clear()
    _bvh_cache.clear()

def clear_all_caches():
    """Leert alle Caches"""
    from . import constants
    
    # Count before clearing
    cache_counts = {
        'aura': len(_aura_cache),
        'bvh': len(_bvh_cache),
        'connectors': len(_existing_connectors),
        'pairs': len(_remembered_pairs),
        'sphere_registry': len(_sphere_registry)
    }
    
    # Clear auras first (removes objects)
    clear_auras()
    
    # Clear other caches
    _bvh_cache.clear()
    _existing_connectors.clear() 
    _remembered_pairs.clear()
    
    # Clear sphere collections but not registry (handled by spheres module)
    for collection_type in _sphere_collections:
        _sphere_collections[collection_type].clear()
    
    total_cleared = sum(cache_counts.values())
    
    if constants.debug:
        constants.debug.info('CACHE', f"Cleared all caches: {total_cleared} entries")
        constants.debug.trace('CACHE', f"Cache breakdown: {cache_counts}")
    
    return cache_counts

def cleanup_dead_references():
    """Bereinigt tote Referenzen in allen Caches"""
    from . import constants
    
    cleaned = {
        'aura': 0,
        'bvh': 0,
        'connectors': 0,
        'sphere_registry': 0,
        'sphere_collections': 0
    }
    
    # Aura Cache - prüfe ob Objekte noch existieren
    dead_auras = []
    for cache_key, aura in _aura_cache.items():
        try:
            if aura.name not in bpy.data.objects:
                dead_auras.append(cache_key)
        except ReferenceError:
            dead_auras.append(cache_key)
    
    for cache_key in dead_auras:
        _aura_cache.pop(cache_key, None)
        cleaned['aura'] += 1
    
    # BVH Cache - prüfe ob Objekte noch existieren
    dead_bvh = []
    for obj_name in _bvh_cache:
        if obj_name not in bpy.data.objects:
            dead_bvh.append(obj_name)
    
    for obj_name in dead_bvh:
        _bvh_cache.pop(obj_name, None)
        cleaned['bvh'] += 1
    
    # Connector Cache - prüfe ob Connector-Objekte noch existieren
    dead_connectors = []
    for pair_key, connector in _existing_connectors.items():
        try:
            if connector.name not in bpy.data.objects:
                dead_connectors.append(pair_key)
        except ReferenceError:
            dead_connectors.append(pair_key)
    
    for pair_key in dead_connectors:
        _existing_connectors.pop(pair_key, None)
        cleaned['connectors'] += 1
    
    # Sphere Registry - prüfe ob Sphere-Objekte noch existieren
    dead_spheres = []
    for sphere_name in _sphere_registry:
        if sphere_name not in bpy.data.objects:
            dead_spheres.append(sphere_name)
    
    for sphere_name in dead_spheres:
        _sphere_registry.pop(sphere_name, None)
        cleaned['sphere_registry'] += 1
    
    # Sphere Collections - prüfe alle Listen
    for collection_type, sphere_list in _sphere_collections.items():
        dead_sphere_names = []
        for sphere_name in sphere_list:
            if sphere_name not in bpy.data.objects:
                dead_sphere_names.append(sphere_name)
        
        for sphere_name in dead_sphere_names:
            sphere_list.remove(sphere_name)
            cleaned['sphere_collections'] += 1
    
    total_cleaned = sum(cleaned.values())
    
    if constants.debug and total_cleaned > 0:
        constants.debug.info('CACHE', f"Cleaned {total_cleaned} dead references")
        constants.debug.trace('CACHE', f"Cleanup breakdown: {cleaned}")
    
    return cleaned

# =========================================
# CACHE VALIDATION
# =========================================

def validate_cache_integrity() -> Dict[str, Any]:
    """
    Validiert Integrität aller Caches
    
    Returns:
        Validierungs-Report
    """
    from . import constants
    
    report = {
        'cache_sizes': {
            'aura_cache': len(_aura_cache),
            'bvh_cache': len(_bvh_cache),
            'connectors': len(_existing_connectors),
            'sphere_registry': len(_sphere_registry),
            'sphere_collections_total': sum(len(lst) for lst in _sphere_collections.values())
        },
        'validation_errors': [],
        'warnings': [],
        'recommendations': []
    }
    
    # Validiere Aura Cache
    for cache_key, aura in _aura_cache.items():
        try:
            if aura.name not in bpy.data.objects:
                report['validation_errors'].append(f"Dead aura reference: {cache_key}")
        except ReferenceError:
            report['validation_errors'].append(f"Invalid aura reference: {cache_key}")
    
    # Validiere BVH Cache
    for obj_name in _bvh_cache:
        if obj_name not in bpy.data.objects:
            report['validation_errors'].append(f"Dead BVH reference: {obj_name}")
    
    # Validiere Connector Cache
    for pair_key, connector in _existing_connectors.items():
        try:
            if connector.name not in bpy.data.objects:
                report['validation_errors'].append(f"Dead connector reference: {pair_key}")
        except ReferenceError:
            report['validation_errors'].append(f"Invalid connector reference: {pair_key}")
    
    # Validiere Sphere Registry vs Collections
    registry_spheres = set(_sphere_registry.keys())
    collection_spheres = set()
    for sphere_list in _sphere_collections.values():
        collection_spheres.update(sphere_list)
    
    # Spheres in Registry aber nicht in Collections
    orphaned_registry = registry_spheres - collection_spheres
    if orphaned_registry:
        report['warnings'].append(f"Spheres in registry but not in collections: {len(orphaned_registry)}")
    
    # Spheres in Collections aber nicht in Registry
    orphaned_collections = collection_spheres - registry_spheres
    if orphaned_collections:
        report['warnings'].append(f"Spheres in collections but not in registry: {len(orphaned_collections)}")
    
    # Performance-Warnungen
    if len(_aura_cache) > 10:
        report['warnings'].append(f"Large aura cache: {len(_aura_cache)} entries")
        report['recommendations'].append("Consider clearing aura cache")
    
    if len(_bvh_cache) > 20:
        report['warnings'].append(f"Large BVH cache: {len(_bvh_cache)} entries")
        report['recommendations'].append("Consider clearing BVH cache")
    
    # Erfolgs-Status
    report['is_valid'] = len(report['validation_errors']) == 0
    report['needs_cleanup'] = len(report['validation_errors']) > 0 or len(report['warnings']) > 0
    
    if constants.debug:
        constants.debug.trace('CACHE', f"Cache validation: {len(report['validation_errors'])} errors, {len(report['warnings'])} warnings")
    
    return report

# =========================================
# CACHE STATISTICS
# =========================================

def get_cache_statistics() -> Dict[str, Any]:
    """
    Gibt detaillierte Cache-Statistiken zurück
    
    Returns:
        Cache-Statistiken
    """
    stats = {
        'cache_counts': {
            'aura_cache': len(_aura_cache),
            'bvh_cache': len(_bvh_cache),
            'existing_connectors': len(_existing_connectors),
            'remembered_pairs': len(_remembered_pairs),
            'sphere_registry': len(_sphere_registry)
        },
        'sphere_collections': {},
        'memory_usage': {},
        'cache_efficiency': {}
    }
    
    # Sphere Collections im Detail
    for collection_type, sphere_list in _sphere_collections.items():
        stats['sphere_collections'][collection_type] = len(sphere_list)
    
    # Memory Usage (grobe Schätzung)
    stats['memory_usage'] = {
        'aura_objects': len(_aura_cache),  # Jede Aura = 1 Mesh-Objekt
        'bvh_trees': len(_bvh_cache),     # BVH Trees können groß sein
        'registry_entries': len(_sphere_registry) * 10,  # ~10 Properties pro Sphere
        'collection_refs': sum(len(lst) for lst in _sphere_collections.values())
    }
    
    # Cache Efficiency (Hit-Rate simulation)
    # In einer echten Implementierung würden wir Hit/Miss zählen
    total_cache_entries = sum(stats['cache_counts'].values())
    if total_cache_entries > 0:
        stats['cache_efficiency'] = {
            'total_entries': total_cache_entries,
            'estimated_hit_rate': min(0.9, total_cache_entries / 100),  # Simulation
            'cache_pressure': 'low' if total_cache_entries < 50 else 'high'
        }
    
    return stats

# =========================================
# CACHE MAINTENANCE
# =========================================

def perform_cache_maintenance() -> Dict[str, Any]:
    """
    Führt vollständige Cache-Wartung durch
    
    Returns:
        Wartungs-Report
    """
    from . import constants
    
    if constants.debug:
        constants.debug.info('CACHE', "Starting cache maintenance")
    
    # Cleanup tote Referenzen
    cleaned = cleanup_dead_references()
    
    # Validiere Integrität
    validation = validate_cache_integrity()
    
    # Statistiken
    stats = get_cache_statistics()
    
    # Wartungs-Empfehlungen
    recommendations = []
    
    if stats['cache_counts']['aura_cache'] > 10:
        recommendations.append("Clear aura cache - too many cached auras")
    
    if stats['cache_counts']['bvh_cache'] > 20:
        recommendations.append("Clear BVH cache - memory usage high")
    
    if len(validation['validation_errors']) > 0:
        recommendations.append("Fix cache integrity errors")
    
    # Report zusammenstellen
    maintenance_report = {
        'cleanup_performed': cleaned,
        'validation_results': validation,
        'current_statistics': stats,
        'recommendations': recommendations,
        'maintenance_needed': len(recommendations) > 0
    }
    
    if constants.debug:
        constants.debug.info('CACHE', f"Cache maintenance completed: {sum(cleaned.values())} items cleaned")
    
    return maintenance_report

# =========================================
# SYSTEM HEALTH
# =========================================

def validate_system_health() -> Dict[str, Any]:
    """Validiert gesamte System-Gesundheit"""
    from . import constants
    
    health = {
        'cache_health': validate_cache_integrity(),
        'cache_statistics': get_cache_statistics(),
        'collections': {},
        'performance': None
    }
    
    # Collection Health
    kettentool_collections = [
        'KonstruktionsKette',
        'KonstruktionsKette_Spheres', 
        'KonstruktionsKette_Connectors',
        'KonstruktionsKette_Temp'
    ]
    
    for coll_name in kettentool_collections:
        collection = bpy.data.collections.get(coll_name)
        if collection:
            health['collections'][coll_name] = {
                'exists': True,
                'object_count': len(collection.objects),
                'child_count': len(collection.children)
            }
        else:
            health['collections'][coll_name] = {
                'exists': False,
                'object_count': 0,
                'child_count': 0
            }
    
    # Performance Monitor
    if constants.performance_monitor:
        health['performance'] = constants.performance_monitor.report()
    
    # Overall Health Score
    errors = len(health['cache_health']['validation_errors'])
    warnings = len(health['cache_health']['warnings'])
    
    if errors == 0 and warnings == 0:
        health['overall_status'] = 'excellent'
    elif errors == 0 and warnings < 3:
        health['overall_status'] = 'good'
    elif errors < 3:
        health['overall_status'] = 'fair'
    else:
        health['overall_status'] = 'poor'
    
    return health

# =========================================
# REGISTRATION
# =========================================

def register():
    """Registriert Cache Manager"""
    pass

def unregister():
    """Deregistriert Cache Manager"""
    clear_all_caches()
