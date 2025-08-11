"""
Collection Management - ERWEITERT
Mit hierarchischer Collection-Struktur
"""

import bpy
from ..core.constants import COLLECTION_STRUCTURE

# =========================================
# COLLECTION UTILITIES
# =========================================

def ensure_collection_structure() -> dict:
    """
    Erstellt komplette Collection-Struktur
    
    Returns:
        Dictionary mit allen Collections
    """
    from ..core import constants
    
    collections = {}
    
    # Root Collection
    root_name = COLLECTION_STRUCTURE['root']
    root_col = bpy.data.collections.get(root_name)
    if not root_col:
        if constants.debug:
            constants.debug.info('COLLECTIONS', f"Creating root collection: {root_name}")
        root_col = bpy.data.collections.new(root_name)
        bpy.context.scene.collection.children.link(root_col)
    collections['root'] = root_col
    
    # Sub-Collections
    for key, name in COLLECTION_STRUCTURE.items():
        if key == 'root':
            continue
            
        col = bpy.data.collections.get(name)
        if not col:
            if constants.debug:
                constants.debug.trace('COLLECTIONS', f"Creating sub-collection: {name}")
            col = bpy.data.collections.new(name)
            root_col.children.link(col)
        collections[key] = col
    
    return collections

def ensure_collection(collection_type: str = "spheres") -> bpy.types.Collection:
    """
    Erstellt oder holt spezifische Collection
    
    Args:
        collection_type: 'root', 'spheres', 'connectors', 'temp'
        
    Returns:
        Collection-Objekt
    """
    from ..core import constants
    
    if constants.performance_monitor:
        constants.performance_monitor.start_timer("ensure_collection")
    
    collections = ensure_collection_structure()
    result = collections.get(collection_type, collections['root'])
    
    if constants.performance_monitor:
        constants.performance_monitor.end_timer("ensure_collection")
    
    return result

def organize_object_in_collection(obj: bpy.types.Object, collection_type: str):
    """
    Organisiert Objekt in richtige Collection
    
    Args:
        obj: Blender Object
        collection_type: Ziel-Collection Type ('spheres', 'connectors', 'temp')
    """
    from ..core import constants
    
    target_collection = ensure_collection(collection_type)
    
    # Entferne aus anderen Collections
    for collection in obj.users_collection:
        collection.objects.unlink(obj)
    
    # Füge zu Ziel-Collection hinzu
    target_collection.objects.link(obj)
    
    if constants.debug:
        constants.debug.trace('COLLECTIONS', f"Moved {obj.name} to {target_collection.name}")

def get_collection_stats() -> dict:
    """
    Gibt Statistiken über Collections zurück
    
    Returns:
        Dictionary mit Collection-Statistiken
    """
    stats = {}
    
    for collection_type, collection_name in COLLECTION_STRUCTURE.items():
        collection = bpy.data.collections.get(collection_name)
        if collection:
            stats[collection_type] = {
                'name': collection_name,
                'object_count': len(collection.objects),
                'child_count': len(collection.children)
            }
        else:
            stats[collection_type] = {
                'name': collection_name,
                'object_count': 0,
                'child_count': 0
            }
    
    return stats

def cleanup_empty_collections():
    """Entfernt leere Collections"""
    from ..core import constants
    
    removed = 0
    
    # Prüfe alle Kettentool-Collections (außer Root)
    for key, col_name in COLLECTION_STRUCTURE.items():
        if key == 'root':
            continue
            
        collection = bpy.data.collections.get(col_name)
        if collection and not collection.objects and not collection.children:
            try:
                bpy.data.collections.remove(collection)
                removed += 1
                if constants.debug:
                    constants.debug.trace('COLLECTIONS', f"Removed empty collection: {col_name}")
            except:
                pass
    
    if constants.debug and removed > 0:
        constants.debug.info('COLLECTIONS', f"Removed {removed} empty collections")

def validate_collection_structure() -> bool:
    """
    Validiert die Collection-Struktur
    
    Returns:
        True wenn alle Collections korrekt verlinkt sind
    """
    from ..core import constants
    
    try:
        collections = ensure_collection_structure()
        root_col = collections['root']
        
        # Prüfe ob alle Sub-Collections korrekt verlinkt sind
        for key in ['spheres', 'connectors', 'temp']:
            sub_col = collections[key]
            if sub_col not in root_col.children.values():
                if constants.debug:
                    constants.debug.warning('COLLECTIONS', f"Collection {sub_col.name} not linked to root")
                root_col.children.link(sub_col)
        
        return True
        
    except Exception as e:
        if constants.debug:
            constants.debug.error('COLLECTIONS', f"Collection validation failed: {e}")
        return False

# =========================================
# LEGACY SUPPORT
# =========================================

def ensure_collection_legacy(name: str = "KonstruktionsKette") -> bpy.types.Collection:
    """
    Legacy-Unterstützung für alte ensure_collection Aufrufe
    
    Args:
        name: Collection-Name (wird auf neues System gemappt)
        
    Returns:
        Collection-Objekt
    """
    # Mappe alte Namen auf neue Struktur
    if name == "KonstruktionsKette":
        return ensure_collection("root")
    else:
        return ensure_collection("spheres")

# =========================================
# REGISTRATION
# =========================================

def register():
    """Registriert Collections-Modul"""
    from ..core import constants
    
    # Erstelle Collection-Struktur bei Registrierung
    ensure_collection_structure()
    
    if constants.debug:
        constants.debug.info('COLLECTIONS', "Collection structure initialized")

def unregister():
    """Deregistriert Collections-Modul"""
    cleanup_empty_collections()
