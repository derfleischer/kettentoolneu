"""
Blender Collection Management Utilities
"""

import bpy
from typing import Optional
import kettentool.core.constants as constants

# =========================================
# COLLECTION MANAGEMENT
# =========================================

def ensure_collection(name: str, parent: Optional[bpy.types.Collection] = None) -> bpy.types.Collection:
    """Ensure a collection exists, create if necessary"""
    # Check if collection exists
    if name in bpy.data.collections:
        collection = bpy.data.collections[name]
    else:
        # Create new collection
        collection = bpy.data.collections.new(name)
        
        if constants.debug:
            constants.debug.info('COLLECTION', f"Created collection: {name}")
    
    # Link to parent or scene
    if parent:
        if collection.name not in parent.children:
            parent.children.link(collection)
    else:
        # Link to scene collection if not already linked
        if collection.name not in bpy.context.scene.collection.children:
            bpy.context.scene.collection.children.link(collection)
    
    return collection

def get_chain_collection() -> bpy.types.Collection:
    """Get or create main chain collection"""
    return ensure_collection(constants.COLLECTION_MAIN)

def get_spheres_collection() -> bpy.types.Collection:
    """Get or create spheres collection"""
    main_collection = get_chain_collection()
    return ensure_collection(constants.COLLECTION_SPHERES, main_collection)

def get_connectors_collection() -> bpy.types.Collection:
    """Get or create connectors collection"""
    main_collection = get_chain_collection()
    return ensure_collection(constants.COLLECTION_CONNECTORS, main_collection)

def move_to_collection(obj: bpy.types.Object, collection_name: str):
    """Move object to specified collection"""
    # Ensure collection exists
    collection = ensure_collection(collection_name)
    
    # Remove from all collections
    for coll in obj.users_collection:
        coll.objects.unlink(obj)
    
    # Add to target collection
    collection.objects.link(obj)
    
    if constants.debug:
        constants.debug.debug('COLLECTION', f"Moved {obj.name} to {collection_name}")

def cleanup_empty_collections():
    """Remove empty collections"""
    removed = []
    
    # Find empty collections
    for collection in bpy.data.collections:
        if len(collection.objects) == 0 and len(collection.children) == 0:
            # Don't remove main collections
            if collection.name not in [constants.COLLECTION_MAIN, 
                                      constants.COLLECTION_SPHERES,
                                      constants.COLLECTION_CONNECTORS]:
                removed.append(collection.name)
    
    # Remove empty collections
    for name in removed:
        collection = bpy.data.collections[name]
        bpy.data.collections.remove(collection)
    
    if removed and constants.debug:
        constants.debug.info('COLLECTION', f"Removed {len(removed)} empty collections")
    
    return len(removed)

def get_objects_in_collection(collection_name: str) -> list:
    """Get all objects in a collection"""
    if collection_name not in bpy.data.collections:
        return []
    
    collection = bpy.data.collections[collection_name]
    return list(collection.objects)

def clear_collection(collection_name: str, remove_collection: bool = False):
    """Clear all objects from collection"""
    if collection_name not in bpy.data.collections:
        return
    
    collection = bpy.data.collections[collection_name]
    
    # Remove all objects
    for obj in list(collection.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    
    # Optionally remove collection itself
    if remove_collection:
        bpy.data.collections.remove(collection)
    
    if constants.debug:
        constants.debug.info('COLLECTION', f"Cleared collection: {collection_name}")

def organize_scene():
    """Organize all chain objects into proper collections"""
    organized = 0
    
    # Find all chain objects
    for obj in bpy.data.objects:
        if obj.name.startswith("Kugel"):
            move_to_collection(obj, constants.COLLECTION_SPHERES)
            organized += 1
        elif obj.name.startswith("Connector") or obj.name.startswith("CurvedConnector"):
            move_to_collection(obj, constants.COLLECTION_CONNECTORS)
            organized += 1
    
    if organized > 0 and constants.debug:
        constants.debug.info('COLLECTION', f"Organized {organized} objects")
    
    return organized

# =========================================
# REGISTRATION
# =========================================

def register():
    """Register collections module"""
    pass

def unregister():
    """Unregister collections module"""
    pass
