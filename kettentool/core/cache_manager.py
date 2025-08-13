"""
Cache Management System
"""

import bpy
from typing import Dict, Any, Optional
import kettentool.core.constants as constants

# =========================================
# CACHE MANAGER
# =========================================

class CacheManager:
    """Manages various caches for performance"""
    
    def __init__(self):
        self.caches = {}
    
    def create_cache(self, name: str) -> Dict:
        """Create a new cache"""
        if name not in self.caches:
            self.caches[name] = {}
        return self.caches[name]
    
    def get_cache(self, name: str) -> Optional[Dict]:
        """Get a cache by name"""
        return self.caches.get(name)
    
    def clear_cache(self, name: str):
        """Clear a specific cache"""
        if name in self.caches:
            self.caches[name].clear()
    
    def clear_all(self):
        """Clear all caches"""
        for cache in self.caches.values():
            cache.clear()
    
    def remove_cache(self, name: str):
        """Remove a cache entirely"""
        if name in self.caches:
            del self.caches[name]
    
    def get_cache_size(self, name: str) -> int:
        """Get the size of a cache"""
        if name in self.caches:
            return len(self.caches[name])
        return 0
    
    def get_total_size(self) -> int:
        """Get total size of all caches"""
        return sum(len(cache) for cache in self.caches.values())

# Global cache manager instance
_cache_manager = CacheManager()

# =========================================
# PUBLIC FUNCTIONS
# =========================================

def get_or_create_cache(name: str) -> Dict:
    """Get or create a cache"""
    return _cache_manager.create_cache(name)

def clear_cache(name: str):
    """Clear a specific cache"""
    _cache_manager.clear_cache(name)
    
    # Also clear from constants
    if name == 'aura':
        constants.get_aura_cache().clear()
    elif name == 'spheres':
        constants.get_sphere_registry().clear()
    elif name == 'connectors':
        constants.get_connector_registry().clear()

def cleanup_all_caches():
    """Clear all caches"""
    _cache_manager.clear_all()
    constants.clear_all_caches()
    
    if constants.debug:
        constants.debug.info('CACHE', "All caches cleared")

def invalidate_object_cache(obj_name: str):
    """Invalidate caches related to an object"""
    # Clear aura cache for this object
    aura_cache = constants.get_aura_cache()
    if obj_name in aura_cache:
        del aura_cache[obj_name]
    
    # Clear from cache manager
    for cache in _cache_manager.caches.values():
        if obj_name in cache:
            del cache[obj_name]

def get_cache_statistics() -> Dict[str, Any]:
    """Get cache statistics"""
    stats = {
        'total_caches': len(_cache_manager.caches),
        'total_entries': _cache_manager.get_total_size(),
        'caches': {}
    }
    
    for name, cache in _cache_manager.caches.items():
        stats['caches'][name] = {
            'size': len(cache),
            'keys': list(cache.keys())[:10]  # First 10 keys
        }
    
    # Add constants caches
    stats['caches']['sphere_registry'] = {
        'size': len(constants.get_sphere_registry())
    }
    stats['caches']['connector_registry'] = {
        'size': len(constants.get_connector_registry())
    }
    stats['caches']['aura_cache'] = {
        'size': len(constants.get_aura_cache())
    }
    
    return stats

# =========================================
# CLEANUP UTILITIES
# =========================================

def cleanup_invalid_references():
    """Remove invalid object references from caches"""
    cleaned = 0
    
    # Clean sphere registry
    sphere_reg = constants.get_sphere_registry()
    invalid_spheres = []
    for key, obj in sphere_reg.items():
        try:
            # Test if object is still valid
            _ = obj.name
        except ReferenceError:
            invalid_spheres.append(key)
    
    for key in invalid_spheres:
        del sphere_reg[key]
        cleaned += 1
    
    # Clean connector registry
    conn_reg = constants.get_connector_registry()
    invalid_conns = []
    for key, obj in conn_reg.items():
        try:
            _ = obj.name
        except ReferenceError:
            invalid_conns.append(key)
    
    for key in invalid_conns:
        del conn_reg[key]
        cleaned += 1
    
    if cleaned > 0 and constants.debug:
        constants.debug.info('CACHE', f"Cleaned {cleaned} invalid references")
    
    return cleaned

# =========================================
# REGISTRATION
# =========================================

def register():
    """Register cache manager"""
    pass

def unregister():
    """Unregister cache manager"""
    cleanup_all_caches()
