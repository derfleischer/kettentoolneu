"""
Chain Tool V4 - Caching System
==============================
Caching für Performance-Optimierung
"""

import time
import hashlib
import pickle
from typing import Dict, Any, Optional, Callable, Tuple
from functools import wraps
from mathutils.bvhtree import BVHTree
import bpy

class CacheManager:
    """Central cache management for Chain Tool"""
    
    def __init__(self, max_size_mb: float = 100):
        self.caches: Dict[str, Dict[str, Any]] = {}
        self.max_size_mb = max_size_mb
        self.access_times: Dict[str, float] = {}
        self.hit_count = 0
        self.miss_count = 0
    
    def create_cache(self, name: str) -> None:
        """Create a new cache category"""
        if name not in self.caches:
            self.caches[name] = {}
    
    def get(self, cache_name: str, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        if cache_name in self.caches and key in self.caches[cache_name]:
            self.hit_count += 1
            self.access_times[f"{cache_name}:{key}"] = time.time()
            return self.caches[cache_name][key]
        
        self.miss_count += 1
        return default
    
    def set(self, cache_name: str, key: str, value: Any) -> None:
        """Set value in cache"""
        if cache_name not in self.caches:
            self.create_cache(cache_name)
        
        self.caches[cache_name][key] = value
        self.access_times[f"{cache_name}:{key}"] = time.time()
        
        # Simple size management - remove oldest if needed
        if len(self.access_times) > 1000:  # Arbitrary limit
            self._cleanup_old_entries()
    
    def invalidate(self, cache_name: Optional[str] = None, key: Optional[str] = None) -> None:
        """Invalidate cache entries"""
        if cache_name is None:
            # Clear all caches
            self.caches.clear()
            self.access_times.clear()
        elif key is None:
            # Clear specific cache
            if cache_name in self.caches:
                self.caches[cache_name].clear()
                # Remove access times for this cache
                keys_to_remove = [k for k in self.access_times if k.startswith(f"{cache_name}:")]
                for k in keys_to_remove:
                    del self.access_times[k]
        else:
            # Clear specific entry
            if cache_name in self.caches and key in self.caches[cache_name]:
                del self.caches[cache_name][key]
                full_key = f"{cache_name}:{key}"
                if full_key in self.access_times:
                    del self.access_times[full_key]
    
    def _cleanup_old_entries(self, keep_ratio: float = 0.7) -> None:
        """Remove oldest cache entries"""
        if not self.access_times:
            return
        
        # Sort by access time
        sorted_items = sorted(self.access_times.items(), key=lambda x: x[1])
        
        # Remove oldest entries
        remove_count = int(len(sorted_items) * (1 - keep_ratio))
        for full_key, _ in sorted_items[:remove_count]:
            cache_name, key = full_key.split(':', 1)
            if cache_name in self.caches and key in self.caches[cache_name]:
                del self.caches[cache_name][key]
            if full_key in self.access_times:
                del self.access_times[full_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_entries = sum(len(cache) for cache in self.caches.values())
        hit_rate = self.hit_count / (self.hit_count + self.miss_count) if (self.hit_count + self.miss_count) > 0 else 0
        
        return {
            'total_entries': total_entries,
            'cache_categories': len(self.caches),
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'caches': {name: len(cache) for name, cache in self.caches.items()}
        }
    
    def clear(self) -> None:
        """Clear all caches"""
        self.caches.clear()
        self.access_times.clear()
        self.hit_count = 0
        self.miss_count = 0

class BVHCache:
    """Specialized cache for BVH trees"""
    
    def __init__(self):
        self.cache: Dict[str, Tuple[BVHTree, float, int]] = {}
        self.max_entries = 10
    
    def get_key(self, obj: bpy.types.Object) -> str:
        """Generate cache key for object"""
        # Use object name and modification time
        key_parts = [
            obj.name,
            str(obj.data.vertices.__len__()) if hasattr(obj.data, 'vertices') else '0',
        ]
        
        # Add modifier info
        for mod in obj.modifiers:
            if mod.show_viewport:
                key_parts.append(f"{mod.name}:{mod.type}")
        
        return hashlib.md5('|'.join(key_parts).encode()).hexdigest()
    
    def get(self, obj: bpy.types.Object) -> Optional[BVHTree]:
        """Get BVH tree from cache"""
        key = self.get_key(obj)
        
        if key in self.cache:
            bvh_tree, timestamp, vertex_count = self.cache[key]
            
            # Validate cache entry
            current_vertex_count = len(obj.data.vertices) if hasattr(obj.data, 'vertices') else 0
            if vertex_count == current_vertex_count:
                return bvh_tree
            else:
                # Invalidate outdated entry
                del self.cache[key]
        
        return None
    
    def set(self, obj: bpy.types.Object, bvh_tree: BVHTree) -> None:
        """Store BVH tree in cache"""
        # Limit cache size
        if len(self.cache) >= self.max_entries:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        key = self.get_key(obj)
        vertex_count = len(obj.data.vertices) if hasattr(obj.data, 'vertices') else 0
        self.cache[key] = (bvh_tree, time.time(), vertex_count)
    
    def invalidate(self, obj: Optional[bpy.types.Object] = None) -> None:
        """Invalidate cache entries"""
        if obj is None:
            self.cache.clear()
        else:
            key = self.get_key(obj)
            if key in self.cache:
                del self.cache[key]
    
    def clear(self) -> None:
        """Clear all cache entries"""
        self.cache.clear()

def cached(cache_name: str, key_func: Optional[Callable] = None):
    """
    Decorator for caching function results
    
    Args:
        cache_name: Name of the cache to use
        key_func: Function to generate cache key from arguments
    
    Usage:
        @cached("my_cache")
        def expensive_function(param):
            # Expensive computation
            return result
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                # Simple key generation
                key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Check cache
            result = cache_manager.get(cache_name, key)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_name, key, result)
            
            return result
        
        return wrapper
    return decorator

# Global instances
cache_manager = CacheManager()
bvh_cache = BVHCache()

def clear_all_caches() -> None:
    """Clear all global caches"""
    cache_manager.clear()
    bvh_cache.clear()
    print("Chain Tool V4: All caches cleared")

def get_cache_stats() -> Dict[str, Any]:
    """Get statistics for all caches"""
    return {
        'general_cache': cache_manager.get_stats(),
        'bvh_cache': {
            'entries': len(bvh_cache.cache),
            'max_entries': bvh_cache.max_entries
        }
    }

# Export
__all__ = [
    'CacheManager',
    'BVHCache',
    'cached',
    'cache_manager',
    'bvh_cache',
    'clear_all_caches',
    'get_cache_stats',
]
