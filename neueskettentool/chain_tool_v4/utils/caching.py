"""
Caching System for Chain Tool V4
Intelligent caching for performance optimization
"""

import time
import hashlib
import weakref
from typing import Dict, Any, Tuple, Optional, Callable
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
import bpy

from config import ENABLE_BVH_CACHE, CACHE_SIZE_LIMIT_MB, CACHE_TIMEOUT_SECONDS
from utils.debug import debug

# ============================================
# CACHE MANAGER
# ============================================

class CacheManager:
    """Central cache management system"""
    
    def __init__(self):
        self.caches = {}
        self.total_size = 0
        self.hit_count = 0
        self.miss_count = 0
        
    def register_cache(self, name: str, cache_instance):
        """Register a cache instance"""
        self.caches[name] = cache_instance
        debug.debug('CACHE', f"Registered cache: {name}")
        
    def clear_all(self):
        """Clear all registered caches"""
        for name, cache in self.caches.items():
            if hasattr(cache, 'clear'):
                cache.clear()
        
        self.total_size = 0
        self.hit_count = 0
        self.miss_count = 0
        debug.info('CACHE', "All caches cleared")
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            'total_caches': len(self.caches),
            'total_size_mb': self.total_size / 1024 / 1024,
            'hit_rate': self.hit_count / (self.hit_count + self.miss_count) if (self.hit_count + self.miss_count) > 0 else 0,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'cache_details': {}
        }
        
        for name, cache in self.caches.items():
            if hasattr(cache, 'get_stats'):
                stats['cache_details'][name] = cache.get_stats()
                
        return stats
        
    def record_hit(self):
        """Record a cache hit"""
        self.hit_count += 1
        
    def record_miss(self):
        """Record a cache miss"""
        self.miss_count += 1

# ============================================
# BVH CACHE
# ============================================

class BVHCache:
    """Cache for BVH trees to avoid reconstruction"""
    
    def __init__(self):
        self._cache = {}
        self._last_cleanup = time.time()
        self.enabled = ENABLE_BVH_CACHE
        
        # Register with manager
        cache_manager.register_cache('BVH', self)
        
    def get_or_create(self, obj: bpy.types.Object) -> Tuple[BVHTree, Matrix]:
        """Get cached BVH tree or create new one"""
        if not self.enabled or obj.type != 'MESH':
            return self._create_bvh(obj)
            
        # Generate cache key
        cache_key = self._generate_key(obj)
        
        # Check cache
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            
            # Validate cache entry
            if self._is_valid(obj, entry):
                cache_manager.record_hit()
                debug.trace('CACHE', f"BVH cache hit for {obj.name}")
                return entry['bvh'], entry['matrix'].copy()
                
        # Cache miss - create new
        cache_manager.record_miss()
        bvh, matrix = self._create_bvh(obj)
        
        # Store in cache
        self._cache[cache_key] = {
            'bvh': bvh,
            'matrix': matrix.copy(),
            'timestamp': time.time(),
            'vertex_count': len(obj.data.vertices),
            'update_tag': obj.data.tag
        }
        
        # Cleanup old entries periodically
        self._cleanup_if_needed()
        
        debug.debug('CACHE', f"Created new BVH for {obj.name}")
        return bvh, matrix
        
    def _create_bvh(self, obj: bpy.types.Object) -> Tuple[BVHTree, Matrix]:
        """Create new BVH tree"""
        import bmesh
        
        # Get mesh data
        mesh = obj.data
        matrix = obj.matrix_world.copy()
        
        # Create BVH
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.transform(matrix)
        
        bvh = BVHTree.FromBMesh(bm)
        bm.free()
        
        return bvh, matrix
        
    def _generate_key(self, obj: bpy.types.Object) -> str:
        """Generate unique cache key for object"""
        # Include object name and data block name
        key_string = f"{obj.name}_{obj.data.name}_{id(obj.data)}"
        return hashlib.md5(key_string.encode()).hexdigest()
        
    def _is_valid(self, obj: bpy.types.Object, entry: Dict) -> bool:
        """Check if cache entry is still valid"""
        # Check if mesh was modified
        if obj.data.tag != entry['update_tag']:
            return False
            
        # Check vertex count
        if len(obj.data.vertices) != entry['vertex_count']:
            return False
            
        # Check timeout
        if time.time() - entry['timestamp'] > CACHE_TIMEOUT_SECONDS:
            return False
            
        return True
        
    def _cleanup_if_needed(self):
        """Remove old cache entries if needed"""
        current_time = time.time()
        
        # Only cleanup every 60 seconds
        if current_time - self._last_cleanup < 60:
            return
            
        self._last_cleanup = current_time
        
        # Remove expired entries
        expired_keys = []
        for key, entry in self._cache.items():
            if current_time - entry['timestamp'] > CACHE_TIMEOUT_SECONDS:
                expired_keys.append(key)
                
        for key in expired_keys:
            del self._cache[key]
            
        if expired_keys:
            debug.debug('CACHE', f"Cleaned {len(expired_keys)} expired BVH entries")
            
    def invalidate(self, obj: bpy.types.Object):
        """Invalidate cache for specific object"""
        cache_key = self._generate_key(obj)
        if cache_key in self._cache:
            del self._cache[cache_key]
            debug.debug('CACHE', f"Invalidated BVH cache for {obj.name}")
            
    def clear(self):
        """Clear all cached BVH trees"""
        self._cache.clear()
        debug.info('CACHE', "BVH cache cleared")
        
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'entries': len(self._cache),
            'enabled': self.enabled
        }

# ============================================
# PATTERN CACHE
# ============================================

class PatternCache:
    """Cache for generated patterns"""
    
    def __init__(self):
        self._cache = {}
        self._size_limit = CACHE_SIZE_LIMIT_MB * 1024 * 1024  # Convert to bytes
        self._current_size = 0
        
        # Register with manager
        cache_manager.register_cache('Pattern', self)
        
    def get(self, key: str) -> Optional[Any]:
        """Get cached pattern data"""
        if key in self._cache:
            entry = self._cache[key]
            entry['last_access'] = time.time()
            entry['access_count'] += 1
            cache_manager.record_hit()
            
            debug.trace('CACHE', f"Pattern cache hit: {key}")
            return entry['data']
            
        cache_manager.record_miss()
        return None
        
    def set(self, key: str, data: Any, size_estimate: int = 1000):
        """Cache pattern data"""
        # Check size limit
        if self._current_size + size_estimate > self._size_limit:
            self._evict_lru()
            
        self._cache[key] = {
            'data': data,
            'timestamp': time.time(),
            'last_access': time.time(),
            'access_count': 1,
            'size': size_estimate
        }
        
        self._current_size += size_estimate
        debug.debug('CACHE', f"Cached pattern: {key}")
        
    def _evict_lru(self):
        """Evict least recently used entries"""
        if not self._cache:
            return
            
        # Sort by last access time
        sorted_entries = sorted(self._cache.items(), 
                               key=lambda x: x[1]['last_access'])
        
        # Remove oldest 25%
        remove_count = max(1, len(sorted_entries) // 4)
        
        for key, entry in sorted_entries[:remove_count]:
            self._current_size -= entry['size']
            del self._cache[key]
            
        debug.debug('CACHE', f"Evicted {remove_count} pattern cache entries")
        
    def clear(self):
        """Clear pattern cache"""
        self._cache.clear()
        self._current_size = 0
        debug.info('CACHE', "Pattern cache cleared")
        
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'entries': len(self._cache),
            'size_mb': self._current_size / 1024 / 1024,
            'limit_mb': self._size_limit / 1024 / 1024
        }

# ============================================
# GENERIC OBJECT CACHE
# ============================================

class ObjectCache:
    """Generic cache for any objects with weak references"""
    
    def __init__(self, name: str):
        self.name = name
        self._cache = weakref.WeakValueDictionary()
        
        # Register with manager
        cache_manager.register_cache(name, self)
        
    def get(self, key: str) -> Optional[Any]:
        """Get cached object"""
        obj = self._cache.get(key)
        if obj:
            cache_manager.record_hit()
            debug.trace('CACHE', f"{self.name} cache hit: {key}")
        else:
            cache_manager.record_miss()
        return obj
        
    def set(self, key: str, obj: Any):
        """Cache object with weak reference"""
        self._cache[key] = obj
        debug.trace('CACHE', f"{self.name} cached: {key}")
        
    def clear(self):
        """Clear cache"""
        self._cache.clear()
        debug.debug('CACHE', f"{self.name} cache cleared")
        
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'entries': len(self._cache)
        }

# ============================================
# MEMOIZATION DECORATOR
# ============================================

def memoize(cache_name: str = "Function", timeout: float = 300):
    """Decorator for memoizing function results"""
    cache = {}
    
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key = str(args) + str(sorted(kwargs.items()))
            key_hash = hashlib.md5(key.encode()).hexdigest()
            
            # Check cache
            if key_hash in cache:
                entry = cache[key_hash]
                if time.time() - entry['timestamp'] < timeout:
                    debug.trace('CACHE', f"Memoized result for {func.__name__}")
                    return entry['result']
                    
            # Compute result
            result = func(*args, **kwargs)
            
            # Cache result
            cache[key_hash] = {
                'result': result,
                'timestamp': time.time()
            }
            
            # Cleanup old entries
            current_time = time.time()
            expired = [k for k, v in cache.items() 
                      if current_time - v['timestamp'] > timeout]
            for k in expired:
                del cache[k]
                
            return result
            
        wrapper._cache = cache
        wrapper._clear_cache = lambda: cache.clear()
        
        return wrapper
    return decorator

# ============================================
# GLOBAL INSTANCES
# ============================================

# Create global cache instances
cache_manager = CacheManager()
bvh_cache = BVHCache()
pattern_cache = PatternCache()
mesh_cache = ObjectCache("Mesh")
material_cache = ObjectCache("Material")

# ============================================
# UTILITY FUNCTIONS
# ============================================

def clear_all_caches():
    """Clear all caches"""
    cache_manager.clear_all()
    
def get_cache_report() -> str:
    """Get formatted cache report"""
    stats = cache_manager.get_statistics()
    
    report = [
        "=== Cache Statistics ===",
        f"Total Caches: {stats['total_caches']}",
        f"Total Size: {stats['total_size_mb']:.2f} MB",
        f"Hit Rate: {stats['hit_rate']:.1%}",
        f"Hits: {stats['hit_count']} | Misses: {stats['miss_count']}",
        "",
        "=== Cache Details ==="
    ]
    
    for name, details in stats['cache_details'].items():
        report.append(f"\n{name} Cache:")
        for key, value in details.items():
            report.append(f"  {key}: {value}")
            
    return "\n".join(report)

# ============================================
# OPERATORS
# ============================================

class CHAIN_TOOL_OT_clear_caches(bpy.types.Operator):
    """Clear all caches"""
    bl_idname = "chain_tool.clear_caches"
    bl_label = "Clear Caches"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        clear_all_caches()
        self.report({'INFO'}, "All caches cleared")
        return {'FINISHED'}

class CHAIN_TOOL_OT_cache_report(bpy.types.Operator):
    """Show cache statistics"""
    bl_idname = "chain_tool.cache_report"
    bl_label = "Cache Report"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        report = get_cache_report()
        for line in report.split('\n'):
            if line.strip():
                self.report({'INFO'}, line)
        return {'FINISHED'}

# Classes to register
CLASSES = [
    CHAIN_TOOL_OT_clear_caches,
    CHAIN_TOOL_OT_cache_report,
]
