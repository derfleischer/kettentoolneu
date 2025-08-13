"""
Global Constants and Shared Variables
"""

import bpy
from typing import Dict, Set, Any, Optional

# =========================================
# CONSTANTS
# =========================================

EPSILON = 0.001
DEFAULT_SPHERE_RADIUS = 0.01
DEFAULT_CONNECTOR_RADIUS = 0.003
DEFAULT_SURFACE_OFFSET = 0.002

# Collection Names
COLLECTION_SPHERES = "Chain_Spheres"
COLLECTION_CONNECTORS = "Chain_Connectors"
COLLECTION_MAIN = "Chain_Construction"

# Material Names
MAT_SPHERE_START = "Chain_Sphere_Start"
MAT_SPHERE_DEFAULT = "Chain_Sphere_Default"
MAT_CONNECTOR = "Chain_Connector"

# Colors
COLOR_START = (1.0, 0.0, 0.0, 1.0)  # Red
COLOR_DEFAULT = (0.5, 0.5, 0.5, 1.0)  # Gray
COLOR_CONNECTOR = (0.2, 0.2, 0.2, 1.0)  # Dark Gray

# =========================================
# GLOBAL CACHES
# =========================================

# Sphere Registry
_sphere_registry: Dict[str, Any] = {}
_sphere_counter: int = 0

# Connector Registry  
_connector_registry: Dict[str, Any] = {}
_connector_pairs: Set[tuple] = set()

# Surface Cache
_aura_cache: Dict[str, Any] = {}

# Debug System Instance
debug: Optional[Any] = None
performance_monitor: Optional[Any] = None

# =========================================
# CACHE ACCESS FUNCTIONS
# =========================================

def get_sphere_registry():
    """Returns sphere registry"""
    global _sphere_registry
    return _sphere_registry

def get_connector_registry():
    """Returns connector registry"""
    global _connector_registry
    return _connector_registry

def get_aura_cache():
    """Returns aura cache"""
    global _aura_cache
    return _aura_cache

def clear_all_caches():
    """Clears all global caches"""
    global _sphere_registry, _connector_registry, _connector_pairs, _aura_cache
    _sphere_registry.clear()
    _connector_registry.clear()
    _connector_pairs.clear()
    _aura_cache.clear()

def increment_sphere_counter():
    """Increments and returns sphere counter"""
    global _sphere_counter
    _sphere_counter += 1
    return _sphere_counter

def reset_sphere_counter():
    """Resets sphere counter"""
    global _sphere_counter
    _sphere_counter = 0

# =========================================
# REGISTRATION
# =========================================

def register():
    """Register constants module"""
    pass

def unregister():
    """Unregister constants module"""
    clear_all_caches()
