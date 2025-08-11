"""
Konstanten und globale Variablen für das Kettentool
ERWEITERT: Mit Sphere Registry und Collection-Struktur
"""

# =========================================
# CONSTANTS
# =========================================

EPSILON = 1e-6

# =========================================
# COLLECTION STRUKTUR
# =========================================

COLLECTION_STRUCTURE = {
    'root': 'KonstruktionsKette',
    'spheres': 'KonstruktionsKette_Spheres',
    'connectors': 'KonstruktionsKette_Connectors',
    'temp': 'KonstruktionsKette_Temp'
}

# =========================================
# GLOBAL CACHES
# =========================================

# Aura Cache - {cache_key: aura_object}
_aura_cache = {}

# BVH Cache - {object_name: (bvh_tree, matrix, data_id)}
_bvh_cache = {}

# Connector Cache - {(sphere1_name, sphere2_name): connector_object}
_existing_connectors = {}

# Remembered Pairs - {sphere_name: [connected_sphere_names]}
_remembered_pairs = {}

# =========================================
# ERWEITERTE SPHERE CACHES
# =========================================

# Sphere Registry - {sphere_name: sphere_data}
_sphere_registry = {}

# Sphere Collections - {collection_type: [sphere_names]}
_sphere_collections = {
    'start_spheres': [],     # Start-Kugeln (rot)
    'painted_spheres': [],   # Gemalte Kugeln (blau)
    'manual_spheres': [],    # Manuell platzierte (grau)
    'generated_spheres': []  # Automatisch generierte (grün)
}

# =========================================
# GLOBAL INSTANCES
# =========================================

# Wird in __init__.py initialisiert
debug = None
performance_monitor = None

# =========================================
# REGISTRATION
# =========================================

def register():
    """Registriert Constants-Modul"""
    pass

def unregister():
    """Deregistriert Constants-Modul"""
    pass
