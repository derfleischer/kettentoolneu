"""
Constants and Enums for Chain Tool V4
All constant values and enumerations used throughout the addon
"""

from enum import Enum, auto
import bpy

# ============================================
# PATTERN TYPES
# ============================================

class PatternType(Enum):
    """Available pattern generation types"""
    # Surface Patterns
    VORONOI = "voronoi"
    HEXAGONAL = "hexagonal"
    TRIANGULAR = "triangular"
    ISLANDS = "islands"
    
    # Edge Patterns
    CONTOUR = "contour"
    RIM_REINFORCEMENT = "rim"
    STRESS_EDGE = "stress_edge"
    
    # Hybrid Patterns
    GRADIENT_DENSITY = "gradient"
    ZONAL = "zonal"
    ADAPTIVE = "adaptive"
    
    # Special
    PAINT = "paint"
    CUSTOM = "custom"

# ============================================
# CONNECTION TYPES
# ============================================

class ConnectionType(Enum):
    """Types of connections between points"""
    LINEAR = "linear"
    CURVED = "curved"
    TRIANGULATED = "triangulated"
    EDGE_BASED = "edge_based"
    NEAREST_NEIGHBOR = "nearest"

# ============================================
# EXPORT MODES
# ============================================

class ExportMode(Enum):
    """Export configurations"""
    SINGLE_MESH = "single"
    DUAL_MATERIAL = "dual"
    SEPARATE_LAYERS = "separate"
    WITH_SUPPORTS = "supports"

# ============================================
# ORTHOTIC TYPES
# ============================================

class OrthoticType(Enum):
    """Types of orthotics for specialized patterns"""
    AFO = "afo"  # Ankle Foot Orthosis
    KAFO = "kafo"  # Knee Ankle Foot Orthosis  
    WHO = "who"  # Wrist Hand Orthosis
    TLSO = "tlso"  # Thoraco Lumbo Sacral Orthosis
    CUSTOM = "custom"

# ============================================
# COLLECTIONS
# ============================================

# Collection names
COLLECTIONS = {
    'MAIN': "ChainTool_Objects",
    'PATTERN': "CT_Pattern",
    'SHELL': "CT_Shell", 
    'TEMP': "CT_Temp",
    'EXPORT': "CT_Export"
}

# ============================================
# OBJECT NAMING
# ============================================

# Naming conventions
OBJECT_PREFIXES = {
    'SPHERE': "CT_Sphere_",
    'CONNECTOR': "CT_Connector_",
    'PATTERN': "CT_Pattern_",
    'SHELL': "CT_Shell_",
    'EDGE': "CT_Edge_",
    'PAINT': "CT_Paint_"
}

# ============================================
# PROPERTY ENUMS FOR BLENDER
# ============================================

# Pattern type items for EnumProperty
PATTERN_TYPE_ITEMS = [
    ('SURFACE', "Surface Patterns", "Large area coverage patterns", 'MESH_GRID', 0),
    ('VORONOI', "Voronoi", "Organic cell structure", 'MESH_MONKEY', 1),
    ('HEXAGONAL', "Hexagonal", "Honeycomb pattern", 'MESH_CIRCLE', 2),
    ('TRIANGULAR', "Triangular", "Triangle mesh pattern", 'MESH_CONE', 3),
    ('ISLANDS', "Islands", "Scattered island pattern", 'MESH_TORUS', 4),
    
    ('EDGE', "Edge Patterns", "Edge reinforcement patterns", 'MESH_CUBE', 10),
    ('CONTOUR', "Contour Bands", "Multiple edge bands", 'CURVE_BEZCIRCLE', 11),
    ('RIM', "Rim Reinforcement", "Strong outer rim", 'CURVE_NCIRCLE', 12),
    
    ('HYBRID', "Hybrid Patterns", "Combined patterns", 'MESH_UVSPHERE', 20),
    ('GRADIENT', "Gradient Density", "Variable density pattern", 'MESH_ICOSPHERE', 21),
    ('ADAPTIVE', "Adaptive", "Stress-based adaptation", 'MESH_CYLINDER', 22),
]

# Connection mode items
CONNECTION_MODE_ITEMS = [
    ('AUTO', "Auto", "Automatic triangulation", 'MESH_DATA', 0),
    ('NEAREST', "Nearest", "Connect to nearest neighbors", 'PARTICLES', 1),
    ('TRIANGULATE', "Triangulate", "Delaunay triangulation", 'MESH_CONE', 2),
    ('PAINT', "Paint", "Manual paint connections", 'BRUSH_DATA', 3),
]

# Material preset items
MATERIAL_PRESET_ITEMS = [
    ('TPU_PETG', "TPU + PETG", "Soft shell with rigid pattern", 'SHADING_RENDERED', 0),
    ('TPU_PETG_CF', "TPU + PETG-CF", "Soft shell with carbon fiber pattern", 'SHADING_SOLID', 1),
    ('SINGLE_PETG', "Single PETG", "All PETG construction", 'SHADING_WIRE', 2),
    ('CUSTOM', "Custom", "User defined materials", 'SHADING_TEXTURE', 3),
]

# ============================================
# MATHEMATICAL CONSTANTS
# ============================================

# Epsilon for float comparisons
EPSILON = 0.0001
ANGLE_EPSILON = 0.1  # degrees

# Default sizes (in mm)
DEFAULT_SPHERE_RADIUS = 1.5
DEFAULT_CONNECTOR_RADIUS = 0.8
MIN_CONNECTOR_LENGTH = 0.5
MAX_CONNECTOR_LENGTH = 50.0

# Pattern constraints
MIN_PATTERN_SPACING = 2.0
MAX_PATTERN_SPACING = 50.0
MIN_EDGE_DISTANCE = 1.0

# ============================================
# UI CONSTANTS
# ============================================

# Icon mappings
ICON_MAPPING = {
    'PATTERN': 'MESH_GRID',
    'EDGE': 'MESH_CUBE',
    'PAINT': 'BRUSH_DATA',
    'EXPORT': 'EXPORT',
    'SETTINGS': 'PREFERENCES',
    'HELP': 'QUESTION',
}

# UI Layout
UI_SCALE = 1.0
PANEL_SPACING = 5
BUTTON_HEIGHT = 25

# ============================================
# ERROR MESSAGES
# ============================================

ERROR_MESSAGES = {
    'NO_TARGET': "No target object selected",
    'INVALID_MESH': "Selected object is not a valid mesh",
    'NO_PATTERN': "No pattern points generated",
    'EXPORT_FAILED': "Export failed - check file permissions",
    'PATTERN_FAILED': "Pattern generation failed",
    'CONNECTION_FAILED': "Connection generation failed",
}

# Warning messages
WARNING_MESSAGES = {
    'HIGH_VERTEX_COUNT': "High vertex count may affect performance",
    'LONG_CONNECTIONS': "Some connections exceed recommended length",
    'OVERLAPPING_POINTS': "Detected overlapping points",
}

# ============================================
# PERFORMANCE THRESHOLDS
# ============================================

# When to show warnings
PERFORMANCE_WARNING_VERTICES = 50000
PERFORMANCE_WARNING_POINTS = 5000
PERFORMANCE_WARNING_CONNECTIONS = 10000

# Background processing thresholds
BACKGROUND_PROCESSING_VERTICES = 100000
BACKGROUND_PROCESSING_POINTS = 10000

# ============================================
# COLOR DEFINITIONS
# ============================================

# Material colors (RGBA)
COLORS = {
    'SHELL_MATERIAL': (0.9, 0.9, 0.9, 1.0),
    'PATTERN_MATERIAL': (0.2, 0.4, 0.8, 1.0),
    'EDGE_HIGHLIGHT': (1.0, 0.5, 0.0, 1.0),
    'ERROR': (1.0, 0.0, 0.0, 1.0),
    'WARNING': (1.0, 0.7, 0.0, 1.0),
    'SUCCESS': (0.0, 1.0, 0.0, 1.0),
}

# ============================================
# HELPER FUNCTIONS
# ============================================

def get_collection(collection_type):
    """Get or create a collection by type"""
    name = COLLECTIONS.get(collection_type, "ChainTool_Misc")
    
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    
    # Create new collection
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    return collection

def get_object_name(prefix_type, index=None):
    """Generate consistent object names"""
    prefix = OBJECT_PREFIXES.get(prefix_type, "CT_Object_")
    
    if index is not None:
        return f"{prefix}{index:04d}"
    
    # Find next available index
    existing = [obj.name for obj in bpy.data.objects if obj.name.startswith(prefix)]
    index = len(existing)
    
    while f"{prefix}{index:04d}" in existing:
        index += 1
    
    return f"{prefix}{index:04d}"

# ============================================
# VALIDATION FUNCTIONS
# ============================================

def is_valid_mesh(obj):
    """Check if object is a valid mesh"""
    return obj and obj.type == 'MESH' and len(obj.data.vertices) > 0

def is_valid_pattern_spacing(spacing):
    """Validate pattern spacing value"""
    return MIN_PATTERN_SPACING <= spacing <= MAX_PATTERN_SPACING
