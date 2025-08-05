"""
Chain Tool V4 - Constants and Enums
===================================
Zentrale Konstanten und Enumerationen
"""

from enum import Enum

# Version Information
ADDON_VERSION = (4, 0, 0)
BLENDER_MIN_VERSION = (4, 0, 0)

# Default Values
DEFAULT_SPHERE_SIZE = 2.0
DEFAULT_CONNECTION_THICKNESS = 0.8
DEFAULT_PATTERN_DENSITY = 5.0
DEFAULT_SHELL_THICKNESS = 1.2
DEFAULT_EDGE_ANGLE = 30.0
DEFAULT_EDGE_WIDTH = 5.0
DEFAULT_BRUSH_SIZE = 10.0

# Limits
MIN_SPHERE_SIZE = 0.1
MAX_SPHERE_SIZE = 10.0
MIN_CONNECTION_THICKNESS = 0.1
MAX_CONNECTION_THICKNESS = 5.0
MIN_PATTERN_DENSITY = 0.1
MAX_PATTERN_DENSITY = 20.0
MIN_SHELL_THICKNESS = 0.1
MAX_SHELL_THICKNESS = 5.0

# Pattern Types
class PatternType(Enum):
    """Main pattern types"""
    SURFACE = "surface"
    EDGE = "edge"
    HYBRID = "hybrid"
    PAINT = "paint"

class SurfacePatternType(Enum):
    """Surface pattern subtypes"""
    VORONOI = "voronoi"
    HEXAGONAL = "hexagonal"
    TRIANGULAR = "triangular"
    RANDOM = "random"
    ADAPTIVE = "adaptive"

class EdgePatternType(Enum):
    """Edge pattern subtypes"""
    RIM = "rim"
    CONTOUR = "contour"
    STRESS = "stress"
    CORNER = "corner"

class HybridPatternType(Enum):
    """Hybrid pattern subtypes"""
    CORE_SHELL = "core_shell"
    GRADIENT = "gradient"
    ZONAL = "zonal"

# Material Types
class MaterialType(Enum):
    """3D printing material types"""
    TPU = "tpu"
    PETG = "petg"
    PLA = "pla"
    ABS = "abs"
    NYLON = "nylon"

# Export Formats
class ExportFormat(Enum):
    """Supported export formats"""
    STL = "stl"
    OBJ = "obj"
    PLY = "ply"
    FBX = "fbx"
    GLTF = "gltf"

# Quality Levels
class QualityLevel(Enum):
    """Quality presets"""
    DRAFT = "draft"
    NORMAL = "normal"
    HIGH = "high"
    ULTRA = "ultra"

# Printer Profiles
class PrinterProfile(Enum):
    """Common 3D printer profiles"""
    PRUSA_MK3 = "prusa_mk3"
    ULTIMAKER_S5 = "ultimaker_s5"
    CREALITY_ENDER3 = "creality_ender3"
    CUSTOM = "custom"

# Algorithm Settings
class AlgorithmType(Enum):
    """Algorithm types for pattern generation"""
    DELAUNAY = "delaunay"
    VORONOI = "voronoi"
    POISSON = "poisson"
    LLOYD = "lloyd"

# Load Cases (for stress analysis)
class LoadCase(Enum):
    """Biomechanical load cases"""
    WEIGHT_BEARING = "weight_bearing"
    FLEXION = "flexion"
    EXTENSION = "extension"
    ROTATION = "rotation"
    LATERAL = "lateral"
    COMBINED = "combined"

# Stress Types
class StressType(Enum):
    """Types of stress analysis"""
    COMPRESSION = "compression"
    TENSION = "tension"
    SHEAR = "shear"
    BENDING = "bending"
    TORSION = "torsion"

# Collision Types
class CollisionType(Enum):
    """Types of collisions in pattern"""
    SPHERE_SPHERE = "sphere_sphere"
    SPHERE_CONNECTOR = "sphere_connector"
    CONNECTOR_CONNECTOR = "connector_connector"
    SPHERE_SURFACE = "sphere_surface"
    CONNECTOR_SURFACE = "connector_surface"

# Resolution Strategies
class ResolutionStrategy(Enum):
    """Strategies for collision resolution"""
    MOVE_SMALLEST = "move_smallest"
    MOVE_BOTH = "move_both"
    RESIZE_SMALLER = "resize_smaller"
    REMOVE_WEAKEST = "remove_weakest"
    REROUTE_CONNECTION = "reroute_connection"
    SURFACE_PROJECT = "surface_project"

# Gap Types
class GapType(Enum):
    """Types of gaps in patterns"""
    SMALL_HOLE = "small_hole"
    LARGE_VOID = "large_void"
    EDGE_GAP = "edge_gap"
    CONNECTION_BREAK = "connection_break"
    STRESS_CONCENTRATION = "stress_concentration"
    SURFACE_DEVIATION = "surface_deviation"

# Fill Strategies
class FillStrategy(Enum):
    """Strategies for gap filling"""
    SPHERE_INSERTION = "sphere_insertion"
    CONNECTION_BRIDGING = "connection_bridging"
    PATTERN_EXTENSION = "pattern_extension"
    MESH_DENSIFICATION = "mesh_densification"
    ADAPTIVE_SCALING = "adaptive_scaling"

# Connection Types
class ConnectionType(Enum):
    """Types of connections between spheres"""
    STRAIGHT = "straight"
    CURVED = "curved"
    ADAPTIVE = "adaptive"
    SPRING = "spring"
    RIGID = "rigid"

# Triangulation Methods
class TriangulationMethod(Enum):
    """Methods for triangulation"""
    DELAUNAY_2D = "delaunay_2d"
    DELAUNAY_3D = "delaunay_3d"
    CONSTRAINED = "constrained"
    CONFORMING = "conforming"
    INCREMENTAL = "incremental"

# UI Icons
ICON_MAPPING = {
    PatternType.SURFACE: 'MESH_UVSPHERE',
    PatternType.EDGE: 'EDGESEL',
    PatternType.HYBRID: 'MOD_LATTICE',
    PatternType.PAINT: 'BRUSH_DATA',
    
    SurfacePatternType.VORONOI: 'MESH_ICOSPHERE',
    SurfacePatternType.HEXAGONAL: 'MESH_CIRCLE',
    SurfacePatternType.TRIANGULAR: 'MESH_CONE',
    SurfacePatternType.RANDOM: 'PARTICLE_DATA',
    SurfacePatternType.ADAPTIVE: 'MOD_ADAPTIVE_SUBDIVISION',
    
    MaterialType.TPU: 'MATERIAL',
    MaterialType.PETG: 'MATERIAL_DATA',
    
    QualityLevel.DRAFT: 'RENDER_RESULT',
    QualityLevel.NORMAL: 'RENDER_ANIMATION',
    QualityLevel.HIGH: 'RENDER_STILL',
    QualityLevel.ULTRA: 'SHADING_RENDERED',
}

# Colors (for debug visualization)
DEBUG_COLORS = {
    'sphere': (1.0, 0.0, 0.0, 1.0),      # Red
    'connection': (0.0, 1.0, 0.0, 1.0),   # Green
    'edge': (0.0, 0.0, 1.0, 1.0),         # Blue
    'warning': (1.0, 1.0, 0.0, 1.0),      # Yellow
    'error': (1.0, 0.0, 1.0, 1.0),        # Magenta
    'success': (0.0, 1.0, 1.0, 1.0),      # Cyan
}

# File Extensions
FILE_EXTENSIONS = {
    ExportFormat.STL: '.stl',
    ExportFormat.OBJ: '.obj',
    ExportFormat.PLY: '.ply',
    ExportFormat.FBX: '.fbx',
    ExportFormat.GLTF: '.gltf',
}

# Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    'max_vertices': 100000,
    'max_faces': 50000,
    'max_spheres': 5000,
    'max_connections': 10000,
    'cache_size_mb': 100,
    'generation_timeout': 60.0,  # seconds
}

# Export
__all__ = [
    # Constants
    'ADDON_VERSION',
    'BLENDER_MIN_VERSION',
    'DEFAULT_SPHERE_SIZE',
    'DEFAULT_CONNECTION_THICKNESS',
    'DEFAULT_PATTERN_DENSITY',
    'DEFAULT_SHELL_THICKNESS',
    'DEFAULT_EDGE_ANGLE',
    'DEFAULT_EDGE_WIDTH',
    'DEFAULT_BRUSH_SIZE',
    'MIN_SPHERE_SIZE',
    'MAX_SPHERE_SIZE',
    'MIN_CONNECTION_THICKNESS',
    'MAX_CONNECTION_THICKNESS',
    'MIN_PATTERN_DENSITY',
    'MAX_PATTERN_DENSITY',
    'MIN_SHELL_THICKNESS',
    'MAX_SHELL_THICKNESS',
    
    # Enums
    'PatternType',
    'SurfacePatternType',
    'EdgePatternType',
    'HybridPatternType',
    'MaterialType',
    'ExportFormat',
    'QualityLevel',
    'PrinterProfile',
    'AlgorithmType',
    'LoadCase',
    'StressType',
    'CollisionType',
    'ResolutionStrategy',
    'GapType',
    'FillStrategy',
    'ConnectionType',
    'TriangulationMethod',
    
    # Mappings
    'ICON_MAPPING',
    'DEBUG_COLORS',
    'FILE_EXTENSIONS',
    'PERFORMANCE_THRESHOLDS',
]
