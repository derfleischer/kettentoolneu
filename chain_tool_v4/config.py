"""
Global Configuration for Chain Tool V4
Central place for all configuration values
"""

import os
from pathlib import Path

# ============================================
# ADDON PATHS
# ============================================

ADDON_PATH = Path(__file__).parent
PRESETS_PATH = ADDON_PATH / "presets"
ICONS_PATH = ADDON_PATH / "icons"
TEMP_PATH = Path(os.path.expanduser("~")) / "ChainTool_Temp"

# Create temp directory if needed
TEMP_PATH.mkdir(exist_ok=True)

# ============================================
# DEBUG SETTINGS
# ============================================

DEBUG_ENABLED = False  # Global debug switch
DEBUG_LEVEL = "INFO"  # ERROR, WARNING, INFO, DEBUG, TRACE
DEBUG_CATEGORIES = {
    'GENERAL': True,
    'PATTERN': True,
    'GEOMETRY': True,
    'TRIANGULATION': True,
    'PAINT': True,
    'EXPORT': True,
    'PERFORMANCE': True,
    'CACHE': True,
}

# ============================================
# PERFORMANCE SETTINGS
# ============================================

# Caching
ENABLE_BVH_CACHE = True
ENABLE_PATTERN_CACHE = True
CACHE_SIZE_LIMIT_MB = 100
CACHE_TIMEOUT_SECONDS = 300  # 5 minutes

# Processing limits
MAX_VERTICES_REALTIME = 10000  # Switch to background processing above this
MAX_PATTERN_POINTS = 5000
CHUNK_SIZE = 100  # For batch processing

# ============================================
# PATTERN DEFAULTS
# ============================================

# Surface patterns
DEFAULT_CELL_SIZE = 12.0  # mm
DEFAULT_EDGE_THICKNESS = 2.5  # mm
DEFAULT_FILL_RATIO = 0.7  # 70% fill

# Connection settings
MAX_CONNECTION_LENGTH = 25.0  # mm
MIN_CONNECTION_ANGLE = 20.0  # degrees
CONNECTION_OVERLAP_THRESHOLD = 0.1  # mm

# Triangulation
DELAUNAY_MIN_ANGLE = 20.0  # degrees
DELAUNAY_MAX_EDGE_RATIO = 3.0
QUALITY_THRESHOLD = 0.6

# ============================================
# MATERIAL SETTINGS
# ============================================

# Material presets (thickness in mm)
MATERIAL_PRESETS = {
    'TPU_SHELL': {
        'thickness': 1.2,
        'color': (0.9, 0.9, 0.9, 1.0),
        'name': 'TPU_95A_Shell'
    },
    'PETG_PATTERN': {
        'thickness': 0.8,
        'color': (0.2, 0.4, 0.8, 1.0),
        'name': 'PETG_Pattern'
    },
    'PETG_CF_PATTERN': {
        'thickness': 0.6,
        'color': (0.1, 0.1, 0.1, 1.0),
        'name': 'PETG_CF_Pattern'
    }
}

# ============================================
# UI SETTINGS
# ============================================

# Preview settings
PREVIEW_RESOLUTION = 32
OVERLAY_LINE_WIDTH = 2
OVERLAY_POINT_SIZE = 6
PREVIEW_UPDATE_DELAY = 0.1  # seconds

# Icon settings
ICON_SIZE = 32
USE_CUSTOM_ICONS = True

# Panel defaults
PANEL_WIDTH_MIN = 280
SUBPANEL_INDENT = 10

# ============================================
# EXPORT SETTINGS
# ============================================

# File formats
SUPPORTED_EXPORT_FORMATS = ['.stl', '.obj', '.ply']
DEFAULT_EXPORT_FORMAT = '.stl'

# Export options
EXPORT_SCALE_FACTOR = 1.0  # 1.0 = Blender units, 10.0 = cm to mm
APPLY_MODIFIERS_ON_EXPORT = True
TRIANGULATE_ON_EXPORT = True

# ============================================
# PAINT MODE SETTINGS
# ============================================

# Brush settings
PAINT_BRUSH_MIN = 2.0
PAINT_BRUSH_MAX = 50.0
PAINT_BRUSH_DEFAULT = 8.0
PAINT_FALLOFF_TYPES = ['SMOOTH', 'SPHERE', 'ROOT', 'SHARP', 'LINEAR']

# Stroke settings
STROKE_SAMPLE_DISTANCE = 2.0  # mm
STROKE_SMOOTH_ITERATIONS = 2
STROKE_SIMPLIFY_THRESHOLD = 0.5

# ============================================
# PATTERN LIBRARY SETTINGS
# ============================================

# Library paths
USER_LIBRARY_PATH = Path(os.path.expanduser("~")) / "Documents" / "ChainTool_Patterns"
SYSTEM_LIBRARY_PATH = PRESETS_PATH / "pattern_library"

# Library organization
PATTERN_CATEGORIES = [
    'medical',
    'structural', 
    'aesthetic',
    'custom',
    'experimental'
]

# ============================================
# PHYSICS SETTINGS (Future)
# ============================================

# Stress analysis
ENABLE_STRESS_ANALYSIS = False  # Future feature
STRESS_SAFETY_FACTOR = 2.5

# ============================================
# HELPER FUNCTIONS
# ============================================

def get_temp_file(filename):
    """Get path for temporary file"""
    return TEMP_PATH / filename

def get_preset_path(category, filename):
    """Get path for preset file"""
    return PRESETS_PATH / category / filename

def ensure_paths():
    """Ensure all required paths exist"""
    paths = [
        TEMP_PATH,
        USER_LIBRARY_PATH,
    ]
    
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)

# Initialize paths on import
ensure_paths()
