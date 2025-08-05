"""
Chain Tool V4 - Geometry Module
===============================
Geometric processing and analysis
"""

# Import all geometry components with error handling
classes = []

try:
    from .mesh_analyzer import MeshAnalyzer, MeshData, analyze_mesh, get_mesh_statistics
    classes.extend([MeshAnalyzer])
except ImportError as e:
    print(f"Chain Tool V4 - Geometry: Could not import mesh_analyzer: {e}")
    # Fallback definitions
    MeshAnalyzer = None
    MeshData = None
    analyze_mesh = None
    get_mesh_statistics = None

try:
    from .triangulation import TriangulationEngine, TriangulationResult, triangulate_points, create_delaunay_3d
except ImportError:
    # These might not exist yet
    TriangulationEngine = None
    TriangulationResult = None
    triangulate_points = None
    create_delaunay_3d = None

try:
    from .edge_detector import EdgeDetector, EdgeLoop, detect_edges, find_boundary_loops
except ImportError:
    EdgeDetector = None
    EdgeLoop = None
    detect_edges = None
    find_boundary_loops = None

try:
    from .surface_sampler import SurfaceSampler, SamplePoint, sample_surface, distribute_points
except ImportError:
    SurfaceSampler = None
    SamplePoint = None
    sample_surface = None
    distribute_points = None

try:
    from .dual_layer_optimizer import DualLayerOptimizer, LayerConfig
except ImportError:
    DualLayerOptimizer = None
    LayerConfig = None

# Export what we have
__all__ = []

# Add to __all__ only if successfully imported
if MeshAnalyzer:
    __all__.extend(['MeshAnalyzer', 'MeshData', 'analyze_mesh', 'get_mesh_statistics'])

if TriangulationEngine:
    __all__.extend(['TriangulationEngine', 'TriangulationResult', 'triangulate_points', 'create_delaunay_3d'])

if EdgeDetector:
    __all__.extend(['EdgeDetector', 'EdgeLoop', 'detect_edges', 'find_boundary_loops'])

if SurfaceSampler:
    __all__.extend(['SurfaceSampler', 'SamplePoint', 'sample_surface', 'distribute_points'])

if DualLayerOptimizer:
    __all__.extend(['DualLayerOptimizer', 'LayerConfig'])

# Add classes list for registration
__all__.append('classes')

def register():
    """Register geometry module"""
    print(f"Chain Tool V4: Geometry module registered ({len(classes)} classes)")

def unregister():
    """Unregister geometry module"""
    pass
