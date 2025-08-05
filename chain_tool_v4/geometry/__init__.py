"""
Chain Tool V4 - Geometry Module
===============================
Geometric processing and analysis
"""

# Import all geometry components with error handling
classes = []

# Try to import mesh_analyzer
try:
    from .mesh_analyzer import MeshAnalyzer, MeshData, analyze_mesh, get_mesh_statistics
    classes.append(MeshAnalyzer)
    print("  ✓ mesh_analyzer imported")
except ImportError as e:
    print(f"  ✗ Could not import mesh_analyzer: {e}")
    MeshAnalyzer = None
    MeshData = None
    analyze_mesh = None
    get_mesh_statistics = None

# Try to import triangulation (if exists)
try:
    from .triangulation import TriangulationEngine, TriangulationResult, triangulate_points, create_delaunay_3d
    if TriangulationEngine:
        classes.append(TriangulationEngine)
    print("  ✓ triangulation imported")
except ImportError:
    # Module doesn't exist yet
    TriangulationEngine = None
    TriangulationResult = None
    triangulate_points = None
    create_delaunay_3d = None

# Try to import edge_detector (if exists)
try:
    from .edge_detector import EdgeDetector, EdgeLoop, detect_edges, find_boundary_loops
    if EdgeDetector:
        classes.append(EdgeDetector)
    print("  ✓ edge_detector imported")
except ImportError:
    EdgeDetector = None
    EdgeLoop = None
    detect_edges = None
    find_boundary_loops = None

# Try to import surface_sampler (if exists)
try:
    from .surface_sampler import SurfaceSampler, SamplePoint, sample_surface, distribute_points
    if SurfaceSampler:
        classes.append(SurfaceSampler)
    print("  ✓ surface_sampler imported")
except ImportError:
    SurfaceSampler = None
    SamplePoint = None
    sample_surface = None
    distribute_points = None

# Try to import dual_layer_optimizer (if exists)
try:
    from .dual_layer_optimizer import DualLayerOptimizer, LayerConfig
    if DualLayerOptimizer:
        classes.append(DualLayerOptimizer)
    print("  ✓ dual_layer_optimizer imported")
except ImportError:
    DualLayerOptimizer = None
    LayerConfig = None

# Export what we have
__all__ = ['classes']

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

def register():
    """Register geometry module"""
    print(f"Chain Tool V4: Geometry module registered ({len(classes)} classes)")

def unregister():
    """Unregister geometry module"""
    pass
