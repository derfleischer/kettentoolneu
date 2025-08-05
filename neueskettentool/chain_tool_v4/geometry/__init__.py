"""
Geometry module initialization
Geometric processing and analysis
"""

# Import all geometry components
from .mesh_analyzer import MeshAnalyzer, MeshData
from .triangulation import TriangulationEngine, TriangulationResult
from .edge_detector import EdgeDetector, EdgeLoop
from .surface_sampler import SurfaceSampler, SamplePoint
from .dual_layer_optimizer import DualLayerOptimizer, LayerConfig

# Convenience imports
from .mesh_analyzer import analyze_mesh, get_mesh_statistics
from .triangulation import triangulate_points, create_delaunay_3d
from .edge_detector import detect_edges, find_boundary_loops
from .surface_sampler import sample_surface, distribute_points

__all__ = [
    # Classes
    'MeshAnalyzer',
    'MeshData',
    'TriangulationEngine',
    'TriangulationResult',
    'EdgeDetector',
    'EdgeLoop',
    'SurfaceSampler',
    'SamplePoint',
    'DualLayerOptimizer',
    'LayerConfig',
    
    # Functions
    'analyze_mesh',
    'get_mesh_statistics',
    'triangulate_points',
    'create_delaunay_3d',
    'detect_edges',
    'find_boundary_loops',
    'sample_surface',
    'distribute_points',
]
