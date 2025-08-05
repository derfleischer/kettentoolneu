"""
Pattern Module for Chain Tool V4
Provides various pattern generation strategies
"""

# Base pattern class
from .base_pattern import BasePattern, PatternResult

# Surface patterns (large area coverage)
from .surface_patterns import (
    VoronoiPattern,
    HexagonalGrid,
    TriangularMesh,
    AdaptiveIslands
)

# Edge patterns (reinforcement)
from .edge_patterns import (
    RimReinforcement,
    ContourBanding,
    StressEdgeLoop
)

# Hybrid patterns (combination strategies)
from .hybrid_patterns import (
    CoreShellPattern,
    GradientDensity,
    ZonalReinforcement
)

# Paint-based patterns
from .paint_patterns import (
    PaintPattern,
    StrokeConnector,
    PaintDensityMap
)

# Pattern registry for easy access
PATTERN_REGISTRY = {
    # Surface patterns
    'voronoi': VoronoiPattern,
    'hexagonal': HexagonalGrid,
    'triangular': TriangularMesh,
    'islands': AdaptiveIslands,
    
    # Edge patterns
    'rim': RimReinforcement,
    'contour': ContourBanding,
    'stress_edge': StressEdgeLoop,
    
    # Hybrid patterns
    'core_shell': CoreShellPattern,
    'gradient': GradientDensity,
    'zonal': ZonalReinforcement,
    
    # Paint patterns
    'paint': PaintPattern,
    'stroke': StrokeConnector,
    'density_map': PaintDensityMap
}

# Pattern categories for UI
PATTERN_CATEGORIES = {
    'Surface': ['voronoi', 'hexagonal', 'triangular', 'islands'],
    'Edge': ['rim', 'contour', 'stress_edge'],
    'Hybrid': ['core_shell', 'gradient', 'zonal'],
    'Paint': ['paint', 'stroke', 'density_map']
}

def get_pattern_class(pattern_type: str) -> BasePattern:
    """Get pattern class by type name"""
    if pattern_type not in PATTERN_REGISTRY:
        raise ValueError(f"Unknown pattern type: {pattern_type}")
    return PATTERN_REGISTRY[pattern_type]

def get_available_patterns() -> dict:
    """Get all available patterns organized by category"""
    return PATTERN_CATEGORIES.copy()

def create_pattern(pattern_type: str, **kwargs) -> BasePattern:
    """Create pattern instance with parameters"""
    pattern_class = get_pattern_class(pattern_type)
    return pattern_class(**kwargs)

# Version info
__version__ = "4.0.0"
__all__ = [
    # Base
    'BasePattern',
    'PatternResult',
    
    # Surface
    'VoronoiPattern',
    'HexagonalGrid', 
    'TriangularMesh',
    'AdaptiveIslands',
    
    # Edge
    'RimReinforcement',
    'ContourBanding',
    'StressEdgeLoop',
    
    # Hybrid
    'CoreShellPattern',
    'GradientDensity',
    'ZonalReinforcement',
    
    # Paint
    'PaintPattern',
    'StrokeConnector',
    'PaintDensityMap',
    
    # Functions
    'get_pattern_class',
    'get_available_patterns',
    'create_pattern',
    
    # Data
    'PATTERN_REGISTRY',
    'PATTERN_CATEGORIES'
]
