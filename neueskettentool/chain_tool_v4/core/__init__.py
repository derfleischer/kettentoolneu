"""
Core module initialization
"""

# Import all core components
from . import constants
from . import properties
from . import preferences
from . import state_manager

# Make key items available at package level
from .constants import PatternType, ConnectionType, OrthoticType
from .properties import ChainToolProperties
from .state_manager import StateManager, state

__all__ = [
    'constants',
    'properties', 
    'preferences',
    'state_manager',
    'PatternType',
    'ConnectionType',
    'OrthoticType',
    'ChainToolProperties',
    'StateManager',
    'state'
]
