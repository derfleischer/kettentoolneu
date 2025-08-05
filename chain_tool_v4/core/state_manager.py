"""
Chain Tool V4 - State Manager
=============================
Zentrale Zustandsverwaltung für Chain Tool
"""

import bpy
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
import json

@dataclass
class PatternState:
    """State for active pattern"""
    pattern_type: str = ""
    pattern_object: Optional[str] = None
    sphere_count: int = 0
    connection_count: int = 0
    is_generated: bool = False
    is_preview: bool = False
    generation_time: float = 0.0
    last_settings: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PaintState:
    """State for paint mode"""
    is_active: bool = False
    stroke_count: int = 0
    current_stroke: List[tuple] = field(default_factory=list)
    painted_spheres: Set[str] = field(default_factory=set)
    brush_settings: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExportState:
    """State for export operations"""
    is_ready: bool = False
    has_warnings: bool = False
    warnings: List[str] = field(default_factory=list)
    last_export_path: str = ""
    export_settings: Dict[str, Any] = field(default_factory=dict)

class StateManager:
    """
    Singleton state manager for Chain Tool V4
    Manages global addon state across all modules
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StateManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # Initialize states
        self.pattern_state = PatternState()
        self.paint_state = PaintState()
        self.export_state = ExportState()
        
        # Global flags
        self.is_generating = False
        self.is_paint_mode = False
        self.is_paint_mode_active = False
        self.has_active_pattern = False
        self.has_detected_edges = False
        self.has_paint_data = False
        self.is_print_ready = False
        self.has_export_warnings = False
        
        # Cache for temporary data
        self._cache: Dict[str, Any] = {}
        
        # Undo stack
        self._undo_stack: List[Dict[str, Any]] = []
        self._redo_stack: List[Dict[str, Any]] = []
        
        self._initialized = True
    
    def initialize(self) -> None:
        """Initialize or reset state manager"""
        self.reset()
    
    def reset(self) -> None:
        """Reset all states to default"""
        self.pattern_state = PatternState()
        self.paint_state = PaintState()
        self.export_state = ExportState()
        
        self.is_generating = False
        self.is_paint_mode = False
        self.is_paint_mode_active = False
        self.has_active_pattern = False
        self.has_detected_edges = False
        self.has_paint_data = False
        self.is_print_ready = False
        self.has_export_warnings = False
        
        self._cache.clear()
        self._undo_stack.clear()
        self._redo_stack.clear()
    
    # Pattern State Management
    def set_pattern_type(self, pattern_type: str) -> None:
        """Set the active pattern type"""
        self.pattern_state.pattern_type = pattern_type
        self.save_state()
    
    def set_pattern_generated(self, obj_name: str, sphere_count: int, 
                            connection_count: int, generation_time: float) -> None:
        """Mark pattern as generated"""
        self.pattern_state.pattern_object = obj_name
        self.pattern_state.sphere_count = sphere_count
        self.pattern_state.connection_count = connection_count
        self.pattern_state.generation_time = generation_time
        self.pattern_state.is_generated = True
        self.has_active_pattern = True
        self.save_state()
    
    def clear_pattern(self) -> None:
        """Clear pattern state"""
        self.pattern_state = PatternState()
        self.has_active_pattern = False
    
    # Paint State Management
    def enter_paint_mode(self) -> None:
        """Enter paint mode"""
        self.paint_state.is_active = True
        self.is_paint_mode = True
        self.is_paint_mode_active = True
        self.save_state()
    
    def exit_paint_mode(self) -> None:
        """Exit paint mode"""
        self.paint_state.is_active = False
        self.is_paint_mode = False
        self.is_paint_mode_active = False
        self.save_state()
    
    def add_paint_stroke(self, points: List[tuple]) -> None:
        """Add paint stroke"""
        self.paint_state.current_stroke = points
        self.paint_state.stroke_count += 1
        self.has_paint_data = True
    
    def add_painted_sphere(self, sphere_name: str) -> None:
        """Register painted sphere"""
        self.paint_state.painted_spheres.add(sphere_name)
    
    def clear_paint_data(self) -> None:
        """Clear all paint data"""
        self.paint_state = PaintState()
        self.has_paint_data = False
    
    # Export State Management
    def set_export_ready(self, is_ready: bool) -> None:
        """Set export ready state"""
        self.export_state.is_ready = is_ready
        self.is_print_ready = is_ready
    
    def add_export_warning(self, warning: str) -> None:
        """Add export warning"""
        self.export_state.warnings.append(warning)
        self.export_state.has_warnings = True
        self.has_export_warnings = True
    
    def clear_export_warnings(self) -> None:
        """Clear export warnings"""
        self.export_state.warnings.clear()
        self.export_state.has_warnings = False
        self.has_export_warnings = False
    
    def set_last_export_path(self, path: str) -> None:
        """Set last export path"""
        self.export_state.last_export_path = path
    
    # Cache Management
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        return self._cache.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache"""
        self._cache[key] = value
    
    def delete(self, key: str) -> None:
        """Delete value from cache"""
        if key in self._cache:
            del self._cache[key]
    
    def clear_cache(self) -> None:
        """Clear entire cache"""
        self._cache.clear()
    
    # Undo/Redo Management
    def save_state(self) -> None:
        """Save current state to undo stack"""
        state = {
            'pattern': self._serialize_dataclass(self.pattern_state),
            'paint': self._serialize_dataclass(self.paint_state),
            'export': self._serialize_dataclass(self.export_state),
            'flags': {
                'is_generating': self.is_generating,
                'is_paint_mode': self.is_paint_mode,
                'has_active_pattern': self.has_active_pattern,
                'has_detected_edges': self.has_detected_edges,
                'has_paint_data': self.has_paint_data,
                'is_print_ready': self.is_print_ready,
                'has_export_warnings': self.has_export_warnings,
            }
        }
        
        self._undo_stack.append(state)
        
        # Limit undo stack size
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)
        
        # Clear redo stack on new action
        self._redo_stack.clear()
    
    def undo(self) -> bool:
        """Undo last action"""
        if len(self._undo_stack) > 1:
            # Move current state to redo stack
            self._redo_stack.append(self._undo_stack.pop())
            
            # Restore previous state
            state = self._undo_stack[-1]
            self._restore_state(state)
            return True
        return False
    
    def redo(self) -> bool:
        """Redo previously undone action"""
        if self._redo_stack:
            state = self._redo_stack.pop()
            self._undo_stack.append(state)
            self._restore_state(state)
            return True
        return False
    
    def _serialize_dataclass(self, obj) -> Dict[str, Any]:
        """Serialize dataclass to dictionary"""
        result = {}
        for field_name in obj.__dataclass_fields__:
            value = getattr(obj, field_name)
            if isinstance(value, (str, int, float, bool, list, dict, set, type(None))):
                if isinstance(value, set):
                    value = list(value)
                result[field_name] = value
        return result
    
    def _restore_state(self, state: Dict[str, Any]) -> None:
        """Restore state from dictionary"""
        # Restore pattern state
        for key, value in state.get('pattern', {}).items():
            setattr(self.pattern_state, key, value)
        
        # Restore paint state
        for key, value in state.get('paint', {}).items():
            if key == 'painted_spheres' and isinstance(value, list):
                value = set(value)
            setattr(self.paint_state, key, value)
        
        # Restore export state
        for key, value in state.get('export', {}).items():
            setattr(self.export_state, key, value)
        
        # Restore flags
        flags = state.get('flags', {})
        for key, value in flags.items():
            setattr(self, key, value)
    
    # Persistence - OHNE context!
    def save_to_scene(self) -> None:
        """Save state to Blender scene for persistence"""
        # Nutze bpy.data statt context
        try:
            for scene in bpy.data.scenes:
                state_data = {
                    'pattern': self._serialize_dataclass(self.pattern_state),
                    'paint': self._serialize_dataclass(self.paint_state),
                    'export': self._serialize_dataclass(self.export_state),
                }
                
                # Store as JSON string in scene custom property
                scene['chain_tool_state'] = json.dumps(state_data)
                break  # Nur erste Scene
        except Exception as e:
            print(f"Chain Tool V4: Could not save state: {e}")
    
    def load_from_scene(self) -> None:
        """Load state from Blender scene"""
        # Nutze bpy.data statt context
        try:
            for scene in bpy.data.scenes:
                if 'chain_tool_state' in scene:
                    state_data = json.loads(scene['chain_tool_state'])
                    
                    # Restore states
                    for key, value in state_data.get('pattern', {}).items():
                        setattr(self.pattern_state, key, value)
                    
                    for key, value in state_data.get('paint', {}).items():
                        if key == 'painted_spheres' and isinstance(value, list):
                            value = set(value)
                        setattr(self.paint_state, key, value)
                    
                    for key, value in state_data.get('export', {}).items():
                        setattr(self.export_state, key, value)
                    
                    break  # Nur erste Scene
                    
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Chain Tool V4: Could not load state from scene: {e}")
    
    # Debug
    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of current state"""
        return {
            'pattern': {
                'type': self.pattern_state.pattern_type,
                'is_generated': self.pattern_state.is_generated,
                'sphere_count': self.pattern_state.sphere_count,
            },
            'paint': {
                'is_active': self.paint_state.is_active,
                'stroke_count': self.paint_state.stroke_count,
                'painted_spheres': len(self.paint_state.painted_spheres),
            },
            'export': {
                'is_ready': self.export_state.is_ready,
                'warning_count': len(self.export_state.warnings),
            },
            'cache_size': len(self._cache),
            'undo_stack_size': len(self._undo_stack),
            'redo_stack_size': len(self._redo_stack),
        }

# Export
__all__ = ['StateManager', 'PatternState', 'PaintState', 'ExportState']
