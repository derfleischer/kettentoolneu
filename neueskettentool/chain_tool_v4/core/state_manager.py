"""
State Manager for Chain Tool V4
Global state management and tracking
"""

import bpy
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from enum import Enum, auto
import time

from utils.debug import debug

# ============================================
# STATE ENUMS
# ============================================

class ToolState(Enum):
    """Current tool state"""
    IDLE = auto()
    GENERATING_PATTERN = auto()
    PAINT_MODE = auto()
    CONNECTING = auto()
    EXPORTING = auto()
    ERROR = auto()

class GenerationPhase(Enum):
    """Pattern generation phases"""
    INITIALIZING = auto()
    ANALYZING_MESH = auto()
    GENERATING_POINTS = auto()
    CREATING_CONNECTIONS = auto()
    OPTIMIZING = auto()
    FINALIZING = auto()
    COMPLETE = auto()

# ============================================
# STATE DATA CLASSES
# ============================================

@dataclass
class PatternState:
    """State of current pattern generation"""
    pattern_type: str = ""
    point_count: int = 0
    connection_count: int = 0
    generation_time: float = 0.0
    is_valid: bool = True
    error_message: str = ""
    
    # Generation tracking
    points: List[int] = field(default_factory=list)  # Object indices
    connections: List[tuple] = field(default_factory=list)  # Connection pairs
    
    def reset(self):
        """Reset pattern state"""
        self.pattern_type = ""
        self.point_count = 0
        self.connection_count = 0
        self.generation_time = 0.0
        self.is_valid = True
        self.error_message = ""
        self.points.clear()
        self.connections.clear()

@dataclass
class PaintState:
    """State of paint mode"""
    is_active: bool = False
    stroke_points: List[tuple] = field(default_factory=list)
    current_stroke: List[tuple] = field(default_factory=list)
    painted_objects: Set[str] = field(default_factory=set)
    brush_size: float = 8.0
    
    def reset(self):
        """Reset paint state"""
        self.is_active = False
        self.stroke_points.clear()
        self.current_stroke.clear()
        self.painted_objects.clear()

@dataclass
class ExportState:
    """State of export operations"""
    is_exporting: bool = False
    export_path: str = ""
    exported_files: List[str] = field(default_factory=list)
    total_objects: int = 0
    processed_objects: int = 0
    
    @property
    def progress(self) -> float:
        """Get export progress (0-1)"""
        if self.total_objects == 0:
            return 0.0
        return self.processed_objects / self.total_objects

# ============================================
# STATE MANAGER
# ============================================

class StateManager:
    """Central state management for Chain Tool"""
    
    def __init__(self):
        # Current states
        self.tool_state = ToolState.IDLE
        self.generation_phase = GenerationPhase.INITIALIZING
        
        # State objects
        self.pattern = PatternState()
        self.paint = PaintState()
        self.export = ExportState()
        
        # History tracking
        self.operation_history = []
        self.undo_stack = []
        self.redo_stack = []
        
        # Temporary data
        self._temp_data = {}
        
        # Observers
        self._observers = {}
        
        debug.info('GENERAL', "State Manager initialized")
    
    # ============================================
    # STATE TRANSITIONS
    # ============================================
    
    def set_tool_state(self, new_state: ToolState):
        """Change tool state with validation"""
        old_state = self.tool_state
        
        # Validate transition
        if not self._is_valid_transition(old_state, new_state):
            debug.warning('GENERAL', 
                         f"Invalid state transition: {old_state.name} -> {new_state.name}")
            return False
        
        # Perform transition
        self.tool_state = new_state
        
        # Log transition
        self._log_operation(f"State: {old_state.name} -> {new_state.name}")
        
        # Notify observers
        self._notify_observers('tool_state', old_state, new_state)
        
        debug.info('GENERAL', f"Tool state changed to: {new_state.name}")
        return True
    
    def set_generation_phase(self, phase: GenerationPhase):
        """Update generation phase"""
        old_phase = self.generation_phase
        self.generation_phase = phase
        
        self._notify_observers('generation_phase', old_phase, phase)
        
        debug.debug('PATTERN', f"Generation phase: {phase.name}")
    
    # ============================================
    # PATTERN STATE MANAGEMENT
    # ============================================
    
    def start_pattern_generation(self, pattern_type: str):
        """Begin pattern generation"""
        self.set_tool_state(ToolState.GENERATING_PATTERN)
        self.set_generation_phase(GenerationPhase.INITIALIZING)
        
        self.pattern.reset()
        self.pattern.pattern_type = pattern_type
        self.pattern.generation_time = time.time()
        
        debug.info('PATTERN', f"Started pattern generation: {pattern_type}")
    
    def update_pattern_progress(self, points: int = None, connections: int = None):
        """Update pattern generation progress"""
        if points is not None:
            self.pattern.point_count = points
        if connections is not None:
            self.pattern.connection_count = connections
            
        self._notify_observers('pattern_progress', self.pattern)
    
    def complete_pattern_generation(self, success: bool = True, error: str = ""):
        """Complete pattern generation"""
        self.pattern.generation_time = time.time() - self.pattern.generation_time
        self.pattern.is_valid = success
        self.pattern.error_message = error
        
        self.set_generation_phase(GenerationPhase.COMPLETE)
        self.set_tool_state(ToolState.IDLE)
        
        status = "completed" if success else "failed"
        debug.info('PATTERN', 
                  f"Pattern generation {status} in {self.pattern.generation_time:.2f}s")
    
    # ============================================
    # PAINT STATE MANAGEMENT
    # ============================================
    
    def enter_paint_mode(self):
        """Enter paint mode"""
        if self.set_tool_state(ToolState.PAINT_MODE):
            self.paint.is_active = True
            debug.info('PAINT', "Entered paint mode")
            return True
        return False
    
    def exit_paint_mode(self):
        """Exit paint mode"""
        self.paint.is_active = False
        self.set_tool_state(ToolState.IDLE)
        debug.info('PAINT', "Exited paint mode")
    
    def add_paint_stroke(self, points: List[tuple]):
        """Add paint stroke"""
        self.paint.stroke_points.extend(points)
        self.paint.current_stroke = points.copy()
        
        self._notify_observers('paint_stroke', points)
    
    def add_painted_object(self, obj_name: str):
        """Track painted object"""
        self.paint.painted_objects.add(obj_name)
    
    # ============================================
    # EXPORT STATE MANAGEMENT
    # ============================================
    
    def start_export(self, path: str, object_count: int):
        """Begin export operation"""
        self.set_tool_state(ToolState.EXPORTING)
        
        self.export.is_exporting = True
        self.export.export_path = path
        self.export.total_objects = object_count
        self.export.processed_objects = 0
        self.export.exported_files.clear()
        
        debug.info('EXPORT', f"Started export to: {path}")
    
    def update_export_progress(self, processed: int = None):
        """Update export progress"""
        if processed is not None:
            self.export.processed_objects = processed
            
        self._notify_observers('export_progress', self.export.progress)
    
    def complete_export(self, files: List[str]):
        """Complete export operation"""
        self.export.is_exporting = False
        self.export.exported_files = files
        
        self.set_tool_state(ToolState.IDLE)
        
        debug.info('EXPORT', f"Export completed: {len(files)} files")
    
    # ============================================
    # TEMPORARY DATA MANAGEMENT
    # ============================================
    
    def set_temp_data(self, key: str, value: Any):
        """Store temporary data"""
        self._temp_data[key] = value
    
    def get_temp_data(self, key: str, default: Any = None) -> Any:
        """Retrieve temporary data"""
        return self._temp_data.get(key, default)
    
    def clear_temp_data(self, key: str = None):
        """Clear temporary data"""
        if key:
            self._temp_data.pop(key, None)
        else:
            self._temp_data.clear()
    
    # ============================================
    # HISTORY MANAGEMENT
    # ============================================
    
    def push_undo_state(self, operation: str, data: Dict[str, Any]):
        """Push state to undo stack"""
        self.undo_stack.append({
            'operation': operation,
            'timestamp': time.time(),
            'data': data.copy()
        })
        
        # Clear redo stack on new operation
        self.redo_stack.clear()
        
        # Limit undo stack size
        if len(self.undo_stack) > 50:
            self.undo_stack = self.undo_stack[-50:]
    
    def can_undo(self) -> bool:
        """Check if undo is available"""
        return len(self.undo_stack) > 0
    
    def can_redo(self) -> bool:
        """Check if redo is available"""
        return len(self.redo_stack) > 0
    
    # ============================================
    # OBSERVER PATTERN
    # ============================================
    
    def add_observer(self, event: str, callback: callable):
        """Add observer for state changes"""
        if event not in self._observers:
            self._observers[event] = []
        self._observers[event].append(callback)
    
    def remove_observer(self, event: str, callback: callable):
        """Remove observer"""
        if event in self._observers:
            self._observers[event].remove(callback)
    
    def _notify_observers(self, event: str, *args, **kwargs):
        """Notify all observers of an event"""
        if event in self._observers:
            for callback in self._observers[event]:
                try:
                    callback(*args, **kwargs)
                except Exception as e:
                    debug.error('GENERAL', 
                               f"Observer error: {e}", 
                               event=event)
    
    # ============================================
    # VALIDATION AND HELPERS
    # ============================================
    
    def _is_valid_transition(self, from_state: ToolState, to_state: ToolState) -> bool:
        """Validate state transition"""
        # Define valid transitions
        valid_transitions = {
            ToolState.IDLE: [
                ToolState.GENERATING_PATTERN,
                ToolState.PAINT_MODE,
                ToolState.CONNECTING,
                ToolState.EXPORTING
            ],
            ToolState.GENERATING_PATTERN: [
                ToolState.IDLE,
                ToolState.ERROR
            ],
            ToolState.PAINT_MODE: [
                ToolState.IDLE,
                ToolState.CONNECTING
            ],
            ToolState.CONNECTING: [
                ToolState.IDLE,
                ToolState.ERROR
            ],
            ToolState.EXPORTING: [
                ToolState.IDLE,
                ToolState.ERROR
            ],
            ToolState.ERROR: [
                ToolState.IDLE
            ]
        }
        
        return to_state in valid_transitions.get(from_state, [])
    
    def _log_operation(self, operation: str):
        """Log operation to history"""
        self.operation_history.append({
            'operation': operation,
            'timestamp': time.time(),
            'state': self.tool_state.name
        })
        
        # Limit history size
        if len(self.operation_history) > 100:
            self.operation_history = self.operation_history[-50:]
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of current state"""
        return {
            'tool_state': self.tool_state.name,
            'generation_phase': self.generation_phase.name,
            'pattern': {
                'type': self.pattern.pattern_type,
                'points': self.pattern.point_count,
                'connections': self.pattern.connection_count,
                'valid': self.pattern.is_valid
            },
            'paint': {
                'active': self.paint.is_active,
                'strokes': len(self.paint.stroke_points),
                'objects': len(self.paint.painted_objects)
            },
            'export': {
                'active': self.export.is_exporting,
                'progress': self.export.progress,
                'files': len(self.export.exported_files)
            }
        }
    
    def reset_all(self):
        """Reset all states"""
        self.tool_state = ToolState.IDLE
        self.generation_phase = GenerationPhase.INITIALIZING
        
        self.pattern.reset()
        self.paint.reset()
        self.export = ExportState()
        
        self._temp_data.clear()
        self.operation_history.clear()
        
        debug.info('GENERAL', "All states reset")

# ============================================
# GLOBAL INSTANCE
# ============================================

# Create global state manager
state = StateManager()

# ============================================
# UTILITY FUNCTIONS
# ============================================

def get_state() -> StateManager:
    """Get global state manager"""
    return state

def is_busy() -> bool:
    """Check if any operation is in progress"""
    return state.tool_state != ToolState.IDLE

def can_start_operation() -> bool:
    """Check if new operation can be started"""
    return state.tool_state == ToolState.IDLE
