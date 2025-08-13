"""
Debug and Performance Monitoring System
"""

import bpy
import time
from typing import Dict, List, Optional
from collections import defaultdict

# =========================================
# DEBUG LOGGER
# =========================================

class DebugLogger:
    """Debug logging system"""
    
    def __init__(self):
        self.enabled = False
        self.level = 'INFO'
        self.logs = []
        self.max_logs = 1000
        
        # Log levels - TRACE added as most detailed level
        self.levels = {
            'TRACE': -1,    # Most detailed logging level
            'DEBUG': 0,
            'INFO': 1,
            'WARNING': 2,
            'ERROR': 3
        }
    
    def log(self, level: str, category: str, message: str):
        """Add log entry"""
        if not self.enabled:
            return
            
        if self.levels.get(level, 0) < self.levels.get(self.level, 0):
            return
        
        timestamp = time.time()
        entry = {
            'time': timestamp,
            'level': level,
            'category': category,
            'message': message
        }
        
        self.logs.append(entry)
        
        # Limit log size
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs:]
        
        # Print to console
        print(f"[{level}] {category}: {message}")
    
    def trace(self, category: str, message: str):
        """Trace level log - most detailed logging"""
        self.log('TRACE', category, message)
    
    def debug(self, category: str, message: str):
        """Debug level log"""
        self.log('DEBUG', category, message)
    
    def info(self, category: str, message: str):
        """Info level log"""
        self.log('INFO', category, message)
    
    def warning(self, category: str, message: str):
        """Warning level log"""
        self.log('WARNING', category, message)
    
    def error(self, category: str, message: str):
        """Error level log"""
        self.log('ERROR', category, message)
    
    def clear(self):
        """Clear all logs"""
        self.logs.clear()
    
    def get_logs(self, level: Optional[str] = None, category: Optional[str] = None) -> List[Dict]:
        """Get filtered logs"""
        logs = self.logs
        
        if level:
            logs = [l for l in logs if l['level'] == level]
        
        if category:
            logs = [l for l in logs if l['category'] == category]
        
        return logs
    
    def set_level(self, level: str):
        """Set minimum log level"""
        if level in self.levels:
            self.level = level
            self.info('DEBUG', f"Log level set to {level}")
    
    def get_level_value(self, level: str) -> int:
        """Get numeric value for log level"""
        return self.levels.get(level, 0)

# =========================================
# PERFORMANCE MONITOR
# =========================================

class PerformanceMonitor:
    """Performance monitoring system"""
    
    def __init__(self):
        self.timers = {}
        self.counters = defaultdict(int)
        self.enabled = True
    
    def start_timer(self, name: str):
        """Start a named timer"""
        if not self.enabled:
            return
        self.timers[name] = time.time()
    
    def stop_timer(self, name: str) -> float:
        """Stop timer and return elapsed time"""
        if not self.enabled or name not in self.timers:
            return 0.0
        
        elapsed = time.time() - self.timers[name]
        del self.timers[name]
        return elapsed
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment a counter"""
        if not self.enabled:
            return
        self.counters[name] += value
    
    def get_counter(self, name: str) -> int:
        """Get counter value"""
        return self.counters.get(name, 0)
    
    def reset_counter(self, name: str):
        """Reset a counter"""
        if name in self.counters:
            del self.counters[name]
    
    def reset_all(self):
        """Reset all timers and counters"""
        self.timers.clear()
        self.counters.clear()
    
    def get_report(self) -> str:
        """Generate performance report"""
        lines = ["Performance Report:"]
        lines.append("-" * 40)
        
        # Active timers
        if self.timers:
            lines.append("Active Timers:")
            for name, start_time in self.timers.items():
                elapsed = time.time() - start_time
                lines.append(f"  {name}: {elapsed:.3f}s")
        
        # Counters
        if self.counters:
            lines.append("Counters:")
            for name, count in self.counters.items():
                lines.append(f"  {name}: {count}")
        
        return "\n".join(lines)

# =========================================
# DECORATORS
# =========================================

def timed(func):
    """Decorator to time function execution"""
    def wrapper(*args, **kwargs):
        from kettentool.core import constants
        if constants.performance_monitor and constants.performance_monitor.enabled:
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            if constants.debug:
                constants.debug.debug('PERFORMANCE', f"{func.__name__} took {elapsed:.3f}s")
            return result
        return func(*args, **kwargs)
    return wrapper

def logged(category: str = 'GENERAL'):
    """Decorator to log function calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            from kettentool.core import constants
            if constants.debug and constants.debug.enabled:
                constants.debug.debug(category, f"Calling {func.__name__}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

# =========================================
# REGISTRATION
# =========================================

def register():
    """Register debug module"""
    pass

def unregister():
    """Unregister debug module"""
    pass
