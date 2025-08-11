"""
Debug System und Performance Monitor
"""

import time
from functools import wraps
from typing import List, Dict, Any

# =========================================
# DEBUG LOGGER
# =========================================

class DebugLogger:
    """Zentrales Debug-System"""
    
    def __init__(self):
        self.enabled = True
        self.level = 'INFO'  # DEBUG, INFO, WARNING, ERROR
        self.categories = set()
        self.log_history: List[Dict] = []
        self.max_history = 1000
    
    def _should_log(self, level: str, category: str) -> bool:
        """Entscheidet ob geloggt werden soll"""
        if not self.enabled:
            return False
        
        level_priority = {'DEBUG': 0, 'INFO': 1, 'WARNING': 2, 'ERROR': 3}
        current_priority = level_priority.get(self.level, 1)
        message_priority = level_priority.get(level, 1)
        
        return message_priority >= current_priority
    
    def _log(self, level: str, category: str, message: str, **kwargs):
        """Interne Log-Funktion"""
        if not self._should_log(level, category):
            return
        
        self.categories.add(category)
        
        log_entry = {
            'timestamp': time.time(),
            'level': level,
            'category': category,
            'message': message,
            'extra': kwargs
        }
        
        self.log_history.append(log_entry)
        
        # Begrenze History-Größe
        if len(self.log_history) > self.max_history:
            self.log_history = self.log_history[-self.max_history//2:]
        
        # Konsolen-Ausgabe
        timestamp = time.strftime("%H:%M:%S")
        extra_str = f" {kwargs}" if kwargs else ""
        print(f"[{timestamp}] {level:8} {category:12} {message}{extra_str}")
    
    def debug(self, category: str, message: str, **kwargs):
        self._log('DEBUG', category, message, **kwargs)
    
    def info(self, category: str, message: str, **kwargs):
        self._log('INFO', category, message, **kwargs)
    
    def warning(self, category: str, message: str, **kwargs):
        self._log('WARNING', category, message, **kwargs)
    
    def error(self, category: str, message: str, **kwargs):
        self._log('ERROR', category, message, **kwargs)
    
    def trace(self, category: str, message: str, **kwargs):
        self._log('DEBUG', category, f"TRACE: {message}", **kwargs)

# =========================================
# PERFORMANCE MONITOR
# =========================================

class PerformanceMonitor:
    """Performance-Überwachung"""
    
    def __init__(self):
        self.timers: Dict[str, float] = {}
        self.counters: Dict[str, int] = {}
        self.function_times: Dict[str, List[float]] = {}
    
    def start_timer(self, name: str):
        """Startet Timer"""
        self.timers[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """Beendet Timer und gibt Zeit zurück"""
        if name not in self.timers:
            return 0.0
        
        elapsed = time.time() - self.timers[name]
        del self.timers[name]
        
        # Sammle Statistiken
        if name not in self.function_times:
            self.function_times[name] = []
        self.function_times[name].append(elapsed)
        
        return elapsed
    
    def time_function(self, name: str, category: str = "PERFORMANCE"):
        """Decorator für Funktions-Timing"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                self.start_timer(name)
                try:
                    result = func(*args, **kwargs)
                    elapsed = self.end_timer(name)
                    
                    # Import hier um zirkuläre Imports zu vermeiden
                    from . import constants
                    if constants.debug and elapsed > 0.1:  # Nur bei > 100ms loggen
                        constants.debug.info(category, f"{name} took {elapsed:.3f}s")
                    
                    return result
                except Exception as e:
                    self.end_timer(name)
                    from . import constants
                    if constants.debug:
                        constants.debug.error(category, f"{name} failed: {str(e)}")
                    raise
            return wrapper
        return decorator
    
    def increment_counter(self, name: str, amount: int = 1):
        """Erhöht Counter"""
        self.counters[name] = self.counters.get(name, 0) + amount
    
    def report(self) -> str:
        """Erstellt Performance-Report"""
        lines = ["PERFORMANCE REPORT"]
        lines.append("=" * 50)
        
        if self.function_times:
            lines.append("\nFunction Times (avg/max/count):")
            for name, times in self.function_times.items():
                avg = sum(times) / len(times)
                max_time = max(times)
                count = len(times)
                lines.append(f"  {name:30} {avg:.3f}s / {max_time:.3f}s / {count}")
        
        if self.counters:
            lines.append("\nCounters:")
            for name, count in self.counters.items():
                lines.append(f"  {name:30} {count}")
        
        return "\n".join(lines)

# =========================================
# REGISTRATION
# =========================================

def register():
    """Registriert Debug-Modul"""
    pass

def unregister():
    """Deregistriert Debug-Modul"""
    pass
