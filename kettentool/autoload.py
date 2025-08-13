"""
Improved Autoload Module for Blender Addons
Handles automatic discovery and registration of modules
"""

import os
import sys
import importlib
import traceback
from pathlib import Path
import bpy

# Module configuration
MODULE_NAMES = [
    "core",
    "utils", 
    "creation",
    "operators",
    "ui.property_groups",
    "ui.panels",
]

class AutoloadManager:
    """Manages module loading and registration with recursion protection"""
    
    def __init__(self, addon_name):
        self.addon_name = addon_name
        self.loaded_modules = []
        self.registered_classes = []
        self._is_registering = False
        self._is_unregistering = False
        
    def get_module_path(self, module_name):
        """Get the full module path"""
        return f"{self.addon_name}.{module_name}"
    
    def import_module(self, module_name):
        """Safely import a module"""
        full_path = self.get_module_path(module_name)
        try:
            # Check if already imported
            if full_path in sys.modules:
                module = sys.modules[full_path]
                importlib.reload(module)
            else:
                module = importlib.import_module(full_path)
            
            self.loaded_modules.append(full_path)
            print(f"[AUTOLOAD] ✅ Loaded: {full_path}")
            return module
            
        except Exception as e:
            print(f"[AUTOLOAD] ❌ Failed to load {full_path}: {e}")
            return None
    
    def register_module(self, module):
        """Register all classes in a module"""
        if not module:
            return
            
        # Check for register function
        if hasattr(module, 'register'):
            try:
                module.register()
                print(f"[AUTOLOAD] ✅ Registered module: {module.__name__}")
            except Exception as e:
                print(f"[AUTOLOAD] ⚠️ Failed to register {module.__name__}: {e}")
        
        # Register individual classes
        for attr_name in dir(module):
            if attr_name.startswith('_'):
                continue
                
            attr = getattr(module, attr_name)
            
            # Check if it's a registerable class
            if isinstance(attr, type) and any([
                issubclass(attr, getattr(bpy.types, base, type))
                for base in ['Panel', 'Operator', 'PropertyGroup', 'Menu', 'Header', 'UIList']
            ]):
                try:
                    if attr.__name__ not in [c.__name__ for c in self.registered_classes]:
                        bpy.utils.register_class(attr)
                        self.registered_classes.append(attr)
                        print(f"[AUTOLOAD] ✅ Registered class: {attr.__name__}")
                except Exception as e:
                    if "already registered" not in str(e):
                        print(f"[AUTOLOAD] ⚠️ Failed to register {attr.__name__}: {e}")
    
    def unregister_module(self, module):
        """Unregister all classes in a module"""
        if not module:
            return
            
        # Unregister individual classes first
        for attr_name in dir(module):
            if attr_name.startswith('_'):
                continue
                
            attr = getattr(module, attr_name)
            
            # Check if it's a registered class
            if isinstance(attr, type) and attr in self.registered_classes:
                try:
                    bpy.utils.unregister_class(attr)
                    self.registered_classes.remove(attr)
                    print(f"[AUTOLOAD] ✅ Unregistered class: {attr.__name__}")
                except Exception as e:
                    if "not registered" not in str(e):
                        print(f"[AUTOLOAD] ⚠️ Failed to unregister {attr.__name__}: {e}")
        
        # Check for unregister function
        if hasattr(module, 'unregister'):
            try:
                module.unregister()
                print(f"[AUTOLOAD] ✅ Unregistered module: {module.__name__}")
            except Exception as e:
                print(f"[AUTOLOAD] ⚠️ Failed to unregister {module.__name__}: {e}")
    
    def register_all(self):
        """Register all modules with recursion protection"""
        if self._is_registering:
            print("[AUTOLOAD] ⚠️ Already registering, skipping to prevent recursion")
            return
            
        self._is_registering = True
        print("[AUTOLOAD] Starting registration...")
        
        try:
            # Import and register modules in order
            for module_name in MODULE_NAMES:
                module = self.import_module(module_name)
                if module:
                    self.register_module(module)
            
            print(f"[AUTOLOAD] ✅ Registered {len(self.registered_classes)} classes")
            
        except Exception as e:
            print(f"[AUTOLOAD] ❌ Registration failed: {e}")
            traceback.print_exc()
            
        finally:
            self._is_registering = False
    
    def unregister_all(self):
        """Unregister all modules with recursion protection"""
        if self._is_unregistering:
            print("[AUTOLOAD] ⚠️ Already unregistering, skipping to prevent recursion")
            return
            
        self._is_unregistering = True
        print("[AUTOLOAD] Starting unregistration...")
        
        try:
            # Unregister in reverse order
            for module_name in reversed(MODULE_NAMES):
                full_path = self.get_module_path(module_name)
                if full_path in sys.modules:
                    module = sys.modules[full_path]
                    self.unregister_module(module)
            
            # Clear loaded modules
            self.loaded_modules.clear()
            self.registered_classes.clear()
            
            print("[AUTOLOAD] ✅ Unregistration complete")
            
        except Exception as e:
            print(f"[AUTOLOAD] ❌ Unregistration failed: {e}")
            traceback.print_exc()
            
        finally:
            self._is_unregistering = False

# Global autoload manager instance
_manager = None

def init():
    """Initialize the autoload system"""
    global _manager
    if not _manager:
        # Get addon name from module
        addon_name = __name__.split('.')[0]
        _manager = AutoloadManager(addon_name)
        print(f"[AUTOLOAD] Initialized for addon: {addon_name}")

def register():
    """Register all modules"""
    global _manager
    if _manager:
        _manager.register_all()
    else:
        print("[AUTOLOAD] ❌ Manager not initialized")

def unregister():
    """Unregister all modules"""
    global _manager
    if _manager:
        _manager.unregister_all()
        _manager = None  # Clear the manager after unregistration
    else:
        print("[AUTOLOAD] ⚠️ Manager not initialized or already cleared")
