"""
Auto-Load System für automatische Modul-Registrierung
"""

import bpy
import importlib
from pathlib import Path

# =========================================
# MODULE DISCOVERY
# =========================================

modules = None

def init():
    """Initialisiert das Auto-Load System"""
    global modules
    modules = get_all_submodules(Path(__file__).parent)

def get_all_submodules(directory):
    """Findet alle Python-Module im Verzeichnis"""
    return list(iter_submodules(directory, directory.name))

def iter_submodules(path, package_name):
    """Iteriert über alle Submodule"""
    for name in sorted(iter_submodule_names(path)):
        yield importlib.import_module("." + name, package_name)

def iter_submodule_names(path, root=""):
    """Iteriert über alle Submodul-Namen"""
    for file in path.iterdir():
        if file.is_file():
            if file.name.endswith(".py") and file.name != "__init__.py":
                yield root + file.stem
        elif file.is_dir():
            if file.joinpath("__init__.py").exists():
                yield from iter_submodule_names(file, root + file.name + ".")

# =========================================
# REGISTRATION
# =========================================

def register():
    """Registriert alle Module"""
    for module in modules:
        if hasattr(module, "register"):
            try:
                module.register()
            except Exception as e:
                print(f"Error registering module {module.__name__}: {e}")

def unregister():
    """Deregistriert alle Module in umgekehrter Reihenfolge"""
    for module in reversed(modules):
        if hasattr(module, "unregister"):
            try:
                module.unregister()
            except Exception as e:
                print(f"Error unregistering module {module.__name__}: {e}")
