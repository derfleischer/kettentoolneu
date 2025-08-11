# =========================================
# 6. operators/__init__.py - MODUL-REGISTRIERUNG
# =========================================

"""
Operators Module Registration
"""

from . import sphere_ops
from . import connector_ops  
from . import surface_ops
from . import utility_ops
from . import system_ops

modules = [
    sphere_ops,
    connector_ops,
    surface_ops, 
    utility_ops,
    system_ops,
]

def register():
    """Registriert alle Operator-Module"""
    for module in modules:
        module.register()

def unregister():
    """Deregistriert alle Operator-Module"""
    for module in reversed(modules):
        module.unregister()

print("✅ Operator-Restructure definiert!")
print("\n🎯 FUNKTIONSLOGIK-OPERATOREN:")
print("• sphere_ops.py - Kugel-Erstellung & Management")
print("• connector_ops.py - Verbinder-Erstellung & Management")
print("• surface_ops.py - Surface Snapping & Projektion")
print("• utility_ops.py - Clear, Cleanup (funktionsbezogen)")
print("\n🔧 SYSTEM-MANAGEMENT-OPERATOREN:")
print("• system_ops.py - Debug, Performance, Cache, Statistics")
print("\n🚀 VORTEILE:")
print("• Saubere Trennung der Verantwortlichkeiten")
print("• Bessere Wartbarkeit und Verständlichkeit")
print("• Einfachere Erweiterung (neue Funktionen → funktionslogik)")
print("• System-Features isoliert (Debug, Performance, etc.)")
