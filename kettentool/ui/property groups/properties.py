"""
Property Groups für UI
"""

import bpy
from bpy.props import (
    FloatProperty, IntProperty, BoolProperty, 
    EnumProperty, PointerProperty, StringProperty
)
from bpy.types import PropertyGroup

# =========================================
# MAIN PROPERTY GROUP
# =========================================

class ChainConstructionProperties(PropertyGroup):
    """Hauptgruppe für Chain Construction Properties"""
    
    # =========================================
    # BASIC SETTINGS
    # =========================================
    
    kette_target_obj: PointerProperty(
        name="Zielobjekt",
        type=bpy.types.Object,
        description="Objekt auf dem die Kette platziert wird",
        poll=lambda self, obj: obj.type == 'MESH'
    )
    
    kette_kugelradius: FloatProperty(
        name="Kugelradius",
        default=0.5,
        min=0.01,
        max=5.0,
        description="Radius der Kugeln"
    )
    
    kette_aufloesung: IntProperty(
        name="Auflösung",
        default=8,
        min=3,
        max=32,
        description="Anzahl der Segmente für Kugeln"
    )
    
    kette_abstand: FloatProperty(
        name="Oberflächenabstand",
        default=0.1,
        min=-2.0,
        max=2.0,
        description="Abstand der Kugeln zur Oberfläche"
    )
    
    kette_global_offset: FloatProperty(
        name="Globaler Offset",
        default=0.0,
        min=-5.0,
        max=5.0,
        description="Zusätzlicher Offset für alle Kugeln"
    )
    
    kette_aura_dichte: FloatProperty(
        name="Aura Dicke",
        default=0.05,
        min=0.0,
        max=1.0,
        description="Dicke der Aura für Oberflächenprojektion"
    )
    
    # =========================================
    # CONNECTOR SETTINGS
    # =========================================
    
    kette_use_curved_connectors: BoolProperty(
        name="Gebogene Verbinder",
        default=True,
        description="Verwende gebogene statt gerade Verbinder"
    )
    
    kette_connector_segs: IntProperty(
        name="Kurven-Segmente",
        default=12,
        min=3,
        max=32,
        description="Anzahl Segmente für gebogene Verbinder"
    )
    
    kette_connector_radius_factor: FloatProperty(
        name="Verbinder-Radius",
        default=0.8,
        min=0.1,
        max=2.0,
        subtype='FACTOR',
        description="Radius-Faktor für Verbinder"
    )
    
    kette_zyl_segs: IntProperty(
        name="Zylinder-Segmente",
        default=8,
        min=3,
        max=32,
        description="Anzahl Segmente für Zylinder-Verbinder"
    )
    
    # =========================================
    # CONNECTION MODE
    # =========================================
    
    kette_conn_mode: EnumProperty(
        name="Verbindungsmodus",
        items=[
            ('SEQ', 'Sequentiell', 'Verbinde in Reihenfolge'),
            ('NEIGHBOR', 'Nachbarn', 'Verbinde zu nächsten Nachbarn'),
            ('DELAUNAY', 'Delaunay', 'Optimierte Triangulation'),
        ],
        default='NEIGHBOR'
    )
    
    kette_max_abstand: FloatProperty(
        name="Max. Abstand",
        default=0,
        min=0,
        max=10,
        description="Maximaler Verbindungsabstand (0 = unbegrenzt)"
    )
    
    kette_max_links: IntProperty(
        name="Max. Verbindungen",
        default=4,
        min=1,
        max=10,
        description="Maximale Verbindungen pro Kugel"
    )
    
    # =========================================
    # PAINT MODE SETTINGS
    # =========================================
    
    paint_brush_spacing: FloatProperty(
        name="Pinsel-Abstand",
        default=0.5,
        min=0.1,
        max=2.0,
        subtype='FACTOR',
        description="Minimaler Abstand zwischen gemalten Kugeln (Faktor des Radius)"
    )
    
    paint_use_symmetry: BoolProperty(
        name="Symmetrie",
        default=False,
        description="Aktiviert symmetrisches Malen"
    )
    
    paint_symmetry_axis: EnumProperty(
        name="Symmetrie-Achse",
        items=[
            ('X', 'X-Achse', 'Spiegelung an X-Achse'),
            ('Y', 'Y-Achse', 'Spiegelung an Y-Achse'),
            ('Z', 'Z-Achse', 'Spiegelung an Z-Achse'),
        ],
        default='X'
    )
    
    paint_auto_connect: BoolProperty(
        name="Auto-Verbinden",
        default=True,
        description="Verbinde gemalte Kugeln automatisch"
    )
    
    # =========================================
    # DEBUG SETTINGS
    # =========================================
    
    debug_enabled: BoolProperty(
        name="Debug-Modus",
        default=False,
        description="Aktiviert Debug-Ausgabe"
    )
    
    debug_level: EnumProperty(
        name="Debug-Level",
        items=[
            ('DEBUG', 'Debug', 'Alle Nachrichten'),
            ('INFO', 'Info', 'Info und höher'),
            ('WARNING', 'Warning', 'Warnungen und Fehler'),
            ('ERROR', 'Error', 'Nur Fehler'),
        ],
        default='INFO'
    )

# =========================================
# STATISTICS PROPERTIES
# =========================================

class ChainStatistics(PropertyGroup):
    """Statistiken für Chain Construction"""
    
    sphere_count: IntProperty(
        name="Kugeln",
        default=0,
        description="Anzahl der Kugeln"
    )
    
    connector_count: IntProperty(
        name="Verbinder",
        default=0,
        description="Anzahl der Verbinder"
    )
    
    total_length: FloatProperty(
        name="Gesamtlänge",
        default=0.0,
        description="Gesamtlänge aller Verbinder"
    )

# =========================================
# UPDATE FUNCTIONS
# =========================================

def update_debug_settings(self, context):
    """Update-Funktion für Debug-Einstellungen"""
    from ..core import constants
    
    if constants.debug:
        constants.debug.enabled = self.debug_enabled
        constants.debug.level = self.debug_level
        
        if self.debug_enabled:
            constants.debug.info('UI', f"Debug mode enabled, level: {self.debug_level}")
        else:
            constants.debug.info('UI', "Debug mode disabled")

# Verbinde Update-Funktionen
ChainConstructionProperties.debug_enabled = BoolProperty(
    name="Debug-Modus",
    default=False,
    description="Aktiviert Debug-Ausgabe",
    update=update_debug_settings
)

ChainConstructionProperties.debug_level = EnumProperty(
    name="Debug-Level",
    items=[
        ('DEBUG', 'Debug', 'Alle Nachrichten'),
        ('INFO', 'Info', 'Info und höher'),
        ('WARNING', 'Warning', 'Warnungen und Fehler'),
        ('ERROR', 'Error', 'Nur Fehler'),
    ],
    default='INFO',
    update=update_debug_settings
)

# =========================================
# REGISTRATION
# =========================================

classes = [
    ChainConstructionProperties,
    ChainStatistics,
]

def register():
    """Registriert Properties"""
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    """Deregistriert Properties"""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
