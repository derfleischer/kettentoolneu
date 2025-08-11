"""
UI Panels für Kettentool - ERWEITERT
Mit verbesserter Statistik-Anzeige und Auto-Snap Info
"""

import bpy
from bpy.types import Panel

# =========================================
# MAIN PANEL
# =========================================

class KETTE_PT_main_panel(Panel):
    """Haupt-Panel für Chain Construction"""
    bl_label = "Chain Construction Tool"
    bl_idname = "KETTE_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tools"
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.chain_construction_props
        
        # Header mit Version
        box = layout.box()
        row = box.row()
        row.label(text="Kettentool v3.6.0", icon='LINKED')
        
        # Target Object mit Status
        box = layout.box()
        box.label(text="Zielobjekt", icon='OBJECT_DATA')
        box.prop(props, "kette_target_obj", text="")
        
        if props.kette_target_obj:
            # Auto-Snap Info
            info_box = box.box()
            info_box.label(text="✅ Auto-Snap aktiviert", icon='SNAP_ON')
            info_box.label(text=f"→ {props.kette_target_obj.name}")
        else:
            # Warning Box
            warning_box = box.box()
            warning_box.alert = True
            warning_box.label(text="⚠️ Kein Auto-Snap", icon='ERROR')
            warning_box.label(text="Setze Zielobjekt für")
            warning_box.label(text="automatisches Surface-Snapping")

# =========================================
# BASIC TOOLS PANEL
# =========================================

class KETTE_PT_basic_tools(Panel):
    """Panel für grundlegende Werkzeuge"""
    bl_label = "Grundfunktionen"
    bl_idname = "KETTE_PT_basic_tools"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tools"
    bl_parent_id = "KETTE_PT_main_panel"
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.chain_construction_props
        
        # Kugel-Erstellung
        box = layout.box()
        box.label(text="Kugeln", icon='MESH_UVSPHERE')
        
        # Basic Settings
        col = box.column(align=True)
        col.prop(props, "kette_kugelradius")
        col.prop(props, "kette_aufloesung")
        
        col.separator()
        
        # Kugel-Buttons mit Types
        grid = col.grid_flow(columns=2, align=True)
        
        # Start Sphere (Standard)
        op = grid.operator("kette.place_sphere", text="Start", icon='RADIOBUT_ON')
        op.sphere_type = 'start'
        
        # Manual Sphere
        op = grid.operator("kette.place_sphere", text="Manual", icon='RADIOBUT_OFF')
        op.sphere_type = 'manual'
        
        # Surface Snapping
        if props.kette_target_obj:
            col.separator()
            col.operator("kette.snap_to_surface", text="Auf Oberfläche snappen", icon='SNAP_ON')
        
        # Verbinder
        box = layout.box()
        box.label(text="Verbinder", icon='LINKED')
        
        col = box.column(align=True)
        col.prop(props, "kette_use_curved_connectors", text="Gebogen")
        col.prop(props, "kette_connector_radius_factor")
        
        col.separator()
        
        # Verbinder-Buttons
        row = col.row(align=True)
        op_curved = row.operator("kette.connect_selected", text="Gebogen")
        op_curved.connector_type = True
        
        op_straight = row.operator("kette.connect_selected", text="Gerade")
        op_straight.connector_type = False
        
        col.operator("kette.refresh_connectors", text="Aktualisieren")

# =========================================
# SURFACE SETTINGS PANEL
# =========================================

class KETTE_PT_surface_settings(Panel):
    """Panel für Oberflächeneinstellungen"""
    bl_label = "Oberflächenprojektion"
    bl_idname = "KETTE_PT_surface_settings"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tools"
    bl_parent_id = "KETTE_PT_main_panel"
    
    @classmethod
    def poll(cls, context):
        props = context.scene.chain_construction_props
        return props.kette_target_obj is not None
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.chain_construction_props
        
        col = layout.column(align=True)
        col.prop(props, "kette_abstand")
        col.prop(props, "kette_global_offset")
        col.prop(props, "kette_aura_dichte")

# =========================================
# STATISTICS PANEL
# =========================================

class KETTE_PT_statistics(Panel):
    """Panel für Statistiken - ERWEITERT"""
    bl_label = "Statistiken"
    bl_idname = "KETTE_PT_statistics"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tools"
    bl_parent_id = "KETTE_PT_main_panel"
    
    def draw(self, context):
        layout = self.layout
        
        # Import hier um Zirkular-Import zu vermeiden
        try:
            from ..creation.spheres import get_sphere_statistics
            from ..creation.connectors import get_connector_count
            
            # Sphere Stats
            stats = get_sphere_statistics()
            
            # Spheres Box
            box = layout.box()
            box.label(text="Kugeln", icon='MESH_UVSPHERE')
            
            if stats['total'] > 0:
                col = box.column(align=True)
                
                # Type breakdown
                type_icons = {
                    'start': 'RADIOBUT_ON',
                    'painted': 'BRUSH_DATA', 
                    'manual': 'RADIOBUT_OFF',
                    'generated': 'AUTO'
                }
                
                type_names = {
                    'start': 'Start',
                    'painted': 'Paint',
                    'manual': 'Manual', 
                    'generated': 'Auto'
                }
                
                for sphere_type, count in stats.items():
                    if sphere_type != 'total' and count > 0:
                        row = col.row()
                        icon = type_icons.get(sphere_type, 'DOT')
                        name = type_names.get(sphere_type, sphere_type.title())
                        row.label(text=f"{name}:", icon=icon)
                        row.label(text=str(count))
                
                # Total
                col.separator()
                row = col.row()
                row.label(text="Total:", icon='MESH_UVSPHERE')
                row.label(text=str(stats['total']))
                
            else:
                box.label(text="Keine Kugeln")
            
            # Connectors Box
            connector_count = get_connector_count()
            
            box = layout.box()
            box.label(text="Verbinder", icon='LINKED')
            
            col = box.column(align=True)
            row = col.row()
            row.label(text="Anzahl:")
            row.label(text=str(connector_count))
            
            # Actions
            if stats['total'] > 0 or connector_count > 0:
                box = layout.box()
                col = box.column(align=True)
                col.operator("kette.sphere_statistics", text="Detaillierte Stats", icon='INFO')
                col.operator("kette.system_status", text="System Status", icon='SETTINGS')
                col.separator()
                col.operator("kette.clear_all", text="Alle löschen", icon='TRASH')
                
        except ImportError:
            # Fallback wenn Module nicht verfügbar
            box = layout.box()
            box.label(text="Statistiken nicht verfügbar", icon='ERROR')

# =========================================
# DEBUG PANEL
# =========================================

class KETTE_PT_debug(Panel):
    """Panel für Debug-Funktionen - ERWEITERT"""
    bl_label = "Debug & Performance"
    bl_idname = "KETTE_PT_debug"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tools"
    bl_parent_id = "KETTE_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.chain_construction_props
        
        # Debug Settings
        box = layout.box()
        box.label(text="Debug", icon='CONSOLE')
        
        col = box.column(align=True)
        col.prop(props, "debug_enabled")
        
        if props.debug_enabled:
            col.prop(props, "debug_level")
        
        col.separator()
        col.operator("kette.toggle_debug", text="Debug umschalten")
        
        # Performance
        box = layout.box()
        box.label(text="Performance", icon='TIME')
        
        col = box.column(align=True)
        col.operator("kette.performance_report", text="Performance Report")
        col.operator("kette.clear_caches", text="Caches leeren")
        
        # System Info
        from ..core import constants
        
        if constants.debug or constants.performance_monitor:
            box = layout.box()
            box.label(text="System Info", icon='INFO')
            
            col = box.column(align=True)
            
            # Debug Status
            if constants.debug:
                debug_status = "🟢 ON" if constants.debug.enabled else "🔴 OFF"
                col.label(text=f"Debug: {debug_status}")
                if constants.debug.enabled:
                    col.label(text=f"Level: {constants.debug.level}")
            
            # Performance Monitor
            if constants.performance_monitor:
                col.label(text="Monitor: 🟢 ON")
            
            # Cache Counts (simplified)
            cache_total = (len(constants._aura_cache) + 
                          len(constants._bvh_cache) + 
                          len(constants._existing_connectors) +
                          len(constants._sphere_registry))
            
            if cache_total > 0:
                col.label(text=f"Caches: {cache_total} Einträge")

# =========================================
# TEST PANEL
# =========================================

class KETTE_PT_test_panel(Panel):
    """Test-Panel für Entwicklung - ERWEITERT"""
    bl_label = "🧪 Test Zone"
    bl_idname = "KETTE_PT_test_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tools"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.chain_construction_props
        
        # Quick Test Workflow
        box = layout.box()
        box.label(text="Schnelltest", icon='EXPERIMENTAL')
        
        col = box.column(align=True)
        
        # Workflow Steps
        col.label(text="1. Zielobjekt setzen:")
        if not props.kette_target_obj:
            sub = col.column()
            sub.alert = True
            sub.label(text="⚠️ Kein Zielobjekt!")
        else:
            col.label(text=f"✅ {props.kette_target_obj.name}")
        
        col.separator()
        col.label(text="2. Kugel platzieren:")
        op = col.operator("kette.place_sphere", text="▶️ Start-Kugel erstellen")
        op.sphere_type = 'start'
        
        col.separator()
        col.label(text="3. Zweite Kugel + Verbinden:")
        row = col.row(align=True)
        op = row.operator("kette.place_sphere", text="Manual", icon='ADD')
        op.sphere_type = 'manual'
        row.operator("kette.connect_selected", text="Verbinden", icon='LINKED')
        
        col.separator()
        col.label(text="4. Status prüfen:")
        col.operator("kette.system_status", text="▶️ System Report")
        
        # Auto-Snap Test
        if props.kette_target_obj:
            box = layout.box()
            box.label(text="Auto-Snap Test", icon='SNAP_ON')
            
            col = box.column(align=True)
            col.label(text="✅ Zielobjekt gesetzt")
            col.label(text="Cursor bewegen → Kugel platzieren")
            col.label(text="→ Sollte automatisch snappen")
            
            op = col.operator("kette.place_sphere", text="Test Auto-Snap", icon='SNAP_ON')
            op.sphere_type = 'manual'
        
        # System Info Compact
        box = layout.box()
        box.label(text="System Info", icon='INFO')
        
        from ..core import constants
        
        col = box.column(align=True)
        
        # Debug Status
        debug_status = "✅ ON" if constants.debug and constants.debug.enabled else "❌ OFF"
        col.label(text=f"Debug: {debug_status}")
        
        # Monitor Status  
        monitor_status = "✅ ON" if constants.performance_monitor else "❌ OFF"
        col.label(text=f"Monitor: {monitor_status}")
        
        # Quick Stats
        try:
            from ..creation.spheres import get_sphere_statistics
            from ..creation.connectors import get_connector_count
            
            sphere_stats = get_sphere_statistics()
            connector_count = get_connector_count()
            
            if sphere_stats['total'] > 0 or connector_count > 0:
                col.separator()
                col.label(text=f"Objekte: {sphere_stats['total']}🔮 + {connector_count}🔗")
            
        except ImportError:
            pass
        
        # Cache Status
        cache_info = []
        if constants._aura_cache:
            cache_info.append(f"A:{len(constants._aura_cache)}")
        if constants._bvh_cache:
            cache_info.append(f"B:{len(constants._bvh_cache)}")
        if constants._existing_connectors:
            cache_info.append(f"C:{len(constants._existing_connectors)}")
        if constants._sphere_registry:
            cache_info.append(f"S:{len(constants._sphere_registry)}")
        
        if cache_info:
            col.label(text=f"Caches: {' '.join(cache_info)}")
        else:
            col.label(text="Caches: Leer")

# =========================================
# REGISTRATION
# =========================================

classes = [
    KETTE_PT_main_panel,
    KETTE_PT_basic_tools,
    KETTE_PT_surface_settings,
    KETTE_PT_statistics,
    KETTE_PT_debug,
    KETTE_PT_test_panel,
]

def register():
    """Registriert UI Panels"""
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    """Deregistriert UI Panels"""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
