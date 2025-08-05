"""
Chain Tool V4 - Paint Mode Control Panel
KRITISCHES Interface für Paint-Mode Integration

Features:
- Real-time Brush Controls
- Symmetrie-Tools
- Auto-Connect Visualisierung  
- Surface Snapping
- Stroke Buffer Management
"""

import bpy
from bpy.types import Panel
from ...core.properties import ChainToolProperties
from ...core.state_manager import StateManager
from ...operators.paint_mode import (
    CHAINTOOL_OT_paint_mode,
    CHAINTOOL_OT_paint_stroke,
    CHAINTOOL_OT_paint_clear,
    CHAINTOOL_OT_paint_connect_strokes
)
from ...utils.performance import PerformanceMonitor


class CHAINTOOL_PT_paint_panel(Panel):
    """Paint-Mode Haupt-Panel"""
    bl_label = "Paint Mode"
    bl_idname = "CHAINTOOL_PT_paint_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tool"
    bl_parent_id = "CHAINTOOL_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        """Nur anzeigen wenn Pattern-Type PAINT ist"""
        props = getattr(context.scene, 'chain_tool_properties', None)
        return props and props.pattern_type == 'PAINT'
    
    def draw_header(self, context):
        """Header mit Paint-Mode Status"""
        layout = self.layout
        state_manager = StateManager()
        
        if state_manager.is_paint_mode_active:
            layout.label(text="", icon='BRUSH_DATA')
            layout.label(text="AKTIV", icon='REC')
        else:
            layout.label(text="", icon='BRUSH_DATA')
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.chain_tool_properties
        state_manager = StateManager()
        
        # Paint Mode Toggle
        self._draw_paint_mode_toggle(layout, props, state_manager)
        
        # Brush Settings (nur wenn Paint Mode aktiv)
        if state_manager.is_paint_mode_active:
            self._draw_brush_settings(layout, props)
            self._draw_symmetry_controls(layout, props)
            self._draw_surface_snapping(layout, props)
            self._draw_stroke_buffer_controls(layout, props, state_manager)
            self._draw_auto_connect_settings(layout, props)
        
        # Paint Statistics
        self._draw_paint_statistics(layout, state_manager)
    
    def _draw_paint_mode_toggle(self, layout, props, state_manager):
        """Paint-Mode Ein/Ausschalten"""
        box = layout.box()
        
        # Hauptschalter
        row = box.row(align=True)
        row.scale_y = 1.5
        
        if state_manager.is_paint_mode_active:
            row.operator("chaintool.paint_mode", 
                        text="Paint Mode Beenden", 
                        icon='PAUSE',
                        depress=True)
        else:
            row.operator("chaintool.paint_mode", 
                        text="Paint Mode Starten", 
                        icon='PLAY')
        
        # Quick Actions
        if state_manager.is_paint_mode_active:
            action_row = box.row(align=True)
            action_row.operator("chaintool.paint_clear", 
                              text="Löschen", 
                              icon='TRASH')
            action_row.operator("chaintool.paint_connect_strokes", 
                              text="Verbinden", 
                              icon='STICKY_UVS_LOC')
        
        # Status Info
        if state_manager.is_paint_mode_active:
            status_box = box.box()
            status_box.scale_y = 0.8
            status_box.label(text="Linksklick: Malen | Rechtsklick: Beenden", icon='INFO')
    
    def _draw_brush_settings(self, layout, props):
        """Pinsel-Einstellungen"""
        box = layout.box()
        box.label(text="Pinsel-Einstellungen", icon='BRUSH_DATA')
        
        # Brush Size mit Visualisierung
        col = box.column(align=True)
        
        # Brush Size Slider mit Preview
        size_row = col.row(align=True)
        size_row.prop(props, "paint_brush_size", text="Größe")
        size_row.operator("chaintool.paint_brush_preview", 
                         text="", 
                         icon='RESTRICT_VIEW_OFF')
        
        # Brush Spacing
        col.prop(props, "paint_brush_spacing", text="Abstand")
        
        # Brush Pressure (wenn Tablet vorhanden)
        if self._has_pressure_input(bpy.context):
            col.separator()
            col.prop(props, "paint_use_pressure", text="Druck-Sensitivität")
            if props.paint_use_pressure:
                subcol = col.column(align=True)
                subcol.enabled = props.paint_use_pressure
                subcol.prop(props, "paint_pressure_size", text="Druck → Größe")
                subcol.prop(props, "paint_pressure_spacing", text="Druck → Abstand")
        
        # Advanced Brush Settings
        if props.show_advanced_paint_settings:
            self._draw_advanced_brush_settings(box, props)
        
        toggle_row = box.row()
        toggle_row.prop(props, "show_advanced_paint_settings", 
                       text="Erweiterte Pinsel-Optionen",
                       icon='TRIA_DOWN' if props.show_advanced_paint_settings else 'TRIA_RIGHT')
    
    def _draw_advanced_brush_settings(self, layout, props):
        """Erweiterte Pinsel-Einstellungen"""
        adv_box = layout.box()
        adv_box.label(text="Erweiterte Optionen", icon='SETTINGS')
        
        col = adv_box.column(align=True)
        col.prop(props, "paint_brush_randomize", text="Zufälligkeit")
        col.prop(props, "paint_brush_jitter", text="Jitter")
        col.prop(props, "paint_adaptive_spacing", text="Adaptive Abstände")
        
        # Surface Following
        col.separator()
        col.prop(props, "paint_surface_follow_strength", text="Oberflächen-Verfolgung")
        col.prop(props, "paint_normal_influence", text="Normalen-Einfluss")
    
    def _draw_symmetry_controls(self, layout, props):
        """Symmetrie-Steuerung"""
        box = layout.box()
        box.label(text="Symmetrie", icon='MOD_MIRROR')
        
        # Symmetrie Ein/Aus
        row = box.row(align=True)
        row.prop(props, "paint_use_symmetry", text="Symmetrie")
        if props.paint_use_symmetry:
            row.prop(props, "paint_symmetry_axis", text="")
        
        if props.paint_use_symmetry:
            sym_box = box.box()
            
            # Symmetrie-Einstellungen
            col = sym_box.column(align=True)
            col.prop(props, "paint_symmetry_offset", text="Offset")
            col.prop(props, "paint_symmetry_tolerance", text="Toleranz")
            
            # Multi-Axis Symmetrie
            if props.show_multi_axis_symmetry:
                col.separator()
                col.prop(props, "paint_symmetry_x", text="X-Achse")
                col.prop(props, "paint_symmetry_y", text="Y-Achse") 
                col.prop(props, "paint_symmetry_z", text="Z-Achse")
            
            sym_row = sym_box.row()
            sym_row.prop(props, "show_multi_axis_symmetry",
                        text="Multi-Achsen Symmetrie",
                        icon='TRIA_DOWN' if props.show_multi_axis_symmetry else 'TRIA_RIGHT')
    
    def _draw_surface_snapping(self, layout, props):
        """Oberflächen-Snapping Einstellungen"""
        box = layout.box()
        box.label(text="Oberflächen-Snapping", icon='SNAP_ON')
        
        # Snapping Modi
        col = box.column(align=True)
        col.prop(props, "paint_snap_mode", text="Modus")
        
        if props.paint_snap_mode != 'NONE':
            snap_box = box.box()
            
            # Snapping-Einstellungen
            snap_col = snap_box.column(align=True)
            snap_col.prop(props, "paint_snap_distance", text="Snap-Distanz")
            snap_col.prop(props, "paint_use_normal_offset", text="Normalen-Offset")
            
            if props.paint_use_normal_offset:
                snap_col.prop(props, "paint_normal_offset_distance", text="Offset-Distanz")
            
            # Performance Settings
            snap_col.separator()
            snap_col.prop(props, "paint_snap_use_bvh", text="BVH Beschleunigung")
            snap_col.prop(props, "paint_snap_max_faces", text="Max. Faces für BVH")
    
    def _draw_stroke_buffer_controls(self, layout, props, state_manager):
        """Stroke Buffer Management"""
        box = layout.box()
        box.label(text="Stroke Buffer", icon='GREASEPENCIL')
        
        # Buffer Status
        stroke_count = state_manager.get_stroke_count()
        point_count = state_manager.get_point_count()
        
        status_row = box.row(align=True)
        status_row.label(text=f"Strokes: {stroke_count}")
        status_row.label(text=f"Punkte: {point_count}")
        
        # Buffer Settings
        col = box.column(align=True)
        col.prop(props, "paint_buffer_size", text="Buffer-Größe")
        col.prop(props, "paint_smooth_strokes", text="Stroke-Glättung")
        
        if props.paint_smooth_strokes:
            smooth_col = col.column(align=True)
            smooth_col.enabled = props.paint_smooth_strokes
            smooth_col.prop(props, "paint_smooth_iterations", text="Glättungs-Iterationen")
            smooth_col.prop(props, "paint_smooth_factor", text="Glättungs-Stärke")
        
        # Memory Management
        if stroke_count > 0:
            mem_box = box.box()
            mem_box.scale_y = 0.8
            
            memory_usage = state_manager.get_stroke_memory_usage()
            mem_box.label(text=f"Speicher: {memory_usage:.1f}MB", icon='FILE_CACHE')
            
            if memory_usage > 50.0:  # Warning bei >50MB
                mem_box.label(text="⚠ Hoher Speicherverbrauch", icon='ERROR')
                mem_box.operator("chaintool.paint_optimize_buffer", 
                               text="Buffer Optimieren",
                               icon='FILE_REFRESH')
    
    def _draw_auto_connect_settings(self, layout, props):
        """Auto-Connect Einstellungen"""
        box = layout.box()
        box.label(text="Auto-Verbindung", icon='STICKY_UVS_LOC')
        
        # Auto-Connect Toggle
        row = box.row(align=True)
        row.prop(props, "paint_auto_connect", text="Auto-Verbinden")
        if props.paint_auto_connect:
            row.prop(props, "paint_auto_connect_realtime", text="", icon='TIME')
        
        if props.paint_auto_connect:
            connect_box = box.box()
            
            # Connection Settings
            col = connect_box.column(align=True)
            col.prop(props, "paint_connect_distance", text="Verbindungs-Distanz")
            col.prop(props, "paint_connect_angle", text="Max. Winkel")
            col.prop(props, "paint_avoid_intersections", text="Kollisionen Vermeiden")
            
            # Advanced Connection
            if props.show_advanced_connect_settings:
                adv_col = col.column(align=True)
                adv_col.separator()
                adv_col.prop(props, "paint_connect_algorithm", text="Algorithmus")
                adv_col.prop(props, "paint_connect_optimization", text="Optimierung")
                adv_col.prop(props, "paint_connect_max_connections", text="Max. Verbindungen")
            
            toggle_row = connect_box.row()
            toggle_row.prop(props, "show_advanced_connect_settings",
                           text="Erweiterte Verbindungs-Optionen", 
                           icon='TRIA_DOWN' if props.show_advanced_connect_settings else 'TRIA_RIGHT')
    
    def _draw_paint_statistics(self, layout, state_manager):
        """Paint-Statistiken und Performance"""
        if not state_manager.has_paint_data:
            return
        
        box = layout.box()
        box.label(text="Statistiken", icon='SORTTIME')
        box.scale_y = 0.8
        
        # Performance Stats
        perf_monitor = PerformanceMonitor()
        stats = perf_monitor.get_paint_stats()
        
        if stats:
            col = box.column(align=True)
            col.label(text=f"Durchschn. FPS: {stats['avg_fps']:.1f}")
            col.label(text=f"Stroke-Rate: {stats['stroke_rate']:.1f}/s")
            col.label(text=f"GPU-Auslastung: {stats['gpu_usage']:.1f}%")
            
            # Warnings
            if stats['avg_fps'] < 30:
                warning_box = box.box()
                warning_box.alert = True
                warning_box.label(text="⚠ Niedrige FPS!", icon='ERROR')
                warning_box.operator("chaintool.optimize_paint_performance", 
                                   text="Performance Optimieren")
    
    # Helper Methods
    def _has_pressure_input(self, context):
        """Prüft ob Tablet/Druck-Input verfügbar"""
        # Vereinfachte Tablet-Erkennung
        # In echter Implementierung würde hier Tablet-Hardware geprüft
        return hasattr(context.window_manager, 'tablet_api')
    
    def _get_stroke_preview_icon(self, stroke_count):
        """Icon basierend auf Stroke-Anzahl"""
        if stroke_count == 0:
            return 'DOT'
        elif stroke_count < 10:
            return 'CURVE_DATA'
        elif stroke_count < 50:
            return 'MESH_DATA'
        else:
            return 'MESH_MONKEY'


class CHAINTOOL_PT_paint_tools(Panel):
    """Zusätzliche Paint-Tools"""
    bl_label = "Paint Tools"
    bl_idname = "CHAINTOOL_PT_paint_tools"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tool"
    bl_parent_id = "CHAINTOOL_PT_paint_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        props = getattr(context.scene, 'chain_tool_properties', None)
        state_manager = StateManager()
        return (props and props.pattern_type == 'PAINT' and 
                state_manager.has_paint_data)
    
    def draw(self, context):
        layout = self.layout
        
        # Stroke Editing Tools
        box = layout.box()
        box.label(text="Stroke-Bearbeitung", icon='GREASEPENCIL')
        
        col = box.column(align=True)
        col.operator("chaintool.paint_select_stroke", text="Stroke Auswählen", icon='RESTRICT_SELECT_OFF')
        col.operator("chaintool.paint_delete_stroke", text="Stroke Löschen", icon='X')
        col.operator("chaintool.paint_duplicate_stroke", text="Stroke Duplizieren", icon='DUPLICATE')
        
        # Transform Tools
        col.separator()
        col.operator("chaintool.paint_move_stroke", text="Stroke Verschieben", icon='TRANSFORM_MOVE')
        col.operator("chaintool.paint_rotate_stroke", text="Stroke Rotieren", icon='TRANSFORM_ROTATE')
        col.operator("chaintool.paint_scale_stroke", text="Stroke Skalieren", icon='TRANSFORM_SCALE')
        
        # Pattern Tools
        pattern_box = layout.box()
        pattern_box.label(text="Pattern-Tools", icon='MOD_ARRAY')
        
        pattern_col = pattern_box.column(align=True)
        pattern_col.operator("chaintool.paint_stroke_to_array", text="Array-Modifier", icon='MOD_ARRAY')
        pattern_col.operator("chaintool.paint_stroke_to_curve", text="Zu Kurve", icon='CURVE_DATA')
        pattern_col.operator("chaintool.paint_stroke_mirror", text="Spiegeln", icon='MOD_MIRROR')


# Registration
classes = [
    CHAINTOOL_PT_paint_panel,
    CHAINTOOL_PT_paint_tools,
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
