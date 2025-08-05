"""
Chain Tool V4 - Main UI Panel
Hauptinterface für Pattern-Auswahl und Basis-Controls

Integration mit:
- core/properties.py (ChainToolProperties)
- operators/pattern_generation.py
- Moderne Blender 4.x UI-Patterns
"""

import bpy
from bpy.types import Panel
from ...core.properties import ChainToolProperties
from ...core.state_manager import StateManager
from ...operators.pattern_generation import (
    CHAINTOOL_OT_generate_surface_pattern,
    CHAINTOOL_OT_generate_edge_pattern,
    CHAINTOOL_OT_generate_hybrid_pattern
)
from ...operators.paint_mode import CHAINTOOL_OT_paint_mode


class CHAINTOOL_PT_main_panel(Panel):
    """Haupt-Panel für Chain Tool V4"""
    bl_label = "Chain Tool V4"
    bl_idname = "CHAINTOOL_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tool"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw_header(self, context):
        """Header mit Icon und Status"""
        layout = self.layout
        row = layout.row(align=True)
        row.label(text="", icon='MESH_MONKEY')
        
        # Status Indicator
        state_manager = StateManager()
        if state_manager.is_generating:
            row.label(text="", icon='TIME')
        elif state_manager.has_active_pattern:
            row.label(text="", icon='CHECKMARK')
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.chain_tool_properties
        
        # Error handling
        if not props:
            box = layout.box()
            box.label(text="Chain Tool Properties nicht gefunden!", icon='ERROR')
            box.operator("chaintool.initialize_properties", text="Properties Initialisieren")
            return
        
        # Target Object Selection
        self._draw_target_selection(layout, props)
        
        # Pattern Type Selection
        self._draw_pattern_selection(layout, props)
        
        # Basic Controls
        self._draw_basic_controls(layout, props)
        
        # Generation Buttons
        self._draw_generation_buttons(layout, props)
        
        # Status Info
        self._draw_status_info(layout, context)
    
    def _draw_target_selection(self, layout, props):
        """Zielobjekt-Auswahl Sektion"""
        box = layout.box()
        box.label(text="Zielobjekt", icon='MESH_DATA')
        
        row = box.row(align=True)
        row.prop(props, "target_object", text="")
        row.operator("chaintool.select_active_object", text="", icon='EYEDROPPER')
        
        # Validation Status
        if props.target_object:
            if props.target_object.type != 'MESH':
                box.label(text="⚠ Nur Mesh-Objekte unterstützt", icon='ERROR')
            else:
                mesh_stats = self._get_mesh_stats(props.target_object)
                box.label(text=f"✓ {mesh_stats['faces']} Faces, {mesh_stats['verts']} Verts", icon='INFO')
        else:
            box.label(text="Kein Objekt ausgewählt", icon='ERROR')
    
    def _draw_pattern_selection(self, layout, props):
        """Pattern-Typ Auswahl"""
        box = layout.box()
        box.label(text="Pattern-Typ", icon='MODIFIER')
        
        # Pattern Type Dropdown
        row = box.row()
        row.prop(props, "pattern_type", text="")
        
        # Pattern-spezifische Info
        pattern_info = self._get_pattern_info(props.pattern_type)
        if pattern_info:
            info_box = box.box()
            info_box.scale_y = 0.8
            info_box.label(text=pattern_info['description'], icon='INFO')
            info_box.label(text=f"Empfohlen für: {pattern_info['use_case']}")
    
    def _draw_basic_controls(self, layout, props):
        """Basis-Steuerungen"""
        box = layout.box()
        box.label(text="Basis-Einstellungen", icon='SETTINGS')
        
        # Sphere Settings
        col = box.column(align=True)
        col.prop(props, "sphere_radius", text="Kugel-Radius")
        col.prop(props, "sphere_resolution", text="Auflösung")
        
        # Density Controls
        col.separator()
        col.prop(props, "aura_density", text="Aura-Dichte")
        col.prop(props, "surface_offset", text="Oberflächen-Offset")
        
        # Advanced Toggle
        if props.show_advanced_settings:
            self._draw_advanced_controls(box, props)
        
        row = box.row()
        row.prop(props, "show_advanced_settings", 
                text="Erweiterte Einstellungen", 
                icon='TRIA_DOWN' if props.show_advanced_settings else 'TRIA_RIGHT')
    
    def _draw_advanced_controls(self, layout, props):
        """Erweiterte Steuerungen"""
        box = layout.box()
        box.label(text="Erweiterte Optionen", icon='PREFERENCES')
        
        col = box.column(align=True)
        col.prop(props, "max_spheres", text="Max. Kugeln")
        col.prop(props, "overlap_threshold", text="Überlappungs-Schwelle")
        col.prop(props, "connection_angle", text="Verbindungswinkel")
        
        # Performance Settings
        col.separator()
        col.prop(props, "use_bvh_acceleration", text="BVH Beschleunigung")
        col.prop(props, "cache_results", text="Ergebnisse Zwischenspeichern")
    
    def _draw_generation_buttons(self, layout, props):
        """Pattern-Generierungs-Buttons"""
        box = layout.box()
        box.label(text="Pattern Generierung", icon='PLAY')
        
        # Validation Check
        can_generate = self._can_generate_pattern(props)
        
        if not can_generate['valid']:
            warning_box = box.box()
            warning_box.alert = True
            warning_box.label(text=can_generate['reason'], icon='ERROR')
        
        # Generation Buttons basierend auf Pattern-Typ
        col = box.column(align=True)
        col.enabled = can_generate['valid']
        
        if props.pattern_type == 'SURFACE':
            col.operator("chaintool.generate_surface_pattern", 
                        text="Oberflächen-Pattern Generieren", 
                        icon='MESH_UVSPHERE')
        
        elif props.pattern_type == 'EDGE':
            col.operator("chaintool.generate_edge_pattern", 
                        text="Kanten-Pattern Generieren", 
                        icon='EDGESEL')
        
        elif props.pattern_type == 'HYBRID':
            col.operator("chaintool.generate_hybrid_pattern", 
                        text="Hybrid-Pattern Generieren", 
                        icon='MOD_LATTICE')
        
        elif props.pattern_type == 'PAINT':
            # Paint Mode hat spezielles Interface
            paint_row = col.row(align=True)
            paint_row.operator("chaintool.paint_mode", 
                             text="Paint Modus", 
                             icon='BRUSH_DATA')
            paint_row.operator("chaintool.clear_paint_strokes", 
                             text="", 
                             icon='X')
        
        # Quick Actions
        col.separator()
        action_row = col.row(align=True)
        action_row.operator("chaintool.clear_pattern", 
                          text="Löschen", 
                          icon='TRASH')
        action_row.operator("chaintool.export_pattern", 
                          text="Exportieren", 
                          icon='EXPORT')
    
    def _draw_status_info(self, layout, context):
        """Status-Informationen"""
        state_manager = StateManager()
        
        if state_manager.last_generation_time:
            box = layout.box()
            box.scale_y = 0.7
            
            # Performance Info
            gen_time = state_manager.last_generation_time
            sphere_count = state_manager.last_sphere_count
            
            row = box.row()
            row.label(text=f"Generierungszeit: {gen_time:.2f}s", icon='TIME')
            
            row = box.row()
            row.label(text=f"Kugeln erstellt: {sphere_count}", icon='MESH_UVSPHERE')
            
            # Memory Usage
            if hasattr(state_manager, 'memory_usage'):
                row = box.row()
                row.label(text=f"Speicher: {state_manager.memory_usage:.1f}MB", icon='FILE_CACHE')
    
    # Helper Methods
    def _get_mesh_stats(self, obj):
        """Mesh-Statistiken abrufen"""
        if not obj or obj.type != 'MESH':
            return {'faces': 0, 'verts': 0}
        
        mesh = obj.data
        return {
            'faces': len(mesh.polygons),
            'verts': len(mesh.vertices)
        }
    
    def _get_pattern_info(self, pattern_type):
        """Pattern-Informationen abrufen"""
        pattern_info = {
            'SURFACE': {
                'description': 'Gleichmäßige Verteilung auf Oberflächen',
                'use_case': 'Orthesen, Strukturelle Verstärkung'
            },
            'EDGE': {
                'description': 'Verstärkung entlang Kanten und Konturen',
                'use_case': 'Randverstärkung, Belastungspunkte'
            },
            'HYBRID': {
                'description': 'Kombiniert Oberflächen- und Kanten-Pattern',
                'use_case': 'Komplexe Geometrien, Optimale Festigkeit'
            },
            'PAINT': {
                'description': 'Interaktives Malen von Pattern-Bereichen',
                'use_case': 'Präzise Kontrolle, Künstlerische Designs'
            }
        }
        return pattern_info.get(pattern_type)
    
    def _can_generate_pattern(self, props):
        """Überprüfung ob Pattern generiert werden kann"""
        if not props.target_object:
            return {'valid': False, 'reason': 'Kein Zielobjekt ausgewählt'}
        
        if props.target_object.type != 'MESH':
            return {'valid': False, 'reason': 'Zielobjekt muss Mesh sein'}
        
        if not props.target_object.data.polygons:
            return {'valid': False, 'reason': 'Zielobjekt hat keine Faces'}
        
        if props.sphere_radius <= 0:
            return {'valid': False, 'reason': 'Kugel-Radius muss größer als 0 sein'}
        
        # Pattern-spezifische Validierung
        if props.pattern_type == 'PAINT':
            state_manager = StateManager()
            if not state_manager.has_paint_strokes:
                return {'valid': False, 'reason': 'Keine Paint-Strokes vorhanden'}
        
        return {'valid': True, 'reason': ''}


class CHAINTOOL_PT_quick_actions(Panel):
    """Schnellzugriff-Panel für häufige Aktionen"""
    bl_label = "Schnellzugriff"
    bl_idname = "CHAINTOOL_PT_quick_actions"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tool"
    bl_parent_id = "CHAINTOOL_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        
        # Quick Generation Presets
        box = layout.box()
        box.label(text="Voreinstellungen", icon='PRESET')
        
        col = box.column(align=True)
        col.operator("chaintool.apply_preset", 
                    text="Orthese Standard").preset_name = "orthotic_standard"
        col.operator("chaintool.apply_preset", 
                    text="Dichte Verstärkung").preset_name = "dense_reinforcement"
        col.operator("chiantool.apply_preset", 
                    text="Leichtbau").preset_name = "lightweight"
        
        # Pattern Tools
        box = layout.box()
        box.label(text="Pattern-Tools", icon='TOOL_SETTINGS')
        
        col = box.column(align=True)
        col.operator("chaintool.mirror_pattern", text="Pattern Spiegeln", icon='MOD_MIRROR')
        col.operator("chaintool.scale_pattern", text="Pattern Skalieren", icon='FULLSCREEN_ENTER')
        col.operator("chiantool.optimize_connections", text="Verbindungen Optimieren", icon='STICKY_UVS_LOC')


# Registration Helper
classes = [
    CHAINTOOL_PT_main_panel,
    CHAINTOOL_PT_quick_actions,
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
