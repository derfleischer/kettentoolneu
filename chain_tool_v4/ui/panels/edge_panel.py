"""
Chain Tool V4 - Edge Tools Panel
Erweiterte Kanten-Erkennung und Verstärkungs-Tools

Features:
- Edge Detection Parameter
- Rim Reinforcement Controls
- Connection Tools
- Stress-based Edge Enhancement
"""

import bpy
from bpy.types import Panel
from ...core.properties import ChainToolProperties
from ...core.state_manager import StateManager
from ...operators.edge_tools import (
    CHAINTOOL_OT_detect_edges,
    CHAINTOOL_OT_reinforce_rim,
    CHAINTOOL_OT_enhance_corners,
    CHAINTOOL_OT_create_edge_connections
)
from ...geometry.edge_detector import EdgeDetector
from ...utils.performance import PerformanceMonitor


class CHAINTOOL_PT_edge_panel(Panel):
    """Edge-Tools Haupt-Panel"""
    bl_label = "Edge Tools"
    bl_idname = "CHAINTOOL_PT_edge_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tool"
    bl_parent_id = "CHAINTOOL_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        """Nur anzeigen wenn Zielobjekt vorhanden"""
        props = getattr(context.scene, 'chain_tool_properties', None)
        return props and props.target_object and props.target_object.type == 'MESH'
    
    def draw_header(self, context):
        """Header mit Edge-Detection Status"""
        layout = self.layout  
        state_manager = StateManager()
        
        layout.label(text="", icon='EDGESEL')
        
        if state_manager.has_detected_edges:
            layout.label(text="", icon='CHECKMARK')
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.chain_tool_properties
        state_manager = StateManager()
        
        # Edge Detection
        self._draw_edge_detection(layout, props, state_manager)
        
        # Rim Reinforcement (nur wenn Edges erkannt)
        if state_manager.has_detected_edges:
            self._draw_rim_reinforcement(layout, props)
            self._draw_corner_enhancement(layout, props)
            self._draw_connection_tools(layout, props)
        
        # Edge Analysis Tools
        self._draw_edge_analysis(layout, props, state_manager)
    
    def _draw_edge_detection(self, layout, props, state_manager):
        """Edge Detection Einstellungen"""
        box = layout.box()
        box.label(text="Kanten-Erkennung", icon='SELECT_EXTEND')
        
        # Detection Method
        col = box.column(align=True)
        col.prop(props, "edge_detection_method", text="Methode")
        
        # Method-specific Settings
        if props.edge_detection_method == 'ANGLE_BASED':
            self._draw_angle_based_settings(box, props)
        elif props.edge_detection_method == 'CURVATURE_BASED':
            self._draw_curvature_based_settings(box, props) 
        elif props.edge_detection_method == 'HYBRID':
            self._draw_hybrid_detection_settings(box, props)
        
        # Detection Controls
        detect_row = box.row(align=True)
        detect_row.scale_y = 1.2
        
        if not state_manager.has_detected_edges:
            detect_row.operator("chaintool.detect_edges", 
                              text="Kanten Erkennen", 
                              icon='PLAY')
        else:
            detect_row.operator("chaintool.detect_edges", 
                              text="Neu Erkennen", 
                              icon='FILE_REFRESH')
            detect_row.operator("chaintool.clear_detected_edges", 
                              text="Löschen", 
                              icon='X')
        
        # Detection Results
        if state_manager.has_detected_edges:
            result_box = box.box()
            result_box.scale_y = 0.8
            
            edge_stats = state_manager.get_edge_statistics()
            if edge_stats:
                col = result_box.column(align=True)
                col.label(text=f"Erkannte Kanten: {edge_stats['edge_count']}")
                col.label(text=f"Scharfe Ecken: {edge_stats['corner_count']}")
                col.label(text=f"Rand-Länge: {edge_stats['perimeter_length']:.2f}m")
    
    def _draw_angle_based_settings(self, layout, props):
        """Winkel-basierte Edge Detection"""
        angle_box = layout.box()
        angle_box.label(text="Winkel-Parameter", icon='DRIVER_ROTATIONAL_DIFFERENCE')
        
        col = angle_box.column(align=True)
        col.prop(props, "edge_angle_threshold", text="Winkel-Schwelle")
        col.prop(props, "edge_smooth_threshold", text="Glättungs-Schwelle")
        col.prop(props, "edge_min_length", text="Min. Kanten-Länge")
        
        # Advanced Angle Settings
        if props.show_advanced_edge_detection:
            adv_col = col.column(align=True)
            adv_col.separator()
            adv_col.prop(props, "edge_angle_weighting", text="Winkel-Gewichtung")
            adv_col.prop(props, "edge_neighbor_influence", text="Nachbar-Einfluss")
    
    def _draw_curvature_based_settings(self, layout, props):
        """Krümmungs-basierte Edge Detection"""
        curve_box = layout.box()
        curve_box.label(text="Krümmungs-Parameter", icon='SURFACE_NCURVE')
        
        col = curve_box.column(align=True)
        col.prop(props, "edge_curvature_threshold", text="Krümmungs-Schwelle")
        col.prop(props, "edge_curvature_radius", text="Analyse-Radius")
        col.prop(props, "edge_gaussian_weighting", text="Gaussian-Gewichtung")
        
        # Curvature Analysis Type
        col.separator()
        col.prop(props, "edge_curvature_type", text="Krümmungs-Typ")
    
    def _draw_hybrid_detection_settings(self, layout, props):
        """Hybrid Edge Detection"""
        hybrid_box = layout.box()
        hybrid_box.label(text="Hybrid-Parameter", icon='MOD_LATTICE')
        
        col = hybrid_box.column(align=True)
        col.prop(props, "edge_angle_weight", text="Winkel-Gewicht")
        col.prop(props, "edge_curvature_weight", text="Krümmungs-Gewicht") 
        col.prop(props, "edge_geometry_weight", text="Geometrie-Gewicht")
        
        # Normalization
        col.separator()
        col.prop(props, "edge_normalize_weights", text="Gewichte Normalisieren")
    
    def _draw_rim_reinforcement(self, layout, props):
        """Rand-Verstärkung Einstellungen"""
        box = layout.box()
        box.label(text="Rand-Verstärkung", icon='MOD_SOLIDIFY')
        
        # Reinforcement Type
        col = box.column(align=True)
        col.prop(props, "rim_reinforcement_type", text="Verstärkungs-Typ")
        
        # Type-specific Settings
        if props.rim_reinforcement_type == 'DENSE_SPHERES':
            self._draw_dense_sphere_settings(box, props)
        elif props.rim_reinforcement_type == 'STRUCTURAL_BEAMS':
            self._draw_structural_beam_settings(box, props)
        elif props.rim_reinforcement_type == 'ADAPTIVE_THICKNESS':
            self._draw_adaptive_thickness_settings(box, props)
        
        # Generation Controls
        reinforce_row = box.row(align=True)
        reinforce_row.scale_y = 1.2
        reinforce_row.operator("chaintool.reinforce_rim", 
                             text="Rand Verstärken", 
                             icon='MOD_SOLIDIFY')
        reinforce_row.operator("chaintool.preview_rim_reinforcement", 
                             text="", 
                             icon='HIDE_OFF')
    
    def _draw_dense_sphere_settings(self, layout, props):
        """Dichte Kugel-Verstärkung"""
        sphere_box = layout.box()
        sphere_box.label(text="Kugel-Verstärkung", icon='MESH_UVSPHERE')
        
        col = sphere_box.column(align=True)
        col.prop(props, "rim_sphere_density", text="Kugel-Dichte")
        col.prop(props, "rim_sphere_size_factor", text="Größen-Faktor")
        col.prop(props, "rim_sphere_offset", text="Rand-Offset")
        
        # Advanced Sphere Settings
        col.separator()
        col.prop(props, "rim_sphere_vary_size", text="Größe Variieren")
        if props.rim_sphere_vary_size:
            col.prop(props, "rim_sphere_size_variation", text="Größen-Variation")
    
    def _draw_structural_beam_settings(self, layout, props):  
        """Strukturelle Träger-Verstärkung"""
        beam_box = layout.box()
        beam_box.label(text="Träger-Verstärkung", icon='MESH_CYLINDER')
        
        col = beam_box.column(align=True)
        col.prop(props, "rim_beam_width", text="Träger-Breite")
        col.prop(props, "rim_beam_height", text="Träger-Höhe")
        col.prop(props, "rim_beam_segments", text="Träger-Segmente")
        
        # Beam Pattern
        col.separator()
        col.prop(props, "rim_beam_pattern", text="Träger-Muster")
        col.prop(props, "rim_beam_spacing", text="Träger-Abstand")
    
    def _draw_adaptive_thickness_settings(self, layout, props):
        """Adaptive Dicken-Verstärkung"""
        thickness_box = layout.box()
        thickness_box.label(text="Adaptive Dicke", icon='MOD_THICKNESS')
        
        col = thickness_box.column(align=True)
        col.prop(props, "rim_thickness_min", text="Min. Dicke")
        col.prop(props, "rim_thickness_max", text="Max. Dicke")
        col.prop(props, "rim_thickness_curve", text="Dicken-Kurve")
        
        # Stress-based Adaptation
        col.separator()
        col.prop(props, "rim_use_stress_analysis", text="Stress-basiert")
        if props.rim_use_stress_analysis:
            col.prop(props, "rim_stress_factor", text="Stress-Faktor")
    
    def _draw_corner_enhancement(self, layout, props):
        """Ecken-Verstärkung"""
        box = layout.box()
        box.label(text="Ecken-Verstärkung", icon='MESH_ICOSPHERE')
        
        # Corner Detection
        col = box.column(align=True)
        col.prop(props, "corner_detection_angle", text="Ecken-Winkel")
        col.prop(props, "corner_min_significance", text="Min. Bedeutung")
        
        # Enhancement Settings
        col.separator()
        col.prop(props, "corner_enhancement_type", text="Verstärkungs-Typ")
        col.prop(props, "corner_enhancement_radius", text="Verstärkungs-Radius")
        col.prop(props, "corner_sphere_count", text="Zusätzliche Kugeln")
        
        # Controls
        corner_row = box.row(align=True)
        corner_row.operator("chaintool.enhance_corners", 
                          text="Ecken Verstärken", 
                          icon='MESH_ICOSPHERE')
        corner_row.operator("chaintool.detect_corners", 
                          text="Ecken Erkennen", 
                          icon='SELECT_EXTEND')
    
    def _draw_connection_tools(self, layout, props):
        """Verbindung-Tools für Edges"""
        box = layout.box()
        box.label(text="Kanten-Verbindungen", icon='STICKY_UVS_LOC')
        
        # Connection Algorithm
        col = box.column(align=True)
        col.prop(props, "edge_connection_algorithm", text="Verbindungs-Algorithmus")
        col.prop(props, "edge_connection_density", text="Verbindungs-Dichte")
        
        # Connection Quality
        col.separator()
        col.prop(props, "edge_max_connection_angle", text="Max. Verbindungswinkel")
        col.prop(props, "edge_min_connection_strength", text="Min. Verbindungs-Stärke")
        col.prop(props, "edge_avoid_crossings", text="Kreuzungen Vermeiden")
        
        # Advanced Connection Settings
        if props.show_advanced_edge_connections:
            adv_box = box.box()
            adv_box.label(text="Erweiterte Verbindungen", icon='SETTINGS')
            
            adv_col = adv_box.column(align=True)
            adv_col.prop(props, "edge_connection_optimization", text="Optimierung")
            adv_col.prop(props, "edge_connection_redundancy", text="Redundanz")
            adv_col.prop(props, "edge_dynamic_connection_strength", text="Dynamische Stärke")
        
        # Connection Controls
        connect_row = box.row(align=True)
        connect_row.operator("chaintool.create_edge_connections", 
                           text="Verbindungen Erstellen", 
                           icon='STICKY_UVS_LOC')
        connect_row.operator("chaintool.optimize_edge_connections", 
                           text="Optimieren", 
                           icon='MESH_DATA')
        
        toggle_row = box.row()
        toggle_row.prop(props, "show_advanced_edge_connections",
                       text="Erweiterte Verbindungs-Optionen",
                       icon='TRIA_DOWN' if props.show_advanced_edge_connections else 'TRIA_RIGHT')
    
    def _draw_edge_analysis(self, layout, props, state_manager):
        """Edge Analysis Tools"""
        box = layout.box()
        box.label(text="Kanten-Analyse", icon='VIEWZOOM')
        
        # Analysis Tools
        analysis_col = box.column(align=True)
        analysis_col.operator("chaintool.analyze_edge_quality", 
                            text="Kanten-Qualität Analysieren")
        analysis_col.operator("chaintool.stress_test_edges", 
                            text="Belastungstest")
        analysis_col.operator("chaintool.edge_optimization_report", 
                            text="Optimierungs-Bericht")
        
        # Visual Analysis
        if state_manager.has_detected_edges:
            visual_box = box.box()
            visual_box.label(text="Visuelle Analyse", icon='HIDE_OFF')
            
            visual_col = visual_box.column(align=True)
            visual_col.prop(props, "edge_show_stress_colors", text="Stress-Farben")
            visual_col.prop(props, "edge_show_quality_overlay", text="Qualitäts-Overlay")
            visual_col.prop(props, "edge_show_connection_lines", text="Verbindungs-Linien")
            
            # Color Coding Legend
            if any([props.edge_show_stress_colors, 
                   props.edge_show_quality_overlay]):
                legend_box = visual_box.box()
                legend_box.scale_y = 0.7
                legend_box.label(text="Farb-Legende:", icon='COLOR')
                legend_box.label(text="🔴 Hoch | 🟡 Mittel | 🟢 Niedrig")


class CHAINTOOL_PT_edge_presets(Panel):
    """Edge-Tool Presets"""
    bl_label = "Edge Presets"
    bl_idname = "CHAINTOOL_PT_edge_presets"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tool"
    bl_parent_id = "CHAINTOOL_PT_edge_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        
        # Reinforcement Presets
        box = layout.box()
        box.label(text="Verstärkungs-Presets", icon='PRESET')
        
        col = box.column(align=True)
        col.operator("chaintool.apply_edge_preset", 
                    text="Orthese Standard").preset_name = "orthotic_edge"
        col.operator("chaintool.apply_edge_preset", 
                    text="Hochbelastung").preset_name = "high_stress_edge"  
        col.operator("chaintool.apply_edge_preset", 
                    text="Leichtbau").preset_name = "lightweight_edge"
        col.operator("chaintool.apply_edge_preset", 
                    text="Flexible Kanten").preset_name = "flexible_edge"
        
        # Detection Presets
        detect_box = layout.box()
        detect_box.label(text="Erkennungs-Presets", icon='SELECT_EXTEND')
        
        detect_col = detect_box.column(align=True)
        detect_col.operator("chaintool.apply_detection_preset", 
                          text="Scharfe Kanten").preset_name = "sharp_edges"
        detect_col.operator("chaintool.apply_detection_preset", 
                          text="Weiche Übergänge").preset_name = "soft_edges"
        detect_col.operator("chaintool.apply_detection_preset", 
                          text="Alle Kanten").preset_name = "all_edges"


class CHAINTOOL_PT_edge_performance(Panel):
    """Edge-Tool Performance Monitoring"""
    bl_label = "Performance"
    bl_idname = "CHAINTOOL_PT_edge_performance"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tool"
    bl_parent_id = "CHAINTOOL_PT_edge_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        """Nur anzeigen wenn Performance-Monitoring aktiv"""
        perf_monitor = PerformanceMonitor()
        return perf_monitor.is_monitoring_edges
    
    def draw(self, context):
        layout = self.layout
        perf_monitor = PerformanceMonitor()
        
        # Performance Stats
        stats_box = layout.box()
        stats_box.label(text="Performance-Statistiken", icon='SORTTIME')
        
        edge_stats = perf_monitor.get_edge_performance_stats()
        if edge_stats:
            col = stats_box.column(align=True)
            col.label(text=f"Edge Detection: {edge_stats['detection_time']:.2f}s")
            col.label(text=f"Verstärkung: {edge_stats['reinforcement_time']:.2f}s")
            col.label(text=f"Speicher: {edge_stats['memory_usage']:.1f}MB")
            col.label(text=f"BVH-Effizienz: {edge_stats['bvh_efficiency']:.1f}%")
        
        # Optimization Suggestions  
        if edge_stats and edge_stats.get('suggestions'):
            suggest_box = layout.box()
            suggest_box.label(text="Optimierungs-Vorschläge", icon='INFO')
            
            for suggestion in edge_stats['suggestions']:
                suggest_box.label(text=f"• {suggestion}")


# Registration
classes = [
    CHAINTOOL_PT_edge_panel,
    CHAINTOOL_PT_edge_presets,
    CHAINTOOL_PT_edge_performance,
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
