"""
Chain Tool V4 - Pattern Configuration Panel
Erweiterte Pattern-Parameter für alle Pattern-Typen

Features:
- Pattern-spezifische Einstellungen (Voronoi, Hexagonal, etc.)
- Live-Preview Integration
- Pattern-Density Controls
- Hybrid-Pattern Konfiguration
"""

import bpy
from bpy.types import Panel
from ...core.properties import ChainToolProperties
from ...core.state_manager import StateManager
from ...patterns.surface_patterns import VoronoiPattern, HexagonalGrid, RandomDistribution
from ...patterns.edge_patterns import RimReinforcement, ContourBanding
from ...patterns.hybrid_patterns import CoreShellPattern, GradientDensity
from ...utils.debug import DebugSystem


class CHAINTOOL_PT_pattern_panel(Panel):
    """Pattern-Konfiguration Haupt-Panel"""
    bl_label = "Pattern Konfiguration"
    bl_idname = "CHAINTOOL_PT_pattern_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tool"
    bl_parent_id = "CHAINTOOL_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        """Nur anzeigen wenn Pattern-Type nicht PAINT ist"""
        props = getattr(context.scene, 'chain_tool_properties', None)
        return props and props.pattern_type != 'PAINT'
    
    def draw_header(self, context):
        """Header mit Pattern-Type Icon"""
        layout = self.layout
        props = context.scene.chain_tool_properties
        
        pattern_icons = {
            'SURFACE': 'MESH_UVSPHERE',
            'EDGE': 'EDGESEL', 
            'HYBRID': 'MOD_LATTICE'
        }
        
        icon = pattern_icons.get(props.pattern_type, 'MODIFIER')
        layout.label(text="", icon=icon)
        
        # Live Preview Indicator
        if props.pattern_live_preview:
            layout.label(text="", icon='HIDE_OFF')
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.chain_tool_properties
        
        # Live Preview Toggle
        self._draw_live_preview_controls(layout, props)
        
        # Pattern-spezifische Einstellungen
        if props.pattern_type == 'SURFACE':
            self._draw_surface_pattern_settings(layout, props)
        elif props.pattern_type == 'EDGE':
            self._draw_edge_pattern_settings(layout, props)
        elif props.pattern_type == 'HYBRID':
            self._draw_hybrid_pattern_settings(layout, props)
        
        # Gemeinsame Pattern-Tools
        self._draw_common_pattern_tools(layout, props)
    
    def _draw_live_preview_controls(self, layout, props):
        """Live-Preview Steuerung"""
        box = layout.box()
        
        # Preview Toggle
        row = box.row(align=True)
        row.prop(props, "pattern_live_preview", text="Live-Vorschau", icon='HIDE_OFF')
        
        if props.pattern_live_preview:
            row.prop(props, "pattern_preview_quality", text="", icon='CAMERA_DATA')
        
        # Preview Settings
        if props.pattern_live_preview:
            preview_box = box.box()
            preview_box.scale_y = 0.9
            
            col = preview_box.column(align=True)
            col.prop(props, "pattern_preview_update_rate", text="Update-Rate")
            col.prop(props, "pattern_preview_max_spheres", text="Max. Vorschau-Kugeln")
            
            # Performance Warning
            state_manager = StateManager()
            if state_manager.preview_performance_warning:
                warning_row = preview_box.row()
                warning_row.alert = True
                warning_row.label(text="⚠ Preview verlangsamt System", icon='ERROR')
                warning_row.operator("chaintool.disable_preview", text="", icon='X')
    
    def _draw_surface_pattern_settings(self, layout, props):
        """Surface Pattern Einstellungen"""
        box = layout.box()
        box.label(text="Oberflächen-Pattern", icon='MESH_UVSPHERE')
        
        # Pattern Subtype
        col = box.column(align=True)
        col.prop(props, "surface_pattern_type", text="Typ")
        
        # Pattern-spezifische Einstellungen
        if props.surface_pattern_type == 'VORONOI':
            self._draw_voronoi_settings(box, props)
        elif props.surface_pattern_type == 'HEXAGONAL':
            self._draw_hexagonal_settings(box, props)
        elif props.surface_pattern_type == 'RANDOM':
            self._draw_random_settings(box, props)
        elif props.surface_pattern_type == 'FIBONACCI':
            self._draw_fibonacci_settings(box, props)
        
        # Common Surface Settings
        self._draw_common_surface_settings(box, props)
    
    def _draw_voronoi_settings(self, layout, props):
        """Voronoi Pattern Einstellungen"""
        voronoi_box = layout.box()
        voronoi_box.label(text="Voronoi Parameter", icon='MOD_SCREW')
        
        col = voronoi_box.column(align=True)
        col.prop(props, "voronoi_seed", text="Seed")
        col.prop(props, "voronoi_density", text="Dichte")
        col.prop(props, "voronoi_relaxation", text="Relaxation")
        
        # Advanced Voronoi
        if props.show_advanced_voronoi:
            adv_col = col.column(align=True)
            adv_col.separator()
            adv_col.prop(props, "voronoi_lloyd_iterations", text="Lloyd Iterationen")
            adv_col.prop(props, "voronoi_boundary_treatment", text="Rand-Behandlung")
            adv_col.prop(props, "voronoi_cell_size_variation", text="Zell-Größen Variation")
            adv_col.prop(props, "voronoi_use_3d_noise", text="3D Noise Modulation")
        
        toggle_row = voronoi_box.row()
        toggle_row.prop(props, "show_advanced_voronoi",
                       text="Erweiterte Voronoi-Optionen",
                       icon='TRIA_DOWN' if props.show_advanced_voronoi else 'TRIA_RIGHT')
    
    def _draw_hexagonal_settings(self, layout, props):
        """Hexagonal Pattern Einstellungen"""
        hex_box = layout.box()
        hex_box.label(text="Hexagonal Parameter", icon='MESH_GRID')
        
        col = hex_box.column(align=True)
        col.prop(props, "hex_grid_size", text="Gitter-Größe")
        col.prop(props, "hex_offset_variation", text="Offset-Variation")
        col.prop(props, "hex_rotation", text="Rotation")
        
        # Hexagonal Variations
        col.separator()
        col.prop(props, "hex_pattern_variation", text="Muster-Variation")
        
        if props.hex_pattern_variation != 'REGULAR':
            var_col = col.column(align=True)
            var_col.prop(props, "hex_variation_strength", text="Variations-Stärke")
            var_col.prop(props, "hex_noise_scale", text="Noise-Skalierung")
    
    def _draw_random_settings(self, layout, props):
        """Random Pattern Einstellungen"""
        random_box = layout.box()
        random_box.label(text="Zufalls Parameter", icon='FORCE_TURBULENCE')
        
        col = random_box.column(align=True)
        col.prop(props, "random_seed", text="Seed")
        col.prop(props, "random_density", text="Dichte")
        col.prop(props, "random_min_distance", text="Min. Abstand")
        col.prop(props, "random_distribution", text="Verteilung")
        
        # Clustering Controls
        if props.random_distribution == 'CLUSTERED':
            cluster_col = col.column(align=True)
            cluster_col.separator()
            cluster_col.prop(props, "random_cluster_count", text="Cluster-Anzahl")
            cluster_col.prop(props, "random_cluster_radius", text="Cluster-Radius")
            cluster_col.prop(props, "random_cluster_density", text="Cluster-Dichte")
    
    def _draw_fibonacci_settings(self, layout, props):
        """Fibonacci Spiral Pattern Einstellungen"""
        fib_box = layout.box()
        fib_box.label(text="Fibonacci Parameter", icon='FORCE_VORTEX')
        
        col = fib_box.column(align=True)
        col.prop(props, "fibonacci_count", text="Punkt-Anzahl")
        col.prop(props, "fibonacci_golden_angle", text="Goldener Winkel")
        col.prop(props, "fibonacci_projection_method", text="Projektion")
        
        # Spiral Modifications
        col.separator()
        col.prop(props, "fibonacci_spiral_tightness", text="Spiral-Dichte")
        col.prop(props, "fibonacci_noise_influence", text="Noise-Einfluss")
    
    def _draw_common_surface_settings(self, layout, props):
        """Gemeinsame Surface Pattern Einstellungen"""
        common_box = layout.box()
        common_box.label(text="Oberflächen-Anpassung", icon='SURFACE_DATA')
        
        col = common_box.column(align=True)
        col.prop(props, "surface_adaptive_density", text="Adaptive Dichte")
        
        if props.surface_adaptive_density:
            adapt_col = col.column(align=True)
            adapt_col.prop(props, "surface_curvature_influence", text="Krümmungs-Einfluss")
            adapt_col.prop(props, "surface_area_influence", text="Flächen-Einfluss")
            adapt_col.prop(props, "surface_edge_influence", text="Kanten-Einfluss")
        
        col.separator()
        col.prop(props, "surface_projection_method", text="Projektions-Methode")
        col.prop(props, "surface_normal_offset", text="Normalen-Offset")
    
    def _draw_edge_pattern_settings(self, layout, props):
        """Edge Pattern Einstellungen"""
        box = layout.box()
        box.label(text="Kanten-Pattern", icon='EDGESEL')
        
        # Edge Detection Settings
        detect_box = box.box()
        detect_box.label(text="Kanten-Erkennung", icon='SELECT_EXTEND')
        
        col = detect_box.column(align=True)
        col.prop(props, "edge_detection_method", text="Methode")
        col.prop(props, "edge_angle_threshold", text="Winkel-Schwelle")
        col.prop(props, "edge_length_threshold", text="Längen-Schwelle")
        
        # Reinforcement Settings
        reinforce_box = box.box()
        reinforce_box.label(text="Verstärkung", icon='MOD_SOLIDIFY')
        
        col = reinforce_box.column(align=True)
        col.prop(props, "edge_reinforcement_type", text="Verstärkungstyp")
        col.prop(props, "edge_reinforcement_density", text="Verstärkungs-Dichte")
        col.prop(props, "edge_reinforcement_width", text="Verstärkungs-Breite")
        
        # Advanced Edge Settings
        if props.show_advanced_edge:
            adv_box = box.box()
            adv_box.label(text="Erweiterte Kanten-Optionen", icon='SETTINGS')
            
            adv_col = adv_box.column(align=True)
            adv_col.prop(props, "edge_follow_contours", text="Konturen Folgen")
            adv_col.prop(props, "edge_branch_probability", text="Verzweigungs-Wahrscheinlichkeit")
            adv_col.prop(props, "edge_taper_strength", text="Verjüngungs-Stärke")
        
        toggle_row = box.row()
        toggle_row.prop(props, "show_advanced_edge",
                       text="Erweiterte Kanten-Optionen",
                       icon='TRIA_DOWN' if props.show_advanced_edge else 'TRIA_RIGHT')
    
    def _draw_hybrid_pattern_settings(self, layout, props):
        """Hybrid Pattern Einstellungen"""
        box = layout.box()
        box.label(text="Hybrid-Pattern", icon='MOD_LATTICE')
        
        # Hybrid Composition
        comp_box = box.box()
        comp_box.label(text="Pattern-Komposition", icon='NODE_COMPOSITING')
        
        col = comp_box.column(align=True)
        col.prop(props, "hybrid_primary_pattern", text="Primär-Pattern")
        col.prop(props, "hybrid_secondary_pattern", text="Sekundär-Pattern")
        col.prop(props, "hybrid_blend_mode", text="Misch-Modus")
        col.prop(props, "hybrid_blend_factor", text="Misch-Faktor")
        
        # Zone-based Mixing
        zone_box = box.box()
        zone_box.label(text="Zonen-basierte Mischung", icon='GROUP')
        
        zone_col = zone_box.column(align=True)
        zone_col.prop(props, "hybrid_use_zones", text="Zonen Verwenden")
        
        if props.hybrid_use_zones:
            zone_subcol = zone_col.column(align=True)
            zone_subcol.prop(props, "hybrid_zone_count", text="Zonen-Anzahl")
            zone_subcol.prop(props, "hybrid_zone_transition", text="Übergangs-Breite")
            zone_subcol.prop(props, "hybrid_zone_seed", text="Zonen-Seed")
        
        # Gradient Controls
        gradient_box = box.box()
        gradient_box.label(text="Gradienten-Steuerung", icon='COLOR')
        
        grad_col = gradient_box.column(align=True)
        grad_col.prop(props, "hybrid_use_gradient", text="Gradient Verwenden")
        
        if props.hybrid_use_gradient:
            grad_subcol = grad_col.column(align=True)
            grad_subcol.prop(props, "hybrid_gradient_direction", text="Richtung")
            grad_subcol.prop(props, "hybrid_gradient_curve", text="Kurven-Typ")
            grad_subcol.prop(props, "hybrid_gradient_noise", text="Gradient-Noise")
    
    def _draw_common_pattern_tools(self, layout, props):
        """Gemeinsame Pattern-Tools"""
        box = layout.box()
        box.label(text="Pattern-Tools", icon='TOOL_SETTINGS')
        
        # Optimization Tools
        opt_row = box.row(align=True)
        opt_row.operator("chaintool.optimize_pattern_density", 
                        text="Dichte Optimieren", 
                        icon='MESH_DATA')
        opt_row.operator("chaintool.validate_pattern", 
                        text="Validieren", 
                        icon='CHECKMARK')
        
        # Modification Tools
        mod_row = box.row(align=True)
        mod_row.operator("chaintool.randomize_pattern", 
                        text="Randomisieren", 
                        icon='FORCE_TURBULENCE')
        mod_row.operator("chaintool.smooth_pattern", 
                        text="Glätten", 
                        icon='MOD_SMOOTH')
        
        # Advanced Tools
        if props.show_advanced_pattern_tools:
            adv_box = box.box()
            adv_box.label(text="Erweiterte Tools", icon='MODIFIER')
            
            adv_col = adv_box.column(align=True)
            adv_col.operator("chaintool.pattern_stress_analysis", 
                           text="Stress-Analyse", 
                           icon='FORCE_LENNNARDJ')
            adv_col.operator("chaintool.pattern_material_optimization", 
                           text="Material-Optimierung", 
                           icon='MATERIAL')
            adv_col.operator("chaintool.pattern_export_data", 
                           text="Pattern-Daten Exportieren", 
                           icon='EXPORT')
        
        toggle_row = box.row()
        toggle_row.prop(props, "show_advanced_pattern_tools",
                       text="Erweiterte Pattern-Tools",
                       icon='TRIA_DOWN' if props.show_advanced_pattern_tools else 'TRIA_RIGHT')


class CHAINTOOL_PT_pattern_presets(Panel):
    """Pattern-Preset Management"""
    bl_label = "Pattern-Presets"
    bl_idname = "CHAINTOOL_PT_pattern_presets"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tool"
    bl_parent_id = "CHAINTOOL_PT_pattern_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.chain_tool_properties
        
        # Preset Library
        box = layout.box()
        box.label(text="Preset-Bibliothek", icon='PRESET')
        
        # Preset List
        if hasattr(props, 'pattern_presets') and props.pattern_presets:
            preset_box = box.box()
            preset_box.template_list("CHAINTOOL_UL_pattern_presets", "", 
                                   props, "pattern_presets",
                                   props, "active_preset_index")
            
            # Preset Actions
            preset_row = preset_box.row(align=True) 
            preset_row.operator("chaintool.load_pattern_preset", 
                              text="Laden", 
                              icon='IMPORT')
            preset_row.operator("chaintool.save_pattern_preset", 
                              text="Speichern", 
                              icon='EXPORT')
            preset_row.operator("chaintool.delete_pattern_preset", 
                              text="", 
                              icon='TRASH')
        else:
            box.label(text="Keine Presets verfügbar", icon='INFO')
            box.operator("chaintool.create_default_presets", 
                        text="Standard-Presets Erstellen")
        
        # Quick Presets
        quick_box = layout.box()
        quick_box.label(text="Schnell-Presets", icon='LIGHTPROBE')
        
        quick_col = quick_box.column(align=True)
        quick_col.operator("chaintool.apply_orthotic_preset", 
                         text="Orthese Standard")
        quick_col.operator("chaintool.apply_prosthetic_preset", 
                         text="Prothese Standard")
        quick_col.operator("chaintool.apply_lightweight_preset", 
                         text="Leichtbau")
        quick_col.operator("chaintool.apply_reinforced_preset", 
                         text="Verstärkt")


class CHAINTOOL_PT_pattern_analysis(Panel):
    """Pattern-Analyse und Statistiken"""
    bl_label = "Pattern-Analyse"
    bl_idname = "CHAINTOOL_PT_pattern_analysis"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tool"
    bl_parent_id = "CHAINTOOL_PT_pattern_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        state_manager = StateManager()
        return state_manager.has_active_pattern
    
    def draw(self, context):
        layout = self.layout
        state_manager = StateManager()
        
        # Pattern Statistics
        stats_box = layout.box()
        stats_box.label(text="Pattern-Statistiken", icon='SORTTIME')
        
        stats = state_manager.get_pattern_statistics()
        if stats:
            col = stats_box.column(align=True)
            col.label(text=f"Kugeln: {stats['sphere_count']}")
            col.label(text=f"Verbindungen: {stats['connection_count']}")
            col.label(text=f"Oberflächen-Abdeckung: {stats['surface_coverage']:.1f}%")
            col.label(text=f"Durchschn. Kugel-Abstand: {stats['avg_distance']:.2f}")
            
            # Quality Metrics
            quality_col = col.column(align=True)
            quality_col.separator()
            quality_col.label(text=f"Pattern-Qualität: {stats['pattern_quality']:.1f}/10")
            quality_col.label(text=f"Strukturelle Integrität: {stats['structural_integrity']:.1f}/10")
        
        # Analysis Tools
        analysis_box = layout.box()
        analysis_box.label(text="Analyse-Tools", icon='VIEWZOOM')
        
        analysis_col = analysis_box.column(align=True)
        analysis_col.operator("chaintool.analyze_pattern_distribution", 
                            text="Verteilungs-Analyse")
        analysis_col.operator("chaintool.analyze_connection_quality", 
                            text="Verbindungs-Qualität")
        analysis_col.operator("chaintool.generate_pattern_report", 
                            text="Analyse-Bericht Erstellen")


# Registration
classes = [
    CHAINTOOL_PT_pattern_panel,
    CHAINTOOL_PT_pattern_presets, 
    CHAINTOOL_PT_pattern_analysis,
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
