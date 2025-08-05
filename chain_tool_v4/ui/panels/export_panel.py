"""
Chain Tool V4 - Export Control Panel
Dual-Material 3D-Druck Export mit TPU/PETG Support

Features:
- Dual-Material STL Export (TPU + PETG)
- Multi-Material 3D-Printer Support
- Qualitätskontrolle vor Export
- Material-Zuordnung und Optimierung
"""

import bpy
from bpy.types import Panel
from ...core.properties import ChainToolProperties
from ...core.state_manager import StateManager
from ...operators.export_tools import (
    CHAINTOOL_OT_export_dual_material,
    CHAINTOOL_OT_export_stl_separated,
    CHAINTOOL_OT_prepare_for_printing,
    CHAINTOOL_OT_validate_printability
)
from ...presets.material_presets import MaterialPresets
from ...utils.performance import PerformanceMonitor


class CHAINTOOL_PT_export_panel(Panel):
    """Export-Control Haupt-Panel"""
    bl_label = "Export & 3D-Druck"
    bl_idname = "CHAINTOOL_PT_export_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tool"
    bl_parent_id = "CHAINTOOL_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        """Nur anzeigen wenn Pattern vorhanden"""
        state_manager = StateManager()
        return state_manager.has_active_pattern
    
    def draw_header(self, context):
        """Header mit Export-Status"""
        layout = self.layout
        state_manager = StateManager()
        
        layout.label(text="", icon='EXPORT')
        
        if state_manager.is_print_ready:
            layout.label(text="", icon='CHECKMARK')
        elif state_manager.has_export_warnings:
            layout.label(text="", icon='ERROR')
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.chain_tool_properties
        state_manager = StateManager()
        
        # Pre-Export Validation
        self._draw_export_validation(layout, props, state_manager)
        
        # Material Assignment
        self._draw_material_assignment(layout, props)
        
        # Export Settings
        self._draw_export_settings(layout, props)
        
        # 3D-Printer Presets
        self._draw_printer_presets(layout, props)
        
        # Export Actions
        self._draw_export_actions(layout, props, state_manager)
        
        # Post-Export Tools
        if state_manager.has_exported_files:
            self._draw_post_export_tools(layout, props, state_manager)
    
    def _draw_export_validation(self, layout, props, state_manager):
        """Pre-Export Validierung"""
        box = layout.box()
        box.label(text="Export-Validierung", icon='CHECKMARK')
        
        # Validation Status
        validation_results = state_manager.get_validation_results()
        
        if validation_results:
            # Overall Status
            if validation_results['is_valid']:
                status_box = box.box()
                status_box.label(text="✓ Export bereit", icon='CHECKMARK')
            else:
                status_box = box.box()
                status_box.alert = True
                status_box.label(text="⚠ Export-Probleme gefunden", icon='ERROR')
            
            # Detailed Validation Results
            if validation_results.get('issues'):
                issues_box = box.box()
                issues_box.label(text="Gefundene Probleme:", icon='INFO')
                
                for issue in validation_results['issues']:
                    issue_row = issues_box.row()
                    if issue['severity'] == 'ERROR':
                        issue_row.alert = True
                        issue_row.label(text=f"❌ {issue['message']}")
                    elif issue['severity'] == 'WARNING':
                        issue_row.label(text=f"⚠ {issue['message']}", icon='ERROR')
                    else:
                        issue_row.label(text=f"ℹ {issue['message']}", icon='INFO')
        
        # Validation Controls
        val_row = box.row(align=True)
        val_row.operator("chaintool.validate_printability", 
                        text="Druckbarkeit Prüfen", 
                        icon='CHECKMARK')
        val_row.operator("chaintool.fix_export_issues", 
                        text="Probleme Beheben", 
                        icon='TOOL_SETTINGS')
    
    def _draw_material_assignment(self, layout, props):
        """Material-Zuordnung für Dual-Material Druck"""
        box = layout.box()
        box.label(text="Material-Zuordnung", icon='MATERIAL')
        
        # Material Strategy
        strategy_col = box.column(align=True)
        strategy_col.prop(props, "export_material_strategy", text="Zuordnungs-Strategie")
        
        # Strategy-specific Settings
        if props.export_material_strategy == 'AUTOMATIC':
            self._draw_automatic_material_assignment(box, props)
        elif props.export_material_strategy == 'MANUAL':
            self._draw_manual_material_assignment(box, props)
        elif props.export_material_strategy == 'ZONE_BASED':
            self._draw_zone_based_assignment(box, props)
        
        # Material Properties
        self._draw_material_properties(box, props)
    
    def _draw_automatic_material_assignment(self, layout, props):
        """Automatische Material-Zuordnung"""
        auto_box = layout.box()
        auto_box.label(text="Automatische Zuordnung", icon='AUTO')
        
        col = auto_box.column(align=True)
        col.prop(props, "auto_assign_by_stress", text="Nach Belastung")
        col.prop(props, "auto_assign_by_location", text="Nach Position")
        col.prop(props, "auto_assign_by_thickness", text="Nach Dicke")
        
        # Thresholds
        col.separator()
        col.prop(props, "stress_threshold_tpu", text="TPU Stress-Schwelle")
        col.prop(props, "thickness_threshold_petg", text="PETG Dicken-Schwelle")
    
    def _draw_manual_material_assignment(self, layout, props):
        """Manuelle Material-Zuordnung"""
        manual_box = layout.box()
        manual_box.label(text="Manuelle Zuordnung", icon='RESTRICT_SELECT_OFF')
        
        col = manual_box.column(align=True)
        col.operator("chaintool.select_tpu_components", 
                    text="TPU Komponenten Auswählen", 
                    icon='MATERIAL')
        col.operator("chaintool.select_petg_components", 
                    text="PETG Komponenten Auswählen", 
                    icon='MATERIAL')
        
        # Selection Tools
        col.separator()
        sel_row = col.row(align=True)
        sel_row.operator("chaintool.select_by_pattern_type", text="Nach Pattern")
        sel_row.operator("chaintool.select_by_size", text="Nach Größe")
    
    def _draw_zone_based_assignment(self, layout, props):
        """Zonen-basierte Material-Zuordnung"""
        zone_box = layout.box()
        zone_box.label(text="Zonen-basierte Zuordnung", icon='GROUP')
        
        col = zone_box.column(align=True)
        col.prop(props, "material_zone_count", text="Anzahl Zonen")
        col.prop(props, "zone_transition_blend", text="Übergangs-Mischung")
        
        # Zone Configuration
        if props.material_zone_count > 0:
            zones_box = zone_box.box()
            zones_box.label(text="Zonen-Konfiguration:", icon='SETTINGS')
            
            # Simplified zone display (in real implementation would be more complex)
            for i in range(min(props.material_zone_count, 4)):  # Limit display
                zone_row = zones_box.row(align=True)
                zone_row.label(text=f"Zone {i+1}:")
                zone_row.prop(props, f"zone_{i}_material", text="", icon='MATERIAL')
                zone_row.prop(props, f"zone_{i}_priority", text="", icon='SORTBYEXT')
    
    def _draw_material_properties(self, layout, props):
        """Material-Eigenschaften"""
        mat_box = layout.box()
        mat_box.label(text="Material-Eigenschaften", icon='PROPERTIES')
        
        # TPU Settings
        tpu_box = mat_box.box()
        tpu_box.label(text="TPU (Flexibel)", icon='MESH_UVSPHERE')
        tpu_col = tpu_box.column(align=True)
        tpu_col.prop(props, "tpu_shore_hardness", text="Shore-Härte")
        tpu_col.prop(props, "tpu_layer_height", text="Schichthöhe")
        tpu_col.prop(props, "tpu_infill_density", text="Füllung")
        
        # PETG Settings  
        petg_box = mat_box.box()
        petg_box.label(text="PETG (Rigid)", icon='MESH_CUBE')
        petg_col = petg_box.column(align=True)
        petg_col.prop(props, "petg_layer_height", text="Schichthöhe")
        petg_col.prop(props, "petg_infill_pattern", text="Füll-Muster")
        petg_col.prop(props, "petg_infill_density", text="Füllung")
        
        # Advanced Material Settings
        if props.show_advanced_material_settings:
            adv_box = mat_box.box()
            adv_box.label(text="Erweiterte Einstellungen", icon='MODIFIER')
            
            adv_col = adv_box.column(align=True)
            adv_col.prop(props, "material_interface_thickness", text="Interface-Dicke")
            adv_col.prop(props, "material_adhesion_strength", text="Haftungs-Stärke")
            adv_col.prop(props, "material_thermal_expansion", text="Thermische Ausdehnung")
        
        toggle_row = mat_box.row()
        toggle_row.prop(props, "show_advanced_material_settings",
                       text="Erweiterte Material-Optionen",
                       icon='TRIA_DOWN' if props.show_advanced_material_settings else 'TRIA_RIGHT')
    
    def _draw_export_settings(self, layout, props):
        """Export-Einstellungen"""
        box = layout.box()
        box.label(text="Export-Einstellungen", icon='EXPORT')
        
        # File Format
        format_col = box.column(align=True)
        format_col.prop(props, "export_file_format", text="Datei-Format")
        
        # Format-specific Settings
        if props.export_file_format == 'STL':
            self._draw_stl_settings(box, props)
        elif props.export_file_format == '3MF':
            self._draw_3mf_settings(box, props)
        elif props.export_file_format == 'OBJ':
            self._draw_obj_settings(box, props)
        
        # Export Options
        options_box = box.box()
        options_box.label(text="Export-Optionen", icon='SETTINGS')
        
        opt_col = options_box.column(align=True)
        opt_col.prop(props, "export_separate_materials", text="Materialien Getrennt")
        opt_col.prop(props, "export_include_supports", text="Stützen Einbeziehen")
        opt_col.prop(props, "export_merge_overlapping", text="Überlappungen Zusammenfassen")
        opt_col.prop(props, "export_apply_smoothing", text="Glättung Anwenden")
        
        # Quality Settings
        quality_col = opt_col.column(align=True)
        quality_col.separator()
        quality_col.prop(props, "export_mesh_resolution", text="Mesh-Auflösung")
        quality_col.prop(props, "export_precision", text="Export-Präzision")
    
    def _draw_stl_settings(self, layout, props):
        """STL-spezifische Einstellungen"""
        stl_box = layout.box()
        stl_box.label(text="STL-Optionen", icon='MESH_DATA')
        
        col = stl_box.column(align=True)
        col.prop(props, "stl_ascii_format", text="ASCII Format")
        col.prop(props, "stl_precision", text="Dezimal-Präzision")
        col.prop(props, "stl_merge_vertices", text="Vertices Zusammenfassen")
    
    def _draw_3mf_settings(self, layout, props):
        """3MF-spezifische Einstellungen"""
        mf_box = layout.box()
        mf_box.label(text="3MF-Optionen", icon='PACKAGE')
        
        col = mf_box.column(align=True)
        col.prop(props, "mf_include_materials", text="Material-Info Einbeziehen")
        col.prop(props, "mf_include_textures", text="Texturen Einbeziehen")
        col.prop(props, "mf_compression", text="Komprimierung")
    
    def _draw_obj_settings(self, layout, props):
        """OBJ-spezifische Einstellungen"""
        obj_box = layout.box()
        obj_box.label(text="OBJ-Optionen", icon='OBJECT_DATA')
        
        col = obj_box.column(align=True)
        col.prop(props, "obj_include_materials", text="MTL-Datei Erstellen")
        col.prop(props, "obj_smooth_normals", text="Glatte Normalen")
        col.prop(props, "obj_group_by_material", text="Nach Material Gruppieren")
    
    def _draw_printer_presets(self, layout, props):
        """3D-Drucker Presets"""
        box = layout.box()
        box.label(text="3D-Drucker Presets", icon='SETTINGS')
        
        # Printer Selection
        printer_col = box.column(align=True)
        printer_col.prop(props, "target_printer", text="Ziel-Drucker")
        
        # Printer-specific Settings
        if props.target_printer != 'CUSTOM':
            printer_box = box.box()
            printer_info = self._get_printer_info(props.target_printer)
            
            if printer_info:
                info_col = printer_box.column(align=True)
                info_col.label(text=f"Bauraum: {printer_info['build_volume']}")
                info_col.label(text=f"Dual-Material: {'Ja' if printer_info['dual_material'] else 'Nein'}")
                info_col.label(text=f"Max. Auflösung: {printer_info['max_resolution']}")
                
                # Auto-optimize for printer
                info_col.separator()
                info_col.operator("chaintool.optimize_for_printer", 
                                text=f"Für {props.target_printer} Optimieren",
                                icon='AUTO')
        
        # Common Printer Presets
        preset_box = box.box()
        preset_box.label(text="Häufige Drucker", icon='PRESET')
        
        preset_col = preset_box.column(align=True)
        preset_col.operator("chaintool.apply_printer_preset", 
                          text="Prusa i3 MK3S+").printer_name = "PRUSA_MK3S"
        preset_col.operator("chaintool.apply_printer_preset", 
                          text="Bambu Lab X1 Carbon").printer_name = "BAMBU_X1"
        preset_col.operator("chaintool.apply_printer_preset", 
                          text="Ultimaker S3/S5").printer_name = "ULTIMAKER_S5"
        preset_col.operator("chaintool.apply_printer_preset", 
                          text="Generic Dual Extruder").printer_name = "GENERIC_DUAL"
    
    def _draw_export_actions(self, layout, props, state_manager):
        """Export-Aktionen"""
        box = layout.box()
        box.label(text="Export-Aktionen", icon='PLAY')
        
        # Export Buttons
        export_col = box.column(align=True)
        export_col.scale_y = 1.3
        
        # Primary Export
        if props.export_separate_materials:
            export_col.operator("chaintool.export_dual_material", 
                              text="Dual-Material Export", 
                              icon='MATERIAL')
        else:
            export_col.operator("chaintool.export_stl_separated", 
                              text="Getrennte STL-Dateien", 
                              icon='EXPORT')
        
        # Secondary Actions
        action_row = box.row(align=True)
        action_row.operator("chaintool.prepare_for_printing", 
                          text="Druck Vorbereiten", 
                          icon='TOOL_SETTINGS')
        action_row.operator("chaintool.preview_slicing", 
                          text="Slicing-Vorschau", 
                          icon='HIDE_OFF')
        
        # Export Path
        path_box = box.box()
        path_box.prop(props, "export_file_path", text="Export-Pfad")
        path_row = path_box.row(align=True)
        path_row.operator("chaintool.select_export_folder", 
                        text="Ordner Wählen", 
                        icon='FILE_FOLDER')
        path_row.operator("chaintool.open_export_folder", 
                        text="Ordner Öffnen", 
                        icon='FOLDER_REDIRECT')
    
    def _draw_post_export_tools(self, layout, props, state_manager):
        """Post-Export Tools"""
        box = layout.box()
        box.label(text="Nach Export", icon='CHECKMARK')
        
        # Export Results
        results = state_manager.get_export_results()
        if results:
            result_box = box.box()
            result_box.scale_y = 0.8
            
            col = result_box.column(align=True)
            col.label(text=f"Exportierte Dateien: {results['file_count']}")
            col.label(text=f"Gesamt-Größe: {results['total_size_mb']:.1f}MB")
            col.label(text=f"Export-Zeit: {results['export_time']:.1f}s")
        
        # Post-Processing
        post_col = box.column(align=True)
        post_col.operator("chaintool.generate_print_report", 
                        text="Druck-Bericht Erstellen", 
                        icon='TEXT')
        post_col.operator("chaintool.estimate_print_time", 
                        text="Druck-Zeit Schätzen", 
                        icon='TIME')
        post_col.operator("chaintool.calculate_material_cost", 
                        text="Material-Kosten Berechnen", 
                        icon='FUND')
    
    # Helper Methods
    def _get_printer_info(self, printer_name):
        """Drucker-Informationen abrufen"""
        printer_db = {
            'PRUSA_MK3S': {
                'build_volume': '250×210×210mm',
                'dual_material': False,
                'max_resolution': '0.05mm'
            },
            'BAMBU_X1': {
                'build_volume': '256×256×256mm', 
                'dual_material': True,
                'max_resolution': '0.08mm'
            },
            'ULTIMAKER_S5': {
                'build_volume': '330×240×300mm',
                'dual_material': True,
                'max_resolution': '0.06mm'
            },
            'GENERIC_DUAL': {
                'build_volume': '200×200×200mm',
                'dual_material': True,
                'max_resolution': '0.1mm'
            }
        }
        return printer_db.get(printer_name)


class CHAINTOOL_PT_export_quality(Panel):
    """Export-Qualitätskontrolle"""
    bl_label = "Qualitätskontrolle"
    bl_idname = "CHAINTOOL_PT_export_quality"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tool"
    bl_parent_id = "CHAINTOOL_PT_export_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        state_manager = StateManager()
        
        # Quality Checks
        box = layout.box()
        box.label(text="Qualitäts-Prüfungen", icon='CHECKMARK')
        
        check_col = box.column(align=True)
        check_col.operator("chaintool.check_manifold_geometry", 
                         text="Manifold-Geometrie Prüfen")
        check_col.operator("chaintool.check_minimum_thickness", 
                         text="Mindest-Wandstärke Prüfen")
        check_col.operator("chaintool.check_overhangs", 
                         text="Überhänge Analysieren")
        check_col.operator("chaintool.check_printability", 
                         text="Allgemeine Druckbarkeit")
        
        # Quality Results
        quality_results = state_manager.get_quality_check_results()
        if quality_results:
            result_box = layout.box()
            result_box.label(text="Prüf-Ergebnisse", icon='INFO')
            
            for check, result in quality_results.items():
                result_row = result_box.row()
                if result['passed']:
                    result_row.label(text=f"✓ {check}", icon='CHECKMARK')
                else:
                    result_row.alert = True
                    result_row.label(text=f"❌ {check}: {result['issue']}", icon='ERROR')


class CHAINTOOL_PT_export_advanced(Panel):
    """Erweiterte Export-Optionen"""
    bl_label = "Erweiterte Optionen"
    bl_idname = "CHAINTOOL_PT_export_advanced"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chain Tool"  
    bl_parent_id = "CHAINTOOL_PT_export_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.chain_tool_properties
        
        # Batch Export
        batch_box = layout.box()
        batch_box.label(text="Batch-Export", icon='DUPLICATE')
        
        batch_col = batch_box.column(align=True)
        batch_col.prop(props, "batch_export_variations", text="Variationen Exportieren")
        batch_col.prop(props, "batch_export_different_sizes", text="Verschiedene Größen")
        batch_col.operator("chaintool.setup_batch_export", text="Batch-Export Einrichten")
        
        # Custom Scripts
        script_box = layout.box()
        script_box.label(text="Custom Scripts", icon='SCRIPT')
        
        script_col = script_box.column(align=True)
        script_col.prop(props, "run_pre_export_script", text="Pre-Export Script")
        script_col.prop(props, "run_post_export_script", text="Post-Export Script")
        
        if props.run_pre_export_script:
            script_col.prop(props, "pre_export_script_path", text="Script-Pfad")
        
        # Cloud Integration
        cloud_box = layout.box()
        cloud_box.label(text="Cloud-Integration", icon='CLOUD')
        
        cloud_col = cloud_box.column(align=True)
        cloud_col.prop(props, "upload_to_cloud", text="In Cloud Hochladen")
        
        if props.upload_to_cloud:
            cloud_col.prop(props, "cloud_service", text="Service")
            cloud_col.operator("chaintool.setup_cloud_sync", text="Cloud Einrichten")


# Registration
classes = [
    CHAINTOOL_PT_export_panel,
    CHAINTOOL_PT_export_quality,
    CHAINTOOL_PT_export_advanced,
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
