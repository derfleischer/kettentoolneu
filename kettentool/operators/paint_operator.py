"""
Paint Mode Operator - Vollständig integriert mit Pattern System
"""

import bpy
from bpy.types import Operator
from bpy.props import BoolProperty
from typing import List
import kettentool.creation.spheres as spheres
import kettentool.creation.connectors as connectors
import kettentool.utils.paintmode as paintmode
import kettentool.core.constants as constants


class KETTE_OT_paint_mode(Operator):
    """Interaktiver Paint Modus für Kugeln mit Pattern-Unterstützung"""
    bl_idname = "kette.paint_mode"
    bl_label = "Chain Paint Mode"
    bl_options = {'REGISTER', 'UNDO', 'BLOCKING'}
    bl_description = "Male Kugeln auf die Oberfläche mit Auto-Verbindung"
    
    # Internal properties (not exposed to UI)
    use_pattern_connect: BoolProperty(
        name="Pattern verwenden",
        description="Verwende Pattern statt linearer Verbindung",
        default=False
    )
    
    def __init__(self):
        """Initialize paint mode"""
        self.draw_handler = None
        self.painting = False
        self.navigation_mode = False
        self.last_position = None
        self.painted_spheres: List[bpy.types.Object] = []
        self.cursor_3d_pos = None
        self.target_warning_shown = False
        self.pattern_mode = False
    
    @classmethod
    def poll(cls, context):
        """Check if paint mode can be started"""
        props = context.scene.chain_props
        return (context.area and 
                context.area.type == 'VIEW_3D' and 
                props.kette_target_obj is not None)
    
    def modal(self, context, event):
        """Main modal handler"""
        context.area.tag_redraw()
        props = context.scene.chain_props
        
        # Check for target object
        if not props.kette_target_obj:
            if not self.target_warning_shown:
                self.report({'WARNING'}, "⚠️ Zielobjekt verloren!")
                self.target_warning_shown = True
        else:
            self.target_warning_shown = False
        
        # Update cursor position (only in paint mode)
        if event.type == 'MOUSEMOVE' and not self.navigation_mode:
            if props.kette_target_obj:
                self.cursor_3d_pos = paintmode.get_paint_surface_position(
                    context, event, props.kette_target_obj
                )
            else:
                self.cursor_3d_pos = None
        
        # === KEY BINDINGS ===
        
        # SPACE - Toggle navigation
        if event.type == 'SPACE' and event.value == 'PRESS':
            self.navigation_mode = not self.navigation_mode
            mode = "NAVIGATION" if self.navigation_mode else "PAINT"
            self.report({'INFO'}, f"Modus: {mode}")
            return {'RUNNING_MODAL'}
        
        # Pass through navigation events
        if self.navigation_mode:
            if event.type not in {'ESC', 'SPACE'}:
                return {'PASS_THROUGH'}
            else:
                return {'RUNNING_MODAL'}
        
        # ESC - Exit
        if event.type in {'ESC'}:
            return self.finish(context)
        
        # === PAINTING ===
        
        # LEFT MOUSE - Paint spheres
        if event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                self.painting = True
                self.paint_sphere(context, event)
            elif event.value == 'RELEASE':
                self.painting = False
                self.last_position = None
        
        # Paint while dragging
        if self.painting and event.type == 'MOUSEMOVE':
            self.paint_sphere(context, event)
        
        # RIGHT MOUSE - Remove spheres
        if event.type == 'RIGHTMOUSE' and event.value == 'PRESS':
            self.remove_sphere_at_cursor(context, event)
        
        # === CONNECTION CONTROLS ===
        
        # C - Connect with linear pattern
        if event.type == 'C' and event.value == 'PRESS' and not event.shift:
            self.connect_painted_linear(context)
        
        # P - Open pattern dialog
        if event.type == 'P' and event.value == 'PRESS' and not event.shift:
            self.apply_pattern_dialog(context)
        
        # T - Quick triangulation
        if event.type == 'T' and event.value == 'PRESS' and not event.shift:
            self.quick_triangulate(context)
        
        # V - Quick Voronoi
        if event.type == 'V' and event.value == 'PRESS' and not event.shift:
            self.quick_voronoi(context)
        
        # A - Toggle auto-connect
        if event.type == 'A' and event.value == 'PRESS' and not event.shift:
            props.kette_auto_connect = not props.kette_auto_connect
            status = 'AN' if props.kette_auto_connect else 'AUS'
            self.report({'INFO'}, f"Auto-Connect: {status}")
        
        # S - Toggle symmetry
        if event.type == 'S' and event.value == 'PRESS' and not event.shift:
            self.toggle_symmetry(context)
        
        # === ADJUSTMENTS ===
        
        # Scroll - Adjust sphere radius
        if event.type == 'WHEELUPMOUSE':
            props.kette_kugelradius = min(props.kette_kugelradius * 1.1, 50.0)
            self.report({'INFO'}, f"Radius: {props.kette_kugelradius:.3f}")
        elif event.type == 'WHEELDOWNMOUSE':
            props.kette_kugelradius = max(props.kette_kugelradius * 0.9, 0.001)
            self.report({'INFO'}, f"Radius: {props.kette_kugelradius:.3f}")
        
        # SHIFT+Scroll - Adjust spacing
        if event.shift:
            if event.type == 'WHEELUPMOUSE':
                if hasattr(props, 'paint_brush_spacing'):
                    props.paint_brush_spacing = min(props.paint_brush_spacing * 1.1, 10.0)
                    self.report({'INFO'}, f"Spacing: {props.paint_brush_spacing:.2f}")
            elif event.type == 'WHEELDOWNMOUSE':
                if hasattr(props, 'paint_brush_spacing'):
                    props.paint_brush_spacing = max(props.paint_brush_spacing * 0.9, 1.0)
                    self.report({'INFO'}, f"Spacing: {props.paint_brush_spacing:.2f}")
        
        # D - Delete all painted spheres
        if event.type == 'D' and event.value == 'PRESS' and event.shift:
            self.delete_all_painted(context)
        
        # O - Optimize for 3D print
        if event.type == 'O' and event.value == 'PRESS' and not event.shift:
            self.optimize_for_print(context)
        
        return {'RUNNING_MODAL'}
    
    def invoke(self, context, event):
        """Initialize paint mode"""
        if context.space_data.type != 'VIEW_3D':
            self.report({'WARNING'}, "Muss in 3D View ausgeführt werden")
            return {'CANCELLED'}
        
        # Initialize
        self.painted_spheres = []
        self.painting = False
        self.navigation_mode = False
        self.last_position = None
        self.cursor_3d_pos = None
        self.pattern_mode = False
        
        # Setup draw handler
        args = (self, context)
        self.draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            paintmode.draw_paint_preview, args, 'WINDOW', 'POST_VIEW'
        )
        
        # Add modal handler
        context.window_manager.modal_handler_add(self)
        
        # Show help
        self.report({'INFO'}, 
                   "Paint Mode aktiv | H für Hilfe | ESC zum Beenden")
        self.show_help_brief()
        
        return {'RUNNING_MODAL'}
    
    def finish(self, context):
        """Clean up and exit paint mode"""
        # Remove draw handler
        if self.draw_handler:
            bpy.types.SpaceView3D.draw_handler_remove(self.draw_handler, 'WINDOW')
        
        # Final auto-connect if enabled
        props = context.scene.chain_props
        if props.kette_auto_connect and len(self.painted_spheres) > 1:
            self.connect_painted_linear(context)
        
        # Report statistics
        sphere_count = len(self.painted_spheres)
        connector_count = len([obj for obj in bpy.data.objects 
                             if "Connector" in obj.name and 
                             any(obj.get("sphere1") == s.name or 
                                obj.get("sphere2") == s.name 
                                for s in self.painted_spheres)])
        
        self.report({'INFO'}, 
                   f"Paint Mode beendet | {sphere_count} Kugeln, "
                   f"{connector_count} Verbindungen erstellt")
        
        # Clear painted list
        self.painted_spheres.clear()
        
        return {'FINISHED'}
    
    # =========================================
    # PAINTING FUNCTIONS
    # =========================================
    
    def paint_sphere(self, context, event):
        """Paint a sphere at cursor position"""
        props = context.scene.chain_props
        
        if not props.kette_target_obj:
            return
        
        # Get surface position
        position = paintmode.get_paint_surface_position(
            context, event, props.kette_target_obj
        )
        
        if not position:
            return
        
        # Check minimum distance
        spacing_factor = props.paint_brush_spacing if hasattr(props, 'paint_brush_spacing') else 2.5
        min_dist = props.kette_kugelradius * spacing_factor
        
        if self.last_position:
            if (position - self.last_position).length < min_dist:
                return
        
        # Check distance to existing spheres
        if not paintmode.check_minimum_distance(
            position, self.painted_spheres, props.kette_kugelradius * 2
        ):
            return
        
        # Get positions (with symmetry if enabled)
        positions = [position]
        if hasattr(props, 'paint_use_symmetry') and props.paint_use_symmetry:
            axis = props.paint_symmetry_axis if hasattr(props, 'paint_symmetry_axis') else 'X'
            positions = paintmode.get_symmetry_positions(position, axis)
        
        # Create spheres
        for pos in positions:
            sphere = spheres.create_sphere(pos, props.kette_kugelradius)
            
            if sphere:
                sphere.name = f"Kugel_Paint_{len(self.painted_spheres):03d}"
                sphere["painted"] = True
                self.painted_spheres.append(sphere)
                
                # Auto-connect if enabled
                if props.kette_auto_connect and len(self.painted_spheres) > 1:
                    prev_sphere = self.painted_spheres[-2]
                    dist = (sphere.location - prev_sphere.location).length
                    max_dist = props.kette_kugelradius * 6
                    
                    if dist < max_dist:
                        connector = connectors.connect_spheres(
                            prev_sphere, sphere, 
                            props.kette_use_curved, props
                        )
                        if connector:
                            connector["paint_connected"] = True
        
        # Update last position
        self.last_position = position
        
        if constants.debug:
            constants.debug.trace('PAINT', f"Painted sphere at {position}")
    
    def remove_sphere_at_cursor(self, context, event):
        """Remove sphere under cursor"""
        sphere = paintmode.find_sphere_at_cursor(context, event)
        
        if sphere and sphere in self.painted_spheres:
            self.painted_spheres.remove(sphere)
            
            # Remove associated connectors
            for obj in list(bpy.data.objects):
                if "Connector" in obj.name:
                    if (obj.get("sphere1") == sphere.name or 
                        obj.get("sphere2") == sphere.name):
                        bpy.data.objects.remove(obj, do_unlink=True)
            
            # Remove sphere
            bpy.data.objects.remove(sphere, do_unlink=True)
            
            self.report({'INFO'}, "Kugel entfernt")
    
    # =========================================
    # CONNECTION FUNCTIONS
    # =========================================
    
    def connect_painted_linear(self, context):
        """Connect painted spheres with linear pattern"""
        if len(self.painted_spheres) < 2:
            self.report({'WARNING'}, "Mindestens 2 Kugeln für Verbindung nötig")
            return
        
        props = context.scene.chain_props
        
        # Import pattern connector
        from kettentool.operators.pattern_connector import KETTE_OT_apply_pattern
        
        # Apply linear pattern
        operator = KETTE_OT_apply_pattern()
        operator.pattern_type = 'LINEAR'
        operator.remove_existing = False
        
        connections = operator.get_pattern_connections(self.painted_spheres)
        created = 0
        
        for s1, s2 in connections:
            if not operator.are_connected(s1, s2):
                connector = connectors.connect_spheres(
                    s1, s2, props.kette_use_curved, props
                )
                if connector:
                    connector["paint_pattern"] = "LINEAR"
                    created += 1
        
        self.report({'INFO'}, f"Linear: {created} Verbindungen erstellt")
    
    def apply_pattern_dialog(self, context):
        """Open pattern dialog for painted spheres"""
        if len(self.painted_spheres) < 2:
            self.report({'WARNING'}, "Mindestens 2 Kugeln für Pattern nötig")
            return
        
        # Select painted spheres
        bpy.ops.object.select_all(action='DESELECT')
        for sphere in self.painted_spheres:
            if sphere.name in bpy.data.objects:
                sphere.select_set(True)
        
        # Open pattern dialog
        try:
            bpy.ops.kette.apply_pattern('INVOKE_DEFAULT')
            self.report({'INFO'}, "Pattern Dialog geöffnet")
        except Exception as e:
            self.report({'ERROR'}, f"Pattern Fehler: {e}")
    
    def quick_triangulate(self, context):
        """Quick triangulation of painted spheres"""
        if len(self.painted_spheres) < 3:
            self.report({'WARNING'}, "Mindestens 3 Kugeln für Triangulation")
            return
        
        # Select and apply
        bpy.ops.object.select_all(action='DESELECT')
        for sphere in self.painted_spheres:
            if sphere.name in bpy.data.objects:
                sphere.select_set(True)
        
        try:
            bpy.ops.kette.apply_pattern(
                pattern_type='TRIANGULATE',
                remove_existing=True
            )
            self.report({'INFO'}, "Triangulation angewendet")
        except:
            self.report({'ERROR'}, "Triangulation fehlgeschlagen")
    
    def quick_voronoi(self, context):
        """Quick Voronoi pattern"""
        if len(self.painted_spheres) < 3:
            self.report({'WARNING'}, "Mindestens 3 Kugeln für Voronoi")
            return
        
        # Select and apply
        bpy.ops.object.select_all(action='DESELECT')
        for sphere in self.painted_spheres:
            if sphere.name in bpy.data.objects:
                sphere.select_set(True)
        
        try:
            bpy.ops.kette.apply_pattern(
                pattern_type='VORONOI',
                remove_existing=True
            )
            self.report({'INFO'}, "Voronoi angewendet")
        except:
            self.report({'ERROR'}, "Voronoi fehlgeschlagen")
    
    def optimize_for_print(self, context):
        """Optimize painted spheres for 3D printing"""
        if len(self.painted_spheres) < 2:
            self.report({'WARNING'}, "Mindestens 2 Kugeln für Optimierung")
            return
        
        # Select painted spheres
        bpy.ops.object.select_all(action='DESELECT')
        for sphere in self.painted_spheres:
            if sphere.name in bpy.data.objects:
                sphere.select_set(True)
        
        # Apply optimized pattern
        try:
            bpy.ops.kette.apply_pattern(
                pattern_type='TRIANGULATE',
                optimize_for_print=True,
                add_intermediate_spheres=True,
                max_connection_length=5.0,
                ensure_minimum_angles=True,
                minimum_angle=30.0,
                check_printability=True
            )
            self.report({'INFO'}, "3D-Druck Optimierung angewendet")
        except Exception as e:
            self.report({'ERROR'}, f"Optimierung fehlgeschlagen: {e}")
    
    # =========================================
    # UTILITY FUNCTIONS
    # =========================================
    
    def toggle_symmetry(self, context):
        """Toggle symmetry mode"""
        props = context.scene.chain_props
        
        if hasattr(props, 'paint_use_symmetry'):
            props.paint_use_symmetry = not props.paint_use_symmetry
            status = 'AN' if props.paint_use_symmetry else 'AUS'
            
            if props.paint_use_symmetry and hasattr(props, 'paint_symmetry_axis'):
                axis = props.paint_symmetry_axis
                self.report({'INFO'}, f"Symmetrie: {status} ({axis}-Achse)")
            else:
                self.report({'INFO'}, f"Symmetrie: {status}")
    
    def delete_all_painted(self, context):
        """Delete all painted spheres and their connections"""
        if not self.painted_spheres:
            self.report({'INFO'}, "Keine gemalten Kugeln vorhanden")
            return
        
        # Remove connectors first
        for obj in list(bpy.data.objects):
            if "Connector" in obj.name:
                for sphere in self.painted_spheres:
                    if (obj.get("sphere1") == sphere.name or 
                        obj.get("sphere2") == sphere.name):
                        bpy.data.objects.remove(obj, do_unlink=True)
                        break
        
        # Remove spheres
        count = len(self.painted_spheres)
        for sphere in self.painted_spheres:
            if sphere.name in bpy.data.objects:
                bpy.data.objects.remove(sphere, do_unlink=True)
        
        self.painted_spheres.clear()
        self.report({'INFO'}, f"{count} Kugeln gelöscht")
    
    def show_help_brief(self):
        """Show brief help message"""
        help_text = (
            "LMB: Malen | RMB: Löschen | SPACE: Navigation | "
            "C: Verbinden | P: Pattern | T: Triangulation | "
            "O: 3D-Druck Opt. | ESC: Beenden"
        )
        self.report({'INFO'}, help_text)


# =========================================
# REGISTRATION
# =========================================

classes = [
    KETTE_OT_paint_mode,
]

def register():
    """Register paint operators"""
    for cls in classes:
        bpy.utils.register_class(cls)
    
    if constants.debug:
        constants.debug.info('OPERATORS', "Paint mode registered")

def unregister():
    """Unregister paint operators"""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
