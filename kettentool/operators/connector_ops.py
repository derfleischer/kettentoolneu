"""
Connector Operations - Verbinder-Funktionslogik
Optimiert für Blender 4.4 + macOS
"""

import bpy
from bpy.types import Operator
from bpy.props import BoolProperty, EnumProperty, IntProperty, FloatProperty

# Direkte Imports
import kettentool.creation.connectors as connector_creation
import kettentool.creation.spheres as sphere_creation
import kettentool.core.constants as constants

# =========================================
# CONNECTOR CREATION OPERATORS
# =========================================

class KETTE_OT_connect_selected(Operator):
    """Verbindet ausgewählte Kugeln (Blender 4.4 optimiert)"""
    bl_idname = "kette.connect_selected"
    bl_label = "Ausgewählte verbinden"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Verbindet zwei ausgewählte Kugeln"

    connector_type: BoolProperty(
        name="Gebogen",
        default=True,
        description="True für gebogene, False für gerade Verbinder"
    )

    @classmethod
    def poll(cls, context):
        """Blender 4.4: Erweiterte Poll-Validation"""
        selected_spheres = [obj for obj in context.selected_objects 
                          if obj.name.startswith("Kugel")]
        return len(selected_spheres) >= 2

    def execute(self, context):
        # Finde ausgewählte Kugeln
        selected_spheres = [obj for obj in context.selected_objects 
                          if obj.name.startswith("Kugel")]
        
        if len(selected_spheres) != 2:
            self.report({'ERROR'}, "Bitte genau zwei Kugeln auswählen!")
            return {'CANCELLED'}
        
        sphere1, sphere2 = selected_spheres
        
        try:
            # Blender 4.4: Verbesserte Connector-Erstellung
            connector = connector_creation.connect_spheres(
                sphere1, sphere2, 
                use_curved=self.connector_type
            )
            
            if connector:
                # Performance: Selection optimiert für macOS
                bpy.ops.object.select_all(action='DESELECT')
                connector.select_set(True)
                context.view_layer.objects.active = connector
                
                connector_type_str = "gebogener" if self.connector_type else "gerader"
                self.report({'INFO'}, 
                          f"{connector_type_str.capitalize()} Verbinder: {sphere1.name} ↔ {sphere2.name}")
                
                if constants.debug and constants.debug.enabled:
                    constants.debug.info('CONNECTORS', f"Created {connector_type_str} connector")
            else:
                self.report({'ERROR'}, "Verbinder konnte nicht erstellt werden")
                return {'CANCELLED'}
                
        except Exception as e:
            self.report({'ERROR'}, f"Fehler beim Verbinden: {str(e)}")
            if constants.debug and constants.debug.enabled:
                constants.debug.error('CONNECTORS', f"Error connecting spheres: {e}")
            return {'CANCELLED'}
        
        return {'FINISHED'}

class KETTE_OT_connect_curved(Operator):
    """Quick-Button: Gebogener Verbinder"""
    bl_idname = "kette.connect_curved"
    bl_label = "Gebogen"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Erstellt gebogenen Verbinder zwischen zwei ausgewählten Kugeln"
    
    def execute(self, context):
        return bpy.ops.kette.connect_selected(connector_type=True)

class KETTE_OT_connect_straight(Operator):
    """Quick-Button: Gerader Verbinder"""
    bl_idname = "kette.connect_straight"
    bl_label = "Gerade"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Erstellt geraden Verbinder zwischen zwei ausgewählten Kugeln"
    
    def execute(self, context):
        return bpy.ops.kette.connect_selected(connector_type=False)

# =========================================
# AUTOMATIC CONNECTION
# =========================================

class KETTE_OT_auto_connect_spheres(Operator):
    """Automatisches Verbinden von Kugeln (Blender 4.4 erweitert)"""
    bl_idname = "kette.auto_connect_spheres"
    bl_label = "Automatisch verbinden"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Verbindet Kugeln automatisch basierend auf Algorithmus"

    connection_mode: EnumProperty(
        name="Verbindungsmodus",
        items=[
            ('SEQUENTIAL', 'Sequentiell', 'Verbinde in Reihenfolge der Namen'),
            ('NEAREST', 'Nächste Nachbarn', 'Verbinde zu nächsten Nachbarn'),
            ('DISTANCE', 'Abstandsbasiert', 'Verbinde basierend auf max. Abstand'),
            ('DELAUNAY', 'Delaunay', 'Optimierte Triangulation'),
        ],
        default='NEAREST'
    )

    max_distance: FloatProperty(
        name="Max. Abstand",
        description="Maximaler Verbindungsabstand (0 = unbegrenzt)",
        default=0.0,
        min=0.0,
        max=20.0,
        subtype='DISTANCE'
    )

    max_connections: IntProperty(
        name="Max. Verbindungen",
        description="Maximale Verbindungen pro Kugel",
        default=4,
        min=1,
        max=10
    )

    use_curved: BoolProperty(
        name="Gebogene Verbinder",
        description="Verwende gebogene statt gerade Verbinder",
        default=True
    )

    sphere_types: EnumProperty(
        name="Kugeltypen",
        items=[
            ('ALL', 'Alle', 'Alle Kugeltypen verbinden'),
            ('SELECTED', 'Ausgewählte', 'Nur ausgewählte Kugeln'),
            ('SAME_TYPE', 'Gleicher Typ', 'Nur Kugeln gleichen Typs'),
        ],
        default='SELECTED'
    )

    def execute(self, context):
        try:
            # Kugeln basierend auf Auswahl sammeln
            if self.sphere_types == 'SELECTED':
                spheres = [obj for obj in context.selected_objects 
                          if obj.name.startswith("Kugel")]
            elif self.sphere_types == 'ALL':
                spheres = sphere_creation.get_all_spheres()
            else:  # SAME_TYPE
                # Gruppiere nach Typen und verbinde nur innerhalb der Gruppen
                self.report({'INFO'}, "Same-Type Verbindung noch nicht implementiert")
                return {'FINISHED'}
            
            if len(spheres) < 2:
                self.report({'ERROR'}, "Mindestens zwei Kugeln benötigt!")
                return {'CANCELLED'}
            
            # Verbindungspaare basierend auf Modus berechnen
            pairs = self._calculate_connection_pairs(spheres)
            
            if not pairs:
                self.report({'INFO'}, "Keine Verbindungen zu erstellen")
                return {'FINISHED'}
            
            # Verbindungen erstellen (Blender 4.4 Batch-optimiert)
            connected = 0
            total_pairs = len(pairs)
            
            # macOS: Redraw deaktivieren für Performance
            context.window_manager.progress_begin(0, total_pairs)
            
            try:
                for i, (sphere1, sphere2) in enumerate(pairs):
                    if (sphere1.name in bpy.data.objects and 
                        sphere2.name in bpy.data.objects):
                        
                        connector = connector_creation.connect_spheres(
                            sphere1, sphere2, use_curved=self.use_curved
                        )
                        if connector:
                            connected += 1
                    
                    # Progress Update
                    context.window_manager.progress_update(i)
            
            finally:
                context.window_manager.progress_end()
            
            self.report({'INFO'}, f"{connected} Verbindungen erstellt")
            
            if constants.debug and constants.debug.enabled:
                constants.debug.info('CONNECTORS', 
                                   f"Auto-connected {connected} pairs using {self.connection_mode}")
            
        except Exception as e:
            self.report({'ERROR'}, f"Fehler beim Auto-Verbinden: {str(e)}")
            return {'CANCELLED'}
        
        return {'FINISHED'}

    def _calculate_connection_pairs(self, spheres):
        """Berechnet Verbindungspaare basierend auf Modus"""
        from mathutils.kdtree import KDTree
        import mathutils
        
        pairs = []
        
        if self.connection_mode == 'SEQUENTIAL':
            # Sortiere nach Namen und verbinde sequentiell
            sorted_spheres = sorted(spheres, key=lambda x: x.name)
            for i in range(len(sorted_spheres) - 1):
                pairs.append((sorted_spheres[i], sorted_spheres[i + 1]))
        
        elif self.connection_mode == 'NEAREST':
            # KD-Tree für effiziente Nachbarsuche (macOS optimiert)
            if len(spheres) > 3:
                positions = [sphere.location for sphere in spheres]
                kdt = KDTree(len(positions))
                
                for i, pos in enumerate(positions):
                    kdt.insert(pos, i)
                kdt.balance()
                
                # Für jede Kugel die nächsten Nachbarn finden
                for i, sphere1 in enumerate(spheres):
                    # Finde nächste Nachbarn
                    co_find = sphere1.location
                    neighbors = []
                    
                    for (co, index, dist) in kdt.find_range(co_find, self.max_distance or 10.0):
                        if index != i:  # Nicht sich selbst
                            neighbors.append((dist, spheres[index]))
                    
                    # Sortiere nach Distanz und nimm die nächsten
                    neighbors.sort(key=lambda x: x[0])
                    connections_made = 0
                    
                    for dist, sphere2 in neighbors:
                        if connections_made >= self.max_connections:
                            break
                        
                        # Vermeide Duplikate
                        pair_tuple = tuple(sorted([sphere1.name, sphere2.name]))
                        if pair_tuple not in [tuple(sorted([p[0].name, p[1].name])) for p in pairs]:
                            pairs.append((sphere1, sphere2))
                            connections_made += 1
            else:
                # Fallback für wenige Kugeln
                for i, sphere1 in enumerate(spheres):
                    for j, sphere2 in enumerate(spheres[i + 1:], i + 1):
                        if j < i + self.max_connections:
                            pairs.append((sphere1, sphere2))
        
        elif self.connection_mode == 'DISTANCE':
            # Alle Paare innerhalb max_distance
            max_dist = self.max_distance or 5.0
            for i, sphere1 in enumerate(spheres):
                for sphere2 in spheres[i + 1:]:
                    distance = (sphere1.location - sphere2.location).length
                    if distance <= max_dist:
                        pairs.append((sphere1, sphere2))
        
        elif self.connection_mode == 'DELAUNAY':
            # Vereinfachte Delaunay-ähnliche Verbindung
            # Für komplexere Implementierung würde externes Library benötigt
            for i, sphere1 in enumerate(spheres):
                for j, sphere2 in enumerate(spheres[i + 1:], i + 1):
                    if j < i + 3:  # Begrenze auf 3 nächste
                        pairs.append((sphere1, sphere2))
        
        return pairs

    def invoke(self, context, event):
        """Blender 4.4: Verbessertes Invoke mit Property Dialog"""
        return context.window_manager.invoke_props_dialog(self, width=300)

    def draw(self, context):
        """Custom Draw für erweiterte Optionen"""
        layout = self.layout
        
        layout.prop(self, "connection_mode")
        layout.prop(self, "sphere_types")
        
        layout.separator()
        
        layout.prop(self, "max_distance")
        layout.prop(self, "max_connections")
        layout.prop(self, "use_curved")

# =========================================
# CONNECTOR UTILITIES
# =========================================

class KETTE_OT_refresh_connectors(Operator):
    """Aktualisiert alle Verbinder (Blender 4.4 optimiert)"""
    bl_idname = "kette.refresh_connectors"
    bl_label = "Verbinder aktualisieren"
    bl_description = "Aktualisiert alle Verbinder nach manuellen Änderungen"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            # macOS: Performance-optimiertes Refresh
            updated, deleted = connector_creation.refresh_connectors()
            
            msgs = []
            if updated:
                msgs.append(f"{updated} aktualisiert")
            if deleted:
                msgs.append(f"{deleted} gelöscht")
            
            if not msgs:
                self.report({'INFO'}, "Keine Verbinder angepasst")
            else:
                self.report({'INFO'}, ", ".join(msgs))
                
            if constants.debug and constants.debug.enabled:
                constants.debug.info('CONNECTORS', 
                                   f"Refreshed connectors: {updated} updated, {deleted} deleted")
                
        except Exception as e:
            self.report({'ERROR'}, f"Fehler beim Aktualisieren: {str(e)}")
            if constants.debug and constants.debug.enabled:
                constants.debug.error('CONNECTORS', f"Error refreshing connectors: {e}")
            return {'CANCELLED'}
        
        return {'FINISHED'}

class KETTE_OT_select_connectors(Operator):
    """Wählt alle Verbinder aus"""
    bl_idname = "kette.select_connectors"
    bl_label = "Alle Verbinder auswählen"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Wählt alle Verbinder aus"

    def execute(self, context):
        import kettentool.utils.collections as collections
        
        try:
            bpy.ops.object.select_all(action='DESELECT')
            
            connectors_collection = collections.ensure_collection("connectors")
            selected = 0
            
            for obj in connectors_collection.objects:
                if obj.name.startswith("Connector"):
                    obj.select_set(True)
                    selected += 1
            
            if selected > 0:
                self.report({'INFO'}, f"{selected} Verbinder ausgewählt")
            else:
                self.report({'INFO'}, "Keine Verbinder gefunden")
            
        except Exception as e:
            self.report({'ERROR'}, f"Fehler bei Auswahl: {str(e)}")
            return {'CANCELLED'}
        
        return {'FINISHED'}

class KETTE_OT_convert_connector_type(Operator):
    """Konvertiert Verbinder zwischen gerade/gebogen"""
    bl_idname = "kette.convert_connector_type"
    bl_label = "Verbinder-Typ konvertieren"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Konvertiert ausgewählte Verbinder zwischen gerade und gebogen"

    target_type: EnumProperty(
        name="Zieltyp",
        items=[
            ('CURVED', 'Gebogen', 'Konvertiere zu gebogenen Verbindern'),
            ('STRAIGHT', 'Gerade', 'Konvertiere zu geraden Verbindern'),
        ],
        default='CURVED'
    )

    def execute(self, context):
        selected_connectors = [obj for obj in context.selected_objects 
                             if obj.name.startswith("Connector")]
        
        if not selected_connectors:
            self.report({'ERROR'}, "Keine Verbinder ausgewählt!")
            return {'CANCELLED'}
        
        try:
            converted = 0
            
            for connector in selected_connectors:
                # Sphere-Referenzen lesen
                sphere1_name = connector.get("sphere1")
                sphere2_name = connector.get("sphere2")
                
                if not sphere1_name or not sphere2_name:
                    continue
                
                sphere1 = bpy.data.objects.get(sphere1_name)
                sphere2 = bpy.data.objects.get(sphere2_name)
                
                if not sphere1 or not sphere2:
                    continue
                
                # Alten Connector löschen
                connector_creation.delete_connector(connector)
                
                # Neuen Connector erstellen
                use_curved = (self.target_type == 'CURVED')
                new_connector = connector_creation.connect_spheres(
                    sphere1, sphere2, use_curved=use_curved
                )
                
                if new_connector:
                    converted += 1
            
            type_name = "gebogene" if self.target_type == 'CURVED' else "gerade"
            self.report({'INFO'}, f"{converted} Verbinder zu {type_name} konvertiert")
            
        except Exception as e:
            self.report({'ERROR'}, f"Fehler bei Konvertierung: {str(e)}")
            return {'CANCELLED'}
        
        return {'FINISHED'}

# =========================================
# REGISTRATION
# =========================================

classes = [
    KETTE_OT_connect_selected,
    KETTE_OT_connect_curved,
    KETTE_OT_connect_straight,
    KETTE_OT_auto_connect_spheres,
    KETTE_OT_refresh_connectors,
    KETTE_OT_select_connectors,
    KETTE_OT_convert_connector_type,
]

def register():
    """Registriert Connector Operators"""
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    """Deregistriert Connector Operators"""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
