"""
Pattern Connector Operator - Vollständig integriert mit Kettentool
"""

import bpy
from bpy.types import Operator
from bpy.props import (
    EnumProperty, FloatProperty, BoolProperty, 
    IntProperty, StringProperty
)
from mathutils import Vector
from typing import List, Tuple, Set
import math

# Import kettentool modules
import kettentool.creation.connectors as connectors
import kettentool.creation.spheres as spheres
import kettentool.creation.pattern_geometry as pattern_geometry
import kettentool.core.constants as constants


class KETTE_OT_apply_pattern(Operator):
    """Wendet Verbindungsmuster auf Kugeln an - optimiert für 3D-Druck"""
    bl_idname = "kette.apply_pattern"
    bl_label = "Pattern anwenden"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Verbindet Kugeln mit verschiedenen Mustern"
    
    # =========================================
    # PATTERN SELECTION
    # =========================================
    
    pattern_type: EnumProperty(
        name="Muster",
        description="Verbindungsmuster",
        items=[
            ('LINEAR', 'Linear', 'Einfache Kette', 'IPO_LINEAR', 0),
            ('CLOSED', 'Geschlossen', 'Geschlossene Kette', 'CURVE_BEZCIRCLE', 1),
            ('ZIGZAG', 'Zickzack', 'Zickzack-Muster', 'IPO_SINE', 2),
            ('STAR', 'Stern', 'Sternförmige Verbindungen', 'LIGHT_SUN', 3),
            ('TREE', 'Baum', 'Binäre Baumstruktur', 'OUTLINER', 4),
            ('WEB', 'Netz', 'Alle innerhalb Distanz', 'UV_SYNC_SELECT', 5),
            ('TRIANGULATE', 'Triangulation', 'Delaunay Triangulation', 'MESH_DATA', 6),
            ('VORONOI', 'Voronoi', 'Voronoi-Diagramm', 'TEXTURE', 7),
            ('GRID', 'Gitter', 'Regelmäßiges Gitter', 'MESH_GRID', 8),
            ('SPIRAL', 'Spirale', 'Spiralförmig', 'CURVE_SPIRALS', 9),
            ('MST', 'Min. Spannbaum', 'Minimal Spanning Tree', 'PHYSICS', 10),
            ('NEAREST_N', 'Nächste N', 'N nächste Nachbarn', 'CONSTRAINT', 11),
        ],
        default='LINEAR'
    )
    
    # =========================================
    # GENERAL SETTINGS
    # =========================================
    
    apply_to_all: BoolProperty(
        name="Auf alle anwenden",
        description="Wende Muster auf alle Kugeln an (nicht nur ausgewählte)",
        default=False
    )
    
    remove_existing: BoolProperty(
        name="Bestehende entfernen",
        description="Entferne bestehende Verbindungen",
        default=True
    )
    
    # Pattern specific
    max_distance: FloatProperty(
        name="Max. Distanz",
        description="Maximale Verbindungsdistanz",
        default=10.0,
        min=0.0,
        max=100.0,
        unit='LENGTH'
    )
    
    nearest_count: IntProperty(
        name="Nachbar-Anzahl",
        description="Anzahl nächster Nachbarn",
        default=3,
        min=1,
        max=10
    )
    
    # =========================================
    # 3D PRINT OPTIMIZATION
    # =========================================
    
    optimize_for_print: BoolProperty(
        name="3D-Druck Optimierung",
        description="Optimiere Geometrie für 3D-Druck",
        default=False
    )
    
    add_intermediate_spheres: BoolProperty(
        name="Zwischen-Kugeln",
        description="Füge Kugeln bei langen Verbindungen ein",
        default=True
    )
    
    max_connection_length: FloatProperty(
        name="Max. Verbindungslänge",
        description="Maximale Länge ohne Zwischen-Kugel",
        default=5.0,
        min=1.0,
        max=50.0,
        unit='LENGTH'
    )
    
    intermediate_sphere_scale: FloatProperty(
        name="Zwischen-Kugel Größe",
        description="Größenfaktor für Zwischen-Kugeln",
        default=0.8,
        min=0.5,
        max=1.5,
        subtype='FACTOR'
    )
    
    ensure_minimum_angles: BoolProperty(
        name="Mindestwinkel",
        description="Verhindere zu spitze Winkel",
        default=True
    )
    
    minimum_angle: FloatProperty(
        name="Min. Winkel",
        description="Minimaler Winkel zwischen Verbindungen",
        default=30.0,
        min=15.0,
        max=90.0,
        subtype='ANGLE'
    )
    
    merge_close_spheres: BoolProperty(
        name="Nahe verschmelzen",
        description="Verschmelze zu nahe Kugeln",
        default=False
    )
    
    merge_threshold: FloatProperty(
        name="Verschmelz-Abstand",
        description="Minimaler Abstand (Faktor)",
        default=1.5,
        min=0.5,
        max=5.0,
        subtype='FACTOR'
    )
    
    # =========================================
    # PRINT VALIDATION
    # =========================================
    
    check_printability: BoolProperty(
        name="Druckbarkeit prüfen",
        description="Prüfe Geometrie auf Druckbarkeit",
        default=False
    )
    
    min_wall_thickness: FloatProperty(
        name="Min. Wandstärke",
        description="Minimale Wandstärke (mm)",
        default=1.5,
        min=0.5,
        max=5.0,
        unit='LENGTH'
    )
    
    support_angle_threshold: FloatProperty(
        name="Support-Winkel",
        description="Max. Überhang ohne Support",
        default=45.0,
        min=30.0,
        max=90.0,
        subtype='ANGLE'
    )
    
    @classmethod
    def poll(cls, context):
        """Check if we can run"""
        all_spheres = [obj for obj in bpy.data.objects 
                      if obj.name.startswith("Kugel")]
        return len(all_spheres) >= 2
    
    def execute(self, context):
        """Execute pattern connection"""
        props = context.scene.chain_props
        
        # Get spheres
        if self.apply_to_all:
            sphere_list = [obj for obj in bpy.data.objects 
                          if obj.name.startswith("Kugel")]
        else:
            sphere_list = [obj for obj in context.selected_objects 
                          if obj.name.startswith("Kugel")]
        
        if len(sphere_list) < 2:
            self.report({'ERROR'}, "Mindestens 2 Kugeln benötigt!")
            return {'CANCELLED'}
        
        # Remove existing if requested
        if self.remove_existing:
            self.remove_existing_connections(sphere_list)
        
        # Get pattern connections
        connections = self.get_pattern_connections(sphere_list)
        
        # Filter by distance if needed
        if self.max_distance > 0 and self.pattern_type in ['WEB', 'GRID']:
            connections = self.filter_by_distance(connections, self.max_distance)
        
        # Optimize for 3D print if requested
        if self.optimize_for_print:
            settings = {
                'max_connection_length': self.max_connection_length,
                'add_intermediate_spheres': self.add_intermediate_spheres,
                'intermediate_sphere_scale': self.intermediate_sphere_scale,
                'ensure_minimum_angles': self.ensure_minimum_angles,
                'minimum_angle': self.minimum_angle,
                'merge_close_spheres': self.merge_close_spheres,
                'merge_threshold': self.merge_threshold,
            }
            
            # Use pattern_geometry module for optimization
            geometry = pattern_geometry.PatternGeometry()
            sphere_list, connections = geometry.optimize_for_3d_print(
                sphere_list, connections, props, settings
            )
        
        # Create physical connections
        created = self.create_connections(sphere_list, connections, props)
        
        # Check printability if requested
        if self.check_printability:
            settings = {
                'min_wall_thickness': self.min_wall_thickness,
                'support_angle_threshold': self.support_angle_threshold,
            }
            
            geometry = pattern_geometry.PatternGeometry()
            issues = geometry.check_printability(sphere_list, connections, settings)
            
            if issues:
                self.report({'WARNING'}, f"Druckbarkeit: {', '.join(issues)}")
        
        self.report({'INFO'}, 
                   f"Erstellt: {created} Verbindungen, "
                   f"{len(sphere_list)} Kugeln total")
        
        return {'FINISHED'}
    
    def invoke(self, context, event):
        """Show dialog"""
        return context.window_manager.invoke_props_dialog(self, width=350)
    
    def draw(self, context):
        """Draw UI"""
        layout = self.layout
        
        # Pattern selection
        layout.prop(self, "pattern_type")
        
        # Pattern specific settings
        if self.pattern_type == 'NEAREST_N':
            layout.prop(self, "nearest_count")
        
        if self.pattern_type in ['WEB', 'GRID']:
            layout.prop(self, "max_distance")
        
        # General settings
        layout.separator()
        row = layout.row()
        row.prop(self, "apply_to_all")
        row.prop(self, "remove_existing")
        
        # 3D Print optimization
        layout.separator()
        box = layout.box()
        box.prop(self, "optimize_for_print", icon='MOD_REMESH')
        
        if self.optimize_for_print:
            col = box.column(align=True)
            col.prop(self, "add_intermediate_spheres")
            if self.add_intermediate_spheres:
                col.prop(self, "max_connection_length")
                col.prop(self, "intermediate_sphere_scale")
            
            col.separator()
            col.prop(self, "ensure_minimum_angles")
            if self.ensure_minimum_angles:
                col.prop(self, "minimum_angle")
            
            col.separator()
            col.prop(self, "merge_close_spheres")
            if self.merge_close_spheres:
                col.prop(self, "merge_threshold")
        
        # Printability check
        if self.optimize_for_print:
            layout.separator()
            box = layout.box()
            box.prop(self, "check_printability", icon='INFO')
            if self.check_printability:
                col = box.column(align=True)
                col.prop(self, "min_wall_thickness")
                col.prop(self, "support_angle_threshold")
        
        # Info
        layout.separator()
        spheres_count = len([obj for obj in bpy.data.objects 
                           if obj.name.startswith("Kugel")])
        selected_count = len([obj for obj in context.selected_objects 
                            if obj.name.startswith("Kugel")])
        
        if self.apply_to_all:
            layout.label(text=f"Wird auf {spheres_count} Kugeln angewendet", icon='INFO')
        else:
            layout.label(text=f"Wird auf {selected_count} Kugeln angewendet", icon='INFO')
    
    # =========================================
    # PATTERN IMPLEMENTATIONS
    # =========================================
    
    def get_pattern_connections(self, spheres: List) -> List[Tuple]:
        """Get connections based on pattern"""
        
        if self.pattern_type == 'LINEAR':
            return self.pattern_linear(spheres)
        elif self.pattern_type == 'CLOSED':
            return self.pattern_closed(spheres)
        elif self.pattern_type == 'ZIGZAG':
            return self.pattern_zigzag(spheres)
        elif self.pattern_type == 'STAR':
            return self.pattern_star(spheres)
        elif self.pattern_type == 'TREE':
            return self.pattern_tree(spheres)
        elif self.pattern_type == 'WEB':
            return self.pattern_web(spheres)
        elif self.pattern_type == 'TRIANGULATE':
            return self.pattern_triangulate(spheres)
        elif self.pattern_type == 'VORONOI':
            return self.pattern_voronoi(spheres)
        elif self.pattern_type == 'GRID':
            return self.pattern_grid(spheres)
        elif self.pattern_type == 'SPIRAL':
            return self.pattern_spiral(spheres)
        elif self.pattern_type == 'MST':
            return self.pattern_mst(spheres)
        elif self.pattern_type == 'NEAREST_N':
            return self.pattern_nearest_n(spheres, self.nearest_count)
        
        return []
    
    def pattern_linear(self, spheres: List) -> List[Tuple]:
        """Linear chain"""
        connections = []
        spheres_sorted = sorted(spheres, key=lambda s: s.name)
        for i in range(len(spheres_sorted) - 1):
            connections.append((spheres_sorted[i], spheres_sorted[i + 1]))
        return connections
    
    def pattern_closed(self, spheres: List) -> List[Tuple]:
        """Closed loop"""
        connections = self.pattern_linear(spheres)
        if len(spheres) > 2:
            spheres_sorted = sorted(spheres, key=lambda s: s.name)
            connections.append((spheres_sorted[-1], spheres_sorted[0]))
        return connections
    
    def pattern_zigzag(self, spheres: List) -> List[Tuple]:
        """Zigzag pattern"""
        connections = []
        spheres_sorted = sorted(spheres, key=lambda s: s.location.x)
        
        for i in range(len(spheres_sorted) - 1):
            connections.append((spheres_sorted[i], spheres_sorted[i + 1]))
            if i % 2 == 0 and i + 2 < len(spheres_sorted):
                connections.append((spheres_sorted[i], spheres_sorted[i + 2]))
        
        return connections
    
    def pattern_star(self, spheres: List) -> List[Tuple]:
        """Star from center"""
        connections = []
        center_pos = sum([s.location for s in spheres], Vector()) / len(spheres)
        center = min(spheres, key=lambda s: (s.location - center_pos).length)
        
        for sphere in spheres:
            if sphere != center:
                connections.append((center, sphere))
        
        return connections
    
    def pattern_tree(self, spheres: List) -> List[Tuple]:
        """Binary tree"""
        connections = []
        spheres_sorted = sorted(spheres, key=lambda s: s.name)
        
        for i, sphere in enumerate(spheres_sorted):
            left_idx = 2 * i + 1
            right_idx = 2 * i + 2
            
            if left_idx < len(spheres_sorted):
                connections.append((sphere, spheres_sorted[left_idx]))
            if right_idx < len(spheres_sorted):
                connections.append((sphere, spheres_sorted[right_idx]))
        
        return connections
    
    def pattern_web(self, spheres: List) -> List[Tuple]:
        """Web connections"""
        connections = []
        max_dist = self.max_distance if self.max_distance > 0 else float('inf')
        
        for i, s1 in enumerate(spheres):
            for s2 in spheres[i + 1:]:
                if (s1.location - s2.location).length <= max_dist:
                    connections.append((s1, s2))
        
        return connections
    
    def pattern_triangulate(self, spheres: List) -> List[Tuple]:
        """Delaunay triangulation"""
        connections = set()
        points_2d = [(s.location.x, s.location.y) for s in spheres]
        
        try:
            from mathutils.geometry import delaunay_2d_cdt
            
            result = delaunay_2d_cdt(points_2d, [], [], 0, 0.0001)
            
            if result:
                vertices_out, edges_out, faces_out = result
                
                for face in faces_out:
                    for i in range(len(face)):
                        v1 = face[i]
                        v2 = face[(i + 1) % len(face)]
                        if v1 < len(spheres) and v2 < len(spheres):
                            connections.add(tuple(sorted([spheres[v1], spheres[v2]], 
                                                        key=lambda x: x.name)))
        except Exception as e:
            self.report({'WARNING'}, f"Triangulation failed: {e}")
            return self.pattern_web(spheres)
        
        return list(connections)
    
    def pattern_voronoi(self, spheres: List) -> List[Tuple]:
        """Voronoi diagram"""
        connections = []
        
        for i, s1 in enumerate(spheres):
            for s2 in spheres[i + 1:]:
                midpoint = (s1.location + s2.location) / 2
                dist_to_mid = (s1.location - midpoint).length
                
                is_neighbor = True
                for other in spheres:
                    if other != s1 and other != s2:
                        if (other.location - midpoint).length < dist_to_mid - 0.001:
                            is_neighbor = False
                            break
                
                if is_neighbor:
                    connections.append((s1, s2))
        
        return connections
    
    def pattern_grid(self, spheres: List) -> List[Tuple]:
        """Grid connections"""
        connections = []
        threshold = self.max_distance if self.max_distance > 0 else 5.0
        
        for sphere in spheres:
            for other in spheres:
                if other != sphere:
                    diff = other.location - sphere.location
                    
                    # Horizontal
                    if abs(diff.y) < threshold * 0.3 and abs(diff.x) < threshold:
                        if diff.x > 0:
                            connections.append((sphere, other))
                    # Vertical
                    elif abs(diff.x) < threshold * 0.3 and abs(diff.y) < threshold:
                        if diff.y > 0:
                            connections.append((sphere, other))
        
        return connections
    
    def pattern_spiral(self, spheres: List) -> List[Tuple]:
        """Spiral pattern"""
        connections = []
        center = sum([s.location for s in spheres], Vector()) / len(spheres)
        
        def get_angle(sphere):
            diff = sphere.location - center
            return math.atan2(diff.y, diff.x)
        
        spheres_sorted = sorted(spheres, key=get_angle)
        
        for i in range(len(spheres_sorted)):
            next_idx = (i + 1) % len(spheres_sorted)
            connections.append((spheres_sorted[i], spheres_sorted[next_idx]))
            
            if i + 3 < len(spheres_sorted):
                connections.append((spheres_sorted[i], spheres_sorted[i + 3]))
        
        return connections
    
    def pattern_mst(self, spheres: List) -> List[Tuple]:
        """Minimal spanning tree"""
        connections = []
        
        # Create edges with weights
        edges = []
        for i, s1 in enumerate(spheres):
            for s2 in spheres[i + 1:]:
                dist = (s1.location - s2.location).length
                edges.append((dist, s1, s2))
        
        edges.sort(key=lambda x: x[0])
        
        # Kruskal's algorithm
        parent = {s: s for s in spheres}
        
        def find(s):
            if parent[s] != s:
                parent[s] = find(parent[s])
            return parent[s]
        
        def union(s1, s2):
            root1, root2 = find(s1), find(s2)
            if root1 != root2:
                parent[root1] = root2
                return True
            return False
        
        for dist, s1, s2 in edges:
            if union(s1, s2):
                connections.append((s1, s2))
                if len(connections) == len(spheres) - 1:
                    break
        
        return connections
    
    def pattern_nearest_n(self, spheres: List, n: int) -> List[Tuple]:
        """Connect to N nearest"""
        connections = set()
        
        for sphere in spheres:
            distances = []
            for other in spheres:
                if other != sphere:
                    dist = (other.location - sphere.location).length
                    distances.append((dist, other))
            
            distances.sort(key=lambda x: x[0])
            
            for i in range(min(n, len(distances))):
                connection = tuple(sorted([sphere, distances[i][1]], 
                                        key=lambda x: x.name))
                connections.add(connection)
        
        return list(connections)
    
    # =========================================
    # HELPER METHODS
    # =========================================
    
    def create_connections(self, sphere_list, connections, props):
        """Create physical connections using existing connector system"""
        created = 0
        
        for s1, s2 in connections:
            if not self.are_connected(s1, s2):
                # Calculate optimal thickness if optimizing
                if self.optimize_for_print:
                    geometry = pattern_geometry.PatternGeometry()
                    thickness = geometry.calculate_optimal_thickness(
                        s1, s2,
                        props.kette_kugelradius * props.kette_connector_radius_factor,
                        self.max_connection_length
                    )
                    
                    # Temporarily set thickness
                    original_factor = props.kette_connector_radius_factor
                    props.kette_connector_radius_factor = thickness / props.kette_kugelradius
                
                # Use existing connector creation
                use_curved = props.kette_use_curved
                if self.optimize_for_print:
                    # Use curved for longer connections
                    distance = (s2.location - s1.location).length
                    use_curved = distance > self.max_connection_length * 0.7
                
                connector = connectors.connect_spheres(s1, s2, use_curved, props)
                
                if self.optimize_for_print:
                    # Restore original factor
                    props.kette_connector_radius_factor = original_factor
                
                if connector:
                    connector["pattern_type"] = self.pattern_type
                    created += 1
        
        return created
    
    def remove_existing_connections(self, spheres):
        """Remove existing connections between spheres"""
        sphere_names = set(s.name for s in spheres)
        
        for obj in list(bpy.data.objects):
            if "Connector" in obj.name or "Verbinder" in obj.name:
                s1 = obj.get("sphere1")
                s2 = obj.get("sphere2")
                if s1 in sphere_names and s2 in sphere_names:
                    bpy.data.objects.remove(obj, do_unlink=True)
    
    def are_connected(self, s1, s2):
        """Check if spheres are connected"""
        for obj in bpy.data.objects:
            if "Connector" in obj.name or "Verbinder" in obj.name:
                sphere1 = obj.get("sphere1")
                sphere2 = obj.get("sphere2")
                if sphere1 and sphere2:
                    if ((sphere1 == s1.name and sphere2 == s2.name) or 
                        (sphere1 == s2.name and sphere2 == s1.name)):
                        return True
        return False
    
    def filter_by_distance(self, connections, max_dist):
        """Filter connections by distance"""
        return [(s1, s2) for s1, s2 in connections 
                if (s1.location - s2.location).length <= max_dist]


# =========================================
# QUICK ACCESS OPERATORS
# =========================================

class KETTE_OT_pattern_quick(Operator):
    """Quick pattern application"""
    bl_idname = "kette.pattern_quick"
    bl_label = "Quick Pattern"
    bl_options = {'REGISTER', 'UNDO'}
    
    pattern: StringProperty()
    
    def execute(self, context):
        bpy.ops.kette.apply_pattern(pattern_type=self.pattern)
        return {'FINISHED'}


# =========================================
# REGISTRATION
# =========================================

classes = [
    KETTE_OT_apply_pattern,
    KETTE_OT_pattern_quick,
]

def register():
    """Register pattern operators"""
    for cls in classes:
        bpy.utils.register_class(cls)
    
    if constants.debug:
        constants.debug.info('OPERATORS', "Pattern connector registered")

def unregister():
    """Unregister pattern operators"""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
