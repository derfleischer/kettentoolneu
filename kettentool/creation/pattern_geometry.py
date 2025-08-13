"""
Pattern Geometry - Geometrie-Optimierung für 3D-Druck
Teil des Kettentool Pattern Systems
"""

import bpy
from mathutils import Vector
import math
from typing import List, Tuple, Dict, Set

# Import existing kettentool modules
import kettentool.creation.spheres as spheres
import kettentool.creation.connectors as connectors
import kettentool.core.constants as constants


class PatternGeometry:
    """Geometrie-Optimierung für Pattern Connections"""
    
    @staticmethod
    def optimize_for_3d_print(
        sphere_list: List[bpy.types.Object],
        connections: List[Tuple],
        props,
        settings: Dict
    ) -> Tuple[List, List]:
        """
        Optimiert Geometrie für 3D-Druck
        
        Args:
            sphere_list: Liste von Kugel-Objekten
            connections: Liste von Verbindungs-Tupeln
            props: Chain Properties
            settings: Optimierungs-Einstellungen
            
        Returns:
            (optimierte_kugeln, optimierte_verbindungen)
        """
        
        # Extract settings
        max_length = settings.get('max_connection_length', 5.0)
        add_intermediate = settings.get('add_intermediate_spheres', True)
        intermediate_scale = settings.get('intermediate_sphere_scale', 0.8)
        ensure_angles = settings.get('ensure_minimum_angles', True)
        min_angle = settings.get('minimum_angle', 30.0)
        merge_close = settings.get('merge_close_spheres', True)
        merge_threshold = settings.get('merge_threshold', 1.5)
        
        # Get sphere radius from props
        sphere_radius = props.kette_kugelradius if props else 0.5
        
        new_spheres = []
        optimized_connections = []
        
        # Process each connection
        for s1, s2 in connections:
            distance = (s2.location - s1.location).length
            
            # Check if intermediate spheres needed
            if add_intermediate and distance > max_length:
                # Add intermediate spheres
                intermediates = PatternGeometry.create_intermediate_spheres(
                    s1, s2, 
                    distance / max_length,  # number needed
                    sphere_radius * intermediate_scale,
                    props
                )
                
                # Create chain of connections
                chain = [s1] + intermediates + [s2]
                for i in range(len(chain) - 1):
                    optimized_connections.append((chain[i], chain[i + 1]))
                
                new_spheres.extend(intermediates)
            else:
                # Keep original connection
                optimized_connections.append((s1, s2))
        
        # Combine all spheres
        all_spheres = sphere_list + new_spheres
        
        # Merge close spheres if requested
        if merge_close:
            all_spheres = PatternGeometry.merge_nearby_spheres(
                all_spheres,
                sphere_radius * merge_threshold
            )
            # Update connections after merge
            optimized_connections = PatternGeometry.validate_connections(
                optimized_connections, all_spheres
            )
        
        # Ensure minimum angles
        if ensure_angles:
            optimized_connections = PatternGeometry.filter_by_angle(
                all_spheres, optimized_connections, min_angle
            )
        
        return all_spheres, optimized_connections
    
    @staticmethod
    def create_intermediate_spheres(
        s1: bpy.types.Object,
        s2: bpy.types.Object,
        count: float,
        radius: float,
        props
    ) -> List[bpy.types.Object]:
        """
        Erstellt Zwischen-Kugeln entlang einer Verbindung
        
        Args:
            s1, s2: Start- und End-Kugel
            count: Anzahl Zwischen-Kugeln (wird gerundet)
            radius: Radius der Zwischen-Kugeln
            props: Chain Properties
            
        Returns:
            Liste der erstellten Zwischen-Kugeln
        """
        num_spheres = max(1, int(count))
        intermediate = []
        
        start = s1.location
        end = s2.location
        direction = end - start
        
        # Add slight curve for organic look
        perpendicular = direction.normalized().cross(Vector((0, 0, 1)))
        if perpendicular.length < 0.01:
            perpendicular = direction.normalized().cross(Vector((1, 0, 0)))
        perpendicular.normalize()
        
        for i in range(1, num_spheres + 1):
            t = i / (num_spheres + 1)
            
            # Linear interpolation
            position = start.lerp(end, t)
            
            # Add subtle curve
            curve_offset = math.sin(t * math.pi) * radius * 0.3
            position += perpendicular * curve_offset
            
            # Create sphere using existing sphere creation
            sphere = spheres.create_sphere(position, radius)
            
            if sphere:
                # Mark as intermediate
                sphere["intermediate"] = True
                sphere["print_optimized"] = True
                sphere.name = f"Kugel_Inter_{len(intermediate):03d}"
                
                intermediate.append(sphere)
        
        if constants.debug:
            constants.debug.trace('PATTERN', 
                f"Created {len(intermediate)} intermediate spheres")
        
        return intermediate
    
    @staticmethod
    def merge_nearby_spheres(
        sphere_list: List[bpy.types.Object],
        threshold: float
    ) -> List[bpy.types.Object]:
        """
        Verschmilzt zu nahe Kugeln
        
        Args:
            sphere_list: Liste von Kugeln
            threshold: Minimaler Abstand
            
        Returns:
            Bereinigte Liste von Kugeln
        """
        merged = []
        processed = set()
        
        for sphere in sphere_list:
            if sphere in processed or not sphere.name in bpy.data.objects:
                continue
            
            # Find nearby spheres
            nearby = []
            for other in sphere_list:
                if (other != sphere and 
                    other not in processed and 
                    other.name in bpy.data.objects):
                    
                    if (sphere.location - other.location).length < threshold:
                        nearby.append(other)
                        processed.add(other)
            
            if nearby:
                # Calculate merged position
                positions = [sphere.location] + [s.location for s in nearby]
                center = sum(positions, Vector()) / len(positions)
                
                # Update main sphere
                sphere.location = center
                
                # Slightly increase size
                if not sphere.get("intermediate"):
                    sphere.scale *= 1.1
                
                # Remove nearby spheres
                for other in nearby:
                    bpy.data.objects.remove(other, do_unlink=True)
            
            merged.append(sphere)
            processed.add(sphere)
        
        return merged
    
    @staticmethod
    def filter_by_angle(
        sphere_list: List[bpy.types.Object],
        connections: List[Tuple],
        min_angle_deg: float
    ) -> List[Tuple]:
        """
        Filtert Verbindungen mit zu spitzen Winkeln
        
        Args:
            sphere_list: Liste von Kugeln
            connections: Verbindungen
            min_angle_deg: Minimaler Winkel in Grad
            
        Returns:
            Gefilterte Verbindungen
        """
        min_angle = math.radians(min_angle_deg)
        filtered = []
        
        # Build connectivity map
        connectivity = {s: [] for s in sphere_list}
        for s1, s2 in connections:
            if s1 in sphere_list and s2 in sphere_list:
                connectivity[s1].append(s2)
                connectivity[s2].append(s1)
        
        # Check each connection
        for s1, s2 in connections:
            if s1 not in sphere_list or s2 not in sphere_list:
                continue
                
            keep = True
            
            # Check angles at both ends
            for other in connectivity[s1]:
                if other != s2:
                    v1 = (s2.location - s1.location).normalized()
                    v2 = (other.location - s1.location).normalized()
                    angle = math.acos(min(max(v1.dot(v2), -1), 1))
                    
                    if angle < min_angle:
                        keep = False
                        break
            
            if keep:
                filtered.append((s1, s2))
        
        return filtered
    
    @staticmethod
    def validate_connections(
        connections: List[Tuple],
        valid_spheres: List[bpy.types.Object]
    ) -> List[Tuple]:
        """
        Validiert Verbindungen nach Sphere-Merge
        
        Args:
            connections: Original-Verbindungen
            valid_spheres: Gültige Kugeln nach Merge
            
        Returns:
            Gültige Verbindungen
        """
        valid_names = set(s.name for s in valid_spheres if s.name in bpy.data.objects)
        validated = []
        
        for s1, s2 in connections:
            if (s1.name in valid_names and 
                s2.name in valid_names and
                s1.name in bpy.data.objects and
                s2.name in bpy.data.objects):
                validated.append((s1, s2))
        
        return validated
    
    @staticmethod
    def check_printability(
        sphere_list: List[bpy.types.Object],
        connections: List[Tuple],
        settings: Dict
    ) -> List[str]:
        """
        Prüft Druckbarkeit der Geometrie
        
        Args:
            sphere_list: Kugeln
            connections: Verbindungen
            settings: Druckeinstellungen
            
        Returns:
            Liste von Warnungen
        """
        issues = []
        
        min_wall = settings.get('min_wall_thickness', 1.5)
        max_overhang = settings.get('support_angle_threshold', 45.0)
        
        # Check minimum sizes
        for sphere in sphere_list:
            radius = sphere.dimensions[0] / 2
            if radius < min_wall / 2:
                issues.append(f"Kugel '{sphere.name}' zu klein ({radius:.1f}mm)")
                break  # Only report once
        
        # Check overhangs
        for s1, s2 in connections:
            direction = (s2.location - s1.location).normalized()
            angle_to_vertical = math.degrees(
                math.acos(min(max(abs(direction.z), 0), 1))
            )
            
            if angle_to_vertical > max_overhang:
                issues.append(f"Überhang {angle_to_vertical:.0f}° (Support nötig)")
                break  # Only report once
        
        # Check connectivity
        if not PatternGeometry.is_fully_connected(sphere_list, connections):
            issues.append("Nicht alle Teile verbunden")
        
        return issues
    
    @staticmethod
    def is_fully_connected(
        sphere_list: List[bpy.types.Object],
        connections: List[Tuple]
    ) -> bool:
        """
        Prüft ob alle Kugeln verbunden sind
        
        Args:
            sphere_list: Kugeln
            connections: Verbindungen
            
        Returns:
            True wenn vollständig verbunden
        """
        if not sphere_list:
            return True
        
        # Build adjacency
        adjacency = {s: set() for s in sphere_list}
        for s1, s2 in connections:
            if s1 in sphere_list and s2 in sphere_list:
                adjacency[s1].add(s2)
                adjacency[s2].add(s1)
        
        # BFS
        visited = set()
        queue = [sphere_list[0]]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            queue.extend(adjacency[current] - visited)
        
        return len(visited) == len(sphere_list)
    
    @staticmethod
    def calculate_optimal_thickness(
        s1: bpy.types.Object,
        s2: bpy.types.Object,
        base_thickness: float,
        max_length: float
    ) -> float:
        """
        Berechnet optimale Connector-Dicke basierend auf Länge
        
        Args:
            s1, s2: Verbundene Kugeln
            base_thickness: Basis-Dicke
            max_length: Maximale Länge für Skalierung
            
        Returns:
            Optimierte Dicke
        """
        distance = (s2.location - s1.location).length
        
        # Scale thickness based on length
        length_factor = min(distance / max_length, 2.0)
        thickness = base_thickness * (1.0 + length_factor * 0.3)
        
        # Consider sphere sizes
        avg_sphere_size = (s1.dimensions[0] + s2.dimensions[0]) / 4
        thickness = min(thickness, avg_sphere_size * 0.8)
        
        return thickness
