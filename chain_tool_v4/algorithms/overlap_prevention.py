"""
Chain Tool V4 - Overlap Prevention Module
=========================================

BVH-Tree basierte Kollisionserkennung für Orthesen-Verstärkungen
- Effiziente Überlappungsprüfung zwischen Spheres und Verbindungen
- Automatische Konfliktlösung durch adaptive Repositionierung
- Multi-Layer Kollisionserkennung (Spheres, Connectors, Surface)
- Integration mit Aura-System für optimale Performance

Author: Chain Tool V4 Development Team  
Version: 4.0.0
"""

import bpy
import bmesh
import numpy as np
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from mathutils.geometry import intersect_line_sphere, intersect_sphere_sphere_3d
from typing import List, Tuple, Dict, Optional, Set, Any, Union
import time
from enum import Enum
from dataclasses import dataclass, field

# V4 Module imports
from ..core.constants import CollisionType, ResolutionStrategy
from ..core.state_manager import StateManager
from ..utils.performance import measure_time, PerformanceTracker
from ..utils.debug import DebugManager
from ..utils.caching import CacheManager
from ..utils.math_utils import safe_normalize, calculate_distance_3d

# Initialize systems
debug = DebugManager()
cache = CacheManager()
perf = PerformanceTracker()

class CollisionType(Enum):
    """Typen von Kollisionen"""
    SPHERE_SPHERE = "sphere_sphere"           # Sphere-Sphere Überlappung
    SPHERE_CONNECTOR = "sphere_connector"     # Sphere-Verbindung Überlappung
    CONNECTOR_CONNECTOR = "connector_connector" # Verbindung-Verbindung Kreuzung
    SPHERE_SURFACE = "sphere_surface"         # Sphere durchdringt Oberfläche
    CONNECTOR_SURFACE = "connector_surface"   # Verbindung durchdringt Oberfläche

class ResolutionStrategy(Enum):
    """Strategien zur Konfliktlösung"""
    MOVE_SMALLEST = "move_smallest"           # Bewege kleineres Objekt
    MOVE_BOTH = "move_both"                  # Bewege beide Objekte anteilig
    RESIZE_SMALLER = "resize_smaller"         # Verkleinere Objekte
    REMOVE_WEAKEST = "remove_weakest"        # Entferne schwächste Verbindung
    REROUTE_CONNECTION = "reroute_connection" # Leite Verbindung um
    SURFACE_PROJECT = "surface_project"       # Projiziere auf Oberfläche

@dataclass
class CollisionInfo:
    """Information über eine erkannte Kollision"""
    type: CollisionType
    objects: Tuple[Any, ...]  # Beteiligte Objekte
    overlap_amount: float     # Stärke der Überlappung
    collision_point: Vector   # Punkt der Kollision
    resolution_priority: float = 1.0  # Priorität für Auflösung
    suggested_strategy: Optional[ResolutionStrategy] = None

@dataclass  
class SpatialHashCell:
    """Zelle im Spatial Hash Grid"""
    objects: Set[int] = field(default_factory=set)
    spheres: Set[int] = field(default_factory=set)
    connectors: Set[int] = field(default_factory=set)

class SpatialHashGrid:
    """Spatial Hash Grid für effiziente Kollisionserkennung"""
    
    def __init__(self, cell_size: float = 10.0):
        self.cell_size = cell_size
        self.grid: Dict[Tuple[int, int, int], SpatialHashCell] = {}
        self.object_positions: Dict[int, Vector] = {}
        
    def _hash_position(self, position: Vector) -> Tuple[int, int, int]:
        """Berechnet Grid-Koordinaten für Position"""
        return (
            int(position.x // self.cell_size),
            int(position.y // self.cell_size),
            int(position.z // self.cell_size)
        )
    
    def add_sphere(self, sphere_id: int, position: Vector, radius: float):
        """Fügt Sphere zum Grid hinzu"""
        self.object_positions[sphere_id] = position
        
        # Bestimme alle Zellen die die Sphere überlappen könnte
        cells_to_update = set()
        
        # Hauptzelle
        main_cell = self._hash_position(position)
        cells_to_update.add(main_cell)
        
        # Nachbarzellen basierend auf Radius
        cell_radius = int(np.ceil(radius / self.cell_size))
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                for dz in range(-cell_radius, cell_radius + 1):
                    cell_coord = (
                        main_cell[0] + dx,
                        main_cell[1] + dy, 
                        main_cell[2] + dz
                    )
                    cells_to_update.add(cell_coord)
        
        # Update Grid
        for cell_coord in cells_to_update:
            if cell_coord not in self.grid:
                self.grid[cell_coord] = SpatialHashCell()
            self.grid[cell_coord].objects.add(sphere_id)
            self.grid[cell_coord].spheres.add(sphere_id)
    
    def add_connector(self, connector_id: int, start_pos: Vector, end_pos: Vector):
        """Fügt Connector zum Grid hinzu"""
        # Bestimme alle Zellen die der Connector durchläuft
        cells_to_update = set()
        
        # Sample Punkte entlang der Verbindung
        num_samples = max(3, int((end_pos - start_pos).length / self.cell_size) + 1)
        for i in range(num_samples):
            t = i / (num_samples - 1) if num_samples > 1 else 0
            sample_pos = start_pos.lerp(end_pos, t)
            cell_coord = self._hash_position(sample_pos)
            cells_to_update.add(cell_coord)
        
        # Update Grid
        for cell_coord in cells_to_update:
            if cell_coord not in self.grid:
                self.grid[cell_coord] = SpatialHashCell()
            self.grid[cell_coord].objects.add(connector_id)
            self.grid[cell_coord].connectors.add(connector_id)
    
    def get_potential_collisions(self, position: Vector, radius: float = 0.0) -> Set[int]:
        """Holt potentielle Kollisionen für Position"""
        potential_objects = set()
        
        # Bestimme zu prüfende Zellen
        main_cell = self._hash_position(position)
        cell_radius = int(np.ceil(radius / self.cell_size)) + 1  # +1 für Sicherheit
        
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                for dz in range(-cell_radius, cell_radius + 1):
                    cell_coord = (
                        main_cell[0] + dx,
                        main_cell[1] + dy,
                        main_cell[2] + dz
                    )
                    
                    if cell_coord in self.grid:
                        potential_objects.update(self.grid[cell_coord].objects)
        
        return potential_objects

class OverlapPrevention:
    """
    Hauptklasse für Überlappungs-Vermeidung
    
    Features:
    - BVH-Tree basierte Kollisionserkennung
    - Spatial Hash Grid für Performance
    - Multiple Auflösungsstrategien
    - Surface-Integration
    - Batch-Processing für viele Objekte
    """
    
    def __init__(self, target_surface: Optional[bpy.types.Object] = None):
        """
        Initialisiert Overlap Prevention System
        
        Args:
            target_surface: Ziel-Oberfläche für Surface-Kollisionen
        """
        self.target_surface = target_surface
        self.surface_bvh: Optional[BVHTree] = None
        self.spatial_grid = SpatialHashGrid(cell_size=8.0)
        
        # Collision detection parameters
        self.collision_tolerance = 0.001  # Minimaler Abstand
        self.surface_clearance = 0.5      # Abstand zur Oberfläche
        
        # Resolution parameters
        self.max_resolution_iterations = 10
        self.resolution_step_size = 0.1
        
        # Caching
        self.sphere_data: Dict[int, Dict[str, Any]] = {}
        self.connector_data: Dict[int, Dict[str, Any]] = {}
        
        # Statistics
        self.collision_stats = {
            'total_checks': 0,
            'collisions_found': 0,
            'collisions_resolved': 0,
            'resolution_time': 0.0
        }
        
        # Initialize surface BVH if available
        if target_surface:
            self.update_surface_target(target_surface)
    
    @measure_time
    def check_sphere_overlaps(self, spheres: List[bpy.types.Object]) -> List[CollisionInfo]:
        """
        Prüft Überlappungen zwischen Spheres
        
        Args:
            spheres: Liste von Sphere-Objekten
            
        Returns:
            Liste erkannter Kollisionen
        """
        if len(spheres) < 2:
            return []
        
        debug.info('OVERLAP', f"Checking overlaps for {len(spheres)} spheres")
        start_time = time.time()
        
        # Update spatial grid
        self._update_spatial_grid_spheres(spheres)
        
        collisions = []
        
        # Paarweise Kollisionsprüfung mit Spatial Hash Optimierung
        for i, sphere_a in enumerate(spheres):
            pos_a = Vector(sphere_a.location)
            radius_a = sphere_a.get("kette_radius", max(sphere_a.dimensions) / 2.0)
            
            # Hole potentielle Kollisions-Partner
            potential_partners = self.spatial_grid.get_potential_collisions(pos_a, radius_a)
            
            for j in range(i + 1, len(spheres)):
                if j not in potential_partners:
                    continue  # Skip weit entfernte Spheres
                
                sphere_b = spheres[j]
                pos_b = Vector(sphere_b.location)
                radius_b = sphere_b.get("kette_radius", max(sphere_b.dimensions) / 2.0)
                
                # Sphere-Sphere Kollisionsprüfung
                collision = self._check_sphere_sphere_collision(
                    sphere_a, pos_a, radius_a,
                    sphere_b, pos_b, radius_b
                )
                
                if collision:
                    collisions.append(collision)
        
        # Update statistics
        check_time = time.time() - start_time
        self.collision_stats['total_checks'] += len(spheres) * (len(spheres) - 1) // 2
        self.collision_stats['collisions_found'] += len(collisions)
        
        debug.info('OVERLAP', f"Found {len(collisions)} sphere overlaps in {check_time:.3f}s")
        return collisions
    
    def _update_spatial_grid_spheres(self, spheres: List[bpy.types.Object]):
        """Aktualisiert Spatial Grid mit Sphere-Daten"""
        self.spatial_grid = SpatialHashGrid(cell_size=8.0)
        
        for i, sphere in enumerate(spheres):
            position = Vector(sphere.location)
            radius = sphere.get("kette_radius", max(sphere.dimensions) / 2.0)
            
            self.spatial_grid.add_sphere(i, position, radius)
            
            # Cache sphere data
            self.sphere_data[i] = {
                'object': sphere,
                'position': position,
                'radius': radius,
                'strength': sphere.get("strength_requirement", 1.0),
                'is_anchor': sphere.get("is_anchor", False)
            }
    
    def _check_sphere_sphere_collision(self, sphere_a: bpy.types.Object, pos_a: Vector, radius_a: float,
                                     sphere_b: bpy.types.Object, pos_b: Vector, radius_b: float) -> Optional[CollisionInfo]:
        """Prüft Kollision zwischen zwei Spheres"""
        distance = (pos_a - pos_b).length
        min_distance = radius_a + radius_b + self.collision_tolerance
        
        if distance < min_distance:
            overlap_amount = min_distance - distance
            collision_point = pos_a.lerp(pos_b, radius_a / (radius_a + radius_b))
            
            # Bestimme Auflösungsstrategie
            if radius_a < radius_b:
                strategy = ResolutionStrategy.MOVE_SMALLEST
            elif abs(radius_a - radius_b) < 0.1:
                strategy = ResolutionStrategy.MOVE_BOTH
            else:
                strategy = ResolutionStrategy.RESIZE_SMALLER
            
            return CollisionInfo(
                type=CollisionType.SPHERE_SPHERE,
                objects=(sphere_a, sphere_b),
                overlap_amount=overlap_amount,
                collision_point=collision_point,
                suggested_strategy=strategy,
                resolution_priority=overlap_amount  # Größere Überlappungen zuerst
            )
        
        return None
    
    @measure_time
    def check_connector_overlaps(self, sphere_pairs: List[Tuple[bpy.types.Object, bpy.types.Object]]) -> List[CollisionInfo]:
        """
        Prüft Überlappungen zwischen Connectors
        
        Args:
            sphere_pairs: Liste von Sphere-Paaren (Verbindungen)
            
        Returns:
            Liste erkannter Kollisionen
        """
        if len(sphere_pairs) < 2:
            return []
        
        debug.info('OVERLAP', f"Checking connector overlaps for {len(sphere_pairs)} connections")
        
        collisions = []
        
        # Update spatial grid mit Connectors
        self._update_spatial_grid_connectors(sphere_pairs)
        
        # Paarweise Connector-Kollisionsprüfung
        for i, (sphere_a1, sphere_a2) in enumerate(sphere_pairs):
            pos_a1, pos_a2 = Vector(sphere_a1.location), Vector(sphere_a2.location)
            radius_a = min(
                sphere_a1.get("kette_radius", 1.0),
                sphere_a2.get("kette_radius", 1.0)
            )
            
            for j in range(i + 1, len(sphere_pairs)):
                sphere_b1, sphere_b2 = sphere_pairs[j]
                pos_b1, pos_b2 = Vector(sphere_b1.location), Vector(sphere_b2.location)
                radius_b = min(
                    sphere_b1.get("kette_radius", 1.0),
                    sphere_b2.get("kette_radius", 1.0)
                )
                
                # Line-Line Kollisionsprüfung
                collision = self._check_line_line_collision(
                    pos_a1, pos_a2, radius_a,
                    pos_b1, pos_b2, radius_b,
                    sphere_pairs[i], sphere_pairs[j]
                )
                
                if collision:
                    collisions.append(collision)
        
        debug.info('OVERLAP', f"Found {len(collisions)} connector overlaps")
        return collisions
    
    def _update_spatial_grid_connectors(self, sphere_pairs: List[Tuple[bpy.types.Object, bpy.types.Object]]):
        """Aktualisiert Spatial Grid mit Connector-Daten"""
        for i, (sphere_a, sphere_b) in enumerate(sphere_pairs):
            start_pos = Vector(sphere_a.location)
            end_pos = Vector(sphere_b.location)
            
            self.spatial_grid.add_connector(i + 1000, start_pos, end_pos)  # Offset für Connector IDs
            
            # Cache connector data
            self.connector_data[i] = {
                'sphere_pair': (sphere_a, sphere_b),
                'start_pos': start_pos,
                'end_pos': end_pos,
                'length': (end_pos - start_pos).length
            }
    
    def _check_line_line_collision(self, pos_a1: Vector, pos_a2: Vector, radius_a: float,
                                 pos_b1: Vector, pos_b2: Vector, radius_b: float,
                                 pair_a: Tuple, pair_b: Tuple) -> Optional[CollisionInfo]:
        """Prüft Kollision zwischen zwei Linien (Connectors)"""
        
        # Berechne kürzeste Distanz zwischen zwei Linien im 3D
        line_a = pos_a2 - pos_a1
        line_b = pos_b2 - pos_b1
        w = pos_a1 - pos_b1
        
        a = line_a.dot(line_a)
        b = line_a.dot(line_b)
        c = line_b.dot(line_b)
        d = line_a.dot(w)
        e = line_b.dot(w)
        
        denom = a * c - b * b
        
        if abs(denom) < 1e-6:  # Parallele Linien
            return None
        
        # Parameter für nächste Punkte
        t_a = (b * e - c * d) / denom
        t_b = (a * e - b * d) / denom
        
        # Beschränke auf Liniensegmente
        t_a = max(0.0, min(1.0, t_a))
        t_b = max(0.0, min(1.0, t_b))
        
        # Nächste Punkte
        closest_a = pos_a1 + t_a * line_a
        closest_b = pos_b1 + t_b * line_b
        
        distance = (closest_a - closest_b).length
        min_distance = radius_a + radius_b + self.collision_tolerance
        
        if distance < min_distance:
            overlap_amount = min_distance - distance
            collision_point = closest_a.lerp(closest_b, 0.5)
            
            return CollisionInfo(
                type=CollisionType.CONNECTOR_CONNECTOR,
                objects=(pair_a, pair_b),
                overlap_amount=overlap_amount,
                collision_point=collision_point,
                suggested_strategy=ResolutionStrategy.REROUTE_CONNECTION,
                resolution_priority=overlap_amount
            )
        
        return None
    
    @measure_time
    def check_surface_overlaps(self, spheres: List[bpy.types.Object]) -> List[CollisionInfo]:
        """
        Prüft Kollisionen mit Ziel-Oberfläche
        
        Args:
            spheres: Liste von Sphere-Objekten
            
        Returns:
            Liste erkannter Surface-Kollisionen
        """
        if not self.surface_bvh:
            return []
        
        debug.info('OVERLAP', f"Checking surface overlaps for {len(spheres)} spheres")
        
        collisions = []
        
        for sphere in spheres:
            position = Vector(sphere.location)
            radius = sphere.get("kette_radius", max(sphere.dimensions) / 2.0)
            
            # Finde nächsten Surface-Punkt
            surface_point, surface_normal, _, _ = self.surface_bvh.find_nearest(position)
            
            if surface_point:
                distance_to_surface = (position - surface_point).length
                min_distance = radius + self.surface_clearance
                
                if distance_to_surface < min_distance:
                    overlap_amount = min_distance - distance_to_surface
                    
                    collision = CollisionInfo(
                        type=CollisionType.SPHERE_SURFACE,
                        objects=(sphere, self.target_surface),
                        overlap_amount=overlap_amount,
                        collision_point=surface_point,
                        suggested_strategy=ResolutionStrategy.SURFACE_PROJECT,
                        resolution_priority=overlap_amount
                    )
                    
                    collisions.append(collision)
        
        debug.info('OVERLAP', f"Found {len(collisions)} surface overlaps")
        return collisions
    
    @measure_time
    def resolve_collisions(self, collisions: List[CollisionInfo], 
                         max_iterations: int = None) -> Dict[str, Any]:
        """
        Löst erkannte Kollisionen auf
        
        Args:
            collisions: Liste von Kollisionen
            max_iterations: Maximale Iterationen
            
        Returns:
            Auflösungs-Statistiken
        """
        if not collisions:
            return {'resolved': 0, 'remaining': 0, 'iterations': 0}
        
        max_iterations = max_iterations or self.max_resolution_iterations
        debug.info('OVERLAP', f"Resolving {len(collisions)} collisions (max {max_iterations} iterations)")
        
        start_time = time.time()
        resolved_count = 0
        
        # Sortiere nach Priorität (höchste Überlappung zuerst)
        sorted_collisions = sorted(collisions, key=lambda c: c.resolution_priority, reverse=True)
        
        for iteration in range(max_iterations):
            iteration_resolved = 0
            
            for collision in sorted_collisions:
                if self._resolve_single_collision(collision):
                    iteration_resolved += 1
                    resolved_count += 1
            
            debug.info('OVERLAP', f"Iteration {iteration + 1}: resolved {iteration_resolved} collisions")
            
            # Prüfe ob alle Kollisionen gelöst
            if iteration_resolved == 0:
                break
            
            # Re-check remaining collisions
            remaining_collisions = []
            for collision in sorted_collisions:
                if not self._is_collision_resolved(collision):
                    remaining_collisions.append(collision)
            
            sorted_collisions = remaining_collisions
            
            if not remaining_collisions:
                break
        
        resolution_time = time.time() - start_time
        self.collision_stats['collisions_resolved'] += resolved_count
        self.collision_stats['resolution_time'] += resolution_time
        
        result = {
            'resolved': resolved_count,
            'remaining': len(sorted_collisions),
            'iterations': iteration + 1 if collisions else 0,
            'resolution_time': resolution_time
        }
        
        debug.info('OVERLAP', f"Resolution completed: {result}")
        return result
    
    def _resolve_single_collision(self, collision: CollisionInfo) -> bool:
        """Löst eine einzelne Kollision auf"""
        strategy = collision.suggested_strategy
        
        try:
            if collision.type == CollisionType.SPHERE_SPHERE:
                return self._resolve_sphere_sphere_collision(collision, strategy)
            elif collision.type == CollisionType.CONNECTOR_CONNECTOR:
                return self._resolve_connector_collision(collision, strategy)
            elif collision.type == CollisionType.SPHERE_SURFACE:
                return self._resolve_surface_collision(collision, strategy)
            else:
                debug.warning('OVERLAP', f"Unknown collision type: {collision.type}")
                return False
                
        except Exception as e:
            debug.error('OVERLAP', f"Failed to resolve collision: {e}")
            return False
    
    def _resolve_sphere_sphere_collision(self, collision: CollisionInfo, 
                                       strategy: ResolutionStrategy) -> bool:
        """Löst Sphere-Sphere Kollision auf"""
        sphere_a, sphere_b = collision.objects
        pos_a = Vector(sphere_a.location)
        pos_b = Vector(sphere_b.location)
        
        direction = (pos_b - pos_a).normalized()
        move_distance = collision.overlap_amount * 0.5 + self.resolution_step_size
        
        if strategy == ResolutionStrategy.MOVE_SMALLEST:
            # Bewege kleineres Objekt
            radius_a = sphere_a.get("kette_radius", 1.0)
            radius_b = sphere_b.get("kette_radius", 1.0)
            
            if radius_a <= radius_b:
                sphere_a.location = pos_a - direction * move_distance
            else:
                sphere_b.location = pos_b + direction * move_distance
                
        elif strategy == ResolutionStrategy.MOVE_BOTH:
            # Bewege beide anteilig
            sphere_a.location = pos_a - direction * (move_distance * 0.5)
            sphere_b.location = pos_b + direction * (move_distance * 0.5)
            
        elif strategy == ResolutionStrategy.RESIZE_SMALLER:
            # Verkleinere beide Spheres
            shrink_factor = 0.9
            sphere_a["kette_radius"] = sphere_a.get("kette_radius", 1.0) * shrink_factor
            sphere_b["kette_radius"] = sphere_b.get("kette_radius", 1.0) * shrink_factor
            
            # Update Blender object scale
            sphere_a.scale *= shrink_factor
            sphere_b.scale *= shrink_factor
        
        return True
    
    def _resolve_connector_collision(self, collision: CollisionInfo, 
                                   strategy: ResolutionStrategy) -> bool:
        """Löst Connector-Connector Kollision auf"""
        pair_a, pair_b = collision.objects
        
        if strategy == ResolutionStrategy.REROUTE_CONNECTION:
            # Einfache Umleitung: bewege einen der Endpunkte leicht
            sphere_a1, sphere_a2 = pair_a
            
            # Bewege einen Endpunkt perpendicular zur Verbindung
            connection_dir = (Vector(sphere_a2.location) - Vector(sphere_a1.location)).normalized()
            perpendicular = Vector((-connection_dir.y, connection_dir.x, 0)).normalized()
            
            if perpendicular.length < 0.1:  # Falls parallel zu Z-Achse
                perpendicular = Vector((0, -connection_dir.z, connection_dir.y)).normalized()
            
            move_distance = self.resolution_step_size * 2
            sphere_a1.location = Vector(sphere_a1.location) + perpendicular * move_distance
            
        elif strategy == ResolutionStrategy.REMOVE_WEAKEST:
            # Markiere schwächere Verbindung für Entfernung
            # (Implementierung hängt von übergeordnetem System ab)
            debug.info('OVERLAP', "Marking weaker connection for removal")
            
        return True
    
    def _resolve_surface_collision(self, collision: CollisionInfo, 
                                 strategy: ResolutionStrategy) -> bool:
        """Löst Surface-Kollision auf"""
        sphere, surface = collision.objects
        
        if strategy == ResolutionStrategy.SURFACE_PROJECT:
            position = Vector(sphere.location)
            radius = sphere.get("kette_radius", 1.0)
            
            # Finde Surface-Punkt und Normal
            surface_point, surface_normal, _, _ = self.surface_bvh.find_nearest(position)
            
            if surface_point and surface_normal:
                # Projiziere mit Clearance
                target_distance = radius + self.surface_clearance
                new_position = surface_point + surface_normal * target_distance
                sphere.location = new_position
                
        return True
    
    def _is_collision_resolved(self, collision: CollisionInfo) -> bool:
        """Prüft ob Kollision aufgelöst wurde"""
        # Re-check collision based on type
        if collision.type == CollisionType.SPHERE_SPHERE:
            sphere_a, sphere_b = collision.objects
            pos_a = Vector(sphere_a.location)
            pos_b = Vector(sphere_b.location)
            radius_a = sphere_a.get("kette_radius", max(sphere_a.dimensions) / 2.0)
            radius_b = sphere_b.get("kette_radius", max(sphere_b.dimensions) / 2.0)
            
            distance = (pos_a - pos_b).length
            min_distance = radius_a + radius_b + self.collision_tolerance
            
            return distance >= min_distance
            
        # Für andere Typen: vereinfachte Prüfung
        return True
    
    def update_surface_target(self, target_obj: bpy.types.Object):
        """Aktualisiert Ziel-Oberfläche für Kollisionsprüfung"""
        self.target_surface = target_obj
        
        if target_obj and target_obj.type == 'MESH':
            # Erstelle BVH-Tree
            depsgraph = bpy.context.evaluated_depsgraph_get()
            obj_eval = target_obj.evaluated_get(depsgraph)
            mesh = obj_eval.to_mesh()
            
            self.surface_bvh = BVHTree.FromMesh(mesh)
            debug.info('OVERLAP', f"Surface BVH updated: {target_obj.name}")
    
    def get_collision_report(self) -> Dict[str, Any]:
        """Erstellt detaillierten Kollisions-Bericht"""
        return {
            'statistics': self.collision_stats.copy(),
            'parameters': {
                'collision_tolerance': self.collision_tolerance,
                'surface_clearance': self.surface_clearance,
                'max_resolution_iterations': self.max_resolution_iterations,
                'resolution_step_size': self.resolution_step_size
            },
            'grid_info': {
                'cell_size': self.spatial_grid.cell_size,
                'active_cells': len(self.spatial_grid.grid),
                'cached_spheres': len(self.sphere_data),
                'cached_connectors': len(self.connector_data)
            },
            'surface_integration': {
                'surface_available': self.target_surface is not None,
                'bvh_available': self.surface_bvh is not None
            }
        }


# =========================================
# CONVENIENCE FUNCTIONS
# =========================================

@measure_time
def detect_and_resolve_overlaps(spheres: List[bpy.types.Object],
                              sphere_pairs: List[Tuple[bpy.types.Object, bpy.types.Object]] = None,
                              surface_obj: Optional[bpy.types.Object] = None,
                              auto_resolve: bool = True) -> Dict[str, Any]:
    """
    Convenience-Funktion für vollständige Überlappungs-Behandlung
    
    Args:
        spheres: Liste von Sphere-Objekten
        sphere_pairs: Optional Verbindungen zwischen Spheres
        surface_obj: Optional Ziel-Oberfläche
        auto_resolve: Automatische Auflösung von Kollisionen
        
    Returns:
        Detaillierter Bericht über Kollisionen und Auflösungen
    """
    overlap_system = OverlapPrevention(surface_obj)
    
    # Sammle alle Kollisionen
    all_collisions = []
    
    # Sphere-Sphere Kollisionen
    sphere_collisions = overlap_system.check_sphere_overlaps(spheres)
    all_collisions.extend(sphere_collisions)
    
    # Connector-Kollisionen
    if sphere_pairs:
        connector_collisions = overlap_system.check_connector_overlaps(sphere_pairs)
        all_collisions.extend(connector_collisions)
    
    # Surface-Kollisionen
    if surface_obj:
        surface_collisions = overlap_system.check_surface_overlaps(spheres)
        all_collisions.extend(surface_collisions)
    
    # Automatische Auflösung
    resolution_result = {}
    if auto_resolve and all_collisions:
        resolution_result = overlap_system.resolve_collisions(all_collisions)
    
    # Zusammenfassender Bericht
    report = {
        'collision_summary': {
            'sphere_overlaps': len(sphere_collisions),
            'connector_overlaps': len(connector_collisions) if sphere_pairs else 0,
            'surface_overlaps': len(surface_collisions) if surface_obj else 0,
            'total_collisions': len(all_collisions)
        },
        'resolution_summary': resolution_result,
        'system_report': overlap_system.get_collision_report()
    }
    
    return report


def batch_collision_check(object_groups: List[List[bpy.types.Object]],
                         surface_obj: Optional[bpy.types.Object] = None) -> List[Dict[str, Any]]:
    """
    Batch-Kollisionsprüfung für mehrere Objektgruppen
    
    Args:
        object_groups: Liste von Sphere-Listen
        surface_obj: Optional Ziel-Oberfläche
        
    Returns:
        Liste von Kollisions-Berichten pro Gruppe
    """
    results = []
    overlap_system = OverlapPrevention(surface_obj)
    
    for i, spheres in enumerate(object_groups):
        debug.info('OVERLAP', f"Processing group {i + 1}/{len(object_groups)} ({len(spheres)} spheres)")
        
        collisions = overlap_system.check_sphere_overlaps(spheres)
        if surface_obj:
            surface_collisions = overlap_system.check_surface_overlaps(spheres)
            collisions.extend(surface_collisions)
        
        group_result = {
            'group_index': i,
            'sphere_count': len(spheres),
            'collision_count': len(collisions),
            'collisions': collisions
        }
        
        results.append(group_result)
    
    return results
