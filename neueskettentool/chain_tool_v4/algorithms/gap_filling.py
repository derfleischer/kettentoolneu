"""
Chain Tool V4 - Gap Filling Module
==================================

Lückenerkennung und -füllung für Orthesen-Verstärkungen
- Automatische Erkennung von Pattern-Lücken
- Adaptive Füllstrategien basierend auf Belastungsanalyse
- Nahtlose Integration neuer Elemente
- Surface-aware Gap-Filling mit Aura-System Integration

Author: Chain Tool V4 Development Team
Version: 4.0.0
"""

import bpy
import bmesh
import numpy as np
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from mathutils.geometry import intersect_point_tri, barycentric_transform
from typing import List, Tuple, Dict, Optional, Set, Any, Union
import time
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict

# V4 Module imports
from ..core.constants import GapType, FillStrategy, PatternType
from ..core.state_manager import StateManager
from ..utils.performance import measure_time, PerformanceTracker
from ..utils.debug import DebugManager
from ..utils.caching import CacheManager
from ..utils.math_utils import safe_normalize, calculate_distance_3d

# Initialize systems
debug = DebugManager()
cache = CacheManager()
perf = PerformanceTracker()

class GapType(Enum):
    """Typen von Lücken in Patterns"""
    SMALL_HOLE = "small_hole"                 # Kleine isolierte Lücken
    LARGE_VOID = "large_void"                # Große zusammenhängende Bereiche
    EDGE_GAP = "edge_gap"                    # Lücken an Pattern-Rändern
    CONNECTION_BREAK = "connection_break"     # Unterbrochene Verbindungen
    STRESS_CONCENTRATION = "stress_concentration"  # Bereiche hoher Belastung
    SURFACE_DEVIATION = "surface_deviation"   # Bereiche mit Surface-Abweichung

class FillStrategy(Enum):
    """Strategien zum Lücken-Füllen"""
    SPHERE_INSERTION = "sphere_insertion"     # Neue Spheres einfügen
    CONNECTION_BRIDGING = "connection_bridging"  # Verbindungen überbrücken
    PATTERN_EXTENSION = "pattern_extension"   # Pattern erweitern
    MESH_DENSIFICATION = "mesh_densification" # Mesh verdichten
    ADAPTIVE_SCALING = "adaptive_scaling"     # Adaptive Skalierung
    HYBRID_APPROACH = "hybrid_approach"       # Kombination mehrerer Strategien

@dataclass
class GapInfo:
    """Information über eine erkannte Lücke"""
    type: GapType
    center: Vector                           # Zentrum der Lücke
    size: float                             # Charakteristische Größe
    boundary_points: List[Vector] = field(default_factory=list)  # Randpunkte
    affected_spheres: Set[int] = field(default_factory=set)      # Betroffene Spheres
    stress_level: float = 1.0               # Belastungsniveau
    priority: float = 1.0                   # Füll-Priorität
    suggested_strategy: Optional[FillStrategy] = None
    surface_normal: Optional[Vector] = None  # Oberflächen-Normal
    
@dataclass
class FillSolution:
    """Lösung für Lücken-Füllung"""
    gap_info: GapInfo
    strategy: FillStrategy
    new_spheres: List[Dict[str, Any]] = field(default_factory=list)
    new_connections: List[Tuple[int, int]] = field(default_factory=list)
    modifications: List[Dict[str, Any]] = field(default_factory=list)
    quality_score: float = 0.0

class VoronoiAnalyzer:
    """Analysiert Voronoi-Eigenschaften für Gap-Detection"""
    
    def __init__(self, points: List[Vector]):
        self.points = points
        self.voronoi_cells: Dict[int, List[Vector]] = {}
        self.cell_neighbors: Dict[int, Set[int]] = defaultdict(set)
        
    def compute_voronoi_properties(self) -> Dict[int, Dict[str, float]]:
        """Berechnet Voronoi-Eigenschaften für alle Punkte"""
        properties = {}
        
        for i, point in enumerate(self.points):
            # Finde nächste Nachbarn
            neighbors = self._find_natural_neighbors(i)
            
            # Berechne Voronoi-Zell-Eigenschaften
            cell_area = self._estimate_cell_area(i, neighbors)
            cell_perimeter = self._estimate_cell_perimeter(i, neighbors)
            
            # Regularität (wie nah an regulärem Polygon)
            regularity = self._calculate_regularity(i, neighbors)
            
            properties[i] = {
                'area': cell_area,
                'perimeter': cell_perimeter,
                'neighbor_count': len(neighbors),
                'regularity': regularity,
                'density': len(neighbors) / max(cell_area, 0.001)
            }
            
        return properties
    
    def _find_natural_neighbors(self, point_index: int, max_neighbors: int = 8) -> List[int]:
        """Findet natürliche Nachbarn für Voronoi-Analyse"""
        center = self.points[point_index]
        distances = []
        
        for i, point in enumerate(self.points):
            if i != point_index:
                dist = (center - point).length
                distances.append((dist, i))
        
        # Sortiere nach Distanz und nimm nächste Nachbarn
        distances.sort(key=lambda x: x[0])
        
        # Adaptive Nachbaranzahl basierend auf Distanz-Sprüngen
        neighbors = []
        prev_dist = 0
        
        for dist, idx in distances[:max_neighbors]:
            # Großer Distanz-Sprung = Ende der natürlichen Nachbarschaft
            if prev_dist > 0 and dist > prev_dist * 2.0 and len(neighbors) >= 3:
                break
            neighbors.append(idx)
            prev_dist = dist
            
        return neighbors[:max_neighbors]
    
    def _estimate_cell_area(self, point_index: int, neighbors: List[int]) -> float:
        """Schätzt Voronoi-Zell-Fläche"""
        if len(neighbors) < 3:
            return 1.0
        
        center = self.points[point_index]
        
        # Approximation durch Polygon-Fläche
        # Sortiere Nachbarn nach Winkel
        angles = []
        for neighbor_idx in neighbors:
            neighbor = self.points[neighbor_idx]
            direction = neighbor - center
            angle = np.arctan2(direction.y, direction.x)
            angles.append((angle, neighbor_idx))
        
        angles.sort(key=lambda x: x[0])
        
        # Berechne Polygon-Fläche
        total_area = 0.0
        for i in range(len(angles)):
            j = (i + 1) % len(angles)
            p1 = self.points[angles[i][1]]
            p2 = self.points[angles[j][1]]
            
            # Mittelpunkt der Kante als Polygon-Eckpunkt
            mid_point = (p1 + p2) * 0.5
            
            # Dreiecks-Fläche zwischen Center und Kantenmittelpunkten
            if i < len(angles) - 1:
                next_mid = (self.points[angles[j][1]] + self.points[angles[(j + 1) % len(angles)][1]]) * 0.5
                triangle_area = 0.5 * abs((mid_point - center).cross(next_mid - center).length)
                total_area += triangle_area
        
        return max(total_area, 1.0)
    
    def _estimate_cell_perimeter(self, point_index: int, neighbors: List[int]) -> float:
        """Schätzt Voronoi-Zell-Perimeter"""
        center = self.points[point_index]
        
        # Durchschnittliche Distanz zu Nachbarn * Anzahl Nachbarn
        total_dist = sum((center - self.points[idx]).length for idx in neighbors)
        avg_dist = total_dist / max(len(neighbors), 1)
        
        return avg_dist * len(neighbors)
    
    def _calculate_regularity(self, point_index: int, neighbors: List[int]) -> float:
        """Berechnet Regularität der Voronoi-Zelle"""
        if len(neighbors) < 3:
            return 0.0
        
        center = self.points[point_index]
        
        # Berechne Winkel zwischen aufeinanderfolgenden Nachbarn
        angles = []
        for neighbor_idx in neighbors:
            neighbor = self.points[neighbor_idx]
            direction = neighbor - center
            angle = np.arctan2(direction.y, direction.x)
            angles.append(angle)
        
        angles.sort()
        
        # Berechne Winkel-Unterschiede 
        ideal_angle = 2 * np.pi / len(neighbors)
        angle_diffs = []
        
        for i in range(len(angles)):
            j = (i + 1) % len(angles)
            diff = angles[j] - angles[i]
            if diff < 0:
                diff += 2 * np.pi
            angle_diffs.append(abs(diff - ideal_angle))
        
        # Regularität = 1 - (durchschnittliche Abweichung / max mögliche Abweichung)
        avg_deviation = sum(angle_diffs) / len(angle_diffs)
        max_deviation = np.pi  # Maximale mögliche Abweichung
        
        return max(0.0, 1.0 - avg_deviation / max_deviation)

class GapDetector:
    """Erkennt Lücken in Sphere-Patterns"""
    
    def __init__(self, target_surface: Optional[bpy.types.Object] = None):
        """
        Initialisiert Gap Detector
        
        Args:
            target_surface: Ziel-Oberfläche für Surface-aware Detection
        """
        self.target_surface = target_surface
        self.surface_bvh: Optional[BVHTree] = None
        
        # Detection parameters
        self.min_gap_size = 2.0      # Minimale Lückengröße für Erkennung
        self.max_neighbor_distance = 15.0  # Max Distanz für Nachbarschafts-Analyse
        self.density_threshold = 0.3  # Schwellwert für Dichte-basierte Gap-Erkennung
        
        # Voronoi analyzer
        self.voronoi_analyzer: Optional[VoronoiAnalyzer] = None
        
        # Initialize surface BVH
        if target_surface:
            self.update_surface_target(target_surface)
    
    @measure_time
    def detect_gaps(self, spheres: List[bpy.types.Object],
                   stress_data: Optional[Dict[int, float]] = None) -> List[GapInfo]:
        """
        Hauptfunktion: Erkennt Lücken in Sphere-Pattern
        
        Args:
            spheres: Liste von Sphere-Objekten
            stress_data: Optional Stress-Daten pro Sphere
            
        Returns:
            Liste erkannter Lücken
        """
        if len(spheres) < 3:
            debug.warning('GAP_FILL', "Insufficient spheres for gap detection")
            return []
        
        debug.info('GAP_FILL', f"Starting gap detection for {len(spheres)} spheres")
        
        # Extract positions and properties
        positions = [Vector(sphere.location) for sphere in spheres]
        self.voronoi_analyzer = VoronoiAnalyzer(positions)
        
        # Analyze Voronoi properties
        voronoi_props = self.voronoi_analyzer.compute_voronoi_properties()
        
        detected_gaps = []
        
        # 1. Density-based gap detection
        density_gaps = self._detect_density_gaps(spheres, voronoi_props)
        detected_gaps.extend(density_gaps)
        
        # 2. Connection-based gap detection  
        connection_gaps = self._detect_connection_gaps(spheres)
        detected_gaps.extend(connection_gaps)
        
        # 3. Stress-based gap detection
        if stress_data:
            stress_gaps = self._detect_stress_gaps(spheres, stress_data)
            detected_gaps.extend(stress_gaps)
        
        # 4. Surface-based gap detection
        if self.surface_bvh:
            surface_gaps = self._detect_surface_gaps(spheres)
            detected_gaps.extend(surface_gaps)
        
        # 5. Edge gap detection
        edge_gaps = self._detect_edge_gaps(spheres)
        detected_gaps.extend(edge_gaps)
        
        # Remove duplicates and prioritize
        final_gaps = self._consolidate_gaps(detected_gaps)
        
        debug.info('GAP_FILL', f"Detected {len(final_gaps)} gaps total")
        return final_gaps
    
    def _detect_density_gaps(self, spheres: List[bpy.types.Object], 
                           voronoi_props: Dict[int, Dict[str, float]]) -> List[GapInfo]:
        """Erkennt Lücken basierend auf lokaler Dichte"""
        gaps = []
        
        # Finde Bereiche mit ungewöhnlich niedriger Dichte
        densities = [props['density'] for props in voronoi_props.values()]
        avg_density = sum(densities) / len(densities)
        density_threshold = avg_density * self.density_threshold
        
        for i, (sphere_idx, props) in enumerate(voronoi_props.items()):
            if props['density'] < density_threshold and props['area'] > self.min_gap_size:
                
                center = Vector(spheres[sphere_idx].location)
                size = np.sqrt(props['area'])  # Charakteristische Größe
                
                # Bestimme Gap-Typ basierend auf Eigenschaften
                if props['neighbor_count'] < 3:
                    gap_type = GapType.LARGE_VOID
                elif props['regularity'] < 0.3:
                    gap_type = GapType.SMALL_HOLE
                else:
                    gap_type = GapType.CONNECTION_BREAK
                
                gap = GapInfo(
                    type=gap_type,
                    center=center,
                    size=size,
                    affected_spheres={sphere_idx},
                    priority=1.0 - props['density'] / avg_density,  # Niedrigere Dichte = höhere Priorität
                    suggested_strategy=self._suggest_fill_strategy(gap_type, size)
                )
                
                gaps.append(gap)
        
        debug.info('GAP_FILL', f"Found {len(gaps)} density-based gaps")
        return gaps
    
    def _detect_connection_gaps(self, spheres: List[bpy.types.Object]) -> List[GapInfo]:
        """Erkennt Lücken in Verbindungsstrukturen"""
        gaps = []
        
        # Baue Nachbarschaftsgraph
        connections = self._build_connection_graph(spheres)
        
        # Finde isolierte Bereiche oder schwache Verbindungen
        for i, sphere in enumerate(spheres):
            neighbors = connections.get(i, [])
            
            if len(neighbors) < 2:  # Isolierte oder schwach verbundene Spheres
                position = Vector(sphere.location)
                
                # Suche nächste Nachbarn für potentielle Verbindungen
                potential_connections = self._find_potential_connections(i, spheres)
                
                if potential_connections:
                    # Berechne Zentrum des Verbindungs-Gaps
                    gap_center = position
                    for neighbor_idx, _ in potential_connections[:3]:
                        gap_center += Vector(spheres[neighbor_idx].location)
                    gap_center /= min(4, len(potential_connections) + 1)
                    
                    # Geschätzte Gap-Größe
                    gap_size = min([dist for _, dist in potential_connections[:3]])
                    
                    gap = GapInfo(
                        type=GapType.CONNECTION_BREAK,
                        center=gap_center,
                        size=gap_size,
                        affected_spheres={i},
                        priority=1.0 / max(len(neighbors), 1),  # Weniger Verbindungen = höhere Priorität
                        suggested_strategy=FillStrategy.CONNECTION_BRIDGING
                    )
                    
                    gaps.append(gap)
        
        debug.info('GAP_FILL', f"Found {len(gaps)} connection gaps")
        return gaps
    
    def _detect_stress_gaps(self, spheres: List[bpy.types.Object], 
                          stress_data: Dict[int, float]) -> List[GapInfo]:
        """Erkennt Lücken in hoch belasteten Bereichen"""
        gaps = []
        
        # Finde Bereiche mit hoher Belastung aber geringer Verstärkung
        stress_values = list(stress_data.values())
        avg_stress = sum(stress_values) / len(stress_values)
        high_stress_threshold = avg_stress * 1.5
        
        for sphere_idx, stress_level in stress_data.items():
            if stress_level > high_stress_threshold and sphere_idx < len(spheres):
                sphere = spheres[sphere_idx]
                position = Vector(sphere.location)
                
                # Prüfe lokale Verstärkung (Anzahl Nachbarn)
                neighbors = self._find_nearby_spheres(sphere_idx, spheres, radius=10.0)
                
                # Hohe Belastung + wenige Nachbarn = Gap
                if len(neighbors) < 4:  # Threshold konfigurierbar
                    gap = GapInfo(
                        type=GapType.STRESS_CONCENTRATION,
                        center=position,
                        size=stress_level,  # Stress als Größenmaß
                        affected_spheres={sphere_idx},
                        stress_level=stress_level,
                        priority=stress_level / avg_stress,
                        suggested_strategy=FillStrategy.MESH_DENSIFICATION
                    )
                    
                    gaps.append(gap)
        
        debug.info('GAP_FILL', f"Found {len(gaps)} stress-based gaps")
        return gaps
    
    def _detect_surface_gaps(self, spheres: List[bpy.types.Object]) -> List[GapInfo]:
        """Erkennt Lücken basierend auf Surface-Abweichung"""
        gaps = []
        
        if not self.surface_bvh:
            return gaps
        
        for i, sphere in enumerate(spheres):
            position = Vector(sphere.location)
            
            # Finde nächsten Surface-Punkt
            surface_point, surface_normal, _, _ = self.surface_bvh.find_nearest(position)
            
            if surface_point:
                distance_to_surface = (position - surface_point).length
                
                # Große Abweichung von Oberfläche = potentieller Gap
                if distance_to_surface > 3.0:  # Threshold konfigurierbar
                    
                    # Suche nach anderen Spheres in der Nähe des Surface-Punkts
                    nearby_spheres = self._find_spheres_near_point(surface_point, spheres, radius=5.0)
                    
                    if len(nearby_spheres) < 2:  # Wenig Abdeckung an Surface
                        gap = GapInfo(
                            type=GapType.SURFACE_DEVIATION,
                            center=surface_point,
                            size=distance_to_surface,
                            affected_spheres={i},
                            surface_normal=surface_normal,
                            priority=distance_to_surface / 10.0,  # Normalisierte Priorität
                            suggested_strategy=FillStrategy.SURFACE_PROJECT
                        )
                        
                        gaps.append(gap)
        
        debug.info('GAP_FILL', f"Found {len(gaps)} surface gaps")
        return gaps
    
    def _detect_edge_gaps(self, spheres: List[bpy.types.Object]) -> List[GapInfo]:
        """Erkennt Lücken an Pattern-Rändern"""
        gaps = []
        
        # Identifiziere Rand-Spheres (wenige Nachbarn)
        edge_spheres = []
        
        for i, sphere in enumerate(spheres):
            neighbors = self._find_nearby_spheres(i, spheres, radius=12.0)
            
            if len(neighbors) <= 3:  # Rand-Kriterium
                edge_spheres.append((i, sphere, neighbors))
        
        # Analysiere Lücken zwischen Rand-Spheres
        for i, (sphere_idx, sphere, neighbors) in enumerate(edge_spheres):
            position = Vector(sphere.location)
            
            # Suche nach anderen Rand-Spheres in mittlerer Entfernung
            for j, (other_idx, other_sphere, _) in enumerate(edge_spheres[i+1:], i+1):
                other_position = Vector(other_sphere.location)
                distance = (position - other_position).length
                
                # Mittlere Distanz deutet auf Gap hin
                if 8.0 < distance < 20.0:  # Konfigurierbare Schwellwerte
                    
                    # Prüfe ob Zwischenbereich spärlich besetzt
                    mid_point = (position + other_position) * 0.5
                    nearby_count = len(self._find_spheres_near_point(mid_point, spheres, radius=6.0))
                    
                    if nearby_count < 2:
                        gap = GapInfo(
                            type=GapType.EDGE_GAP,
                            center=mid_point,
                            size=distance,
                            affected_spheres={sphere_idx, other_idx},
                            priority=1.0 / max(nearby_count, 1),
                            suggested_strategy=FillStrategy.PATTERN_EXTENSION
                        )
                        
                        gaps.append(gap)
        
        debug.info('GAP_FILL', f"Found {len(gaps)} edge gaps")
        return gaps
    
    def _build_connection_graph(self, spheres: List[bpy.types.Object]) -> Dict[int, List[int]]:
        """Baut Graph der Sphere-Verbindungen"""
        connections = defaultdict(list)
        
        for i, sphere_a in enumerate(spheres):
            pos_a = Vector(sphere_a.location)
            
            for j, sphere_b in enumerate(spheres[i+1:], i+1):
                pos_b = Vector(sphere_b.location)
                distance = (pos_a - pos_b).length
                
                # Verbindung wenn in angemessener Distanz
                if distance <= self.max_neighbor_distance:
                    connections[i].append(j)
                    connections[j].append(i)
        
        return dict(connections)
    
    def _find_potential_connections(self, sphere_idx: int, 
                                  spheres: List[bpy.types.Object]) -> List[Tuple[int, float]]:
        """Findet potentielle Verbindungen für isolierte Sphere"""
        sphere = spheres[sphere_idx]
        position = Vector(sphere.location)
        
        candidates = []
        for i, other_sphere in enumerate(spheres):
            if i != sphere_idx:
                other_pos = Vector(other_sphere.location)
                distance = (position - other_pos).length
                
                if distance <= self.max_neighbor_distance * 1.5:  # Erweiterte Suche
                    candidates.append((i, distance))
        
        # Sortiere nach Distanz
        candidates.sort(key=lambda x: x[1])
        return candidates
    
    def _find_nearby_spheres(self, center_idx: int, spheres: List[bpy.types.Object], 
                           radius: float) -> List[int]:
        """Findet Spheres in der Nähe einer gegebenen Sphere"""
        center_pos = Vector(spheres[center_idx].location)
        nearby = []
        
        for i, sphere in enumerate(spheres):
            if i != center_idx:
                distance = (center_pos - Vector(sphere.location)).length
                if distance <= radius:
                    nearby.append(i)
        
        return nearby
    
    def _find_spheres_near_point(self, point: Vector, spheres: List[bpy.types.Object], 
                               radius: float) -> List[int]:
        """Findet Spheres in der Nähe eines gegebenen Punkts"""
        nearby = []
        
        for i, sphere in enumerate(spheres):
            distance = (point - Vector(sphere.location)).length
            if distance <= radius:
                nearby.append(i)
        
        return nearby
    
    def _suggest_fill_strategy(self, gap_type: GapType, size: float) -> FillStrategy:
        """Schlägt Fill-Strategie basierend auf Gap-Typ vor"""
        if gap_type == GapType.SMALL_HOLE:
            return FillStrategy.SPHERE_INSERTION
        elif gap_type == GapType.LARGE_VOID:
            return FillStrategy.PATTERN_EXTENSION if size > 10.0 else FillStrategy.MESH_DENSIFICATION
        elif gap_type == GapType.CONNECTION_BREAK:
            return FillStrategy.CONNECTION_BRIDGING
        elif gap_type == GapType.STRESS_CONCENTRATION:
            return FillStrategy.MESH_DENSIFICATION
        elif gap_type == GapType.SURFACE_DEVIATION:
            return FillStrategy.SURFACE_PROJECT
        elif gap_type == GapType.EDGE_GAP:
            return FillStrategy.PATTERN_EXTENSION
        else:
            return FillStrategy.HYBRID_APPROACH
    
    def _consolidate_gaps(self, gaps: List[GapInfo]) -> List[GapInfo]:
        """Konsolidiert überlappende oder ähnliche Gaps"""
        if not gaps:
            return gaps
        
        # Sortiere nach Priorität
        sorted_gaps = sorted(gaps, key=lambda g: g.priority, reverse=True)
        
        consolidated = []
        used_positions = set()
        
        for gap in sorted_gaps:
            # Prüfe ob Gap zu nah an bereits verwendeter Position
            too_close = False
            for used_pos in used_positions:
                if (gap.center - used_pos).length < gap.size * 0.5:
                    too_close = True
                    break
            
            if not too_close:
                consolidated.append(gap)
                used_positions.add(gap.center)
        
        return consolidated
    
    def update_surface_target(self, target_obj: bpy.types.Object):
        """Aktualisiert Ziel-Oberfläche"""
        self.target_surface = target_obj
        
        if target_obj and target_obj.type == 'MESH':
            depsgraph = bpy.context.evaluated_depsgraph_get()
            obj_eval = target_obj.evaluated_get(depsgraph)
            mesh = obj_eval.to_mesh()
            
            self.surface_bvh = BVHTree.FromMesh(mesh)
            debug.info('GAP_FILL', f"Surface BVH updated: {target_obj.name}")

class GapFiller:
    """Füllt erkannte Lücken mit geeigneten Strategien"""
    
    def __init__(self, target_surface: Optional[bpy.types.Object] = None):
        """
        Initialisiert Gap Filler
        
        Args:
            target_surface: Ziel-Oberfläche für Surface-aware Filling
        """
        self.target_surface = target_surface
        self.surface_bvh: Optional[BVHTree] = None
        
        # Filling parameters
        self.min_sphere_radius = 0.5
        self.max_sphere_radius = 3.0
        self.surface_offset = 1.0
        
        if target_surface:
            self.update_surface_target(target_surface)
    
    @measure_time
    def fill_gaps(self, gaps: List[GapInfo], 
                 existing_spheres: List[bpy.types.Object]) -> List[FillSolution]:
        """
        Hauptfunktion: Füllt erkannte Lücken
        
        Args:
            gaps: Liste erkannter Lücken
            existing_spheres: Bestehende Sphere-Objekte
            
        Returns:
            Liste von Fill-Lösungen
        """
        if not gaps:
            return []
        
        debug.info('GAP_FILL', f"Filling {len(gaps)} gaps")
        
        solutions = []
        
        for gap in gaps:
            strategy = gap.suggested_strategy or FillStrategy.SPHERE_INSERTION
            
            try:
                if strategy == FillStrategy.SPHERE_INSERTION:
                    solution = self._fill_with_sphere_insertion(gap, existing_spheres)
                elif strategy == FillStrategy.CONNECTION_BRIDGING:
                    solution = self._fill_with_connection_bridging(gap, existing_spheres)
                elif strategy == FillStrategy.PATTERN_EXTENSION:
                    solution = self._fill_with_pattern_extension(gap, existing_spheres)
                elif strategy == FillStrategy.MESH_DENSIFICATION:
                    solution = self._fill_with_mesh_densification(gap, existing_spheres)
                elif strategy == FillStrategy.SURFACE_PROJECT:
                    solution = self._fill_with_surface_projection(gap, existing_spheres)
                else:  # HYBRID_APPROACH
                    solution = self._fill_with_hybrid_approach(gap, existing_spheres)
                
                if solution:
                    solutions.append(solution)
                    
            except Exception as e:
                debug.error('GAP_FILL', f"Failed to fill gap: {e}")
        
        debug.info('GAP_FILL', f"Generated {len(solutions)} fill solutions")
        return solutions
    
    def _fill_with_sphere_insertion(self, gap: GapInfo, 
                                  existing_spheres: List[bpy.types.Object]) -> Optional[FillSolution]:
        """Füllt Lücke durch Einfügen neuer Spheres"""
        
        # Bestimme optimale Sphere-Position
        position = gap.center
        
        # Surface-Projektion wenn verfügbar
        if self.surface_bvh and gap.surface_normal:
            surface_point, surface_normal, _, _ = self.surface_bvh.find_nearest(position)
            if surface_point:
                position = surface_point + surface_normal * self.surface_offset
        
        # Bestimme Sphere-Radius basierend auf lokaler Dichte
        nearby_spheres = self._find_nearby_spheres_for_position(position, existing_spheres)
        
        if nearby_spheres:
            avg_distance = sum(dist for _, dist in nearby_spheres[:3]) / min(3, len(nearby_spheres))
            radius = max(self.min_sphere_radius, min(self.max_sphere_radius, avg_distance * 0.3))
        else:
            radius = (self.min_sphere_radius + self.max_sphere_radius) * 0.5
        
        # Erstelle neue Sphere-Definition
        new_sphere = {
            'location': position,
            'radius': radius,
            'strength_requirement': gap.stress_level,
            'gap_fill': True
        }
        
        # Erstelle Verbindungen zu nächsten Nachbarn
        new_connections = []
        for sphere_idx, _ in nearby_spheres[:4]:  # Max 4 Verbindungen
            new_connections.append((len(existing_spheres), sphere_idx))  # Neue Sphere zu existierender
        
        solution = FillSolution(
            gap_info=gap,
            strategy=FillStrategy.SPHERE_INSERTION,
            new_spheres=[new_sphere],
            new_connections=new_connections,
            quality_score=self._calculate_solution_quality(gap, [new_sphere], new_connections)
        )
        
        return solution
    
    def _fill_with_connection_bridging(self, gap: GapInfo, 
                                     existing_spheres: List[bpy.types.Object]) -> Optional[FillSolution]:
        """Füllt Lücke durch Überbrücken bestehender Spheres"""
        
        # Finde Spheres die verbunden werden sollen
        affected_spheres = list(gap.affected_spheres)
        nearby_spheres = self._find_nearby_spheres_for_position(gap.center, existing_spheres, radius=15.0)
        
        # Kombiniere mit betroffenen Spheres
        all_candidates = set(affected_spheres)
        for sphere_idx, _ in nearby_spheres[:6]:
            all_candidates.add(sphere_idx)
        
        # Erstelle optimale Verbindungen
        new_connections = []
        candidates = list(all_candidates)
        
        for i, sphere_idx_a in enumerate(candidates):
            for sphere_idx_b in candidates[i+1:]:
                pos_a = Vector(existing_spheres[sphere_idx_a].location)
                pos_b = Vector(existing_spheres[sphere_idx_b].location)
                distance = (pos_a - pos_b).length
                
                # Verbinde wenn angemessene Distanz
                if 5.0 < distance < 20.0:  # Konfigurierbar
                    new_connections.append((sphere_idx_a, sphere_idx_b))
        
        solution = FillSolution(
            gap_info=gap,
            strategy=FillStrategy.CONNECTION_BRIDGING,
            new_connections=new_connections,
            quality_score=len(new_connections) * 0.5  # Einfache Qualitäts-Metrik
        )
        
        return solution
    
    def _fill_with_pattern_extension(self, gap: GapInfo, 
                                   existing_spheres: List[bpy.types.Object]) -> Optional[FillSolution]:
        """Füllt Lücke durch Pattern-Erweiterung"""
        
        # Analysiere lokales Pattern
        nearby_spheres = self._find_nearby_spheres_for_position(gap.center, existing_spheres, radius=20.0)
        
        if len(nearby_spheres) < 3:
            return None
        
        # Extrahiere Pattern-Eigenschaften
        positions = [Vector(existing_spheres[idx].location) for idx, _ in nearby_spheres[:6]]
        avg_distance = sum(dist for _, dist in nearby_spheres[:3]) / min(3, len(nearby_spheres))
        
        # Generiere neue Spheres basierend auf Pattern
        new_spheres = []
        
        # Einfache Gitter-Erweiterung
        direction_vectors = []
        center_pos = gap.center
        
        for pos in positions[:3]:
            direction = (pos - center_pos).normalized()
            direction_vectors.append(direction)
        
        # Erstelle neue Positionen
        for i, direction in enumerate(direction_vectors):
            new_pos = center_pos + direction * avg_distance * 0.8
            
            # Surface-Projektion
            if self.surface_bvh:
                surface_point, surface_normal, _, _ = self.surface_bvh.find_nearest(new_pos)
                if surface_point:
                    new_pos = surface_point + surface_normal * self.surface_offset
            
            new_sphere = {
                'location': new_pos,
                'radius': avg_distance * 0.2,  # Adaptive Größe
                'pattern_extension': True
            }
            
            new_spheres.append(new_sphere)
        
        # Erstelle Verbindungen zwischen neuen Spheres und zu bestehenden
        new_connections = []
        start_idx = len(existing_spheres)
        
        # Verbinde neue Spheres untereinander
        for i in range(len(new_spheres)):
            for j in range(i + 1, len(new_spheres)):
                new_connections.append((start_idx + i, start_idx + j))
        
        # Verbinde zu bestehenden Spheres
        for i, new_sphere in enumerate(new_spheres):
            nearby = self._find_nearby_spheres_for_position(new_sphere['location'], existing_spheres, radius=avg_distance * 1.2)
            for sphere_idx, _ in nearby[:2]:  # Max 2 Verbindungen pro neue Sphere
                new_connections.append((start_idx + i, sphere_idx))
        
        solution = FillSolution(
            gap_info=gap,
            strategy=FillStrategy.PATTERN_EXTENSION,
            new_spheres=new_spheres,
            new_connections=new_connections,
            quality_score=len(new_spheres) * 0.8 + len(new_connections) * 0.2
        )
        
        return solution
    
    def _fill_with_mesh_densification(self, gap: GapInfo, 
                                    existing_spheres: List[bpy.types.Object]) -> Optional[FillSolution]:
        """Füllt Lücke durch lokale Mesh-Verdichtung"""
        
        # Finde bestehende Spheres in Gap-Bereich
        affected_sphere_indices = list(gap.affected_spheres)
        nearby_indices = [idx for idx, _ in self._find_nearby_spheres_for_position(gap.center, existing_spheres, radius=12.0)]
        
        all_indices = set(affected_sphere_indices + nearby_indices)
        
        new_spheres = []
        new_connections = []
        
        # Für jede betroffene Sphere: erstelle zusätzliche Spheres in der Nähe
        for sphere_idx in all_indices:
            sphere = existing_spheres[sphere_idx]
            base_pos = Vector(sphere.location)
            base_radius = sphere.get("kette_radius", 1.0)
            
            # Erstelle Ring von Spheres um existierende Sphere
            num_additional = 3  # Anzahl zusätzlicher Spheres
            radius_offset = base_radius * 2.5
            
            for i in range(num_additional):
                angle = (i / num_additional) * 2 * np.pi
                offset = Vector((np.cos(angle), np.sin(angle), 0)) * radius_offset
                new_pos = base_pos + offset
                
                # Surface-Projektion
                if self.surface_bvh:
                    surface_point, surface_normal, _, _ = self.surface_bvh.find_nearest(new_pos)
                    if surface_point:
                        new_pos = surface_point + surface_normal * self.surface_offset
                
                new_sphere = {
                    'location': new_pos,
                    'radius': base_radius * 0.7,  # Etwas kleiner
                    'densification': True,
                    'parent_sphere': sphere_idx
                }
                
                new_spheres.append(new_sphere)
                
                # Verbinde zu Parent-Sphere
                new_sphere_idx = len(existing_spheres) + len(new_spheres) - 1
                new_connections.append((new_sphere_idx, sphere_idx))
        
        solution = FillSolution(
            gap_info=gap,
            strategy=FillStrategy.MESH_DENSIFICATION,
            new_spheres=new_spheres,
            new_connections=new_connections,
            quality_score=len(new_spheres) * 0.6 + gap.stress_level * 0.4
        )
        
        return solution
    
    def _fill_with_surface_projection(self, gap: GapInfo, 
                                    existing_spheres: List[bpy.types.Object]) -> Optional[FillSolution]:
        """Füllt Lücke durch Surface-Projektion"""
        
        if not self.surface_bvh:
            return None
        
        # Sample Punkte auf Surface in Gap-Bereich
        sample_positions = []
        center = gap.center
        sample_radius = gap.size
        
        # Erstelle Grid von Sample-Punkten
        num_samples = max(3, int(gap.size / 2))  # Adaptive Sample-Anzahl
        
        for i in range(num_samples):
            for j in range(num_samples):
                u = (i / (num_samples - 1) - 0.5) * 2  # -1 bis 1
                v = (j / (num_samples - 1) - 0.5) * 2  # -1 bis 1
                
                sample_point = center + Vector((u, v, 0)) * sample_radius
                
                # Projiziere auf Surface
                surface_point, surface_normal, _, _ = self.surface_bvh.find_nearest(sample_point)
                
                if surface_point:
                    projected_pos = surface_point + surface_normal * self.surface_offset
                    sample_positions.append(projected_pos)
        
        # Filtere Sample-Positionen (entferne zu nahe an bestehenden Spheres)
        filtered_positions = []
        min_distance = 3.0  # Mindestabstand
        
        for pos in sample_positions:
            too_close = False
            for sphere in existing_spheres:
                if (pos - Vector(sphere.location)).length < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                filtered_positions.append(pos)
        
        # Erstelle neue Spheres
        new_spheres = []
        for pos in filtered_positions[:5]:  # Max 5 neue Spheres
            new_sphere = {
                'location': pos,
                'radius': self.min_sphere_radius * 1.5,
                'surface_projected': True
            }
            new_spheres.append(new_sphere)
        
        # Erstelle Verbindungen
        new_connections = []
        start_idx = len(existing_spheres)
        
        # Verbinde neue Spheres zu nächsten bestehenden
        for i, new_sphere in enumerate(new_spheres):
            nearby = self._find_nearby_spheres_for_position(new_sphere['location'], existing_spheres, radius=10.0)
            for sphere_idx, _ in nearby[:2]:
                new_connections.append((start_idx + i, sphere_idx))
        
        solution = FillSolution(
            gap_info=gap,
            strategy=FillStrategy.SURFACE_PROJECT,
            new_spheres=new_spheres,
            new_connections=new_connections,
            quality_score=len(new_spheres) * 0.7 + len(filtered_positions) * 0.1
        )
        
        return solution
    
    def _fill_with_hybrid_approach(self, gap: GapInfo, 
                                 existing_spheres: List[bpy.types.Object]) -> Optional[FillSolution]:
        """Füllt Lücke mit Hybrid-Ansatz (Kombination mehrerer Strategien)"""
        
        # Versuche mehrere Strategien und wähle beste
        strategies = [
            FillStrategy.SPHERE_INSERTION,
            FillStrategy.CONNECTION_BRIDGING,
            FillStrategy.PATTERN_EXTENSION
        ]
        
        best_solution = None
        best_score = 0.0
        
        for strategy in strategies:
            temp_gap = GapInfo(
                type=gap.type,
                center=gap.center,
                size=gap.size,
                affected_spheres=gap.affected_spheres.copy(),
                stress_level=gap.stress_level,
                priority=gap.priority,
                suggested_strategy=strategy,
                surface_normal=gap.surface_normal
            )
            
            solution = None
            if strategy == FillStrategy.SPHERE_INSERTION:
                solution = self._fill_with_sphere_insertion(temp_gap, existing_spheres)
            elif strategy == FillStrategy.CONNECTION_BRIDGING:
                solution = self._fill_with_connection_bridging(temp_gap, existing_spheres)
            elif strategy == FillStrategy.PATTERN_EXTENSION:
                solution = self._fill_with_pattern_extension(temp_gap, existing_spheres)
            
            if solution and solution.quality_score > best_score:
                best_solution = solution
                best_score = solution.quality_score
        
        if best_solution:
            best_solution.strategy = FillStrategy.HYBRID_APPROACH
        
        return best_solution
    
    def _find_nearby_spheres_for_position(self, position: Vector, spheres: List[bpy.types.Object], 
                                        radius: float = 15.0) -> List[Tuple[int, float]]:
        """Findet Spheres in der Nähe einer Position"""
        nearby = []
        
        for i, sphere in enumerate(spheres):
            distance = (position - Vector(sphere.location)).length
            if distance <= radius:
                nearby.append((i, distance))
        
        # Sortiere nach Distanz
        nearby.sort(key=lambda x: x[1])
        return nearby
    
    def _calculate_solution_quality(self, gap: GapInfo, new_spheres: List[Dict], 
                                  new_connections: List[Tuple[int, int]]) -> float:
        """Berechnet Qualitäts-Score für Fill-Solution"""
        
        # Basis-Score basierend auf Gap-Eigenschaften
        base_score = gap.priority * gap.size
        
        # Bonus für neue Spheres (aber nicht zu viele)
        sphere_score = len(new_spheres) * 2.0 - (len(new_spheres) ** 2) * 0.1
        
        # Bonus für Verbindungen
        connection_score = len(new_connections) * 1.0
        
        # Stress-Level Berücksichtigung
        stress_score = gap.stress_level * 0.5
        
        total_score = base_score + sphere_score + connection_score + stress_score
        return max(0.0, total_score)
    
    def update_surface_target(self, target_obj: bpy.types.Object):
        """Aktualisiert Ziel-Oberfläche"""
        self.target_surface = target_obj
        
        if target_obj and target_obj.type == 'MESH':
            depsgraph = bpy.context.evaluated_depsgraph_get()
            obj_eval = target_obj.evaluated_get(depsgraph)
            mesh = obj_eval.to_mesh()
            
            self.surface_bvh = BVHTree.FromMesh(mesh)
            debug.info('GAP_FILL', f"Surface BVH updated: {target_obj.name}")


# =========================================
# CONVENIENCE FUNCTIONS
# =========================================

@measure_time
def detect_and_fill_gaps(spheres: List[bpy.types.Object],
                        surface_obj: Optional[bpy.types.Object] = None,
                        stress_data: Optional[Dict[int, float]] = None,
                        auto_fill: bool = True) -> Dict[str, Any]:
    """
    Convenience-Funktion für Gap-Detection und -Filling
    
    Args:
        spheres: Liste von Sphere-Objekten
        surface_obj: Optional Ziel-Oberfläche
        stress_data: Optional Stress-Daten pro Sphere
        auto_fill: Automatisches Füllen der Lücken
        
    Returns:
        Umfassender Bericht über Gaps und Fill-Solutions
    """
    
    # Gap Detection
    detector = GapDetector(surface_obj)
    detected_gaps = detector.detect_gaps(spheres, stress_data)
    
    fill_solutions = []
    if auto_fill and detected_gaps:
        # Gap Filling
        filler = GapFiller(surface_obj)
        fill_solutions = filler.fill_gaps(detected_gaps, spheres)
    
    # Erstelle zusammenfassenden Bericht
    gap_summary = {
        gap_type.value: len([g for g in detected_gaps if g.type == gap_type])
        for gap_type in GapType
    }
    
    strategy_summary = {
        strategy.value: len([s for s in fill_solutions if s.strategy == strategy])
        for strategy in FillStrategy
    }
    
    report = {
        'gap_detection': {
            'total_gaps': len(detected_gaps),
            'gap_types': gap_summary,
            'gaps': detected_gaps
        },
        'gap_filling': {
            'total_solutions': len(fill_solutions),
            'strategies_used': strategy_summary,
            'solutions': fill_solutions,
            'total_new_spheres': sum(len(s.new_spheres) for s in fill_solutions),
            'total_new_connections': sum(len(s.new_connections) for s in fill_solutions)
        },
        'quality_metrics': {
            'avg_gap_priority': sum(g.priority for g in detected_gaps) / len(detected_gaps) if detected_gaps else 0.0,
            'avg_solution_quality': sum(s.quality_score for s in fill_solutions) / len(fill_solutions) if fill_solutions else 0.0,
            'coverage_improvement': len(fill_solutions) / max(len(detected_gaps), 1)
        }
    }
    
    return report


def apply_fill_solutions(fill_solutions: List[FillSolution], 
                        collection_name: str = "GapFill_Spheres") -> List[bpy.types.Object]:
    """
    Wendet Fill-Solutions in Blender an (erstellt tatsächliche Objekte)
    
    Args:
        fill_solutions: Liste von Fill-Solutions
        collection_name: Name der Collection für neue Objekte
        
    Returns:
        Liste der erstellten Sphere-Objekte
    """
    
    if not fill_solutions:
        return []
    
    # Erstelle oder hole Collection
    if collection_name in bpy.data.collections:
        collection = bpy.data.collections[collection_name]
    else:
        collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(collection)
    
    created_spheres = []
    
    for solution in fill_solutions:
        debug.info('GAP_FILL', f"Applying {solution.strategy.value} solution with {len(solution.new_spheres)} spheres")
        
        # Erstelle neue Spheres
        for i, sphere_data in enumerate(solution.new_spheres):
            location = sphere_data['location']
            radius = sphere_data['radius']
            
            # Erstelle Sphere-Mesh
            bpy.ops.mesh.primitive_uv_sphere_add(location=location, radius=radius)
            new_sphere = bpy.context.active_object
            new_sphere.name = f"GapFill_Sphere_{len(created_spheres):03d}"
            
            # Custom Properties übertragen
            for key, value in sphere_data.items():
                if key not in ['location', 'radius']:
                    new_sphere[key] = value
            
            # Zu Collection hinzufügen
            for coll in new_sphere.users_collection:
                coll.objects.unlink(new_sphere)
            collection.objects.link(new_sphere)
            
            created_spheres.append(new_sphere)
    
    debug.info('GAP_FILL', f"Created {len(created_spheres)} gap-fill spheres")
    return created_spheres
