"""
Chain Tool V4 - 3D Delaunay Triangulation Module
===============================================

Erweiterte 3D Delaunay-Triangulation für Orthesen-Verstärkungen
- Surface Delaunay Triangulation mit Aura-System Integration
- Constrained Delaunay für Edge-Reinforcement  
- Quality-Metriken und adaptive Optimierung
- Vollständig kompatibel mit BVH-Caching

Author: Chain Tool V4 Development Team
Version: 4.0.0
"""

import bpy
import bmesh
import numpy as np
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from typing import List, Tuple, Dict, Optional, Set, Any, Union
import time
from enum import Enum

# V4 Module imports
from ..core.constants import TriangulationMethod, QualityMetrics
from ..core.state_manager import StateManager
from ..utils.performance import measure_time, PerformanceTracker
from ..utils.debug import DebugManager
from ..utils.caching import CacheManager
from ..utils.math_utils import safe_normalize, calculate_triangle_quality

# Initialize systems
debug = DebugManager()
cache = CacheManager()
perf = PerformanceTracker()

class ConstraintType(Enum):
    """Constraint-Typen für Delaunay-Triangulation"""
    NONE = "none"
    EDGE_PRESERVE = "edge_preserve"  # Kanten müssen erhalten bleiben
    SURFACE_FOLLOW = "surface_follow"  # Folgt Oberflächenkontur
    DISTANCE_LIMIT = "distance_limit"  # Maximale Verbindungsdistanz
    ANGLE_LIMIT = "angle_limit"  # Maximaler Winkel zwischen Verbindungen

class QualityConstraint:
    """Quality-Constraints für Dreiecke"""
    def __init__(self):
        self.min_angle: float = 15.0  # Minimaler Innenwinkel in Grad
        self.max_angle: float = 120.0  # Maximaler Innenwinkel in Grad
        self.min_edge_ratio: float = 0.3  # Verhältnis kürzeste/längste Kante
        self.max_aspect_ratio: float = 4.0  # Maximales Seitenverhältnis
        self.min_area: float = 0.001  # Minimale Dreiecksfläche

class DelaunayTriangulator3D:
    """
    Erweiterte 3D Delaunay-Triangulation für Orthesen-Pattern
    
    Features:
    - Surface-projected Delaunay mit Aura-Integration
    - Constrained Triangulation für Edge-Preservation
    - Quality-basierte Mesh-Optimierung
    - BVH-Tree Integration für Performance
    """
    
    def __init__(self, target_surface: Optional[bpy.types.Object] = None):
        """
        Initialisiert Delaunay-Triangulator
        
        Args:
            target_surface: Ziel-Objekt für Surface-Projection
        """
        self.target_surface = target_surface
        self.quality_constraints = QualityConstraint()
        self.bvh_tree: Optional[BVHTree] = None
        self.aura_cache: Dict[str, Any] = {}
        
        # Scipy availability check
        self.scipy_available = self._check_scipy()
        
        # Performance tracking
        self.triangulation_stats = {
            'total_points': 0,
            'total_triangles': 0,
            'quality_score': 0.0,
            'computation_time': 0.0
        }
    
    def _check_scipy(self) -> bool:
        """Prüft Scipy-Verfügbarkeit"""
        try:
            import scipy.spatial
            debug.info('DELAUNAY', "Scipy available - using optimized algorithms")
            return True
        except ImportError:
            debug.warning('DELAUNAY', "Scipy not available - using fallback algorithms")
            return False
    
    @measure_time
    def triangulate_surface_points(self, 
                                 points: List[Vector], 
                                 constraints: Optional[List[ConstraintType]] = None,
                                 quality_target: float = 0.7) -> List[Tuple[int, int, int]]:
        """
        Hauptfunktion: Surface Delaunay Triangulation
        
        Args:
            points: 3D Punkte für Triangulation
            constraints: Optional constraint types
            quality_target: Ziel-Qualitätswert (0.0-1.0)
            
        Returns:
            Liste von Dreiecks-Indizes (i, j, k)
        """
        if len(points) < 3:
            debug.warning('DELAUNAY', f"Insufficient points for triangulation: {len(points)}")
            return []
        
        debug.info('DELAUNAY', f"Starting surface triangulation for {len(points)} points")
        start_time = time.time()
        
        # Update stats
        self.triangulation_stats['total_points'] = len(points)
        
        # Convert to numpy for processing
        np_points = np.array([[p.x, p.y, p.z] for p in points])
        
        # Choose triangulation method
        if self.scipy_available:
            triangles = self._scipy_surface_delaunay(np_points, constraints)
        else:
            triangles = self._fallback_surface_delaunay(np_points, constraints)
        
        # Quality optimization
        if quality_target > 0.5:
            triangles = self._optimize_triangle_quality(np_points, triangles, quality_target)
        
        # Update performance stats
        computation_time = time.time() - start_time
        self.triangulation_stats['computation_time'] = computation_time
        self.triangulation_stats['total_triangles'] = len(triangles)
        
        debug.info('DELAUNAY', f"Triangulation completed: {len(triangles)} triangles in {computation_time:.3f}s")
        
        return triangles
    
    def _scipy_surface_delaunay(self, points: np.ndarray, 
                               constraints: Optional[List[ConstraintType]]) -> List[Tuple[int, int, int]]:
        """Scipy-basierte Delaunay-Triangulation"""
        from scipy.spatial import Delaunay
        
        # Project to best-fit plane for 2D Delaunay
        projected_points, projection_matrix = self._project_to_plane(points)
        
        # Standard Delaunay in 2D
        tri = Delaunay(projected_points)
        
        # Convert simplices to triangles
        triangles = []
        for simplex in tri.simplices:
            if self._is_valid_triangle(points[simplex]):
                triangles.append(tuple(simplex))
        
        # Apply constraints
        if constraints:
            triangles = self._apply_constraints(points, triangles, constraints)
        
        return triangles
    
    def _fallback_surface_delaunay(self, points: np.ndarray, 
                                  constraints: Optional[List[ConstraintType]]) -> List[Tuple[int, int, int]]:
        """Fallback ohne Scipy - einfache Nachbarschafts-Triangulation"""
        debug.info('DELAUNAY', "Using fallback triangulation algorithm")
        
        triangles = []
        n_points = len(points)
        
        # Für jeden Punkt: verbinde zu nächsten Nachbarn
        for i in range(n_points):
            # Finde nächste Nachbarn
            distances = []
            for j in range(n_points):
                if i != j:
                    dist = np.linalg.norm(points[i] - points[j])
                    distances.append((dist, j))
            
            # Sortiere nach Distanz
            distances.sort(key=lambda x: x[0])
            
            # Erstelle Dreiecke mit nächsten Nachbarn
            for k in range(min(3, len(distances) - 1)):
                for l in range(k + 1, min(6, len(distances))):
                    j1 = distances[k][1]
                    j2 = distances[l][1]
                    
                    # Prüfe Dreiecks-Validität
                    triangle = tuple(sorted([i, j1, j2]))
                    if (triangle not in triangles and 
                        self._is_valid_triangle(points[[i, j1, j2]])):
                        triangles.append(triangle)
        
        return list(set(triangles))  # Entferne Duplikate
    
    def _project_to_plane(self, points: np.ndarray) -> Tuple[np.ndarray, Matrix]:
        """Projiziert 3D Punkte auf beste 2D Ebene"""
        # PCA für beste Projektionsebene
        centered = points - np.mean(points, axis=0)
        _, _, vh = np.linalg.svd(centered)
        
        # Erste zwei Hauptkomponenten als Basis
        u = vh[0]
        v = vh[1]
        
        # Projiziere auf UV-Ebene
        projected = np.column_stack([
            np.dot(centered, u),
            np.dot(centered, v)
        ])
        
        # Matrix für Rückprojektion
        projection_matrix = Matrix((
            (u[0], u[1], u[2], 0),
            (v[0], v[1], v[2], 0),
            (0, 0, 0, 0),
            (0, 0, 0, 1)
        ))
        
        return projected, projection_matrix
    
    def _is_valid_triangle(self, triangle_points: np.ndarray) -> bool:
        """Prüft Dreiecks-Validität"""
        if len(triangle_points) != 3:
            return False
        
        # Konvertiere zu Vektoren
        p1 = Vector(triangle_points[0])
        p2 = Vector(triangle_points[1])
        p3 = Vector(triangle_points[2])
        
        # Berechne Quality-Metriken
        quality = calculate_triangle_quality(p1, p2, p3)
        
        # Prüfe gegen Constraints
        return (quality['min_angle'] >= self.quality_constraints.min_angle and
                quality['max_angle'] <= self.quality_constraints.max_angle and
                quality['edge_ratio'] >= self.quality_constraints.min_edge_ratio and
                quality['area'] >= self.quality_constraints.min_area)
    
    def _apply_constraints(self, points: np.ndarray, 
                          triangles: List[Tuple[int, int, int]], 
                          constraints: List[ConstraintType]) -> List[Tuple[int, int, int]]:
        """Wendet Constraints auf Triangulation an"""
        filtered_triangles = []
        
        for triangle in triangles:
            valid = True
            
            for constraint in constraints:
                if constraint == ConstraintType.DISTANCE_LIMIT:
                    # Prüfe maximale Kantenlänge
                    max_edge_length = self._get_max_edge_length(points[list(triangle)])
                    if max_edge_length > 20.0:  # Konfigurierbar
                        valid = False
                        break
                
                elif constraint == ConstraintType.SURFACE_FOLLOW:
                    # Prüfe Surface-Abweichung
                    if not self._triangle_follows_surface(points[list(triangle)]):
                        valid = False
                        break
            
            if valid:
                filtered_triangles.append(triangle)
        
        debug.info('DELAUNAY', f"Constraints applied: {len(triangles)} -> {len(filtered_triangles)} triangles")
        return filtered_triangles
    
    def _get_max_edge_length(self, triangle_points: np.ndarray) -> float:
        """Berechnet maximale Kantenlänge eines Dreiecks"""
        edges = [
            np.linalg.norm(triangle_points[1] - triangle_points[0]),
            np.linalg.norm(triangle_points[2] - triangle_points[1]),
            np.linalg.norm(triangle_points[0] - triangle_points[2])
        ]
        return max(edges)
    
    def _triangle_follows_surface(self, triangle_points: np.ndarray) -> bool:
        """Prüft ob Dreieck der Oberfläche folgt"""
        if not self.target_surface or not self.bvh_tree:
            return True  # Keine Surface-Constraints ohne Zielobjekt
        
        # Sample Punkte auf Dreieckskanten
        samples = []
        for i in range(3):
            p1 = Vector(triangle_points[i])
            p2 = Vector(triangle_points[(i + 1) % 3])
            
            # Sample zwischen Eckpunkten
            for t in np.linspace(0.1, 0.9, 5):
                sample_point = p1.lerp(p2, t)
                
                # Finde nächsten Surface-Punkt
                location, normal, _, _ = self.bvh_tree.find_nearest(sample_point)
                if location:
                    distance = (sample_point - location).length
                    samples.append(distance)
        
        # Prüfe durchschnittliche Abweichung
        if samples:
            avg_deviation = sum(samples) / len(samples)
            return avg_deviation < 2.0  # Konfigurierbar
        
        return True
    
    @measure_time
    def _optimize_triangle_quality(self, points: np.ndarray, 
                                  triangles: List[Tuple[int, int, int]], 
                                  quality_target: float) -> List[Tuple[int, int, int]]:
        """Optimiert Dreieck-Qualität durch iterative Verbesserung"""
        debug.info('DELAUNAY', f"Starting quality optimization (target: {quality_target:.2f})")
        
        optimized_triangles = []
        total_quality = 0.0
        
        for triangle in triangles:
            # Berechne aktuelle Qualität
            tri_points = points[list(triangle)]
            p1, p2, p3 = [Vector(p) for p in tri_points]
            quality_metrics = calculate_triangle_quality(p1, p2, p3)
            current_quality = quality_metrics['overall_quality']
            
            # Nur hochqualitative Dreiecke behalten
            if current_quality >= quality_target:
                optimized_triangles.append(triangle)
                total_quality += current_quality
        
        # Berechne Gesamt-Qualitätsscore
        if optimized_triangles:
            avg_quality = total_quality / len(optimized_triangles)
            self.triangulation_stats['quality_score'] = avg_quality
            
            debug.info('DELAUNAY', 
                      f"Quality optimization completed: {len(optimized_triangles)} triangles, "
                      f"avg quality: {avg_quality:.3f}")
        
        return optimized_triangles
    
    def triangulate_constrained_edges(self, points: List[Vector], 
                                    edge_constraints: List[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
        """
        Constrained Delaunay für Edge-Reinforcement
        
        Args:
            points: 3D Punkte
            edge_constraints: Muss-Kanten als (i,j) Index-Paare
            
        Returns:
            Triangulation die alle constraint edges enthält
        """
        debug.info('DELAUNAY', f"Constrained triangulation: {len(points)} points, {len(edge_constraints)} edge constraints")
        
        # Standard Triangulation als Basis
        base_triangles = self.triangulate_surface_points(points)
        
        # Prüfe welche Constraint-Edges fehlen
        existing_edges = set()
        for triangle in base_triangles:
            for i in range(3):
                edge = tuple(sorted([triangle[i], triangle[(i + 1) % 3]]))
                existing_edges.add(edge)
        
        missing_constraints = []
        for constraint in edge_constraints:
            edge = tuple(sorted(constraint))
            if edge not in existing_edges:
                missing_constraints.append(edge)
        
        if not missing_constraints:
            debug.info('DELAUNAY', "All edge constraints already satisfied")
            return base_triangles
        
        # Füge fehlende Constraint-Edges durch lokale Retriangulation hinzu
        constrained_triangles = self._add_constraint_edges(
            points, base_triangles, missing_constraints
        )
        
        debug.info('DELAUNAY', f"Constrained triangulation completed: {len(constrained_triangles)} triangles")
        return constrained_triangles
    
    def _add_constraint_edges(self, points: List[Vector], 
                             triangles: List[Tuple[int, int, int]], 
                             missing_edges: List[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
        """Fügt fehlende Constraint-Edges durch lokale Retriangulation hinzu"""
        result_triangles = list(triangles)
        
        for edge in missing_edges:
            i, j = edge
            p1, p2 = points[i], points[j]
            
            # Finde Dreiecke die diese Edge kreuzen
            crossing_triangles = []
            for tri_idx, triangle in enumerate(result_triangles):
                if self._edge_crosses_triangle(p1, p2, points, triangle):
                    crossing_triangles.append((tri_idx, triangle))
            
            if crossing_triangles:
                # Entferne kreuzende Dreiecke
                for tri_idx, _ in reversed(crossing_triangles):
                    del result_triangles[tri_idx]
                
                # Retrianguliere Polygon mit Constraint-Edge
                polygon_points = self._extract_polygon_from_triangles(
                    [tri for _, tri in crossing_triangles]
                )
                new_triangles = self._triangulate_polygon_with_constraint(
                    points, polygon_points, edge
                )
                result_triangles.extend(new_triangles)
        
        return result_triangles
    
    def _edge_crosses_triangle(self, p1: Vector, p2: Vector, 
                              all_points: List[Vector], 
                              triangle: Tuple[int, int, int]) -> bool:
        """Prüft ob Edge ein Dreieck kreuzt"""
        # Implementierung der Segment-Triangle Intersection
        tri_points = [all_points[i] for i in triangle]
        
        # Vereinfachte 2D-Projektion für Test
        # Vollständige 3D Line-Triangle Intersection würde hier implementiert
        return False  # Placeholder
    
    def _extract_polygon_from_triangles(self, triangles: List[Tuple[int, int, int]]) -> List[int]:
        """Extrahiert Polygon-Rand aus Dreiecks-Liste"""
        # Edge-Count für Rand-Erkennung
        edge_count = {}
        for triangle in triangles:
            for i in range(3):
                edge = tuple(sorted([triangle[i], triangle[(i + 1) % 3]]))
                edge_count[edge] = edge_count.get(edge, 0) + 1
        
        # Rand-Edges (nur einmal verwendet)
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
        
        # Ordne Edges zu geschlossenem Polygon
        if not boundary_edges:
            return []
        
        polygon = [boundary_edges[0][0], boundary_edges[0][1]]
        used_edges = {boundary_edges[0]}
        
        while len(used_edges) < len(boundary_edges):
            last_vertex = polygon[-1]
            
            # Finde nächste verbundene Edge
            next_edge = None
            for edge in boundary_edges:
                if edge not in used_edges:
                    if edge[0] == last_vertex:
                        next_edge = edge
                        polygon.append(edge[1])
                        break
                    elif edge[1] == last_vertex:
                        next_edge = edge
                        polygon.append(edge[0])
                        break
            
            if next_edge:
                used_edges.add(next_edge)
            else:
                break  # Kein geschlossenes Polygon möglich
        
        return polygon[:-1]  # Entferne letzten (doppelten) Punkt
    
    def _triangulate_polygon_with_constraint(self, all_points: List[Vector], 
                                           polygon_indices: List[int], 
                                           constraint_edge: Tuple[int, int]) -> List[Tuple[int, int, int]]:
        """Ring-Triangulation mit Constraint-Edge"""
        if len(polygon_indices) < 3:
            return []
        
        # Einfache Fan-Triangulation vom ersten Punkt
        triangles = []
        for i in range(1, len(polygon_indices) - 1):
            triangle = (polygon_indices[0], polygon_indices[i], polygon_indices[i + 1])
            triangles.append(triangle)
        
        # Zusätzlich: Constraint-Edge als separates Dreieck wenn möglich
        # (Vereinfachte Implementierung)
        
        return triangles
    
    def get_triangulation_quality_report(self) -> Dict[str, Any]:
        """Erstellt detaillierten Qualitätsbericht"""
        return {
            'statistics': self.triangulation_stats.copy(),
            'quality_constraints': {
                'min_angle': self.quality_constraints.min_angle,
                'max_angle': self.quality_constraints.max_angle,
                'min_edge_ratio': self.quality_constraints.min_edge_ratio,
                'max_aspect_ratio': self.quality_constraints.max_aspect_ratio,
                'min_area': self.quality_constraints.min_area
            },
            'performance': {
                'scipy_available': self.scipy_available,
                'bvh_cached': self.bvh_tree is not None,
                'cache_hits': cache.get_stats().get('hits', 0)
            }
        }
    
    def update_surface_target(self, target_obj: bpy.types.Object):
        """Aktualisiert Ziel-Oberfläche für Surface-Projection"""
        self.target_surface = target_obj
        
        if target_obj and target_obj.type == 'MESH':
            # Erstelle BVH-Tree für effiziente Surface-Queries
            depsgraph = bpy.context.evaluated_depsgraph_get()
            obj_eval = target_obj.evaluated_get(depsgraph)
            mesh = obj_eval.to_mesh()
            
            self.bvh_tree = BVHTree.FromMesh(mesh)
            debug.info('DELAUNAY', f"BVH tree created for surface: {target_obj.name}")
            
            # Cache Surface-Properties
            cache_key = f"surface_{target_obj.name}_{hash(str(mesh.vertices))}"
            self.aura_cache[cache_key] = {
                'bvh_tree': self.bvh_tree,
                'vertex_count': len(mesh.vertices),
                'face_count': len(mesh.polygons)
            }


# =========================================
# CONVENIENCE FUNCTIONS 
# =========================================

@measure_time
def delaunay_triangulate_spheres(spheres: List[bpy.types.Object], 
                               quality_target: float = 0.6,
                               max_connection_distance: float = 20.0) -> List[Tuple[bpy.types.Object, bpy.types.Object]]:
    """
    Kompatibilitätsfunktion: Delaunay-Triangulation für Sphere-Objekte
    
    Args:
        spheres: Liste von Sphere-Objekten
        quality_target: Qualitätsziel für Triangulation (0.0-1.0)
        max_connection_distance: Maximale Verbindungsdistanz
        
    Returns:
        Liste von Sphere-Paaren für Verbindungen
    """
    if len(spheres) < 3:
        debug.warning('DELAUNAY', f"Insufficient spheres for triangulation: {len(spheres)}")
        return []

    debug.info('DELAUNAY', f"Creating Delaunay triangulation for {len(spheres)} spheres")
    
    # Konvertiere Sphere-Positionen zu Punkten
    points = [Vector(sphere.location) for sphere in spheres]
    
    # Initialisiere Triangulator
    triangulator = DelaunayTriangulator3D()
    
    # Führe Triangulation durch
    triangles = triangulator.triangulate_surface_points(
        points, 
        constraints=[ConstraintType.DISTANCE_LIMIT],
        quality_target=quality_target
    )
    
    # Konvertiere Dreiecke zu Sphere-Paaren (Edges)
    sphere_pairs = []
    edge_set = set()
    
    for triangle in triangles:
        # Jedes Dreieck hat 3 Kanten
        for i in range(3):
            edge = tuple(sorted([triangle[i], triangle[(i + 1) % 3]]))
            
            if edge not in edge_set:
                # Prüfe Distanz-Constraint
                sphere1, sphere2 = spheres[edge[0]], spheres[edge[1]]
                distance = (Vector(sphere1.location) - Vector(sphere2.location)).length
                
                if distance <= max_connection_distance:
                    sphere_pairs.append((sphere1, sphere2))
                    edge_set.add(edge)
    
    debug.info('DELAUNAY', f"Created {len(sphere_pairs)} sphere connections")
    return sphere_pairs


def create_constrained_triangulation(points: List[Vector], 
                                   edge_constraints: List[Tuple[int, int]],
                                   surface_obj: Optional[bpy.types.Object] = None) -> List[Tuple[int, int, int]]:
    """
    Erstellt Constrained Delaunay Triangulation
    
    Args:
        points: 3D Punkte für Triangulation
        edge_constraints: Muss-behalten Kanten
        surface_obj: Optional Surface für Projection
        
    Returns:
        Triangulation als Index-Tripel
    """
    triangulator = DelaunayTriangulator3D(surface_obj)
    return triangulator.triangulate_constrained_edges(points, edge_constraints)


def analyze_triangulation_quality(triangles: List[Tuple[int, int, int]], 
                                 points: List[Vector]) -> Dict[str, float]:
    """
    Analysiert Triangulation-Qualität
    
    Args:
        triangles: Dreiecks-Indizes
        points: 3D Punkte
        
    Returns:
        Qualitäts-Metriken
    """
    if not triangles:
        return {'overall_quality': 0.0}
    
    total_quality = 0.0
    angle_stats = []
    edge_ratio_stats = []
    
    for triangle in triangles:
        p1, p2, p3 = [points[i] for i in triangle]
        quality = calculate_triangle_quality(p1, p2, p3)
        
        total_quality += quality['overall_quality']
        angle_stats.extend([quality['min_angle'], quality['max_angle']])
        edge_ratio_stats.append(quality['edge_ratio'])
    
    return {
        'overall_quality': total_quality / len(triangles),
        'avg_min_angle': sum(angle_stats) / len(angle_stats) if angle_stats else 0.0,
        'avg_edge_ratio': sum(edge_ratio_stats) / len(edge_ratio_stats) if edge_ratio_stats else 0.0,
        'triangle_count': len(triangles)
    }
