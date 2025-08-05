"""
Chain Tool V4 - Connection Optimizer Module
==========================================

Graph-basierte Verbindungsoptimierung für Orthesen-Verstärkungen
- Minimum Spanning Tree für optimale Konnektivität
- Stress-basierte Pfadfindung für biomechanische Optimierung
- Multi-Kriterien Optimierung (Gewicht, Stabilität, Druckbarkeit)
- Integration mit Aura-System und Surface-Projection

Author: Chain Tool V4 Development Team
Version: 4.0.0
"""

import bpy
import bmesh
import numpy as np
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from typing import List, Tuple, Dict, Optional, Set, Any, Union
import heapq
import time
from enum import Enum
from dataclasses import dataclass, field

# V4 Module imports
from ..core.constants import OptimizationMethod, ConnectionType
from ..core.state_manager import StateManager
from ..utils.performance import measure_time, PerformanceTracker
from ..utils.debug import DebugManager
from ..utils.caching import CacheManager
from ..utils.math_utils import safe_normalize, calculate_distance_3d

# Initialize systems
debug = DebugManager()
cache = CacheManager()
perf = PerformanceTracker()

class OptimizationCriteria(Enum):
    """Optimierungs-Kriterien für Verbindungen"""
    MINIMUM_WEIGHT = "minimum_weight"          # Minimales Gesamtgewicht
    MAXIMUM_STABILITY = "maximum_stability"    # Maximale strukturelle Stabilität
    MINIMUM_STRESS = "minimum_stress"          # Minimale Spannungskonzentration
    PRINT_FRIENDLY = "print_friendly"          # 3D-Druck optimiert
    BIOMECHANICAL = "biomechanical"           # Biomechanisch optimiert
    HYBRID = "hybrid"                         # Multi-Kriterien Balance

class PathfindingAlgorithm(Enum):
    """Algorithmen für Pfadfindung"""
    DIJKSTRA = "dijkstra"                     # Kürzeste Pfade
    A_STAR = "a_star"                         # A* mit Heuristik
    MINIMUM_SPANNING_TREE = "mst"             # MST für Konnektivität
    STEINER_TREE = "steiner"                  # Optimaler Steiner-Baum
    FORCE_DIRECTED = "force_directed"         # Kraftbasierte Optimierung

@dataclass
class ConnectionNode:
    """Repräsentiert einen Knoten im Verbindungsgraph"""
    id: int
    position: Vector
    sphere_obj: Optional[bpy.types.Object] = None
    radius: float = 1.0
    strength_requirement: float = 1.0
    stress_capacity: float = 100.0
    connections: Set[int] = field(default_factory=set)
    is_anchor: bool = False
    biomech_importance: float = 1.0

@dataclass
class ConnectionEdge:
    """Repräsentiert eine Verbindung zwischen Knoten"""
    node_a: int
    node_b: int
    weight: float
    length: float
    stress_factor: float = 1.0
    printability_score: float = 1.0
    surface_deviation: float = 0.0
    biomech_efficiency: float = 1.0
    is_critical: bool = False

class ConnectionGraph:
    """Graph-Struktur für Verbindungsoptimierung"""
    
    def __init__(self):
        self.nodes: Dict[int, ConnectionNode] = {}
        self.edges: Dict[Tuple[int, int], ConnectionEdge] = {}
        self.adjacency_list: Dict[int, Set[int]] = {}
        
    def add_node(self, node: ConnectionNode):
        """Fügt Knoten zum Graph hinzu"""
        self.nodes[node.id] = node
        if node.id not in self.adjacency_list:
            self.adjacency_list[node.id] = set()
    
    def add_edge(self, edge: ConnectionEdge):
        """Fügt Kante zum Graph hinzu"""
        edge_key = tuple(sorted([edge.node_a, edge.node_b]))
        self.edges[edge_key] = edge
        
        # Update adjacency list
        self.adjacency_list[edge.node_a].add(edge.node_b)
        self.adjacency_list[edge.node_b].add(edge.node_a)
    
    def get_edge(self, node_a: int, node_b: int) -> Optional[ConnectionEdge]:
        """Holt Kante zwischen zwei Knoten"""
        edge_key = tuple(sorted([node_a, node_b]))
        return self.edges.get(edge_key)
    
    def get_neighbors(self, node_id: int) -> Set[int]:
        """Holt Nachbarn eines Knotens"""
        return self.adjacency_list.get(node_id, set())

class ConnectionOptimizer:
    """
    Hauptklasse für Verbindungsoptimierung
    
    Features:
    - Multi-Kriterien Optimierung
    - Graph-basierte Algorithmen (MST, Dijkstra, A*)
    - Stress-Analyse Integration
    - Surface-Aware Pathfinding
    - Biomechanische Optimierung
    """
    
    def __init__(self, target_surface: Optional[bpy.types.Object] = None):
        """
        Initialisiert Connection Optimizer
        
        Args:
            target_surface: Ziel-Oberfläche für Surface-Projection
        """
        self.target_surface = target_surface
        self.bvh_tree: Optional[BVHTree] = None
        self.graph = ConnectionGraph()
        
        # Optimization parameters
        self.optimization_weights = {
            'weight': 0.3,
            'stability': 0.25, 
            'stress': 0.2,
            'printability': 0.15,
            'biomechanics': 0.1
        }
        
        # Performance tracking
        self.optimization_stats = {
            'total_nodes': 0,
            'total_edges': 0,
            'optimization_time': 0.0,
            'algorithm_used': '',
            'improvement_factor': 1.0
        }
        
        # Surface projection cache
        self.surface_cache: Dict[str, Vector] = {}
        
        # Initialize BVH if surface available
        if target_surface:
            self.update_surface_target(target_surface)
    
    @measure_time
    def optimize_connections(self, 
                           spheres: List[bpy.types.Object],
                           criteria: OptimizationCriteria = OptimizationCriteria.HYBRID,
                           algorithm: PathfindingAlgorithm = PathfindingAlgorithm.MINIMUM_SPANNING_TREE,
                           max_connections_per_node: int = 6) -> List[Tuple[bpy.types.Object, bpy.types.Object]]:
        """
        Hauptfunktion: Optimiert Verbindungen zwischen Spheres
        
        Args:
            spheres: Liste von Sphere-Objekten
            criteria: Optimierungskriterium
            algorithm: Verwendeter Algorithmus
            max_connections_per_node: Max. Verbindungen pro Knoten
            
        Returns:
            Optimierte Liste von Sphere-Paaren
        """
        if len(spheres) < 2:
            debug.warning('OPTIMIZER', f"Insufficient spheres for optimization: {len(spheres)}")
            return []
        
        debug.info('OPTIMIZER', f"Starting connection optimization for {len(spheres)} spheres")
        debug.info('OPTIMIZER', f"Criteria: {criteria.value}, Algorithm: {algorithm.value}")
        
        start_time = time.time()
        
        # Step 1: Build graph from spheres
        self._build_graph_from_spheres(spheres)
        
        # Step 2: Calculate all possible connections
        self._calculate_potential_connections(criteria)
        
        # Step 3: Apply optimization algorithm
        if algorithm == PathfindingAlgorithm.MINIMUM_SPANNING_TREE:
            optimized_edges = self._minimum_spanning_tree()
        elif algorithm == PathfindingAlgorithm.DIJKSTRA:
            optimized_edges = self._dijkstra_optimization()
        elif algorithm == PathfindingAlgorithm.A_STAR:
            optimized_edges = self._astar_optimization()
        elif algorithm == PathfindingAlgorithm.FORCE_DIRECTED:
            optimized_edges = self._force_directed_optimization()
        else:
            optimized_edges = self._minimum_spanning_tree()  # Fallback
        
        # Step 4: Apply connection limits
        final_edges = self._apply_connection_limits(optimized_edges, max_connections_per_node)
        
        # Step 5: Convert back to sphere pairs
        sphere_pairs = self._edges_to_sphere_pairs(final_edges)
        
        # Update stats
        optimization_time = time.time() - start_time
        self.optimization_stats.update({
            'total_nodes': len(self.graph.nodes),
            'total_edges': len(final_edges),
            'optimization_time': optimization_time,
            'algorithm_used': algorithm.value,
            'improvement_factor': self._calculate_improvement_factor(len(spheres), len(final_edges))
        })
        
        debug.info('OPTIMIZER', f"Optimization completed: {len(final_edges)} connections in {optimization_time:.3f}s")
        
        return sphere_pairs
    
    def _build_graph_from_spheres(self, spheres: List[bpy.types.Object]):
        """Erstellt Graph aus Sphere-Objekten"""
        self.graph = ConnectionGraph()
        
        for i, sphere in enumerate(spheres):
            # Extract sphere properties
            radius = sphere.get("kette_radius", max(sphere.dimensions) / 2.0)
            strength = sphere.get("strength_requirement", 1.0)
            biomech_importance = sphere.get("biomech_importance", 1.0)
            is_anchor = sphere.get("is_anchor", False)
            
            node = ConnectionNode(
                id=i,
                position=Vector(sphere.location),
                sphere_obj=sphere,
                radius=radius,
                strength_requirement=strength,
                is_anchor=is_anchor,
                biomech_importance=biomech_importance
            )
            
            self.graph.add_node(node)
        
        debug.info('OPTIMIZER', f"Graph built with {len(self.graph.nodes)} nodes")
    
    def _calculate_potential_connections(self, criteria: OptimizationCriteria):
        """Berechnet alle potentiellen Verbindungen mit Gewichtung"""
        nodes = list(self.graph.nodes.values())
        
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node_a, node_b = nodes[i], nodes[j]
                
                # Calculate basic metrics
                distance = calculate_distance_3d(node_a.position, node_b.position)
                
                # Skip if too far
                max_distance = 30.0  # Konfigurierbar
                if distance > max_distance:
                    continue
                
                # Calculate edge weight based on criteria
                weight = self._calculate_edge_weight(node_a, node_b, distance, criteria)
                
                # Calculate additional metrics
                stress_factor = self._calculate_stress_factor(node_a, node_b)
                printability = self._calculate_printability_score(node_a, node_b)
                surface_dev = self._calculate_surface_deviation(node_a.position, node_b.position)
                biomech_eff = self._calculate_biomech_efficiency(node_a, node_b)
                
                edge = ConnectionEdge(
                    node_a=node_a.id,
                    node_b=node_b.id,
                    weight=weight,
                    length=distance,
                    stress_factor=stress_factor,
                    printability_score=printability,
                    surface_deviation=surface_dev,
                    biomech_efficiency=biomech_eff
                )
                
                self.graph.add_edge(edge)
        
        debug.info('OPTIMIZER', f"Calculated {len(self.graph.edges)} potential connections")
    
    def _calculate_edge_weight(self, node_a: ConnectionNode, node_b: ConnectionNode, 
                              distance: float, criteria: OptimizationCriteria) -> float:
        """Berechnet Kantengewicht basierend auf Kriterien"""
        
        if criteria == OptimizationCriteria.MINIMUM_WEIGHT:
            # Einfach: Distanz als Gewicht
            return distance
        
        elif criteria == OptimizationCriteria.MAXIMUM_STABILITY:
            # Stabilität: Kurze, dicke Verbindungen bevorzugen
            radius_factor = (node_a.radius + node_b.radius) / 2.0
            return distance / max(radius_factor, 0.1)
        
        elif criteria == OptimizationCriteria.MINIMUM_STRESS:
            # Stress: Gleichmäßige Lastverteilung
            stress_balance = abs(node_a.strength_requirement - node_b.strength_requirement)
            return distance * (1.0 + stress_balance)
        
        elif criteria == OptimizationCriteria.PRINT_FRIENDLY:
            # 3D-Druck: Horizontale Verbindungen bevorzugen
            height_diff = abs(node_a.position.z - node_b.position.z)
            angle_penalty = height_diff / distance if distance > 0 else 0
            return distance * (1.0 + angle_penalty)
        
        elif criteria == OptimizationCriteria.BIOMECHANICAL:
            # Biomechanik: Wichtige Knoten stärker verbinden
            importance_factor = (node_a.biomech_importance + node_b.biomech_importance) / 2.0
            return distance / max(importance_factor, 0.1)
        
        else:  # HYBRID
            # Multi-Kriterien gewichtete Summe
            weight_component = distance * self.optimization_weights['weight']
            
            radius_factor = (node_a.radius + node_b.radius) / 2.0
            stability_component = (distance / max(radius_factor, 0.1)) * self.optimization_weights['stability']
            
            stress_balance = abs(node_a.strength_requirement - node_b.strength_requirement)
            stress_component = (distance * (1.0 + stress_balance)) * self.optimization_weights['stress']
            
            height_diff = abs(node_a.position.z - node_b.position.z)
            angle_penalty = height_diff / distance if distance > 0 else 0
            print_component = (distance * (1.0 + angle_penalty)) * self.optimization_weights['printability']
            
            importance_factor = (node_a.biomech_importance + node_b.biomech_importance) / 2.0
            biomech_component = (distance / max(importance_factor, 0.1)) * self.optimization_weights['biomechanics']
            
            return weight_component + stability_component + stress_component + print_component + biomech_component
    
    def _calculate_stress_factor(self, node_a: ConnectionNode, node_b: ConnectionNode) -> float:
        """Berechnet Stress-Faktor für Verbindung"""
        # Vereinfachte Stress-Berechnung basierend auf Knoteneigenschaften
        combined_strength = node_a.strength_requirement + node_b.strength_requirement
        combined_capacity = node_a.stress_capacity + node_b.stress_capacity
        
        if combined_capacity > 0:
            return combined_strength / combined_capacity
        return 1.0
    
    def _calculate_printability_score(self, node_a: ConnectionNode, node_b: ConnectionNode) -> float:
        """Berechnet 3D-Druck Freundlichkeits-Score"""
        # Faktoren: Überhang-Winkel, Support-Notwendigkeit, Brücken-Länge
        
        # Überhang-Winkel
        direction = node_b.position - node_a.position
        if direction.length > 0:
            direction.normalize()
            overhang_angle = abs(direction.z)  # Z-Komponente als Überhang-Indikator
        else:
            overhang_angle = 0
        
        # Support-Score (0 = viel Support nötig, 1 = kein Support)
        support_score = 1.0 - min(overhang_angle, 1.0)
        
        # Brücken-Score (kurze Brücken besser)
        bridge_length = direction.length
        max_bridge_length = 20.0
        bridge_score = max(0, 1.0 - bridge_length / max_bridge_length)
        
        return (support_score + bridge_score) / 2.0
    
    def _calculate_surface_deviation(self, pos_a: Vector, pos_b: Vector) -> float:
        """Berechnet Abweichung von Ziel-Oberfläche"""
        if not self.bvh_tree:
            return 0.0
        
        # Sample Punkte entlang der Verbindung
        deviations = []
        for t in np.linspace(0.1, 0.9, 5):
            sample_point = pos_a.lerp(pos_b, t)
            
            # Finde nächsten Surface-Punkt
            location, normal, _, _ = self.bvh_tree.find_nearest(sample_point)
            if location:
                deviation = (sample_point - location).length
                deviations.append(deviation)
        
        return sum(deviations) / len(deviations) if deviations else 0.0
    
    def _calculate_biomech_efficiency(self, node_a: ConnectionNode, node_b: ConnectionNode) -> float:
        """Berechnet biomechanische Effizienz"""
        # Faktoren: Lastpfad-Optimierung, anatomische Ausrichtung
        
        # Importance-Balance
        importance_diff = abs(node_a.biomech_importance - node_b.biomech_importance)
        balance_score = 1.0 - min(importance_diff, 1.0)
        
        # Anchor-Verbindungen bevorzugen
        anchor_bonus = 0.2 if (node_a.is_anchor or node_b.is_anchor) else 0.0
        
        return balance_score + anchor_bonus
    
    @measure_time
    def _minimum_spanning_tree(self) -> List[ConnectionEdge]:
        """Kruskal's MST Algorithmus"""
        debug.info('OPTIMIZER', "Running Minimum Spanning Tree optimization")
        
        # Sortiere Kanten nach Gewicht
        sorted_edges = sorted(self.graph.edges.values(), key=lambda e: e.weight)
        
        # Union-Find für Kreis-Erkennung
        parent = {}
        rank = {}
        
        def find(x):
            if x not in parent:
                parent[x] = x
                rank[x] = 0
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            return True
        
        # MST konstruieren
        mst_edges = []
        for edge in sorted_edges:
            if union(edge.node_a, edge.node_b):
                mst_edges.append(edge)
                
                # Stop wenn genug Kanten (n-1 für n Knoten)
                if len(mst_edges) >= len(self.graph.nodes) - 1:
                    break
        
        debug.info('OPTIMIZER', f"MST completed with {len(mst_edges)} edges")
        return mst_edges
    
    def _dijkstra_optimization(self) -> List[ConnectionEdge]:
        """Dijkstra-basierte Optimierung für kürzeste Pfade"""
        debug.info('OPTIMIZER', "Running Dijkstra optimization")
        
        # Finde wichtigste Knoten (Anker, hohe biomech. Bedeutung)
        important_nodes = []
        for node in self.graph.nodes.values():
            if node.is_anchor or node.biomech_importance > 1.5:
                important_nodes.append(node.id)
        
        if not important_nodes:
            # Fallback: Verwende alle Knoten
            important_nodes = list(self.graph.nodes.keys())
        
        # Für jeden wichtigen Knoten: Finde kürzeste Pfade zu anderen
        selected_edges = set()
        
        for start_node in important_nodes:
            # Dijkstra von start_node
            distances = {node_id: float('inf') for node_id in self.graph.nodes}
            distances[start_node] = 0.0
            previous = {}
            unvisited = set(self.graph.nodes.keys())
            
            while unvisited:
                # Finde Knoten mit minimaler Distanz
                current = min(unvisited, key=lambda x: distances[x])
                if distances[current] == float('inf'):
                    break
                
                unvisited.remove(current)
                
                # Update Nachbarn
                for neighbor in self.graph.get_neighbors(current):
                    if neighbor in unvisited:
                        edge = self.graph.get_edge(current, neighbor)
                        if edge:
                            alt_distance = distances[current] + edge.weight
                            if alt_distance < distances[neighbor]:
                                distances[neighbor] = alt_distance
                                previous[neighbor] = current
            
            # Rekonstruiere Pfade zu anderen wichtigen Knoten
            for target_node in important_nodes:
                if target_node != start_node and target_node in previous:
                    # Baue Pfad zurück
                    path = []
                    current = target_node
                    while current in previous:
                        prev = previous[current]
                        edge = self.graph.get_edge(current, prev)
                        if edge:
                            path.append(edge)
                        current = prev
                    
                    # Füge Pfad-Kanten hinzu
                    for edge in path:
                        selected_edges.add(edge)
        
        debug.info('OPTIMIZER', f"Dijkstra completed with {len(selected_edges)} edges")
        return list(selected_edges)
    
    def _astar_optimization(self) -> List[ConnectionEdge]:
        """A*-basierte Optimierung mit Heuristik"""
        debug.info('OPTIMIZER', "Running A* optimization")
        
        # Vereinfachte A*: Nutze MST als Basis mit heuristischer Verbesserung
        base_edges = self._minimum_spanning_tree()
        
        # Heuristik: Ersetze schlechte Kanten durch bessere Alternativen
        improved_edges = []
        
        for edge in base_edges:
            # Suche nach besseren Alternativen
            node_a = self.graph.nodes[edge.node_a]
            node_b = self.graph.nodes[edge.node_b]
            
            # Heuristik: Biomechanische Bedeutung
            heuristic_value = (node_a.biomech_importance + node_b.biomech_importance) / edge.weight
            
            # Nur gute Kanten behalten
            if heuristic_value > 0.5:  # Threshold konfigurierbar
                improved_edges.append(edge)
        
        debug.info('OPTIMIZER', f"A* completed with {len(improved_edges)} edges")
        return improved_edges
    
    def _force_directed_optimization(self) -> List[ConnectionEdge]:
        """Kraftbasierte Optimierung (vereinfacht)"""
        debug.info('OPTIMIZER', "Running force-directed optimization")
        
        # Starte mit MST
        mst_edges = self._minimum_spanning_tree()
        
        # Simuliere Kräfte für lokale Verbesserungen
        # (Vereinfachte Implementierung - könnte erweitert werden)
        
        optimized_edges = []
        for edge in mst_edges:
            # "Kraft" basierend auf Stress und Biomechanik
            force_factor = edge.stress_factor * edge.biomech_efficiency
            
            # Nur "starke" Verbindungen behalten
            if force_factor > 0.3:  # Threshold konfigurierbar
                optimized_edges.append(edge)
        
        debug.info('OPTIMIZER', f"Force-directed completed with {len(optimized_edges)} edges")
        return optimized_edges
    
    def _apply_connection_limits(self, edges: List[ConnectionEdge], 
                               max_connections: int) -> List[ConnectionEdge]:
        """Begrenzt Verbindungen pro Knoten"""
        # Zähle Verbindungen pro Knoten
        node_connections = {}
        for edge in edges:
            node_connections[edge.node_a] = node_connections.get(edge.node_a, 0) + 1
            node_connections[edge.node_b] = node_connections.get(edge.node_b, 0) + 1
        
        # Entferne schwächste Verbindungen wenn Limit überschritten
        filtered_edges = []
        node_counts = {node_id: 0 for node_id in self.graph.nodes}
        
        # Sortiere Kanten nach Qualität (niedrigeres Gewicht = besser)
        sorted_edges = sorted(edges, key=lambda e: e.weight)
        
        for edge in sorted_edges:
            # Prüfe ob beide Knoten noch Kapazität haben
            if (node_counts[edge.node_a] < max_connections and 
                node_counts[edge.node_b] < max_connections):
                filtered_edges.append(edge)
                node_counts[edge.node_a] += 1
                node_counts[edge.node_b] += 1
        
        debug.info('OPTIMIZER', f"Connection limits applied: {len(edges)} -> {len(filtered_edges)} edges")
        return filtered_edges
    
    def _edges_to_sphere_pairs(self, edges: List[ConnectionEdge]) -> List[Tuple[bpy.types.Object, bpy.types.Object]]:
        """Konvertiert Kanten zurück zu Sphere-Paaren"""
        sphere_pairs = []
        
        for edge in edges:
            node_a = self.graph.nodes[edge.node_a]
            node_b = self.graph.nodes[edge.node_b]
            
            if node_a.sphere_obj and node_b.sphere_obj:
                sphere_pairs.append((node_a.sphere_obj, node_b.sphere_obj))
        
        return sphere_pairs
    
    def _calculate_improvement_factor(self, num_spheres: int, num_connections: int) -> float:
        """Berechnet Verbesserungs-Faktor gegenüber vollständigem Graph"""
        max_possible_connections = (num_spheres * (num_spheres - 1)) // 2
        if max_possible_connections > 0:
            return max_possible_connections / max(num_connections, 1)
        return 1.0
    
    def update_surface_target(self, target_obj: bpy.types.Object):
        """Aktualisiert Ziel-Oberfläche"""
        self.target_surface = target_obj
        
        if target_obj and target_obj.type == 'MESH':
            # Erstelle BVH-Tree
            depsgraph = bpy.context.evaluated_depsgraph_get()
            obj_eval = target_obj.evaluated_get(depsgraph)
            mesh = obj_eval.to_mesh()
            
            self.bvh_tree = BVHTree.FromMesh(mesh)
            debug.info('OPTIMIZER', f"BVH tree updated for surface: {target_obj.name}")
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Erstellt detaillierten Optimierungs-Bericht"""
        return {
            'statistics': self.optimization_stats.copy(),
            'graph_info': {
                'nodes': len(self.graph.nodes),
                'edges': len(self.graph.edges),
                'density': len(self.graph.edges) / max(len(self.graph.nodes), 1)
            },
            'optimization_weights': self.optimization_weights.copy(),
            'surface_integration': {
                'surface_available': self.target_surface is not None,
                'bvh_cached': self.bvh_tree is not None,
                'surface_cache_size': len(self.surface_cache)
            }
        }


# =========================================
# CONVENIENCE FUNCTIONS
# =========================================

@measure_time 
def optimize_sphere_connections(spheres: List[bpy.types.Object],
                              optimization_method: OptimizationCriteria = OptimizationCriteria.HYBRID,
                              surface_obj: Optional[bpy.types.Object] = None,
                              max_connections: int = 6) -> List[Tuple[bpy.types.Object, bpy.types.Object]]:
    """
    Convenience-Funktion für Sphere-Verbindungsoptimierung
    
    Args:
        spheres: Liste von Sphere-Objekten
        optimization_method: Optimierungskriterium
        surface_obj: Optional Ziel-Oberfläche
        max_connections: Maximale Verbindungen pro Sphere
        
    Returns:
        Optimierte Liste von Sphere-Paaren
    """
    optimizer = ConnectionOptimizer(surface_obj)
    return optimizer.optimize_connections(
        spheres, 
        criteria=optimization_method,
        max_connections_per_node=max_connections
    )


def create_stress_optimized_connections(spheres: List[bpy.types.Object],
                                      stress_data: Dict[int, float] = None) -> List[Tuple[bpy.types.Object, bpy.types.Object]]:
    """
    Erstellt stress-optimierte Verbindungen
    
    Args:
        spheres: Liste von Sphere-Objekten  
        stress_data: Optional Stress-Daten pro Sphere-Index
        
    Returns:
        Stress-optimierte Verbindungen
    """
    # Update stress requirements in spheres
    if stress_data:
        for i, sphere in enumerate(spheres):
            if i in stress_data:
                sphere["strength_requirement"] = stress_data[i]
    
    return optimize_sphere_connections(
        spheres,
        optimization_method=OptimizationCriteria.MINIMUM_STRESS
    )


def create_print_friendly_connections(spheres: List[bpy.types.Object]) -> List[Tuple[bpy.types.Object, bpy.types.Object]]:
    """
    Erstellt 3D-Druck freundliche Verbindungen
    
    Args:
        spheres: Liste von Sphere-Objekten
        
    Returns:
        Druckfreundliche Verbindungen
    """
    return optimize_sphere_connections(
        spheres,
        optimization_method=OptimizationCriteria.PRINT_FRIENDLY
    )
