"""
Chain Tool V4 - Stress Analysis Module
======================================

Vereinfachte FEA-Simulation für Orthesen-Verstärkungen
- Biomechanische Lastfall-Definition für Orthesen
- Spannungsvisualisierung und Hotspot-Erkennung  
- Material-Properties Integration (TPU/PETG)
- Surface-aware Stress-Verteilung mit Aura-System

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
from dataclasses import dataclass, field
from collections import defaultdict

# V4 Module imports
from ..core.constants import LoadCase, MaterialType, StressType
from ..core.state_manager import StateManager
from ..utils.performance import measure_time, PerformanceTracker
from ..utils.debug import DebugManager
from ..utils.caching import CacheManager
from ..utils.math_utils import safe_normalize, calculate_distance_3d

# Initialize systems
debug = DebugManager()
cache = CacheManager()
perf = PerformanceTracker()

class LoadCase(Enum):
    """Biomechanische Lastfälle für Orthesen"""
    WEIGHT_BEARING = "weight_bearing"         # Gewichtsbelastung (Stehen/Gehen)
    FLEXION = "flexion"                      # Beugung (z.B. Knie beugen)
    EXTENSION = "extension"                  # Streckung
    ROTATION = "rotation"                    # Rotation/Torsion
    LATERAL_BENDING = "lateral_bending"      # Seitliche Beugung
    IMPACT = "impact"                        # Stoßbelastung
    FATIGUE = "fatigue"                      # Ermüdungsbelastung
    COMBINED = "combined"                    # Kombinierte Belastung

class MaterialType(Enum):
    """Material-Typen für Dual-Material System"""
    TPU_FLEXIBLE = "tpu_flexible"            # TPU Shell - flexibel
    PETG_RIGID = "petg_rigid"               # PETG Pattern - steif
    COMPOSITE = "composite"                  # Verbundmaterial

class StressType(Enum):
    """Spannungs-Typen"""
    VON_MISES = "von_mises"                 # Von-Mises Vergleichsspannung
    PRINCIPAL_MAX = "principal_max"          # Hauptspannung maximal
    PRINCIPAL_MIN = "principal_min"          # Hauptspannung minimal
    SHEAR_MAX = "shear_max"                 # Maximale Schubspannung
    HYDROSTATIC = "hydrostatic"             # Hydrostatische Spannung

@dataclass
class MaterialProperties:
    """Material-Eigenschaften für FEA"""
    name: str
    youngs_modulus: float      # E-Modul in MPa
    poisson_ratio: float       # Poisson-Zahl
    yield_strength: float      # Streckgrenze in MPa
    ultimate_strength: float   # Bruchfestigkeit in MPa
    density: float            # Dichte in g/cm³
    fatigue_limit: float      # Dauerfestigkeit in MPa
    
    # 3D-Druck spezifische Eigenschaften
    layer_adhesion: float = 0.8    # Schichthaftung (0-1)
    anisotropy_factor: float = 0.9  # Anisotropie-Faktor
    surface_quality: float = 1.0    # Oberflächenqualität

@dataclass
class LoadCondition:
    """Belastungsbedingung"""
    load_case: LoadCase
    magnitude: float           # Belastungsgröße in N
    direction: Vector         # Belastungsrichtung
    application_point: Vector # Angriffspunkt
    duration: float = 1.0     # Belastungsdauer in s
    frequency: float = 0.0    # Frequenz in Hz (für Ermüdung)

@dataclass
class StressResult:
    """Spannungs-Ergebnis an einem Punkt"""
    position: Vector
    stress_magnitude: float
    stress_type: StressType
    principal_stresses: Tuple[float, float, float]
    stress_tensor: np.ndarray
    safety_factor: float
    material_utilization: float

class MaterialDatabase:
    """Datenbank für Material-Eigenschaften"""
    
    def __init__(self):
        self.materials = self._initialize_materials()
    
    def _initialize_materials(self) -> Dict[MaterialType, MaterialProperties]:
        """Initialisiert Material-Datenbank"""
        return {
            MaterialType.TPU_FLEXIBLE: MaterialProperties(
                name="TPU Shore 95A",
                youngs_modulus=12.0,      # MPa - sehr flexibel
                poisson_ratio=0.45,       # Gummi-ähnlich
                yield_strength=8.0,       # MPa
                ultimate_strength=25.0,   # MPa
                density=1.2,              # g/cm³
                fatigue_limit=3.0,        # MPa
                layer_adhesion=0.85,      # Gute Schichthaftung
                anisotropy_factor=0.8     # Mäßige Anisotropie
            ),
            
            MaterialType.PETG_RIGID: MaterialProperties(
                name="PETG",
                youngs_modulus=2100.0,    # MPa - steif
                poisson_ratio=0.38,       # Polymer-typisch
                yield_strength=50.0,      # MPa
                ultimate_strength=65.0,   # MPa
                density=1.27,             # g/cm³
                fatigue_limit=20.0,       # MPa
                layer_adhesion=0.95,      # Sehr gute Schichthaftung
                anisotropy_factor=0.85    # Geringe Anisotropie
            ),
            
            MaterialType.COMPOSITE: MaterialProperties(
                name="TPU-PETG Composite",
                youngs_modulus=800.0,     # MPa - Mittelwert
                poisson_ratio=0.40,       # Mischung
                yield_strength=30.0,      # MPa
                ultimate_strength=45.0,   # MPa
                density=1.24,             # g/cm³
                fatigue_limit=15.0,       # MPa
                layer_adhesion=0.90,      # Gute Verbindung
                anisotropy_factor=0.82    # Gemittelt
            )
        }
    
    def get_material(self, material_type: MaterialType) -> MaterialProperties:
        """Holt Material-Eigenschaften"""
        return self.materials.get(material_type, self.materials[MaterialType.COMPOSITE])

class BiomechanicalLoadCases:
    """Biomechanische Lastfälle für Orthesen"""
    
    @staticmethod
    def get_weight_bearing_load(body_weight: float = 700.0) -> LoadCondition:
        """Gewichtsbelastung (70kg Person = 700N)"""
        return LoadCondition(
            load_case=LoadCase.WEIGHT_BEARING,
            magnitude=body_weight,
            direction=Vector((0, 0, -1)),  # Nach unten
            application_point=Vector((0, 0, 0)),
            duration=1.0
        )
    
    @staticmethod
    def get_flexion_load(flexion_moment: float = 100.0) -> LoadCondition:
        """Beugungsmoment (z.B. Knie beugen)"""
        return LoadCondition(
            load_case=LoadCase.FLEXION,
            magnitude=flexion_moment,
            direction=Vector((1, 0, 0)),  # Um X-Achse
            application_point=Vector((0, 0, 0)),
            duration=2.0
        )
    
    @staticmethod
    def get_impact_load(impact_force: float = 1500.0) -> LoadCondition:
        """Stoßbelastung (1.5x Körpergewicht)"""
        return LoadCondition(
            load_case=LoadCase.IMPACT,
            magnitude=impact_force,
            direction=Vector((0, 0, -1)),
            application_point=Vector((0, 0, 0)),
            duration=0.1  # Kurzer Stoß
        )
    
    @staticmethod
    def get_fatigue_load(cyclic_load: float = 200.0, frequency: float = 1.0) -> LoadCondition:
        """Ermüdungsbelastung (zyklische Belastung)"""
        return LoadCondition(
            load_case=LoadCase.FATIGUE,
            magnitude=cyclic_load,
            direction=Vector((0, 0, -1)),
            application_point=Vector((0, 0, 0)),
            duration=3600.0,  # 1 Stunde
            frequency=frequency  # 1 Hz = Gehgeschwindigkeit
        )

class SimplifiedFEA:
    """
    Vereinfachte Finite-Element-Analyse
    
    Features:
    - Truss-Element basierte Analyse
    - Biomechanische Lastfälle
    - Material-abhängige Berechnung
    - Stress-Visualisierung
    """
    
    def __init__(self, target_surface: Optional[bpy.types.Object] = None):
        """
        Initialisiert FEA System
        
        Args:
            target_surface: Ziel-Oberfläche für Surface-aware Analysis
        """
        self.target_surface = target_surface
        self.surface_bvh: Optional[BVHTree] = None
        self.material_db = MaterialDatabase()
        
        # FEA Mesh
        self.nodes: Dict[int, Vector] = {}
        self.elements: Dict[int, Tuple[int, int]] = {}  # Truss elements
        self.node_materials: Dict[int, MaterialType] = {}
        self.element_properties: Dict[int, Dict[str, float]] = {}
        
        # Boundary conditions
        self.fixed_nodes: Set[int] = set()
        self.loaded_nodes: Dict[int, LoadCondition] = {}
        
        # Results
        self.displacements: Dict[int, Vector] = {}
        self.stress_results: Dict[int, StressResult] = {}
        self.reaction_forces: Dict[int, Vector] = {}
        
        # Analysis parameters
        self.safety_factor_target = 2.0
        self.max_iterations = 100
        self.convergence_tolerance = 1e-6
        
        if target_surface:
            self.update_surface_target(target_surface)
    
    @measure_time
    def build_fea_model(self, spheres: List[bpy.types.Object], 
                       connections: List[Tuple[bpy.types.Object, bpy.types.Object]]) -> bool:
        """
        Baut FEA-Modell aus Spheres und Verbindungen
        
        Args:
            spheres: Liste von Sphere-Objekten (Knoten)
            connections: Liste von Verbindungen (Elemente)
            
        Returns:
            True wenn erfolgreich
        """
        debug.info('STRESS', f"Building FEA model: {len(spheres)} nodes, {len(connections)} elements")
        
        # Clear existing model
        self.nodes.clear()
        self.elements.clear()
        self.node_materials.clear()
        self.element_properties.clear()
        
        # Create nodes from spheres
        for i, sphere in enumerate(spheres):
            self.nodes[i] = Vector(sphere.location)
            
            # Bestimme Material basierend auf Sphere-Eigenschaften
            if sphere.get("is_pattern_element", False):
                self.node_materials[i] = MaterialType.PETG_RIGID
            else:
                self.node_materials[i] = MaterialType.TPU_FLEXIBLE
        
        # Create elements from connections
        sphere_to_node = {sphere: i for i, sphere in enumerate(spheres)}
        
        for elem_id, (sphere_a, sphere_b) in enumerate(connections):
            if sphere_a in sphere_to_node and sphere_b in sphere_to_node:
                node_a = sphere_to_node[sphere_a]
                node_b = sphere_to_node[sphere_b]
                
                self.elements[elem_id] = (node_a, node_b)
                
                # Element-Eigenschaften
                radius_a = sphere_a.get("kette_radius", 1.0)
                radius_b = sphere_b.get("kette_radius", 1.0)
                avg_radius = (radius_a + radius_b) * 0.5
                
                # Cross-sectional area (circular)
                area = np.pi * (avg_radius ** 2)
                length = (self.nodes[node_a] - self.nodes[node_b]).length
                
                self.element_properties[elem_id] = {
                    'area': area,
                    'length': length,
                    'radius': avg_radius
                }
        
        debug.info('STRESS', f"FEA model built successfully")
        return True
    
    @measure_time
    def apply_boundary_conditions(self, 
                                load_conditions: List[LoadCondition],
                                fixed_points: List[Vector] = None) -> bool:
        """
        Wendet Randbedingungen an
        
        Args:
            load_conditions: Liste von Belastungsbedingungen
            fixed_points: Liste von festen Punkten (Einspannungen)
            
        Returns:
            True wenn erfolgreich
        """
        debug.info('STRESS', f"Applying boundary conditions: {len(load_conditions)} loads")
        
        self.fixed_nodes.clear()
        self.loaded_nodes.clear()
        
        # Apply loads
        for load in load_conditions:
            # Finde nächsten Knoten zum Angriffspunkt
            closest_node = self._find_closest_node(load.application_point)
            if closest_node is not None:
                self.loaded_nodes[closest_node] = load
        
        # Apply fixations
        if fixed_points:
            for fix_point in fixed_points:
                closest_node = self._find_closest_node(fix_point)
                if closest_node is not None:
                    self.fixed_nodes.add(closest_node)
        else:
            # Auto-detect fixed points (nodes with high support)
            self._auto_detect_fixed_points()
        
        debug.info('STRESS', f"Applied {len(self.loaded_nodes)} loads, {len(self.fixed_nodes)} fixations")
        return True
    
    def _find_closest_node(self, point: Vector) -> Optional[int]:
        """Findet nächsten Knoten zu gegebenem Punkt"""
        if not self.nodes:
            return None
        
        min_distance = float('inf')
        closest_node = None
        
        for node_id, node_pos in self.nodes.items():
            distance = (point - node_pos).length
            if distance < min_distance:
                min_distance = distance
                closest_node = node_id
        
        return closest_node
    
    def _auto_detect_fixed_points(self):
        """Automatische Erkennung von Einspannpunkten"""
        if self.surface_bvh:
            # Knoten nahe der Oberfläche als eingespannt betrachten
            for node_id, position in self.nodes.items():
                surface_point, _, _, _ = self.surface_bvh.find_nearest(position)
                if surface_point:
                    distance = (position - surface_point).length
                    if distance < 2.0:  # Nah an Oberfläche
                        self.fixed_nodes.add(node_id)
        else:
            # Fallback: unterste Knoten als eingespannt
            min_z = min(pos.z for pos in self.nodes.values())
            threshold = min_z + 2.0
            
            for node_id, position in self.nodes.items():
                if position.z <= threshold:
                    self.fixed_nodes.add(node_id)
    
    @measure_time
    def solve_linear_system(self) -> bool:
        """
        Löst lineares Gleichungssystem K*u = F
        
        Returns:
            True wenn konvergiert
        """
        debug.info('STRESS', "Solving linear FEA system")
        
        if not self.nodes or not self.elements:
            debug.error('STRESS', "No FEA model available")
            return False
        
        num_nodes = len(self.nodes)
        num_dof = num_nodes * 3  # 3 DOF per node (x, y, z)
        
        # Assembliere globale Steifigkeitsmatrix
        K_global = np.zeros((num_dof, num_dof))
        F_global = np.zeros(num_dof)
        
        # Assembliere Element-Steifigkeitsmatrizen
        for elem_id, (node_a, node_b) in self.elements.items():
            K_elem = self._compute_element_stiffness(elem_id, node_a, node_b)
            
            # Assembly: füge Element-Matrix in globale Matrix ein
            dof_indices = [
                node_a * 3, node_a * 3 + 1, node_a * 3 + 2,
                node_b * 3, node_b * 3 + 1, node_b * 3 + 2
            ]
            
            for i, global_i in enumerate(dof_indices):
                for j, global_j in enumerate(dof_indices):
                    K_global[global_i, global_j] += K_elem[i, j]
        
        # Assembliere Lastvektor
        for node_id, load in self.loaded_nodes.items():
            force_vector = load.direction * load.magnitude
            F_global[node_id * 3:node_id * 3 + 3] = [force_vector.x, force_vector.y, force_vector.z]
        
        # Wende Randbedingungen an (Einspannungen)
        for node_id in self.fixed_nodes:
            for dof in range(3):
                global_dof = node_id * 3 + dof
                # Setze Zeile und Spalte auf Null, Diagonalelement auf 1
                K_global[global_dof, :] = 0
                K_global[:, global_dof] = 0
                K_global[global_dof, global_dof] = 1
                F_global[global_dof] = 0
        
        # Löse Gleichungssystem
        try:
            u_global = np.linalg.solve(K_global, F_global)
            
            # Extrahiere Knotenverschiebungen
            self.displacements.clear()
            for node_id in self.nodes.keys():
                u_x = u_global[node_id * 3]
                u_y = u_global[node_id * 3 + 1] 
                u_z = u_global[node_id * 3 + 2]
                self.displacements[node_id] = Vector((u_x, u_y, u_z))
            
            debug.info('STRESS', "Linear system solved successfully")
            return True
            
        except np.linalg.LinAlgError as e:
            debug.error('STRESS', f"Failed to solve linear system: {e}")
            return False
    
    def _compute_element_stiffness(self, elem_id: int, node_a: int, node_b: int) -> np.ndarray:
        """Berechnet Element-Steifigkeitsmatrix für Truss-Element"""
        pos_a = self.nodes[node_a]
        pos_b = self.nodes[node_b]
        
        # Element-Eigenschaften
        props = self.element_properties[elem_id]
        area = props['area']
        length = props['length']
        
        # Material-Eigenschaften (nimm Material von node_a)
        material_type = self.node_materials.get(node_a, MaterialType.COMPOSITE)
        material = self.material_db.get_material(material_type)
        E = material.youngs_modulus
        
        # Richtungsvektor
        direction = (pos_b - pos_a) / length
        l, m, n = direction.x, direction.y, direction.z
        
        # Element-Steifigkeitsmatrix (3D Truss)
        factor = (E * area) / length
        
        K_elem = factor * np.array([
            [ l*l,  l*m,  l*n, -l*l, -l*m, -l*n],
            [ l*m,  m*m,  m*n, -l*m, -m*m, -m*n],
            [ l*n,  m*n,  n*n, -l*n, -m*n, -n*n],
            [-l*l, -l*m, -l*n,  l*l,  l*m,  l*n],
            [-l*m, -m*m, -m*n,  l*m,  m*m,  m*n],
            [-l*n, -m*n, -n*n,  l*n,  m*n,  n*n]
        ])
        
        return K_elem
    
    @measure_time
    def compute_stresses(self) -> Dict[int, StressResult]:
        """
        Berechnet Spannungen in allen Elementen
        
        Returns:
            Dictionary mit Stress-Ergebnissen pro Knoten
        """
        debug.info('STRESS', "Computing element stresses")
        
        self.stress_results.clear()
        
        if not self.displacements:
            debug.warning('STRESS', "No displacement results available")
            return {}
        
        # Berechne Spannungen für jedes Element
        element_stresses = {}
        
        for elem_id, (node_a, node_b) in self.elements.items():
            # Verschiebungen der Endknoten
            u_a = self.displacements.get(node_a, Vector((0, 0, 0)))
            u_b = self.displacements.get(node_b, Vector((0, 0, 0)))
            
            # Element-Eigenschaften
            props = self.element_properties[elem_id]
            length = props['length']
            
            # Material-Eigenschaften
            material_type = self.node_materials.get(node_a, MaterialType.COMPOSITE)
            material = self.material_db.get_material(material_type)
            E = material.youngs_modulus
            
            # Richtungsvektor
            pos_a = self.nodes[node_a]
            pos_b = self.nodes[node_b]
            direction = (pos_b - pos_a).normalized()
            
            # Axiale Dehnung
            delta_u = u_b - u_a
            axial_strain = delta_u.dot(direction) / length
            
            # Axiale Spannung (Truss-Element)
            axial_stress = E * axial_strain
            
            element_stresses[elem_id] = {
                'axial_stress': axial_stress,
                'strain': axial_strain,
                'safety_factor': material.yield_strength / max(abs(axial_stress), 0.001),
                'material': material
            }
        
        # Übertrage Element-Spannungen auf Knoten (Mittelwertbildung)
        node_stress_contributions = defaultdict(list)
        
        for elem_id, stress_data in element_stresses.items():
            node_a, node_b = self.elements[elem_id]
            
            # Beide Endknoten bekommen die Element-Spannung
            for node_id in [node_a, node_b]:
                node_stress_contributions[node_id].append(stress_data)
        
        # Berechne Knoten-Spannungen
        for node_id, stress_list in node_stress_contributions.items():
            if not stress_list:
                continue
            
            # Durchschnittliche Spannung
            avg_stress = sum(s['axial_stress'] for s in stress_list) / len(stress_list)
            avg_strain = sum(s['strain'] for s in stress_list) / len(stress_list)
            min_safety_factor = min(s['safety_factor'] for s in stress_list)
            
            # Material-Utilization
            material = stress_list[0]['material']  # Nimm erstes Material
            utilization = abs(avg_stress) / material.yield_strength
            
            # Von-Mises Spannung (vereinfacht für Truss: = axiale Spannung)
            von_mises_stress = abs(avg_stress)
            
            # Principal stresses (für Truss: axial, 0, 0)
            principal_stresses = (avg_stress, 0.0, 0.0)
            
            # Stress tensor (vereinfacht)
            stress_tensor = np.array([
                [avg_stress, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ])
            
            result = StressResult(
                position=self.nodes[node_id],
                stress_magnitude=von_mises_stress,
                stress_type=StressType.VON_MISES,
                principal_stresses=principal_stresses,
                stress_tensor=stress_tensor,
                safety_factor=min_safety_factor,
                material_utilization=utilization
            )
            
            self.stress_results[node_id] = result
        
        debug.info('STRESS', f"Computed stresses for {len(self.stress_results)} nodes")
        return self.stress_results
    
    def find_stress_hotspots(self, threshold_factor: float = 0.8) -> List[Tuple[int, StressResult]]:
        """
        Findet Stress-Hotspots (hochbelastete Bereiche)
        
        Args:
            threshold_factor: Schwellwert als Faktor der Streckgrenze
            
        Returns:
            Liste von (node_id, stress_result) für Hotspots
        """
        hotspots = []
        
        for node_id, stress_result in self.stress_results.items():
            if stress_result.material_utilization > threshold_factor:
                hotspots.append((node_id, stress_result))
        
        # Sortiere nach Material-Utilization (höchste zuerst)
        hotspots.sort(key=lambda x: x[1].material_utilization, reverse=True)
        
        debug.info('STRESS', f"Found {len(hotspots)} stress hotspots")
        return hotspots
    
    def get_analysis_report(self) -> Dict[str, Any]:
        """Erstellt detaillierten Analyse-Bericht"""
        if not self.stress_results:
            return {'error': 'No stress analysis results available'}
        
        # Statistiken
        stress_values = [result.stress_magnitude for result in self.stress_results.values()]
        safety_factors = [result.safety_factor for result in self.stress_results.values()]
        utilizations = [result.material_utilization for result in self.stress_results.values()]
        
        max_stress = max(stress_values) if stress_values else 0
        min_safety = min(safety_factors) if safety_factors else float('inf')
        max_utilization = max(utilizations) if utilizations else 0
        
        # Hotspots
        hotspots = self.find_stress_hotspots(threshold_factor=0.7)
        
        # Critical elements (safety factor < 2.0)
        critical_nodes = [
            (node_id, result) for node_id, result in self.stress_results.items()
            if result.safety_factor < self.safety_factor_target
        ]
        
        report = {
            'model_info': {
                'nodes': len(self.nodes),
                'elements': len(self.elements),
                'loaded_nodes': len(self.loaded_nodes),
                'fixed_nodes': len(self.fixed_nodes)
            },
            'stress_statistics': {
                'max_stress_mpa': max_stress,
                'min_safety_factor': min_safety,
                'max_utilization': max_utilization,
                'avg_stress_mpa': sum(stress_values) / len(stress_values) if stress_values else 0,
                'avg_safety_factor': sum(safety_factors) / len(safety_factors) if safety_factors else 0
            },
            'critical_assessment': {
                'hotspot_count': len(hotspots),
                'critical_node_count': len(critical_nodes),
                'design_adequate': min_safety >= self.safety_factor_target and max_utilization < 0.8,
                'recommendations': self._generate_recommendations(hotspots, critical_nodes)
            },
            'material_usage': {
                material_type.value: len([
                    node_id for node_id, mat_type in self.node_materials.items() 
                    if mat_type == material_type
                ]) for material_type in MaterialType
            }
        }
        
        return report
    
    def _generate_recommendations(self, hotspots: List, critical_nodes: List) -> List[str]:
        """Generiert Design-Empfehlungen basierend auf Analyse"""
        recommendations = []
        
        if hotspots:
            recommendations.append(f"Consider reinforcing {len(hotspots)} stress hotspots with additional pattern elements")
        
        if critical_nodes:
            recommendations.append(f"Increase material thickness or add connections near {len(critical_nodes)} critical nodes")
        
        # Material-spezifische Empfehlungen
        tpu_nodes = len([n for n, m in self.node_materials.items() if m == MaterialType.TPU_FLEXIBLE])
        petg_nodes = len([n for n, m in self.node_materials.items() if m == MaterialType.PETG_RIGID])
        
        if tpu_nodes > petg_nodes * 3:
            recommendations.append("Consider increasing PETG pattern density for better structural support")
        
        max_utilization = max([r.material_utilization for r in self.stress_results.values()], default=0)
        if max_utilization > 0.9:
            recommendations.append("Material utilization very high - consider stronger materials or larger cross-sections")
        
        return recommendations
    
    def update_surface_target(self, target_obj: bpy.types.Object):
        """Aktualisiert Ziel-Oberfläche"""
        self.target_surface = target_obj
        
        if target_obj and target_obj.type == 'MESH':
            depsgraph = bpy.context.evaluated_depsgraph_get()
            obj_eval = target_obj.evaluated_get(depsgraph)
            mesh = obj_eval.to_mesh()
            
            self.surface_bvh = BVHTree.FromMesh(mesh)
            debug.info('STRESS', f"Surface BVH updated: {target_obj.name}")


# =========================================
# CONVENIENCE FUNCTIONS
# =========================================

@measure_time
def perform_stress_analysis(spheres: List[bpy.types.Object],
                          connections: List[Tuple[bpy.types.Object, bpy.types.Object]],
                          load_conditions: List[LoadCondition],
                          surface_obj: Optional[bpy.types.Object] = None) -> Dict[str, Any]:
    """
    Convenience-Funktion für vollständige Stress-Analyse
    
    Args:
        spheres: Liste von Sphere-Objekten
        connections: Liste von Verbindungen
        load_conditions: Belastungsbedingungen
        surface_obj: Optional Ziel-Oberfläche
        
    Returns:
        Vollständiger Analyse-Bericht
    """
    
    # Initialisiere FEA
    fea = SimplifiedFEA(surface_obj)
    
    # Baue Modell
    if not fea.build_fea_model(spheres, connections):
        return {'error': 'Failed to build FEA model'}
    
    # Wende Randbedingungen an
    if not fea.apply_boundary_conditions(load_conditions):
        return {'error': 'Failed to apply boundary conditions'}
    
    # Löse System
    if not fea.solve_linear_system():
        return {'error': 'Failed to solve linear system'}
    
    # Berechne Spannungen
    stress_results = fea.compute_stresses()
    
    if not stress_results:
        return {'error': 'Failed to compute stresses'}
    
    # Erstelle Bericht
    report = fea.get_analysis_report()
    report['stress_results'] = stress_results
    
    return report


def create_orthotic_load_case(body_weight: float = 70.0, 
                            activity_factor: float = 1.0) -> List[LoadCondition]:
    """
    Erstellt typische Orthesen-Belastungsfälle
    
    Args:
        body_weight: Körpergewicht in kg
        activity_factor: Aktivitätsfaktor (1.0 = normal, 2.0 = Sport)
        
    Returns:
        Liste von Belastungsbedingungen
    """
    weight_force = body_weight * 9.81 * activity_factor  # N
    
    load_cases = BiomechanicalLoadCases()
    
    return [
        load_cases.get_weight_bearing_load(weight_force),
        load_cases.get_flexion_load(weight_force * 0.3),  # 30% des Gewichts als Moment
        load_cases.get_impact_load(weight_force * 2.0)    # 2x Gewicht bei Impact
    ]


def analyze_material_efficiency(stress_results: Dict[int, StressResult]) -> Dict[str, float]:
    """
    Analysiert Material-Effizienz basierend auf Stress-Ergebnissen
    
    Args:
        stress_results: Stress-Ergebnisse pro Knoten
        
    Returns:
        Material-Effizienz Metriken
    """
    if not stress_results:
        return {}
    
    utilizations = [result.material_utilization for result in stress_results.values()]
    safety_factors = [result.safety_factor for result in stress_results.values()]
    
    return {
        'average_utilization': sum(utilizations) / len(utilizations),
        'max_utilization': max(utilizations),
        'underutilized_ratio': len([u for u in utilizations if u < 0.3]) / len(utilizations),
        'overutilized_ratio': len([u for u in utilizations if u > 0.8]) / len(utilizations),
        'average_safety_factor': sum(safety_factors) / len(safety_factors),
        'efficiency_score': sum(utilizations) / len(utilizations) * (1.0 - len([u for u in utilizations if u > 0.9]) / len(utilizations))
    }


def visualize_stress_results(stress_results: Dict[int, StressResult], 
                           spheres: List[bpy.types.Object],
                           collection_name: str = "StressVisualization") -> List[bpy.types.Object]:
    """
    Visualisiert Stress-Ergebnisse in Blender
    
    Args:
        stress_results: Stress-Ergebnisse
        spheres: Original Sphere-Objekte  
        collection_name: Name der Visualization-Collection
        
    Returns:
        Liste der erstellten Visualisierungs-Objekte
    """
    
    if not stress_results:
        return []
    
    # Erstelle oder hole Collection
    if collection_name in bpy.data.collections:
        collection = bpy.data.collections[collection_name]
    else:
        collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(collection)
    
    visualization_objects = []
    
    # Normalisiere Stress-Werte für Farbkodierung
    stress_values = [result.stress_magnitude for result in stress_results.values()]
    if stress_values:
        max_stress = max(stress_values)
        min_stress = min(stress_values)
        stress_range = max_stress - min_stress if max_stress > min_stress else 1.0
    else:
        return []
    
    # Erstelle farbkodierte Visualisierung
    for node_id, stress_result in stress_results.items():
        if node_id < len(spheres):
            original_sphere = spheres[node_id]
            
            # Erstelle Visualisierungs-Objekt
            bpy.ops.mesh.primitive_uv_sphere_add(
                location=stress_result.position,
                radius=max(original_sphere.dimensions) * 0.6
            )
            vis_obj = bpy.context.active_object
            vis_obj.name = f"Stress_Node_{node_id:03d}"
            
            # Farbkodierung basierend auf Spannung
            normalized_stress = (stress_result.stress_magnitude - min_stress) / stress_range
            
            # Erstelle Material mit Stress-Farbe
            material = bpy.data.materials.new(name=f"StressMat_{node_id}")
            material.use_nodes = True
            bsdf = material.node_tree.nodes["Principled BSDF"]
            
            # Rot für hohe Spannung, Grün für niedrige
            if normalized_stress > 0.8:
                color = (1.0, 0.0, 0.0, 1.0)  # Rot
            elif normalized_stress > 0.5:
                color = (1.0, 0.5, 0.0, 1.0)  # Orange
            elif normalized_stress > 0.3:
                color = (1.0, 1.0, 0.0, 1.0)  # Gelb
            else:
                color = (0.0, 1.0, 0.0, 1.0)  # Grün
            
            bsdf.inputs[0].default_value = color
            vis_obj.data.materials.append(material)
            
            # Custom Properties für Daten
            vis_obj["stress_magnitude"] = stress_result.stress_magnitude
            vis_obj["safety_factor"] = stress_result.safety_factor
            vis_obj["material_utilization"] = stress_result.material_utilization
            
            # Zu Collection hinzufügen
            for coll in vis_obj.users_collection:
                coll.objects.unlink(vis_obj)
            collection.objects.link(vis_obj)
            
            visualization_objects.append(vis_obj)
    
    debug.info('STRESS', f"Created {len(visualization_objects)} stress visualization objects")
    return visualization_objects
