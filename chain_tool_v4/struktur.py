chain_orthotic_tool/
├── __init__.py
├── bl_info.py                          # Version 4.0.0
├── config.py                           # Zentrale Konfiguration
│
├── core/
│   ├── __init__.py
│   ├── properties.py                   # Alle Properties zentral
│   ├── preferences.py                  # Addon Preferences
│   ├── constants.py                    # Enums, Defaults
│   └── state_manager.py                # Global State Management
│
├── geometry/
│   ├── __init__.py
│   ├── mesh_analyzer.py                # Mesh-Analyse & BVH
│   ├── triangulation.py                # Delaunay Triangulation Engine
│   ├── edge_detector.py                # Kanten-Erkennung
│   ├── surface_sampler.py              # Oberflächen-Sampling
│   └── dual_layer_optimizer.py         # 2-Schicht Optimierung
│
├── patterns/
│   ├── __init__.py
│   ├── base_pattern.py                 # Abstract Pattern Class
│   ├── surface_patterns.py             # Großflächige Patterns
│   │   ├── VoronoiPattern             # Voronoi-Zellen
│   │   ├── HexagonalGrid              # Hexagon-Gitter
│   │   ├── TriangularMesh             # Dreiecks-Netz
│   │   └── AdaptiveIslands            # Adaptive Inseln
│   ├── edge_patterns.py                # Rand-Verstärkungen
│   │   ├── RimReinforcement           # Randverstärkung
│   │   ├── ContourBanding             # Kontur-Bänder
│   │   └── StressEdgeLoop             # Belastungs-Ränder
│   ├── hybrid_patterns.py              # Kombinierte Patterns
│   │   ├── CoreShellPattern           # Kern-Schale
│   │   ├── GradientDensity            # Dichte-Gradient
│   │   └── ZonalReinforcement         # Zonen-Verstärkung
│   └── paint_patterns.py               # Paint-basierte Patterns
│
├── operators/
│   ├── __init__.py
│   ├── pattern_generation.py           # Pattern-Generierung
│   ├── paint_mode.py                   # Paint Tools
│   ├── edge_tools.py                   # Kanten-Werkzeuge
│   ├── connection_tools.py             # Verbindungs-Tools
│   ├── triangulation_ops.py            # Triangulation Operators
│   └── export_tools.py                 # Dual-Material Export
│
├── algorithms/
│   ├── __init__.py
│   ├── delaunay_3d.py                  # 3D Delaunay Triangulation
│   ├── connection_optimizer.py         # Verbindungs-Optimierung
│   ├── overlap_prevention.py           # Überlappungs-Vermeidung
│   ├── gap_filling.py                  # Lücken-Füllung
│   └── stress_analysis.py              # Spannungs-Analyse
│
├── ui/
│   ├── __init__.py
│   ├── panels/
│   │   ├── __init__.py
│   │   ├── main_panel.py               # Haupt-Panel
│   │   ├── pattern_panel.py            # Pattern-Auswahl
│   │   ├── edge_panel.py               # Kanten-Tools
│   │   ├── paint_panel.py              # Paint-Mode
│   │   └── export_panel.py             # Export-Optionen
│   ├── overlays/
│   │   ├── __init__.py
│   │   ├── pattern_preview.py          # Pattern-Vorschau
│   │   ├── connection_preview.py       # Verbindungs-Vorschau
│   │   └── paint_overlay.py            # Paint-Overlay
│   └── operators_menu.py               # Operator-Menüs
│
├── utils/
│   ├── __init__.py
│   ├── caching.py                      # Smart Caching System
│   ├── performance.py                  # Performance Tools
│   ├── debug.py                        # Debug System
│   └── math_utils.py                   # Mathematische Hilfsfunktionen
│
└── presets/
    ├── __init__.py
    ├── orthotic_presets.py             # Orthesen-Vorlagen
    ├── material_presets.py             # Material-Einstellungen
    └── pattern_library/                # Pattern-Bibliothek
        ├── medical/                    # Medizinische Patterns
        ├── structural/                 # Strukturelle Patterns
        └── aesthetic/                  # Ästhetische Patterns
