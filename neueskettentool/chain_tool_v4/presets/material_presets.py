"""
Chain Tool V4 - Material Presets
===============================

Comprehensive material preset system for dual-material 3D printing.
Optimized for orthotic applications with TPU (flexible) and PETG (rigid) materials.

Includes printer-specific settings and quality profiles.
"""

from typing import Dict, List, Any, Optional
import bpy

# TPU (Thermoplastic Polyurethane) Presets - Flexible Material
TPU_PRESETS = {
    "flexible_standard": {
        "name": "TPU Standard Flexible",
        "description": "Standard flexible TPU for general orthotic applications",
        "material_properties": {
            "shore_hardness": 85,          # Shore A hardness
            "tensile_strength": 25,        # MPa
            "elongation_at_break": 400,    # %
            "density": 1.2,                # g/cm³
            "glass_transition_temp": -30   # °C
        },
        "printing_parameters": {
            "nozzle_temperature": 230,     # °C
            "bed_temperature": 50,         # °C
            "print_speed": 25,             # mm/s
            "layer_height": 0.2,           # mm
            "infill_density": 20,          # %
            "infill_pattern": "gyroid",
            "retraction_distance": 2.0,    # mm
            "retraction_speed": 25         # mm/s
        },
        "advanced_settings": {
            "linear_advance": 0.3,
            "coasting_enable": True,
            "z_hop": 0.2,
            "supports_enable": False,
            "brim_width": 3.0
        }
    },
    
    "flexible_soft": {
        "name": "TPU Soft Flexible", 
        "description": "Very soft TPU for comfort-critical areas",
        "material_properties": {
            "shore_hardness": 75,
            "tensile_strength": 20,
            "elongation_at_break": 500,
            "density": 1.15,
            "glass_transition_temp": -35
        },
        "printing_parameters": {
            "nozzle_temperature": 225,
            "bed_temperature": 45,
            "print_speed": 20,
            "layer_height": 0.25,
            "infill_density": 15,
            "infill_pattern": "gyroid",
            "retraction_distance": 1.5,
            "retraction_speed": 20
        },
        "advanced_settings": {
            "linear_advance": 0.4,
            "coasting_enable": True,
            "z_hop": 0.3,
            "supports_enable": False,
            "brim_width": 4.0
        }
    },
    
    "flexible_durable": {
        "name": "TPU Durable Flexible",
        "description": "Higher durability TPU for high-stress applications", 
        "material_properties": {
            "shore_hardness": 95,
            "tensile_strength": 35,
            "elongation_at_break": 300,
            "density": 1.25,
            "glass_transition_temp": -25
        },
        "printing_parameters": {
            "nozzle_temperature": 235,
            "bed_temperature": 60,
            "print_speed": 30,
            "layer_height": 0.15,
            "infill_density": 30,
            "infill_pattern": "triangular",
            "retraction_distance": 2.5,
            "retraction_speed": 30
        },
        "advanced_settings": {
            "linear_advance": 0.25,
            "coasting_enable": False,
            "z_hop": 0.1,
            "supports_enable": True,
            "brim_width": 2.0
        }
    },
    
    "medical_grade_tpu": {
        "name": "Medical Grade TPU",
        "description": "Biocompatible TPU for medical/skin contact applications",
        "material_properties": {
            "shore_hardness": 80,
            "tensile_strength": 28,
            "elongation_at_break": 450,
            "density": 1.18,
            "glass_transition_temp": -32,
            "biocompatible": True,
            "sterilizable": True
        },
        "printing_parameters": {
            "nozzle_temperature": 228,
            "bed_temperature": 48,
            "print_speed": 22,
            "layer_height": 0.2,
            "infill_density": 25,
            "infill_pattern": "honeycomb",
            "retraction_distance": 1.8,
            "retraction_speed": 22
        },
        "advanced_settings": {
            "linear_advance": 0.35,
            "coasting_enable": True,
            "z_hop": 0.25,
            "supports_enable": False,
            "brim_width": 3.5,
            "enclosure_required": True
        }
    }
}

# PETG (Polyethylene Terephthalate Glycol) Presets - Rigid Material  
PETG_PRESETS = {
    "rigid_standard": {
        "name": "PETG Standard Rigid",
        "description": "Standard PETG for structural orthotic components",
        "material_properties": {
            "tensile_strength": 50,        # MPa
            "flexural_strength": 69,       # MPa
            "impact_resistance": "High",
            "density": 1.27,               # g/cm³
            "glass_transition_temp": 80,   # °C
            "crystallinity": "Low"
        },
        "printing_parameters": {
            "nozzle_temperature": 250,     # °C
            "bed_temperature": 80,         # °C
            "print_speed": 50,             # mm/s
            "layer_height": 0.2,           # mm
            "infill_density": 30,          # %
            "infill_pattern": "grid",
            "retraction_distance": 4.0,    # mm
            "retraction_speed": 45         # mm/s
        },
        "advanced_settings": {
            "linear_advance": 0.08,
            "coasting_enable": False,
            "z_hop": 0.5,
            "supports_enable": True,
            "brim_width": 2.0
        }
    },
    
    "rigid_high_strength": {
        "name": "PETG High Strength",
        "description": "High-strength PETG for load-bearing structures",
        "material_properties": {
            "tensile_strength": 55,
            "flexural_strength": 75,
            "impact_resistance": "Very High",
            "density": 1.29,
            "glass_transition_temp": 85,
            "crystallinity": "Medium"
        },
        "printing_parameters": {
            "nozzle_temperature": 255,
            "bed_temperature": 85,
            "print_speed": 45,
            "layer_height": 0.15,
            "infill_density": 40,
            "infill_pattern": "triangular",
            "retraction_distance": 4.5,
            "retraction_speed": 50
        },
        "advanced_settings": {
            "linear_advance": 0.06,
            "coasting_enable": False,
            "z_hop": 0.3,
            "supports_enable": True,
            "brim_width": 1.5,
            "post_processing": "annealing_recommended"
        }
    },
    
    "rigid_lightweight": {
        "name": "PETG Lightweight",
        "description": "Lightweight PETG for reduced-weight applications",
        "material_properties": {
            "tensile_strength": 45,
            "flexural_strength": 62,
            "impact_resistance": "Medium",
            "density": 1.24,
            "glass_transition_temp": 78,
            "crystallinity": "Low"
        },
        "printing_parameters": {
            "nozzle_temperature": 245,
            "bed_temperature": 75,
            "print_speed": 55,
            "layer_height": 0.25,
            "infill_density": 20,
            "infill_pattern": "honeycomb",
            "retraction_distance": 3.5,
            "retraction_speed": 40
        },
        "advanced_settings": {
            "linear_advance": 0.1,
            "coasting_enable": True,
            "z_hop": 0.4,
            "supports_enable": True,
            "brim_width": 2.5
        }
    },
    
    "medical_grade_petg": {
        "name": "Medical Grade PETG",
        "description": "Biocompatible PETG for medical applications",
        "material_properties": {
            "tensile_strength": 52,
            "flexural_strength": 71,
            "impact_resistance": "High",
            "density": 1.28,
            "glass_transition_temp": 82,
            "crystallinity": "Low",
            "biocompatible": True,
            "chemical_resistance": "High"
        },
        "printing_parameters": {
            "nozzle_temperature": 248,
            "bed_temperature": 78,
            "print_speed": 48,
            "layer_height": 0.18,
            "infill_density": 35,
            "infill_pattern": "grid",
            "retraction_distance": 4.2,
            "retraction_speed": 47
        },
        "advanced_settings": {
            "linear_advance": 0.07,
            "coasting_enable": False,
            "z_hop": 0.35,
            "supports_enable": True,
            "brim_width": 2.2,
            "enclosure_required": True,
            "post_processing": "sterilization_compatible"
        }
    }
}

# Printer-specific profiles for common 3D printers
PRINTER_PROFILES = {
    "prusa_i3_mk3s": {
        "name": "Prusa i3 MK3S+",
        "build_volume": (250, 210, 200),  # mm
        "nozzle_diameter": 0.4,
        "max_temp_nozzle": 300,
        "max_temp_bed": 120,
        "dual_material_support": False,
        "recommended_materials": ["TPU", "PETG"],
        "profile_adjustments": {
            "TPU": {
                "print_speed_modifier": 0.8,  # 20% slower
                "retraction_modifier": 0.9     # 10% less retraction
            },
            "PETG": {
                "print_speed_modifier": 1.0,
                "bed_adhesion": "smooth_pei_sheet"
            }
        }
    },
    
    "ultimaker_s3": {
        "name": "Ultimaker S3",
        "build_volume": (230, 190, 200),
        "nozzle_diameter": 0.4,
        "max_temp_nozzle": 280,
        "max_temp_bed": 100,
        "dual_material_support": True,
        "recommended_materials": ["TPU", "PETG"],
        "profile_adjustments": {
            "TPU": {
                "print_speed_modifier": 0.7,
                "flow_rate": 95
            },
            "PETG": {
                "print_speed_modifier": 1.1,
                "cooling": "minimal"
            }
        }
    },
    
    "bambu_x1_carbon": {
        "name": "Bambu Lab X1 Carbon",
        "build_volume": (256, 256, 256),
        "nozzle_diameter": 0.4,
        "max_temp_nozzle": 300,
        "max_temp_bed": 120,
        "dual_material_support": True,
        "auto_material_system": True,
        "recommended_materials": ["TPU", "PETG"],
        "profile_adjustments": {
            "TPU": {
                "print_speed_modifier": 0.9,
                "vibration_compensation": True
            },
            "PETG": {
                "print_speed_modifier": 1.2,
                "ai_monitoring": True
            }
        }
    },
    
    "generic_dual_extruder": {
        "name": "Generic Dual Extruder",
        "build_volume": (200, 200, 150),
        "nozzle_diameter": 0.4,
        "max_temp_nozzle": 260,
        "max_temp_bed": 90,
        "dual_material_support": True,
        "recommended_materials": ["TPU", "PETG"],
        "profile_adjustments": {
            "TPU": {
                "print_speed_modifier": 0.75,
                "temperature_tower_recommended": True
            },
            "PETG": {
                "print_speed_modifier": 0.95,
                "temperature_tower_recommended": True
            }
        }
    }
}

# Quality profiles for different applications
QUALITY_PROFILES = {
    "prototype": {
        "name": "Prototype Quality",
        "description": "Fast printing for concept verification",
        "layer_height": 0.3,
        "infill_density": 15,
        "print_speed_modifier": 1.5,
        "supports_minimal": True,
        "surface_finish": "rough"
    },
    
    "standard": {
        "name": "Standard Quality", 
        "description": "Balanced quality and speed for general use",
        "layer_height": 0.2,
        "infill_density": 25,
        "print_speed_modifier": 1.0,
        "supports_standard": True,
        "surface_finish": "good"
    },
    
    "high_quality": {
        "name": "High Quality",
        "description": "High resolution for final products",
        "layer_height": 0.15,
        "infill_density": 35,
        "print_speed_modifier": 0.7,
        "supports_detailed": True,
        "surface_finish": "smooth"
    },
    
    "medical_grade": {
        "name": "Medical Grade",
        "description": "Medical/clinical applications with strict requirements",
        "layer_height": 0.1,
        "infill_density": 40,
        "print_speed_modifier": 0.5,
        "supports_soluble": True,
        "surface_finish": "very_smooth",
        "post_processing_required": True,
        "quality_control": "strict"
    }
}


def get_material_preset(material_type: str, preset_name: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific material preset.
    
    Args:
        material_type: 'TPU' or 'PETG'
        preset_name: Name of the preset
        
    Returns:
        Preset dictionary or None if not found
    """
    
    if material_type.upper() == 'TPU':
        return TPU_PRESETS.get(preset_name)
    elif material_type.upper() == 'PETG':
        return PETG_PRESETS.get(preset_name)
    else:
        return None


def get_printer_profile(printer_name: str) -> Optional[Dict[str, Any]]:
    """
    Get printer-specific profile.
    
    Args:
        printer_name: Name of the printer profile
        
    Returns:
        Printer profile dictionary or None if not found
    """
    
    return PRINTER_PROFILES.get(printer_name)


def get_quality_profile(quality_name: str) -> Optional[Dict[str, Any]]:
    """
    Get quality profile settings.
    
    Args:
        quality_name: Name of the quality profile
        
    Returns:
        Quality profile dictionary or None if not found
    """
    
    return QUALITY_PROFILES.get(quality_name)


def apply_printer_adjustments(material_preset: Dict, printer_profile: Dict, material_type: str) -> Dict[str, Any]:
    """
    Apply printer-specific adjustments to material preset.
    
    Args:
        material_preset: Base material preset
        printer_profile: Printer profile with adjustments
        material_type: 'TPU' or 'PETG'
        
    Returns:
        Adjusted material preset
    """
    
    adjusted_preset = material_preset.copy()
    
    if 'profile_adjustments' in printer_profile:
        adjustments = printer_profile['profile_adjustments'].get(material_type.upper(), {})
        
        # Apply print speed modifier
        if 'print_speed_modifier' in adjustments:
            modifier = adjustments['print_speed_modifier']
            adjusted_preset['printing_parameters']['print_speed'] *= modifier
        
        # Apply retraction modifier
        if 'retraction_modifier' in adjustments:
            modifier = adjustments['retraction_modifier']
            adjusted_preset['printing_parameters']['retraction_distance'] *= modifier
        
        # Apply other specific adjustments
        for key, value in adjustments.items():
            if key not in ['print_speed_modifier', 'retraction_modifier']:
                adjusted_preset['advanced_settings'][key] = value
    
    return adjusted_preset


def create_dual_material_profile(tpu_preset: str, petg_preset: str, printer: str = None, quality: str = "standard") -> Dict[str, Any]:
    """
    Create a complete dual-material printing profile.
    
    Args:
        tpu_preset: TPU preset name
        petg_preset: PETG preset name  
        printer: Printer profile name (optional)
        quality: Quality profile name
        
    Returns:
        Complete dual-material profile
    """
    
    # Get base presets
    tpu_base = get_material_preset('TPU', tpu_preset)
    petg_base = get_material_preset('PETG', petg_preset)
    
    if not tpu_base or not petg_base:
        raise ValueError(f"Invalid preset names: TPU='{tpu_preset}', PETG='{petg_preset}'")
    
    # Apply printer adjustments if specified
    if printer:
        printer_profile = get_printer_profile(printer)
        if printer_profile:
            tpu_base = apply_printer_adjustments(tpu_base, printer_profile, 'TPU')
            petg_base = apply_printer_adjustments(petg_base, printer_profile, 'PETG')
    
    # Apply quality profile
    quality_profile = get_quality_profile(quality)
    if quality_profile:
        # Apply quality adjustments to both materials
        for material in [tpu_base, petg_base]:
            if 'layer_height' in quality_profile:
                material['printing_parameters']['layer_height'] = quality_profile['layer_height']
            if 'infill_density' in quality_profile:
                material['printing_parameters']['infill_density'] = quality_profile['infill_density']
            if 'print_speed_modifier' in quality_profile:
                material['printing_parameters']['print_speed'] *= quality_profile['print_speed_modifier']
    
    # Create combined profile
    dual_profile = {
        'profile_name': f"Dual_{tpu_preset}_{petg_preset}_{quality}",
        'creation_timestamp': bpy.context.scene.frame_current,  # Use frame as timestamp alternative
        'materials': {
            'flexible': {
                'type': 'TPU',
                'preset': tpu_preset,
                'settings': tpu_base
            },
            'rigid': {
                'type': 'PETG', 
                'preset': petg_preset,
                'settings': petg_base
            }
        },
        'printer_profile': printer,
        'quality_profile': quality,
        'dual_printing_settings': {
            'interface_layers': 2,
            'tower_enable': True,
            'ooze_shield': True,
            'material_transition_length': 150,  # mm
            'purge_volume': 25  # mm³
        }
    }
    
    return dual_profile


def get_all_preset_names() -> Dict[str, List[str]]:
    """
    Get all available preset names organized by category.
    
    Returns:
        Dictionary with preset categories and names
    """
    
    return {
        'tpu_presets': list(TPU_PRESETS.keys()),
        'petg_presets': list(PETG_PRESETS.keys()),
        'printer_profiles': list(PRINTER_PROFILES.keys()),
        'quality_profiles': list(QUALITY_PROFILES.keys())
    }


def validate_preset_compatibility(tpu_preset: str, petg_preset: str) -> Dict[str, Any]:
    """
    Validate compatibility between TPU and PETG presets.
    
    Args:
        tpu_preset: TPU preset name
        petg_preset: PETG preset name
        
    Returns:
        Validation results with warnings and recommendations
    """
    
    tpu = get_material_preset('TPU', tpu_preset)
    petg = get_material_preset('PETG', petg_preset)
    
    if not tpu or not petg:
        return {'valid': False, 'error': 'Invalid preset names'}
    
    validation = {
        'valid': True,
        'warnings': [],
        'recommendations': [],
        'compatibility_score': 100
    }
    
    # Check temperature compatibility
    tpu_temp = tpu['printing_parameters']['nozzle_temperature']
    petg_temp = petg['printing_parameters']['nozzle_temperature']
    temp_diff = abs(tpu_temp - petg_temp)
    
    if temp_diff > 25:
        validation['warnings'].append(f"Large temperature difference: {temp_diff}°C")
        validation['compatibility_score'] -= 20
    
    # Check bed temperature compatibility
    tpu_bed = tpu['printing_parameters']['bed_temperature']
    petg_bed = petg['printing_parameters']['bed_temperature']
    bed_diff = abs(tpu_bed - petg_bed)
    
    if bed_diff > 20:
        validation['warnings'].append(f"Bed temperature difference: {bed_diff}°C")
        validation['recommendations'].append("Consider using average bed temperature")
        validation['compatibility_score'] -= 10
    
    # Check print speed compatibility
    tpu_speed = tpu['printing_parameters']['print_speed']
    petg_speed = petg['printing_parameters']['print_speed']
    
    if petg_speed / tpu_speed > 2.0:
        validation['recommendations'].append("Consider matching print speeds for better layer adhesion")
        validation['compatibility_score'] -= 5
    
    return validation


# Registration functions for Blender integration
def register():
    """Register material preset system."""
    print("Chain Tool V4: Material presets loaded")
    print(f"  - {len(TPU_PRESETS)} TPU presets")
    print(f"  - {len(PETG_PRESETS)} PETG presets") 
    print(f"  - {len(PRINTER_PROFILES)} printer profiles")
    print(f"  - {len(QUALITY_PROFILES)} quality profiles")


def unregister():
    """Unregister material preset system."""
    print("Chain Tool V4: Material presets unloaded")


# Development utilities
def debug_presets():
    """Print all available presets for debugging."""
    
    print("Chain Tool V4 - Material Presets Debug:")
    
    print("\nTPU Presets:")
    for name, preset in TPU_PRESETS.items():
        shore = preset['material_properties']['shore_hardness']
        temp = preset['printing_parameters']['nozzle_temperature']
        print(f"  - {name}: Shore {shore}A, {temp}°C")
    
    print("\nPETG Presets:")
    for name, preset in PETG_PRESETS.items():
        strength = preset['material_properties']['tensile_strength']
        temp = preset['printing_parameters']['nozzle_temperature']
        print(f"  - {name}: {strength}MPa, {temp}°C")
    
    print("\nPrinter Profiles:")
    for name, profile in PRINTER_PROFILES.items():
        dual = "✓" if profile['dual_material_support'] else "✗"
        print(f"  - {name}: Dual material {dual}")
    
    print("\nQuality Profiles:")
    for name, profile in QUALITY_PROFILES.items():
        layer = profile['layer_height']
        infill = profile['infill_density']
        print(f"  - {name}: {layer}mm layers, {infill}% infill")
