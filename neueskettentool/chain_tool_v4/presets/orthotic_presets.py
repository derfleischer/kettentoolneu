"""
Chain Tool V4 - Orthotic Presets
================================

Predefined orthotic templates and configurations for different medical applications.
"""

import bpy
from typing import Dict, List, Any, Optional

# Orthotic template definitions
ORTHOTIC_TEMPLATES = {
    "ankle_foot_orthosis": {
        "name": "Ankle-Foot Orthosis (AFO)",
        "description": "Standard ankle-foot orthosis for drop foot correction",
        "category": "lower_extremity",
        "application": "Drop foot, muscle weakness, spasticity",
        "pattern_config": {
            "pattern_type": "HYBRID",
            "surface_subtype": "VORONOI",
            "density": 0.8,
            "connection_type": "ADAPTIVE",
            "max_distance": 0.12,
            "thickness": 0.025
        },
        "material_config": {
            "tpu_zones": ["ankle_joint", "foot_interface"],
            "petg_zones": ["structural_frame", "mounting_points"],
            "tpu_preset": "flexible_standard",
            "petg_preset": "rigid_standard"
        },
        "design_parameters": {
            "ankle_angle": 90,  # degrees
            "heel_height": 0.02,  # meters
            "toe_clearance": 0.015,
            "strapping_points": 3
        }
    },
    
    "knee_ankle_foot_orthosis": {
        "name": "Knee-Ankle-Foot Orthosis (KAFO)",
        "description": "Extended orthosis including knee joint support",
        "category": "lower_extremity",
        "application": "Knee instability, quadriceps weakness",
        "pattern_config": {
            "pattern_type": "HYBRID", 
            "surface_subtype": "HEXAGONAL",
            "density": 0.9,
            "connection_type": "ADAPTIVE",
            "max_distance": 0.15,
            "thickness": 0.03
        },
        "material_config": {
            "tpu_zones": ["knee_joint", "ankle_joint", "skin_interface"],
            "petg_zones": ["thigh_frame", "shin_frame", "foot_plate"],
            "tpu_preset": "flexible_durable",
            "petg_preset": "rigid_high_strength"
        },
        "design_parameters": {
            "knee_angle": 5,  # slight flexion
            "ankle_angle": 90,
            "thigh_length_ratio": 0.6,
            "strapping_points": 5
        }
    },
    
    "wrist_hand_orthosis": {
        "name": "Wrist-Hand Orthosis (WHO)",
        "description": "Wrist and hand support orthosis",
        "category": "upper_extremity",
        "application": "Carpal tunnel, wrist fractures, tendon injuries",
        "pattern_config": {
            "pattern_type": "SURFACE",
            "surface_subtype": "TRIANGULAR",
            "density": 1.2,
            "connection_type": "CURVED",
            "max_distance": 0.08,
            "thickness": 0.015
        },
        "material_config": {
            "tpu_zones": ["wrist_joint", "palm_interface"],
            "petg_zones": ["forearm_frame", "thumb_support"],
            "tpu_preset": "flexible_soft",
            "petg_preset": "rigid_lightweight"
        },
        "design_parameters": {
            "wrist_angle": 15,  # extension
            "thumb_position": "opposed",
            "finger_clearance": 0.005,
            "ventilation_holes": True
        }
    },
    
    "spinal_orthosis": {
        "name": "Thoraco-Lumbar-Sacral Orthosis (TLSO)",
        "description": "Spinal support orthosis for thoraco-lumbar region",
        "category": "spinal",
        "application": "Scoliosis, compression fractures, post-surgical support",
        "pattern_config": {
            "pattern_type": "HYBRID",
            "surface_subtype": "VORONOI",
            "density": 0.7,
            "connection_type": "ADAPTIVE",
            "max_distance": 0.2,
            "thickness": 0.04
        },
        "material_config": {
            "tpu_zones": ["pressure_relief_pads", "breathing_expansion"],
            "petg_zones": ["structural_supports", "closure_mechanism"],
            "tpu_preset": "medical_grade_tpu",
            "petg_preset": "medical_grade_petg"
        },
        "design_parameters": {
            "correction_force": "moderate",
            "breathing_room": 0.02,
            "pressure_distribution": "even",
            "closure_type": "velcro_straps"
        }
    },
    
    "pediatric_afo": {
        "name": "Pediatric Ankle-Foot Orthosis",
        "description": "Child-sized AFO with growth considerations",
        "category": "pediatric",
        "application": "Cerebral palsy, developmental delays",
        "pattern_config": {
            "pattern_type": "SURFACE",
            "surface_subtype": "HEXAGONAL",
            "density": 1.0,
            "connection_type": "CURVED",
            "max_distance": 0.06,
            "thickness": 0.02
        },
        "material_config": {
            "tpu_zones": ["entire_interface"],  # More comfortable for children
            "petg_zones": ["minimal_structure"],
            "tpu_preset": "flexible_soft",
            "petg_preset": "rigid_lightweight"
        },
        "design_parameters": {
            "growth_allowance": 0.01,  # 1cm growth room
            "weight_optimization": True,
            "fun_colors": True,
            "easy_donning": True
        }
    },
    
    "sports_ankle_brace": {
        "name": "Sports Ankle Brace",
        "description": "Athletic ankle support for sports activities",
        "category": "sports",
        "application": "Ankle sprains, sports prevention, return to play",
        "pattern_config": {
            "pattern_type": "EDGE",
            "surface_subtype": "TRIANGULAR",
            "density": 1.5,
            "connection_type": "STRAIGHT",
            "max_distance": 0.05,
            "thickness": 0.012
        },
        "material_config": {
            "tpu_zones": ["dynamic_support_bands"],
            "petg_zones": ["rigid_side_supports"],
            "tpu_preset": "flexible_durable",
            "petg_preset": "rigid_lightweight"
        },
        "design_parameters": {
            "flexibility_retention": 0.8,  # 80% natural motion
            "impact_protection": True,
            "moisture_wicking": True,
            "low_profile": True
        }
    }
}

# Sizing templates
SIZING_TEMPLATES = {
    "adult_male": {
        "foot_length": 0.27,  # 27cm average
        "ankle_circumference": 0.25,
        "calf_circumference": 0.38,
        "weight_factor": 1.0
    },
    "adult_female": {
        "foot_length": 0.24,  # 24cm average  
        "ankle_circumference": 0.23,
        "calf_circumference": 0.35,
        "weight_factor": 0.85
    },
    "child_6_10": {
        "foot_length": 0.18,  # 18cm average
        "ankle_circumference": 0.18,
        "calf_circumference": 0.25,
        "weight_factor": 0.4
    },
    "child_11_15": {
        "foot_length": 0.22,  # 22cm average
        "ankle_circumference": 0.21,
        "calf_circumference": 0.30,
        "weight_factor": 0.6
    }
}

def get_orthotic_template(template_name: str) -> Optional[Dict[str, Any]]:
    """
    Get orthotic template by name
    
    Args:
        template_name: Name of the template
        
    Returns:
        Template dictionary or None if not found
    """
    return ORTHOTIC_TEMPLATES.get(template_name)

def get_templates_by_category(category: str) -> List[Dict[str, Any]]:
    """
    Get all templates for a specific category
    
    Args:
        category: Category name (lower_extremity, upper_extremity, spinal, pediatric, sports)
        
    Returns:
        List of matching templates
    """
    return [template for template in ORTHOTIC_TEMPLATES.values() 
            if template.get('category') == category]

def get_available_categories() -> List[str]:
    """Get list of available orthotic categories"""
    categories = set()
    for template in ORTHOTIC_TEMPLATES.values():
        categories.add(template.get('category', 'other'))
    return sorted(list(categories))

def create_orthotic_preset(template_name: str, size_template: str = "adult_male", 
                          custom_params: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Create a complete orthotic preset from template and sizing
    
    Args:
        template_name: Name of orthotic template
        size_template: Sizing template name
        custom_params: Custom parameter overrides
        
    Returns:
        Complete orthotic preset
    """
    # Get base template
    orthotic_template = get_orthotic_template(template_name)
    if not orthotic_template:
        raise ValueError(f"Unknown orthotic template: {template_name}")
    
    # Get sizing information
    sizing = SIZING_TEMPLATES.get(size_template, SIZING_TEMPLATES["adult_male"])
    
    # Create complete preset
    preset = {
        'name': f"{orthotic_template['name']} - {size_template}",
        'template_base': template_name,
        'size_template': size_template,
        'orthotic_config': orthotic_template.copy(),
        'sizing_config': sizing.copy(),
        'creation_timestamp': bpy.context.scene.frame_current,
        'custom_modifications': custom_params or {}
    }
    
    # Apply custom parameter overrides
    if custom_params:
        _apply_custom_parameters(preset, custom_params)
    
    # Apply size-based adjustments
    _apply_size_adjustments(preset)
    
    return preset

def _apply_custom_parameters(preset: Dict, custom_params: Dict):
    """Apply custom parameter modifications to preset"""
    
    # Pattern modifications
    if 'pattern' in custom_params:
        preset['orthotic_config']['pattern_config'].update(custom_params['pattern'])
    
    # Material modifications
    if 'materials' in custom_params:
        preset['orthotic_config']['material_config'].update(custom_params['materials'])
    
    # Design modifications
    if 'design' in custom_params:
        preset['orthotic_config']['design_parameters'].update(custom_params['design'])

def _apply_size_adjustments(preset: Dict):
    """Apply size-based adjustments to preset parameters"""
    
    sizing = preset['sizing_config']
    pattern_config = preset['orthotic_config']['pattern_config']
    
    # Scale connection distances based on foot size
    if 'foot_length' in sizing:
        scale_factor = sizing['foot_length'] / SIZING_TEMPLATES['adult_male']['foot_length']
        pattern_config['max_distance'] *= scale_factor
        pattern_config['thickness'] *= scale_factor
    
    # Adjust density based on weight factor
    if 'weight_factor' in sizing:
        pattern_config['density'] *= sizing['weight_factor']

def validate_orthotic_preset(preset: Dict) -> Dict[str, Any]:
    """
    Validate orthotic preset for completeness and consistency
    
    Args:
        preset: Orthotic preset to validate
        
    Returns:
        Validation results
    """
    validation = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'score': 100
    }
    
    # Check required fields
    required_fields = ['name', 'orthotic_config', 'sizing_config']
    for field in required_fields:
        if field not in preset:
            validation['errors'].append(f"Missing required field: {field}")
            validation['is_valid'] = False
    
    # Check orthotic config
    if 'orthotic_config' in preset:
        config = preset['orthotic_config']
        
        # Check pattern config
        if 'pattern_config' not in config:
            validation['errors'].append("Missing pattern_config")
        else:
            pattern = config['pattern_config']
            if pattern.get('thickness', 0) < 0.01:
                validation['warnings'].append("Very thin orthotic - may lack strength")
            if pattern.get('density', 0) > 2.0:
                validation['warnings'].append("Very dense pattern - may be heavy")
        
        # Check material config
        if 'material_config' not in config:
            validation['errors'].append("Missing material_config")
        else:
            materials = config['material_config']
            if not materials.get('tpu_zones') and not materials.get('petg_zones'):
                validation['warnings'].append("No material zones defined")
    
    # Calculate validation score
    validation['score'] -= len(validation['errors']) * 25
    validation['score'] -= len(validation['warnings']) * 5
    validation['score'] = max(0, validation['score'])
    
    return validation

def get_orthotic_recommendations(patient_condition: str, age_group: str = "adult") -> List[str]:
    """
    Get orthotic recommendations based on patient condition
    
    Args:
        patient_condition: Medical condition or requirement
        age_group: Age group (pediatric, adult, elderly)
        
    Returns:
        List of recommended orthotic template names
    """
    recommendations = []
    
    condition_lower = patient_condition.lower()
    
    # Condition-based recommendations
    if 'drop foot' in condition_lower or 'foot drop' in condition_lower:
        recommendations.append('ankle_foot_orthosis')
        if age_group == 'pediatric':
            recommendations.append('pediatric_afo')
    
    elif 'ankle sprain' in condition_lower or 'sports' in condition_lower:
        recommendations.append('sports_ankle_brace')
    
    elif 'knee' in condition_lower:
        recommendations.append('knee_ankle_foot_orthosis')
    
    elif 'wrist' in condition_lower or 'carpal tunnel' in condition_lower:
        recommendations.append('wrist_hand_orthosis')
    
    elif 'spine' in condition_lower or 'scoliosis' in condition_lower:
        recommendations.append('spinal_orthosis')
    
    elif 'cerebral palsy' in condition_lower or age_group == 'pediatric':
        recommendations.append('pediatric_afo')
    
    # Default fallback
    if not recommendations:
        recommendations.append('ankle_foot_orthosis')  # Most common
    
    return recommendations

# Registration functions
def register():
    """Register orthotic presets"""
    print(f"Chain Tool V4: {len(ORTHOTIC_TEMPLATES)} orthotic templates loaded")

def unregister():
    """Unregister orthotic presets"""
    print("Chain Tool V4: Orthotic templates unloaded")

# Development utilities
def debug_orthotic_templates():
    """Print all available orthotic templates"""
    print("Chain Tool V4 - Orthotic Templates:")
    
    categories = get_available_categories()
    for category in categories:
        print(f"\n{category.upper()}:")
        templates = get_templates_by_category(category)
        for template in templates:
            print(f"  - {template['name']}")
            print(f"    Application: {template['application']}")
