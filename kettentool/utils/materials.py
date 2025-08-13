"""
Material Creation and Management
"""

import bpy
from typing import Optional, Tuple
import kettentool.core.constants as constants

# =========================================
# MATERIAL CREATION
# =========================================

def create_material(name: str, color: Tuple[float, float, float, float]) -> bpy.types.Material:
    """Create or get material with specified color"""
    # Check if material exists
    if name in bpy.data.materials:
        mat = bpy.data.materials[name]
    else:
        # Create new material
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        
        # Setup nodes
        nodes = mat.node_tree.nodes
        nodes.clear()
        
        # Add Principled BSDF
        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf.inputs['Base Color'].default_value = color
        bsdf.inputs['Metallic'].default_value = 0.5
        bsdf.inputs['Roughness'].default_value = 0.4
        bsdf.location = (0, 0)
        
        # Add Output node
        output = nodes.new(type='ShaderNodeOutputMaterial')
        output.location = (300, 0)
        
        # Connect nodes
        mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        
        if constants.debug:
            constants.debug.info('MATERIAL', f"Created material: {name}")
    
    return mat

def get_sphere_material(sphere_type: str = 'default') -> bpy.types.Material:
    """Get material for sphere based on type"""
    if sphere_type == 'start':
        return create_material(constants.MAT_SPHERE_START, constants.COLOR_START)
    else:
        return create_material(constants.MAT_SPHERE_DEFAULT, constants.COLOR_DEFAULT)

def get_connector_material() -> bpy.types.Material:
    """Get material for connectors"""
    return create_material(constants.MAT_CONNECTOR, constants.COLOR_CONNECTOR)

def apply_material(obj: bpy.types.Object, material: bpy.types.Material):
    """Apply material to object"""
    # Clear existing materials
    obj.data.materials.clear()
    
    # Add new material
    obj.data.materials.append(material)
    
    if constants.debug:
        constants.debug.debug('MATERIAL', f"Applied {material.name} to {obj.name}")

def create_emission_material(name: str, color: Tuple[float, float, float, float], 
                           strength: float = 1.0) -> bpy.types.Material:
    """Create emission material for highlighting"""
    # Check if material exists
    if name in bpy.data.materials:
        mat = bpy.data.materials[name]
    else:
        # Create new material
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        
        # Setup nodes
        nodes = mat.node_tree.nodes
        nodes.clear()
        
        # Add Emission shader
        emission = nodes.new(type='ShaderNodeEmission')
        emission.inputs['Color'].default_value = color
        emission.inputs['Strength'].default_value = strength
        emission.location = (0, 0)
        
        # Add Output node
        output = nodes.new(type='ShaderNodeOutputMaterial')
        output.location = (300, 0)
        
        # Connect nodes
        mat.node_tree.links.new(emission.outputs['Emission'], output.inputs['Surface'])
    
    return mat

def update_material_color(material_name: str, color: Tuple[float, float, float, float]):
    """Update color of existing material"""
    if material_name not in bpy.data.materials:
        return
    
    mat = bpy.data.materials[material_name]
    if not mat.use_nodes:
        return
    
    # Find Principled BSDF node
    for node in mat.node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            node.inputs['Base Color'].default_value = color
            break
        elif node.type == 'EMISSION':
            node.inputs['Color'].default_value = color
            break

def cleanup_unused_materials():
    """Remove unused materials from the file"""
    removed = 0
    
    # Find unused materials
    for mat in list(bpy.data.materials):
        if mat.users == 0:
            bpy.data.materials.remove(mat)
            removed += 1
    
    if removed > 0 and constants.debug:
        constants.debug.info('MATERIAL', f"Removed {removed} unused materials")
    
    return removed

def get_material_by_prefix(prefix: str) -> list:
    """Get all materials with specified prefix"""
    materials = []
    for mat in bpy.data.materials:
        if mat.name.startswith(prefix):
            materials.append(mat)
    return materials

# =========================================
# REGISTRATION
# =========================================

def register():
    """Register materials module"""
    pass

def unregister():
    """Unregister materials module"""
    pass


def set_material(obj: bpy.types.Object, mat_name: str, color: tuple,
                alpha: float = 1.0, metallic: float = 0.0, roughness: float = 0.5):
    """Apply or create material for object"""
    import bpy
    
    # Get or create material
    mat = bpy.data.materials.get(mat_name)
    if not mat:
        mat = bpy.data.materials.new(mat_name)
        mat.use_nodes = True
        
        # Setup nodes
        bsdf = mat.node_tree.nodes.get('Principled BSDF')
        if bsdf:
            bsdf.inputs['Base Color'].default_value = (*color, alpha)
            bsdf.inputs['Metallic'].default_value = metallic
            bsdf.inputs['Roughness'].default_value = roughness
            if alpha < 1.0:
                mat.blend_method = 'BLEND'
    
    # Apply to object
    if obj.data and hasattr(obj.data, 'materials'):
        if len(obj.data.materials) > 0:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)
    
    return mat
