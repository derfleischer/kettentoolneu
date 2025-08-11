"""
Material System
"""

import bpy
from typing import Tuple

# =========================================
# MATERIAL UTILITIES
# =========================================

def set_material(obj: bpy.types.Object, mat_name: str, color: Tuple[float, float, float],
                 alpha: float = 1.0, metallic: float = 0.0, roughness: float = 0.5):
    """Erstellt und setzt Material"""
    from ..core import constants
    
    if constants.performance_monitor:
        constants.performance_monitor.start_timer("set_material")
    
    mat = bpy.data.materials.get(mat_name)
    if not mat:
        if constants.debug:
            constants.debug.trace('MATERIALS', f"Creating material: {mat_name}")
        mat = bpy.data.materials.new(mat_name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get('Principled BSDF')
        if bsdf:
            bsdf.inputs['Base Color'].default_value = (*color, 1)
            bsdf.inputs['Alpha'].default_value = alpha
            bsdf.inputs['Metallic'].default_value = metallic
            bsdf.inputs['Roughness'].default_value = roughness
        mat.blend_method = 'BLEND' if alpha < 1.0 else 'OPAQUE'
    
    if obj.data and hasattr(obj.data, 'materials'):
        if len(obj.data.materials) > 0:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)
    
    if constants.performance_monitor:
        constants.performance_monitor.end_timer("set_material")

def create_default_materials():
    """Erstellt Standard-Materialien"""
    from ..core import constants
    
    materials = {
        "KetteSchwarz": (0.02, 0.02, 0.02, 1.0, 0.8, 0.2),
        "KetteStart": (1.0, 0.0, 0.0, 1.0, 0.5, 0.5),
        "KetteVerbinder": (0.1, 0.1, 0.1, 1.0, 0.9, 0.3),
        "KettePaint": (0.5, 0.7, 1.0, 1.0, 0.3, 0.7)
    }
    
    # Dummy-Objekt für Material-Erstellung
    temp_mesh = bpy.data.meshes.new("temp")
    temp_obj = bpy.data.objects.new("temp", temp_mesh)
    
    for mat_name, (r, g, b, alpha, metallic, roughness) in materials.items():
        if not bpy.data.materials.get(mat_name):
            set_material(temp_obj, mat_name, (r, g, b), alpha, metallic, roughness)
    
    # Cleanup
    bpy.data.objects.remove(temp_obj)
    bpy.data.meshes.remove(temp_mesh)
    
    if constants.debug:
        constants.debug.info('MATERIALS', f"Created {len(materials)} default materials")

# =========================================
# REGISTRATION
# =========================================

def register():
    """Registriert Materials-Modul"""
    create_default_materials()

def unregister():
    """Deregistriert Materials-Modul"""
    pass
