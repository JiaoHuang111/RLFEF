from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from pymatgen.vis.structure_vtk import StructureVis

# 加载晶体结构
structure = CifParser("example.cif").get_structures()[0]
print('1')
# 可视化晶体结构
vis = StructureVis()
print('2')
vis.set_structure(structure)
print('3')
# 保存图片
output_path = "crystal_structure.png"
vis.save_image(output_path)

print(f"Crystal structure image saved to {output_path}")
