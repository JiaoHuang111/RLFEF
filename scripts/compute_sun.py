import os
import pandas as pd
from pymatgen.core import Structure  # 修改后的导入
from pymatgen.analysis.structure_matcher import StructureMatcher


def process_cif_files():
    # 定义路径
    cif_folder = "all_cif_FE"
    data_folder = "data/mp_20"

    # 输出文件
    output_file_stable = "stable_cifs.txt"
    output_file_unique = "unique_cifs.txt"
    output_file_novel = "novel_cifs.txt"
    output_file_unique_novel = "unique_novel_cifs.txt"

    # 1. 统计 all_cifs_FE 下的 CIF 文件数量
    cif_files = [f for f in os.listdir(cif_folder) if f.endswith(".cif")]
    total_crystals = len(cif_files)
    print(f"Total crystals in all_cifs_FE: {total_crystals}")

    # 2. 读取 CSV 文件并构建化学式到能量数据的映射
    energy_data = {}
    mp_structures = []  # 存储 MP 数据集中的所有结构

    for csv_file in ["train.csv", "val.csv", "test.csv"]:
        csv_path = os.path.join(data_folder, csv_file)
        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            formula = row["pretty_formula"]
            formation_energy = row["formation_energy_per_atom"]
            e_above_hull = row["e_above_hull"]

            # 计算 convex hull 能量
            convex_hull_energy = formation_energy - e_above_hull
            energy_data[formula] = convex_hull_energy

            # 加载 MP 数据集中的结构（假设 CIF 文件名格式为 {formula}_{id}.cif）
            mp_cif_path = os.path.join(data_folder, f"{formula}.cif")  # 假设路径
            if os.path.exists(mp_cif_path):
                struct = Structure.from_file(mp_cif_path)
                mp_structures.append(struct)

    # 3. 筛选稳定晶体（prop <= convex_hull_energy）
    stable_cifs = []
    stable_info = {}  # {cif_file: structure}

    for cif_file in cif_files:
        parts = cif_file[:-4].split("_")
        prop_str = parts[-1]
        formula = "_".join(parts[:-1])

        try:
            prop = float(prop_str)
        except ValueError:
            print(f"Warning: Could not parse formation energy from file {cif_file}")
            continue

        if formula in energy_data and prop <= energy_data[formula]+0.1:
            stable_cifs.append(cif_file)
            # 加载结构
            struct = Structure.from_file(os.path.join(cif_folder, cif_file))
            stable_info[cif_file] = struct

    print(f"Number of stable crystals: {len(stable_cifs)}")

    # 4. 计算 Unique 晶体（在稳定晶体中结构唯一）
    unique_cifs = []
    structure_matcher = StructureMatcher()  # 默认参数

    for i, (cif_i, struct_i) in enumerate(stable_info.items()):
        is_unique = True
        formula_i = "_".join(cif_i.split("_")[:-1])

        for j, (cif_j, struct_j) in enumerate(stable_info.items()):
            if i == j:
                continue
            formula_j = "_".join(cif_j.split("_")[:-1])

            # 仅比较化学式相同的晶体
            # if formula_i == formula_j:
            if structure_matcher.fit(struct_i, struct_j):
                is_unique = False
                break

        if is_unique:
            unique_cifs.append(cif_i)

    print(f"Number of unique stable crystals: {len(unique_cifs)}")

    # 5. 计算 Novel 晶体（在 MP 数据集中不存在相似结构）
    novel_cifs = []

    for cif_file, struct in stable_info.items():
        is_novel = True

        for mp_struct in mp_structures:
            if structure_matcher.fit(struct, mp_struct):
                is_novel = False
                break

        if is_novel:
            novel_cifs.append(cif_file)

    print(f"Number of novel stable crystals: {len(novel_cifs)}")

    # 6. 计算 Unique + Novel 晶体
    unique_novel_cifs = list(set(unique_cifs) & set(novel_cifs))
    print(f"Number of unique AND novel stable crystals: {len(unique_novel_cifs)}")

    # 7. 保存结果到文件
    with open(output_file_stable, "w") as f:
        f.write("\n".join(stable_cifs))

    with open(output_file_unique, "w") as f:
        f.write("\n".join(unique_cifs))

    with open(output_file_novel, "w") as f:
        f.write("\n".join(novel_cifs))

    with open(output_file_unique_novel, "w") as f:
        f.write("\n".join(unique_novel_cifs))

    return {
        "stable": len(stable_cifs),
        "unique": len(unique_cifs),
        "novel": len(novel_cifs),
        "unique_novel": len(unique_novel_cifs),
    }


if __name__ == "__main__":
    results = process_cif_files()