import os
import pandas as pd


def process_cif_files():
    # 定义路径
    cif_folder = "all_cifs_FE"
    data_folder = "data/mp_20"
    output_file_ehull_0 = "ehull_0_files.txt"  # 存储能量<=hull的文件名
    output_file_ehull_01 = "ehull_01_files.txt"  # 存储能量<=hull+0.1的文件名

    # 1. 统计all_cifs_FE下的CIF文件数量
    cif_files = [f for f in os.listdir(cif_folder) if f.endswith(".cif")]
    total_crystals = len(cif_files)
    print(f"Total crystals in all_cifs_FE: {total_crystals}")

    # 2. 读取CSV文件并构建化学式到能量数据的映射
    energy_data = {}

    for csv_file in ["train.csv", "val.csv", "test.csv"]:
        csv_path = os.path.join(data_folder, csv_file)
        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            formula = row["pretty_formula"]
            formation_energy = row["formation_energy_per_atom"]
            e_above_hull = row["e_above_hull"]

            # 计算convex hull能量
            convex_hull_energy = formation_energy - e_above_hull

            energy_data[formula] = {
                "convex_hull_energy": convex_hull_energy
            }

    # 3. 准备写入结果文件
    with open(output_file_ehull_0, 'w') as f0, open(output_file_ehull_01, 'w') as f01:
        # 遍历CIF文件并统计
        count_ehull_0 = 0
        count_ehull_01 = 0

        for cif_file in cif_files:
            # 解析化学式和形成能
            parts = cif_file[:-4].split("_")
            prop_str = parts[-1]
            formula = "_".join(parts[:-1])

            try:
                prop = float(prop_str)
            except ValueError:
                print(f"Warning: Could not parse formation energy from file {cif_file}")
                continue

            # 获取convex hull能量
            if formula in energy_data:
                convex_hull_energy = energy_data[formula]["convex_hull_energy"]

                # 判断并写入相应文件
                if prop <= convex_hull_energy:
                    count_ehull_0 += 1
                    f0.write(cif_file + '\n')

                if prop <= convex_hull_energy + 0.1:
                    count_ehull_01 += 1
                    f01.write(cif_file + '\n')

    # 打印结果
    print(f"Number of crystals with energy <= convex hull: {count_ehull_0}")
    print(f"Number of crystals with energy <= convex hull + 0.1: {count_ehull_01}")
    print(f"Results saved to {output_file_ehull_0} and {output_file_ehull_01}")


if __name__ == "__main__":
    process_cif_files()