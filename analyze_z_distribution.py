import numpy as np
from plyfile import PlyData

def analyze_z_distribution(ply_path):
    print(f"正在分析文件: {ply_path}")
    
    # 1. 加载数据
    plydata = PlyData.read(ply_path)
    v_data = plydata['vertex']
    z_coords = np.array(v_data['z'])
    labels = np.array(v_data['scalar_Label']).astype(np.int32)

    # 2. 全局统计
    print("-" * 30)
    print("【全局高度统计】")
    print(f"绝对最小值 (Min Z): {z_coords.min():.4f} m")
    print(f"绝对最大值 (Max Z): {z_coords.max():.4f} m")
    
    # 使用百分位数排除离群点（噪声）
    z_99 = np.percentile(z_coords, [0.1, 1, 99, 99.9])
    print(f"0.1% 分位数 (建议地板下限): {z_99[0]:.4f} m")
    print(f"99.9% 分位数 (建议天花板上限): {z_99[3]:.4f} m")

    # 3. 按类别统计（重点分析 Auditorium）
    class_names = {
        1: 'auditorium',
        2: 'conferenceRoom',
        3: 'hallway',
        4: 'office',
        5: 'storage',
        6: 'WC'
    }

    print("-" * 30)
    print("【各区域高度详情】")
    print(f"{'Class ID':<10} | {'Name':<15} | {'Min Z':<10} | {'Max Z':<10}")
    
    for label_id, name in class_names.items():
        mask = (labels == label_id)
        if np.any(mask):
            class_z = z_coords[mask]
            print(f"{label_id:<10} | {name:<15} | {class_z.min():.4f} | {class_z.max():.4f}")
        else:
            print(f"{label_id:<10} | {name:<15} | {'无数据':<10}")

if __name__ == "__main__":
    # 请确保路径指向你的 Area 2 文件
    path = "../ply_files/S3DIS_Area2_6Classes.ply" 
    analyze_z_distribution(path)