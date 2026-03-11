import os
import numpy as np
from pathlib import Path
import struct

def export_area_original_rgb(dataset_root, area_name, output_filename):
    area_path = Path(dataset_root) / area_name
    
    if not area_path.exists():
        print(f"❌ 错误：在 {dataset_root} 下找不到 {area_name}")
        return

    all_xyz = []
    all_rgb = []
    
    print(f"🔍 正在从 {area_name} 提取原始颜色数据...")

    # 1. 遍历所有房间
    rooms = list(area_path.iterdir())
    for i, room_dir in enumerate(rooms):
        if not room_dir.is_dir(): continue
        
        anno_path = room_dir / 'Annotations'
        if not anno_path.exists(): continue
        
        # 2. 遍历房间内所有物体的 .txt 文件
        for obj_file in anno_path.glob('*.txt'):
            if 'ceiling' in obj_file.name.lower():
                # print(f"  [跳过] {room_name} 的天花板")
                continue
            try:
                # S3DIS 格式: X Y Z R G B (通常还有一列分类，我们忽略)
                data = np.loadtxt(obj_file)
                if data.size == 0: continue
                if len(data.shape) == 1: data = data.reshape(1, -1)
                
                # 提取 XYZ (前3列) 和 RGB (4-6列)
                all_xyz.append(data[:, :3].astype(np.float32))
                all_rgb.append(data[:, 3:6].astype(np.uint8))
                
            except Exception as e:
                print(f"读取 {obj_file.name} 时出错: {e}")
        
        if (i + 1) % 10 == 0:
            print(f"  进度: 已处理 {i + 1}/{len(rooms)} 个文件夹")

    if not all_xyz:
        print("❌ 未发现有效点云数据。")
        return

    # 3. 合并数据
    final_xyz = np.vstack(all_xyz)
    final_rgb = np.vstack(all_rgb)
    num_points = len(final_xyz)

    # 4. 写入二进制 PLY 文件
    print(f"💾 正在写入二进制 PLY (共 {num_points} 个点)...")
    
    with open(output_filename, 'wb') as f:
        # 写入 PLY 文件头
        header = (f"ply\n"
                  f"format binary_little_endian 1.0\n"
                  f"element vertex {num_points}\n"
                  f"property float x\n"
                  f"property float y\n"
                  f"property float z\n"
                  f"property uchar red\n"
                  f"property uchar green\n"
                  f"property uchar blue\n"
                  f"end_header\n").encode('utf-8')
        f.write(header)
        
        # 使用 numpy 的 tobytes() 配合结构化数组实现极速写入
        # 构造一个复合数据类型：3个float32 + 3个uint8
        dt = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                       ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        
        v_data = np.empty(num_points, dtype=dt)
        v_data['x'] = final_xyz[:, 0]
        v_data['y'] = final_xyz[:, 1]
        v_data['z'] = final_xyz[:, 2]
        v_data['red'] = final_rgb[:, 0]
        v_data['green'] = final_rgb[:, 1]
        v_data['blue'] = final_rgb[:, 2]
        
        f.write(v_data.tobytes())

    print(f"✨ 完成！原始颜色点云已保存至: {output_filename}")

# --- 配置路径 ---
# 确保此路径指向包含 Area_1, Area_2... 的文件夹
DATASET_ROOT = '../Stanford3dDataset_v1.2'
OUTPUT_FILE = './ply_files/S3DIS_Area2_Original.ply'

export_area_original_rgb(DATASET_ROOT, 'Area_2', OUTPUT_FILE)