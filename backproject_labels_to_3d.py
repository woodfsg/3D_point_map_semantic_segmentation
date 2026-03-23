import numpy as np
import cv2
import pickle
import os

def backproject_labels_to_3d(ply_path, map_2d_path, mapping_pkl_path, output_ply_path):
    """
    根据 .pkl 映射文件，将 2D 语义标签反投回原始 3D 点云。
    """
    print("⏳ 正在加载文件，请稍候...")
    
    # 1. 加载 2D 语义分割图
    if not os.path.exists(map_2d_path):
        raise FileNotFoundError(f"找不到 2D 语义图: {map_2d_path}")
    # 灰度模式读取，每个像素的值就是类别 ID
    semantic_map_2d = cv2.imread(map_2d_path, cv2.IMREAD_GRAYSCALE)
    height, width = semantic_map_2d.shape
    print(f"✅ 2D 语义图加载完成，尺寸: {width} x {height}")

    # 2. 加载投影映射字典 (.pkl)
    if not os.path.exists(mapping_pkl_path):
        raise FileNotFoundError(f"找不到映射文件: {mapping_pkl_path}")
    with open(mapping_pkl_path, "rb") as f:
        mapping_index = pickle.load(f)
    print(f"✅ 映射字典加载完成，共包含 {len(mapping_index)} 个有效像素坐标")

    # 3. 加载原始 3D 点云 (以二进制方式高速读取)
    print(f"📂 正在读取原始 PLY 文件: {ply_path}")
    origin_dt = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                          ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    
    with open(ply_path, 'rb') as f:
        header_lines = []
        while True:
            line = f.readline().decode('utf-8')
            header_lines.append(line)
            if line.strip() == "end_header":
                break
        
        # 提取点云总数
        num_points = 0
        for line in header_lines:
            if line.startswith("element vertex"):
                num_points = int(line.split()[2])
                break
                
        # 读取二进制顶点数据
        vertex_data = np.frombuffer(f.read(), dtype=origin_dt)

    # 4. 初始化标签数组并进行反向映射
    print("🔍 正在执行反向投影...")
    # 初始化所有点的标签为 0 (背景)
    point_labels = np.zeros(num_points, dtype=np.uint8)
    
    # 遍历字典中的每一个 (u, v) 像素坐标及其对应的 3D 点索引列表
    for (u, v), point_indices in mapping_index.items():
        # 边界安全检查，防止索引越界
        if 0 <= u < width and 0 <= v < height:
            # 获取该像素在 2D 图上的类别 ID
            label_id = semantic_map_2d[v, u] 
            
            # 将该类别 ID 批量赋给所有对应的 3D 点
            point_labels[point_indices] = label_id

    # 5. 构建并保存带有语义标签的新点云
    print("💾 正在写入带有语义标签的新 PLY 文件...")
    # 新增 'scalar_Label' 字段
    new_dt = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                       ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), 
                       ('scalar_Label', 'f4')])
    
    new_vertex_data = np.empty(num_points, dtype=new_dt)
    for name in origin_dt.names:
        new_vertex_data[name] = vertex_data[name]
    new_vertex_data['scalar_Label'] = point_labels.astype(np.float32)

    # 写入新文件
    with open(output_ply_path, 'wb') as f:
        header = (f"ply\n"
                  f"format binary_little_endian 1.0\n"
                  f"element vertex {num_points}\n"
                  f"property float x\n"
                  f"property float y\n"
                  f"property float z\n"
                  f"property uchar red\n"
                  f"property uchar green\n"
                  f"property uchar blue\n"
                  f"property float scalar_Label\n"  
                  f"end_header\n").encode('utf-8')
        f.write(header)
        f.write(new_vertex_data.tobytes())

    print(f"✨ 流程全部走完！带语义标签的 3D 点云已成功保存至: {output_ply_path}")

# ==========================================
# 执行入口
# ==========================================
if __name__ == '__main__':
    # 请根据您本地的实际路径进行替换
    ORIGIN_PLY = '../ply_files/S3DIS_Area2_Original.ply' # 或者使用 S3DIS_Area2_6Classes.ply
    MAP_2D_PNG = '../2d_segmentation_files/2d_semantic_segmentation_map.png'      # 第3步生成的图片
    MAPPING_PKL = '../bev_files/bev_mapping_-50_50.pkl'         # 您的映射文件
    OUTPUT_PLY = '../ply_files/S3DIS_Area2_Final_Segmented.ply'
    
    backproject_labels_to_3d(ORIGIN_PLY, MAP_2D_PNG, MAPPING_PKL, OUTPUT_PLY)