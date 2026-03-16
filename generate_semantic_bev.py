import open3d as o3d
import numpy as np
import cv2
from plyfile import PlyData 

def generate_semantic_bev(ply_path, resolution=0.05, z_range=[-50.0, 50.0]):
    class_map = {
        1: [255, 0, 0],      # auditorium
        2: [0, 255, 0],      # conferenceRoom
        3: [0, 0, 255],      # hallway
        4: [255, 255, 0],    # office
        5: [0, 255, 255],    # storage
        6: [255, 0, 255]     # WC
    }

    print(f"正在使用 plyfile 加载点云: {ply_path}")
    
    # 1. 使用 plyfile 读取所有属性
    plydata = PlyData.read(ply_path)
    v_data = plydata['vertex']
    
    # 提取坐标和标签
    points = np.stack([v_data['x'], v_data['y'], v_data['z']], axis=1)
    # 关键点：直接通过属性名提取 scalar_Label
    labels = np.array(v_data['scalar_Label']).astype(np.int32)

    # 2. 高度过滤 (与之前逻辑一致)
    mask = (points[:, 2] > z_range[0]) & (points[:, 2] < z_range[1])
    f_points = points[mask]
    f_labels = labels[mask]

    # 3. 计算画布尺寸
    x_min, y_min = np.min(f_points[:, 0]), np.min(f_points[:, 1])
    x_max, y_max = np.max(f_points[:, 0]), np.max(f_points[:, 1])
    
    width = int(np.ceil((x_max - x_min) / resolution)) + 1
    height = int(np.ceil((y_max - y_min) / resolution)) + 1
    
    # 4. 映射坐标并生成语义 BEV
    u = ((f_points[:, 0] - x_min) / resolution).astype(np.int32)
    v = ((f_points[:, 1] - y_min) / resolution).astype(np.int32)

    semantic_bev = np.zeros((height, width, 3), dtype=np.uint8)

    # 5. 填充颜色 (BGR 顺序)
    for i in range(len(f_points)):
        label = f_labels[i]
        if label in class_map:
            c = class_map[label]
            semantic_bev[v[i], u[i]] = [c[2], c[1], c[0]]

    return semantic_bev


# 运行
if __name__ == "__main__":
    file_path = "../ply_files/S3DIS_Area2_6Classes.ply" # 替换为你的路径
    res_img = generate_semantic_bev(file_path)
    
    if res_img is not None:
        cv2.imwrite("../semantic_files/S3DIS_Area2_Semantic_GT.png", res_img)
        print("语义真值 BEV 已生成！你可以将其与大模型推理结果进行对比。")