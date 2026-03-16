import open3d as o3d
import numpy as np
import cv2
import pickle

def generate_bev_with_mapping(ply_path, resolution=0.05, z_range=[1.5, 2.0]):
    """
    生成高分辨率BEV图像并保留点到像素的映射索引
    :param ply_path: S3DIS PLY文件路径
    :param resolution: 每个像素代表的实际尺寸（米），默认0.05m (5cm)
    :param z_range: 高度过滤范围，过滤掉地面和天花板以突出功能区
    """
    print(f"正在加载点云: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    
    # 1. 预处理：高度过滤
    mask = (points[:, 2] > z_range[0]) & (points[:, 2] < z_range[1])
    filtered_points = points[mask]
    original_indices = np.where(mask)[0] # 记录过滤后点在原点云中的索引

    x_coords = filtered_points[:, 0]
    y_coords = filtered_points[:, 1]
    z_coords = filtered_points[:, 2]

    # 2. 计算边界与图像尺寸
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    width = int(np.ceil((x_max - x_min) / resolution)) + 1
    height = int(np.ceil((y_max - y_min) / resolution)) + 1
    print(f"生成的BEV尺寸为: {width} x {height}")

    # 3. 映射 3D 点到 2D 像素坐标
    u = ((x_coords - x_min) / resolution).astype(np.int32)
    v = ((y_coords - y_min) / resolution).astype(np.int32)

    # 4. 构建映射索引 (Mapping Index)
    # 使用字典存储，key为(u, v)，value为原始点云索引列表
    mapping_index = {}
    
    # 5. 初始化特征统计矩阵
    # 我们需要统计：Max Z (R), Count (G), Sum Z (用于计算B通道的Mean Z)
    z_max_img = np.zeros((height, width), dtype=np.float32)
    count_img = np.zeros((height, width), dtype=np.float32)
    z_sum_img = np.zeros((height, width), dtype=np.float32)

    # 遍历点云进行填充（此步骤可使用 numba 加速，但在几千万点规模下 NumPy 迭代尚可接受）
    for i in range(len(filtered_points)):
        curr_u, curr_v = u[i], v[i]
        orig_idx = original_indices[i]
        
        # 更新映射索引
        pix_coord = (curr_u, curr_v)
        if pix_coord not in mapping_index:
            mapping_index[pix_coord] = []
        mapping_index[pix_coord].append(int(orig_idx))

        # 更新特征统计
        count_img[curr_v, curr_u] += 1
        z_sum_img[curr_v, curr_u] += z_coords[i]
        if z_coords[i] > z_max_img[curr_v, curr_u]:
            z_max_img[curr_v, curr_u] = z_coords[i]

    # 6. 生成 RGB 通道
    # R: Max Height 归一化
    r_channel = ((z_max_img - z_range[0]) / (z_range[1] - z_range[0]) * 255)
    
    # G: Density 归一化 (使用log处理以防局部密度过高导致图像太暗)
    g_channel = np.log1p(count_img)
    g_channel = (g_channel / g_channel.max() * 255)
    
    # B: Mean Height 归一化
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_z = z_sum_img / count_img
        mean_z[np.isnan(mean_z)] = 0
    b_channel = ((mean_z - z_range[0]) / (z_range[1] - z_range[0]) * 255)

    # 合成 BEV 图像
    bev_img = np.dstack((b_channel, g_channel, r_channel)).astype(np.uint8) # OpenCV使用BGR

    return bev_img, mapping_index, (x_min, y_min, resolution)

# 测试运行
if __name__ == "__main__":
    # 请替换为你本地 S3DIS Area_2 的实际路径
    ply_file_path = "../ply_files/S3DIS_Area2_6Classes.ply" 
    
    bev, mapping, meta = generate_bev_with_mapping(ply_file_path)

    # 保存结果
    bev_folder_path="../bev_files/"
    png_file_path=bev_folder_path+"S3DIS_Area2_BEV.png"
    mapping_file_path=bev_folder_path+"bev_mapping.pkl"
    cv2.imwrite(png_file_path, bev)
    with open(mapping_file_path, "wb") as f:
        pickle.dump(mapping, f)
    
    print("处理完成！BEV 图像与索引文件已保存。")