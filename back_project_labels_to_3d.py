import open3d as o3d
import numpy as np
import pickle
import cv2

def back_project_labels_to_3d(original_ply_path, label_img_path, mapping_pkl_path, output_path):
    """
    将 2D 语义标签图映射回 3D 点云
    :param original_ply_path: 原始 S3DIS Area2 点云路径
    :param label_img_path: 2D 语义分割结果图（像素值代表类别索引）
    :param mapping_pkl_path: 之前生成的映射索引文件
    :param output_path: 带有语义标签的 3D 点云保存路径
    """
    # 1. 加载原始点云
    print("正在加载原始点云...")
    pcd = o3d.io.read_point_cloud(original_ply_path)
    num_points = len(pcd.points)
    
    # 初始化 3D 标签数组（默认为 0，代表未分类或背景）
    point_labels = np.zeros(num_points, dtype=np.int32)

    # 2. 加载 2D 语义标签图
    # 假设标签图是单通道图像，像素值即为类别 ID
    print("正在加载 2D 语义标签...")
    label_img = cv2.imread(label_img_path, cv2.IMREAD_GRAYSCALE)
    if label_img is None:
        raise ValueError("无法读取标签图像，请检查路径。")

    # 3. 加载映射索引
    print("正在加载映射索引...")
    with open(mapping_pkl_path, "rb") as f:
        mapping_index = pickle.load(f)

    # 4. 执行反向映射
    print("正在执行反向映射...")
    # 遍历映射索引中记录的所有有效像素位置
    for (u, v), point_indices in mapping_index.items():
        # 获取该像素在 2D 图中的语义标签
        # 注意：OpenCV 图像索引是 [row, col]，即 [v, u]
        if v < label_img.shape[0] and u < label_img.shape[1]:
            semantic_label = label_img[v, u]
            
            # 将该标签赋予所有对应的 3D 点索引
            # 这是一个高效的批量赋值操作
            point_labels[point_indices] = semantic_label

    # 5. 保存结果
    # S3DIS 原本就有 scalar_Label 属性，我们可以替换它
    # 或者为了可视化，将标签映射为颜色
    print(f"正在保存结果至: {output_path}")
    
    # 创建包含自定义属性的 PCD 结构（或直接导出为 PLY）
    # 这里我们演示如何将标签转化为颜色以便在 CloudCompare 中直接观察
    max_label = np.max(point_labels)
    # 为每个标签生成随机颜色颜色
    colors = np.random.uniform(0, 1, size=(max_label + 1, 3))
    colors[0] = [0.5, 0.5, 0.5] # 标签0设为灰色
    
    pcd.colors = o3d.utility.Vector3dVector(colors[point_labels])
    
    o3d.io.write_point_cloud(output_path, pcd)
    print("反向映射完成！")

if __name__ == "__main__":
    # 使用示例
    back_project_labels_to_3d(
        original_ply_path="Area_2.ply",
        label_img_path="S3DIS_Area2_Semantic_Labels.png", # 这是你大模型输出的图
        mapping_pkl_path="bev_mapping.pkl",
        output_path="Area_2_Semantic_Result.ply"
    )