import cv2
import numpy as np
import os

def generate_2d_segmentation_map(bev_image_path, mask_info_list, output_path):
    """
    根据 BEV 投影图和多个语义掩码图生成 2D 语义分割结果图。
    
    参数:
        bev_image_path (str): BEV 投影图的路径 (用于获取基准尺寸)。
        mask_info_list (list of dict): 包含掩码路径和对应类别 ID 的字典列表。
                                       例如: [{'path': 'mask_auditorium.jpg', 'class_id': 1}, ...]
        output_path (str): 分割结果图的保存路径 (强烈建议使用 .png)。
    """
    
    # 1. 前期准备与定义
    print(f"正在读取 BEV 投影图: {bev_image_path}")
    bev_image = cv2.imread(bev_image_path)
    if bev_image is None:
        raise FileNotFoundError(f"无法读取 BEV 投影图，请检查路径: {bev_image_path}")
    
    # 获取基准尺寸 (高 H, 宽 W)
    base_h, base_w = bev_image.shape[:2]
    print(f"基准尺寸获取成功: 宽 {base_w}, 高 {base_h}")

    # 2. 初始化结果画布
    # 创建一个单通道的二维数组，初始值为 0 (背景)，数据类型使用 uint8 (0-255)
    segmentation_map = np.zeros((base_h, base_w), dtype=np.uint8)

    # 3 & 4. 掩码图的循环处理与冲突处理 (后写入者覆盖)
    for mask_info in mask_info_list:
        mask_path = mask_info['path']
        class_id = mask_info['class_id']
        
        if not os.path.exists(mask_path):
            print(f"警告: 找不到掩码文件 {mask_path}，已跳过。")
            continue
            
        print(f"正在处理掩码图: {mask_path} -> 映射为类别 ID: {class_id}")
        
        # 以灰度模式读取掩码图
        mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 强制缩放 (Resize) 到基准尺寸，必须使用最近邻插值 (INTER_NEAREST)
        resized_mask = cv2.resize(mask_image, (base_w, base_h), interpolation=cv2.INTER_NEAREST)
        
        # 二值化处理: 设定阈值 128，大于 128 的像素视为目标区域 (True)
        # 这样可以过滤掉 JPEG 压缩或模型生成时边缘的灰色噪点
        _, binary_mask = cv2.threshold(resized_mask, 128, 255, cv2.THRESH_BINARY)
        bool_mask = binary_mask > 0
        
        # 映射到画布: 将掩码区域内的像素值更新为当前的类别 ID
        segmentation_map[bool_mask] = class_id

    # 5. 结果保存
    # 确保保存为 PNG 格式，防止数据被有损压缩破坏
    if not output_path.lower().endswith('.png'):
        print("警告: 建议将输出文件扩展名改为 .png 以保证标签索引不被破坏。")
        
    cv2.imwrite(output_path, segmentation_map)
    print(f"✅ 2D 语义分割结果图已成功保存至: {output_path}")

# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    # 替换为您本地的实际文件路径
    BEV_IMAGE = "../bev_files/S3DIS_Area2_BEV_-50_50.png" 
    
    # 模拟您拥有的掩码图列表 (路径 + 对应的类别 ID)
    # 假设：背景=0, 礼堂=1, 走廊=2, 办公室=3
    MASKS_TO_PROCESS = [
        # 您上传的礼堂掩码图
        {'path': '../mask_files/mask.png', 'class_id': 1}, 
        # 假设您后续生成的其他掩码图
        # {'path': 'corridor_mask.jpg', 'class_id': 2},
        # {'path': 'office_mask.jpg', 'class_id': 3},
    ]
    
    OUTPUT_MAP = "../2d_segmentation_files/2d_semantic_segmentation_map2.png"
    
    generate_2d_segmentation_map(BEV_IMAGE, MASKS_TO_PROCESS, OUTPUT_MAP)