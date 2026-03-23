import cv2
import numpy as np

def calculate_overlap(img1_path, img2_path):
    # 1. 读取图片
    # 读取第一张图（彩色模式，OpenCV 默认读取为 BGR 格式）
    img1 = cv2.imread(img1_path)
    # 读取第二张图（灰度模式，因为只有黑白掩码）
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("图片读取失败，请检查路径。")
        return

    # 2. 统一尺寸
    # 获取第一张图的高和宽
    h1, w1 = img1.shape[:2]
    # 将第二张图缩放到与第一张图相同的尺寸
    # 注意：对于掩码图像，必须使用 INTER_NEAREST (最近邻插值)，防止边缘出现灰色过渡像素
    img2_resized = cv2.resize(img2, (w1, h1), interpolation=cv2.INTER_NEAREST)

    # 3. 提取目标区域掩码 (Mask)
    # 提取第一张图的红色区域
    # OpenCV 是 BGR 格式，纯红色的 BGR 值为 (0, 0, 255)
    # 为了增加鲁棒性，我们使用一个颜色范围来提取红色
    lower_red = np.array([0, 0, 200])   # B, G, R 下限
    upper_red = np.array([50, 50, 255]) # B, G, R 上限
    mask1_red = cv2.inRange(img1, lower_red, upper_red)
    mask1_bool = mask1_red > 0 # 转换为布尔型掩码

    # 提取第二张图的白色区域
    # 二值化处理，将大于 127 的像素视为白色
    _, mask2_white = cv2.threshold(img2_resized, 127, 255, cv2.THRESH_BINARY)
    mask2_bool = mask2_white > 0 # 转换为布尔型掩码

    # 4. 计算重叠比例
    # 计算交集 (Intersection)：两个掩码中都为 True 的区域
    intersection = np.logical_and(mask1_bool, mask2_bool)
    # 计算并集 (Union)：两个掩码中至少有一个为 True 的区域
    union = np.logical_or(mask1_bool, mask2_bool)

    # 计算像素面积
    area_intersect = np.sum(intersection)
    area_union = np.sum(union)
    area_red = np.sum(mask1_bool)
    area_white = np.sum(mask2_bool)

    # 5. 计算各项评估指标
    # 交并比 IoU (Intersection over Union)，这是计算机视觉中最常用的重叠度量
    iou = area_intersect / area_union if area_union > 0 else 0
    # 红色区域被覆盖的比例 (Recall-like)
    overlap_red = area_intersect / area_red if area_red > 0 else 0
    # 白色掩码落在红色区域内的比例 (Precision-like)
    overlap_white = area_intersect / area_white if area_white > 0 else 0

    print("=== 重叠比例计算结果 ===")
    print(f"交并比 (IoU): {iou:.4f} ({iou * 100:.2f}%)")
    print(f"第一张图的红色区域中，有 {overlap_red * 100:.2f}% 在第二张图被标为白色。")
    print(f"第二张图的白色区域中，有 {overlap_white * 100:.2f}% 准确落在了红色区域内。")

    return iou, overlap_red, overlap_white

# 使用示例（请将路径替换为你图片的实际路径）
iou, cover_red, cover_white = calculate_overlap('../semantic_files/S3DIS_Area2_Semantic_GT_turn90.png', '../mask_files/mask.png')