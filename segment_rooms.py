import cv2
import numpy as np
import os
import argparse

def segment_rooms_optimal(input_path, output_path, threshold_val=120):
    """
    基于分水岭算法的房间实例分割
    :param input_path: 输入BEV图像路径
    :param output_path: 输出结果保存路径
    :param threshold_val: 墙体强化的二值化阈值，根据点云密度感官调整
    """
    # 1. 安全读取
    if not os.path.exists(input_path):
        print(f"[-] 错误: 找不到输入文件 -> {os.path.abspath(input_path)}")
        return

    print(f"[*] 正在处理: {input_path}")
    img = cv2.imread(input_path)
    if img is None:
        print(f"[-] 错误: OpenCV 无法加载图像，请检查文件完整性。")
        return

    g_density = img[:, :, 1] # 提取G通道（点云密度） [cite: 1]

    # 2. 墙体强化与去噪
    denoised = cv2.medianBlur(g_density, 5)
    _, walls = cv2.threshold(denoised, threshold_val, 255, cv2.THRESH_BINARY)

    # 3. 寻找房间种子 (距离变换)
    # 反转图像：让房间内部变成高亮，墙壁变成黑色
    dist_not_walls = cv2.bitwise_not(walls)
    dist_transform = cv2.distanceTransform(dist_not_walls, cv2.DIST_L2, 5)
    
    # 提取种子点：局部距离最大的区域（通常是房间中心）
    _, seeds = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    seeds = np.uint8(seeds)

    # 4. 标记连通域
    num_labels, markers = cv2.connectedComponents(seeds)
    # 分水岭算法要求背景标签为0，我们将现有标记加1，墙壁（高密度区）设为0
    markers = markers + 1
    markers[walls == 255] = 0

    # 5. 执行分水岭分割 [cite: 1]
    final_markers = cv2.watershed(img, markers)
    
    # 6. 生成可视化结果
    result_img = np.zeros_like(img)
    # 从标签2开始分配颜色（1是背景）
    for i in range(2, num_labels + 2):
        random_color = [np.random.randint(0, 255) for _ in range(3)]
        result_img[final_markers == i] = random_color

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[*] 已创建输出目录: {output_dir}")

    cv2.imwrite(output_path, result_img)
    print(f"[+] 房间分割预选方案已保存至: {output_path}")

if __name__ == "__main__":
    # 使用 argparse 方便修改路径
    parser = argparse.ArgumentParser(description="S3DIS BEV 房间实例分割脚本")
    
    # 添加参数：默认值设为你目前使用的文件名
    parser.add_argument("--input", "-i", type=str, default="../bev_files/S3DIS_Area2_BEV.png", help="输入BEV图像的路径")
    parser.add_argument("--output", "-o", type=str, default="../segment_files/Room_Segmentation_Proposals.png", help="输出结果保存的路径")
    parser.add_argument("--thresh", "-t", type=int, default=120, help="墙体检测阈值 (0-255)")

    args = parser.parse_args()

    # 执行主程序
    segment_rooms_optimal(args.input, args.output, args.thresh)