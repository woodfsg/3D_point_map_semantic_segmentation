import cv2
import numpy as np

def extract_wall_skeleton(bev_img_path, output_path):
    """
    从全量 BEV 图像中提取墙体骨架
    """
    # 1. 读取图像
    # 注意：OpenCV 读取的顺序是 BGR
    img = cv2.imread(bev_img_path)
    if img is None:
        print("无法加载图像，请检查路径。")
        return

    # 2. 提取 G 通道 (索引为 1)
    # G 通道代表点云密度，墙体在此通道亮度最高
    g_channel = img[:, :, 1]

    # 3. 预处理：去噪
    # 使用高斯模糊平滑微小噪声，使墙体边缘更连续
    # blurred = cv2.GaussianBlur(g_channel, (3, 3), 0)
    blurred = g_channel

    # 4. 二值化分割
    # 使用 Otsu 自动阈值处理，或者设置固定高阈值（如 150-200）来提取高密度区域
    # _, binary_walls = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binary_walls = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

    # 5. 形态学处理：细化骨架
    # 先执行闭运算填充墙体内部的小空洞
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(binary_walls, cv2.MORPH_CLOSE, kernel)

    # 使用 OpenCV 的细化算法获取单像素宽度的骨架
    # 如果你的环境中没有 ximgproc，可以使用简单的腐蚀（Erosion）代替
    # try:
    #     skeleton = cv2.ximgproc.thinning(closed)
    # except AttributeError:
    #     # 备选方案：通过腐蚀变细
    #     skeleton = cv2.erode(closed, kernel, iterations=1)
    #     print("提示：未找到 ximgproc，使用腐蚀操作代替细化。")
    skeleton = closed

    # 6. 保存并显示结果
    cv2.imwrite(output_path, skeleton)
    print(f"墙体骨架已提取并保存至: {output_path}")

    # 可视化对比
    # 将骨架叠加回原图显示（以红色高亮显示）
    # overlay = img.copy()
    # overlay[skeleton == 255] = [0, 0, 255] # 在 BGR 中 [0,0,255] 是纯红
    # cv2.imwrite("Wall_Overlay_Comparison.png", overlay)

if __name__ == "__main__":
    # 使用你上传的图片路径
    extract_wall_skeleton("../bev_files/S3DIS_Area2_BEV_-50_50.png", "../skeleton_files/Wall_Skeleton.png")