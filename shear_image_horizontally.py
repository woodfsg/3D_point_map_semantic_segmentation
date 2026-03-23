import cv2
import numpy as np
import math

def shear_image_horizontally(image_path, output_path, angle_degrees):
    # 1. 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图像，请检查路径。")
        return

    h, w = img.shape[:2]

    # 2. 将角度转换为弧度并计算正切值
    angle_radians = math.radians(angle_degrees)
    tan_theta = math.tan(angle_radians)

    # 3. 计算倾斜后的新宽度，以防止图像边缘被裁剪
    # 底部的最大水平偏移量为 h * tan(θ)
    offset = int(h * abs(tan_theta))
    new_w = w + offset

    # 4. 构建仿射变换矩阵
    # [1, tan(θ), tx]
    # [0, 1,      ty]
    # 如果角度为正，底部向右倾斜；如果为负，需要给 x 加上初始偏移量 tx 防止左侧被裁剪
    if angle_degrees > 0:
        M = np.float32([
            [1, tan_theta, 0],
            [0, 1,         0]
        ])
    else:
        M = np.float32([
            [1, tan_theta, offset],
            [0, 1,         0]
        ])

    # 5. 应用仿射变换 (borderValue 为背景填充色，默认为黑色)
    sheared_img = cv2.warpAffine(img, M, (new_w, h), borderValue=(0, 0, 0))

    # 6. 保存结果
    cv2.imwrite(output_path, sheared_img)
    print(f"处理完成，图像已保存至: {output_path}")

# 使用示例
input_file = '../semantic_files/S3DIS_Area2_Semantic_GT.png' # 替换为您的图片路径
output_file = '../semantic_files/sheared_output.png'
# 水平倾斜 10 度
shear_image_horizontally(input_file, output_file, 10)