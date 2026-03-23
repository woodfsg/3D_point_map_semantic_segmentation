# 3D点云区域级语义分割流程
3D点云的文件头：

ply
format binary_little_endian 1.0
element vertex 37311701
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header

1. 运行generate_bev_with_mapping.py生成3D点云的bev投影图和投影映射文件
2. 使用多模态大模型输入bev投影图和语义图生成语义掩码图
3. 检查生成的掩码图是正确的概率，高于一定的阈值才允许进入下一步。
4. 运行generate_2d_segmentation_map.py使用投影图和语义掩码图生成 2D 语义分割结果图（像素值代表类别索引
5. 使用2D语义分割结果图和投影映射文件生成带有语义标签的3D点云保存路径