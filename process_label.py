import os
import numpy as np
from pathlib import Path
import uuid
from xml.dom import minidom
import xml.etree.ElementTree as ET

def export_color_scale_xml(class_map, xml_filename):
    root = ET.Element("CloudCompare")
    cs = ET.SubElement(root, "ColorScale", version="1")
    
    # 1. 设置属性
    props = ET.SubElement(cs, "Properties")
    ET.SubElement(props, "name").text = "S3DIS_6Classes"
    ET.SubElement(props, "uuid").text = f"{{{uuid.uuid4()}}}"
    ET.SubElement(props, "absolute").text = "1"
    
    ids = [info['id'] for info in class_map.values()]
    min_id, max_id = min(ids), max(ids)
    range_val = max_id - min_id
    
    ET.SubElement(props, "minValue").text = str(min_id)
    ET.SubElement(props, "range").text = str(range_val)
    
    # 2. 写入数据
    data = ET.SubElement(cs, "Data")
    sorted_classes = sorted(class_map.items(), key=lambda x: x[1]['id'])
    
    for name, info in sorted_classes:
        label_id = info['id']
        color = info['color']
        pos = (label_id - min_id) / range_val if range_val > 0 else 0
        
        # 写入 Step
        ET.SubElement(data, "step", 
                      r=str(color[0]), g=str(color[1]), b=str(color[2]), 
                      pos=f"{pos:.4f}")
        # 写入 Label
        ET.SubElement(data, "label", val=str(label_id))

    # 格式化并保存
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
    with open(xml_filename, "w", encoding="utf-8") as f:
        f.write(xml_str)

def export_s3dis_area_to_ply_binary(dataset_root, area_name, output_filename):
    class_map = {
        'auditorium':     {'id': 1, 'color': [255, 0, 0]},      
        'conferenceRoom': {'id': 2, 'color': [0, 255, 0]},      
        'hallway':        {'id': 3, 'color': [0, 0, 255]},      
        'office':         {'id': 4, 'color': [255, 255, 0]},    
        'storage':        {'id': 5, 'color': [0, 255, 255]},    
        'WC':             {'id': 6, 'color': [255, 0, 255]}     
    }
    
    area_path = Path(dataset_root) / area_name
    all_points = []
    
    if not area_path.exists():
        print(f"错误：找不到路径 {area_path}")
        return

    print(f"开始提取 {area_name} 数据并保留原始色彩...")

    for room_dir in area_path.iterdir():
        if not room_dir.is_dir(): continue
            
        room_name = room_dir.name
        matched_class = next((cls for cls in class_map if cls.lower() in room_name.lower()), None)
        
        if not matched_class: continue
            
        print(f"  正在处理: {room_name}")
        class_info = class_map[matched_class]
        anno_path = room_dir / 'Annotations'
        
        if not anno_path.exists(): continue
            
        for obj_file in anno_path.glob('*.txt'):
            if 'ceiling' in obj_file.name.lower(): continue
            try:
                # S3DIS 格式通常为: X Y Z R G B
                data = np.loadtxt(obj_file)
                if data.size == 0: continue
                if len(data.shape) == 1: data = data.reshape(1, -1)
                
                xyz = data[:, :3]
                # --- 修改点 1：保留点云原本的色彩 ---
                rgb = data[:, 3:6].astype(np.uint8) 
                labels = np.full((xyz.shape[0], 1), class_info['id'], dtype=np.int32)
                
                # 暂时存入 list，注意合并时类型统一为 float64 后面写入再转换
                all_points.append(np.hstack((xyz, rgb, labels)))
            except Exception as e:
                print(f"    读取文件 {obj_file.name} 出错: {e}")

    if not all_points:
        print("未提取到有效数据。")
        return

    print("正在以 Binary 格式写入文件...")
    final_data = np.vstack(all_points)
    num_points = len(final_data)
    
    # --- 修改点 2：构造二进制 PLY 文件头 ---
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {num_points}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property float scalar_Label\n"
        "end_header\n"
    )

    # 使用结构化数组来确保二进制排列紧凑且类型正确
    # f4: float32, u1: uint8, i4: int32
    dt = np.dtype([
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('scalar_Label', 'f4')
    ])
    
    binary_data = np.empty(num_points, dtype=dt)
    binary_data['x'] = final_data[:, 0]
    binary_data['y'] = final_data[:, 1]
    binary_data['z'] = final_data[:, 2]
    binary_data['red'] = final_data[:, 3]
    binary_data['green'] = final_data[:, 4]
    binary_data['blue'] = final_data[:, 5]
    binary_data['scalar_Label'] = final_data[:, 6].astype(np.float32)

    with open(output_filename, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(binary_data.tobytes())

    # --- 修改点 3：生成配套的 XML 颜色表 ---
    xml_output = str(Path(output_filename).with_suffix('.xml'))
    export_color_scale_xml(class_map, xml_output)

    print(f"完成！\n点云文件: {output_filename}\n颜色配置: {xml_output}")

# --- 配置路径 ---
DATASET_ROOT = '../Stanford3dDataset_v1.2'
OUTPUT_FILE = './ply_files/S3DIS_Area2_6Classes.ply'

os.makedirs('./ply_files', exist_ok=True)
export_s3dis_area_to_ply_binary(DATASET_ROOT, 'Area_2', OUTPUT_FILE)