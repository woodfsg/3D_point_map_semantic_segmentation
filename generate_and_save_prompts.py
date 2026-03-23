import os

# 您提供的类别与 RGB 颜色映射表
class_map = {
    1: [255, 0, 0],      # auditorium
    2: [0, 255, 0],      # conferenceRoom
    3: [0, 0, 255],      # hallway
    4: [255, 255, 0],    # office
    5: [0, 255, 255],    # storage
    6: [255, 0, 255]     # WC
}

# 定义 RGB 到英文颜色单词的辅助映射
rgb_to_color_name = {
    (255, 0, 0): "RED",
    (0, 255, 0): "GREEN",
    (0, 0, 255): "BLUE",
    (255, 255, 0): "YELLOW",
    (0, 255, 255): "CYAN",
    (255, 0, 255): "MAGENTA"
}

# 定义类别 ID 到类别名称的辅助映射
class_id_to_name = {
    1: 'auditorium',
    2: 'conferenceRoom',
    3: 'hallway',
    4: 'office',
    5: 'storage',
    6: 'WC'
}

# 定义 Prompt 模板
prompt_template = """You are provided with two images: a Bird's-Eye View (BEV) projection of an indoor 3D point cloud (Image 1), and a semantic ground truth map (Image 2).

Your task is to generate a precise black-and-white mask image for the '{class_name}' areas based on Image 1.

Instructions:

1. Look at Image 2 (the semantic map) and identify the areas colored in solid {color_name}. These represent the {class_plural}.
2. Locate these corresponding {class_name} structures in Image 1 (the BEV image). 
3. Ignore the rotation, tilt, or scale differences between Image 1 and Image 2. Rely on the visual architectural features in Image 1 to determine the exact boundaries.
4. Generate a flat, 2D black-and-white mask that is perfectly aligned with the dimensions and perspective of Image 1 (the BEV image). The {class_name} areas should be solid white, and all other areas (including the background) should be solid black. Do not add any stylistic elements or textures."""

def generate_and_save_prompts(output_folder="prompts_output"):
    """生成并保存所有类别的 Prompt 到指定文件夹"""
    
    # 如果文件夹不存在，则创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 已创建输出文件夹: {output_folder}")
    
    for class_id, rgb_list in class_map.items():
        class_name = class_id_to_name[class_id]
        rgb_tuple = tuple(rgb_list)
        color_name = rgb_to_color_name.get(rgb_tuple, "UNKNOWN_COLOR")
        
        # 处理复数形式
        class_plural = class_name + 's' if class_name != 'WC' else 'WCs'
        
        # 格式化模板
        final_prompt = prompt_template.format(
            class_name=class_name,
            color_name=color_name,
            class_plural=class_plural
        )
        
        # 定义文件名并保存
        file_name = f"prompt_{class_name}.txt"
        file_path = os.path.join(output_folder, file_name)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(final_prompt)
            
        print(f"✅ 成功保存: {file_name}")

if __name__ == "__main__":
    output_folder="../prompt_files"
    generate_and_save_prompts(output_folder)
    print("\n🎉 所有 Prompt 已生成完毕！")