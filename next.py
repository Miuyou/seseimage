import os
import shutil
from PIL import Image
import torch
from transformers import AutoProcessor, CLIPModel
from deepdanbooru_onnx import DeepDanbooru, process_image
from tqdm import tqdm
import traceback
danbooru = DeepDanbooru()
# ==============================================================================
# ### 第 0 部分: 全局配置 ###
# ==============================================================================
print("--- 正在初始化配置 ---")


# --- 筛选阈值配置 ---
# CLIP模型判断为“真人”的置信度阈值 (0.0 - 1.0, 越高越严格)
# DeepDanbooru 判断为 NSFW 的分数阈值 (0.0 - 1.0)
NSFW_THRESHOLD = 0.4
# 质量筛选：图片最小尺寸
MIN_WIDTH = 512
MIN_HEIGHT = 512
MIN_TOTAL_PIXELS = 0 # 最小总像素，0表示不启用

# --- 其他配置 ---
SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

# ==============================================================================
# ### 第 1 部分: 初始化、加载模型与准备文件夹 ###
# ==============================================================================

# --- 1.1 模型加载 ---
print("\n--- 正在加载模型 (这可能需要一些时间) ---")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用的设备: {DEVICE}")

# 加载 CLIP 模型
clip_model = None
clip_processor = None
try:
    print("正在加载 CLIP 模型...")
    CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
    clip_processor = AutoProcessor.from_pretrained(CLIP_MODEL_ID)
    # 使用 use_safetensors=True 来避免 torch.load 的安全漏洞问题
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID, use_safetensors=True).to(DEVICE)
    print(" CLIP 模型加载完成！")
except Exception as e:
    print(f" 加载 CLIP 模型失败！错误: {e}")
    traceback.print_exc()
    # 如果核心模型加载失败，则无法继续，退出程序
    exit()

# 加载 DeepDanbooru 模型
danbooru = None
try:
    print("正在加载 DeepDanbooru ONNX 模型...")
    danbooru = DeepDanbooru()
    print(" DeepDanbooru 模型加载完成！")
except Exception as e:
    print(f" 加载 DeepDanbooru 模型失败！错误: {e}")
    traceback.print_exc()
    exit()

import os
import shutil
from PIL import Image
import torch
from tqdm import tqdm
# 假设你的模型和处理器已经加载好了，例如：
# from transformers import CLIPProcessor, CLIPModel
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# --- 路径和参数配置 (请根据你的实际情况修改) ---
SOURCE_FOLDER = r'folder'
DESTINATION_BASE_FOLDER = r'folder'

# --- 新增：批处理大小 ---
# BATCH_SIZE可以根据你的GPU显存进行调整。32或64是常见的起始值。
# 如果遇到 "CUDA out of memory" 错误，请调低此数值。
BATCH_SIZE = 32

# --- 图片质量配置 ---
MIN_WIDTH = 512
MIN_HEIGHT = 512
MIN_TOTAL_PIXELS = MIN_WIDTH * MIN_HEIGHT
SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')


# ==============================================================================
# ### 准备工作 (与原代码相同) ###
# ==============================================================================

# --- 1.2 文件夹准备 ---
print("\n--- 正在准备文件夹结构 ---")
real_folder = os.path.join(DESTINATION_BASE_FOLDER, '真人')
anime_folder = os.path.join(DESTINATION_BASE_FOLDER, '动漫')
anime_normal_folder = os.path.join(anime_folder, 'normal')
anime_nsfw_folder = os.path.join(anime_folder, 'nsfw')
low_quality_folder = os.path.join(DESTINATION_BASE_FOLDER, 'low_quality')
error_folder = os.path.join(DESTINATION_BASE_FOLDER, 'error')
more_folder = os.path.join(DESTINATION_BASE_FOLDER, '杂项')
TEMP_ANIME_FOLDER = os.path.join(DESTINATION_BASE_FOLDER, '_temp_anime_processing')

all_folders_to_create = [
    real_folder, anime_normal_folder, anime_nsfw_folder,
    low_quality_folder, error_folder, TEMP_ANIME_FOLDER,more_folder
]
for folder in all_folders_to_create:
    os.makedirs(folder, exist_ok=True)
print(" 文件夹准备完成！")


# ==============================================================================
# ### 第 2 部分: 使用 CLIP 进行一级分类 (批处理优化版) ###
# ==============================================================================
print("\n--- (阶段 1/2) 开始使用 CLIP 进行一级分类 (批处理优化) ---")

if not os.path.isdir(SOURCE_FOLDER):
    print(f" 致命错误：源文件夹 '{SOURCE_FOLDER}' 不存在！请检查路径。")
else:
    image_files = [f for f in os.listdir(SOURCE_FOLDER) if os.path.isfile(os.path.join(SOURCE_FOLDER, f)) and f.lower().endswith(SUPPORTED_FORMATS)]
    print(f"在源文件夹中发现 {len(image_files)} 个待处理的图片文件。")

    clip_text_prompts = [
        "a high-resolution photograph of a real person, a real life scene, realistic photo, dslr quality",
        "anime style drawing, illustration, digital art, character art, manga, cartoon, 2d art"
    ]

    # --- 批处理循环 ---
    # 我们将 image_files 列表按 BATCH_SIZE 分块处理
    for i in tqdm(range(0, len(image_files), BATCH_SIZE), desc="CLIP 分类批处理"):
        # 获取当前批次的
        batch_filenames = image_files[i:i + BATCH_SIZE]
        
        # 用于存储当前批次中通过了质量检查的图片和其原始路径
        batch_images_for_clip = []
        batch_source_paths = []

        # 1. 预处理当前批次：加载、检查质量
        for filename in batch_filenames:
            image_path = os.path.join(SOURCE_FOLDER, filename)
            destination_path=None
            try:
                with Image.open(image_path) as img:
                    # 质量检查
                    width, height = img.size
                    
                    if width < MIN_WIDTH or height < MIN_HEIGHT :
                        destination_path = os.path.join(low_quality_folder, filename)
                        
                    else:
                        img_rgb = img.convert("RGB")
                        batch_images_for_clip.append(img_rgb)
                        batch_source_paths.append(image_path)
            except Exception as e:
                shutil.move(image_path, error_folder)
            if destination_path:
                shutil.move(image_path, destination_path)

        
        # 2. 对整个批次进行CLIP分类 (如果批次中有有效图片)
        if not batch_images_for_clip:
            continue # 如果当前批次所有图片都是低质量或错误，则跳过

        #try:
        with torch.no_grad():
            # 一次性处理整个批次的图片
            inputs = clip_processor(
                text=clip_text_prompts, 
                images=batch_images_for_clip, 
                return_tensors="pt", 
                padding=True
            ).to(DEVICE)
            
            outputs = clip_model(**inputs)
            # `probs` 现在是一个张量，每行对应一批次中的一张图片
            probs = outputs.logits_per_image.softmax(dim=1)

        # 3. 根据批处理结果，移动文件
        for idx, source_path in enumerate(batch_source_paths):
            prob_real = probs[idx][0].item()
            filename = os.path.basename(source_path)
            
            if prob_real >= 0.5:
                destination_path = os.path.join(real_folder, filename)
            else:
                destination_path = os.path.join(TEMP_ANIME_FOLDER, filename) 
            shutil.move(source_path, destination_path)

        #except Exception as e:
        #    print(f"处理批次 {i//BATCH_SIZE + 1} 时发生模型错误: {e}")
        #    # 如果模型在整个批次上失败，尝试将这个批次的文件移到错误文件夹
        #    for source_path in batch_source_paths:
        #        try:
        #            shutil.move(source_path, os.path.join(error_folder, os.path.basename(source_path)))
        #        except Exception as move_e:
        #            print(f"移动文件 {os.path.basename(source_path)} 到 'error' 文件夹失败: {move_e}")

    print(" CLIP 一级分类完成！")

    print("\n--- (阶段 2/2) 开始使用 DeepDanbooru 对动漫图片进行二级分类 ---")
nsfw_keys = [
    # 身体部位
    'breasts',       # 胸部
    'cleavage',      # 乳沟
    'ass',           # 臀部
    'navel',         # 肚脐
    'nipples',       # 乳头
    'pussy',         # 阴部
    'armpits',       # 腋窝
    'thighs',        # 大腿
    'groin',         # 腹股沟
    'sideboob',      # 侧乳
    'female_pubic_hair',  # 女性阴毛
    'pubic_hair',    # 阴毛
    'large_breasts', # 大胸
    'butt_crack',    # 臀缝
    
    # 姿势/状态
    'spread_legs',   # 张开双腿
    'bent_over',     # 弯腰
    'from_behind',   # 从后面
    'on_back',       # 仰卧
    'on_stomach',    # 俯卧
    'lying',         # 躺着
    'open_clothes',  # 敞开的衣服
    'strap_slip',    # 肩带滑落
    
    # 衣物/物品
    'underwear',     # 内衣
    'panties',       # 内裤
    'bra',           # 胸罩
    'pasties',       # 乳贴
    'torn_clothes',  # 破损的衣服
    'condom_wrapper',# 避孕套包装
    
    # 行为/状态
    'lactation',     # 哺乳
    'prostitution',  # 卖淫
    'uncensored',    # 无码
    'nude',          # 裸体
    'nude_cover',    # 裸体遮盖
    
    # 评级标签
    'rating:questionable',  # 可疑评级
    'rating:nsfw',          # NSFW评级
    
    # 可继续添加的标签...
]
if not os.path.isdir(TEMP_ANIME_FOLDER):
    print(f"⚠️ 警告：临时文件夹 '{TEMP_ANIME_FOLDER}' 未找到，可能没有需要二次处理的图片。")
else:
    anime_files = [f for f in os.listdir(TEMP_ANIME_FOLDER) if os.path.isfile(os.path.join(TEMP_ANIME_FOLDER, f)) and f.lower().endswith(SUPPORTED_FORMATS)]
    if not anime_files:
        print(" 临时文件夹中没有需要处理的动漫图片。")
    else:
        print(f"发现 {len(anime_files)} 个待处理的动漫图片。")
        for filename in tqdm(anime_files, desc="DeepDanbooru 分类进度"):
            image_path = os.path.join(TEMP_ANIME_FOLDER, filename)
            img=[image_path]
            destination_path = None
            results = list(danbooru(img))
            if results[0].get('1girl',0.0)+results[0].get('2girls',0.0)<0.5:
                shutil.move(image_path,more_folder)
                continue
            try:
                nsfw_score = max(results[0].get(key, 0.0) for key in nsfw_keys)
                #print(nsfw_score)
                if nsfw_score > 0.6:
                    destination_path=anime_nsfw_folder
                else:
                    destination_path=anime_normal_folder
                shutil.move(image_path, destination_path)

            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e} ")
                try:
                    shutil.move(image_path,error_folder)
                except Exception as move_e:
                    print(f"移动文件 {filename} 到 'error' 文件夹也失败了: {move_e}")
    
    # 清理临时文件夹
    try:
        if os.path.exists(TEMP_ANIME_FOLDER) and not os.listdir(TEMP_ANIME_FOLDER):
            print("清理空的临时文件夹...")
            os.rmdir(TEMP_ANIME_FOLDER)
    except OSError as e:
        print(f"清理临时文件夹失败: {e}")

print("\n 所有处理流程完成！")