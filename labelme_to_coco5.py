import json
import os
import numpy as np
import random
import shutil
from collections import defaultdict


def create_tokens_positive(caption):
    """
    为caption创建tokens_positive
    将整个caption分词并创建token位置区间

    Args:
        caption: 文本描述

    Returns:
        tokens_positive: 列表，每个元素是包含两个整数的列表 [start, end]
    """
    if not caption:
        return []

    # 简单的分词：按空格分割
    words = caption.split()
    tokens_positive = []

    # 记录每个词的位置
    start_idx = 0
    for word in words:
        end_idx = start_idx + len(word)
        # 每个词作为一个token区间，必须是 [start, end] 格式
        tokens_positive.append([start_idx, end_idx])
        start_idx = end_idx + 1  # +1 for space

    return tokens_positive


def create_phrase_tokens_positive(caption, phrases):
    """
    为特定短语创建tokens_positive

    Args:
        caption: 完整的文本描述
        phrases: 短语列表，每个短语对应一个标注框

    Returns:
        每个短语对应的token区间列表，格式为 [[start, end], [start, end], ...]
    """
    if not caption or not phrases:
        return []

    tokens_positive = []
    for phrase in phrases:
        # 在caption中查找短语
        start_idx = caption.find(phrase)
        if start_idx != -1:
            end_idx = start_idx + len(phrase)
            tokens_positive.append([start_idx, end_idx])
        else:
            # 如果找不到短语，使用整个caption
            tokens_positive.append([0, len(caption)])

    return tokens_positive


def merge_labelme_to_coco_with_split(input_dir, output_dir=None, train_ratio=0.8, random_seed=42):
    """
    将多个labelme文件合并转换为COCO格式，并划分训练集和验证集（同时分开图片文件）

    Args:
        input_dir: 包含labelme文件和图片的输入目录
        output_dir: 输出目录（可选）
        train_ratio: 训练集比例（默认0.8）
        random_seed: 随机种子，确保结果可重复
    """
    if output_dir is None:
        output_dir = os.path.join(input_dir, "coco_output")

    # 创建输出目录结构
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    annotations_dir = os.path.join(output_dir, "annotations")

    for dir_path in [train_dir, val_dir, annotations_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # 设置随机种子
    random.seed(random_seed)

    # 获取所有json文件
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]

    if not json_files:
        print("错误：没有找到JSON文件")
        return None

    print(f"找到 {len(json_files)} 个JSON文件")

    # 初始化COCO数据结构
    coco_data = {
        "images": [],
        "categories": [],
        "annotations": []
    }

    # 收集所有不同的标签
    all_labels = set()
    file_data_list = []

    # 第一次遍历：收集所有标签
    for json_file in json_files:
        file_path = os.path.join(input_dir, json_file)
        with open(file_path, 'r', encoding='utf-8') as f:
            labelme_data = json.load(f)

        file_data_list.append((json_file, labelme_data))

        # 收集标签
        for shape in labelme_data.get("shapes", []):
            label = shape.get("label", "")
            # 提取主要类别（取第一个词作为类别）
            main_label = label.split('，')[0].split(' ')[0] if label else "unknown"
            all_labels.add(main_label)

    # 创建categories
    categories_dict = {}
    for idx, label in enumerate(sorted(all_labels), 1):
        categories_dict[label] = idx
        coco_data["categories"].append({
            "id": idx,
            "name": label
        })

    print(f"发现 {len(categories_dict)} 个类别: {list(categories_dict.keys())}")

    # 第二次遍历：处理annotations
    image_id = 1
    annotation_id = 1
    image_info_list = []  # 存储所有图片信息，用于后续划分

    for json_file, labelme_data in file_data_list:
        image_path = labelme_data.get("imagePath", "")

        # 收集该图片的所有标注的caption
        img_captions = []
        img_phrases = []

        # 先收集所有标注的caption
        for shape in labelme_data.get("shapes", []):
            label = shape.get("label", "")
            if label:
                img_captions.append(label)
                img_phrases.append(label)

        # 创建图片级别的caption
        img_caption = " ".join(img_captions) if img_captions else ""

        # 处理images信息 - 注意：这里没有tokens_positive字段，因为这是ModulatedDataset使用的格式
        image_info = {
            "id": image_id,
            "file_name": os.path.basename(image_path),
            "width": labelme_data.get("imageWidth", 0),
            "height": labelme_data.get("imageHeight", 0),
            "original_json": json_file,
            "original_image": image_path,
            "caption": img_caption  # ModulatedDataset期望在images中有caption字段
        }
        coco_data["images"].append(image_info)
        image_info_list.append(image_info)

        # 为每个标注创建对应的tokens_positive
        if img_phrases:
            phrase_tokens = create_phrase_tokens_positive(img_caption, img_phrases)
        else:
            phrase_tokens = []

        # 处理该图像的所有标注
        for shape_idx, shape in enumerate(labelme_data.get("shapes", [])):
            points = shape.get("points", [])
            label = shape.get("label", "")

            # 提取主要类别
            main_label = label.split('，')[0].split(' ')[0] if label else "unknown"
            cat_id = categories_dict.get(main_label, 1)

            # 计算bbox
            if points and len(points) >= 3:
                points_array = np.array(points)
                x_min = float(np.min(points_array[:, 0]))
                y_min = float(np.min(points_array[:, 1]))
                x_max = float(np.max(points_array[:, 0]))
                y_max = float(np.max(points_array[:, 1]))

                width = x_max - x_min
                height = y_max - y_min

                if width > 0 and height > 0:
                    bbox = [x_min, y_min, width, height]
                    area = width * height

                    # 获取当前标注对应的tokens_positive - 格式为 [[start, end], ...]
                    if shape_idx < len(phrase_tokens):
                        ann_tokens_positive = [phrase_tokens[shape_idx]]  # 每个标注可以有多个token区间，这里只有一个
                    else:
                        # 如果没有对应的短语，使用整个caption
                        ann_tokens_positive = [[0, len(img_caption)]] if img_caption else []

                    # 创建annotation - 注意：tokens_positive字段在annotations中
                    annotation = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": cat_id,
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0,
                        "caption": label,
                        "tokens_positive": ann_tokens_positive,  # 格式: [[start, end], ...]
                        "segmentation": [np.array(points).flatten().tolist()]
                    }

                    coco_data["annotations"].append(annotation)
                    annotation_id += 1

        image_id += 1

    print(f"总计处理了 {image_id - 1} 张图片，{annotation_id - 1} 个标注")

    # 确保所有必要字段都存在且格式正确
    for img in coco_data["images"]:
        if "caption" not in img:
            img["caption"] = ""

    for ann in coco_data["annotations"]:
        if "caption" not in ann:
            ann["caption"] = ""
        if "tokens_positive" not in ann:
            ann["tokens_positive"] = []
        # 确保tokens_positive的格式正确：[[start, end], ...]
        # 不需要额外转换，已经正确

    # 划分训练集和验证集
    split_data = split_coco_dataset_with_images(
        coco_data,
        image_info_list,
        input_dir,
        train_dir,
        val_dir,
        train_ratio,
        random_seed
    )

    # 保存训练集和验证集的annotation文件
    train_ann_path = os.path.join(annotations_dir, "instances_train.json")
    val_ann_path = os.path.join(annotations_dir, "instances_val.json")

    with open(train_ann_path, 'w', encoding='utf-8') as f:
        json.dump(split_data['train'], f, indent=2, ensure_ascii=False)

    with open(val_ann_path, 'w', encoding='utf-8') as f:
        json.dump(split_data['val'], f, indent=2, ensure_ascii=False)

    print(f"\n训练集标注文件已保存到: {train_ann_path}")
    print(f"验证集标注文件已保存到: {val_ann_path}")

    # 保存完整的COCO文件
    full_output_path = os.path.join(annotations_dir, "instances_full.json")
    with open(full_output_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)
    print(f"完整标注文件已保存到: {full_output_path}")

    # 打印统计信息
    print_split_statistics(split_data, train_dir, val_dir)

    return split_data


def split_coco_dataset_with_images(coco_data, image_info_list, source_dir, train_dir, val_dir, train_ratio=0.8,
                                   random_seed=42):
    """
    将COCO数据集划分为训练集和验证集，并复制图片文件

    Args:
        coco_data: COCO格式的数据字典
        image_info_list: 图片信息列表
        source_dir: 源图片目录
        train_dir: 训练集图片输出目录
        val_dir: 验证集图片输出目录
        train_ratio: 训练集比例
        random_seed: 随机种子

    Returns:
        包含train和val的字典
    """
    random.seed(random_seed)

    # 获取所有图像ID并打乱
    all_image_ids = [img['id'] for img in coco_data['images']]
    random.shuffle(all_image_ids)

    # 计算划分点
    split_point = int(len(all_image_ids) * train_ratio)
    train_image_ids = set(all_image_ids[:split_point])
    val_image_ids = set(all_image_ids[split_point:])

    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_image_ids)} 张图片")
    print(f"  验证集: {len(val_image_ids)} 张图片")

    # 创建训练集和验证集
    train_data = {
        "images": [],
        "categories": coco_data['categories'].copy(),
        "annotations": []
    }

    val_data = {
        "images": [],
        "categories": coco_data['categories'].copy(),
        "annotations": []
    }

    # 分配图片并复制文件
    copied_train = 0
    copied_val = 0
    missing_files = []

    for img_info in coco_data['images']:
        img_id = img_info['id']
        original_filename = img_info['file_name']

        # 查找源图片路径
        source_path = os.path.join(source_dir, original_filename)

        if img_id in train_image_ids:
            # 添加到训练集
            train_data['images'].append(img_info)

            # 复制图片到训练目录
            dest_path = os.path.join(train_dir, original_filename)
            try:
                if os.path.exists(source_path):
                    shutil.copy2(source_path, dest_path)
                    copied_train += 1
                else:
                    missing_files.append(original_filename)
            except Exception as e:
                print(f"  复制文件失败 {original_filename}: {str(e)}")
                missing_files.append(original_filename)

        elif img_id in val_image_ids:
            # 添加到验证集
            val_data['images'].append(img_info)

            # 复制图片到验证目录
            dest_path = os.path.join(val_dir, original_filename)
            try:
                if os.path.exists(source_path):
                    shutil.copy2(source_path, dest_path)
                    copied_val += 1
                else:
                    missing_files.append(original_filename)
            except Exception as e:
                print(f"  复制文件失败 {original_filename}: {str(e)}")
                missing_files.append(original_filename)

    # 分配标注
    for ann in coco_data['annotations']:
        if ann['image_id'] in train_image_ids:
            train_data['annotations'].append(ann)
        else:
            val_data['annotations'].append(ann)

    # 打印复制结果
    print(f"\n图片复制结果:")
    print(f"  训练集: 成功复制 {copied_train} 张图片")
    print(f"  验证集: 成功复制 {copied_val} 张图片")

    if missing_files:
        print(f"  警告: {len(missing_files)} 张图片未找到:")
        for f in missing_files[:5]:
            print(f"    - {f}")
        if len(missing_files) > 5:
            print(f"    ... 等{len(missing_files)}个文件")

    return {'train': train_data, 'val': val_data}


def print_split_statistics(split_data, train_dir, val_dir):
    """
    打印数据集划分的统计信息
    """
    print("\n" + "=" * 60)
    print("数据集统计信息:")
    print("=" * 60)

    for split_name, data in split_data.items():
        print(f"\n{split_name.upper()} 集:")
        print(f"  图片数量: {len(data['images'])}")
        print(f"  标注数量: {len(data['annotations'])}")

        # 验证必要字段
        img_has_caption = all('caption' in img for img in data['images'])
        print(f"  图片caption字段: {'✓ 存在' if img_has_caption else '✗ 缺失'}")

        ann_has_caption = all('caption' in ann for ann in data['annotations'])
        ann_has_tokens = all('tokens_positive' in ann for ann in data['annotations'])
        print(f"  标注caption字段: {'✓ 存在' if ann_has_caption else '✗ 缺失'}")
        print(f"  标注tokens_positive字段: {'✓ 存在' if ann_has_tokens else '✗ 缺失'}")

        # 检查tokens_positive的格式
        if data['annotations'] and ann_has_tokens:
            sample_tokens = data['annotations'][0].get('tokens_positive', [])
            print(f"  标注tokens_positive格式示例: {sample_tokens}")

        if split_name == 'train':
            print(f"  图片目录: {train_dir}")
        else:
            print(f"  图片目录: {val_dir}")

        # 统计每个类别的数量
        category_counts = defaultdict(int)
        for ann in data['annotations']:
            category_counts[ann['category_id']] += 1

        if category_counts:
            print("  类别分布:")
            cat_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
            for cat_id, count in sorted(category_counts.items()):
                cat_name = cat_id_to_name.get(cat_id, f"类别{cat_id}")
                print(f"    {cat_name}: {count} 个")


# 使用示例
if __name__ == "__main__":
    # 设置输入目录（包含所有labelme JSON文件和图片的文件夹）


    input_directory = "C:/SynMetaAI/projects/datasets/coco128-seg/coco128-seg/images/train2017"  # 替换为你的文件夹路径

    output_dir = "C:/SynMetaAI/projects/datasets/coco128-seg/coco128-seg/images/coco_dataset0227"


    # 合并、转换并划分数据集
    split_data = merge_labelme_to_coco_with_split(
        input_dir=input_directory,
        output_dir=output_dir,
        train_ratio=0.8,
        random_seed=42
    )


