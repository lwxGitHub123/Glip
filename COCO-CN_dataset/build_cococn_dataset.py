#!/usr/bin/env python3
"""
构建符合GLIP要求的COCO-CN数据集（标准COCO格式）
使用独立的训练集和验证集标注文件
"""

import json
import os
from collections import defaultdict
from tqdm import tqdm

# ====================================================
# 只需修改这里的路径变量
# ====================================================

# COCO训练集标注文件
COCO_TRAIN_ANN = "C:/SynMetaAI/projects/clip/GLIP/coco/coco2014/annotations/instances_train2014.json"
# COCO验证集标注文件
COCO_VAL_ANN = "C:/SynMetaAI/projects/clip/GLIP/coco/coco2014/annotations/instances_val2014.json"
# COCO-CN标签文件（训练集和验证集共用一个标签文件）
COCOCN_TAGS_FILE = "C:/SynMetaAI/projects/clip/GLIP/COCO-CN_dataset/coco-cn-version1805v1.1/imageid.human-written-tags.txt"
# 训练集划分文件
TRAIN_SPLIT_FILE = "C:/SynMetaAI/projects/clip/GLIP/COCO-CN_dataset/coco-cn-version1805v1.1/coco-cn_train.txt"
# 验证集划分文件
VAL_SPLIT_FILE = "C:/SynMetaAI/projects/clip/GLIP/COCO-CN_dataset/coco-cn-version1805v1.1/coco-cn_val.txt"
# 输出文件
OUTPUT_TRAIN = "./coco_cn_train.json"
OUTPUT_VAL = "./coco_cn_val.json"

# 参数设置
MIN_AREA = 100  # 最小边界框面积
MAX_BOXES_PER_IMAGE = 30  # 每张图像最大边界框数


# ====================================================
# 以下代码不需要修改
# ====================================================

def load_coco_annotations(coco_ann_file, dataset_name):
    """
    加载COCO标注文件，获取图像尺寸等信息
    """
    print(f"正在加载{dataset_name}COCO标注: {coco_ann_file}")
    with open(coco_ann_file, 'r') as f:
        coco_data = json.load(f)

    # 创建图像ID到信息的映射
    img_id_to_info = {}
    for img in coco_data['images']:
        img_id_to_info[img['id']] = {
            'file_name': img['file_name'],
            'width': img['width'],
            'height': img['height']
        }

    # 创建类别ID到名称的映射
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # 创建图像ID到边界框的映射
    img_id_to_boxes = defaultdict(list)
    for ann in tqdm(coco_data['annotations'], desc=f"处理{dataset_name}COCO标注"):
        img_id = ann['image_id']
        bbox = ann['bbox']  # [x, y, width, height]
        cat_name = cat_id_to_name[ann['category_id']]

        img_id_to_boxes[img_id].append({
            'bbox': bbox,
            'category_id': ann['category_id'],
            'category_name': cat_name,
            'area': ann['area'],
            'segmentation': ann.get('segmentation', [])
        })

    print(f"{dataset_name}COCO图像总数: {len(img_id_to_info)}")
    print(f"{dataset_name}有边界框的图像数: {len(img_id_to_boxes)}")

    return img_id_to_info, img_id_to_boxes


def load_cococn_tags(tags_file):
    """
    加载COCO-CN的中文标签文件
    格式: image_id 标签1 标签2 ...
    """
    print(f"正在加载COCO-CN标签: {tags_file}")
    img_id_to_tags = {}

    with open(tags_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="处理标签"):
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            img_id = parts[0]
            # 提取数字ID
            numeric_id = int(img_id.split('_')[-1])
            tags = parts[1:]
            img_id_to_tags[numeric_id] = {
                'cococn_id': img_id,
                'tags': tags
            }

    print(f"COCO-CN标签总数: {len(img_id_to_tags)}")
    return img_id_to_tags


def load_image_split(split_file, split_name):
    """
    加载图像划分文件（train/val/test），返回数字ID列表
    """
    print(f"正在加载{split_name}划分: {split_file}")
    img_ids = []
    with open(split_file, 'r') as f:
        for line in f:
            cococn_id = line.strip()
            numeric_id = int(cococn_id.split('_')[-1])
            img_ids.append({
                'cococn_id': cococn_id,
                'numeric_id': numeric_id
            })
    print(f"{split_name}图像数: {len(img_ids)}")
    return img_ids


def create_coco_format_dataset(split_ids, split_name, img_id_to_info, img_id_to_boxes,
                               img_id_to_tags, output_file):
    """
    创建COCO格式的数据集
    """
    print(f"\n开始构建{split_name}集...")

    # 收集所有类别
    all_categories = set()
    category_name_to_id = {}

    # 首先收集所有用到的类别
    for item in split_ids:
        numeric_id = item['numeric_id']
        if numeric_id in img_id_to_boxes:
            for box in img_id_to_boxes[numeric_id]:
                cat_name = box['category_name']
                all_categories.add(cat_name)

    # 创建类别映射（使用英文类别名）
    categories = []
    for i, cat_name in enumerate(sorted(all_categories), 1):
        category_name_to_id[cat_name] = i
        categories.append({
            'id': i,
            'name': cat_name,
            'supercategory': 'object'
        })

    print(f"类别数: {len(categories)}")

    # 构建COCO格式
    coco_data = {
        'images': [],
        'annotations': [],
        'categories': categories
    }

    image_id_set = set()
    ann_id = 1
    stats = {
        'with_boxes': 0,
        'with_tags': 0,
        'with_both': 0
    }

    for item in tqdm(split_ids, desc=f"构建{split_name}集"):
        numeric_id = item['numeric_id']

        # 检查图像是否存在
        if numeric_id not in img_id_to_info:
            continue

        # 获取图像信息
        img_info = img_id_to_info[numeric_id]

        # 避免重复的图像ID
        if numeric_id in image_id_set:
            continue
        image_id_set.add(numeric_id)

        # 获取COCO-CN标签
        has_tags = numeric_id in img_id_to_tags
        tags = img_id_to_tags[numeric_id]['tags'] if has_tags else []
        if has_tags:
            stats['with_tags'] += 1

        # 构建图像信息
        image_entry = {
            'id': numeric_id,
            'file_name': img_info['file_name'],
            'width': img_info['width'],
            'height': img_info['height'],
            'cococn_id': item['cococn_id'],
            'cococn_tags': tags,
            'caption': '，'.join(tags) if tags else ''  # 合并标签作为描述
        }
        coco_data['images'].append(image_entry)

        # 获取边界框
        if numeric_id in img_id_to_boxes:
            boxes = img_id_to_boxes[numeric_id]
            stats['with_boxes'] += 1

            # 过滤小框
            valid_boxes = [box for box in boxes if box['area'] >= MIN_AREA]

            # 限制框的数量
            if len(valid_boxes) > MAX_BOXES_PER_IMAGE:
                valid_boxes = sorted(valid_boxes, key=lambda x: x['area'], reverse=True)[:MAX_BOXES_PER_IMAGE]

            # 添加标注
            for box in valid_boxes:
                annotation = {
                    'id': ann_id,
                    'image_id': numeric_id,
                    'category_id': category_name_to_id[box['category_name']],
                    'bbox': box['bbox'],
                    'area': box['area'],
                    'segmentation': box['segmentation'],
                    'iscrowd': 0
                }
                coco_data['annotations'].append(annotation)
                ann_id += 1

            if has_tags:
                stats['with_both'] += 1

    # 输出统计信息
    print(f"\n{split_name}集统计:")
    print(f"  总图像数: {len(coco_data['images'])}")
    print(f"  总标注数: {len(coco_data['annotations'])}")
    print(f"  有边界框的图像: {stats['with_boxes']}")
    print(f"  有标签的图像: {stats['with_tags']}")
    print(f"  同时有框和标签的图像: {stats['with_both']}")

    # 保存文件
    print(f"正在保存到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=2)

    print(f"保存完成！")

    return coco_data


def main():
    print("=" * 60)
    print("开始构建COCO-CN数据集（COCO格式）")
    print("=" * 60)

    # 1. 加载训练集COCO标注
    train_img_info, train_img_boxes = load_coco_annotations(COCO_TRAIN_ANN, "训练集")

    # 2. 加载验证集COCO标注
    val_img_info, val_img_boxes = load_coco_annotations(COCO_VAL_ANN, "验证集")

    # 3. 加载COCO-CN标签（共用一个标签文件）
    img_id_to_tags = load_cococn_tags(COCOCN_TAGS_FILE)

    # 4. 加载划分文件
    train_ids = load_image_split(TRAIN_SPLIT_FILE, "训练集")
    val_ids = load_image_split(VAL_SPLIT_FILE, "验证集")

    # 5. 创建训练集
    print("\n" + "=" * 60)
    create_coco_format_dataset(
        train_ids, "训练", train_img_info, train_img_boxes, img_id_to_tags,
        OUTPUT_TRAIN
    )

    # 6. 创建验证集
    print("\n" + "=" * 60)
    create_coco_format_dataset(
        val_ids, "验证", val_img_info, val_img_boxes, img_id_to_tags,
        OUTPUT_VAL
    )

    print("\n" + "=" * 60)
    print("✅ 数据集构建完成！")
    print(f"训练集: {OUTPUT_TRAIN}")
    print(f"验证集: {OUTPUT_VAL}")
    print("=" * 60)


if __name__ == '__main__':
    main()







