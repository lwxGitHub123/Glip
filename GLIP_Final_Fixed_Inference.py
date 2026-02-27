# GLIP_Final_Fixed_Inference.py
import os
import sys
import torch
import numpy as np
from PIL import Image
import cv2
import time

# 添加GLIP项目根目录到系统路径
GLIP_ROOT = os.path.dirname(os.path.abspath(__file__))
if GLIP_ROOT not in sys.path:
    sys.path.insert(0, GLIP_ROOT)

print(f"✅ GLIP根目录: {GLIP_ROOT}")

# 导入GLIP预测器
try:
    from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
    from maskrcnn_benchmark.engine.predictor_glip import create_positive_map, \
        create_positive_map_label_to_token_from_positive_map
    from maskrcnn_benchmark.config import cfg

    print("✅ 成功导入GLIP预测器")
    print(f"  - GLIPDemo 类型: {type(GLIPDemo)}")
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    sys.exit(1)


class PatchedGLIPDemo(GLIPDemo):  # 修正这里：GLPIDemo -> GLIPDemo
    """
    修复版的GLIPDemo，添加缺少的color属性
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 添加color属性，默认白色
        self.color = 255
        print("✅ PatchedGLIPDemo 初始化完成，color属性已添加")

    def overlay_entity_names(self, image, predictions, names=None, text_size=1.0, text_pixel=2, text_offset=10,
                             text_offset_original=4):
        """
        修复版的overlay_entity_names方法
        """
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels").tolist()
        new_labels = []
        if self.cfg.MODEL.RPN_ARCHITECTURE == "VLDYHEAD":
            plus = 1
        else:
            plus = 0
        self.plus = plus
        if hasattr(self, 'entities') and self.entities and self.plus:
            for i in labels:
                if i <= len(self.entities):
                    new_labels.append(self.entities[i - self.plus])
                else:
                    new_labels.append('object')
        else:
            new_labels = ['object' for i in labels]
        boxes = predictions.bbox

        template = "{}:{:.2f}"
        previous_locations = []

        # 确保color属性存在
        color = getattr(self, 'color', 255)

        for box, score, label in zip(boxes, scores, new_labels):
            x, y = box[:2]
            s = template.format(label, score).replace("_", " ").replace("(", "").replace(")", "")
            for x_prev, y_prev in previous_locations:
                if abs(x - x_prev) < abs(text_offset) and abs(y - y_prev) < abs(text_offset):
                    y -= text_offset

            cv2.putText(
                image, s, (int(x), int(y) - text_offset_original),
                cv2.FONT_HERSHEY_SIMPLEX, text_size,
                (color, color, color), text_pixel, cv2.LINE_AA
            )
            previous_locations.append((int(x), int(y)))

        return image


class FixedGLIPInference:
    def __init__(self, weight_path, config_file='configs/pretrain/glip_Swin_T_O365_GoldG.yaml',
                 device=None, min_image_size=480, confidence_threshold=0.5, color=255):
        """
        修复版GLIP推理器
        """
        # 检查文件是否存在
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"❌ 权重文件不存在: {weight_path}")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"❌ 配置文件不存在: {config_file}")

        # 设置设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"🔧 使用设备: {self.device}")
        print(f"📏 最小图像尺寸: {min_image_size}")

        # 加载配置
        self.cfg = cfg.clone()
        self.cfg.merge_from_file(config_file)
        self.cfg.MODEL.WEIGHTS = weight_path
        self.cfg.MODEL.DEVICE = self.device
        self.cfg.freeze()

        print(f"📄 配置文件: {config_file}")

        # 创建修复版的GLIP演示对象
        print("🏗️ 初始化修复版GLIP预测器...")
        try:
            self.glip_demo = PatchedGLIPDemo(
                cfg=self.cfg,
                min_image_size=min_image_size,
                confidence_threshold=confidence_threshold,
                show_mask_heatmaps=False
            )
            # 设置color属性
            self.glip_demo.color = color
            print("✅ 修复版GLIP预测器初始化完成")
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            raise

    def predict(self, image_path, captions, threshold=0.5):
        """
        使用GLIP进行推理
        """
        start_time = time.time()

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"❌ 图像文件不存在: {image_path}")

        print(f"🖼️ 处理图像: {os.path.basename(image_path)}")
        print(f"📝 检测类别: {captions}")

        # 加载图像
        image = cv2.imread(image_path)
        if image is None:
            image = cv2.cvtColor(np.array(Image.open(image_path).convert('RGB')), cv2.COLOR_RGB2BGR)

        # 运行GLIP推理
        try:
            result, predictions = self.glip_demo.run_on_web_image(
                image,
                captions,
                thresh=threshold
            )
        except Exception as e:
            print(f"⚠️ 推理时出错: {e}")
            # 尝试不带thresh参数
            result, predictions = self.glip_demo.run_on_web_image(
                image,
                captions
            )

        # 提取预测结果
        boxes = []
        scores = []
        labels = []

        if predictions is not None:
            try:
                if hasattr(predictions, 'bbox'):
                    boxes = predictions.bbox.cpu().numpy()
                if hasattr(predictions, 'get_field'):
                    scores = predictions.get_field('scores').cpu().numpy()
                    labels = predictions.get_field('labels').cpu().numpy()
            except Exception as e:
                print(f"⚠️ 提取预测结果时出错: {e}")

        boxes = np.array(boxes) if len(boxes) > 0 else np.array([])
        scores = np.array(scores) if len(scores) > 0 else np.array([])
        labels = np.array(labels) if len(labels) > 0 else np.array([])

        elapsed_time = time.time() - start_time
        print(f"⏱️ 推理耗时: {elapsed_time:.2f}秒")
        print(f"✅ 检测到 {len(boxes)} 个目标")

        return {
            'boxes': boxes,
            'scores': scores,
            'labels': labels,
            'result_image': result[:, :, ::-1] if result is not None else image[:, :, ::-1],
            'original_image': image[:, :, ::-1],
            'image_path': image_path,
            'inference_time': elapsed_time
        }

    def visualize(self, results, save_path=None):
        """保存可视化结果"""
        if save_path is None:
            base_name = os.path.splitext(os.path.basename(results['image_path']))[0]
            save_path = f"{base_name}_detected.jpg"

        os.makedirs(os.path.dirname(os.path.abspath(save_path)) if os.path.dirname(save_path) else '.', exist_ok=True)

        result_image = results['result_image']
        Image.fromarray(result_image).save(save_path)

        print(f"💾 结果已保存到: {save_path}")

        if len(results['boxes']) > 0:
            print(f"📊 检测到 {len(results['boxes'])} 个目标")
            for i, (box, score) in enumerate(zip(results['boxes'][:5], results['scores'][:5])):
                print(f"   目标 {i + 1}: 置信度 {score:.3f}, 框 {box.astype(int)}")
            if len(results['boxes']) > 5:
                print(f"   ... 还有 {len(results['boxes']) - 5} 个目标")

        return save_path


def main():
    """主函数"""
    print("=" * 60)
    print("GLIP 最终修复版推理演示")
    print("=" * 60)

    # 配置参数
    weight_path = 'MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth'
    config_path = 'configs/pretrain/glip_Swin_T_O365_GoldG.yaml'

    # 检查测试图像
    test_dir = './test'
    if not os.path.exists(test_dir):
        os.makedirs(test_dir, exist_ok=True)
        print(f"📁 创建测试目录: {test_dir}")
        print("请放置测试图像到 ./test/ 目录下")
        return

    import glob
    image_files = glob.glob(os.path.join(test_dir, '*.jpg')) + \
                  glob.glob(os.path.join(test_dir, '*.png')) + \
                  glob.glob(os.path.join(test_dir, '*.jpeg'))

    if not image_files:
        print(f"❌ 在 {test_dir} 中没有找到图像文件")
        return

    image_path = image_files[0]
    print(f"📸 测试图像: {image_path}")

    # 检查模型文件
    if not os.path.exists(weight_path):
        print(f"❌ 权重文件不存在: {weight_path}")
        print(
            "请下载: wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/glip_tiny_model_o365_goldg_cc_sbu.pth")
        return

    try:
        # 测试快速模式
        print(f"\n{'=' * 40}")
        print(f"测试: 快速模式 (尺寸: 480)")
        print(f"{'=' * 40}")

        # 初始化推理器
        inferencer = FixedGLIPInference(
            weight_path=weight_path,
            config_file=config_path,
            min_image_size=480,  # 快速模式
            color=0  # 黑色文字
        )

        # 根据图像内容调整检测类别
        if 'cat' in image_path.lower():
            captions = 'cat. kitten.'
        else:
            captions = 'person. dog. car. cat.'

        # 运行推理
        results = inferencer.predict(
            image_path=image_path,
            captions=captions,
            threshold=0.3
        )

        # 保存结果
        save_path = "result_quick.jpg"
        inferencer.visualize(results, save_path=save_path)

        print("\n✅ 推理完成！")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


def quick_test():
    """快速测试函数"""
    print("=" * 60)
    print("GLIP 快速测试")
    print("=" * 60)

    weight_path = 'MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth'
    config_path = 'configs/pretrain/glip_Swin_T_O365_GoldG.yaml'
    image_path = './test/2_1cats.jpg'

    if not os.path.exists(image_path):
        print(f"❌ 测试图像不存在: {image_path}")
        return

    if not os.path.exists(weight_path):
        print(f"❌ 权重不存在: {weight_path}")
        return

    try:
        # 使用快速模式
        inferencer = FixedGLIPInference(
            weight_path=weight_path,
            config_file=config_path,
            min_image_size=480,
            color=0  # 黑色文字
        )

        # 运行推理
        results = inferencer.predict(
            image_path=image_path,
            captions='cat. kitten.',
            threshold=0.3
        )

        # 保存结果
        inferencer.visualize(results, save_path='quick_result.jpg')

        print("\n✅ 测试完成！")

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # 运行主程序
    main()

    # 或者运行快速测试
    # quick_test()

