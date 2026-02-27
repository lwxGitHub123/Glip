# GLIP_Final_Fixed_Inference.py
import os
import sys
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import time
import platform

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


# ====================== 字体设置 ======================
class ChineseFontManager:
    """中文字体管理器 - 单例模式"""
    _instance = None
    _font_path = None
    _font_name = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """初始化字体"""
        self._find_font_file()

    def _find_font_file(self):
        """查找字体文件路径"""
        system = platform.system()

        # 直接指定字体文件路径（根据 fc-list 的输出）
        if system == "Linux":
            font_paths = [
                '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',  # 文泉驿微米黑
                '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',  # 文泉驿正黑
                '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',  # DejaVu Sans（备选）
            ]
        elif system == "Windows":
            font_paths = [
                'C:/Windows/Fonts/msyh.ttc',
                'C:/Windows/Fonts/simhei.ttf',
                'C:/Windows/Fonts/simsun.ttc',
            ]
        elif system == "Darwin":  # macOS
            font_paths = [
                '/System/Library/Fonts/PingFang.ttc',
                '/System/Library/Fonts/STHeiti Light.ttc',
            ]
        else:
            font_paths = []

        # 检查字体文件是否存在
        for path in font_paths:
            if os.path.exists(path):
                self._font_path = path
                self._font_name = os.path.basename(path)
                print(f"✓ 找到中文字体文件: {self._font_path}")
                return True

        print("⚠ 未找到中文字体文件")
        return False

    def get_font(self, size=20):
        """获取PIL字体对象"""
        if self._font_path and os.path.exists(self._font_path):
            try:
                return ImageFont.truetype(self._font_path, size, encoding='utf-8')
            except Exception as e:
                print(f"⚠ 加载字体失败: {e}")

        # 返回默认字体
        return ImageFont.load_default()

    def get_font_name(self):
        """获取字体名称"""
        return self._font_name


# 创建全局字体管理器实例
font_manager = ChineseFontManager()


def draw_chinese_text_cv2(img, text, position, font_size=20, color=(0, 0, 255)):
    """
    使用PIL字体管理器在OpenCV图像上绘制中文文本
    """
    try:
        # 转换为PIL图像
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # 创建绘图对象
        draw = ImageDraw.Draw(img_pil)

        # 获取字体
        font = font_manager.get_font(font_size)

        # 绘制文字（PIL使用RGB颜色）
        draw.text(position, text, font=font, fill=color[::-1])  # BGR转RGB

        # 转换回OpenCV格式
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    except Exception as e:
        print(f"⚠ 绘制中文失败: {e}")
        # 降级到OpenCV（可能显示为乱码）
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return img


class PatchedGLIPDemo(GLIPDemo):
    """
    修复版的GLIPDemo，添加中文支持和红色绘制
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 添加颜色属性（红色）
        self.color = (0, 0, 255)  # BGR格式的红色
        self.font_color = (0, 0, 255)  # 文字颜色也设为红色
        self.font_size = 20  # 字体大小

        # 获取字体信息
        self.font_name = font_manager.get_font_name()
        print("✅ PatchedGLIPDemo 初始化完成，红色属性已添加")
        print(f"🎨 使用红色绘制框和文字")
        if self.font_name:
            print(f"📝 使用字体文件: {self.font_name}")

    def overlay_entity_names(self, image, predictions, names=None, text_size=1.0, text_pixel=2, text_offset=10,
                             text_offset_original=4):
        """
        修复版的overlay_entity_names方法，支持中文
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

        # 确保颜色属性存在（BGR格式）
        color = getattr(self, 'color', (0, 0, 255))
        font_color = getattr(self, 'font_color', (0, 0, 255))

        # 绘制边界框（红色）
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)  # 红色框，线宽2

        # 绘制标签
        for box, score, label in zip(boxes, scores, new_labels):
            x, y = map(int, box[:2])

            # 格式化文本
            if any('\u4e00' <= ch <= '\u9fff' for ch in label):
                s = f"{label}:{score:.2f}"
            else:
                s = template.format(label, score).replace("_", " ").replace("(", "").replace(")", "")

            # 调整位置避免重叠
            for x_prev, y_prev in previous_locations:
                if abs(x - x_prev) < abs(text_offset) and abs(y - y_prev) < abs(text_offset):
                    y -= text_offset

            # 绘制文本（总是使用中文绘制方法，因为我们现在有中文字体）
            image = draw_chinese_text_cv2(
                image, s, (x, y - text_offset_original),
                font_size=self.font_size, color=font_color
            )

            previous_locations.append((x, y))

        return image

    def get_entity_names(self, labels):
        """根据标签索引获取实体名称"""
        new_labels = []
        if self.cfg.MODEL.RPN_ARCHITECTURE == "VLDYHEAD":
            plus = 1
        else:
            plus = 0

        if hasattr(self, 'entities') and self.entities:
            for i in labels:
                if i < len(self.entities):
                    new_labels.append(self.entities[i])
                else:
                    new_labels.append('object')
        else:
            new_labels = ['object' for _ in labels]

        return new_labels


class FixedGLIPInference:
    def __init__(self, weight_path, config_file='configs/pretrain/glip_Swin_T_O365_GoldG.yaml',
                 device=None, min_image_size=480, confidence_threshold=0.5, color=(0, 0, 255)):
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
            # 设置颜色属性（红色）
            self.glip_demo.color = color
            self.glip_demo.font_color = color
            self.glip_demo.font_size = 20
            print("✅ 修复版GLIP预测器初始化完成")
            print(f"🎨 使用红色绘制框和文字")
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
        entity_names = []

        if predictions is not None:
            try:
                if hasattr(predictions, 'bbox'):
                    boxes = predictions.bbox.cpu().numpy()
                if hasattr(predictions, 'get_field'):
                    scores = predictions.get_field('scores').cpu().numpy()
                    labels = predictions.get_field('labels').cpu().numpy()

                    # 获取实体名称
                    if hasattr(self.glip_demo, 'get_entity_names'):
                        entity_names = self.glip_demo.get_entity_names(labels)
                    else:
                        # 备选方法：直接使用entities属性
                        if hasattr(self.glip_demo, 'entities') and self.glip_demo.entities:
                            plus = 1 if self.glip_demo.cfg.MODEL.RPN_ARCHITECTURE == "VLDYHEAD" else 0
                            entity_names = []
                            for i in labels:
                                if i < len(self.glip_demo.entities):
                                    entity_names.append(self.glip_demo.entities[i])
                                else:
                                    entity_names.append('object')
                        else:
                            entity_names = ['object' for _ in labels]
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
            'entity_names': entity_names,
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

        # 如果是RGB格式，转换为BGR保存
        if len(result_image.shape) == 3 and result_image.shape[2] == 3:
            save_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        else:
            save_image = result_image

        cv2.imwrite(save_path, save_image)

        print(f"💾 结果已保存到: {save_path}")

        # 打印详细的目标信息，包括具体标签名称
        if len(results['boxes']) > 0:
            print(f"\n📊 检测到的目标详细信息:")
            print("-" * 60)
            print(f"{'序号':<6}{'标签':<15}{'置信度':<10}{'边界框 (x1,y1,x2,y2)':<30}")
            print("-" * 60)

            for i in range(len(results['boxes'])):
                # 获取标签名称
                if i < len(results.get('entity_names', [])):
                    label_name = results['entity_names'][i]
                else:
                    label_name = f"类别_{results['labels'][i]}" if i < len(results['labels']) else "未知"

                score = results['scores'][i] if i < len(results['scores']) else 0
                box = results['boxes'][i].astype(int) if i < len(results['boxes']) else []

                # 格式化输出
                box_str = f"({box[0]},{box[1]},{box[2]},{box[3]})" if len(box) == 4 else "无效"
                print(f"{i + 1:<6}{label_name:<15}{score:<10.3f}{box_str:<30}")

            print("-" * 60)
            print(f"总计: {len(results['boxes'])} 个目标")
        else:
            print("📊 未检测到目标")

        return save_path


def main():
    """主函数"""
    print("=" * 60)
    print("GLIP 最终修复版推理演示（红色框+中文支持）")
    print("=" * 60)

    # 配置参数
    weight_path = 'output/glip_Swin_T_O365_GoldG_my_model_glip_tiny_model_o365_goldg_cc_sbu/model_final.pth'
    config_path = 'configs/pretrain/glip_Swin_T_O365_GoldG_my_model_predict.yaml'

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

    print(image_files)
    image_path = image_files[1]
    print(f"📸 测试图像: {image_path}")

    # 检查模型文件
    if not os.path.exists(weight_path):
        print(f"❌ 权重文件不存在: {weight_path}")
        print(
            "请下载: wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/glip_tiny_model_o365_goldg_cc_sbu.pth")
        return

    try:
        # 字体管理器已在导入时初始化

        # 初始化推理器（红色框和文字）
        inferencer = FixedGLIPInference(
            weight_path=weight_path,
            config_file=config_path,
            min_image_size=480,  # 快速模式
            color=(0, 0, 255)  # 红色 (BGR格式)
        )

        # 根据图像内容调整检测类别
        if '猫' in image_path.lower():
            captions = '猫. 小猫.'
        else:
            captions = '中间的人'  # '一只白色的猫，在图片左边，右边有一只黄色的猫。'   # '白色的猫'

        # 运行推理
        results = inferencer.predict(
            image_path=image_path,
            captions=captions,
            threshold=0.3
        )

        # 保存结果
        save_path = "result_quick_red_chinese6.jpg"
        inferencer.visualize(results, save_path=save_path)

        print("\n✅ 推理完成！")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


def quick_test():
    """快速测试函数"""
    print("=" * 60)
    print("GLIP 快速测试（红色框+中文支持）")
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
        # 字体管理器已在导入时初始化

        # 使用快速模式（红色框和文字）
        inferencer = FixedGLIPInference(
            weight_path=weight_path,
            config_file=config_path,
            min_image_size=480,
            color=(0, 0, 255)  # 红色 (BGR格式)
        )

        # 运行推理
        results = inferencer.predict(
            image_path=image_path,
            captions='猫. 小猫.',  # 中文类别
            threshold=0.3
        )

        # 保存结果
        inferencer.visualize(results, save_path='quick_result_red.jpg')

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


