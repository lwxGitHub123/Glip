# GLIP_Final_Fixed_Inference.py
import os
import sys
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import time
import platform
import gc  # 垃圾回收

# 添加GLIP项目根目录到系统路径
GLIP_ROOT = os.path.dirname(os.path.abspath(__file__))
if GLIP_ROOT not in sys.path:
    sys.path.insert(0, GLIP_ROOT)

print(f"✅ GLIP根目录: {GLIP_ROOT}")

# 解决 NLTK 下载问题
import nltk

try:
    # 设置 NLTK 数据路径
    nltk.data.path.append('/root/nltk_data')
    # 尝试加载，如果失败则设置离线模式
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("⚠️ NLTK punkt 未找到，尝试手动设置...")
    # 创建必要的目录
    os.makedirs('/root/nltk_data/tokenizers', exist_ok=True)
    os.makedirs('/root/nltk_data/taggers', exist_ok=True)
    # 设置环境变量避免下载
    os.environ['NLTK_DATA'] = '/root/nltk_data'

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
    _font_cache = {}  # 字体缓存，避免重复加载

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
        """获取PIL字体对象（带缓存）"""
        cache_key = f"{self._font_path}_{size}"

        if cache_key in self._font_cache:
            return self._font_cache[cache_key]

        if self._font_path and os.path.exists(self._font_path):
            try:
                font = ImageFont.truetype(self._font_path, size, encoding='utf-8')
                self._font_cache[cache_key] = font
                return font
            except Exception as e:
                print(f"⚠ 加载字体失败: {e}")

        # 返回默认字体
        default_font = ImageFont.load_default()
        self._font_cache[cache_key] = default_font
        return default_font

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

            # 绘制文本
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
                idx = i - plus if plus else i
                if 0 <= idx < len(self.entities):
                    new_labels.append(self.entities[idx])
                else:
                    new_labels.append('object')
        else:
            new_labels = ['object' for _ in labels]

        return new_labels


class FixedGLIPInference:
    def __init__(self, weight_path, config_file='configs/pretrain/glip_Swin_T_O365_GoldG.yaml',
                 device=None, min_image_size=480, confidence_threshold=0.5, color=(0, 0, 255),
                 enable_warmup=True, enable_benchmark=True):
        """
        修复版GLIP推理器（优化版）
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

        # GPU优化设置
        if self.device == 'cuda':
            # 清空缓存
            torch.cuda.empty_cache()
            gc.collect()

            # 启用cuDNN benchmark
            if enable_benchmark:
                torch.backends.cudnn.benchmark = True
                print("⚡ 已启用cuDNN benchmark")

            # 显示GPU信息
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            # 修复：使用正确的显存查询API
            print(f"📊 总显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
            print(f"📊 已分配显存: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")
            print(f"📊 缓存显存: {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")

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
            start_init = time.time()
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
            init_time = time.time() - start_init
            print(f"✅ 修复版GLIP预测器初始化完成 (耗时: {init_time:.2f}秒)")
            print(f"🎨 使用红色绘制框和文字")

            # 预热模型
            if enable_warmup:
                self._warmup()

        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            raise

    def _warmup(self, iterations=2):
        """预热模型，减少首次推理延迟"""
        print("🔥 预热模型中...")
        start_time = time.time()

        # 创建虚拟图像
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_captions = "person. dog. cat."

        try:
            for i in range(iterations):
                self.glip_demo.run_on_web_image(dummy_image, dummy_captions, thresh=0.5)
                print(f"  预热 {i + 1}/{iterations} 完成")

            elapsed = time.time() - start_time
            print(f"✅ 模型预热完成，耗时: {elapsed:.2f}秒")
        except Exception as e:
            print(f"⚠️ 预热失败: {e}")

    def predict(self, image_path, captions, threshold=0.5, max_size=None):
        """
        使用GLIP进行推理（优化版）
        """
        start_time = time.time()

        # 记录开始时的显存
        if self.device == 'cuda':
            gpu_mem_before = torch.cuda.memory_allocated(0) / 1024 ** 2

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"❌ 图像文件不存在: {image_path}")

        print(f"🖼️ 处理图像: {os.path.basename(image_path)}")
        print(f"📝 检测类别: {captions}")

        # 加载图像
        image = cv2.imread(image_path)
        if image is None:
            image = cv2.cvtColor(np.array(Image.open(image_path).convert('RGB')), cv2.COLOR_RGB2BGR)

        # 检查图像尺寸
        h, w = image.shape[:2]
        original_size = (w, h)
        print(f"📐 原始图像尺寸: {w}x{h}")

        # 如果图像太大，调整大小
        if max_size and max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
            print(f"📐 调整后尺寸: {new_w}x{new_h} (缩放因子: {scale:.2f})")

        # 运行GLIP推理
        inference_start = time.time()
        try:
            result, predictions = self.glip_demo.run_on_web_image(
                image,
                captions,
                thresh=threshold
            )
        except Exception as e:
            print(f"⚠️ 推理时出错: {e}")
            result, predictions = self.glip_demo.run_on_web_image(
                image,
                captions
            )
        inference_time = time.time() - inference_start

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
                        if hasattr(self.glip_demo, 'entities') and self.glip_demo.entities:
                            plus = 1 if self.glip_demo.cfg.MODEL.RPN_ARCHITECTURE == "VLDYHEAD" else 0
                            entity_names = []
                            for i in labels:
                                idx = i - plus if plus else i
                                if 0 <= idx < len(self.glip_demo.entities):
                                    entity_names.append(self.glip_demo.entities[idx])
                                else:
                                    entity_names.append('object')
                        else:
                            entity_names = ['object' for _ in labels]
            except Exception as e:
                print(f"⚠️ 提取预测结果时出错: {e}")

        boxes = np.array(boxes) if len(boxes) > 0 else np.array([])
        scores = np.array(scores) if len(scores) > 0 else np.array([])
        labels = np.array(labels) if len(labels) > 0 else np.array([])

        total_time = time.time() - start_time

        # 显示性能信息
        print(f"\n📊 性能统计:")
        print(f"  ⏱️ 推理时间: {inference_time:.3f}秒")
        print(f"  ⏱️ 总耗时: {total_time:.3f}秒")
        print(f"  ✅ 检测到 {len(boxes)} 个目标")

        # GPU显存统计
        if self.device == 'cuda':
            gpu_mem_after = torch.cuda.memory_allocated(0) / 1024 ** 2
            gpu_mem_diff = gpu_mem_after - gpu_mem_before
            gpu_mem_reserved = torch.cuda.memory_reserved(0) / 1024 ** 2
            print(f"  🎮 GPU显存变化: {gpu_mem_diff:+.2f} MB")
            print(f"  🎮 当前已分配显存: {gpu_mem_after:.2f} MB")
            print(f"  🎮 当前缓存显存: {gpu_mem_reserved:.2f} MB")

        return {
            'boxes': boxes,
            'scores': scores,
            'labels': labels,
            'entity_names': entity_names,
            'result_image': result[:, :, ::-1] if result is not None else image[:, :, ::-1],
            'original_image': image[:, :, ::-1],
            'image_path': image_path,
            'inference_time': inference_time,
            'total_time': total_time,
            'original_size': original_size,
            'processed_size': (image.shape[1], image.shape[0]) if image is not None else original_size
        }

    def predict_batch(self, image_paths, captions, threshold=0.5, max_size=None, batch_size=4):
        """批量处理多张图像"""
        results = []
        total_images = len(image_paths)

        print(f"\n📦 开始批量处理 {total_images} 张图像...")

        for i in range(0, total_images, batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_images - 1) // batch_size + 1

            print(f"\n📦 处理批次 {batch_num}/{total_batches}")

            batch_start = time.time()
            for j, image_path in enumerate(batch_paths):
                print(f"  🖼️ 图像 {i + j + 1}/{total_images}: {os.path.basename(image_path)}")
                result = self.predict(image_path, captions, threshold, max_size)
                results.append(result)

            batch_time = time.time() - batch_start
            print(f"  ⏱️ 批次耗时: {batch_time:.2f}秒")

            # 每处理完一批，清理缓存
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

        # 统计总体性能
        if results:
            avg_time = np.mean([r['inference_time'] for r in results])
            print(f"\n📊 批量处理完成:")
            print(f"  📸 总图像数: {total_images}")
            print(f"  ⏱️ 平均推理时间: {avg_time:.3f}秒/张")

        return results

    def visualize(self, results, save_path=None):
        """保存可视化结果（改进版）"""
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

        print(f"\n💾 结果已保存到: {save_path}")

        # 打印详细的目标信息
        if len(results['boxes']) > 0:
            print(f"\n📊 检测到的目标详细信息:")
            print("=" * 80)
            print(f"{'序号':<6}{'标签':<20}{'置信度':<10}{'边界框 (x1,y1,x2,y2)':<35}")
            print("=" * 80)

            for i in range(len(results['boxes'])):
                # 获取标签名称
                if i < len(results.get('entity_names', [])):
                    label_name = results['entity_names'][i]
                else:
                    label_name = f"类别_{results['labels'][i]}" if i < len(results['labels']) else "未知"

                score = results['scores'][i] if i < len(results['scores']) else 0
                box = results['boxes'][i].astype(int) if i < len(results['boxes']) else []

                # 格式化输出
                if len(box) == 4:
                    box_str = f"({box[0]:4d},{box[1]:4d},{box[2]:4d},{box[3]:4d})"
                else:
                    box_str = "无效"

                # 截断过长的标签名
                if len(label_name) > 18:
                    label_name = label_name[:15] + "..."

                print(f"{i + 1:<6}{label_name:<20}{score:<10.3f}{box_str:<35}")

            print("=" * 80)
            print(f"总计: {len(results['boxes'])} 个目标")

            # 按置信度排序显示Top 3
            if len(results['boxes']) > 3:
                print(f"\n🏆 置信度最高的3个目标:")
                indices = np.argsort(results['scores'])[-3:][::-1]
                for idx in indices:
                    label_name = results['entity_names'][idx] if idx < len(results['entity_names']) else "未知"
                    score = results['scores'][idx]
                    print(f"   {label_name}: {score:.3f}")
        else:
            print("📊 未检测到目标")

        return save_path


def main():
    """主函数"""
    print("=" * 60)
    print("GLIP 最终修复版推理演示（红色框+中文支持）")
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
        # 初始化推理器（带优化）
        inferencer = FixedGLIPInference(
            weight_path=weight_path,
            config_file=config_path,
            min_image_size=480,
            color=(0, 0, 255),  # 红色
            enable_warmup=True,  # 预热模型
            enable_benchmark=True  # 启用benchmark
        )

        # 根据图像内容调整检测类别
        if '猫' in image_path.lower():
            captions = '猫. 小猫.'
        else:
            captions = '人. 狗. 小汽车. 猫.'

        # 运行推理（限制图像大小）
        results = inferencer.predict(
            image_path=image_path,
            captions=captions,
            threshold=0.3,
            max_size=1200  # 限制最大尺寸
        )

        # 保存结果
        save_path = "result_quick_red_chinese.jpg"
        inferencer.visualize(results, save_path=save_path)

        print("\n✅ 推理完成！")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


def quick_test():
    """快速测试函数（带性能监控）"""
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
        # 使用快速模式
        inferencer = FixedGLIPInference(
            weight_path=weight_path,
            config_file=config_path,
            min_image_size=480,
            color=(0, 0, 255),  # 红色
            enable_warmup=False,  # 快速测试不预热
            enable_benchmark=True
        )

        # 运行推理
        results = inferencer.predict(
            image_path=image_path,
            captions='猫. 小猫.',
            threshold=0.3,
            max_size=800  # 更小的尺寸，更快
        )

        # 保存结果
        inferencer.visualize(results, save_path='quick_result_red.jpg')

        print("\n✅ 测试完成！")

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


def batch_test():
    """批量测试函数"""
    print("=" * 60)
    print("GLIP 批量测试")
    print("=" * 60)

    weight_path = 'MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth'
    config_path = 'configs/pretrain/glip_Swin_T_O365_GoldG.yaml'

    # 获取所有测试图像
    test_dir = './test'
    import glob
    image_files = glob.glob(os.path.join(test_dir, '*.jpg')) + \
                  glob.glob(os.path.join(test_dir, '*.png'))

    if len(image_files) < 2:
        print("⚠️ 需要至少2张图像进行批量测试")
        return

    print(f"📸 找到 {len(image_files)} 张测试图像")

    try:
        inferencer = FixedGLIPInference(
            weight_path=weight_path,
            config_file=config_path,
            min_image_size=480,
            color=(0, 0, 255),
            enable_warmup=True,
            enable_benchmark=True
        )

        # 批量处理
        results = inferencer.predict_batch(
            image_paths=image_files[:4],  # 测试前4张
            captions='人. 狗. 猫. 小汽车.',
            threshold=0.3,
            max_size=800,
            batch_size=2
        )

        print("\n✅ 批量测试完成！")

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # 选择运行模式
    mode = 'main'  # 可选: 'main', 'quick', 'batch'

    if mode == 'main':
        main()
    elif mode == 'quick':
        quick_test()
    elif mode == 'batch':
        batch_test()


