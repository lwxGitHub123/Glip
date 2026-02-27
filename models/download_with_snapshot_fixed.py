import os
from huggingface_hub import snapshot_download
import time

# 设置镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 目标文件信息
repo_id = "GLIPModel/GLIP"
filename_pattern = "*glip_tiny_model_o365_goldg_cc_sbu.pth*"  # 匹配目标文件的模式
local_dir = "./glip_model_snapshot"

print(f"使用 snapshot_download 开始下载，仅匹配模式: {filename_pattern}")

max_retries = 15  # 增加重试次数
for attempt in range(max_retries):
    try:
        # 下载仓库快照，但通过 allow_patterns 过滤文件
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            allow_patterns=[filename_pattern],  # 只下载匹配模式的文件
            # 注意：不要同时使用 ignore_patterns 和 allow_patterns，可能冲突
            max_workers=4,  # 使用4个线程并发下载
            local_dir_use_symlinks=False,
            resume_download=True,  # 支持断点续传
            # 移除 timeout 参数，因为它在这个版本中不支持
            # 移除 etag_timeout 参数，也不支持
            # 如果需要设置超时，可以通过环境变量设置
        )
        print(f"下载完成！文件保存在目录: {downloaded_path}")
        # 列出下载的文件
        if os.path.exists(local_dir):
            downloaded_files = os.listdir(local_dir)
            print(f"下载的文件: {downloaded_files}")
            
            # 检查文件是否完整下载
            target_file = None
            for f in downloaded_files:
                if "glip_tiny_model_o365_goldg_cc_sbu" in f:
                    target_file = f
                    break
            
            if target_file:
                file_path = os.path.join(local_dir, target_file)
                file_size = os.path.getsize(file_path) / (1024**3)
                print(f"目标文件: {target_file}")
                print(f"文件大小: {file_size:.2f} GB")
        break
    except Exception as e:
        print(f"下载尝试 {attempt + 1}/{max_retries} 失败: {e}")
        if attempt < max_retries - 1:
            wait_time = 15 * (attempt + 1)  # 增加等待时间
            print(f"等待 {wait_time} 秒后重试...")
            time.sleep(wait_time)
        else:
            print("已达到最大重试次数，下载失败。")
