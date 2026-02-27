FROM ubuntu:20.04


# 设置非交互模式 - 必须在最前面
ENV DEBIAN_FRONTEND=noninteractive

# 预先配置时区
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


# 更新源并安装依赖
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    curl \
    git \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update

# 安装Python 3.9及相关工具
RUN apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    python3.9-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 设置python3.9为默认python版本
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# 安装pip
RUN wget https://bootstrap.pypa.io/get-pip.py -O /tmp/get-pip.py \
    && python /tmp/get-pip.py \
    && rm /tmp/get-pip.py

# 验证安装
RUN python --version && pip --version

# 设置工作目录
WORKDIR /workspace



CMD ["/bin/bash"]

