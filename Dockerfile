# Dockerfile.base
FROM python:3.12-slim

WORKDIR /app

# 创建安全用户
ARG APP_UID=9001
ARG APP_GID=9001

# 安装 jemalloc
RUN apt-get update && apt-get install -y libjemalloc2 && rm -rf /var/lib/apt/lists/*

# 设置环境变量预加载 jemalloc
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2

RUN groupadd -r -g ${APP_GID} apprunner \
    && useradd -r -u ${APP_UID} -g ${APP_GID} -s /bin/false apprunner

# 复制统一的 requirements.txt
COPY requirements.txt .

# 安装所有依赖（两个服务共享）
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制所有代码
COPY . .

USER apprunner