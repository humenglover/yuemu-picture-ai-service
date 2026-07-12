# 基础镜像：Python 3.10 轻量化版本
FROM python:3.10-slim

# 解决时区问题
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 升级pip（腾讯云源加速）
RUN pip install --upgrade pip -i https://mirrors.cloud.tencent.com/pypi/simple/ --trusted-host mirrors.cloud.tencent.com

# 安装Python依赖（腾讯云源）
RUN pip install --no-cache-dir -i https://mirrors.cloud.tencent.com/pypi/simple/ --trusted-host mirrors.cloud.tencent.com --prefer-binary -r requirements.txt

# 复制项目代码（.dockerignore 已排除所有运行时数据目录）
COPY src/ /app/src/

# 预建运行时数据目录（部署初始化为空，数据由外部挂载 volume 持久化）
RUN mkdir -p /app/src/knowledge \
    && mkdir -p /app/src/md5 \
    && mkdir -p /app/src/logs \
    && touch /app/src/md5/md5_record.txt

# 清理构建缓存
RUN find /app -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# 暴露端口
EXPOSE 8001

# 启动服务
CMD ["sh", "-c", "find /app -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null; cd /app/src && python main.py"]