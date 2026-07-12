<p align="center">
  <img src="https://static.yuemutuku.com/logo.png" alt="悦木图库" height="80" onerror="this.style.display='none'">
</p>

<h1 align="center">悦木图库 AI 服务</h1>
<p align="center">
  <strong>轻量级社交图库管理系统的智能引擎</strong>
</p>

<p align="center">
  <a href="https://www.yuemutuku.com" target="_blank">🌐 线上地址</a>
  &nbsp;|&nbsp;
  <a href="#快速启动">🚀 快速启动</a>
  &nbsp;|&nbsp;
  <a href="#api-接口">📡 API 文档</a>
</p>

## 🔗 项目仓库

| 仓库 | 说明 |
|------|------|
| [yuemu-picture-backend](https://github.com/humenglover/yuemu-picture-backend) | Java 后端服务（Spring Boot） |
| [yuemu-picture-frontend](https://github.com/humenglover/yuemu-picture-frontend) | Web 前端（Vue） |
| [yuemu-picture-ai-service](https://github.com/humenglover/yuemu-picture-ai-service) | Python AI 服务（本仓库） |
| [yuemu-picture-official-docs](https://github.com/humenglover/yuemu-picture-official-docs) | 官方文档站 |

---

## 📖 项目介绍

悦木图库 Python AI 子服务，基于 **FastAPI + LangChain ReAct Agent** 构建。提供 RAG 智能客服问答、知识库管理、目标检测、AI 图像处理（去背景、人脸模糊、换背景、图片增强）、AI 生图、语音合成（TTS）等能力，作为 [悦木图库](https://www.yuemutuku.com) 的 AI 引擎层独立部署运行。

> 🖼️ **线上产品**：[https://www.yuemutuku.com](https://www.yuemutuku.com) — 一个轻量级的社交图库平台，支持图片上传、AI 修图、智能搜索与社区分享。

## 🛠 技术栈

| 类别 | 技术 |
|------|------|
| Web 框架 | FastAPI + Uvicorn |
| 大模型 | 通义千问（阿里云 DashScope，OpenAI 兼容协议） |
| 向量数据库 | Qdrant + DashScope Embeddings |
| RAG | LangChain + BM25 + RRF 混合检索 |
| Agent | LangGraph ReAct Agent（工具调用 + 流式输出） |
| 目标检测 | NanoDet-Plus（ONNX Runtime） |
| 图像处理 | OpenCV、U²-Net 抠图 |
| TTS | 腾讯云语音合成 |
| 文本审核 | 纯 DFA 高性能过滤（万级词库） |
| 并发 | ThreadPoolExecutor 线程池 |

## 🚀 快速启动

### 1. 安装依赖

```bash
cd src
pip install -r requirements.txt
```

### 2. 启动 Qdrant 向量数据库

```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 API 密钥
```

| 环境变量 | 说明 | 获取地址 |
|---------|------|---------|
| `QWEN_API_KEY` | 通义千问 API Key | [DashScope 控制台](https://dashscope.console.aliyun.com/apiKey) |
| `TAVILY_API_KEY` | Tavily 搜索 Key | [Tavily](https://app.tavily.com/home) |
| `PEXELS_API_KEY` | Pexels 图片 Key | [Pexels API](https://www.pexels.com/api/) |
| `TENCENTCLOUD_SECRET_ID` | 腾讯云 Secret ID | [腾讯云 CAM](https://console.cloud.tencent.com/cam/capi) |
| `TENCENTCLOUD_SECRET_KEY` | 腾讯云 Secret Key | 同上 |

> ⚠️ `.env` 已加入 `.gitignore`，不会被提交到 Git。

### 4. 启动服务

```bash
python main.py
# 默认地址: http://localhost:8001
# 开发模式热重载已启用，代码保存后自动重启
```

### 5. 验证

```bash
curl http://localhost:8001/
```

## 🐳 Docker 部署

```bash
# 构建镜像（不含密钥）
docker build -t yuemu-picture-ai .

# 运行时注入密钥
docker run -d \
  --name yuemu-picture-ai \
  -p 8001:8001 \
  -e QWEN_API_KEY=your_key \
  -e TAVILY_API_KEY=your_key \
  -e PEXELS_API_KEY=your_key \
  -e TENCENTCLOUD_SECRET_ID=your_id \
  -e TENCENTCLOUD_SECRET_KEY=your_key \
  -e SPRING_PROFILES_ACTIVE=prod \
  yuemu-picture-ai
```

## 📡 API 接口

### 🤖 AI 对话 & RAG

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/rag/sync` | 同步 RAG 问答 |
| POST | `/api/rag/stream` | 流式 RAG 问答（SSE） |
| POST | `/api/rag/summarize` | 摘要生成 |
| POST | `/api/ai/pure-chat` | 纯 LLM 对话（不走检索） |
| POST | `/api/ai_post/stream` | AI 一键成帖（流式） |
| POST | `/api/ai_picture/stream` | AI 识图配文（流式） |
| POST | `/api/ai/image-keywords` | 图片关键词提取 |

### 🖼️ AI 图像处理

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/ai/remove_bg` | 智能去背景（返回 PNG） |
| POST | `/api/ai/face_blur` | 人脸自动打码 |
| POST | `/api/ai/change_background` | 智能换背景 |
| POST | `/api/ai/enhance_image` | 图片清晰度增强 |

### 🎯 目标检测

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/detect/objects` | 上传图片检测 |
| GET | `/api/detect/objects-url` | URL 图片检测 |

### 🔊 语音合成

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/tts` | TTS 语音合成 |

### 📚 知识库管理

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/knowledge/upload` | 上传知识文件 |
| GET | `/api/knowledge/list` | 文件列表 |
| POST | `/api/knowledge/delete` | 删除文件 |
| POST | `/api/knowledge/clear-all` | 清空知识库 |

### 🛡 内容审核

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/comment/moderation` | 评论审核 |
| POST | `/api/post/moderation` | 帖子审核 |

## 📁 项目结构

```
src/
├── main.py              # FastAPI 入口，路由定义
├── config/              # YAML 配置文件（非敏感信息）
├── model/
│   ├── factory.py       # 模型工厂（LLM / Embedding）
│   ├── dto/             # 请求/响应 DTO
│   ├── vo/              # 视图对象
│   └── common/          # 通用模型
├── agent/               # ReAct Agent + 工具集
│   ├── react_agent.py   # 智能体核心
│   ├── biz_tool.py      # 图库业务工具
│   ├── pexels_tool.py   # 图片搜索
│   ├── tavily_tool.py   # 网络搜索
│   ├── tts_tool.py      # 语音合成
│   └── ...              # 更多工具
├── rag/                 # RAG 检索
│   ├── vector_store.py  # 向量存储
│   ├── bm25_retriever.py # BM25 检索
│   └── rrf_fusion.py   # RRF 融合
├── service/             # 业务服务层
├── prompts/             # Prompt 模板
├── models/              # ONNX 模型文件
└── utils/               # 工具类
```

## 🔗 与 Java 后端的交互

Java 后端通过 HTTP 调用本服务，作为 AI 引擎层：

```yaml
rag:
  python-service:
    base-url: "http://127.0.0.1:8001"
    sync-endpoint: "/api/rag/sync"
    stream-endpoint: "/api/rag/stream"
```

## 📧 联系方式

- 作者：**鹿梦**
- 邮箱：109484028@qq.com
- 线上产品：[yuemutuku.com](https://www.yuemutuku.com)
