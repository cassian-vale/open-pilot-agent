# Open Pilot Agent

[English](#english) | [简体中文](#简体中文)

---

## English

### Project Overview
Open Pilot Agent is a high-performance, structured AI agent framework based on **LangGraph** and customized **LLMClient**. It provides a collection of specialized agents for various NLP tasks, designed for production use with unified configuration and streaming support.

### Web Demo & Online API
- **URL**: [https://apilot.org](https://apilot.org)
- **Online Experience**: Configure your own third-party model API (OpenAI-compatible) to start using the tools online.
- **API Access**: Create your **Pilot API Key** in the dashboard to call our hosted agent service APIs.
- **Online Documentation**: [https://apilot.org/docs/overview](https://apilot.org/docs/overview)
- **Note**: Supports Google/GitHub account login only. Access from Mainland China requires a VPN (ladder).

### Capabilities (What can this project do?)
- **Cognition & Retrieval**:
    - **Query Rewriting**: Intelligent resolution of pronouns and semantic enhancement for better search results.
    - **Evidence-based Document QA**: Accurate question answering for long documents with source citations.
- **Structure & Data Engineering**:
    - **Information Extraction**: Extracting structured data from unstructured text based on custom JSON Schemas.
    - **Schema Generation**: Automatically generating data structures from natural language descriptions.
    - **Keyword Generation**: High-precision keyword extraction for tagging and SEO.
- **Content Governance**:
    - **Smart Summarization**: Multi-style summarization (News, Academic, Meeting, etc.).
    - **Professional Translation**: Multi-domain translation (Governmental, Technical, Social Media styles).
    - **Text Correction & Classification**: Production-grade grammar correction and flexible text categorization.
- **Vertical Solutions**:
    - **RAG Pipeline**: Complete Retrieval-Augmented Generation flow (Search -> Context -> Reasoning -> Response).
    - **Intelligent Document Processing (IDP)**: Automated data capture and insight generation from documents.

### Documentation Links
- **API Documentation**: [English](./docs/en/api/) | [Chinese](./docs/zh/api/)
- **Installation Guide**: [English](./docs/en/install/docker.md) | [Chinese](./docs/zh/install/docker.md)
- **User Guides**:
  - Basic Guide: [English](./docs/en/instructions/base_agent/BaseAgent_Basic_Guide.md) | [Chinese](./docs/zh/instructions/base_agent/BaseAgent应用基础指南.md)
  - Advanced Guide: [English](./docs/en/instructions/base_agent/BaseAgent_Advanced_Guide.md) | [Chinese](./docs/zh/instructions/base_agent/BaseAgent应用进阶指南.md)

### Quick Start

#### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/open-pilot-agent.git
cd open-pilot-agent/pilot-agent
```

#### 2. Development Environment Setup
We recommend using `uv` for dependency management:
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv pip install -r requirements.txt
```

#### 3. Running Services (Development)
You can run individual agent services directly using `python` or `uvicorn`:
```bash
# Enter the project directory
cd open-pilot-agent

# Example: Run Document QA service
python applications/evidence_based_docqa/doc_qa_app.py

# Or using uvicorn
uvicorn applications.evidence_based_docqa.doc_qa_app:app --host 0.0.0.0 --port 8000
```

#### 4. Deployment with Makefile (Recommended)
`make` simplifies the deployment of all services.

**1. Install Docker & Docker Compose**:
- **Ubuntu/Debian**:
  ```bash
  curl -fsSL https://get.docker.com | sudo sh
  sudo apt install docker-compose-plugin -y
  ```

**2. Install Make**:
- **Ubuntu/Debian**: `sudo apt install make -y`
- **CentOS/RHEL**: `sudo yum install make -y`
- **macOS**: `brew install make`

**3. Deployment Commands**:
```bash
# Build all service images
make build

# Start all services in the background
make up

# View real-time logs
make logs

# Stop all services
make down
```

---

## 简体中文

### 项目概述
Open Pilot Agent 是一个基于 **LangGraph** 和自定义 **LLMClient** 的高性能结构化 AI Agent 框架。它提供了一系列专门针对 NLP 任务优化的智能词服务，专为生产环境设计，支持统一配置、流式输出和深度推理。

### 网页 Demo 与 在线 API
- **地址**: [https://apilot.org](https://apilot.org)
- **在线体验**: 网页版简单配置自己的第三方模型 API (OpenAI 兼容) 后即可直接在线体验各项功能。
- **API 调用**: 在控制台创建 **Pilot API Key** 后即可直接调用在线 Agent 服务接口。
- **在线文档**: [https://apilot.org/docs/overview](https://apilot.org/docs/overview)
- **提示**: 仅支持 Google/GitHub 账号登录，中国大陆地区访问需要 VPN。

### 核心能力（项目可以做什么？）
- **认知与检索**:
    - **查询改写**: 智能识别指代消解（指代消歧）和语义增强，大幅提升搜索召回率。
    - **证据驱动文档问答**: 支持超长文档的精准问答，并提供来源依据。
- **结构化与数据工程**:
    - **信息抽取**: 根据自定义 JSON Schema 从非结构化文本中精准提取结构化数据。
    - **Schema 生成**: 根据自然语言需求自动设计数据结构定义。
    - **关键词生成**: 高精度的关键词识别，适用于内容打标和 SEO。
- **内容治理**:
    - **智能摘要**: 支持多种风格（新闻、学术、会议等）和字数控制。
    - **专业翻译**: 覆盖多种语境（政务、学术、流行语等）的高质量翻译。
    - **纠错与分类**: 生产级的语法自动纠错和灵活的文本自动归类。
- **行业解决方案**:
    - **RAG 链路**: 完整的增强检索生成流程（搜索 -> 上下文配置 -> 推理 -> 回答）。
    - **智能文档处理 (IDP)**: 自动化的文档内容采集、核验与见解生成。

### 文档链接
- **API 文档**: [英文](./docs/en/api/) | [中文](./docs/zh/api/)
- **安装指南**: [英文](./docs/en/install/docker.md) | [中文](./docs/zh/install/docker.md)
- **指南文档**:
  - 基础指南: [英文](./docs/en/instructions/base_agent/BaseAgent_Basic_Guide.md) | [中文](./docs/zh/instructions/base_agent/BaseAgent应用基础指南.md)
  - 进阶指南: [英文](./docs/en/instructions/base_agent/BaseAgent_Advanced_Guide.md) | [中文](./docs/zh/instructions/base_agent/BaseAgent应用进阶指南.md)

### 快速开始

#### 1. 克隆仓库
```bash
git clone https://github.com/cassian-vale/open-pilot-agent.git
```

#### 2. 开发环境配置
推荐使用 `uv` 进行依赖管理：
```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装依赖
uv pip install -r requirements.txt
```

#### 3. 运行服务（开发环境）
您可以直接使用 `python` 或 `uvicorn` 运行单个 Agent 服务：
```bash
# 进入项目目录
cd open-pilot-agent

# 示例：运行文档问答服务
python applications/evidence_based_docqa/doc_qa_app.py

# 或者使用 uvicorn
uvicorn applications.evidence_based_docqa.doc_qa_app:app --host 0.0.0.0 --port 8000
```

#### 4. 使用 Makefile 部署（推荐）
通过 `make` 命令可以简化所有服务的部署流程。

**1. 安装 Docker & Docker Compose**:
- **Ubuntu/Debian**:
  ```bash
  curl -fsSL https://get.docker.com | sudo sh
  sudo apt install docker-compose-plugin -y
  ```

**2. 安装 Make**:
- **Ubuntu/Debian**: `sudo apt install make -y`
- **CentOS/RHEL**: `sudo yum install make -y`
- **macOS**: `brew install make`

**3. 部署常用命令**:
```bash
# 构建所有服务镜像
make build

# 后台启动所有服务
make up

# 查看实时日志
make logs

# 停止所有服务
make down

# 重新构建并重启
make reload
```
所有服务将在各自的端口（8000-8008）上运行。
