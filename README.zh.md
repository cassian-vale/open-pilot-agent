# Open Pilot Agent

<p align="center">
  <a href="README.md">English</a> | <b>简体中文</b>
</p>

---

## 项目概述
Open Pilot Agent 是一个基于 **LangGraph** 和自定义 **LLMClient** 的高性能结构化 AI Agent 框架。它提供了一系列专门针对 NLP 任务优化的智能词服务，专为生产环境设计，支持统一配置、流式输出和深度推理。

## 网页 Demo 与 在线 API
- **地址**: [https://apilot.org](https://apilot.org)
- **在线体验**: 网页版简单配置自己的第三方模型 API (OpenAI 兼容) 后即可直接在线体验各项功能。
- **API 调用**: 在控制台创建 **Pilot API Key** 后即可直接调用在线 Agent 服务接口。
- **在线文档**: [https://apilot.org/docs/overview](https://apilot.org/docs/overview)
- **提示**: 仅支持 Google/GitHub 账号登录，中国大陆地区访问需要 VPN。

## 核心能力（项目可以做什么？）
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

## 文档链接
- **API 文档**: [英文](./docs/en/api/) | [中文](./docs/zh/api/)
- **安装指南**: [英文](./docs/en/install/docker.md) | [中文](./docs/zh/install/docker.md)
- **指南文档**:
  - 基础指南: [英文](./docs/en/instructions/base_agent/BaseAgent_Basic_Guide.md) | [中文](./docs/zh/instructions/base_agent/BaseAgent应用基础指南.md)
  - 进阶指南: [英文](./docs/en/instructions/base_agent/BaseAgent_Advanced_Guide.md) | [中文](./docs/zh/instructions/base_agent/BaseAgent应用进阶指南.md)

## 快速开始

### 1. 克隆仓库
```bash
git clone https://github.com/cassian-vale/open-pilot-agent.git
```

### 2. 开发环境配置
推荐使用 `uv` 进行依赖管理：
```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装依赖
uv pip install -r requirements.txt
```

### 3. 运行服务（开发环境）
您可以直接使用 `python` 或 `uvicorn` 运行单个 Agent 服务：
```bash
# 进入项目目录
cd open-pilot-agent

# 示例：运行文档问答服务
python applications/evidence_based_docqa/doc_qa_app.py

# 或者使用 uvicorn
uvicorn applications.evidence_based_docqa.doc_qa_app:app --host 0.0.0.0 --port 8000
```

### 4. 使用 Makefile 部署（推荐）
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
## Star History (Star 增长趋势)

[![Star History Chart](https://api.star-history.com/svg?repos=cassian-vale/open-pilot-agent&type=Date)](https://star-history.com/#cassian-vale/open-pilot-agent&Date)

## 社区共创与贡献

我们欢迎任何形式的贡献！无论是报告 Bug、建议新功能还是提交代码（Pull Request），您的参与对 Open Pilot Agent 的成长至关重要。

1. **问题反馈**：请使用 GitHub Issues 提交您的疑问或 Bug。
2. **功能建议**：有好的想法？欢迎开启一个 Issue 进行讨论。
3. **提交代码**：
   - Fork 本仓库。
   - 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)。
   - 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)。
   - 推送到该分支 (`git push origin feature/AmazingFeature`)。
   - 开启一个 Pull Request。

## 致谢贡献者

感谢所有为 Open Pilot Agent 做出贡献的朋友们！

<a href="https://github.com/cassian-vale/open-pilot-agent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=cassian-vale/open-pilot-agent" />
</a>

---
<p align="center">
  由 Open Pilot 社区倾情打造 ❤️
</p>
