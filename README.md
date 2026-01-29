# Open Pilot Agent

<p align="center">
  <b>English</b> | <a href="README.zh.md">简体中文</a>
</p>

---

## Project Overview
Open Pilot Agent is a high-performance, structured AI agent framework based on **LangGraph** and customized **LLMClient**. It provides a collection of specialized agents for various NLP tasks, designed for production use with unified configuration and streaming support.

## Web Demo & Online API
- **URL**: [https://apilot.org](https://apilot.org)
- **Online Experience (Free)**: Configure your own third-party model API (OpenAI-compatible) to start using the tools online for free.
- **API Access (Free)**: Create your **Pilot API Key** in the dashboard to call our hosted agent service APIs for free.
- **Online Documentation**: [https://apilot.org/docs/overview](https://apilot.org/docs/overview)
- **Note**: Since only Google/GitHub account login is supported, a VPN is required for the login process from Mainland China.

## Capabilities (What can this project do?)
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

## Documentation Links
- **API Documentation**: [English](./docs/en/api/) | [Chinese](./docs/zh/api/)
- **Installation Guide**: [English](./docs/en/install/docker.md) | [Chinese](./docs/zh/install/docker.md)
- **User Guides**:
  - Basic Guide: [English](./docs/en/instructions/base_agent/BaseAgent_Basic_Guide.md) | [Chinese](./docs/zh/instructions/base_agent/BaseAgent应用基础指南.md)
  - Developer Advanced Guide: [English](./docs/en/instructions/base_agent/BaseAgent_Advanced_Guide.md) | [Chinese](./docs/zh/instructions/base_agent/BaseAgent应用进阶指南.md)

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/open-pilot-agent.git
cd open-pilot-agent/pilot-agent
```

### 2. Development Environment Setup
We recommend using `uv` for dependency management:
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv pip install -r requirements.txt
```

### 3. Running Services (Development)
You can run individual agent services directly using `python` or `uvicorn`:
```bash
# Enter the project directory
cd open-pilot-agent

# Example: Run Document QA service
python applications/evidence_based_docqa/doc_qa_app.py

# Or using uvicorn
uvicorn applications.evidence_based_docqa.doc_qa_app:app --host 0.0.0.0 --port 8000
```

### 4. Deployment with Makefile (Recommended)
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
## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=cassian-vale/open-pilot-agent&type=Date)](https://star-history.com/#cassian-vale/open-pilot-agent&Date)

## Community & Contribution

We welcome all forms of contributions! Whether it's reporting bugs, suggesting new features, or submitting Pull Requests, your help is vital to the growth of Open Pilot Agent.

1. **Bug Reports**: Please use GitHub Issues to report any problems.
2. **Feature Requests**: Have an idea? Open an issue to discuss it.
3. **Pull Requests**:
   - Fork the repository.
   - Create your feature branch (`git checkout -b feature/AmazingFeature`).
   - Commit your changes (`git commit -m 'Add some AmazingFeature'`).
   - Push to the branch (`git push origin feature/AmazingFeature`).
   - Open a Pull Request.

## Contributors

Thank you to everyone who has contributed to the development of Open Pilot Agent!

<a href="https://github.com/cassian-vale/open-pilot-agent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=cassian-vale/open-pilot-agent" />
</a>

---
<p align="center">
  Built with ❤️ by the Open Pilot Community
</p>
