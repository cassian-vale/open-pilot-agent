# 文本摘要服务 API 文档

**版本**: 1.0.0  
**描述**: 基于 LangGraph + LLM 的文本摘要服务，支持多种摘要类型（如新闻、学术、会议等）、字数限制控制、流式输出及思维链（Thinking）过程。  
**Base URL**: `http://<host>:8005/text_summarization/v1`

---

## 1. 通用说明

- **协议**: HTTP/1.1
- **数据格式**: JSON
- **字符集**: UTF-8
- **模型兼容**: 符合 OpenAI 接口规范的 LLM API 均可使用

---

## 2. 接口列表

### 2.1 健康检查

检查服务实例及 Agent 是否初始化完成。

- **接口地址**: `/health`
- **请求方式**: `GET`

#### 响应示例

```json
{
  "status": "OK",
  "agent": "initialized"
}
```

---

### 2.2 文本摘要

统一摘要接口，通过 `stream` 参数控制返回方式。

- **接口地址**: `/chat`
- **请求方式**: `POST`
- **Content-Type**: `application/json`

#### 请求参数 (`SummarizationRequest`)

| 参数名 | 类型 | 必选 | 描述 | 默认值 |
| :--- | :--- | :--- | :--- | :--- |
| `request_id` | string | ✅ | 请求ID，用于全链路追踪 | - |
| `text` | string | ✅ | 需要摘要的原始文本 | - |
| `model` | string | ✅ | 模型名称 (如 deepseek-chat, gpt-4o) | - |
| `base_url` | string | ✅ | LLM API 基础 URL | - |
| `api_key` | string | ✅ | LLM API 密钥 | - |
| `target_words` | int | ❌ | 目标字数，None表示不限制 | `null` |
| `summary_type` | string | ❌ | 摘要类型 | `要点摘要` |
| `ratio` | float | ❌ | 字数调整比例 (仅限字数模式有效) | `1.5` |
| `stream` | bool | ❌ | 是否启用流式输出 | `false` |
| `enable_thinking` | bool | ❌ | 是否启用/返回思考过程 | `false` |
| `temperature` | float | ❌ | 温度参数 | `0.3` |
| `top_p` | float | ❌ | Top-p 采样参数 | `1.0` |
| `max_tokens` | int | ❌ | 最大 Token 数 | `null` |
| `timeout` | float | ❌ | 超时时间 (秒) | `60.0` |
| `max_retries` | int | ❌ | 最大重试次数 | `3` |

**`summary_type` 可选值**:
- `要点摘要`
- `段落摘要`
- `新闻摘要`
- `技术摘要`
- `会议摘要`
- `学术摘要`
- `故事摘要`

#### 请求示例

```json
{
  "request_id": "sum_12345",
  "text": "这里是一篇非常长的关于人工智能发展的文章内容...",
  "target_words": 200,
  "summary_type": "新闻摘要",
  "model": "deepseek-chat",
  "base_url": "https://api.deepseek.com/v1",
  "api_key": "your_api_key",
  "enable_thinking": true,
  "temperature": 0.3
}
```

---

### 非流式响应 (stream=false)

#### 响应参数 (`SummarizationResponse`)

| 参数名 | 类型 | 描述 |
| :--- | :--- | :--- |
| `output` | object | 业务结果，包含摘要内容 |
| `content` | string | 模型最终输出 |
| `confidence` | float | 整体置信度 (0.0 - 1.0) |
| `reasoning_content` | string | 模型的思考/推理过程 |
| `metadata` | object | 元数据 (如 token 消耗) |

#### 响应示例

```json
{
  "output": {
    "summary": "人工智能技术在近年来取得了突破性进展...",
    "word_count": 198
  },
  "reasoning_content": "首先分析文章的核心观点，然后筛选关键信息...",
  "confidence": 1.0,
  "metadata": {
    "usage": {
      "prompt_tokens": 500,
      "completion_tokens": 250
    }
  }
}
```

---

### 流式响应 (stream=true)

使用 Server-Sent Events (SSE) 协议返回。

- **响应类型**: `text/event-stream`

#### 事件类型说明

1. **start**: 任务开始
2. **thinking**: 思考过程 (需开启 enable_thinking)
3. **content**: 摘要内容片段
4. **final**: 最终结果
5. **end**: 任务结束
6. **error**: 处理发生错误

#### 响应流示例

```text
data: {"type": "start", "content": "", "metadata": {"request_id": "sum_12345", "status": "started"}}

data: {"type": "content", "content": "本文主要", "metadata": null}

data: {"type": "content", "content": "讨论了...", "metadata": null}

data: {"type": "final", "content": "", "metadata": {"output": {...}, "confidence": 1.0}}

data: {"type": "end", "content": "", "metadata": {"request_id": "sum_12345", "status": "completed"}}
```

---

## 3. 错误码说明

| 状态码 | 描述 |
|--------|------|
| `200` | 请求成功 |
| `400` | 参数验证失败 |
| `422` | 数据格式校验错误 |
| `499` | 客户端断开连接 |
| `500` | 服务器内部错误 |
| `503` | Agent 服务尚未初始化 |

---

## 4. 客户端示例 (Python)

```python
import requests

url = "http://localhost:8005/text_summarization/v1/chat"
payload = {
    "request_id": "sum_001",
    "text": "这里是一篇非常长的关于人工智能发展的文章内容...",
    "target_words": 200,
    "summary_type": "新闻摘要",
    "model": "deepseek-chat",
    "base_url": "https://api.deepseek.com/v1",
    "api_key": "your_api_key"
}

response = requests.post(url, json=payload)
result = response.json()
print("摘要:", result['output']['summary'])
print("字数:", result['output']['word_count'])
```

---

## 5. 注意事项

1. **必填参数**: `model`、`base_url`、`api_key` 必须在每次请求中提供
2. **模型兼容**: 符合 OpenAI 接口规范的 LLM API 均可使用
3. **字数控制**: 通过 `target_words` 和 `ratio` 参数控制输出字数
4. **摘要类型**: 不同摘要类型会影响输出风格和结构
5. **客户端断开**: 支持客户端断开时自动中断后端推理