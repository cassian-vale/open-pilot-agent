# 文本翻译服务 API 文档

**版本**: 1.0.0  
**描述**: 基于 LangGraph + LLM 的文本翻译服务，支持多种翻译风格（如政务、学术、小红书风等）、多语言方向以及流式输出和思维链（Thinking）过程。  
**Base URL**: `http://<host>:8008/text_translation/v1`

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

### 2.2 文本翻译

统一翻译接口，通过 `stream` 参数控制返回方式。

- **接口地址**: `/chat`
- **请求方式**: `POST`
- **Content-Type**: `application/json`

#### 请求参数 (`TranslationRequest`)

| 参数名 | 类型 | 必选 | 描述 | 默认值 |
| :--- | :--- | :--- | :--- | :--- |
| `request_id` | string | ✅ | 请求ID，用于全链路追踪 | - |
| `text` | string | ✅ | 需要翻译的原始文本 | - |
| `translation_direction` | string | ✅ | 翻译方向 (例如: "中译英", "英译中") | - |
| `model` | string | ✅ | 模型名称 (如 deepseek-chat, gpt-4o) | - |
| `base_url` | string | ✅ | LLM API 基础 URL | - |
| `api_key` | string | ✅ | LLM API 密钥 | - |
| `translation_style` | string | ❌ | 翻译风格 | `普通` |
| `stream` | bool | ❌ | 是否启用流式输出 | `false` |
| `enable_thinking` | bool | ❌ | 是否启用/返回思考过程 | `false` |
| `temperature` | float | ❌ | 温度参数 | `0.3` |
| `top_p` | float | ❌ | Top-p 采样参数 | `1.0` |
| `max_tokens` | int | ❌ | 最大 Token 数 | `null` |
| `timeout` | float | ❌ | 超时时间 (秒) | `60.0` |
| `max_retries` | int | ❌ | 最大重试次数 | `3` |

**`translation_style` 常见示例**:
- `普通`: 标准翻译
- `政务`: 正式、严谨
- `学术`: 专业术语、学术规范
- `小红书`: 活泼、Emoji丰富、口语化
- `科幻`: 未来感、科技感
- `贴吧`: 网络俚语、接地气
- `剧本`: 对话形式、场景感

#### 请求示例

```json
{
  "request_id": "trans_001",
  "text": "人工智能将彻底改变我们的生活方式。",
  "translation_direction": "中译英",
  "translation_style": "学术",
  "model": "deepseek-chat",
  "base_url": "https://api.deepseek.com/v1",
  "api_key": "your_api_key",
  "temperature": 0.3,
  "enable_thinking": true
}
```

---

### 非流式响应 (stream=false)

#### 响应参数 (`TranslationResponse`)

| 参数名 | 类型 | 描述 |
| :--- | :--- | :--- |
| `output` | object | 业务结果，包含 `translated_text` (译文) 和 `quality_score` (质量评分) |
| `content` | string | 模型最终输出 |
| `confidence` | float | 整体置信度 (0.0 - 1.0) |
| `reasoning_content` | string | 模型的思考/推理过程 |
| `metadata` | object | 元数据 (如 token 消耗) |

#### 响应示例

```json
{
  "output": {
    "translated_text": "Artificial Intelligence constitutes a paradigm shift that will fundamentally restructure our mode of existence.",
    "quality_score": 95
  },
  "reasoning_content": "用户要求学术风格。原文含义为AI改变生活。学术表达中可以用 'paradigm shift', 'restructure', 'mode of existence' 等词汇...",
  "confidence": 0.95,
  "metadata": {
    "usage": {
      "prompt_tokens": 100,
      "completion_tokens": 50
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
3. **content**: 翻译内容片段
4. **final**: 最终结果
5. **end**: 任务结束
6. **error**: 处理发生错误

#### 响应流示例

```text
data: {"type": "start", "content": "", "metadata": {"request_id": "trans_001", "status": "started"}}

data: {"type": "content", "content": "Artificial", "metadata": null}

data: {"type": "content", "content": " Intelligence", "metadata": null}

data: {"type": "final", "content": "", "metadata": {"output": {...}, "confidence": 0.95}}

data: {"type": "end", "content": "", "metadata": {"request_id": "trans_001", "status": "completed"}}
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

url = "http://localhost:8008/text_translation/v1/chat"
payload = {
    "request_id": "trans_001",
    "text": "人工智能将彻底改变我们的生活方式。",
    "translation_direction": "中译英",
    "translation_style": "学术",
    "model": "deepseek-chat",
    "base_url": "https://api.deepseek.com/v1",
    "api_key": "your_api_key"
}

response = requests.post(url, json=payload)
print(response.json())
```

---

## 5. 注意事项

1. **必填参数**: `model`、`base_url`、`api_key` 必须在每次请求中提供
2. **模型兼容**: 符合 OpenAI 接口规范的 LLM API 均可使用
3. **翻译风格**: 可根据应用场景选择合适的翻译风格
4. **客户端断开**: 支持客户端断开时自动中断后端推理