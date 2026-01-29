# 文本纠错服务 API 文档

**版本**: 1.0.0  
**描述**: 基于 LangGraph + LLM 的文本纠错服务，支持长文本自动分块处理、流式输出及思维链（Thinking）过程。  
**Base URL**: `http://<host>:8007/text_correction/v1`

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

### 2.2 文本纠错

统一纠错接口，通过 `stream` 参数控制返回方式。

- **接口地址**: `/chat`
- **请求方式**: `POST`
- **Content-Type**: `application/json`

#### 请求参数 (`TextCorrectionRequest`)

| 参数名 | 类型 | 必选 | 描述 | 默认值 |
| :--- | :--- | :--- | :--- | :--- |
| `request_id` | string | ✅ | 请求ID，用于全链路追踪 | - |
| `text` | string | ✅ | 需要纠错的原始文本内容 | - |
| `model` | string | ✅ | 模型名称 (如 deepseek-chat, gpt-4o) | - |
| `base_url` | string | ✅ | LLM API 基础 URL | - |
| `api_key` | string | ✅ | LLM API 密钥 | - |
| `max_chunk_length` | int | ❌ | 文本分块长度 (针对长文本处理) | `512` |
| `stream` | bool | ❌ | 是否启用流式输出 | `false` |
| `enable_thinking` | bool | ❌ | 是否启用/返回思考过程 | `false` |
| `temperature` | float | ❌ | 温度参数 (纠错任务建议为0) | `0.0` |
| `top_p` | float | ❌ | Top-p 采样参数 | `1.0` |
| `max_tokens` | int | ❌ | 最大 Token 数 | `null` |
| `timeout` | float | ❌ | 超时时间 (秒) | `60.0` |
| `max_retries` | int | ❌ | 最大重试次数 | `3` |

#### 请求示例

```json
{
  "request_id": "corr_001",
  "text": "这个项目的进都非常快，我们须要即使跟进。",
  "model": "deepseek-chat",
  "base_url": "https://api.deepseek.com/v1",
  "api_key": "your_api_key",
  "max_chunk_length": 512,
  "temperature": 0.0,
  "enable_thinking": false
}
```

---

### 非流式响应 (stream=false)

#### 响应参数 (`TextCorrectionResponse`)

| 参数名 | 类型 | 描述 |
| :--- | :--- | :--- |
| `output` | object | 业务结果，包含 `corrected_text` 和 `corrections` |
| `content` | string | 模型最终输出 |
| `confidence` | float | 整体置信度 (0.0 - 1.0) |
| `reasoning_content` | string | 模型的思考/推理过程 |
| `metadata` | object | 元数据 (如 token 消耗) |

#### 响应示例

```json
{
  "output": {
    "corrected_text": "这个项目的进度非常快，我们需要及时跟进。",
    "corrections": [
      {
        "original": "进都",
        "corrected": "进度",
        "type": "typo",
        "position": 5
      },
      {
        "original": "须要",
        "corrected": "需要",
        "type": "grammar",
        "position": 12
      },
      {
        "original": "即使",
        "corrected": "及时",
        "type": "typo",
        "position": 14
      }
    ]
  },
  "reasoning_content": "",
  "confidence": 1.0,
  "metadata": {
    "usage": {
      "total_tokens": 200
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
3. **content**: 纠错内容片段
4. **final**: 最终结果
5. **end**: 任务结束
6. **error**: 处理发生错误

#### 响应流示例

```text
data: {"type": "start", "content": "", "metadata": {"request_id": "corr_001", "status": "started"}}

data: {"type": "content", "content": "这个项目的", "metadata": null}

data: {"type": "content", "content": "进度", "metadata": null}

data: {"type": "final", "content": "", "metadata": {"output": {...}, "confidence": 1.0}}

data: {"type": "end", "content": "", "metadata": {"request_id": "corr_001", "status": "completed"}}
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

url = "http://localhost:8007/text_correction/v1/chat"
payload = {
    "request_id": "corr_001",
    "text": "这个项目的进都非常快，我们须要即使跟进。",
    "model": "deepseek-chat",
    "base_url": "https://api.deepseek.com/v1",
    "api_key": "your_api_key"
}

response = requests.post(url, json=payload)
result = response.json()
print("纠错后文本:", result['output']['corrected_text'])
print("错误列表:", result['output']['corrections'])
```

---

## 5. 注意事项

1. **必填参数**: `model`、`base_url`、`api_key` 必须在每次请求中提供
2. **模型兼容**: 符合 OpenAI 接口规范的 LLM API 均可使用
3. **温度建议**: 纠错任务建议使用 0 温度以保证结果准确性
4. **长文本处理**: 通过 `max_chunk_length` 参数自动分块处理长文本
5. **客户端断开**: 支持客户端断开时自动中断后端推理