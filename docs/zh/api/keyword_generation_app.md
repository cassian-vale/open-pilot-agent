# 关键词生成服务 API 文档

**版本**: 1.0.0  
**描述**: 基于 LangGraph + LLM 的智能关键词提取与生成服务，支持领域上下文和流式输出。  
**Base URL**: `http://<host>:8002/keyword_generation/v1`

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

### 2.2 关键词生成

统一生成接口，通过 `stream` 参数控制返回方式。

- **接口地址**: `/chat`
- **请求方式**: `POST`
- **Content-Type**: `application/json`

#### 请求参数 (`KeywordGenerationRequest`)

| 参数名 | 类型 | 必选 | 描述 | 默认值 |
| :--- | :--- | :--- | :--- | :--- |
| `request_id` | string | ✅ | 请求的唯一标识符 | - |
| `content` | string | ✅ | 需要提取关键词的原始文本内容 | - |
| `model` | string | ✅ | 模型名称 (如 deepseek-chat, gpt-4o) | - |
| `base_url` | string | ✅ | LLM API 基础 URL | - |
| `api_key` | string | ✅ | LLM API 密钥 | - |
| `domain_context` | string | ❌ | 领域上下文信息（如"医疗"、"电商"） | `null` |
| `max_keywords` | int | ❌ | 期望生成的最大关键词数量 | `10` |
| `stream` | bool | ❌ | 是否启用流式输出 | `false` |
| `max_tokens` | int | ❌ | 最大生成 Token 数 | `null` |
| `temperature` | float | ❌ | 采样温度 | `0.1` |
| `top_p` | float | ❌ | 核采样参数 | `1.0` |
| `timeout` | float | ❌ | 请求超时时间（秒） | `60.0` |
| `max_retries` | int | ❌ | 最大重试次数 | `3` |
| `enable_thinking` | bool | ❌ | 是否返回模型的思考过程 | `false` |

#### 请求示例

```json
{
  "request_id": "kg_001",
  "content": "深度学习是机器学习的一个子领域，它基于人工神经网络的学习算法。在图像识别、自然语言处理等领域取得了显著成果。",
  "domain_context": "人工智能/计算机科学",
  "max_keywords": 5,
  "model": "deepseek-chat",
  "base_url": "https://api.deepseek.com/v1",
  "api_key": "your_api_key",
  "enable_thinking": true
}
```

---

### 非流式响应 (stream=false)

#### 响应参数 (`KeywordGenerationResponse`)

| 参数名 | 类型 | 描述 |
| :--- | :--- | :--- |
| `output` | List[str] | 生成的关键词字符串列表 |
| `content` | string | 模型最终输出 |
| `reasoning_content` | string | 模型的思考/推理过程 |
| `metadata` | dict | 元数据，包含 token 使用情况等 |
| `confidence` | float | 置信度评分 (0.0 - 1.0) |

#### 响应示例

```json
{
  "output": [
    "深度学习",
    "机器学习",
    "人工神经网络",
    "图像识别",
    "自然语言处理"
  ],
  "reasoning_content": "用户提供的文本主要关于深度学习及其应用...",
  "metadata": {
    "usage": {
      "prompt_tokens": 120,
      "completion_tokens": 30
    }
  },
  "confidence": 1.0
}
```

---

### 流式响应 (stream=true)

使用 Server-Sent Events (SSE) 协议返回。

- **响应类型**: `text/event-stream`

#### 事件类型说明

1. **start**: 任务开始
2. **processing**: 处理中状态更新
3. **thinking**: 模型思考内容片段
4. **content**: 生成的内容片段
5. **final**: 最终完整结果
6. **end**: 任务结束
7. **error**: 发生错误

#### SSE 响应流示例

```text
data: {"type": "start", "content": "", "metadata": {"request_id": "kg_001", "status": "started"}}

data: {"type": "thinking", "content": "分析文本关键词...", "metadata": null}

data: {"type": "content", "content": "深度学习", "metadata": null}

data: {"type": "final", "content": "", "metadata": {"output": ["深度学习", "机器学习"], "confidence": 1.0}}

data: {"type": "end", "content": "", "metadata": {"request_id": "kg_001", "status": "completed"}}
```

---

## 3. 错误码说明

| 状态码 | 描述 |
|--------|------|
| `200` | 请求成功 |
| `400` | 输入参数错误 |
| `499` | 客户端断开连接 |
| `500` | 服务器内部错误 |
| `503` | Agent 服务尚未初始化 |

---

## 4. 客户端示例 (Python)

```python
import requests

url = "http://localhost:8002/keyword_generation/v1/chat"
payload = {
    "request_id": "test_kg",
    "content": "随着新能源汽车的普及，电池回收技术成为了行业关注的焦点。",
    "domain_context": "新能源/环保",
    "max_keywords": 5,
    "model": "deepseek-chat",
    "base_url": "https://api.deepseek.com/v1",
    "api_key": "your_api_key"
}

response = requests.post(url, json=payload)
print("生成的关键词:", response.json()['output'])
```

---

## 5. 注意事项

1. **必填参数**: `model`、`base_url`、`api_key` 必须在每次请求中提供
2. **模型兼容**: 符合 OpenAI 接口规范的 LLM API 均可使用
3. **领域上下文**: 提供领域上下文可提高关键词提取的准确性
4. **客户端断开**: 支持客户端断开时自动中断后端推理