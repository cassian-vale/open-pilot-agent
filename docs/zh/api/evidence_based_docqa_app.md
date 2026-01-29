# 文档问答服务 API 文档

## 概述

基于 LangGraph + LLM 的结构化文档问答服务，支持流式和非流式输出。该服务提供基于证据的文档问答功能，能够处理长文本文档并返回准确的答案。

## 基础信息

- **根路径**: `/evidence_based_docQA/v1`
- **版本**: 1.0.0
- **默认端口**: 8000

---

## 接口列表

### 1. 健康检查接口

**端点**: `GET /health`

**功能**: 检查服务运行状态和 Agent 初始化状态

**响应**:
```json
{
  "status": "OK",
  "agent": "initialized"
}
```

**错误状态**:
- `503 Service Unavailable`: Agent 未初始化

---

### 2. 文档问答接口

**端点**: `POST /chat`

**功能**: 统一文档问答接口，通过 `stream` 参数控制返回方式

**请求体** (`DocQARequest`):

| 字段 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| `request_id` | string | ✅ | - | 请求唯一标识 |
| `doc_text` | string | ✅ | - | 文档文本内容 |
| `query` | string | ✅ | - | 用户查询问题 |
| `model` | string | ✅ | - | 模型名称 (如 deepseek-chat, gpt-4o) |
| `base_url` | string | ✅ | - | API基础URL |
| `api_key` | string | ✅ | - | API密钥 |
| `stream` | boolean | ❌ | `false` | 是否启用流式输出 |
| `max_tokens` | integer | ❌ | `null` | 最大token数 |
| `temperature` | float | ❌ | `0.0` | 温度参数（0.0-2.0） |
| `top_p` | float | ❌ | `1.0` | Top-p参数（0.0-1.0） |
| `timeout` | float | ❌ | `60.0` | 超时时间（秒） |
| `max_retries` | integer | ❌ | `3` | 最大重试次数 |
| `enable_thinking` | boolean | ❌ | `false` | 是否启用思考过程 |
| `chunk_size` | integer | ❌ | `512` | 文本分块大小 |
| `overlap` | integer | ❌ | `100` | 分块重叠大小 |
| `return_sentences` | boolean | ❌ | `true` | 是否返回句子 |

---

### 非流式响应 (stream=false)

**响应体** (`DocQAResponse`):

| 字段 | 类型 | 描述 |
|------|------|------|
| `output` | object | 问答输出结果 |
| `content` | string | 模型最终输出 |
| `reasoning_content` | string | 思考过程内容 |
| `metadata` | object | 元数据信息 |
| `confidence` | float | 置信度 (0.0-1.0) |

**示例请求**:
```bash
curl -X POST "http://localhost:8000/evidence_based_docQA/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "req_123",
    "doc_text": "这里是文档内容...",
    "query": "文档主要讲了什么？",
    "model": "deepseek-chat",
    "base_url": "https://api.deepseek.com/v1",
    "api_key": "your_api_key",
    "temperature": 0.1,
    "enable_thinking": true
  }'
```

---

### 流式响应 (stream=true)

使用 Server-Sent Events (SSE) 协议返回。

**响应格式**: `text/event-stream`

**事件类型**:

1. **开始事件** (`type: "start"`)
   ```json
   {"type": "start", "content": "", "metadata": {"request_id": "请求ID", "status": "started"}}
   ```

2. **思考事件** (`type: "thinking"`)
   ```json
   {"type": "thinking", "content": "思考内容片段", "metadata": {...}}
   ```

3. **内容事件** (`type: "content"`)
   ```json
   {"type": "content", "content": "回答内容片段", "metadata": {...}}
   ```

4. **完成事件** (`type: "final"`)
   ```json
   {"type": "final", "content": "", "metadata": {"output": {...}, "confidence": 0.95}}
   ```

5. **结束事件** (`type: "end"`)
   ```json
   {"type": "end", "content": "", "metadata": {"request_id": "请求ID", "status": "completed"}}
   ```

6. **错误事件** (`type: "error"`)
   ```json
   {"type": "error", "content": "错误信息"}
   ```

---

## 客户端示例 (Python)

### 非流式调用

```python
import requests

url = "http://localhost:8000/evidence_based_docQA/v1/chat"
payload = {
    "request_id": "test_001",
    "doc_text": "人工智能是计算机科学的一个分支...",
    "query": "什么是人工智能？",
    "model": "deepseek-chat",
    "base_url": "https://api.deepseek.com/v1",
    "api_key": "your_api_key"
}

response = requests.post(url, json=payload)
print(response.json())
```

### 流式调用

```python
import requests
import json

url = "http://localhost:8000/evidence_based_docQA/v1/chat"
payload = {
    "request_id": "test_002",
    "doc_text": "人工智能是计算机科学的一个分支...",
    "query": "什么是人工智能？",
    "model": "deepseek-chat",
    "base_url": "https://api.deepseek.com/v1",
    "api_key": "your_api_key",
    "stream": True
}

with requests.post(url, json=payload, stream=True) as response:
    for line in response.iter_lines():
        if line:
            line_text = line.decode('utf-8')
            if line_text.startswith('data: '):
                data = json.loads(line_text[6:])
                print(f"[{data['type']}] {data.get('content', '')}")
```

---

## 错误处理

| 状态码 | 描述 |
|--------|------|
| `200` | 请求成功 |
| `400` | 请求参数错误 |
| `499` | 客户端断开连接 |
| `503` | 服务未就绪（Agent未初始化） |
| `500` | 服务内部错误 |

---

## 启动服务

```bash
# 直接运行
python doc_qa_app.py

# 使用 Uvicorn
uvicorn doc_qa_app:app --host 0.0.0.0 --port 8000
```

---

## 注意事项

1. **模型兼容性**: 符合 OpenAI 接口规范的 LLM API 均可使用
2. **参数传递**: `model`、`base_url`、`api_key` 为必填参数，需在每次请求中提供
3. **超时设置**: 根据文档长度和复杂度合理设置超时时间
4. **分块参数**: 对于长文档，适当调整 `chunk_size` 和 `overlap` 参数可提高问答质量
5. **思考过程**: 启用 `enable_thinking` 可获取模型的推理过程，但会增加响应时间