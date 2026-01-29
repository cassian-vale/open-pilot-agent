# 查询改写服务 API 文档

**版本**: 1.0.0  
**描述**: 基于 LangGraph + LLM 的智能查询改写服务，支持指代消歧、查询扩写、语义增强等策略。  
**Base URL**: `http://<host>:8003/query_rewriting/v1`

---

## 1. 通用说明

- **协议**: HTTP/1.1
- **数据格式**: JSON
- **字符集**: UTF-8
- **模型兼容**: 符合 OpenAI 接口规范的 LLM API 均可使用

**主要策略**:
- **指代消歧**: 结合对话历史，解析代词的具体指代
- **查询扩写**: 添加同义词、相关术语以增加召回率
- **语义增强**: 补全省略的上下文信息
- **格式调整**: 调整语法结构和表达视角

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

### 2.2 查询改写

统一改写接口，通过 `stream` 参数控制返回方式。

- **接口地址**: `/chat`
- **请求方式**: `POST`
- **Content-Type**: `application/json`

#### 请求参数 (`QueryRewriteRequest`)

| 参数名 | 类型 | 必选 | 描述 | 默认值 |
| :--- | :--- | :--- | :--- | :--- |
| `request_id` | string | ✅ | 请求的唯一标识符 | - |
| `query` | string | ✅ | 需要改写的原始查询文本 | - |
| `model` | string | ✅ | 模型名称 (如 deepseek-chat, gpt-4o) | - |
| `base_url` | string | ✅ | LLM API 基础 URL | - |
| `api_key` | string | ✅ | LLM API 密钥 | - |
| `conversation_history` | List[Dict] | ❌ | 对话历史，用于指代消歧 | `null` |
| `domain_context` | string | ❌ | 领域上下文信息 | `null` |
| `max_rewrites` | int | ❌ | 期望生成的最大改写版本数量 | `5` |
| `preserve_system` | bool | ❌ | 是否保留系统预设指令的影响 | `true` |
| `stream` | bool | ❌ | 是否启用流式输出 | `false` |
| `enable_thinking` | bool | ❌ | 是否返回模型的思考过程 | `false` |
| `temperature` | float | ❌ | 采样温度 | `0.3` |
| `top_p` | float | ❌ | 核采样参数 | `1.0` |
| `max_tokens` | int | ❌ | 最大生成 Token 数 | `null` |
| `timeout` | float | ❌ | 请求超时时间（秒） | `60.0` |
| `max_retries` | int | ❌ | 最大重试次数 | `3` |

#### 请求示例

```json
{
  "request_id": "qr_001",
  "query": "它续航怎么样？",
  "conversation_history": [
    {"role": "user", "content": "我想了解一下iPhone 15 Pro。"},
    {"role": "assistant", "content": "iPhone 15 Pro是苹果最新发布的旗舰手机..."}
  ],
  "domain_context": "数码产品导购",
  "max_rewrites": 3,
  "model": "deepseek-chat",
  "base_url": "https://api.deepseek.com/v1",
  "api_key": "your_api_key",
  "enable_thinking": true
}
```

---

### 非流式响应 (stream=false)

#### 响应参数 (`QueryRewriteResponse`)

| 参数名 | 类型 | 描述 |
| :--- | :--- | :--- |
| `output` | Dict | 改写结果，包含 `rewritten_queries` 列表 |
| `content` | string | 模型最终输出 |
| `reasoning_content` | string | 模型的思考/推理过程 |
| `metadata` | dict | 元数据，包含 token 使用情况等 |
| `confidence` | float | 置信度评分 (0.0 - 1.0) |

#### 响应示例

```json
{
  "output": {
    "original_query": "它续航怎么样？",
    "rewritten_queries": [
      "iPhone 15 Pro 电池续航时间",
      "iPhone 15 Pro 待机时长测评",
      "iPhone 15 Pro 实际使用续航表现"
    ],
    "strategy_used": "Coreference Resolution"
  },
  "reasoning_content": "用户问'它续航怎么样'，结合历史记录'iPhone 15 Pro'...",
  "metadata": {
    "usage": {
      "total_tokens": 180
    }
  },
  "confidence": 0.95
}
```

---

### 流式响应 (stream=true)

使用 Server-Sent Events (SSE) 协议返回。

- **响应类型**: `text/event-stream`

#### 事件类型说明

1. **start**: 任务开始
2. **processing**: 内部处理状态更新
3. **thinking**: 模型思考内容片段
4. **content**: 生成的改写内容片段
5. **final**: 最终结果
6. **end**: 任务结束
7. **error**: 发生错误

#### SSE 响应流示例

```text
data: {"type": "start", "content": "", "metadata": {"request_id": "qr_001", "status": "started"}}

data: {"type": "thinking", "content": "分析用户意图...", "metadata": null}

data: {"type": "content", "content": "iPhone 15 Pro 续航能力", "metadata": null}

data: {"type": "final", "content": "", "metadata": {"output": {...}, "confidence": 0.95}}

data: {"type": "end", "content": "", "metadata": {"request_id": "qr_001", "status": "completed"}}
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

url = "http://localhost:8003/query_rewriting/v1/chat"
payload = {
    "request_id": "test_qr",
    "query": "价格是多少",
    "conversation_history": [
        {"role": "user", "content": "帮我查一下去北京的机票"}
    ],
    "max_rewrites": 3,
    "model": "deepseek-chat",
    "base_url": "https://api.deepseek.com/v1",
    "api_key": "your_api_key"
}

response = requests.post(url, json=payload)
print("改写结果:", response.json()['output'])
```

---

## 5. 注意事项

1. **必填参数**: `model`、`base_url`、`api_key` 必须在每次请求中提供
2. **模型兼容**: 符合 OpenAI 接口规范的 LLM API 均可使用
3. **对话历史**: 提供对话历史可帮助进行指代消歧
4. **客户端断开**: 支持客户端断开时自动中断后端推理