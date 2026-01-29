# Schema 生成服务 API 文档

**版本**: 1.0.0  
**描述**: 基于 LangGraph + LLM 的 JSON Schema 生成服务，根据自然语言需求自动生成符合规范的 Schema 结构定义。  
**Base URL**: `http://<host>:8004/schema_generation/v1`

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

### 2.2 Schema 生成

统一生成接口，通过 `stream` 参数控制返回方式。

- **接口地址**: `/chat`
- **请求方式**: `POST`
- **Content-Type**: `application/json`

#### 请求参数 (`SchemaGenerationRequest`)

| 参数名 | 类型 | 必选 | 描述 | 默认值 |
| :--- | :--- | :--- | :--- | :--- |
| `request_id` | string | ✅ | 请求的唯一标识符 | - |
| `user_requirements` | string | ✅ | 用户对 Schema 的自然语言需求描述 | - |
| `model` | string | ✅ | 模型名称 (如 deepseek-chat, gpt-4o) | - |
| `base_url` | string | ✅ | LLM API 基础 URL | - |
| `api_key` | string | ✅ | LLM API 密钥 | - |
| `domain_context` | string | ❌ | 领域上下文信息 | `null` |
| `stream` | bool | ❌ | 是否启用流式输出 | `false` |
| `enable_thinking` | bool | ❌ | 是否返回模型的思考过程 | `false` |
| `temperature` | float | ❌ | 采样温度，建议保持较低值 | `0.1` |
| `top_p` | float | ❌ | 核采样参数 | `1.0` |
| `max_tokens` | int | ❌ | 最大生成 Token 数 | `null` |
| `timeout` | float | ❌ | 请求超时时间（秒） | `60.0` |
| `max_retries` | int | ❌ | 最大重试次数 | `3` |

#### 请求示例

```json
{
  "request_id": "sg_req_001",
  "user_requirements": "我们需要一个员工档案的数据结构。包含姓名（必填）、工号（字符串）、年龄、入职日期。另外还需要一个列表记录他的过往项目经历，每个项目包含项目名和担任角色。",
  "domain_context": "企业HR系统",
  "model": "deepseek-chat",
  "base_url": "https://api.deepseek.com/v1",
  "api_key": "your_api_key",
  "enable_thinking": true
}
```

---

### 非流式响应 (stream=false)

#### 响应参数 (`SchemaGenerationResponse`)

| 参数名 | 类型 | 描述 |
| :--- | :--- | :--- |
| `output` | Dict | 生成的 JSON Schema 对象 |
| `content` | string | 模型最终输出 |
| `reasoning_content` | string | 模型的思考/推理过程 |
| `metadata` | dict | 元数据，包含 token 使用情况等 |
| `confidence` | float | 置信度评分 (0.0 - 1.0) |

#### 响应示例

```json
{
  "output": {
    "type": "object",
    "properties": {
      "name": {
        "type": "string",
        "description": "员工姓名"
      },
      "employee_id": {
        "type": "string",
        "description": "工号"
      },
      "age": {
        "type": "integer",
        "description": "年龄"
      },
      "hire_date": {
        "type": "string",
        "format": "date",
        "description": "入职日期"
      },
      "projects": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "project_name": { "type": "string", "description": "项目名称" },
            "role": { "type": "string", "description": "担任角色" }
          },
          "required": ["project_name"]
        },
        "description": "过往项目经历"
      }
    },
    "required": ["name", "employee_id"]
  },
  "reasoning_content": "用户需要HR系统的员工档案Schema...",
  "metadata": {
    "usage": {
      "total_tokens": 350
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
2. **processing**: 内部处理状态更新
3. **thinking**: 模型思考内容片段
4. **content**: 生成的 Schema 文本片段
5. **final**: 最终完整的 Schema 对象
6. **end**: 任务结束
7. **error**: 发生错误

#### SSE 响应流示例

```text
data: {"type": "start", "content": "", "metadata": {"request_id": "sg_req_001", "status": "started"}}

data: {"type": "thinking", "content": "设计员工字段...", "metadata": null}

data: {"type": "content", "content": "{ \"type\": \"object\",", "metadata": null}

data: {"type": "final", "content": "", "metadata": {"output": { "type": "object", ... }, "confidence": 1.0}}

data: {"type": "end", "content": "", "metadata": {"request_id": "sg_req_001", "status": "completed"}}
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
import json

url = "http://localhost:8004/schema_generation/v1/chat"
payload = {
    "request_id": "test_schema",
    "user_requirements": "创建一个包含书名、作者、出版年份和价格的书籍对象。",
    "domain_context": "图书馆管理",
    "model": "deepseek-chat",
    "base_url": "https://api.deepseek.com/v1",
    "api_key": "your_api_key"
}

response = requests.post(url, json=payload)
result = response.json()
print("生成的Schema:")
print(json.dumps(result['output'], indent=2, ensure_ascii=False))
```

---

## 5. 注意事项

1. **必填参数**: `model`、`base_url`、`api_key` 必须在每次请求中提供
2. **模型兼容**: 符合 OpenAI 接口规范的 LLM API 均可使用
3. **需求描述**: 详细的需求描述可生成更准确的 Schema
4. **温度建议**: Schema 生成建议使用低温度 (0.0-0.2) 以保证结构稳定性
5. **客户端断开**: 支持客户端断开时自动中断后端推理