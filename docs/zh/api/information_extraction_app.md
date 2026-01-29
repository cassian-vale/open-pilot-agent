# 信息抽取服务 API 文档

**版本**: 1.0.0  
**描述**: 基于 LangGraph + LLM 的结构化信息抽取服务，支持根据用户定义的 JSON Schema 从非结构化文本中提取数据。  
**Base URL**: `http://<host>:8001/information_extraction/v1`

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

### 2.2 信息抽取

统一抽取接口，通过 `stream` 参数控制返回方式。

- **接口地址**: `/chat`
- **请求方式**: `POST`
- **Content-Type**: `application/json`

#### 请求参数 (`InformationExtractionRequest`)

| 参数名 | 类型 | 必选 | 描述 | 默认值 |
| :--- | :--- | :--- | :--- | :--- |
| `request_id` | string | ✅ | 请求的唯一标识符 | - |
| `text` | string | ✅ | 待抽取的原始文本内容 | - |
| `schema` | dict | ✅ | 定义目标提取结构的 JSON Schema | - |
| `model` | string | ✅ | 模型名称 (如 deepseek-chat, gpt-4o) | - |
| `base_url` | string | ✅ | LLM API 基础 URL | - |
| `api_key` | string | ✅ | LLM API 密钥 | - |
| `stream` | bool | ❌ | 是否启用流式输出 | `false` |
| `max_tokens` | int | ❌ | 最大生成 Token 数 | `null` |
| `temperature` | float | ❌ | 采样温度，建议保持较低值 | `0.1` |
| `top_p` | float | ❌ | 核采样参数 | `1.0` |
| `timeout` | float | ❌ | 请求超时时间（秒） | `60.0` |
| `max_retries` | int | ❌ | 最大重试次数 | `3` |
| `enable_thinking` | bool | ❌ | 是否返回模型的思考过程 | `false` |

#### 请求示例

```json
{
  "request_id": "req_123456",
  "text": "张三，男，30岁，电话是13800138000，住在北京市朝阳区。",
  "schema": {
    "name": "姓名",
    "age": "年龄 (int)",
    "phone": "手机号",
    "address": "详细地址"
  },
  "model": "deepseek-chat",
  "base_url": "https://api.deepseek.com/v1",
  "api_key": "your_api_key",
  "temperature": 0.1,
  "enable_thinking": true
}
```

---

### 非流式响应 (stream=false)

#### 响应参数 (`InformationExtractionResponse`)

| 参数名 | 类型 | 描述 |
| :--- | :--- | :--- |
| `output` | dict | 符合 Schema 定义的结构化数据 |
| `content` | string | 模型最终输出 |
| `reasoning_content` | string | 模型的思考/推理过程 |
| `metadata` | dict | 元数据，包含 token 使用情况等 |
| `confidence` | float | 置信度评分 (0.0 - 1.0) |

#### 响应示例

```json
{
  "output": {
    "name": "张三",
    "age": 30,
    "phone": "13800138000",
    "address": "北京市朝阳区"
  },
  "reasoning_content": "用户提供了个人信息文本，我需要提取姓名、年龄、电话和地址...",
  "metadata": {
    "usage": {
      "prompt_tokens": 150,
      "completion_tokens": 45
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
2. **processing**: 处理进度
3. **thinking**: 思考过程 (需开启 enable_thinking)
4. **content**: 内容片段
5. **final**: 最终结果
6. **end**: 任务结束
7. **error**: 处理发生错误

#### SSE 响应流示例

```text
data: {"type": "start", "content": "", "metadata": {"request_id": "req_123", "status": "started"}}

data: {"type": "processing", "content": "开始初始化信息抽取任务...", "metadata": null}

data: {"type": "thinking", "content": "分析文本...", "metadata": null}

data: {"type": "content", "content": "{ \"name\":", "metadata": null}

data: {"type": "content", "content": " \"张三\" }", "metadata": null}

data: {"type": "final", "content": "", "metadata": {"output": {"name": "张三"}, "confidence": 1.0}}

data: {"type": "end", "content": "", "metadata": {"request_id": "req_123", "status": "completed"}}
```

---

## 3. Schema 定义格式说明

`schema` 参数是一个 JSON 对象，用于定义希望从文本中抽取的数据结构。

### 支持的数据类型

- **基础类型**: `str`, `int`, `float`, `bool`
- **复合类型**: `dict`, `list`

### 字段配置属性

| 属性名 | 类型 | 必填 | 描述 | 适用类型 |
| :--- | :--- | :--- | :--- | :--- |
| `type` | string | ✅ | 字段数据类型 | 所有 |
| `description` | string | ❌ | 字段含义描述 | 所有 |
| `required` | bool | ❌ | 是否为必填项 | 所有 |
| `properties` | dict | ❌ | 嵌套对象的字段结构 | 仅 `dict` |
| `item_type` | string | ❌ | 列表元素的类型 | 仅 `list` |
| `item_properties` | dict | ❌ | 列表元素为对象时的内部结构 | 仅 `list` |

### Schema 示例

```json
{
  "customer_info": {
    "type": "dict",
    "description": "客户基本信息",
    "required": true,
    "properties": {
      "name": { "type": "str", "description": "客户姓名", "required": true },
      "vip_member": { "type": "bool", "description": "是否为VIP会员" }
    }
  },
  "products": {
    "type": "list",
    "description": "购买的产品清单",
    "item_type": "dict",
    "item_properties": {
      "name": { "type": "str", "description": "产品名称", "required": true },
      "count": { "type": "int", "description": "数量" }
    }
  }
}
```

---

## 4. 错误码说明

| 状态码 | 描述 |
|--------|------|
| `200` | 请求成功 |
| `400` | 输入参数错误或 Schema 校验失败 |
| `499` | 客户端断开连接 |
| `500` | 服务器内部错误 |
| `503` | Agent 服务尚未初始化 |

---

## 5. 客户端示例 (Python)

```python
import requests

url = "http://localhost:8001/information_extraction/v1/chat"
payload = {
    "request_id": "test_001",
    "text": "订购两台iPhone 15，寄到上海市浦东新区。",
    "schema": {
        "product": "产品名称",
        "quantity": "数量(int)",
        "location": "收货地址"
    },
    "model": "deepseek-chat",
    "base_url": "https://api.deepseek.com/v1",
    "api_key": "your_api_key"
}

response = requests.post(url, json=payload)
print(response.json())
```

---

## 6. 注意事项

1. **必填参数**: `model`、`base_url`、`api_key` 必须在每次请求中提供
2. **模型兼容**: 符合 OpenAI 接口规范的 LLM API 均可使用
3. **Schema 设计**: 合理设计 Schema 可提高抽取准确性
4. **温度建议**: 信息抽取任务建议使用低温度 (0.0-0.2)
5. **客户端断开**: 支持客户端断开时自动中断后端推理