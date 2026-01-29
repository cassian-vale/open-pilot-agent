# Information Extraction Service API Documentation

**Version**: 1.0.0  
**Description**: A structured information extraction service based on LangGraph + LLM, supporting data extraction from unstructured text according to a user-defined JSON Schema.  
**Base URL**: `http://<host>:8001/information_extraction/v1`

---

## 1. General Information

- **Protocol**: HTTP/1.1
- **Data Format**: JSON
- **Character Set**: UTF-8
- **Model Compatibility**: Any LLM API conforming to the OpenAI interface specification can be used.

---

## 2. Interface List

### 2.1 Health Check

Check whether the service instance and Agent initialization are completed.

- **Endpoint**: `/health`
- **Method**: `GET`

#### Response Example

```json
{
  "status": "OK",
  "agent": "initialized"
}
```

---

### 2.2 Information Extraction

Unified extraction interface, controlled by the `stream` parameter for the return method.

- **Endpoint**: `/chat`
- **Method**: `POST`
- **Content-Type**: `application/json`

#### Request Parameters (`InformationExtractionRequest`)

| Parameter | Type | Required | Description | Default |
| :--- | :--- | :--- | :--- | :--- |
| `request_id` | string | ✅ | Unique request identifier | - |
| `text` | string | ✅ | Original text content to be extracted | - |
| `schema` | dict | ✅ | JSON Schema defining the target extraction structure | - |
| `model` | string | ✅ | Model name (e.g., deepseek-chat, gpt-4o) | - |
| `base_url` | string | ✅ | LLM API Base URL | - |
| `api_key` | string | ✅ | LLM API Key | - |
| `stream` | bool | ❌ | Whether to enable streaming output | `false` |
| `max_tokens` | int | ❌ | Maximum generated tokens | `null` |
| `temperature` | float | ❌ | Sampling temperature, recommended to keep low | `0.1` |
| `top_p` | float | ❌ | Nucleus sampling parameter | `1.0` |
| `timeout` | float | ❌ | Request timeout (seconds) | `60.0` |
| `max_retries` | int | ❌ | Maximum retry attempts | `3` |
| `enable_thinking` | bool | ❌ | Whether to return the model's reasoning process | `false` |

#### Request Example

```json
{
  "request_id": "req_123456",
  "text": "Zhang San, male, 30 years old, phone number is 13800138000, lives in Chaoyang District, Beijing.",
  "schema": {
    "name": "Name",
    "age": "Age (int)",
    "phone": "Phone Number",
    "address": "Detailed Address"
  },
  "model": "deepseek-chat",
  "base_url": "https://api.deepseek.com/v1",
  "api_key": "your_api_key",
  "temperature": 0.1,
  "enable_thinking": true
}
```

---

### Non-streaming Response (stream=false)

#### Response Parameters (`InformationExtractionResponse`)

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `output` | dict | Structured data conforming to the Schema definition |
| `content` | string | Final model output |
| `reasoning_content` | string | Model's reasoning/inference process |
| `metadata` | dict | Metadata, including token usage, etc. |
| `confidence` | float | Confidence score (0.0 - 1.0) |

#### Response Example

```json
{
  "output": {
    "name": "Zhang San",
    "age": 30,
    "phone": "13800138000",
    "address": "Chaoyang District, Beijing"
  },
  "reasoning_content": "The user provided personal information text, I need to extract name, age, phone, and address...",
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

### Streaming Response (stream=true)

Returned using the Server-Sent Events (SSE) protocol.

- **Response Type**: `text/event-stream`

#### Event Type Descriptions

1. **start**: Task started
2. **processing**: Processing progress
3. **thinking**: Reasoning process (requires enable_thinking)
4. **content**: Content fragment
5. **final**: Final result
6. **end**: Task ended
7. **error**: Error occurred during processing

#### SSE Response Stream Example

```text
data: {"type": "start", "content": "", "metadata": {"request_id": "req_123", "status": "started"}}

data: {"type": "processing", "content": "Initializing information extraction task...", "metadata": null}

data: {"type": "thinking", "content": "Analyzing text...", "metadata": null}

data: {"type": "content", "content": "{ \"name\":", "metadata": null}

data: {"type": "content", "content": " \"Zhang San\" }", "metadata": null}

data: {"type": "final", "content": "", "metadata": {"output": {"name": "Zhang San"}, "confidence": 1.0}}

data: {"type": "end", "content": "", "metadata": {"request_id": "req_123", "status": "completed"}}
```

---

## 3. Schema Definition Format Specification

The `schema` parameter is a JSON object used to define the data structure you want to extract from the text.

### Supported Data Types

- **Basic Types**: `str`, `int`, `float`, `bool`
- **Complex Types**: `dict`, `list`

### Field Configuration Attributes

| Attribute | Type | Required | Description | Applicable Types |
| :--- | :--- | :--- | :--- | :--- |
| `type` | string | ✅ | Field data type | All |
| `description` | string | ❌ | Field meaning description | All |
| `required` | bool | ❌ | Whether it is mandatory | All |
| `properties` | dict | ❌ | Field structure for nested objects | `dict` only |
| `item_type` | string | ❌ | Type of list elements | `list` only |
| `item_properties` | dict | ❌ | Internal structure when list elements are objects | `list` only |

### Schema Example

```json
{
  "customer_info": {
    "type": "dict",
    "description": "Basic customer information",
    "required": true,
    "properties": {
      "name": { "type": "str", "description": "Customer name", "required": true },
      "vip_member": { "type": "bool", "description": "Whether a VIP member" }
    }
  },
  "products": {
    "type": "list",
    "description": "Purchased product list",
    "item_type": "dict",
    "item_properties": {
      "name": { "type": "str", "description": "Product name", "required": true },
      "count": { "type": "int", "description": "Quantity" }
    }
  }
}
```

---

## 4. Error Codes

| Status Code | Description |
|--------|------|
| `200` | Request successful |
| `400` | Input parameter error or Schema validation failure |
| `499` | Client disconnected |
| `500` | Internal server error |
| `503` | Agent service not yet initialized |

---

## 5. Client Example (Python)

```python
import requests

url = "http://localhost:8001/information_extraction/v1/chat"
payload = {
    "request_id": "test_001",
    "text": "Order two iPhone 15s, ship to Pudong New Area, Shanghai.",
    "schema": {
        "product": "Product Name",
        "quantity": "Quantity (int)",
        "location": "Shipping Address"
    },
    "model": "deepseek-chat",
    "base_url": "https://api.deepseek.com/v1",
    "api_key": "your_api_key"
}

response = requests.post(url, json=payload)
print(response.json())
```

---

## 6. Notes

1. **Required Parameters**: `model`, `base_url`, and `api_key` must be provided in every request.
2. **Model Compatibility**: Any LLM API conforming to the OpenAI interface specification can be used.
3. **Schema Design**: Proper Schema design can improve extraction accuracy.
4. **Temperature Suggestion**: Low temperature (0.0-0.2) is recommended for information extraction tasks.
5. **Client Disconnection**: Supports automatic interruption of backend reasoning when the client disconnects.
