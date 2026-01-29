# Schema Generation Service API Documentation

**Version**: 1.0.0  
**Description**: JSON Schema generation service based on LangGraph + LLM, automatically generating compliant Schema structure definitions based on natural language requirements.  
**Base URL**: `http://<host>:8004/schema_generation/v1`

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

### 2.2 Schema Generation

Unified generation interface, controlled by the `stream` parameter for the return method.

- **Endpoint**: `/chat`
- **Method**: `POST`
- **Content-Type**: `application/json`

#### Request Parameters (`SchemaGenerationRequest`)

| Parameter | Type | Required | Description | Default |
| :--- | :--- | :--- | :--- | :--- |
| `request_id` | string | ✅ | Unique request identifier | - |
| `user_requirements` | string | ✅ | User's natural language description of Schema requirements | - |
| `model` | string | ✅ | Model name (e.g., deepseek-chat, gpt-4o) | - |
| `base_url` | string | ✅ | LLM API Base URL | - |
| `api_key` | string | ✅ | LLM API Key | - |
| `domain_context` | string | ❌ | Domain context information | `null` |
| `stream` | bool | ❌ | Whether to enable streaming output | `false` |
| `enable_thinking` | bool | ❌ | Whether to return the model's reasoning process | `false` |
| `temperature` | float | ❌ | Sampling temperature, recommended to keep low | `0.1` |
| `top_p` | float | ❌ | Nucleus sampling parameter | `1.0` |
| `max_tokens` | int | ❌ | Maximum generated tokens | `null` |
| `timeout` | float | ❌ | Request timeout (seconds) | `60.0` |
| `max_retries` | int | ❌ | Maximum retry attempts | `3` |

#### Request Example

```json
{
  "request_id": "sg_req_001",
  "user_requirements": "We need an employee profile data structure. It should include Name (required), Employee ID (string), Age, and Hire Date. Also, we need a list to record their past project experiences, where each project includes project name and role.",
  "domain_context": "Enterprise HR System",
  "model": "deepseek-chat",
  "base_url": "https://api.deepseek.com/v1",
  "api_key": "your_api_key",
  "enable_thinking": true
}
```

---

### Non-streaming Response (stream=false)

#### Response Parameters (`SchemaGenerationResponse`)

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `output` | Dict | Generated JSON Schema object |
| `content` | string | Final model output |
| `reasoning_content` | string | Model's reasoning/inference process |
| `metadata` | dict | Metadata, including token usage, etc. |
| `confidence` | float | Confidence score (0.0 - 1.0) |

#### Response Example

```json
{
  "output": {
    "type": "object",
    "properties": {
      "name": {
        "type": "string",
        "description": "Employee Name"
      },
      "employee_id": {
        "type": "string",
        "description": "Employee ID"
      },
      "age": {
        "type": "integer",
        "description": "Age"
      },
      "hire_date": {
        "type": "string",
        "format": "date",
        "description": "Hire Date"
      },
      "projects": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "project_name": { "type": "string", "description": "Project Name" },
            "role": { "type": "string", "description": "Role" }
          },
          "required": ["project_name"]
        },
        "description": "Past project experiences"
      }
    },
    "required": ["name", "employee_id"]
  },
  "reasoning_content": "User needs an employee profile Schema for an HR system...",
  "metadata": {
    "usage": {
      "total_tokens": 350
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
2. **processing**: Internal processing status update
3. **thinking**: Model reasoning content fragment
4. **content**: Generated Schema text fragment
5. **final**: Final complete Schema object
6. **end**: Task ended
7. **error**: Error occurred

#### SSE Response Stream Example

```text
data: {"type": "start", "content": "", "metadata": {"request_id": "sg_req_001", "status": "started"}}

data: {"type": "thinking", "content": "Designing employee fields...", "metadata": null}

data: {"type": "content", "content": "{ \"type\": \"object\",", "metadata": null}

data: {"type": "final", "content": "", "metadata": {"output": { "type": "object", ... }, "confidence": 1.0}}

data: {"type": "end", "content": "", "metadata": {"request_id": "sg_req_001", "status": "completed"}}
```

---

## 3. Error Codes

| Status Code | Description |
|--------|------|
| `200` | Request successful |
| `400` | Input parameter error |
| `499` | Client disconnected |
| `500` | Internal server error |
| `503` | Agent service not yet initialized |

---

## 4. Client Example (Python)

```python
import requests
import json

url = "http://localhost:8004/schema_generation/v1/chat"
payload = {
    "request_id": "test_schema",
    "user_requirements": "Create a book object containing title, author, publication year, and price.",
    "domain_context": "Library Management",
    "model": "deepseek-chat",
    "base_url": "https://api.deepseek.com/v1",
    "api_key": "your_api_key"
}

response = requests.post(url, json=payload)
result = response.json()
print("Generated Schema:")
print(json.dumps(result['output'], indent=2, ensure_ascii=False))
```

---

## 5. Notes

1. **Required Parameters**: `model`, `base_url`, and `api_key` must be provided in every request.
2. **Model Compatibility**: Any LLM API conforming to the OpenAI interface specification can be used.
3. **Requirement Description**: Detailed requirement descriptions lead to more accurate Schemas.
4. **Temperature Suggestion**: Low temperature (0.0-0.2) is recommended for Schema generation to ensure structural stability.
5. **Client Disconnection**: Supports automatic interruption of backend reasoning when the client disconnects.
