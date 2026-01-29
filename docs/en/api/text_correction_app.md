# Text Correction Service API Documentation

**Version**: 1.0.0  
**Description**: Text correction service based on LangGraph + LLM, supporting automatic chunking for long text, streaming output, and Chain of Thought (Thinking) process.  
**Base URL**: `http://<host>:8007/text_correction/v1`

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

### 2.2 Text Correction

Unified correction interface, controlled by the `stream` parameter for the return method.

- **Endpoint**: `/chat`
- **Method**: `POST`
- **Content-Type**: `application/json`

#### Request Parameters (`TextCorrectionRequest`)

| Parameter | Type | Required | Description | Default |
| :--- | :--- | :--- | :--- | :--- |
| `request_id` | string | ✅ | Request ID for full-link tracing | - |
| `text` | string | ✅ | Original text content to be corrected | - |
| `model` | string | ✅ | Model name (e.g., deepseek-chat, gpt-4o) | - |
| `base_url` | string | ✅ | LLM API Base URL | - |
| `api_key` | string | ✅ | LLM API Key | - |
| `max_chunk_length` | int | ❌ | Text chunk length (for long text processing) | `512` |
| `stream` | bool | ❌ | Whether to enable streaming output | `false` |
| `enable_thinking` | bool | ❌ | Whether to enable/return the reasoning process | `false` |
| `temperature` | float | ❌ | Temperature parameter (recommended as 0 for correction) | `0.0` |
| `top_p` | float | ❌ | Top-p sampling parameter | `1.0` |
| `max_tokens` | int | ❌ | Maximum tokens | `null` |
| `timeout` | float | ❌ | Timeout (seconds) | `60.0` |
| `max_retries` | int | ❌ | Maximum retry attempts | `3` |

#### Request Example

```json
{
  "request_id": "corr_001",
  "text": "This projet is moving very fast, we ned to follow up immedately.",
  "model": "deepseek-chat",
  "base_url": "https://api.deepseek.com/v1",
  "api_key": "your_api_key",
  "max_chunk_length": 512,
  "temperature": 0.0,
  "enable_thinking": false
}
```

---

### Non-streaming Response (stream=false)

#### Response Parameters (`TextCorrectionResponse`)

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `output` | object | Business result, including `corrected_text` and `corrections` |
| `content` | string | Final model output |
| `confidence` | float | Overall confidence (0.0 - 1.0) |
| `reasoning_content` | string | Model's reasoning/inference process |
| `metadata` | object | Metadata (e.g., token consumption) |

#### Response Example

```json
{
  "output": {
    "corrected_text": "This project is moving very fast, we need to follow up immediately.",
    "corrections": [
      {
        "original": "projet",
        "corrected": "project",
        "type": "typo",
        "position": 5
      },
      {
        "original": "ned",
        "corrected": "need",
        "type": "typo",
        "position": 35
      },
      {
        "original": "immedately",
        "corrected": "immediately",
        "type": "typo",
        "position": 53
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

### Streaming Response (stream=true)

Returned using the Server-Sent Events (SSE) protocol.

- **Response Type**: `text/event-stream`

#### Event Type Descriptions

1. **start**: Task started
2. **thinking**: Reasoning process (requires enable_thinking)
3. **content**: Correction content fragment
4. **final**: Final result
5. **end**: Task ended
6. **error**: Error occurred during processing

#### Response Stream Example

```text
data: {"type": "start", "content": "", "metadata": {"request_id": "corr_001", "status": "started"}}

data: {"type": "content", "content": "This project", "metadata": null}

data: {"type": "content", "content": " is moving", "metadata": null}

data: {"type": "final", "content": "", "metadata": {"output": {...}, "confidence": 1.0}}

data: {"type": "end", "content": "", "metadata": {"request_id": "corr_001", "status": "completed"}}
```

---

## 3. Error Codes

| Status Code | Description |
|--------|------|
| `200` | Request successful |
| `400` | Parameter validation failed |
| `422` | Data format validation error |
| `499` | Client disconnected |
| `500` | Internal server error |
| `503` | Agent service not yet initialized |

---

## 4. Client Example (Python)

```python
import requests

url = "http://localhost:8007/text_correction/v1/chat"
payload = {
    "request_id": "corr_001",
    "text": "This projet is moving very fast, we ned to follow up immedately.",
    "model": "deepseek-chat",
    "base_url": "https://api.deepseek.com/v1",
    "api_key": "your_api_key"
}

response = requests.post(url, json=payload)
result = response.json()
print("Corrected Text:", result['output']['corrected_text'])
print("Errors List:", result['output']['corrections'])
```

---

## 5. Notes

1. **Required Parameters**: `model`, `base_url`, and `api_key` must be provided in every request.
2. **Model Compatibility**: Any LLM API conforming to the OpenAI interface specification can be used.
3. **Temperature Suggestion**: Temperature 0 is recommended for correction tasks to ensure accuracy.
4. **Long Text Processing**: Automatically handles long text through chunking with the `max_chunk_length` parameter.
5. **Client Disconnection**: Supports automatic interruption of backend reasoning when the client disconnects.
