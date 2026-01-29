# Text Classification Service API Documentation

**Version**: 1.0.0  
**Description**: Text classification service based on LangGraph + LLM, supporting custom labels, confidence output, and Chain of Thought (Thinking) process.  
**Base URL**: `http://<host>:8006/text_classification/v1`

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

### 2.2 Text Classification

Unified classification interface, controlled by the `stream` parameter for the return method.

- **Endpoint**: `/chat`
- **Method**: `POST`
- **Content-Type**: `application/json`

#### Request Parameters (`TextClassificationRequest`)

| Parameter | Type | Required | Description | Default |
| :--- | :--- | :--- | :--- | :--- |
| `request_id` | string | ✅ | Request ID for full-link tracing | - |
| `text` | string | ✅ | Original text content to be classified | - |
| `candidate_labels` | list[str] | ✅ | List of candidate labels (minimum 2, maximum 20) | - |
| `model` | string | ✅ | Model name (e.g., deepseek-chat, gpt-4o) | - |
| `base_url` | string | ✅ | LLM API Base URL | - |
| `api_key` | string | ✅ | LLM API Key | - |
| `stream` | bool | ❌ | Whether to enable streaming output | `false` |
| `enable_thinking` | bool | ❌ | Whether to enable/return the reasoning process | `false` |
| `temperature` | float | ❌ | Temperature parameter (recommended low for stability) | `0.1` |
| `top_p` | float | ❌ | Top-p sampling parameter | `1.0` |
| `max_tokens` | int | ❌ | Maximum tokens | `10` |
| `timeout` | float | ❌ | Timeout (seconds) | `60.0` |
| `max_retries` | int | ❌ | Maximum retry attempts | `3` |

#### Request Example

```json
{
  "request_id": "req_123456789",
  "text": "The battery life of this phone is terrible; it ran out of power in just half a day.",
  "candidate_labels": ["Positive Review", "Negative Review", "Neutral Review"],
  "model": "deepseek-chat",
  "base_url": "https://api.deepseek.com/v1",
  "api_key": "your_api_key",
  "temperature": 0.1,
  "enable_thinking": true
}
```

---

### Non-streaming Response (stream=false)

#### Response Parameters (`TextClassificationResponse`)

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `output` | object | Business result, including `predicted_label`, etc. |
| `content` | string | Final model output |
| `confidence` | float | Overall confidence (0.0 - 1.0) |
| `reasoning_content` | string | Model's reasoning/inference process |
| `metadata` | object | Metadata (e.g., token consumption) |

#### Response Example

```json
{
  "output": {
    "predicted_label": "Negative Review"
  },
  "content": "Negative Review",
  "reasoning_content": "The user mentioned terrible battery life, expressing dissatisfaction...",
  "confidence": 0.9,
  "metadata": {
    "usage": {
      "prompt_tokens": 150,
      "completion_tokens": 20
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
3. **content**: Classification result fragment
4. **final**: Final result
5. **end**: Task ended
6. **error**: Error occurred during processing

#### Response Stream Example

```text
data: {"type": "start", "content": "", "metadata": {"request_id": "req_123", "status": "started"}}

data: {"type": "content", "content": "Negative Review", "metadata": null}

data: {"type": "final", "content": "", "metadata": {"output": {"predicted_label": "Negative Review"}, "confidence": 0.9}}

data: {"type": "end", "content": "", "metadata": {"request_id": "req_123", "status": "completed"}}
```

---

## 3. Error Codes

| Status Code | Description |
|--------|------|
| `200` | Request successful |
| `400` | Parameter validation failed (e.g., less than 2 labels) |
| `422` | Data format validation error |
| `499` | Client disconnected |
| `500` | Internal server error |
| `503` | Agent service not yet initialized |

---

## 4. Client Example (Python)

```python
import requests

url = "http://localhost:8006/text_classification/v1/chat"
payload = {
    "request_id": "req_123",
    "text": "The battery life of this phone is terrible; it ran out of power in just half a day.",
    "candidate_labels": ["Positive Review", "Negative Review", "Neutral Review"],
    "model": "deepseek-chat",
    "base_url": "https://api.deepseek.com/v1",
    "api_key": "your_api_key"
}

response = requests.post(url, json=payload)
print(response.json())
```

---

## 5. Notes

1. **Required Parameters**: `model`, `base_url`, and `api_key` must be provided in every request.
2. **Model Compatibility**: Any LLM API conforming to the OpenAI interface specification can be used.
3. **Number of Labels**: At least 2 candidate labels are required, with a maximum of 20.
4. **Temperature Suggestion**: Low temperature (0.0-0.2) is recommended for classification tasks to ensure stability.
5. **Client Disconnection**: Supports automatic interruption of backend reasoning when the client disconnects.
