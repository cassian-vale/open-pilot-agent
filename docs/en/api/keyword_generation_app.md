# Keyword Generation Service API Documentation

**Version**: 1.0.0  
**Description**: Intelligent keyword extraction and generation service based on LangGraph + LLM, supporting domain context and streaming output.  
**Base URL**: `http://<host>:8002/keyword_generation/v1`

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

### 2.2 Keyword Generation

Unified generation interface, controlled by the `stream` parameter for the return method.

- **Endpoint**: `/chat`
- **Method**: `POST`
- **Content-Type**: `application/json`

#### Request Parameters (`KeywordGenerationRequest`)

| Parameter | Type | Required | Description | Default |
| :--- | :--- | :--- | :--- | :--- |
| `request_id` | string | ✅ | Unique request identifier | - |
| `content` | string | ✅ | Original text content for keyword extraction | - |
| `model` | string | ✅ | Model name (e.g., deepseek-chat, gpt-4o) | - |
| `base_url` | string | ✅ | LLM API Base URL | - |
| `api_key` | string | ✅ | LLM API Key | - |
| `domain_context` | string | ❌ | Domain context information (e.g., "Medical", "E-commerce") | `null` |
| `max_keywords` | int | ❌ | Maximum number of keywords expected | `10` |
| `stream` | bool | ❌ | Whether to enable streaming output | `false` |
| `max_tokens` | int | ❌ | Maximum generated tokens | `null` |
| `temperature` | float | ❌ | Sampling temperature | `0.1` |
| `top_p` | float | ❌ | Nucleus sampling parameter | `1.0` |
| `timeout` | float | ❌ | Request timeout (seconds) | `60.0` |
| `max_retries` | int | ❌ | Maximum retry attempts | `3` |
| `enable_thinking` | bool | ❌ | Whether to return the model's reasoning process | `false` |

#### Request Example

```json
{
  "request_id": "kg_001",
  "content": "Deep learning is a sub-field of machine learning, based on learning algorithms of artificial neural networks. It has achieved significant results in fields like image recognition and natural language processing.",
  "domain_context": "Artificial Intelligence/Computer Science",
  "max_keywords": 5,
  "model": "deepseek-chat",
  "base_url": "https://api.deepseek.com/v1",
  "api_key": "your_api_key",
  "enable_thinking": true
}
```

---

### Non-streaming Response (stream=false)

#### Response Parameters (`KeywordGenerationResponse`)

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `output` | List[str] | List of generated keyword strings |
| `content` | string | Final model output |
| `reasoning_content` | string | Model's reasoning/inference process |
| `metadata` | dict | Metadata, including token usage, etc. |
| `confidence` | float | Confidence score (0.0 - 1.0) |

#### Response Example

```json
{
  "output": [
    "Deep Learning",
    "Machine Learning",
    "Artificial Neural Networks",
    "Image Recognition",
    "Natural Language Processing"
  ],
  "reasoning_content": "The text provided by the user is mainly about deep learning and its applications...",
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

### Streaming Response (stream=true)

Returned using the Server-Sent Events (SSE) protocol.

- **Response Type**: `text/event-stream`

#### Event Type Descriptions

1. **start**: Task started
2. **processing**: Processing status update
3. **thinking**: Model reasoning content fragment
4. **content**: Generated content fragment
5. **final**: Final complete result
6. **end**: Task ended
7. **error**: Error occurred

#### SSE Response Stream Example

```text
data: {"type": "start", "content": "", "metadata": {"request_id": "kg_001", "status": "started"}}

data: {"type": "thinking", "content": "Analyzing text keywords...", "metadata": null}

data: {"type": "content", "content": "Deep Learning", "metadata": null}

data: {"type": "final", "content": "", "metadata": {"output": ["Deep Learning", "Machine Learning"], "confidence": 1.0}}

data: {"type": "end", "content": "", "metadata": {"request_id": "kg_001", "status": "completed"}}
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

url = "http://localhost:8002/keyword_generation/v1/chat"
payload = {
    "request_id": "test_kg",
    "content": "With the popularity of new energy vehicles, battery recycling technology has become a focus of industry attention.",
    "domain_context": "New Energy/Environmental Protection",
    "max_keywords": 5,
    "model": "deepseek-chat",
    "base_url": "https://api.deepseek.com/v1",
    "api_key": "your_api_key"
}

response = requests.post(url, json=payload)
print("Generated Keywords:", response.json()['output'])
```

---

## 5. Notes

1. **Required Parameters**: `model`, `base_url`, and `api_key` must be provided in every request.
2. **Model Compatibility**: Any LLM API conforming to the OpenAI interface specification can be used.
3. **Domain Context**: Providing domain context can improve the accuracy of keyword extraction.
4. **Client Disconnection**: Supports automatic interruption of backend reasoning when the client disconnects.
