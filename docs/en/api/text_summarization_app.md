# Text Summarization Service API Documentation

**Version**: 1.0.0  
**Description**: Text summarization service based on LangGraph + LLM, supporting multiple summary types (e.g., news, academic, meeting), word count control, streaming output, and Chain of Thought (Thinking) process.  
**Base URL**: `http://<host>:8005/text_summarization/v1`

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

### 2.2 Text Summarization

Unified summarization interface, controlled by the `stream` parameter for the return method.

- **Endpoint**: `/chat`
- **Method**: `POST`
- **Content-Type**: `application/json`

#### Request Parameters (`SummarizationRequest`)

| Parameter | Type | Required | Description | Default |
| :--- | :--- | :--- | :--- | :--- |
| `request_id` | string | ✅ | Request ID for full-link tracing | - |
| `text` | string | ✅ | Original text to be summarized | - |
| `model` | string | ✅ | Model name (e.g., deepseek-chat, gpt-4o) | - |
| `base_url` | string | ✅ | LLM API Base URL | - |
| `api_key` | string | ✅ | LLM API Key | - |
| `target_words` | int | ❌ | Target word count; None means no limit | `null` |
| `summary_type` | string | ❌ | Type of summary | `Point summary` |
| `ratio` | float | ❌ | Word count adjustment ratio (valid only in word count mode) | `1.5` |
| `stream` | bool | ❌ | Whether to enable streaming output | `false` |
| `enable_thinking` | bool | ❌ | Whether to enable/return the reasoning process | `false` |
| `temperature` | float | ❌ | Temperature parameter | `0.3` |
| `top_p` | float | ❌ | Top-p sampling parameter | `1.0` |
| `max_tokens` | int | ❌ | Maximum tokens | `null` |
| `timeout` | float | ❌ | Timeout (seconds) | `60.0` |
| `max_retries` | int | ❌ | Maximum retry attempts | `3` |

**`summary_type` Optional Values**:
- `Point summary`
- `Paragraph summary`
- `News summary`
- `Technical summary`
- `Meeting summary`
- `Academic summary`
- `Story summary`

#### Request Example

```json
{
  "request_id": "sum_12345",
  "text": "Here is a very long article about the development of Artificial Intelligence...",
  "target_words": 200,
  "summary_type": "News summary",
  "model": "deepseek-chat",
  "base_url": "https://api.deepseek.com/v1",
  "api_key": "your_api_key",
  "enable_thinking": true,
  "temperature": 0.3
}
```

---

### Non-streaming Response (stream=false)

#### Response Parameters (`SummarizationResponse`)

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `output` | object | Business result, including summary content |
| `content` | string | Final model output |
| `confidence` | float | Overall confidence (0.0 - 1.0) |
| `reasoning_content` | string | Model's reasoning/inference process |
| `metadata` | object | Metadata (e.g., token consumption) |

#### Response Example

```json
{
  "output": {
    "summary": "Artificial intelligence technology has made breakthrough progress in recent years...",
    "word_count": 198
  },
  "reasoning_content": "First analyze the core viewpoints of the article, then filter key information...",
  "confidence": 1.0,
  "metadata": {
    "usage": {
      "prompt_tokens": 500,
      "completion_tokens": 250
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
3. **content**: Summary content fragment
4. **final**: Final result
5. **end**: Task ended
6. **error**: Error occurred during processing

#### Response Stream Example

```text
data: {"type": "start", "content": "", "metadata": {"request_id": "sum_12345", "status": "started"}}

data: {"type": "content", "content": "This article mainly", "metadata": null}

data: {"type": "content", "content": " discusses...", "metadata": null}

data: {"type": "final", "content": "", "metadata": {"output": {...}, "confidence": 1.0}}

data: {"type": "end", "content": "", "metadata": {"request_id": "sum_12345", "status": "completed"}}
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

url = "http://localhost:8005/text_summarization/v1/chat"
payload = {
    "request_id": "sum_001",
    "text": "Here is a very long article about the development of Artificial Intelligence...",
    "target_words": 200,
    "summary_type": "News summary",
    "model": "deepseek-chat",
    "base_url": "https://api.deepseek.com/v1",
    "api_key": "your_api_key"
}

response = requests.post(url, json=payload)
result = response.json()
print("Summary:", result['output']['summary'])
print("Word Count:", result['output']['word_count'])
```

---

## 5. Notes

1. **Required Parameters**: `model`, `base_url`, and `api_key` must be provided in every request.
2. **Model Compatibility**: Any LLM API conforming to the OpenAI interface specification can be used.
3. **Word Count Control**: Control the output word count via the `target_words` and `ratio` parameters.
4. **Summary Type**: Different summary types will affect the output style and structure.
5. **Client Disconnection**: Supports automatic interruption of backend reasoning when the client disconnects.
