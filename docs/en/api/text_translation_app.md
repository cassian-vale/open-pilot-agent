# Text Translation Service API Documentation

**Version**: 1.0.0  
**Description**: Text translation service based on LangGraph + LLM, supporting multiple translation styles (e.g., governmental, academic, social media style), multi-language directions, streaming output, and Chain of Thought (Thinking) process.  
**Base URL**: `http://<host>:8008/text_translation/v1`

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

### 2.2 Text Translation

Unified translation interface, controlled by the `stream` parameter for the return method.

- **Endpoint**: `/chat`
- **Method**: `POST`
- **Content-Type**: `application/json`

#### Request Parameters (`TranslationRequest`)

| Parameter | Type | Required | Description | Default |
| :--- | :--- | :--- | :--- | :--- |
| `request_id` | string | ✅ | Request ID for full-link tracing | - |
| `text` | string | ✅ | Original text to be translated | - |
| `translation_direction` | string | ✅ | Translation direction (e.g., "Chinese to English", "English to Chinese") | - |
| `model` | string | ✅ | Model name (e.g., deepseek-chat, gpt-4o) | - |
| `base_url` | string | ✅ | LLM API Base URL | - |
| `api_key` | string | ✅ | LLM API Key | - |
| `translation_style` | string | ❌ | Translation style | `Normal` |
| `stream` | bool | ❌ | Whether to enable streaming output | `false` |
| `enable_thinking` | bool | ❌ | Whether to enable/return the reasoning process | `false` |
| `temperature` | float | ❌ | Temperature parameter | `0.3` |
| `top_p` | float | ❌ | Top-p sampling parameter | `1.0` |
| `max_tokens` | int | ❌ | Maximum tokens | `null` |
| `timeout` | float | ❌ | Timeout (seconds) | `60.0` |
| `max_retries` | int | ❌ | Maximum retry attempts | `3` |

**Common `translation_style` Examples**:
- `Normal`: Standard translation
- `Governmental`: Formal and rigorous
- `Academic`: Professional terminology, academic standards
- `Social Media`: Lively, rich in Emojis, colloquial (e.g., "Xiaohongshu style")
- `Sci-Fi`: Futuristic, technological feel
- `Forum`: Internet slang, down-to-earth
- `Script`: Dialogue format, sense of scenario

#### Request Example

```json
{
  "request_id": "trans_001",
  "text": "人工智能将彻底改变我们的生活方式。",
  "translation_direction": "Chinese to English",
  "translation_style": "Academic",
  "model": "deepseek-chat",
  "base_url": "https://api.deepseek.com/v1",
  "api_key": "your_api_key",
  "temperature": 0.3,
  "enable_thinking": true
}
```

---

### Non-streaming Response (stream=false)

#### Response Parameters (`TranslationResponse`)

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `output` | object | Business result, including `translated_text` and `quality_score` |
| `content` | string | Final model output |
| `confidence` | float | Overall confidence (0.0 - 1.0) |
| `reasoning_content` | string | Model's reasoning/inference process |
| `metadata` | object | Metadata (e.g., token consumption) |

#### Response Example

```json
{
  "output": {
    "translated_text": "Artificial Intelligence constitutes a paradigm shift that will fundamentally restructure our mode of existence.",
    "quality_score": 95
  },
  "reasoning_content": "The user requested an academic style. The original meaning is AI changing life. Academic expressions can use words like 'paradigm shift', 'restructure', 'mode of existence'...",
  "confidence": 0.95,
  "metadata": {
    "usage": {
      "prompt_tokens": 100,
      "completion_tokens": 50
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
3. **content**: Translation content fragment
4. **final**: Final result
5. **end**: Task ended
6. **error**: Error occurred during processing

#### Response Stream Example

```text
data: {"type": "start", "content": "", "metadata": {"request_id": "trans_001", "status": "started"}}

data: {"type": "content", "content": "Artificial", "metadata": null}

data: {"type": "content", "content": " Intelligence", "metadata": null}

data: {"type": "final", "content": "", "metadata": {"output": {...}, "confidence": 0.95}}

data: {"type": "end", "content": "", "metadata": {"request_id": "trans_001", "status": "completed"}}
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

url = "http://localhost:8008/text_translation/v1/chat"
payload = {
    "request_id": "trans_001",
    "text": "人工智能将彻底改变我们的生活方式。",
    "translation_direction": "Chinese to English",
    "translation_style": "Academic",
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
3. **Translation Style**: Choose an appropriate translation style based on the application scenario.
4. **Client Disconnection**: Supports automatic interruption of backend reasoning when the client disconnects.
