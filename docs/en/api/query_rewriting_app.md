# Query Rewriting Service API Documentation

**Version**: 1.0.0  
**Description**: Intelligent query rewriting service based on LangGraph + LLM, supporting strategies like coreference resolution, query expansion, and semantic enhancement.  
**Base URL**: `http://<host>:8003/query_rewriting/v1`

---

## 1. General Information

- **Protocol**: HTTP/1.1
- **Data Format**: JSON
- **Character Set**: UTF-8
- **Model Compatibility**: Any LLM API conforming to the OpenAI interface specification can be used.

**Key Strategies**:
- **Coreference Resolution**: Resolves pronouns based on conversation history.
- **Query Expansion**: Adds synonyms and related terms to increase recall.
- **Semantic Enhancement**: Completes omitted contextual information.
- **Format Adjustment**: Adjusts grammatical structure and expression perspective.

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

### 2.2 Query Rewriting

Unified rewriting interface, controlled by the `stream` parameter for the return method.

- **Endpoint**: `/chat`
- **Method**: `POST`
- **Content-Type**: `application/json`

#### Request Parameters (`QueryRewriteRequest`)

| Parameter | Type | Required | Description | Default |
| :--- | :--- | :--- | :--- | :--- |
| `request_id` | string | ✅ | Unique request identifier | - |
| `query` | string | ✅ | Original query text to be rewritten | - |
| `model` | string | ✅ | Model name (e.g., deepseek-chat, gpt-4o) | - |
| `base_url` | string | ✅ | LLM API Base URL | - |
| `api_key` | string | ✅ | LLM API Key | - |
| `conversation_history` | List[Dict] | ❌ | Conversation history for coreference resolution | `null` |
| `domain_context` | string | ❌ | Domain context information | `null` |
| `max_rewrites` | int | ❌ | Maximum number of rewritten versions expected | `5` |
| `preserve_system` | bool | ❌ | Whether to preserve the influence of system preset instructions | `true` |
| `stream` | bool | ❌ | Whether to enable streaming output | `false` |
| `enable_thinking` | bool | ❌ | Whether to return the model's reasoning process | `false` |
| `temperature` | float | ❌ | Sampling temperature | `0.3` |
| `top_p` | float | ❌ | Nucleus sampling parameter | `1.0` |
| `max_tokens` | int | ❌ | Maximum generated tokens | `null` |
| `timeout` | float | ❌ | Request timeout (seconds) | `60.0` |
| `max_retries` | int | ❌ | Maximum retry attempts | `3` |

#### Request Example

```json
{
  "request_id": "qr_001",
  "query": "How is its battery life?",
  "conversation_history": [
    {"role": "user", "content": "I want to know about the iPhone 15 Pro."},
    {"role": "assistant", "content": "The iPhone 15 Pro is the latest flagship phone released by Apple..."}
  ],
  "domain_context": "Digital Products Shopping Guide",
  "max_rewrites": 3,
  "model": "deepseek-chat",
  "base_url": "https://api.deepseek.com/v1",
  "api_key": "your_api_key",
  "enable_thinking": true
}
```

---

### Non-streaming Response (stream=false)

#### Response Parameters (`QueryRewriteResponse`)

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `output` | Dict | Rewriting result, including `rewritten_queries` list |
| `content` | string | Final model output |
| `reasoning_content` | string | Model's reasoning/inference process |
| `metadata` | dict | Metadata, including token usage, etc. |
| `confidence` | float | Confidence score (0.0 - 1.0) |

#### Response Example

```json
{
  "output": {
    "original_query": "How is its battery life?",
    "rewritten_queries": [
      "iPhone 15 Pro battery life duration",
      "iPhone 15 Pro standby time review",
      "iPhone 15 Pro real-world battery performance"
    ],
    "strategy_used": "Coreference Resolution"
  },
  "reasoning_content": "The user asked 'How is its battery life?', based on history 'iPhone 15 Pro'...",
  "metadata": {
    "usage": {
      "total_tokens": 180
    }
  },
  "confidence": 0.95
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
4. **content**: Generated rewriting content fragment
5. **final**: Final result
6. **end**: Task ended
7. **error**: Error occurred

#### SSE Response Stream Example

```text
data: {"type": "start", "content": "", "metadata": {"request_id": "qr_001", "status": "started"}}

data: {"type": "thinking", "content": "Analyzing user intent...", "metadata": null}

data: {"type": "content", "content": "iPhone 15 Pro battery capability", "metadata": null}

data: {"type": "final", "content": "", "metadata": {"output": {...}, "confidence": 0.95}}

data: {"type": "end", "content": "", "metadata": {"request_id": "qr_001", "status": "completed"}}
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

url = "http://localhost:8003/query_rewriting/v1/chat"
payload = {
    "request_id": "test_qr",
    "query": "What is the price?",
    "conversation_history": [
        {"role": "user", "content": "Check flight tickets to Beijing for me."}
    ],
    "max_rewrites": 3,
    "model": "deepseek-chat",
    "base_url": "https://api.deepseek.com/v1",
    "api_key": "your_api_key"
}

response = requests.post(url, json=payload)
print("Rewriting result:", response.json()['output'])
```

---

## 5. Notes

1. **Required Parameters**: `model`, `base_url`, and `api_key` must be provided in every request.
2. **Model Compatibility**: Any LLM API conforming to the OpenAI interface specification can be used.
3. **Conversation History**: Providing conversation history helps with coreference resolution.
4. **Client Disconnection**: Supports automatic interruption of backend reasoning when the client disconnects.
