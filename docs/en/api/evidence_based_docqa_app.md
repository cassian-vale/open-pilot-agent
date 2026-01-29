# Document Question Answering Service API Documentation

## Overview

A structured document question-answering service based on LangGraph + LLM, supporting both streaming and non-streaming outputs. The service provides evidence-based document QA functionality, capable of processing long text documents and returning accurate answers.

## Basic Information

- **Root Path**: `/evidence_based_docQA/v1`
- **Version**: 1.0.0
- **Default Port**: 8000

---

## Interface List

### 1. Health Check Interface

**Endpoint**: `GET /health`

**Function**: Check the service running status and Agent initialization status

**Response**:
```json
{
  "status": "OK",
  "agent": "initialized"
}
```

**Error Status**:
- `503 Service Unavailable`: Agent not initialized

---

### 2. Document QA Interface

**Endpoint**: `POST /chat`

**Function**: Unified document QA interface, controlled by the `stream` parameter for the return method

**Request Body** (`DocQARequest`):

| Field | Type | Required | Default | Description |
|------|------|------|--------|------|
| `request_id` | string | ✅ | - | Unique request identifier |
| `doc_text` | string | ✅ | - | Document text content |
| `query` | string | ✅ | - | User query question |
| `model` | string | ✅ | - | Model name (e.g., deepseek-chat, gpt-4o) |
| `base_url` | string | ✅ | - | API Base URL |
| `api_key` | string | ✅ | - | API Key |
| `stream` | boolean | ❌ | `false` | Whether to enable streaming output |
| `max_tokens` | integer | ❌ | `null` | Maximum tokens |
| `temperature` | float | ❌ | `0.0` | Temperature parameter (0.0-2.0) |
| `top_p` | float | ❌ | `1.0` | Top-p parameter (0.0-1.0) |
| `timeout` | float | ❌ | `60.0` | Timeout (seconds) |
| `max_retries` | integer | ❌ | `3` | Maximum retry attempts |
| `enable_thinking` | boolean | ❌ | `false` | Whether to enable reasoning process |
| `chunk_size` | integer | ❌ | `512` | Text chunk size |
| `overlap` | integer | ❌ | `100` | Chunk overlap size |
| `return_sentences` | boolean | ❌ | `true` | Whether to return sentences |

---

### Non-streaming Response (stream=false)

**Response Body** (`DocQAResponse`):

| Field | Type | Description |
|------|------|------|
| `output` | object | QA output result |
| `content` | string | Final model output |
| `reasoning_content` | string | Reasoning process content |
| `metadata` | object | Metadata information |
| `confidence` | float | Confidence score (0.0-1.0) |

**Example Request**:
```bash
curl -X POST "http://localhost:8000/evidence_based_docQA/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "req_123",
    "doc_text": "Here is the document content...",
    "query": "What is the document mainly about?",
    "model": "deepseek-chat",
    "base_url": "https://api.deepseek.com/v1",
    "api_key": "your_api_key",
    "temperature": 0.1,
    "enable_thinking": true
  }'
```

---

### Streaming Response (stream=true)

Returned using the Server-Sent Events (SSE) protocol.

**Response Format**: `text/event-stream`

**Event Types**:

1. **Start Event** (`type: "start"`)
   ```json
   {"type": "start", "content": "", "metadata": {"request_id": "Request ID", "status": "started"}}
   ```

2. **Reasoning Event** (`type: "thinking"`)
   ```json
   {"type": "thinking", "content": "Reasoning content fragment", "metadata": {...}}
   ```

3. **Content Event** (`type: "content"`)
   ```json
   {"type": "content", "content": "Answer content fragment", "metadata": {...}}
   ```

4. **Final Event** (`type: "final"`)
   ```json
   {"type": "final", "content": "", "metadata": {"output": {...}, "confidence": 0.95}}
   ```

5. **End Event** (`type: "end"`)
   ```json
   {"type": "end", "content": "", "metadata": {"request_id": "Request ID", "status": "completed"}}
   ```

6. **Error Event** (`type: "error"`)
   ```json
   {"type": "error", "content": "Error message"}
   ```

---

## Client Example (Python)

### Non-streaming Call

```python
import requests

url = "http://localhost:8000/evidence_based_docQA/v1/chat"
payload = {
    "request_id": "test_001",
    "doc_text": "Artificial Intelligence is a branch of computer science...",
    "query": "What is Artificial Intelligence?",
    "model": "deepseek-chat",
    "base_url": "https://api.deepseek.com/v1",
    "api_key": "your_api_key"
}

response = requests.post(url, json=payload)
print(response.json())
```

### Streaming Call

```python
import requests
import json

url = "http://localhost:8000/evidence_based_docQA/v1/chat"
payload = {
    "request_id": "test_002",
    "doc_text": "Artificial Intelligence is a branch of computer science...",
    "query": "What is Artificial Intelligence?",
    "model": "deepseek-chat",
    "base_url": "https://api.deepseek.com/v1",
    "api_key": "your_api_key",
    "stream": True
}

with requests.post(url, json=payload, stream=True) as response:
    for line in response.iter_lines():
        if line:
            line_text = line.decode('utf-8')
            if line_text.startswith('data: '):
                data = json.loads(line_text[6:])
                print(f"[{data['type']}] {data.get('content', '')}")
```

---

## Error Handling

| Status Code | Description |
|--------|------|
| `200` | Request successful |
| `400` | Request parameter error |
| `499` | Client disconnected |
| `503` | Service not ready (Agent not initialized) |
| `500` | Internal server error |

---

## Starting the Service

```bash
# Run directly
python doc_qa_app.py

# Using Uvicorn
uvicorn doc_qa_app:app --host 0.0.0.0 --port 8000
```

---

## Notes

1. **Model Compatibility**: Any LLM API conforming to the OpenAI interface specification can be used.
2. **Parameter Passing**: `model`, `base_url`, and `api_key` are required parameters and must be provided in every request.
3. **Timeout Setting**: Set the timeout reasonably based on document length and complexity.
4. **Chunking Parameters**: For long documents, appropriately adjusting `chunk_size` and `overlap` parameters can improve QA quality.
5. **Reasoning Process**: Enabling `enable_thinking` provides the model's reasoning process but may increase response time.
