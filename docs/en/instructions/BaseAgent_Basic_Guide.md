# BaseAgent Application Basic Guide

## 1. Overview

`BaseAgent` is the core abstract base class of Open Pilot Agent, providing a unified LLM calling interface and configuration management mechanism. All specific Agent applications should inherit from this class.

### Core Features

- **Unified LLM Client Management**: Encapsulates `LLMClient`, supporting multiple models.
- **Runtime Configuration Override**: Supports flexible combinations of initialization and runtime configurations.
- **Synchronous/Asynchronous Calls**: Full support for both synchronous and asynchronous LLM calls.
- **Streaming Output**: Supports streaming response processing.
- **Thinking Mode**: Supports reasoning process output for models like DeepSeek.

---

## 2. Quick Start

### 2.1 Creating a Custom Agent

```python
from base_agent import BaseAgent
from typing import Dict, Any

class MyAgent(BaseAgent):
    """Custom Agent Example"""
    
    def __init__(self, **kwargs):
        super().__init__(name="my-agent", **kwargs)
        # Initialize custom components
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build LangGraph graph (Must be implemented)"""
        # Return None for simple scenarios
        return None
    
    def run(self, text: str, **kwargs) -> Dict[str, Any]:
        """Execute Agent logic (Must be implemented)"""
        messages = [{"role": "user", "content": text}]
        response = self.call_llm(messages, runtime_config=kwargs)
        return {"output": response.choices[0].message.content}
```

### 2.2 Using the Agent

```python
# Method 1: Provide complete configuration during initialization
agent = MyAgent(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key="your_api_key"
)
result = agent.run("Hello")

# Method 2: Provide configuration at runtime (Recommended for open-source scenarios)
agent = MyAgent()  # No configuration provided
result = agent.run(
    "Hello",
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key="your_api_key"
)
```

---

## 3. Configuration Parameters

### 3.1 Initialization Parameters

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `name` | str | "base-agent" | Agent name, used for logging identification |
| `model` | str | "deepseek-chat" | Model name |
| `base_url` | str | "https://api.deepseek.com/v1" | API Base URL |
| `api_key` | str | None | API Key |
| `max_tokens` | int | None | Maximum output tokens |
| `temperature` | float | 0.0 | Temperature parameter (0.0-2.0) |
| `top_p` | float | 1.0 | Nucleus sampling parameter |
| `timeout` | float | 60.0 | Request timeout (seconds) |
| `max_retries` | int | 3 | Maximum retry attempts |
| `stream` | bool | False | Whether to enable streaming output |
| `enable_thinking` | bool | False | Whether to enable thinking mode |

### 3.2 Configuration Priority

```
Runtime Configuration > Initialization Configuration > Environment Variables
```

---

## 4. LLM Calling Methods

### 4.1 Synchronous Calls

```python
# Non-streaming
response = agent.call_llm(messages, runtime_config=config)
content = response.choices[0].message.content

# Streaming
for chunk in agent.stream_llm(messages):
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="")
```

### 4.2 Asynchronous Calls

```python
# Non-streaming
response = await agent.acall_llm(messages, runtime_config=config)

# Streaming
async for chunk in agent.astream_llm(messages):
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="")
```

---

## 5. Complete Example: Text Summarization Agent

```python
from base_agent import BaseAgent
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage

class SummarizationAgent(BaseAgent):
    """Text Summarization Agent"""
    
    def __init__(self, max_words: int = 200, **kwargs):
        super().__init__(name="summarization-agent", **kwargs)
        self.max_words = max_words
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build processing graph"""
        graph = StateGraph(dict)
        
        def summarize_node(state: dict) -> dict:
            """Summarization node"""
            text = state["text"]
            runtime_config = state.get("runtime_config", {})
            
            messages = [
                {"role": "system", "content": f"Please summarize the following text into no more than {self.max_words} words"},
                {"role": "user", "content": text}
            ]
            
            response = self.call_llm(messages, runtime_config=runtime_config)
            summary = response.choices[0].message.content
            
            return {**state, "summary": summary}
        
        graph.add_node("summarize", summarize_node)
        graph.add_edge(START, "summarize")
        graph.add_edge("summarize", END)
        
        return graph.compile()
    
    def run(self, text: str, **kwargs) -> Dict[str, Any]:
        """Execute summarization"""
        result = self.graph.invoke({
            "text": text,
            "runtime_config": kwargs
        })
        return {
            "summary": result["summary"],
            "original_length": len(text),
            "summary_length": len(result["summary"])
        }

# Usage Example
agent = SummarizationAgent(max_words=100)
result = agent.run(
    "This is a very long article...",
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key="your_api_key"
)
print(result["summary"])
```

---

## 6. Thinking Mode

### 6.1 What is Thinking Mode?

Thinking Mode allows the model to output its reasoning process, helping users understand how the model reached its conclusion. This is very useful in scenarios requiring transparency and explainability.

### 6.2 Applicable Scenarios

| Scenario | Recommended? | Description |
|------|----------|------|
| Complex Reasoning Tasks | ✅ Yes | Mathematical reasoning, logical analysis, code debugging |
| QA Systems | ✅ Yes | Needs to show reasoning process to enhance credibility |
| Simple Text Generation | ❌ No | Simple tasks like translation or summarization don't need it |
| High-concurrency Scenarios | ⚠️ Use with care | Reasoning process increases token consumption and response time |
| Classification Tasks | ❌ No | Simple classification doesn't require complex reasoning |

### 6.3 Supported Models

Currently supported model types (Configuration file: `llm_api/thinking_config.py`):

| Model Type | Model Example | Thinking Parameter |
|----------|----------|----------|
| **DeepSeek** | `deepseek-reasoner` | No extra parameters needed |
| **GLM** | `glm-4-plus` | `{"thinking": {"type": "enabled"}}` |
| **Qwen** | `qwen-plus` | `{"enable_thinking": True}` |

> 📌 **Note**: Currently only adapted for **OpenAI-compatible format** API interfaces. To use APIs in other formats (like Anthropic native interface), you need to extend `LLMClient` yourself.

### 6.4 Basic Usage

```python
# Enable thinking mode
agent = MyAgent(enable_thinking=True)

# Or enable at runtime
result = agent.run(text, enable_thinking=True)

# Get thinking mode status
status = agent.get_thinking_status()
print(status)  # {"enable_thinking": True, "thinking_params": {...}}

# Toggle dynamically
agent.enable_thinking_mode(True)   # Enable
agent.enable_thinking_mode(False)  # Disable
```

### 6.5 Adding/Modifying Model Thinking Configurations

To support new models, please modify `llm_api/thinking_config.py`:

```python
# File: llm_api/thinking_config.py

class ThinkingConfig(object):
    def __init__(self):
        self.model_type_thinking_params = {
            # Existing configurations
            "glm": {
                "enable_thinking": {"thinking": {"type": "enabled"}},
                "disable_thinking": {"thinking": {"type": "disabled"}}
            },
            "deepseek": {
                "enable_thinking": {},
                "disable_thinking": {}
            },
            "qwen": {
                "enable_thinking": {"enable_thinking": True},
                "disable_thinking": {"enable_thinking": False}
            },
            # Example for adding a new model configuration
            "new_model": {
                "enable_thinking": {"custom_param": "value"},
                "disable_thinking": {"custom_param": "disabled"}
            }
        }

    @staticmethod
    def get_model_type(model_name: str) -> str:
        """Get model type based on model name"""
        if model_name.startswith("glm"):
            return "glm"
        elif model_name.startswith("deepseek"):
            return "deepseek"
        elif model_name.startswith("qwen"):
            return "qwen"
        # New model type matching
        elif model_name.startswith("new_model"):
            return "new_model"
        return ""
```

**Steps**:
1. Add the new model's configuration in the `model_type_thinking_params` dictionary.
2. Add the model name matching rule in the `get_model_type` method.

> ⚠️ **Important Reminder**: Before adding a new model, please consult the official documentation of the corresponding API to understand:
> - Whether the model supports thinking mode.
> - What specific parameters are needed to enable thinking mode.
> - How parameters are passed (e.g., `extra_body`, `headers`, etc.).
> - Whether a specific model version is required (e.g., DeepSeek requires `deepseek-reasoner`).

---

## 7. Configuration Status Check

```python
# Check if LLM is configured
if agent.is_llm_configured:
    print("LLM is configured")

# Get detailed configuration status
status = agent.get_config_status()
print(status)
# {
#     "llm_configured": True,
#     "init_config": {
#         "model": "deepseek-chat",
#         "api_key": "***",  # Desensitized
#         ...
#     }
# }
```

---

## 8. Best Practices

### 8.1 Inheritance Norms

```python
class MyAgent(BaseAgent):
    def __init__(self, custom_param: str, **kwargs):
        # 1. Call parent class initialization first
        super().__init__(name="my-agent", **kwargs)
        
        # 2. Save custom parameters
        self.custom_param = custom_param
        
        # 3. Finally, build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self):
        # Must be implemented
        pass
    
    def run(self, **kwargs):
        # Must be implemented
        pass
```

### 8.2 Runtime Configuration Passing

```python
def run(self, text: str, **kwargs) -> Dict[str, Any]:
    # Extract LLM configuration from kwargs
    runtime_config = {
        "model": kwargs.get("model"),
        "base_url": kwargs.get("base_url"),
        "api_key": kwargs.get("api_key"),
        "temperature": kwargs.get("temperature"),
        # ... other configurations
    }
    # Remove None values
    runtime_config = {k: v for k, v in runtime_config.items() if v is not None}
    
    # Call LLM with configuration
    response = self.call_llm(messages, runtime_config=runtime_config)
```

### 8.3 Error Handling

```python
from typing import Dict, Any

def run(self, **kwargs) -> Dict[str, Any]:
    try:
        response = self.call_llm(messages, runtime_config=kwargs)
        return {"success": True, "output": response.choices[0].message.content}
    except ValueError as e:
        self.logger.error(f"Configuration error: {e}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        self.logger.error(f"Call failed: {e}")
        return {"success": False, "error": str(e)}
```

---

## 9. Next Steps

- Read [BaseAgent Advanced Guide](./BaseAgent_Advanced_Guide.md) to learn about deep LangGraph integration.
- Refer to the complete implementation examples in the `applications/` directory.
