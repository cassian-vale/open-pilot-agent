# BaseAgent 应用基础指南

## 1. 概述

`BaseAgent` 是 Open Pilot Agent 的核心抽象基类，提供了统一的 LLM 调用接口和配置管理机制。所有具体的 Agent 应用都应该继承自此类。

### 核心特性

- **统一的 LLM 客户端管理**: 封装 LLMClient，支持多种模型
- **运行时配置覆盖**: 支持初始化配置和运行时配置的灵活组合
- **同步/异步调用**: 完整支持同步和异步的 LLM 调用
- **流式输出**: 支持流式响应处理
- **思考模式**: 支持 DeepSeek 等模型的思考过程输出

---

## 2. 快速开始

### 2.1 创建自定义 Agent

```python
from base_agent import BaseAgent
from typing import Dict, Any

class MyAgent(BaseAgent):
    """自定义 Agent 示例"""
    
    def __init__(self, **kwargs):
        super().__init__(name="my-agent", **kwargs)
        # 初始化自定义组件
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """构建 LangGraph 图（必须实现）"""
        # 简单场景可以返回 None
        return None
    
    def run(self, text: str, **kwargs) -> Dict[str, Any]:
        """执行 Agent 逻辑（必须实现）"""
        messages = [{"role": "user", "content": text}]
        response = self.call_llm(messages, runtime_config=kwargs)
        return {"output": response.choices[0].message.content}
```

### 2.2 使用 Agent

```python
# 方式1：初始化时提供完整配置
agent = MyAgent(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key="your_api_key"
)
result = agent.run("你好")

# 方式2：运行时提供配置（推荐用于开源场景）
agent = MyAgent()  # 不提供配置
result = agent.run(
    "你好",
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key="your_api_key"
)
```

---

## 3. 配置参数说明

### 3.1 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | str | "base-agent" | Agent 名称，用于日志标识 |
| `model` | str | "deepseek-chat" | 模型名称 |
| `base_url` | str | "https://api.deepseek.com/v1" | API 基础 URL |
| `api_key` | str | None | API 密钥 |
| `max_tokens` | int | None | 最大输出 token 数 |
| `temperature` | float | 0.0 | 温度参数 (0.0-2.0) |
| `top_p` | float | 1.0 | 核采样参数 |
| `timeout` | float | 60.0 | 请求超时时间（秒） |
| `max_retries` | int | 3 | 最大重试次数 |
| `stream` | bool | False | 是否启用流式输出 |
| `enable_thinking` | bool | False | 是否启用思考模式 |

### 3.2 配置优先级

```
运行时配置 > 初始化配置 > 环境变量
```

---

## 4. LLM 调用方法

### 4.1 同步调用

```python
# 非流式
response = agent.call_llm(messages, runtime_config=config)
content = response.choices[0].message.content

# 流式
for chunk in agent.stream_llm(messages):
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="")
```

### 4.2 异步调用

```python
# 非流式
response = await agent.acall_llm(messages, runtime_config=config)

# 流式
async for chunk in agent.astream_llm(messages):
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="")
```

---

## 5. 完整示例：文本摘要 Agent

```python
from base_agent import BaseAgent
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage

class SummarizationAgent(BaseAgent):
    """文本摘要 Agent"""
    
    def __init__(self, max_words: int = 200, **kwargs):
        super().__init__(name="summarization-agent", **kwargs)
        self.max_words = max_words
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """构建处理图"""
        graph = StateGraph(dict)
        
        def summarize_node(state: dict) -> dict:
            """摘要节点"""
            text = state["text"]
            runtime_config = state.get("runtime_config", {})
            
            messages = [
                {"role": "system", "content": f"请将以下文本摘要为不超过{self.max_words}字的内容"},
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
        """执行摘要"""
        result = self.graph.invoke({
            "text": text,
            "runtime_config": kwargs
        })
        return {
            "summary": result["summary"],
            "original_length": len(text),
            "summary_length": len(result["summary"])
        }

# 使用示例
agent = SummarizationAgent(max_words=100)
result = agent.run(
    "这是一篇很长的文章...",
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key="your_api_key"
)
print(result["summary"])
```

---

## 6. 思考模式

### 6.1 什么是思考模式

思考模式（Thinking Mode）允许模型输出其推理过程，帮助用户理解模型如何得出结论。这对于需要透明度和可解释性的场景非常有用。

### 6.2 适用场景

| 场景 | 是否推荐 | 说明 |
|------|----------|------|
| 复杂推理任务 | ✅ 推荐 | 数学推理、逻辑分析、代码调试 |
| 问答系统 | ✅ 推荐 | 需要展示推理过程以增强可信度 |
| 简单文本生成 | ❌ 不推荐 | 翻译、摘要等简单任务不需要 |
| 高并发场景 | ⚠️ 慎用 | 思考过程会增加 token 消耗和响应时间 |
| 分类任务 | ❌ 不推荐 | 简单分类不需要复杂推理 |

### 6.3 支持的模型

当前支持的模型类型（配置文件：`llm_api/thinking_config.py`）：

| 模型类型 | 模型示例 | 思考参数 |
|----------|----------|----------|
| **DeepSeek** | `deepseek-reasoner` | 无需额外参数 |
| **GLM** | `glm-4-plus` | `{"thinking": {"type": "enabled"}}` |
| **Qwen** | `qwen-plus` | `{"enable_thinking": True}` |

> 📌 **注意**：当前仅适配了 **OpenAI 兼容格式** 的 API 接口。如需使用其他格式的 API（如 Anthropic 原生接口），需要自行扩展 `LLMClient`。

### 6.4 基本使用

```python
# 启用思考模式
agent = MyAgent(enable_thinking=True)

# 或运行时启用
result = agent.run(text, enable_thinking=True)

# 获取思考模式状态
status = agent.get_thinking_status()
print(status)  # {"enable_thinking": True, "thinking_params": {...}}

# 动态切换
agent.enable_thinking_mode(True)   # 启用
agent.enable_thinking_mode(False)  # 禁用
```

### 6.5 新增/修改模型思考配置

如需支持新模型，请修改 `llm_api/thinking_config.py`：

```python
# 文件: llm_api/thinking_config.py

class ThinkingConfig(object):
    def __init__(self):
        self.model_type_thinking_params = {
            # 已有配置
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
            # 新增模型配置示例
            "new_model": {
                "enable_thinking": {"custom_param": "value"},
                "disable_thinking": {"custom_param": "disabled"}
            }
        }

    @staticmethod
    def get_model_type(model_name: str) -> str:
        """根据模型名称获取模型类型"""
        if model_name.startswith("glm"):
            return "glm"
        elif model_name.startswith("deepseek"):
            return "deepseek"
        elif model_name.startswith("qwen"):
            return "qwen"
        # 新增模型类型匹配
        elif model_name.startswith("new_model"):
            return "new_model"
        return ""
```

**步骤说明**：
1. 在 `model_type_thinking_params` 字典中添加新模型的配置
2. 在 `get_model_type` 方法中添加模型名称匹配规则

> ⚠️ **重要提醒**：添加新模型前，请先查阅对应 API 的官方文档，了解：
> - 该模型是否支持思考模式
> - 启用思考模式需要哪些特定参数
> - 参数的传递方式（如 extra_body、headers 等）
> - 是否需要使用特定的模型版本（如 DeepSeek 需使用 `deepseek-reasoner`）

---

## 7. 配置状态检查

```python
# 检查 LLM 是否已配置
if agent.is_llm_configured:
    print("LLM 已配置")

# 获取详细配置状态
status = agent.get_config_status()
print(status)
# {
#     "llm_configured": True,
#     "init_config": {
#         "model": "deepseek-chat",
#         "api_key": "***",  # 已脱敏
#         ...
#     }
# }
```

---

## 8. 最佳实践

### 8.1 继承规范

```python
class MyAgent(BaseAgent):
    def __init__(self, custom_param: str, **kwargs):
        # 1. 先调用父类初始化
        super().__init__(name="my-agent", **kwargs)
        
        # 2. 保存自定义配置
        self.custom_param = custom_param
        
        # 3. 最后构建图
        self.graph = self._build_graph()
    
    def _build_graph(self):
        # 必须实现
        pass
    
    def run(self, **kwargs):
        # 必须实现
        pass
```

### 8.2 运行时配置传递

```python
def run(self, text: str, **kwargs) -> Dict[str, Any]:
    # 从 kwargs 提取 LLM 配置
    runtime_config = {
        "model": kwargs.get("model"),
        "base_url": kwargs.get("base_url"),
        "api_key": kwargs.get("api_key"),
        "temperature": kwargs.get("temperature"),
        # ... 其他配置
    }
    # 移除 None 值
    runtime_config = {k: v for k, v in runtime_config.items() if v is not None}
    
    # 使用配置调用 LLM
    response = self.call_llm(messages, runtime_config=runtime_config)
```

### 8.3 错误处理

```python
from typing import Dict, Any

def run(self, **kwargs) -> Dict[str, Any]:
    try:
        response = self.call_llm(messages, runtime_config=kwargs)
        return {"success": True, "output": response.choices[0].message.content}
    except ValueError as e:
        self.logger.error(f"配置错误: {e}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        self.logger.error(f"调用失败: {e}")
        return {"success": False, "error": str(e)}
```

---

## 9. 下一步

- 阅读 [BaseAgent 进阶指南](./BaseAgent应用进阶指南.md) 了解 LangGraph 深度集成
- 参考 `applications/` 目录下的完整实现示例
