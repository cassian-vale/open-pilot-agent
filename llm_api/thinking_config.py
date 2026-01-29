from typing import Any, Dict


class ThinkingConfig(object):
    """思维链配置管理"""

    def __init__(self):
        self.model_type_thinking_params = {
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
            }
        }

    def get_thinking_params(self, model_name: str, enable_thinking: bool) -> Dict[str, Any]:
        """获取思维链参数"""
        model_type = self.get_model_type(model_name)
        key = "enable_thinking" if enable_thinking else "disable_thinking"
        params = self.model_type_thinking_params.get(model_type, {})
        return params.get(key, {})

    @staticmethod
    def get_model_type(model_name: str) -> str:
        """根据模型名称获取模型类型"""
        if model_name.startswith("glm"):
            return "glm"
        elif model_name.startswith("deepseek"):
            return "deepseek"
        elif model_name.startswith("qwen"):
            return "qwen"
        return ""
