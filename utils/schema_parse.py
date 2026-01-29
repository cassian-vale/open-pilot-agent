import json
import re
import traceback
from typing import Type, Dict, Any, Optional
from pydantic import BaseModel, Field


class SchemaParser(object):
    """
    Pydantic Schema 解析工具类
    用于生成JSON Schema提示词和解析LLM响应
    """
    def __init__(self, model_class: Optional[Type[BaseModel]] = None):
        self.type_mapping = {
            'string': "",
            'integer': 0,
            'number': 0.0,
            'boolean': False,
            'array': [],
            'object': {}
        }

        self.model_class = model_class
        self.schema = self._get_schema()
        self.schema_generation_prompt = self._generate_prompt()

    def _get_schema(self) -> Dict[str, Any]:
        """获取schema，支持model_class为None的情况"""
        if self.model_class is None:
            return {}
        return self.model_class.model_json_schema()
        
    def _generate_prompt(self) -> str:
        """生成提示词模板"""
        if not self.schema:
            return "请输出有效的JSON格式数据"
            
        return f"""
请严格按照给定的JSON schema的格式输出结果，不要包含任何其他内容：

JSON schema:
{json.dumps(self.schema, ensure_ascii=False)}

输出要求：
1. 必须直接输出被markdown代码块标记的有效的JSON格式
2. 不要输出其他任何解释性文字
3. 确保所有字段的类型和格式正确

示例格式：
```json
{json.dumps(self.get_example_output(), ensure_ascii=False)}
```
"""
    
    def get_example_output(self) -> Dict[str, Any]:
        """
        根据JSON Schema生成示例输出

        Returns:
            Dict[str, Any]: 示例数据
        """
        if not self.schema:
            return {}
            
        example_data = {}
        properties = self.schema.get('properties', {})

        for field_name, field_info in properties.items():
            example_data[field_name] = self._get_field_example_value(field_info)

        return example_data

    def _get_field_example_value(self, field_info: Dict[str, Any]) -> Any:
        """
        根据字段信息生成示例值

        Args:
            field_info: 字段信息字典

        Returns:
            Any: 示例值
        """
        field_type = field_info.get('type')
        return self.type_mapping.get(field_type, None)

    def parse_response_to_json(self, content: str) -> dict:
        """
        解析LLM响应为指定的BaseModel对象

        Args:
            response: LLM响应对象

        Returns:
            T: dict
        """

        # 尝试多种解析策略
        matching_strategies = [
            # 策略1: 提取markdown JSON代码块 (最常用)
            lambda c: re.search(r'```(?:json)?\s*(\{.*?\})\s*```', c, re.DOTALL).group(1),

            # 策略2: 提取markdown代码块 (无语言标识)
            lambda c: re.search(r'```\s*(\{.*?\})\s*```', c, re.DOTALL).group(1),

            # 策略3: 直接解析整个内容
            lambda c: c
        ]

        parsed_data = {}
        last_error = None

        for strategy in matching_strategies:
            try:
                match_data = strategy(content)
                parsed_data = json.loads(match_data)
                # 验证解析结果是否为字典
                if isinstance(parsed_data, dict):
                    break
            except Exception as e:
                last_error = e
                continue

        return parsed_data

    def parse_response_to_base_model(self, content: str) -> Optional[BaseModel]:
        """
        解析响应为BaseModel对象
        
        Returns:
            Optional[BaseModel]: 当model_class为None时返回None
        """
        if self.model_class is None:
            return None
            
        parsed_data = self.parse_response_to_json(content)
        try:
            model = self.model_class(**parsed_data)
        except Exception as e:
            # 创建空对象作为fallback
            parsed_data = {}
            model = self.model_class(**parsed_data)
            print(traceback.format_exception(e))
        return model

    def update_model_class(self, model_class: Type[BaseModel]) -> None:
        """
        更新model_class并重新生成相关配置
        
        Args:
            model_class: 新的Pydantic模型类
        """
        self.model_class = model_class
        self.schema = self._get_schema()
        self.schema_generation_prompt = self._generate_prompt()