from typing import Dict, Any, List, Type, Union, get_origin, get_args, Optional
from pydantic import BaseModel, create_model, validator, Field
import json
from enum import Enum


class SchemaValidator(object):
    """Schema验证和工具类"""

    # 支持的Python类型映射
    SUPPORTED_TYPES = {"str", "int", "float", "bool", "list", "dict"}

    # 基础类型到Python类型的映射
    TYPE_MAPPING = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": List,
        "dict": Dict
    }

    @classmethod
    def validate_schema(cls, schema: Dict[str, Any], strict: bool = False) -> List[str]:
        """验证Schema格式是否正确
        Args:
            schema: 要验证的schema
            strict: 是否严格模式，严格模式下description和required为必需字段
        """
        errors = []

        if not isinstance(schema, dict):
            return ["Schema必须是一个字典"]

        for field_name, field_config in schema.items():
            errors.extend(cls._validate_field_config(field_name, field_config, strict=strict))

        return errors

    @classmethod
    def _validate_field_config(cls, field_name: str, config: Dict[str, Any], path: str = "", strict: bool = False) -> List[str]:
        """验证单个字段配置"""
        errors = []
        current_path = f"{path}.{field_name}" if path else field_name

        # 检查必需字段
        if "type" not in config:
            errors.append(f"{current_path} 缺少 'type' 字段")
            return errors

        # 检查类型是否支持
        field_type = config["type"]
        if field_type not in cls.SUPPORTED_TYPES:
            errors.append(f"{current_path} 不支持的类型: {field_type}")
            return errors

        # 检查description（在严格模式下为必需）
        if strict and "description" not in config:
            errors.append(f"{current_path} 缺少 'description' 字段")
        elif "description" not in config:
            # 非严格模式下，自动生成一个默认描述
            config["description"] = f"{field_name}字段"

        # 检查required字段（在严格模式下为必需）
        if strict and "required" not in config:
            errors.append(f"{current_path} 缺少 'required' 字段")
        elif "required" not in config:
            # 非严格模式下，默认为False
            config["required"] = False
        elif not isinstance(config["required"], bool):
            errors.append(f"{current_path} 'required' 必须是布尔值")

        # 递归验证嵌套结构
        if field_type == "dict" and "properties" in config:
            if not isinstance(config["properties"], dict):
                errors.append(f"{current_path} 'properties' 必须是字典")
            else:
                for sub_field, sub_config in config["properties"].items():
                    errors.extend(cls._validate_field_config(sub_field, sub_config, current_path, strict))

        elif field_type == "list":
            # 检查item_type
            if "item_type" not in config:
                errors.append(f"{current_path} 列表类型缺少 'item_type' 字段")
            elif config["item_type"] not in cls.SUPPORTED_TYPES:
                errors.append(f"{current_path} 不支持的item_type: {config['item_type']}")

            # 如果item_type是dict，检查item_properties
            if config.get("item_type") == "dict" and "item_properties" in config:
                if not isinstance(config["item_properties"], dict):
                    errors.append(f"{current_path} 'item_properties' 必须是字典")
                else:
                    for sub_field, sub_config in config["item_properties"].items():
                        errors.extend(cls._validate_field_config(sub_field, sub_config, f"{current_path}[]", strict))

        return errors

    @classmethod
    def generate_pydantic_model(cls, schema: Dict[str, Any], model_name: str = "DynamicModel", strict_validation: bool = False) -> Type[BaseModel]:
        """根据Schema动态生成Pydantic模型
        Args:
            schema: 数据schema
            model_name: 模型名称
            strict_validation: 是否严格验证schema
        """

        # 验证schema（非严格模式，允许缺少description和required）
        errors = cls.validate_schema(schema, strict=strict_validation)
        if errors and strict_validation:
            raise ValueError(f"Schema验证失败: {errors}")
        elif errors:
            print(f"⚠️ Schema警告（非严格模式）: {errors}")

        field_definitions = {}

        for field_name, field_config in schema.items():
            field_definitions[field_name] = cls._build_field_definition(field_config)

        return create_model(model_name, **field_definitions)

    @classmethod
    def _build_field_definition(cls, config: Dict[str, Any]) -> tuple:
        """构建Pydantic字段定义"""
        field_type = config["type"]
        required = config.get("required", False)  # 默认为False
        description = config.get("description", f"{field_type}类型字段")

        # 基础类型处理
        if field_type in ["str", "int", "float", "bool"]:
            python_type = cls.TYPE_MAPPING[field_type]
            
            if not required:
                # 对于可选字段，使用Optional并提供默认值
                python_type = Optional[python_type]
                field_info = Field(default=None, description=description)
            else:
                field_info = Field(description=description)

            return (python_type, field_info)

        # 字典类型处理
        elif field_type == "dict":
            if "properties" in config:
                # 递归创建嵌套模型
                nested_model = cls.generate_pydantic_model(
                    config["properties"],
                    f"Nested_{id(config)}",
                    strict_validation=False  # 嵌套模型也使用非严格模式
                )
                
                if not required:
                    nested_model = Optional[nested_model]
                    field_info = Field(default=None, description=description)
                else:
                    field_info = Field(description=description)

                return (nested_model, field_info)
            else:
                # 普通字典
                if not required:
                    python_type = Optional[Dict[str, Any]]
                    field_info = Field(default=None, description=description)
                else:
                    python_type = Dict[str, Any]
                    field_info = Field(description=description)

                return (python_type, field_info)

        # 列表类型处理
        elif field_type == "list":
            item_type = config["item_type"]

            if item_type in ["str", "int", "float", "bool"]:
                # 基础类型列表
                python_type = List[cls.TYPE_MAPPING[item_type]]
                if not required:
                    field_info = Field(default=[], description=description)
                else:
                    field_info = Field(description=description)

                return (python_type, field_info)

            elif item_type == "dict" and "item_properties" in config:
                # 字典列表 - 递归创建嵌套模型
                item_model = cls.generate_pydantic_model(
                    config["item_properties"],
                    f"ListItem_{id(config)}",
                    strict_validation=False  # 嵌套模型也使用非严格模式
                )
                python_type = List[item_model]
                if not required:
                    field_info = Field(default=[], description=description)
                else:
                    field_info = Field(description=description)

                return (python_type, field_info)

            else:
                # 其他类型的列表
                if not required:
                    field_info = Field(default=[], description=description)
                else:
                    field_info = Field(description=description)
                return (List[Any], field_info)

        # 默认情况
        if not required:
            field_info = Field(default=None, description=description)
        else:
            field_info = Field(description=description)
        return (Any, field_info)

    @classmethod
    def validate_data(cls, data: Dict[str, Any], schema: Dict[str, Any], strict_validation: bool = False) -> Dict[str, Any]:
        """使用Pydantic验证数据是否符合Schema
        Args:
            data: 要验证的数据
            schema: 数据schema
            strict_validation: 是否严格验证schema
        """
        try:
            model = cls.generate_pydantic_model(schema, strict_validation=strict_validation)
            validated_data = model(**data)
            return {
                "valid": True,
                "data": validated_data.dict(),
                "errors": []
            }
        except Exception as e:
            return {
                "valid": False,
                "data": None,
                "errors": [str(e)]
            }

    @classmethod
    def generate_schema_description(cls, schema: Dict[str, Any]) -> str:
        """生成Schema的详细描述文档"""

        def build_field_description(field_name: str, config: Dict[str, Any], level: int = 0) -> str:
            """递归构建字段描述"""
            indent = "  " * level
            field_type = config['type']
            description = config.get('description', '暂无描述')
            required = config.get('required', False)  # 默认为False

            # 基础字段信息
            desc = f"{indent} {field_name} ({field_type})"
            desc += f": {description}"

            # 必填标识
            if required:
                desc += " [必填]"
            else:
                desc += " [可选]"

            # 处理不同类型
            if field_type == "dict" and 'properties' in config:
                desc += "\n"
                for sub_field, sub_config in config['properties'].items():
                    desc += build_field_description(sub_field, sub_config, level + 1) + "\n"

            elif field_type == "list":
                item_type = config.get('item_type', 'any')
                desc += f" - 元素类型: {item_type}"

                if item_type == "dict" and 'item_properties' in config:
                    desc += "\n" + indent + "   列表元素结构:\n"
                    for sub_field, sub_config in config['item_properties'].items():
                        desc += build_field_description(sub_field, sub_config, level + 2) + "\n"

            return desc.rstrip()

        # 构建完整文档
        schema_doc = ["## 数据字段规范:\n"]
        for field_name, field_config in schema.items():
            schema_doc.append(build_field_description(field_name, field_config))

        return "\n".join(schema_doc)

    @classmethod
    def generate_example_data(cls, schema: Dict[str, Any]) -> Dict[str, Any]:
        """根据Schema生成示例数据"""

        def generate_field_example(config: Dict[str, Any]) -> Any:
            field_type = config['type']

            # 基础类型示例
            if field_type == "str":
                return "示例文本"
            elif field_type == "int":
                return 42
            elif field_type == "float":
                return 3.14
            elif field_type == "bool":
                return True

            # 字典类型
            elif field_type == "dict":
                if "properties" in config:
                    example_dict = {}
                    for sub_field, sub_config in config["properties"].items():
                        if sub_config.get('required', False):
                            example_dict[sub_field] = generate_field_example(sub_config)
                    return example_dict
                else:
                    return {"key": "value"}

            # 列表类型
            elif field_type == "list":
                item_type = config.get('item_type', 'str')

                if item_type in ["str", "int", "float", "bool"]:
                    type_examples = {
                        "str": ["示例1", "示例2"],
                        "int": [1, 2, 3],
                        "float": [1.1, 2.2],
                        "bool": [True, False]
                    }
                    return type_examples.get(item_type, [])

                elif item_type == "dict" and "item_properties" in config:
                    item_example = {}
                    for sub_field, sub_config in config["item_properties"].items():
                        if sub_config.get('required', False):
                            item_example[sub_field] = generate_field_example(sub_config)
                    return [item_example] if item_example else []

                else:
                    return []

            return None

        example_data = {}
        for field_name, field_config in schema.items():
            if field_config.get('required', False):
                example_data[field_name] = generate_field_example(field_config)

        return example_data
