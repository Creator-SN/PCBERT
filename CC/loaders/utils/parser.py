from __future__ import annotations
from typing import Any, Dict
from distutils.util import strtobool
import json


class KwargItem():
    def __init__(self, name: str, _type: type, optional: bool = False, defaultValue=None, description: str = ""):
        self.name = name
        self.type = _type
        if defaultValue is not None:
            optional = True
        self.optional = optional
        self.defaultValue = defaultValue
        if description == "":
            if not optional:
                description = f"args {name}: required!"
        self.description = description


class KwargsParser():
    def __init__(self, debug: bool = False):
        self.args_name_dict: Dict[KwargItem] = {}
        self.debug = debug
        self.kv = {}

    def add_argument(self, argName: str, argType: type, defaultValue=None, optional: bool = False, description: str = "") -> KwargsParser:
        """add argument

        Args:
            argName (str): arg name
            argType (type): arg type
            defaultValue (any, optional): default value. Defaults to None.
            optional (bool, optional): is optional? Defaults to False.

        Returns:
            [type]: self
        """
        self.args_name_dict[argName] = KwargItem(
            argName, argType, optional=optional, defaultValue=defaultValue, description=description)
        return self

    def parse(self, instance, **kwargs) -> KwargsParser:
        self.kv = {}
        for argName in self.args_name_dict:
            arg: KwargItem = self.args_name_dict[argName]
            value = arg.defaultValue
            if not arg.optional:
                if argName not in kwargs:
                    raise ValueError(arg.description)
                value = kwargs[argName]
            else:
                if argName in kwargs:
                    value = kwargs[argName]
            if arg.optional and value is None:
                setattr(instance, argName, None)
            else:
                setattr(instance, argName, self._convert_to(value, arg.type))
            self.kv[argName] = getattr(instance, argName)
        if self.debug:
            print(
                f"kwargs parser: {json.dumps(self.kv,indent=4,ensure_ascii=False)}")
        return self

    def parse_dict(self, **kwargs) -> Dict[str, Any]:
        self.kv = {}
        for argName in self.args_name_dict:
            arg: KwargItem = self.args_name_dict[argName]
            value = arg.defaultValue
            if not arg.optional:
                if argName not in kwargs:
                    raise ValueError(arg.description)
                value = kwargs[argName]
            else:
                if argName in kwargs:
                    value = kwargs[argName]
            if arg.optional and value is None:
                self.kv[argName] = None
            else:
                self.kv[argName] = self._convert_to(value, arg.type)

        if self.debug:
            print(
                f"kwargs parser: {json.dumps(self.kv,indent=4,ensure_ascii=False)}")
        return self.kv

    def _convert_to(self, value, _type: type):
        if isinstance(value, str):
            if _type is bool:
                return bool(strtobool(value))
            else:
                return _type(value)
        elif isinstance(value, _type):
            return value
        else:
            raise ValueError(f"value {value} could not convert to {_type}")
