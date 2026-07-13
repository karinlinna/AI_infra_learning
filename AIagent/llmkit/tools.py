"""工具（function calling）辅助设施。

@tool 装饰器把一个普通 Python 函数变成：
  1. 一份可暴露给模型的 ToolSpec（自动从类型注解与 docstring 生成 JSON Schema）；
  2. 一个可被 Agent 循环调用的可执行体。

ToolRegistry 负责登记、生成 specs、按名分发调用。

教学要点：工具的本质是"把模型的意图翻译成对世界的操作"。schema 写得好不好，
直接决定模型能否正确调用——这是提示工程在工具层面的延伸。
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from .types import ToolSpec

_PY_TO_JSON = {int: "integer", float: "number", str: "string", bool: "boolean"}


@dataclass
class Tool:
    spec: ToolSpec
    func: Callable[..., Any]

    def __call__(self, **kwargs) -> str:
        result = self.func(**kwargs)
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)


def tool(fn: Callable) -> Tool:
    """把函数转成 Tool。参数类型来自注解，描述来自 docstring 第一行。"""
    sig = inspect.signature(fn)
    props, required = {}, []
    for pname, p in sig.parameters.items():
        json_type = _PY_TO_JSON.get(p.annotation, "string")
        props[pname] = {"type": json_type, "description": f"参数 {pname}"}
        if p.default is inspect.Parameter.empty:
            required.append(pname)
    desc = (fn.__doc__ or fn.__name__).strip().splitlines()[0]
    spec = ToolSpec(
        name=fn.__name__,
        description=desc,
        parameters={"type": "object", "properties": props, "required": required},
    )
    return Tool(spec=spec, func=fn)


class ToolRegistry:
    """登记并分发工具调用。"""

    def __init__(self, *tools: Tool):
        self._tools: Dict[str, Tool] = {}
        for t in tools:
            self.register(t)

    def register(self, t: Tool) -> None:
        self._tools[t.spec.name] = t

    def specs(self) -> List[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def call(self, name: str, arguments: Dict[str, Any]) -> str:
        if name not in self._tools:
            return f"错误：未知工具 {name}"
        try:
            return self._tools[name](**arguments)
        except Exception as e:  # 工具异常必须捕获并回传给模型，让它有机会纠错
            return f"工具 {name} 执行失败：{e}"
