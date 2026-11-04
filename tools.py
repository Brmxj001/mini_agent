"""工具定义、注册表与三个内置工具：calculator / search / todo。"""

from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from storage import Storage


class ToolInputError(Exception):
    """工具参数或执行错误，会被转换成模型可见的错误 tool 消息。"""


@dataclass
class ToolContext:
    storage: "Storage"
    user_id: str
    session_id: int


@dataclass
class ToolResult:
    content: str
    is_error: bool = False


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    execute: Callable[[dict, ToolContext], str]

    def spec(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self._tools = {tool.name: tool for tool in tools}

    def specs(self) -> list[dict]:
        return [tool.spec() for tool in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def execute(self, name: str, args: object, ctx: ToolContext) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(f"[工具错误] 不存在名为 {name} 的工具", True)
        if not isinstance(args, dict):
            return ToolResult("[工具错误] 工具参数必须是 JSON 对象", True)
        try:
            return ToolResult(str(tool.execute(args, ctx)))
        except ToolInputError as exc:
            return ToolResult(f"[工具错误] {exc}", True)
        except Exception as exc:
            return ToolResult(f"[工具错误] {type(exc).__name__}: {exc}", True)


# ---------------------------------------------------------------- calculator

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MAX_RESULT = 1e15


def _bounded(value):
    if isinstance(value, complex) or (
        isinstance(value, float) and not math.isfinite(value)
    ):
        raise ToolInputError("计算结果不是有限实数")
    if abs(value) > _MAX_RESULT:
        raise ToolInputError("计算结果过大")
    return value


def _eval_node(node: ast.AST, depth: int = 0):
    if depth > 50:
        raise ToolInputError("表达式过于复杂")
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, depth + 1)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolInputError("只支持数字常量")
        return node.value
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ToolInputError(f"不支持的运算符：{type(node.op).__name__}")
        left = _eval_node(node.left, depth + 1)
        right = _eval_node(node.right, depth + 1)
        if isinstance(node.op, ast.Pow):
            if isinstance(right, bool) or not isinstance(right, int) or abs(right) > 100:
                raise ToolInputError("指数必须是不超过 100 的整数")
            if left == 0 and right < 0:
                raise ToolInputError("除数不能为 0")
            if abs(left) > 1 and right > 0 and right * math.log10(abs(left)) > 15:
                raise ToolInputError("计算结果过大")
        try:
            return _bounded(op(left, right))
        except ZeroDivisionError as exc:
            raise ToolInputError("除数不能为 0") from exc
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ToolInputError(f"不支持的一元运算符：{type(node.op).__name__}")
        return _bounded(op(_eval_node(node.operand, depth + 1)))
    raise ToolInputError(f"不支持的表达式节点：{type(node).__name__}")


def calculator(args: dict, ctx: ToolContext) -> str:
    expression = args.get("expression")
    if not isinstance(expression, str) or not expression.strip():
        raise ToolInputError("缺少非空字符串参数 expression")
    expression = expression.strip()
    if len(expression) > 200:
        raise ToolInputError("表达式过长（最多 200 字符）")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolInputError(f"表达式语法错误：{exc.msg}") from exc
    value = _bounded(_eval_node(tree))
    text = f"{value:.12g}" if isinstance(value, float) else str(value)
    return f"{expression} = {text}"


# -------------------------------------------------------------------- search

_MOCK_DOCS = [
    {
        "title": "上海天气（Mock）",
        "snippet": "上海今天晴，气温 25°C，东南风 2 级，适合外出。",
        "tags": "天气 weather 上海 预报",
        "url": "mock://weather/shanghai",
    },
    {
        "title": "Python asyncio 入门",
        "snippet": "asyncio 是 Python 标准库中的异步 I/O 框架，使用 async/await 编写协程。",
        "tags": "python asyncio 异步 协程",
        "url": "mock://python/asyncio",
    },
    {
        "title": "SQLite 简介",
        "snippet": "SQLite 是嵌入式关系型数据库，Python 内置 sqlite3 模块即可直接使用。",
        "tags": "sqlite 数据库 持久化",
        "url": "mock://sqlite/intro",
    },
    {
        "title": "DeepSeek API 指南",
        "snippet": "DeepSeek 提供 OpenAI 兼容的 Chat Completions API，支持 function calling。",
        "tags": "deepseek api 大模型 function-calling",
        "url": "mock://deepseek/api",
    },
    {
        "title": "Agent Loop 基础",
        "snippet": "Agent Loop 反复调用模型、执行工具并回传结果，直到模型给出最终回答。",
        "tags": "agent loop 智能体 工具调用",
        "url": "mock://agent/loop",
    },
    {
        "title": "pytest 快速上手",
        "snippet": "pytest 是 Python 常用测试框架，支持 fixture、参数化与断言重写。",
        "tags": "pytest 测试 单元测试",
        "url": "mock://pytest/intro",
    },
]


def _search_score(doc: dict, query: str) -> int:
    haystack = f"{doc['title']} {doc['snippet']} {doc['tags']}".lower()
    tokens = [t for t in re.split(r"\s+", query.lower()) if len(t) >= 2]
    score = sum(1 for token in tokens if token in haystack)
    words = [w for w in re.split(r"\s+", doc["tags"].lower()) if len(w) >= 2]
    score += sum(1 for word in words if word in query.lower())
    return score


def search(args: dict, ctx: ToolContext) -> str:
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ToolInputError("缺少非空字符串参数 query")
    query = query.strip()
    top_k = args.get("top_k", 3)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 5:
        raise ToolInputError("top_k 必须是 1 到 5 之间的整数")

    scored = [(doc, _search_score(doc, query)) for doc in _MOCK_DOCS]
    hits = [pair for pair in scored if pair[1] > 0]
    hits.sort(key=lambda pair: pair[1], reverse=True)
    if not hits:
        return f"未找到与「{query}」相关的 Mock 结果（本地模拟搜索，未联网）。"

    lines = [f"Mock 搜索结果（共 {min(top_k, len(hits))} 条）："]
    for index, (doc, _score) in enumerate(hits[:top_k], start=1):
        lines.append(f"{index}. {doc['title']}\n   {doc['snippet']}\n   来源: {doc['url']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------- todo

def todo(args: dict, ctx: ToolContext) -> str:
    action = args.get("action")

    if action == "add":
        content = args.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ToolInputError("add 操作需要非空字符串参数 content")
        content = content.strip()[:200]
        todo_id = ctx.storage.add_todo(ctx.user_id, ctx.session_id, content)
        return f"已添加待办 #{todo_id}：{content}"

    if action == "list":
        items = ctx.storage.list_todos(ctx.user_id, ctx.session_id)
        if not items:
            return "当前会话没有待办。"
        lines = ["当前会话待办："]
        for item in items[:100]:
            mark = "x" if item.status == "completed" else " "
            lines.append(f"#{item.id} [{mark}] {item.content}")
        return "\n".join(lines)

    if action == "complete":
        todo_id = args.get("todo_id")
        if isinstance(todo_id, bool) or not isinstance(todo_id, int):
            raise ToolInputError("complete 操作需要整数参数 todo_id")
        item = ctx.storage.complete_todo(ctx.user_id, ctx.session_id, todo_id)
        if item is None:
            raise ToolInputError(f"当前会话中不存在待办 #{todo_id}")
        return f"已完成待办 #{item.id}：{item.content}"

    raise ToolInputError("action 必须是 add、list 或 complete")


# ------------------------------------------------------------------ registry

def build_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            Tool(
                name="calculator",
                description="安全计算数学表达式，支持 + - * / // % ** 和括号。需要精确计算时使用。",
                input_schema={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "要计算的数学表达式，例如 (12+8)*3",
                        }
                    },
                    "required": ["expression"],
                },
                execute=calculator,
            ),
            Tool(
                name="search",
                description="在本地 Mock 文档库中搜索资料（不联网），返回标题、摘要和来源。",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索关键词"},
                        "top_k": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 5,
                            "default": 3,
                            "description": "返回结果数量，默认 3",
                        },
                    },
                    "required": ["query"],
                },
                execute=search,
            ),
            Tool(
                name="todo",
                description="管理当前用户、当前会话的待办：add 添加、list 查看、complete 完成。",
                input_schema={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["add", "list", "complete"],
                            "description": "要执行的操作",
                        },
                        "content": {
                            "type": "string",
                            "description": "待办内容，add 时必填",
                        },
                        "todo_id": {
                            "type": "integer",
                            "description": "待办编号，complete 时必填",
                        },
                    },
                    "required": ["action"],
                },
                execute=todo,
            ),
        ]
    )
