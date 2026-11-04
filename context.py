"""System Prompt、上下文组装与历史压缩。

压缩以“完整对话组”为单位：每组从一条真实用户文本消息开始，包含其后的
assistant/tool 交互，直到下一条真实用户消息。只压缩完整组，因此
assistant(tool_calls) 与对应的 tool 结果永远不会被拆散。
"""

from dataclasses import dataclass

from storage import Message, Session, Storage

SYSTEM_PROMPT = """你是一个最小可运行的中文助手，可以调用工具完成计算、Mock 搜索和待办管理。
- 需要精确计算时调用 calculator，不要心算。
- 需要查资料时调用 search；它的结果是本地 Mock 数据，回答时应说明来源为 Mock。
- 用户提到待办、任务清单时调用 todo，增删查都以工具返回结果为准。
- 不要臆造工具结果；工具报错时如实说明并尝试其他办法。
- 回答简洁、直接。"""

SUMMARY_LIMIT = 3000


@dataclass
class Compaction:
    compressed_messages: int
    kept_messages: int
    until_message_id: int
    summary_chars: int


def build_context(storage: Storage, session: Session) -> tuple[str, list[dict]]:
    """返回 (system_prompt, 活跃历史消息)。"""
    fresh = storage.get_session(session.id) or session
    system = SYSTEM_PROMPT
    if fresh.summary.strip():
        system += "\n\n## 历史对话摘要（较早对话已压缩，仅作参考）\n" + fresh.summary
    history = [message.data for message in storage.active_messages(fresh)]
    return system, history


def maybe_compact(
    storage: Storage,
    session: Session,
    *,
    compact_threshold: int = 20,
    keep_recent: int = 10,
) -> Compaction | None:
    fresh = storage.get_session(session.id) or session
    active = storage.active_messages(fresh)
    if len(active) <= compact_threshold:
        return None

    groups = _group_messages(active)
    if len(groups) <= 1:
        return None

    kept_groups: list[list[Message]] = []
    kept_count = 0
    for group in reversed(groups):
        if kept_count >= keep_recent:
            break
        kept_groups.insert(0, group)
        kept_count += len(group)

    compressed_groups = groups[: len(groups) - len(kept_groups)]
    if not compressed_groups:
        return None
    compressed = [message for group in compressed_groups for message in group]
    if not compressed:
        return None

    addition = _summarize(compressed)
    merged = f"{fresh.summary}\n{addition}".strip() if fresh.summary.strip() else addition
    if len(merged) > SUMMARY_LIMIT:
        merged = "……（更早摘要已省略）\n" + merged[-SUMMARY_LIMIT:]

    until_id = compressed[-1].id
    storage.set_summary(fresh.id, merged, until_id)
    session.summary = merged
    session.summary_until_message_id = until_id

    return Compaction(
        compressed_messages=len(compressed),
        kept_messages=kept_count,
        until_message_id=until_id,
        summary_chars=len(merged),
    )


def _is_user_text(message: Message) -> bool:
    data = message.data
    return (
        data.get("role") == "user"
        and isinstance(data.get("content"), str)
        and bool(data["content"].strip())
    )


def _group_messages(messages: list[Message]) -> list[list[Message]]:
    groups: list[list[Message]] = []
    for message in messages:
        if _is_user_text(message) or not groups:
            groups.append([message])
        else:
            groups[-1].append(message)
    return groups


def _summarize(messages: list[Message]) -> str:
    lines: list[str] = []
    tool_names: dict[str, str] = {}
    for message in messages:
        data = message.data
        role = data.get("role")
        content = _flatten(data.get("content"))

        if role == "user":
            if content.strip():
                lines.append(f"用户：{_clip(content, 150)}")
            continue

        if role == "assistant":
            for call in data.get("tool_calls") or []:
                function = call.get("function") or {}
                name = function.get("name") or "unknown"
                tool_names[call.get("id")] = name
                args = _clip(function.get("arguments") or "", 100)
                lines.append(f"助手调用 {name}({args})")
            if content.strip():
                lines.append(f"助手：{_clip(content, 150)}")
            continue

        if role == "tool":
            name = tool_names.get(data.get("tool_call_id"), "工具")
            label = "错误" if content.startswith("[工具错误]") else "结果"
            lines.append(f"{name} {label}：{_clip(content, 150)}")

    return "\n".join(lines)


def _flatten(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return "" if content is None else str(content)


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
