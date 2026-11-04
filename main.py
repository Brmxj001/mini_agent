"""最小可运行 Agent：Agent Loop、控制台 Trace 与 CLI。"""

import argparse
import json
import sys
import time
from datetime import datetime

from context import build_context, maybe_compact
from llm import DeepSeekClient, DeepSeekError
from storage import Storage
from tools import ToolContext, ToolRegistry, ToolResult, build_registry

MAX_TRACE_TEXT = 500


def _clip(text: object, limit: int = MAX_TRACE_TEXT) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


class Trace:
    def __init__(self):
        self.started_at = time.perf_counter()

    def log(self, tag: str, text: str) -> None:
        now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{now}] [{tag}] {text}", flush=True)

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started_at


class Agent:
    def __init__(
        self,
        storage: Storage,
        llm: DeepSeekClient,
        registry: ToolRegistry,
        *,
        max_iterations: int = 8,
        compact_threshold: int = 20,
        keep_recent: int = 10,
    ):
        self.storage = storage
        self.llm = llm
        self.registry = registry
        self.max_iterations = max_iterations
        self.compact_threshold = compact_threshold
        self.keep_recent = keep_recent

    def run_turn(self, user_id: str, session_name: str, user_text: str) -> str:
        trace = Trace()
        session = self.storage.get_or_create_session(user_id, session_name)
        history_count = len(self.storage.active_messages(session))
        trace.log(
            "会话",
            f"{user_id}/{session_name} | 恢复 {history_count} 条历史消息 | "
            f"摘要 {len(session.summary)} 字",
        )
        trace.log("用户", user_text)
        self.storage.append_message(
            session.id, "user", {"role": "user", "content": user_text}
        )

        compaction = maybe_compact(
            self.storage,
            session,
            compact_threshold=self.compact_threshold,
            keep_recent=self.keep_recent,
        )
        if compaction:
            trace.log(
                "压缩",
                f"{compaction.compressed_messages} 条较早消息 -> 摘要"
                f"（{compaction.summary_chars} 字），保留 {compaction.kept_messages} 条，"
                f"边界消息 #{compaction.until_message_id}",
            )

        system, history = build_context(self.storage, session)
        messages = [{"role": "system", "content": system}, *history]
        tool_specs = self.registry.specs()

        for iteration in range(1, self.max_iterations + 1):
            trace.log(
                "请求",
                f"Loop {iteration}/{self.max_iterations} | model={self.llm.model} | "
                f"messages={len(messages)} | tools={','.join(self.registry.names())}",
            )
            started = time.perf_counter()
            try:
                assistant = self.llm.chat(messages, tool_specs)
            except DeepSeekError as exc:
                elapsed = time.perf_counter() - started
                trace.log("错误", f"API {exc.kind}: {exc} ({elapsed:.2f}s)")
                final = "抱歉，本轮模型调用失败，请稍后再试。"
                trace.log("最终", f"{final} | 总耗时 {trace.elapsed:.2f}s")
                return final

            elapsed = time.perf_counter() - started
            tool_calls = assistant.get("tool_calls") or []
            if tool_calls:
                names = ", ".join(
                    (call.get("function") or {}).get("name", "?")
                    for call in tool_calls
                )
                trace.log(
                    "响应",
                    f"Loop {iteration}/{self.max_iterations} | tool_calls: {names} "
                    f"({elapsed:.2f}s)",
                )
            else:
                trace.log(
                    "响应",
                    f"Loop {iteration}/{self.max_iterations} | text ({elapsed:.2f}s)",
                )

            if not tool_calls:
                self.storage.append_message(session.id, "assistant", assistant)
                content = assistant.get("content")
                final = content.strip() if isinstance(content, str) else ""
                if not final:
                    final = "（模型没有返回文本回答）"
                trace.log("最终", f"{final} | 总耗时 {trace.elapsed:.2f}s")
                return final

            if iteration == self.max_iterations:
                trace.log(
                    "错误", f"达到最大循环次数 {self.max_iterations}，停止执行"
                )
                final = f"已达到最大循环次数（{self.max_iterations}），本轮中止。"
                trace.log("最终", f"{final} | 总耗时 {trace.elapsed:.2f}s")
                return final

            tool_messages: list[dict] = []
            for call in tool_calls:
                function = call.get("function") or {}
                name = function.get("name") or ""
                raw_args = function.get("arguments") or ""
                started_tool = time.perf_counter()
                parse_failed = False
                if isinstance(raw_args, dict):
                    raw_display = json.dumps(raw_args, ensure_ascii=False)
                    args: object = raw_args
                else:
                    raw_display = str(raw_args)
                    try:
                        args = json.loads(raw_display) if raw_display.strip() else {}
                    except json.JSONDecodeError:
                        args = None
                        parse_failed = True

                if parse_failed:
                    result = ToolResult("[工具错误] 工具参数不是合法 JSON", True)
                else:
                    result = self.registry.execute(
                        name,
                        args,
                        ToolContext(self.storage, user_id, session.id),
                    )

                tool_elapsed = time.perf_counter() - started_tool
                trace.log(
                    "工具",
                    f"Loop {iteration}/{self.max_iterations} | "
                    f"{name}({_clip(raw_display, 300)}) -> {_clip(result.content)} | "
                    f"is_error={result.is_error} | {tool_elapsed * 1000:.1f}ms",
                )
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id"),
                        "content": result.content,
                    }
                )

            self.storage.append_exchange(session.id, assistant, tool_messages)
            messages.append(assistant)
            messages.extend(tool_messages)

        final = f"已达到最大循环次数（{self.max_iterations}），本轮中止。"
        trace.log("最终", f"{final} | 总耗时 {trace.elapsed:.2f}s")
        return final


def _build_agent(args: argparse.Namespace, storage: Storage, llm: DeepSeekClient) -> Agent:
    return Agent(
        storage,
        llm,
        build_registry(),
        max_iterations=args.max_iterations,
        compact_threshold=args.compact_threshold,
        keep_recent=args.keep_recent,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="最小可运行 DeepSeek Agent（自研 Agent Loop）"
    )
    parser.add_argument("--user", default="default", help="用户 ID，默认 default")
    parser.add_argument("--session", default="default", help="会话名，默认 default")
    parser.add_argument("-m", "--message", help="单轮消息；不传则进入交互模式")
    parser.add_argument("--db", default="data/agent.db", help="SQLite 路径")
    parser.add_argument("--max-iterations", type=int, default=8, help="最大循环次数")
    parser.add_argument(
        "--compact-threshold", type=int, default=20, help="历史消息压缩阈值"
    )
    parser.add_argument(
        "--keep-recent", type=int, default=10, help="压缩后至少保留的最近消息数"
    )
    args = parser.parse_args()

    if args.max_iterations < 1:
        parser.error("--max-iterations 必须 >= 1")
    if args.compact_threshold < 2:
        parser.error("--compact-threshold 必须 >= 2")
    if args.keep_recent < 1:
        parser.error("--keep-recent 必须 >= 1")

    try:
        llm = DeepSeekClient()
    except DeepSeekError as exc:
        print(f"初始化失败：{exc}")
        print("请先设置环境变量，例如：export DEEPSEEK_API_KEY=sk-...")
        return 1

    storage = Storage(args.db)
    agent = _build_agent(args, storage, llm)

    try:
        if args.message:
            agent.run_turn(args.user, args.session, args.message)
            return 0

        print(
            f"DeepSeek Agent | user={args.user} | session={args.session} | db={args.db}"
        )
        print(f"模型：{llm.model} | 输入 /exit 退出\n")
        while True:
            try:
                text = input(f"{args.user}@{args.session}> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text:
                continue
            if text in ("/exit", "/quit"):
                break
            agent.run_turn(args.user, args.session, text)
    finally:
        storage.close()
        llm.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
