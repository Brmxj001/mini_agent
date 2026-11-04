# 最小可运行 Agent（DeepSeek 原生 API）

一个从零实现的最小 Agent Demo：

- 自研 Agent Loop（未使用任何 Agent 框架）。
- 模型自主决定是否调用工具（calculator / search / todo）。
- SQLite 持久化多用户、多 Session 历史与待办。
- 上下文包含 System Prompt、最近消息、历史摘要，并在超阈值时自动压缩。
- 控制台 Trace 记录每一轮请求、工具调用、结果与耗时。
- 使用 DeepSeek 原生 OpenAI 兼容接口（httpx 直连），不依赖 Anthropic/Claude SDK。

> 项目包含 59 个 pytest 自动化用例（`tests/`），默认使用 FakeLLM 和临时 SQLite，不访问网络、不依赖真实 API Key；另有真实 DeepSeek API 的手动验收场景（第 10 节）。

## 1. 环境准备

要求 Python 3.11+（开发环境为 Python 3.12）。

```bash
cd mini_agent

# 方式一：uv（推荐，开发时使用）
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt

# 方式二：venv + pip
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

配置 API Key（代码只从环境变量读取，不写入源码）：

```bash
export DEEPSEEK_API_KEY=sk-...
# 可选：
export DEEPSEEK_BASE_URL=https://api.deepseek.com   # 默认值
export DEEPSEEK_MODEL=deepseek-flash                # 默认值
```

DeepSeek 为国内服务，直连即可；若本机有代理环境变量，必要时用
`env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY ...` 运行。

## 2. 运行方式

```bash
# 交互模式（同一 user + session 会自动恢复历史）
.venv/bin/python main.py --user alice --session session_1

# 单轮模式
.venv/bin/python main.py --user alice --session session_1 -m "请用计算器算 (12+8)*3"

# 指定数据库、最大循环次数、压缩阈值
.venv/bin/python main.py --user alice --session session_1 \
  --db data/agent.db --max-iterations 8 --compact-threshold 20 --keep-recent 10
```

交互模式下输入 `/exit` 或 `/quit` 退出。默认数据库为 `data/agent.db`，首次运行自动创建。

参数说明：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--user` | `default` | 用户 ID |
| `--session` | `default` | 会话名；同一 user 下唯一 |
| `-m/--message` | 无 | 单轮模式；不传则进入 REPL |
| `--db` | `data/agent.db` | SQLite 文件路径 |
| `--max-iterations` | `8` | 单轮最大循环次数 |
| `--compact-threshold` | `20` | 活跃消息数超过该值时压缩 |
| `--keep-recent` | `10` | 压缩时至少保留的最近消息数（按完整对话组扩展） |

## 3. 目录结构

```
mini_agent/
├── main.py           # CLI + Agent Loop + 控制台 Trace
├── llm.py            # DeepSeekClient（httpx 直连 /chat/completions）+ DeepSeekError
├── tools.py          # Tool / ToolRegistry + calculator / search / todo
├── storage.py        # SQLite：sessions / messages / todos
├── context.py        # System Prompt、上下文组装、摘要压缩
├── requirements.txt  # httpx + pytest
├── pytest.ini        # pytest 配置
├── tests/            # FakeLLM + 59 个自动化测试
│   ├── conftest.py
│   ├── fake_llm.py
│   ├── test_agent_loop.py
│   ├── test_tools.py
│   ├── test_sessions.py
│   ├── test_context.py
│   ├── test_trace_errors.py
│   └── test_llm_client.py
├── README.md
└── AI_PROMPTS.md     # System Prompt、开发 Prompt、问题解决记录
```

## 4. 系统设计

```text
CLI (main.py)
   │  每轮用户输入
   ▼
Agent Loop ──────────────► DeepSeek API (/chat/completions)
   │  ▲                         │
   │  └── tool 消息 ◄── ToolRegistry：calculator / search / todo
   │                              │
   └── Context（System + 摘要 + 最近消息）
                                  │
                                  ▼
                          SQLite data/agent.db
                     sessions / messages / todos
```

各模块职责：

- **main.py**：解析 CLI 参数；每轮组装上下文并驱动 Loop；执行工具；处理 API 异常、最大轮次；打印 Trace。
- **llm.py**：唯一的模型调用入口。60 秒超时，把超时、连接失败、HTTP 非 200、响应格式异常统一转换为 `DeepSeekError`。
- **tools.py**：工具注册表；每个工具包含 `name / description / input_schema / execute`；统一捕获工具异常。
- **storage.py**：Session、原始消息、Todo 的 SQLite 读写；消息历史与 Todo 都按 Session 隔离。
- **context.py**：固定 System Prompt；生成上下文；按完整对话组压缩历史并持久化摘要。

### 消息协议

DeepSeek 原生接口是 OpenAI 风格的 Chat Completions，与题目中 Claude 术语的对应关系：

| 题目术语 | 本项目实际使用 |
|---|---|
| assistant 消息 | `{"role": "assistant", "content": ...}` |
| `tool_use` | assistant 消息中的 `tool_calls` |
| `tool_result` | `{"role": "tool", "tool_call_id": ...}` |
| `stop_reason = tool_use` | `finish_reason = "tool_calls"` |

DeepSeek 在 thinking 模式下要求把 assistant 消息里的 `reasoning_content` 原样回传，否则请求返回 400。因此本项目把 assistant 消息**按原始 JSON 完整存储并回放**（包含 `reasoning_content` 和 `tool_calls`），但 Trace、摘要和 README 中都不展示或提取该字段。

## 5. Agent Loop

```text
run_turn(user_id, session_name, text):
  1. 取得/创建 Session；读取历史并写入本轮 user 消息
  2. 若活跃消息数 > 阈值：压缩较早的完整对话组为摘要（写 SQLite）
  3. 组装 System Prompt + 摘要 + 最近消息
  4. for iteration in 1..max_iterations:
       a. 请求 DeepSeek（携带 tools schema）
       b. 若响应没有 tool_calls：保存 assistant 消息，返回最终回答
       c. 若已达到最大循环次数：返回固定提示，不执行、不落盘该 tool_calls
       d. 依次执行全部 tool_calls，生成对应的 tool 消息
       e. 将 assistant(tool_calls) 与全部 tool 消息原子写入 SQLite
       f. 把它们追加到上下文，继续下一轮
```

要点：

- 工具选择完全由模型返回的 `tool_calls` 决定，代码中没有任何按用户关键词选工具的逻辑。
- 一次 assistant 响应中的多个 `tool_calls` 会全部执行，结果按同一顺序作为多条 `role: "tool"` 消息回传。
- 工具失败不会终止 Loop：工具不存在、参数不是合法 JSON、参数校验失败、执行抛异常，都会变成带 `[工具错误]` 前缀的 tool 消息，由模型决定如何向用户解释或重试。
- 最大轮次限制在“执行工具之前”检查，避免数据库中留下没有对应 tool 结果的悬空 `tool_calls`。
- API 超时/连接失败/HTTP 错误会打印 `[错误]` Trace 并返回友好提示，进程不崩溃。

## 6. 工具系统

| 工具 | 参数 | 行为 |
|---|---|---|
| `calculator` | `expression` | AST 白名单求值，不使用 `eval`；支持 `+ - * / // % **`、一元 ±、括号；拒绝变量/函数/属性/下标；限制表达式长度、指数和结果大小；除零与非法表达式返回错误 |
| `search` | `query`、`top_k`(1–5) | 本地固定 Mock 文档库（天气、asyncio、SQLite、DeepSeek、Agent Loop、pytest），按关键词打分返回结果与 `mock://` 来源；不联网 |
| `todo` | `action`(add/list/complete)、`content`、`todo_id` | 严格按当前 `user_id + session_id` 读写；返回 `#id [ ]/[x] 内容`；完成不存在的待办会报错 |

工具统一注册在 `ToolRegistry` 中，`specs()` 输出 DeepSeek/OpenAI 所需的 function schema。

## 7. Memory：召回时机与存储位置

**存储位置**：`data/agent.db`（可通过 `--db` 修改）。

| 表 | 内容 |
|---|---|
| `sessions` | `(user_id, name)` 唯一；`summary` 保存历史摘要；`summary_until_message_id` 保存已压缩边界 |
| `messages` | 每个 Session 的原始消息 JSON（user / assistant / tool），按自增 id 排序 |
| `todos` | 待办内容、状态、所属 `user_id + session_id` |

**召回时机**：

- 每次 `run_turn` 开始时，从 SQLite 读取该 Session 的摘要与边界之后的活跃消息，组装成 System + 最近消息发给模型。
- 压缩在每轮开始、写入本轮 user 消息之后检查并执行；压缩结果立即写回 SQLite，下一轮（包括服务重启后的下一进程）自动重新加载。
- 循环内部的新 assistant/tool 消息直接追加到内存中的消息列表；本轮结束后它们已经落库，下一轮重新从数据库加载。
- Todo 不在 System Prompt 中预加载：模型需要时调用 `todo` 工具，由工具层实时查询当前 `user_id + session_id` 的数据。

**隔离规则**：

- Session 由 `(user_id, name)` 唯一确定；alice/session_1、alice/session_2、bob/session_1 三者的消息完全隔离。
- Todo 的所有查询都同时带 `user_id` 和内部 `session_id`，不同 Session 互相不可见。

## 8. 上下文压缩

- 默认：活跃消息 > 20 条时压缩，保留最近至少 10 条。
- **按完整对话组切分**：每组从一条真实用户文本消息开始，包含后续 assistant 的 tool_calls、全部 tool 结果和最终回答，直到下一条用户文本消息。只压缩完整组，因此不会拆散 `tool_calls` 与对应的 tool 消息。
- 摘要为确定性本地生成，不额外调用模型；包含用户问题、助手回答、工具调用与结果的关键信息（各截断），跳过 `reasoning_content`。摘要与旧摘要合并，总长上限 3000 字符。
- 摘要保存在 `sessions.summary`，边界保存在 `sessions.summary_until_message_id`，二者在同一事务中更新。
- 实际运行示例（真实 API）：session_1 达到 22 条活跃消息时，压缩了 10 条较早消息，生成 680 字摘要，保留 13 条；下一轮从新进程启动后摘要成功恢复，模型据此回答了更早的计算结果 60。

## 9. Trace 示例（真实运行）

```text
[17:12:52.079] [会话] alice/session_1 | 恢复 2 条历史消息 | 摘要 0 字
[17:12:52.079] [用户] 请用计算器算 (12+8)*3
[17:12:52.082] [请求] Loop 1/8 | model=deepseek-flash | messages=4 | tools=calculator,search,todo
[17:12:52.946] [响应] Loop 1/8 | tool_calls: calculator (0.86s)
[17:12:52.946] [工具] Loop 1/8 | calculator({"expression": "(12+8)*3"}) -> (12+8)*3 = 60 | is_error=False | 0.2ms
[17:12:52.950] [请求] Loop 2/8 | model=deepseek-flash | messages=6 | tools=calculator,search,todo
[17:12:53.658] [响应] Loop 2/8 | text (0.71s)
[17:12:53.660] [最终] (12+8)*3 = **60** | 总耗时 1.58s
```

压缩时会出现类似：

```text
[17:13:42.587] [会话] alice/session_1 | 恢复 22 条历史消息 | 摘要 0 字
[17:13:42.591] [压缩] 10 条较早消息 -> 摘要（680 字），保留 13 条，边界消息 #10
[17:13:42.592] [请求] Loop 1/8 | model=deepseek-flash | messages=14 | tools=calculator,search,todo
[17:13:43.588] [响应] Loop 1/8 | text (1.00s)
[17:13:43.591] [最终] (12+8)*3 = **60**（早前由计算器工具算得）。 | 总耗时 1.01s
```

Trace 明确记录：用户输入、当前 Loop 次数、模型请求摘要、工具名、工具参数、工具结果、耗时、最终回答、压缩事件与错误。不会输出 API Key 和 `reasoning_content`。

## 10. 手动验收场景

以下命令均已在真实 DeepSeek API 上运行验证。建议使用独立数据库文件，避免污染默认数据：

```bash
DB=/tmp/mini_agent_acceptance.db
rm -f "$DB"
```

1. **普通对话**
   ```bash
   .venv/bin/python main.py --user alice --session session_1 --db "$DB" -m "你好，简单介绍一下你自己"
   ```
   预期：`[响应] text`，输出助手自我介绍，无工具调用。

2. **Calculator 调用**
   ```bash
   .venv/bin/python main.py --user alice --session session_1 --db "$DB" -m "请用计算器算 (12+8)*3"
   ```
   预期：`[工具] calculator(...) -> (12+8)*3 = 60`，最终回答包含 60；同时验证了上一轮历史恢复。

3. **Search 调用**
   ```bash
   .venv/bin/python main.py --user alice --session session_1 --db "$DB" -m "搜索一下 Python asyncio 是什么"
   ```
   预期：`[工具] search(...)` 返回 `mock://python/asyncio` 等 Mock 结果。

4. **Todo 添加、查询、完成**（每条命令都是新进程，验证持久化）
   ```bash
   .venv/bin/python main.py --user alice --session session_1 --db "$DB" -m "帮我添加待办：写 README"
   .venv/bin/python main.py --user alice --session session_1 --db "$DB" -m "查看当前会话的待办"
   .venv/bin/python main.py --user alice --session session_1 --db "$DB" -m "把待办 #1 标记为完成"
   ```
   预期：依次出现“已添加待办 #1”“`#1 [ ] 写 README`”“已完成待办 #1”。

5. **多工具 / 连续工具调用**
   ```bash
   .venv/bin/python main.py --user alice --session session_2 --db "$DB" -m "帮我用计算器算 2**10，并搜索一下 DeepSeek 的资料"
   ```
   预期：同一条 `[响应]` 中出现 `tool_calls: calculator, search`，Trace 连续打印两个工具结果，随后给出最终回答。

6. **纯对话追问**
   在同一 Session 中继续：
   ```bash
   .venv/bin/python main.py --user alice --session session_1 --db "$DB" -m "刚才算的 (12+8)*3 结果是多少？请根据对话历史直接回答，不要重新计算。"
   ```
   预期：回答 60，且没有再次调用 calculator。

7. **带工具的追问**
   ```bash
   .venv/bin/python main.py --user alice --session session_1 --db "$DB" -m "把刚才那个结果再乘以 2，用计算器算"
   ```
   预期：`[工具] calculator({"expression": "60*2"}) -> 60*2 = 120`，最终回答 120。

8. **两个 Session 隔离**
   ```bash
   .venv/bin/python main.py --user alice --session session_2 --db "$DB" -m "查看当前会话的待办"
   .venv/bin/python main.py --user bob --session session_1 --db "$DB" -m "查看当前会话的待办"
   .venv/bin/python main.py --user alice --session session_1 --db "$DB" -m "查看当前会话的待办"
   ```
   预期：alice/session_2 和 bob/session_1 均为空；alice/session_1 只看到自己的 `#1 写 README`。

9. **历史 Session 恢复**
   上述每条 `-m` 命令都是独立进程。再次访问同一个 `--user + --session` 时，Trace 首行会显示“恢复 N 条历史消息”，模型能引用之前的内容。

10. **上下文压缩**
    使用较低阈值快速触发（每条命令都需要带上阈值参数）：
    ```bash
    .venv/bin/python main.py --user demo --session compact --db "$DB" \
      --compact-threshold 3 --keep-recent 1 -m "第一轮：请记住暗号是蓝色海豚"
    .venv/bin/python main.py --user demo --session compact --db "$DB" \
      --compact-threshold 3 --keep-recent 1 -m "第二轮：请复述刚才的暗号"
    .venv/bin/python main.py --user demo --session compact --db "$DB" \
      --compact-threshold 3 --keep-recent 1 -m "第三轮：暗号是什么？请根据历史摘要回答"
    ```
    预期：第三轮进程启动时 Trace 出现 `[压缩] 4 条较早消息 -> 摘要（76 字），保留 1 条`；只把当前用户消息和摘要发给模型，模型仍能根据摘要回答“蓝色海豚”。默认阈值下的真实运行也验证过：22 条活跃消息 → 压缩 10 条 → 680 字摘要，且压缩后仍能回忆出更早的计算结果 60。

11. **最大循环次数限制**
    ```bash
    .venv/bin/python main.py --user alice --session max_demo --db "$DB" \
      --max-iterations 1 -m "请用计算器算 1+1"
    ```
    预期：第一轮返回 `tool_calls` 后打印 `[错误] 达到最大循环次数 1，停止执行`，最终输出固定提示；工具未执行，数据库中不留下悬空 tool_calls。

12. **异常处理**
    ```bash
    # 工具异常：除零
    .venv/bin/python main.py --user alice --session error_demo --db "$DB" -m "请用计算器算 1/0"
    # API 连接失败
    DEEPSEEK_BASE_URL=http://127.0.0.1:9 .venv/bin/python main.py --user alice --session api_error_demo --db "$DB" -m "你好"
    # API 认证失败
    DEEPSEEK_API_KEY=sk-invalid .venv/bin/python main.py --user alice --session api_error_demo --db "$DB" -m "你好"
    ```
    预期：工具异常会得到 `[工具错误] 除数不能为 0`，模型继续并解释；连接/认证错误会打印 `[错误] API connection/http: ...` 和友好提示，进程不崩溃，之后的正常请求仍可继续。

## 11. 自动化测试

测试位于 `tests/`，使用 FakeLLM 和 `httpx.MockTransport` 模拟模型响应；每个用例使用 pytest `tmp_path` 提供的独立临时 SQLite，不会污染 `data/agent.db`，默认不访问网络，也不需要 `DEEPSEEK_API_KEY`。

```bash
# 项目根目录
.venv/bin/python -m pytest -v

# 如果已激活虚拟环境，等价于
python -m pytest -v
```

测试结构：

| 文件 | 用例数 | 覆盖内容 |
|---|---:|---|
| `tests/fake_llm.py` | - | 脚本化 FakeLLM：普通文本、单个/多个 `tool_calls`、连续多轮响应、异常注入；记录每次请求的 messages/tools |
| `tests/conftest.py` | - | 临时数据库、Agent 工厂 fixture |
| `tests/test_agent_loop.py` | 5 | 普通对话、单次工具调用、并行多工具、连续工具调用、工具后最终回答、最大循环次数 |
| `tests/test_tools.py` | 31 | 工具注册与 Schema、按名称查找、Calculator 正常/非法表达式、Search Mock 结果、Todo 添加/查询/完成、未知工具、参数错误、工具异常回传 |
| `tests/test_sessions.py` | 4 | 同用户多 Session 消息隔离、Todo 隔离、不同用户同名 Session 隔离、重开数据库恢复历史/Todo 并继续对话 |
| `tests/test_context.py` | 8 | 新消息进入 Context、纯对话/工具追问历史、摘要注入 System、超阈值压缩、保留最近消息、摘要持久化、重启加载摘要、压缩不拆散 tool_calls 与 tool 结果 |
| `tests/test_trace_errors.py` | 6 | 正常对话 Trace、工具名称/参数/结果/耗时 Trace、不泄露 API Key 与 reasoning_content、timeout/HTTP/连接错误兜底 |
| `tests/test_llm_client.py` | 5 | DeepSeekClient 原始响应解析、timeout / HTTP / protocol / config 异常映射（MockTransport，不联网） |

覆盖场景与需求对应关系：

- **Agent Loop**：普通对话、单工具、并行多工具、跨轮连续工具、最终回答、最大轮次（`test_agent_loop.py`）。
- **Tool Registry 与工具**：注册、Schema、查找、三个工具的成功与失败路径、Agent 收到错误 tool 消息（`test_tools.py`）。
- **Session 隔离**：消息与 Todo 隔离、不同用户同名 Session、SQLite 重开恢复（`test_sessions.py`）。
- **Context 与 Memory**：追问上下文、摘要加载、压缩触发/保留/持久化/结构完整性（`test_context.py`）。
- **Trace 与异常**：日志内容、工具调用记录、敏感信息不泄露、API 异常不崩溃（`test_trace_errors.py`、`test_llm_client.py`）。

实际运行结果（2026-11-04，Python 3.12.12，pytest 9.2.0）：

```text
============================== 59 passed in 2.08s ==============================
```

无失败、无 skip；生产代码未修改（仅补充测试与在 `requirements.txt` 中加入 pytest）。

## 12. 已知限制

- 搜索是本地 Mock 数据，不联网。
- 摘要为确定性文本压缩，不调用模型生成更自然的摘要。
- 自动化测试使用 FakeLLM 模拟模型响应，不验证 DeepSeek 服务端的真实网络行为；真实 API 行为由第 10 节手动验收覆盖。
- 单进程 SQLite，未做并发控制、用户鉴权、Web 接口或前端。
- API 调用失败时，本轮 user 消息已写入数据库；下一轮仍可正常继续（已实测）。
- 依赖 DeepSeek thinking 模式对 `reasoning_content` 的回传要求；因此消息按原始 JSON 存储。

更详细的 Prompt 与开发问题记录见 [AI_PROMPTS.md](./AI_PROMPTS.md)。
