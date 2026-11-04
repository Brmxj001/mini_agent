# AI Prompt 与问题解决记录

本文记录本项目的 System Prompt、开发过程中使用的关键 AI Prompt，以及实际遇到的问题和解决过程。

## 1. Agent System Prompt

定义在 `context.py` 的 `SYSTEM_PROMPT`，每次请求都会作为第一条 system 消息发送：

```text
你是一个最小可运行的中文助手，可以调用工具完成计算、Mock 搜索和待办管理。
- 需要精确计算时调用 calculator，不要心算。
- 需要查资料时调用 search；它的结果是本地 Mock 数据，回答时应说明来源为 Mock。
- 用户提到待办、任务清单时调用 todo，增删查都以工具返回结果为准。
- 不要臆造工具结果；工具报错时如实说明并尝试其他办法。
- 回答简洁、直接。
```

当存在历史摘要时，追加：

```text
## 历史对话摘要（较早对话已压缩，仅作参考）
<SQLite sessions.summary 中的摘要文本>
```

## 2. 开发过程中使用的关键 Prompt

### 2.1 初始需求（用户提供）

要求从零实现一个最小可运行 Agent，核心要点：

- 自己实现 Agent Loop，禁止使用 LangChain / LangGraph / OpenHands / CrewAI 等框架。
- Tool Registry + calculator / search / todo 三个工具，由模型自主选择工具，禁止关键词硬编码。
- SQLite 持久化多用户、多 Session；Todo 按 user_id + session_id 隔离。
- Context 包含 System Prompt、最近 10 条消息、历史摘要、工具调用与结果；超阈值压缩且不破坏工具消息结构。
- 控制台 Trace：用户输入、Loop 次数、模型请求、工具名/参数/结果/耗时、最终回答。
- 处理 API 超时、工具不存在、参数错误、工具执行失败、最大轮次限制。
- 原题要求 pytest + FakeLLM + 真实 API 测试。

### 2.2 关键约束变更（用户最终确认）

> 不用用 claude api 什么的，Deepseek 的就行。

> 整体技术路线保留，但需要进一步精简：
> 1. 保留 DeepSeek 原生 API + httpx，不使用 Anthropic SDK。
> 2. 保留自定义 Agent Loop、Tool Registry、Calculator、Search、Todo。
> 3. 保留 SQLite、Session 隔离、Context 管理和基础压缩。
> 4. 保留 Trace，但直接集成在 agent.py 或 main.py 中，不需要单独的 trace.py。
> 5. 删除 FakeLLM、pytest、12 组自动化测试和独立真实 API 测试模块。只保留 README 中的手动验收场景。
> 6. 删除独立 real_demo.py 和 real_run.log，直接通过 CLI 展示实际运行效果。
> 7. 简化 Context 压缩，重点保证完整对话轮次及工具调用结果不被拆散。
> 8. 不增加天气工具、RAG、向量数据库、FastAPI、前端、多 Agent 等功能。
> 9. 最终目录尽量控制在 main.py、llm.py、tools.py、storage.py、context.py 五个核心文件，加 README 和 AI_PROMPTS.md。
> 10. 不要提前增加额外架构设计。

### 2.3 探索阶段的 Prompt（摘要）

- 检查项目目录是否为空、Python 版本、uv 是否可用、已有项目约定。
- 在本机寻找 DeepSeek API Key 的存放位置；确认 `DEEPSEEK_API_KEY` 已在 shell 环境中。
- 调研 DeepSeek 原生 API 的模型、工具调用协议、thinking 模式字段要求。
- 让架构 Agent 评审“最小 Agent”方案，重点排查工具消息结构、压缩边界、Session 隔离、最大轮次等风险。

### 2.4 实现阶段的决策 Prompt（摘要）

- 用 curl 直接验证 `/chat/completions` 的请求/响应形状，而不是凭经验假设。
- 验证 assistant 消息包含 `tool_calls` 时，`reasoning_content` 是否必须回传。
- 验证单轮并行多个工具调用的返回结构。
- 先分别验证 calculator/search/storage/context，再接入真实 Agent Loop，避免问题堆到集成阶段。

## 3. 问题解决记录

### 3.1 选择 DeepSeek 原生接口

最初方案曾考虑 DeepSeek 的 Anthropic 兼容端点（`https://api.deepseek.com/anthropic` + Anthropic SDK），实测也能工作，但用户明确要求“不用 Claude API，DeepSeek 的就行”。最终改用原生 OpenAI 兼容接口：

```text
POST https://api.deepseek.com/chat/completions
Authorization: Bearer $DEEPSEEK_API_KEY
model: deepseek-flash
```

工具协议对应关系：

- assistant 的 `tool_calls` ↔ 题目中的 tool_use；
- `role: "tool"` + `tool_call_id` ↔ 题目中的 tool_result；
- `finish_reason: "tool_calls"` ↔ stop_reason=tool_use。

### 3.2 reasoning_content 必须原样回传（本项目的关键坑）

**现象**：把工具结果以 `role: "tool"` 回传后，如果不带 assistant 消息里的 `reasoning_content`，DeepSeek 返回 HTTP 400：

```text
The `reasoning_content` in the thinking mode must be passed back to the API.
```

**解决**：

- assistant 消息不做字段裁剪，按 API 原始 JSON 存入 `messages` 表并原样回放（包含 `reasoning_content`、`tool_calls`）。
- 工具结果与对应的 assistant(tool_calls) 在同一个事务中写入 `append_exchange()`。
- `reasoning_content` 只在协议层回放：不打印、不进摘要、不提取 Chain of Thought；Trace 只记录结构化工具调用。
- 实测：回放时省略 `tool_calls[].index` 也可以正常工作。

### 3.3 并行工具调用

**现象**：一次请求“同时用计算器算 2**10、搜索 DeepSeek 资料”时，DeepSeek 返回一条 assistant 消息，包含两个 `tool_calls`（calculator 与 search）。

**处理**：Loop 遍历所有 `tool_calls`，依次执行并生成多条 tool 消息，然后一起追加到下一轮上下文。实际运行 Trace：

```text
[响应] Loop 1/8 | tool_calls: calculator, search (1.22s)
[工具] ... calculator({"expression": "2**10"}) -> 2**10 = 1024 ...
[工具] ... search({"query": "DeepSeek"}) -> ... mock://deepseek/api ...
[响应] Loop 2/8 | text (0.93s)
```

### 3.4 最大循环次数与悬空 tool_calls

如果模型在最后一次允许的迭代返回 `tool_calls`，此时如果执行并落盘，但下一轮不允许再请求模型，就会在数据库中留下没有对应 tool 结果的 assistant 消息。

**处理**：在“执行工具之前”判断是否达到 `max_iterations`。达到则直接返回固定提示，不执行、不落盘该响应。实测 `--max-iterations 1` 时：

```text
[响应] Loop 1/1 | tool_calls: calculator (0.61s)
[错误] 达到最大循环次数 1，停止执行
[最终] 已达到最大循环次数（1），本轮中止。 | 总耗时 0.61s
```

数据库中该 Session 只保留了 user 消息，没有悬空工具调用。

### 3.5 上下文压缩不破坏工具结构

**策略**：以“真实用户文本消息”为组起点，组内包含 assistant(tool_calls)、全部 tool 消息与最终回答，直到下一条用户文本。压缩只按整组切分，因此不可能出现“assistant tool_calls 还在、tool 结果被压缩掉”的情况。

- 触发条件：活跃消息数 > `compact_threshold`（默认 20）。
- 保留策略：从最新组向前整组保留，直到至少 `keep_recent`（默认 10）条；实际可能多于 10 条，这是为完整性付出的合理代价。
- 摘要为本地确定性生成，包含用户问题、助手回答、工具名/参数/结果摘要，跳过 `reasoning_content`；摘要上限 3000 字符，与旧摘要合并。
- 摘要和边界 `summary_until_message_id` 同事务保存，重启后自动重新加载。

**实际验证**：session_1 达到 22 条活跃消息后，压缩 10 条较早消息为 680 字摘要，保留 13 条；新进程继续提问时：

```text
[会话] alice/session_1 | 恢复 22 条历史消息 | 摘要 0 字
[压缩] 10 条较早消息 -> 摘要（680 字），保留 13 条，边界消息 #10
[最终] (12+8)*3 = **60**（早前由计算器工具算得）。
```

说明即使原始计算记录被压缩，模型仍可依据摘要正确回答，并且后续带工具追问 `60*2` 正常得到 120。

### 3.6 Session / Todo 隔离与恢复

- Session 用 `(user_id, name)` 唯一约束；消息表按内部 `session_id` 隔离。
- Todo 的 add/list/complete 全部同时按 `user_id + session_id` 过滤。
- 每条单轮命令都是新进程，天然的“重启恢复”测试：alice/session_1 的待办在 alice/session_2、bob/session_1 中均不可见；重启后仍能读取自己的待办。
- 修复了一个小问题：早期用 `INSERT OR IGNORE` 创建 Session 会消耗 AUTOINCREMENT id，改为先查询、不存在再插入。

### 3.7 API 异常处理

实际验证了两种异常：

- 连接失败：`DEEPSEEK_BASE_URL=http://127.0.0.1:9` → `[错误] API connection: 网络错误：[Errno 61] Connection refused`，返回友好提示，进程不崩溃。
- 认证失败：`DEEPSEEK_API_KEY=sk-invalid` → `[错误] API http: HTTP 401 ...`，同样正常兜底。
- 异常后再次用正常配置请求同一 Session 可以继续对话（DeepSeek 接受连续 user 消息）。

### 3.8 工具异常

- calculator 的 `1/0` 返回 `[工具错误] 除数不能为 0`，模型收到错误 tool 消息后向用户解释，Loop 正常结束。
- calculator 使用 AST 白名单而非 `eval`；已实测拒绝 `__import__('os')`、`2**999`、未闭合括号等输入。
- 未知工具、非 JSON 参数、非对象参数由 `ToolRegistry` 统一转换为错误 tool 消息。

### 3.9 精简与测试补充

- 最初按用户精简要求删除了 pytest、FakeLLM、自动化测试、独立真实 API 测试脚本和运行日志。
- 后续按新要求补充自动化测试：`tests/` 下新增 FakeLLM 与 59 个 pytest 用例、`pytest.ini`，requirements 增加 pytest；生产代码未修改，59 个用例全部通过。
- 将 Trace 直接集成进 `main.py`，不单独建 `trace.py`。
- 核心代码压缩为五个文件；其余为 README、AI_PROMPTS、requirements.txt、pytest.ini 与 tests/。
- 搜索只保留本地 Mock，不引入天气工具、RAG、向量库、FastAPI、前端、多 Agent。
- 真实 API 验证仍保留为 CLI 手动验收场景，结果记录在 README 第 10 节。

## 4. 运行验证摘要

使用真实 DeepSeek API（`deepseek-flash`）实际跑通：

1. 普通对话与历史恢复；
2. calculator 计算 `(12+8)*3 = 60`；
3. search 返回 Mock 结果；
4. todo add/list/complete 跨进程持久化；
5. 单轮并行 calculator + search；
6. 纯对话追问与带工具追问；
7. alice/session_1、alice/session_2、bob/session_1 数据隔离；
8. 上下文压缩（10 条 → 680 字摘要）及压缩后召回；
9. `--max-iterations 1` 最大轮次限制；
10. 工具除零、API 连接失败、HTTP 401 异常兜底。

后续补充自动化测试：59 个 pytest 用例全部通过（FakeLLM + 临时 SQLite + httpx.MockTransport，默认不访问真实 API）。

详细命令见 README 的“手动验收场景”与“自动化测试”。
