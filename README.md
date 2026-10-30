# dsh-ollama-native

DeepSeek Harness 的 Ollama 模型接入插件。直接走 Ollama 自己的原生协议（`/api/chat` 的
NDJSON 流式输出），不经过任何 OpenAI 兼容层。

- **原生流式**：逐行解析 `{"message":{...},"done":...}`，转换成 harness 的块协议
  （`block-start` / `text-delta` / `reasoning-delta` / `tool-call-delta` / `block-end`）。
  `thinking`、`content`、`tool_calls` 可以在同一行里同时出现。
- **原生思考**：读取 `/api/show` 的 `thinking` 描述，生成 harness 的 reasoning-effort 开关；
  选择结果映射到请求体的 `think` 字段（布尔模型给 `off`/`on`，等级模型原样透传）。
- **原生工具调用**：模型声明 `tools` 能力时才发送工具声明；`tool_calls` 的参数按块累计，
  完成事件统一收敛为 `tool-calls`（有些 Ollama 版本在调用工具时仍回 `done_reason: "stop"`）。
- **视觉输入**：模型声明 `vision` 才把图片块按 base64 附到用户消息；图片读取失败时降级成
  文本占位符，不丢整个回合。
- **模型元数据**：`/api/tags` + `/api/show` 解析上下文长度、参数量化信息、能力集合，5 分钟缓存。
- **模型发现**：实现 `llm.registerModelDiscovery`，设置界面可以直接向 Ollama 查询模型列表。
- **配置即时生效**：连接与采样参数都是 volatile 字段，在 Plugins 页面改动直接作用到运行中的
  适配器，不需要重启。

## 文件结构

| 文件 | 作用 |
| --- | --- |
| `index.js` | 插件入口：注册适配器、可配置 provider、模型发现 |
| `lib/harness.js` | 从运行中的 dsh 运行时加载 `@deepseek-ai/dsh-llm` 等模块（见下） |
| `lib/config.js` | Config 模式（schemastery）、URL 规范化、选项解析 |
| `lib/client.js` | Ollama HTTP 访问与错误映射 |
| `lib/wire.js` | 历史消息 → Ollama 线格式（含图片 base64、工具调用） |
| `lib/body.js` | `/api/chat` 请求体：采样参数、工具、`think`、`keep_alive` |
| `lib/chunks.js` | NDJSON → harness 流块 |
| `lib/ndjson.js` | NDJSON 分帧 |
| `lib/adapter.js` | `LlmAdapter` 实现：元数据缓存、空闲看门狗、失败归类 |
| `test/protocol.test.mjs` | 15 个离线协议测试 |

### 为什么要 `lib/harness.js`

profile 里的插件是按**路径**导入的，插件自己的裸导入（`import ... from '@deepseek-ai/dsh-llm'`）
只会从插件目录向上查找，而 profile 的 `node_modules` 里只有 `schemastery` / `cosmokit`。
`lib/harness.js` 因此按顺序尝试：普通裸导入 → 运行时自带的副本（绝对 file URL）。命中后者时
拿到的是**同一个模块实例**，所以 `extends LlmAdapter`、`instanceof LlmError` 等语义都成立。

## 安装状态

- bundle 名：`dsh-ollama-native`，行 id：`llm-ollama-native`，provider 路由：`ollama`
- 已写入 profile：`C:\Users\dc666\.dsh\profiles\desktop\package.json` 的 dependencies 与
  `dsh.profile.bundles`
- 安装方式是 **link**（junction → 本目录），所以移动/删除本目录会让插件失效
- `cordis.patch.yml` 里有一条 `- id: llm-ollama：disabled: true` 的防御性覆盖：若以后重新安装
  第三方 `dsh-ollama`，它不会和本插件争同一条 `ollama` 路由（匹配不到时会被告警并跳过，无副作用）

## 配置

在 GUI 的 Plugins 页面直接编辑（默认值如下）：

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `baseURL` | `http://127.0.0.1:11434` | Ollama 地址，可带路径前缀；末尾斜杠自动去掉；只接受 http(s) |
| `providerId` | `ollama` | provider 路由，非 volatile：改动会重新挂载该行 |
| `displayName` | `Ollama` | 模型选择器里显示的名字 |
| `defaultContextWindow` | `131072` | 模型不上报上下文长度时的兜底 |
| `maxTokens` | `8192` | 默认输出上限 |
| `streamIdleTimeoutMs` | `300000` | 流式空闲超时；超时以 `TIMEOUT` 结束，可中断挂死的服务 |
| `requestTimeoutMs` | `10000` | `/api/tags`、`/api/show` 这类元数据请求的超时 |
| `models` | `[]` | 覆盖单个模型的上下文长度/模态/输出上限 |
| `params` | `{}` | Ollama 原生选项，放进请求体的 `options`；`keep_alive`、`think` 是顶层字段 |

`params` 的键就是 Ollama 的选项名（`num_ctx`、`top_k`、`top_p`、`min_p`、`seed`、
`repeat_penalty`、`mirostat*`、`num_predict`、`stop` 等）。字符串形式的数字/布尔值会被规范化
（`"0.9"` → `0.9`，`"true"` → `true`），`stop` 接受数组或逗号/换行分隔的字符串。

优先级：调用方显式传入的 temperature / maxTokens / stop 先落地，随后被配置里的 `params` 覆盖——
配置是这个 provider 的用户显式选择。

## 使用

本插件只负责「接入」。本机还没有任何 Ollama 模型，需要先在终端拉一个：

```bash
ollama pull qwen2.5:0.5b     # 或任何你想要的模型
```

拉好后在模型选择器里选择即可，不需要重启。工具调用需要模型自身声明 `tools` 能力（如
`qwen2.5`、`llama3.1`、`qwen3` 等）；思考开关需要模型声明 `thinking` 能力（如 `qwen3`、`deepseek-r1`）。

## 验证状态

已验证（离线）：

- 15 个协议测试全部通过（`node test/protocol.test.mjs`）：NDJSON 分帧、思考/文本/工具块顺序、
  双工具调用拆分、空响应、异常行、provider 错误行、`length`/未知 `done_reason`、消息序列化
  （system/developer/user/图片/assistant 工具调用/tool 结果）、图片字节读取失败时的降级、
  请求体（采样参数、配置覆盖、工具能力门控、`think` 映射、`keep_alive`）。
- Config 模式：默认值、volatile 引用原地改写后的重新解析、URL 规范化与非法值拒绝。

已验证（运行中的进程）：

- `plugin_manager` 安装结果 `application: applied`，`warnings: []`；
  `list_plugins` 报告 `include:llm-ollama-native` 为 `enabled: true, fiberPhase: active`。
- 通过 `llm` 服务实际查询：`listProviders()` 含 `{id: "ollama", name: "Ollama"}`；
  `listConfigurableProviders()` 含 `{provider: "ollama", settingsNs: "llm-ollama-native",
  defaultBaseURL: "http://127.0.0.1:11434"}`；`listModels("ollama")` 返回 `[]`（真实请求了本机
  `/api/tags`，当前无模型）；`resolveModelInfo("ollama", "missing:latest")` 返回
  `contextWindow: 131072`、`defaultMaxTokens: 8192`；`discoverModels("llm-ollama-native", …)`
  返回 `[]`；对照组 `listModels("definitely-not-registered")` 报 `NO_ADAPTER`。
- 服务端契约核对（本机 Ollama 0.40.11）：`/api/version` → `{"version":"0.40.11"}`；
  `/api/tags` → `{"models":[]}`；`/api/chat`、`/api/show` 对不存在的模型返回 404
  `{"error":"model '...' not found"}`，对应本插件的 `NOT_FOUND` 映射。

未验证：

- 真实模型的流式对话与工具回合：本机 Ollama 里没有模型，按你的要求没有下载。
- 界面渲染：这台机器没有可用的浏览器控制通道（`list_windows` 的 browser lane 为空），
  所以模型选择器里的 `Ollama` 条目、Plugins 页面上的标题/图标都没有做视觉确认——只确认了
  Host 侧注册与 schema 投影。

## 回滚

```text
plugin_manager remove_bundle  dsh-ollama-native
```

随后可以从 GitHub 重新安装原来的第三方插件；本插件没有改动 profile 的其它文件。
