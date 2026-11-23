# Vflow: Discovering Optimal Agentic Workflows for Verilog Generation

[English](README.md) | **中文**

论文 [VFlow (arXiv:2504.03723v1)](https://arxiv.org/abs/2504.03723v1) 的实现。如果本项目对你有帮助，请引用论文（见[第 9 节](#9-引用)）。

---

## 0. 结果

每个 workflow 都跑 2 遍取平均。执行器是 Qwen3.8-27B-FP8。

| workflow | 测试通过率 | 每题平均 token | 说明 |
|---|---|---|---|
| IO（起点） | 0.524 | 966 | 直接让模型写一次 |
| **af27 搜索结果** | **0.791** | 4494 | 本目录 `examples/af27_best/` |
| 人工设计 handcrafted | 0.777 | 8092 | 对照：采样 5 份 → 编译修复 → 仿真投票 |

- af27 比 IO 高 **26.7 个点**。按题配对比较是 35 题赢、1 题输。
- af27 和人工设计的 handcrafted 基本持平，token 少一半以上。
- 整个搜索从 IO 起步（`workflows/io_staged`，与 IO 等价）。

---

## 1. 先认识 6 个名词

| 名词 | 是什么 | 在哪 |
|---|---|---|
| **执行器（executor）** | workflow 里实际调用的大模型。af27 用的是本地 vLLM 部署的 Qwen3.8-27B-FP8，也可以换成 OpenAI、DeepSeek 等 API。在 `aflow.toml` 里配置 | `vflowx/llm.py` |
| **workflow** | 一个目录，里面有 `graph.py` 和 `prompt.py`。`graph.py` 必须定义 `async def solve(spec, ops) -> str`：输入自然语言规格，输出 Verilog 代码 | `workflows/`、`examples/af27_best/` |
| **ops（工具箱）** | workflow 能调用的积木，例如 `generate` 生成代码、`repair` 按报错修改、`review` 审查、`lint` 编译检查等 | `vflowx/ops.py` |
| **优化器（optimizer）** | 改 workflow 代码的模型，由一个编程 agent 程序（**harness**：Codex CLI 或 DeepSeek Harness）驱动。两者都在 `aflow.toml` 里配置。它读一份任务说明 `CONTEXT.md`，然后直接修改 workflow 的代码 | `vflowx/search.py` |
| **得分（score）** | `score = 通过率 − 0.0015 × 每题平均token / 1000`。前一项是做对了多少，后一项是花了多少 | `vflowx/evaluate.py` |
| **数据划分** | VerilogEval-v2 共 155 题，固定分成 **val 52 题**（搜索时打分用）和 **test 103 题** | `data/splits.json` |


---

## 2. 搜索一轮在做什么

```
                 ┌─────────────────────────────────────────────────────────────┐
                 │ 排行榜（每个版本在 val 上的 score）+ 经验日志（谁改了什么，有没有用）│
                 └───────────────┬─────────────────────────────────────────────┘
   ① 选父节点：从分数前 4 名里随机抽（分高的更容易被抽到）
                                 ▼
   ② 复制父节点到 rounds/round_N/，写一份 CONTEXT.md 给优化器
      （任务说明、目标函数、排行榜、经验日志、父节点在 val 上做错的 6 道题）
                                 ▼
   ③ 优化器读 CONTEXT.md，修改 graph.py / prompt.py，写 modification.md 说明改了什么
                                 ▼
   ④ 静态检查（禁止读文件、联网等作弊行为）+ 冒烟测试（跑 2 题看会不会崩溃）
      └─ 不通过：让优化器修一次；还不行就丢弃这一版
                                 ▼
   ⑤ 在 val 上完整评估（52 题 × 2 遍），写入排行榜和经验日志
```

- **每次迭代** 并行产生 `width=2` 个新版本。
- **何时停止**：连续 `patience` 次迭代都没刷新最高分就早停，否则跑满 `rounds` 轮。
- **收尾（finalize）**：取 val 分数前 3 的版本，各在 val 上再跑 3 遍，降低运气成分，选出最好的复制到 `runs/<名字>/best/`。
- **父节点怎么选**：采用 AFlow 的混合概率。从前 4 名里，以 30% 的权重均匀抽，70% 的权重按 `softmax(0.2 × score × 100)` 抽。这样大多数时候在好版本上继续改，偶尔也给其他版本机会。起点 round 1 始终留在候选里。
- **每轮的侧重点（focus）**：按轮号轮换"结构 / 提示词与采样 / 效率"三类，避免并行的两个优化器改同一类东西。

---

## 3. 目录结构

```
vflow/
├── README.md               英文说明
├── README_zh.md            ← 你正在看的文件（中文说明）
├── aflow.toml              ★ 模型配置：执行器、优化器用哪个模型，用哪个 harness（只改这里）
├── requirements.txt        Python 依赖（openai、httpx）
├── vflowx/                 核心代码（Python 包）
│   ├── search.py           ★ AFlow 搜索主循环（第 8 节有逐函数导读）
│   ├── evaluate.py         在某个数据划分上跑一个 workflow 并打分
│   ├── ops.py              workflow 可用的工具箱（也就是"搜索空间"）
│   ├── config.py           读 aflow.toml，把 "服务商/模型" 解析成地址、key、参数
│   ├── harness.py          两个优化器 harness（codex / dsh）：用配置里的模型、以非交互方式跑一轮任务
│   ├── llm.py              执行器客户端（vLLM / OpenAI / DeepSeek / 兼容接口），同时统计每题的 token
│   ├── eda.py              iverilog/verilator/yosys 封装：编译、仿真、判分
│   ├── formal.py           yosys 形式等价检查（ops.equiv 会用到）
│   ├── bench.py            读取数据集，固定 val/test 划分
│   └── cli.py              把 workflow 当工具用：输入你的规格，输出 Verilog
├── workflows/
│   ├── io/                 最简单的基线：调用一次模型
│   └── io_staged/          ★ 搜索起点：和 io 完全等价，只是拆成 generate→repair→select 三个函数
├── examples/
│   ├── af27_best/          af27 搜出来的最终 workflow（第 6 节有讲解）
│   └── af27_run/           af27 的排行榜、经验日志、最终选择和完整运行日志
├── scripts/
│   ├── env.example.sh      环境变量模板：路径和 API key（复制成 env.sh 后修改）
│   ├── serve_vllm.sh       启动 vLLM
│   ├── check_setup.py      ★ 环境自检
│   ├── run_aflow.sh        启动一次搜索
│   ├── eval_test.sh        在测试集上评估
│   └── test_config.py      离线测试配置解析（不联网）
└── data/
    └── splits.json         固定的 val/test 划分（务必别改，否则结果没法和 af27 比）
```

---

## 4. 快速开始

### 4.1 安装

前 3 步每个人都要做：

**第 1 步：建 Python 环境（3.12）。**
```bash
python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

**第 2 步：下载数据集。**
```bash
git clone https://github.com/NVlabs/verilog-eval.git data/verilog-eval && git -C data/verilog-eval checkout c498220
```

**第 3 步：配置 env.sh。**
```bash
cp scripts/env.example.sh scripts/env.sh
```
然后打开 `scripts/env.sh`，填两样东西：
- EDA 工具的路径：从 [oss-cad-suite](https://github.com/YosysHQ/oss-cad-suite-build/releases) 下载解压，里面已包含 iverilog、verilator 和 yosys。
- 要用到的 API key。

### 4.2 选模型

在 [`aflow.toml`](aflow.toml) 里改三行。模型的格式都是 `服务商/模型`：

```toml
executor  = "deepseek/deepseek-chat"    # 执行器：写 Verilog 的模型
optimizer = "deepseek/deepseek-flash"   # 优化器：改 workflow 代码的模型
harness   = "dsh"                       # 驱动优化器的编程 agent：codex 或 dsh
```

| 你手上有 | executor | optimizer | harness |
|---|---|---|---|
| 显卡（复现 af27，默认配置） | `local/qwen3.8-27b-fp8` | `""`（codex 自己的登录） | `codex` |
| DeepSeek key | `deepseek/deepseek-chat` | `deepseek/deepseek-flash` | `dsh` |
| OpenAI key | `openai/gpt-5.4-mini` | `openai/gpt-5.5` | `codex` 或 `dsh` |

- **harness 只是 agent 程序，用哪家模型由 `optimizer` 决定。** 两个 harness 都能接 `aflow.toml` 里的任意服务商：
  - `optimizer` 写 `服务商/模型`：key 从该服务商的 `api_key_env` 读。
  - 只写模型名，或者留空：用 harness 自己的登录。
- **安装你选的 harness：**
  ```bash
  npm i -g @openai/codex          # codex；用它自己的登录时再运行 codex login
  ```
  ```bash
  npm i -g @deepseek-ai/dsh       # dsh，需要 Node.js ≥ 24.9
  ```
- **executor 是 `local/…`（本地部署）时**，需要 2 张 32GB 显卡。装 vLLM，再单独开一个终端启动它，并保持运行：
  ```bash
  pip install vllm==0.34.0
  ```
  ```bash
  source scripts/env.sh && bash scripts/serve_vllm.sh /path/to/Qwen3.8-27B-FP8
  ```

### 4.3 运行

```bash
source scripts/env.sh
python scripts/check_setup.py              # 自检：全部 [OK] 再往下
bash scripts/run_aflow.sh demo             # 小规模搜索（8 轮），进度看 tail -f logs/demo.log
bash scripts/eval_test.sh runs/demo/best   # 在测试集上评估搜出来的 workflow
```

- **复现 af27：**
  ```bash
  ROUNDS=48 PATIENCE=8 bash scripts/run_aflow.sh af27_repro
  ```
- **临时换模型：** 在命令后面加参数，不用改 `aflow.toml`。
  ```bash
  bash scripts/run_aflow.sh demo --executor deepseek/deepseek-chat --optimizer deepseek/deepseek-flash --harness dsh
  ```
- **用搜出来的 workflow 生成你自己的模块：**
  ```bash
  python -m vflowx.cli --spec my_spec.txt --workflow runs/demo/best -o my_mod.v
  ```

<details>
<summary><b>更多配置细节</b>（多个 vLLM 副本、其他 API 服务、推理模式、费用与隐私）</summary>

- **多个 vLLM 副本：** 在 `[providers.local]` 的 `base_url` 里写多个地址，请求会自动分摊。启动第二个副本：
  ```bash
  GPUS=2,3 PORT=8001 bash scripts/serve_vllm.sh /path/to/model
  ```
- **其他 OpenAI 兼容服务：** 在 `aflow.toml` 里照抄一段 `[providers.xxx]`，写上 `base_url` 和 `api_key_env`，然后用 `xxx/<模型名>`。
- **可选项：** 每个服务商都可以加 `concurrency`（并发数，API 默认 16）、`max_tokens`（回复上限，默认 6144）、`think_effort`、`think_model`。
- **推理模式（`think=True`）在各家的含义：**

  | 服务商 | think=True 时 |
  |---|---|
  | `local`（vLLM + Qwen3） | 打开 chat template 的 `enable_thinking` |
  | `openai`，推理模型（gpt-5*, o*） | `reasoning_effort` 取 `think_effort`（默认 medium）；think=False 时用 `none` |
  | `deepseek` | 换成 `think_model`（默认 `deepseek-reasoner`） |
  | 其他 | 设置了 `think_model` 就换成它，否则不变 |

- **优化器用本地模型：** 模型要支持工具调用，vLLM 启动时要加 `--enable-auto-tool-choice --tool-call-parser hermes`。
- **推理强度：** `--effort`（默认 high）会传给 harness；dsh 驱动 DeepSeek 模型时换算成它的档位：low→low，medium/high→high，xhigh→max。
- **费用：** 一次完整搜索要评估几十个版本，每个版本是 52 题 × 遍数。先用小的 `ROUNDS` 或 `python -m vflowx.evaluate --limit 3` 估算一下。
- **网络：** 外部 API 走系统代理（`HTTPS_PROXY`），本机地址一律不走代理。
- **隐私：** dsh 默认会把会话日志上传给 DeepSeek，并开启遥测。本项目为每次运行生成的 dsh 配置 `runs/<名字>/dsh.patch.yml` 把两者都关掉了。
</details>

## 5. 输出文件

### 5.1 `runs/<名字>/`
| 文件 | 内容 |
|---|---|
| `leaderboard.json` | 每一轮的 score、pass_rate、avg_tokens、改动说明。最值得看 |
| `experience.jsonl` | 每条记录"父→子"的分数变化和改动内容，优化器下一轮会读到它 |
| `rounds/round_N/graph.py`, `prompt.py` | 第 N 个版本的 workflow |
| `rounds/round_N/CONTEXT.md` | 这一轮发给优化器的完整任务说明。想知道优化器"看到了什么"就读它 |
| `rounds/round_N/modification.md` | 优化器自己写的改动说明 |
| `rounds/round_N/<harness>_opt.log` | 优化器的完整操作过程：读了哪些文件、跑了什么命令 |
| `rounds/round_N/eval/results.jsonl` | 每题、每遍的结果（见 5.2 节） |
| `final.json`, `best/` | 收尾阶段前 3 名的复测结果，以及最终选中的 workflow |

### 5.2 `results.jsonl` 的一行
```
{"pid": "Prob035_count1to10", "rep": 0, "pass": true, "status": "pass", "total_tokens": 1830, "calls": 2, "code": "module TopModule ...", ...}
```
`status` 有以下几种取值：

| status | 含义 |
|---|---|
| `pass` | 通过 |
| `mismatch` | 能编译，但仿真输出和参考答案不一致 |
| `compile` | 编译不过 |
| `syntax` | 输出里根本没有可用的代码 |
| `timeout` | 超时 |
| `workflow_error` | workflow 自己崩溃了 |

### 5.3 日志里的一行
```
21:04:57 round 3: score=0.7765 pass=0.7788 tokens=1595.3 | 'Added a compiler-gated repair loop ...'
21:04:57 best so far: round 3 score=0.7765 (stale=0)
```
`stale` 是连续多少次迭代没有刷新最高分，到 `patience` 就早停。

---

## 6. vflow 搜出了什么

最终版本 round 27 的演化路径是 `1 → 3 → 5 → 6 → 10 → 27`，每一步都是 codex 自己提出的改动：

| 轮 | val 通过率 | token | 改动 |
|---|---|---|---|
| 1 | 0.558 | 1030 | 起点：调用一次模型 |
| 3 | 0.779 | 1595 | **编译不过就按编译器报错修**，最多修两次，每次修完重新编译检查 |
| 5 | 0.760 | 1528 | 用正则自动修一种常见错误（output 和 reg 重复声明），省一次 LLM 调用 |
| 6 | 0.789 | 2478 | 编译通过后，用推理模式做一次**语义审查**；有问题才修，修完还得能编译才采用 |
| 10 | 0.875 | 6850 | 加强审查（codex 没写说明，具体改动见 `rounds/round_10`） |
| 27 | 0.885 | 6381 | 两次修复后仍编译不过，就用推理模式**从规格重新生成一次** |

最终 workflow（[`examples/af27_best/graph.py`](examples/af27_best/graph.py)，共 130 行）的流程是：
```
generate: 普通模式生成 1 份代码
   ▼
repair:   编译 → 不过就先试正则修复 → 再按报错让模型修（最多 2 次）
   ▼
select:   仍编译不过 → 推理模式从规格重写一份，编译通过就用它
          编译通过   → 推理模式审查一遍：说没问题就直接用；
                       指出问题 → 让模型修 → 修完能编译才替换
   ▼
solve:    整体限时 840 秒，超时返回已有的最好结果
```

---

## 7. 常见问题

- **codex / dsh 一直不动 / 退出码非 0：** 先运行 `python scripts/check_setup.py`，再看 `rounds/round_N/<harness>_opt.log` 里的报错。每轮优化器最多给 30 分钟。
- **分数上下跳几个点：** 正常。val 只有 52 题，每题差 1 题就是约 2 个点。所以搜索中每个版本都跑 2 遍，收尾时再对前 3 名各跑 3 遍。比较两个版本时不要只看一次的分数。
- **只想比准确率，不管 token：** 设置 `export VFX_MU=0`，这样 score 就等于通过率。
- **想换数据集：** 准备好 RTLLM-2.0 放到 `data/RTLLM` 后，搜索时加 `--val-split rtllm_val`，评估用 `--split rtllm_test`。
- **API 报 401 / 429：** 401 是 key 不对或没设置，`aflow.toml` 里 `api_key_env` 写的那个变量要 export；429 是被限流，调小那个服务商的 `concurrency`。
- **评估很慢：** 每个 vLLM 副本默认 96 路并发，可以在 `aflow.toml` 里调；仿真并行数用 `VFX_EDA_JOBS` 调，默认 48。最有效的提速办法是多起一个 vLLM 副本。
- **workflow 会不会偷看答案？** 不会。workflow 只拿到自然语言规格，静态检查禁止读文件、调用子进程和联网（`search.py` 的 `FORBIDDEN`）。参考答案和测试平台只在 `eda.score_verilogeval` 里给评估器用。
- **复现结果和 af27 不完全一样：** LLM 采样有随机性，codex 每次的改法也不同，所以同样的设置每次搜出来的 workflow 会不一样。


---

## 8. 代码导读：`vflowx/search.py`

| 行号 | 函数 | 作用 |
|---|---|---|
| 33 | `FORBIDDEN` | 静态检查的黑名单正则 |
| 37 | `FOCI` | 三种轮换的修改侧重点 |
| 97 | `select_parent` | ① 选父节点（混合 softmax） |
| 110 | `failure_digest` | 从父节点的 val 结果里随机挑 6 道错题，附上规格、代码和判分提示 |
| 129 | `executor_text` | 告诉优化器执行器是什么模型、推理模式怎么开 |
| 141 | `write_context` | ② 生成给优化器的 `CONTEXT.md` |
| 200 | `codex` | ③ 用配置的 harness 跑一轮优化器（命令行在 `vflowx/harness.py` 里生成），限时 30 分钟 |
| 214 / 229 | `static_check` / `smoke` | ④ 静态检查和冒烟测试 |
| 236 | `expand` | 一个子节点的完整流程 ②→⑤ |
| 285 | `run_search` | 主循环：评估起点，然后每次迭代并行 `expand` `width` 次，并判断早停 |
| 308 | `finalize` | 收尾：前 3 名在 val 上各跑 3 遍，选出 `best/` |

建议按这个顺序读：`workflows/io_staged/graph.py`（看懂 workflow 长什么样）→ `vflowx/ops.py`（有哪些积木）→ `vflowx/search.py`（搜索怎么改它）→ 任意一轮的 `CONTEXT.md`（codex 实际看到的任务）。

命令行参数：
```bash
python -m vflowx.search --help
```
| 参数 | 默认 | 含义 |
|---|---|---|
| `--run` | 必填 | 输出目录 |
| `--init` | `workflows/io` | 起点 workflow，可以用逗号隔开多个 |
| `--rounds` | 12 | 最多生成多少个版本（包括起点） |
| `--width` | 2 | 每次迭代并行的子节点数 |
| `--repeats` | 1 | 每个版本在 val 上跑几遍（af27 用 2） |
| `--topk` | 4 | 父节点候选池大小 |
| `--patience` | 4 | 连续多少次迭代没进步就早停 |
| `--executor` | `aflow.toml` | 临时换执行器，如 `deepseek/deepseek-chat` |
| `--optimizer` | `aflow.toml` | 临时换优化器模型，如 `deepseek/deepseek-flash` |
| `--harness` | `aflow.toml` | 临时换 harness：`codex` 或 `dsh` |
| `--effort` | high | 优化器的推理强度 |
| `--val-split` | val | `val`（VerilogEval）或 `rtllm_val` |
| `--finalize-only` | | 只做收尾，适合搜索中断后补跑 |

---

## 9. 引用

如果本项目对你的研究有帮助，请引用：

```bibtex
@article{wei2025vflow,
  title   = {VFlow: Discovering Optimal Agentic Workflows for Verilog Generation},
  author  = {Wei, Yangbo and Huang, Zhen and Li, Huang and Xing, Wei W. and Lin, Ting-Jung and He, Lei},
  journal = {arXiv preprint arXiv:2504.03723},
  year    = {2025},
  url     = {https://arxiv.org/abs/2504.03723}
}
```
