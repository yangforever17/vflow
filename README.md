# Vflow: Discovering Optimal Agentic Workflows for Verilog Generation

**English** | [中文](README_zh.md)

An implementation of [VFlow (arXiv:2504.03723v1)](https://arxiv.org/abs/2504.03723v1). If you find this project useful, please cite the paper (see [Section 9](#9-citation)).

---

## 0. Results

Each workflow is run twice and the results are averaged. The executor is Qwen3.8-27B-FP8.

| workflow | test pass rate | avg tokens / problem | notes |
|---|---|---|---|
| IO (starting point) | 0.524 | 966 | ask the model once |
| **af27 search result** | **0.791** | 4494 | `examples/af27_best/` in this repo |
| handcrafted | 0.777 | 8092 | baseline: sample 5 → compile-and-repair → simulation voting |

- af27 beats IO by **26.7 points**. Per-problem paired comparison: 35 wins, 1 loss.
- af27 is on par with the handcrafted workflow while using less than half the tokens.
- The whole search starts from IO (`workflows/io_staged`, equivalent to IO).

---

## 1. Six terms to know first

| Term | What it is | Where |
|---|---|---|
| **Executor** | The LLM actually called inside a workflow. af27 uses Qwen3.8-27B-FP8 served locally with vLLM; it can be swapped for OpenAI, DeepSeek, or other APIs. Configured in `aflow.toml` | `vflowx/llm.py` |
| **Workflow** | A directory containing `graph.py` and `prompt.py`. `graph.py` must define `async def solve(spec, ops) -> str`: natural-language spec in, Verilog code out | `workflows/`, `examples/af27_best/` |
| **ops (toolbox)** | The building blocks a workflow can call, e.g. `generate`, `repair` (fix from error messages), `review`, `lint` (compile check) | `vflowx/ops.py` |
| **Optimizer** | The model that edits workflow code, driven by a coding-agent program (the **harness**: Codex CLI or DeepSeek Harness). Both are configured in `aflow.toml`. It reads a task description `CONTEXT.md` and edits the workflow code directly | `vflowx/search.py` |
| **Score** | `score = pass_rate − 0.0015 × avg_tokens_per_problem / 1000`. The first term is how much it gets right, the second is how much it costs | `vflowx/evaluate.py` |
| **Data split** | VerilogEval-v2 has 155 problems, split once and for all into **val (52)** (scored during search) and **test (103)** | `data/splits.json` |

---

## 2. What one search round does

```
                 ┌──────────────────────────────────────────────────────────────────┐
                 │ Leaderboard (val score of every version) + experience log        │
                 │ (who changed what, and whether it helped)                        │
                 └───────────────┬──────────────────────────────────────────────────┘
   ① Pick a parent: sample from the top 4 (higher scores are more likely)
                                 ▼
   ② Copy the parent to rounds/round_N/ and write CONTEXT.md for the optimizer
      (task, objective, leaderboard, experience log, 6 val problems the parent failed)
                                 ▼
   ③ The optimizer reads CONTEXT.md, edits graph.py / prompt.py, writes modification.md
                                 ▼
   ④ Static check (no file reads, network, or other cheating) + smoke test (2 problems, does it crash?)
      └─ fails: the optimizer gets one fix attempt; if it still fails, the version is discarded
                                 ▼
   ⑤ Full evaluation on val (52 problems × 2 runs), written to the leaderboard and experience log
```

- **Each iteration** produces `width=2` new versions in parallel.
- **Stopping**: early stop after `patience` consecutive iterations without a new best; otherwise run all `rounds`.
- **Finalize**: take the top 3 versions by val score, re-run each 3 more times on val to reduce luck, and copy the best one to `runs/<name>/best/`.
- **Parent selection** uses AFlow's mixed probability: among the top 4, 30% weight uniform and 70% weight `softmax(0.2 × score × 100)`. Most of the time the search builds on good versions, but others still get a chance. Round 1 (the starting point) always stays in the candidate pool.
- **Per-round focus**: rotates by round number among "structure / prompts & sampling / efficiency", so two parallel optimizers don't change the same kind of thing.

---

## 3. Directory layout

```
vflow/
├── README.md               ← this file (English)
├── README_zh.md            Chinese README
├── aflow.toml              ★ model config: executor, optimizer, harness (edit only this)
├── requirements.txt        Python dependencies (openai, httpx)
├── vflowx/                 core code (Python package)
│   ├── search.py           ★ AFlow search loop (walkthrough in Section 8)
│   ├── evaluate.py         run a workflow on a data split and score it
│   ├── ops.py              toolbox available to workflows (i.e. the "search space")
│   ├── config.py           reads aflow.toml, resolves "provider/model" into URL, key, parameters
│   ├── harness.py          the two optimizer harnesses (codex / dsh): run one task non-interactively with the configured model
│   ├── llm.py              executor client (vLLM / OpenAI / DeepSeek / compatible APIs), counts tokens per problem
│   ├── eda.py              iverilog/verilator/yosys wrappers: compile, simulate, grade
│   ├── formal.py           yosys formal equivalence check (used by ops.equiv)
│   ├── bench.py            dataset loading, fixed val/test split
│   └── cli.py              use a workflow as a tool: your spec in, Verilog out
├── workflows/
│   ├── io/                 simplest baseline: one model call
│   └── io_staged/          ★ search starting point: equivalent to io, split into generate→repair→select
├── examples/
│   ├── af27_best/          the final workflow found by af27 (explained in Section 6)
│   └── af27_run/           af27 leaderboard, experience log, final selection and full run log
├── scripts/
│   ├── env.example.sh      env-var template: paths and API keys (copy to env.sh and edit)
│   ├── serve_vllm.sh       start vLLM
│   ├── check_setup.py      ★ environment self-check
│   ├── run_aflow.sh        start a search
│   ├── eval_test.sh        evaluate on the test split
│   └── test_config.py      offline config-parsing test (no network)
└── data/
    └── splits.json         fixed val/test split (do not change it, or results won't be comparable with af27)
```

---

## 4. Quick start

### 4.1 Install

Everyone needs the first three steps:

**Step 1: Python environment (3.12).**
```bash
python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

**Step 2: Download the dataset.**
```bash
git clone https://github.com/NVlabs/verilog-eval.git data/verilog-eval && git -C data/verilog-eval checkout c498220
```

**Step 3: Configure env.sh.**
```bash
cp scripts/env.example.sh scripts/env.sh
```
Then open `scripts/env.sh` and fill in two things:
- The EDA tool path: download and unpack [oss-cad-suite](https://github.com/YosysHQ/oss-cad-suite-build/releases), which includes iverilog, verilator and yosys.
- The API keys you will use.

### 4.2 Choose models

Edit three lines in [`aflow.toml`](aflow.toml). Models are always written as `provider/model`:

```toml
executor  = "deepseek/deepseek-chat"    # executor: the model that writes Verilog
optimizer = "deepseek/deepseek-flash"   # optimizer: the model that edits workflow code
harness   = "dsh"                       # coding agent driving the optimizer: codex or dsh
```

| You have | executor | optimizer | harness |
|---|---|---|---|
| GPUs (reproduce af27, default config) | `local/qwen3.8-27b-fp8` | `""` (codex's own login) | `codex` |
| DeepSeek key | `deepseek/deepseek-chat` | `deepseek/deepseek-flash` | `dsh` |
| OpenAI key | `openai/gpt-5.4-mini` | `openai/gpt-5.5` | `codex` or `dsh` |

- **The harness is just the agent program; `optimizer` decides which model it uses.** Both harnesses can use any provider in `aflow.toml`:
  - `optimizer` set to `provider/model`: the key is read from that provider's `api_key_env`.
  - Only a model name, or empty: uses the harness's own login.
- **Install the harness you chose:**
  ```bash
  npm i -g @openai/codex          # codex; run codex login if using its own login
  ```
  ```bash
  npm i -g @deepseek-ai/dsh       # dsh, requires Node.js ≥ 24.9
  ```
- **If the executor is `local/…` (self-hosted)**, you need 2 × 32 GB GPUs. Install vLLM, then start it in a separate terminal and keep it running:
  ```bash
  pip install vllm==0.34.0
  ```
  ```bash
  source scripts/env.sh && bash scripts/serve_vllm.sh /path/to/Qwen3.8-27B-FP8
  ```

### 4.3 Run

```bash
source scripts/env.sh
python scripts/check_setup.py              # self-check: everything should be [OK] before continuing
bash scripts/run_aflow.sh demo             # small search (8 rounds); follow with tail -f logs/demo.log
bash scripts/eval_test.sh runs/demo/best   # evaluate the found workflow on the test split
```

- **Reproduce af27:**
  ```bash
  ROUNDS=48 PATIENCE=8 bash scripts/run_aflow.sh af27_repro
  ```
- **Switch models for one run:** add flags instead of editing `aflow.toml`.
  ```bash
  bash scripts/run_aflow.sh demo --executor deepseek/deepseek-chat --optimizer deepseek/deepseek-flash --harness dsh
  ```
- **Generate your own module with a found workflow:**
  ```bash
  python -m vflowx.cli --spec my_spec.txt --workflow runs/demo/best -o my_mod.v
  ```

<details>
<summary><b>More configuration details</b> (multiple vLLM replicas, other API services, reasoning mode, cost and privacy)</summary>

- **Multiple vLLM replicas:** list several addresses in `base_url` under `[providers.local]`; requests are spread across them. Start a second replica with:
  ```bash
  GPUS=2,3 PORT=8001 bash scripts/serve_vllm.sh /path/to/model
  ```
- **Other OpenAI-compatible services:** copy a `[providers.xxx]` section in `aflow.toml`, set `base_url` and `api_key_env`, then use `xxx/<model name>`.
- **Optional settings:** every provider accepts `concurrency` (default 16 for APIs), `max_tokens` (reply limit, default 6144), `think_effort`, `think_model`.
- **What reasoning mode (`think=True`) means per provider:**

  | Provider | when think=True |
  |---|---|
  | `local` (vLLM + Qwen3) | turns on `enable_thinking` in the chat template |
  | `openai`, reasoning models (gpt-5*, o*) | `reasoning_effort` = `think_effort` (default medium); `none` when think=False |
  | `deepseek` | switches to `think_model` (default `deepseek-reasoner`) |
  | others | switches to `think_model` if set, otherwise unchanged |

- **Local model as optimizer:** the model must support tool calling, and vLLM must be started with `--enable-auto-tool-choice --tool-call-parser hermes`.
- **Reasoning effort:** `--effort` (default high) is passed to the harness; when dsh drives a DeepSeek model it is mapped to DeepSeek's levels: low→low, medium/high→high, xhigh→max.
- **Cost:** a full search evaluates dozens of versions, each on 52 problems × repeats. Estimate first with a small `ROUNDS` or `python -m vflowx.evaluate --limit 3`.
- **Network:** external APIs go through the system proxy (`HTTPS_PROXY`); local addresses never do.
- **Privacy:** by default dsh uploads session logs to DeepSeek and enables telemetry. The dsh config this project generates for each run (`runs/<name>/dsh.patch.yml`) turns both off.
</details>

## 5. Output files

### 5.1 `runs/<name>/`
| File | Contents |
|---|---|
| `leaderboard.json` | score, pass_rate, avg_tokens and change description of every round. The most useful file |
| `experience.jsonl` | one "parent → child" record per line with the score change and the edit; the optimizer reads it in later rounds |
| `rounds/round_N/graph.py`, `prompt.py` | version N of the workflow |
| `rounds/round_N/CONTEXT.md` | the full task given to the optimizer this round. Read it to see what the optimizer "saw" |
| `rounds/round_N/modification.md` | the optimizer's own description of its change |
| `rounds/round_N/<harness>_opt.log` | the optimizer's full transcript: files read, commands run |
| `rounds/round_N/eval/results.jsonl` | per-problem, per-run results (see 5.2) |
| `final.json`, `best/` | re-test results of the top 3 in the finalize step, and the selected workflow |

### 5.2 One line of `results.jsonl`
```
{"pid": "Prob035_count1to10", "rep": 0, "pass": true, "status": "pass", "total_tokens": 1830, "calls": 2, "code": "module TopModule ...", ...}
```
Possible `status` values:

| status | meaning |
|---|---|
| `pass` | passed |
| `mismatch` | compiles, but simulation output differs from the reference |
| `compile` | does not compile |
| `syntax` | no usable code in the output |
| `timeout` | timed out |
| `workflow_error` | the workflow itself crashed |

### 5.3 One line of the log
```
21:04:57 round 3: score=0.7765 pass=0.7788 tokens=1595.3 | 'Added a compiler-gated repair loop ...'
21:04:57 best so far: round 3 score=0.7765 (stale=0)
```
`stale` is the number of consecutive iterations without a new best; the search stops early when it reaches `patience`.

---

## 6. What vflow found

The final version, round 27, evolved along `1 → 3 → 5 → 6 → 10 → 27`; every step was proposed by codex itself:

| Round | val pass rate | tokens | change |
|---|---|---|---|
| 1 | 0.558 | 1030 | starting point: one model call |
| 3 | 0.779 | 1595 | **if it doesn't compile, repair from the compiler errors**, up to twice, recompiling after each repair |
| 5 | 0.760 | 1528 | regex auto-fix for a common error (duplicate output/reg declaration), saving one LLM call |
| 6 | 0.789 | 2478 | after it compiles, one **semantic review** in reasoning mode; repair only if issues are found, and accept the repair only if it still compiles |
| 10 | 0.875 | 6850 | stronger review (codex wrote no description; see `rounds/round_10` for the diff) |
| 27 | 0.885 | 6381 | if it still doesn't compile after two repairs, **regenerate once from the spec** in reasoning mode |

The final workflow ([`examples/af27_best/graph.py`](examples/af27_best/graph.py), 130 lines):
```
generate: generate 1 candidate in normal mode
   ▼
repair:   compile → on failure, try the regex fix first → then let the model fix from the errors (up to 2 times)
   ▼
select:   still doesn't compile → rewrite from the spec in reasoning mode; use it if it compiles
          compiles              → review once in reasoning mode: if no issues, use it as is;
                                  issues found → model repairs → replace only if the repair compiles
   ▼
solve:    overall time limit 840 s; on timeout return the best result so far
```

---

## 7. FAQ

- **codex / dsh hangs or exits non-zero:** run `python scripts/check_setup.py` first, then check the errors in `rounds/round_N/<harness>_opt.log`. Each optimizer run is capped at 30 minutes.
- **Scores jump by a few points:** this is normal. val has only 52 problems, so one problem is about 2 points. That is why every version is run twice during search and the top 3 are re-run 3 times at the end. Don't compare two versions on a single run.
- **I only care about accuracy, not tokens:** `export VFX_MU=0`, so score equals pass rate.
- **Using another dataset:** put RTLLM-2.0 in `data/RTLLM`, then search with `--val-split rtllm_val` and evaluate with `--split rtllm_test`.
- **API returns 401 / 429:** 401 means the key is wrong or not set (export the variable named by `api_key_env` in `aflow.toml`); 429 means rate limiting, lower that provider's `concurrency`.
- **Evaluation is slow:** each vLLM replica defaults to 96 concurrent requests (configurable in `aflow.toml`); simulation parallelism is set by `VFX_EDA_JOBS` (default 48). The most effective speed-up is another vLLM replica.
- **Can a workflow peek at the answers?** No. A workflow only gets the natural-language spec, and the static check forbids file reads, subprocesses and network access (`FORBIDDEN` in `search.py`). Reference solutions and testbenches are only used by the grader in `eda.score_verilogeval`.
- **My reproduction doesn't match af27 exactly:** LLM sampling is random and codex edits differently every time, so the same settings find a different workflow each run.

---

## 8. Code walkthrough: `vflowx/search.py`

| Line | Function | Purpose |
|---|---|---|
| 33 | `FORBIDDEN` | blacklist regex for the static check |
| 37 | `FOCI` | the three rotating focus areas |
| 97 | `select_parent` | ① parent selection (mixed softmax) |
| 110 | `failure_digest` | pick 6 random failed val problems of the parent, with spec, code and grader hints |
| 129 | `executor_text` | tells the optimizer which executor model is used and how reasoning mode works |
| 141 | `write_context` | ② builds `CONTEXT.md` for the optimizer |
| 200 | `codex` | ③ runs one optimizer pass with the configured harness (command line built in `vflowx/harness.py`), 30-minute limit |
| 214 / 229 | `static_check` / `smoke` | ④ static check and smoke test |
| 236 | `expand` | the full ②→⑤ flow for one child |
| 285 | `run_search` | main loop: evaluate the starting point, then each iteration runs `expand` `width` times in parallel and checks early stopping |
| 308 | `finalize` | finalize: re-run the top 3 on val 3 times each and pick `best/` |

Suggested reading order: `workflows/io_staged/graph.py` (what a workflow looks like) → `vflowx/ops.py` (available building blocks) → `vflowx/search.py` (how the search edits it) → any round's `CONTEXT.md` (what codex actually saw).

Command-line options:
```bash
python -m vflowx.search --help
```
| Option | Default | Meaning |
|---|---|---|
| `--run` | required | output directory |
| `--init` | `workflows/io` | starting workflow(s), comma-separated |
| `--rounds` | 12 | maximum number of versions (including the start) |
| `--width` | 2 | children per iteration, in parallel |
| `--repeats` | 1 | runs per version on val (af27 used 2) |
| `--topk` | 4 | size of the parent candidate pool |
| `--patience` | 4 | iterations without improvement before early stop |
| `--executor` | `aflow.toml` | override the executor, e.g. `deepseek/deepseek-chat` |
| `--optimizer` | `aflow.toml` | override the optimizer model, e.g. `deepseek/deepseek-flash` |
| `--harness` | `aflow.toml` | override the harness: `codex` or `dsh` |
| `--effort` | high | optimizer reasoning effort |
| `--val-split` | val | `val` (VerilogEval) or `rtllm_val` |
| `--finalize-only` | | only run finalize, e.g. after an interrupted search |

---

## 9. Citation

If this project helps your research, please cite:

```bibtex
@article{wei2025vflow,
  title   = {VFlow: Discovering Optimal Agentic Workflows for Verilog Generation},
  author  = {Wei, Yangbo and Huang, Zhen and Li, Huang and Xing, Wei W. and Lin, Ting-Jung and He, Lei},
  journal = {arXiv preprint arXiv:2504.03723},
  year    = {2025},
  url     = {https://arxiv.org/abs/2504.03723}
}
```
