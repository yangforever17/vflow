"""环境自检：在第一次跑搜索之前运行，逐项告诉你哪里还没配好。

    source scripts/env.sh && python scripts/check_setup.py
    python scripts/check_setup.py --skip-llm      # 先不试调执行器（还没起 vLLM / 没填 key 时）
"""
import argparse
import asyncio
import os
import shutil
import subprocess
import sys
import tempfile

OK, BAD = "  [OK] ", "  [!!] "
problems = 0


def report(ok: bool, msg: str, fix: str = ""):
    global problems
    print((OK if ok else BAD) + msg + ("" if ok or not fix else f"\n       -> {fix}"))
    problems += not ok


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-llm", action="store_true", help="不检查执行器 LLM")
    a = ap.parse_args()

    print("1. Python 与依赖")
    report(sys.version_info >= (3, 11), f"Python {sys.version.split()[0]}", "需要 Python >= 3.11（推荐 3.12）")
    try:
        import httpx  # noqa: F401
        import openai  # noqa: F401
        report(True, f"openai {openai.__version__}, httpx {httpx.__version__}")
    except ImportError as e:
        report(False, f"缺少依赖: {e}", "pip install -r requirements.txt")
        return
    try:
        from vflowx import bench, eda
    except ImportError as e:
        report(False, f"导入 vflowx 失败: {e}", "先 source scripts/env.sh（它会设置 PYTHONPATH）")
        return

    print("2. EDA 工具（iverilog / vvp / verilator / yosys）")
    for name, path in (("iverilog", eda.IVERILOG), ("vvp", eda.VVP), ("verilator", eda.VERILATOR),
                       ("yosys", eda.YOSYS)):
        found = os.path.isfile(path) or shutil.which(path)
        report(bool(found), f"{name}: {path}", "装 oss-cad-suite 并在 scripts/env.sh 里设置 VFX_OSS_CAD")
    try:
        r = await eda.lint("module TopModule(input a, output y); assign y = ~a; endmodule")
        report(r["ok"], "能编译一个最小的 Verilog 模块", f"编译失败：{r.get('errors', '')[:300]}")
    except OSError as e:
        report(False, f"无法运行编译器: {e}", "装好上面缺的工具后再试")

    print("3. 数据集（VerilogEval-v2 spec-to-rtl）")
    n = len(bench.load_verilogeval()) if os.path.isdir(bench.VE_DIR) else 0
    report(n == 155, f"{bench.VE_DIR}: {n} 题（应为 155 = 156 - 1 道上游坏题）",
           "git clone https://github.com/NVlabs/verilog-eval.git data/verilog-eval")
    if n:
        s = bench.splits()
        report(len(s["val"]) == 52 and len(s["test"]) == 103, f"划分: val {len(s['val'])} / test {len(s['test'])}",
               "data/splits.json 应与仓库自带的一致（val 52 / test 103）")

    from vflowx import config, harness
    print(f"4. 模型配置（{config.config_path()}）")
    try:
        ex = config.executor_spec()
        h = harness.make(config.harness_name(), config.optimizer_spec(), "high", tempfile.mkdtemp())
        report(True, f"executor = {ex!r}   optimizer = {h.spec!r}   harness = {h.name!r}")
    except Exception as e:  # noqa: BLE001
        report(False, f"配置有问题: {e}", "检查 aflow.toml")
        return

    print(f"5. 优化器：{h.describe()}")
    if h.name == "dsh":
        node = shutil.which("node")
        nv = subprocess.run([node, "-v"], capture_output=True, text=True).stdout.strip() if node else ""
        major_minor = tuple(int(x) for x in nv.lstrip("v").split(".")[:2]) if nv else (0, 0)
        report(major_minor >= (24, 9), f"node: {node or '找不到'} {nv}", "dsh 需要 Node.js >= 24.9")
    install = {"codex": "npm i -g @openai/codex", "dsh": "npm i -g @deepseek-ai/dsh"}[h.name]
    path = shutil.which(h.binary)
    report(bool(path), f"{h.name}: {path or '找不到'}", install)
    if path:
        v = subprocess.run([h.binary, "--version"], capture_output=True, text=True)
        report(v.returncode == 0, f"{h.name} --version: {v.stdout.strip() or v.stderr.strip()[:200]}")
    if h.model and h.model.api_key_env:
        report(bool(h.model.api_key), f"{h.model.api_key_env} {'已设置' if h.model.api_key else '未设置'}",
               f"export {h.model.api_key_env}=...（写进 scripts/env.sh）")
    elif not h.model:
        print("       （用 " + h.name + " 自己的登录：" + {"codex": "codex login", "dsh": "DEEPSEEK_API_KEY"}[h.name] + "）")

    if not a.skip_llm:
        from vflowx.llm import LLM
        print(f"6. 执行器：{ex}")
        try:
            llm = LLM()
        except ValueError as e:
            report(False, str(e), "检查 aflow.toml 的 executor 和 scripts/env.sh 里的 key")
            llm = None
        if llm:
            try:
                model = await asyncio.wait_for(llm.resolve_model(), 30)
                report(True, f"{llm.base_urls} 上的模型: {model}"
                       + (f"（think=True 时用 {llm.think_model}）" if llm.think_model else ""))
                out = await asyncio.wait_for(llm.ask("Reply with the single word: pong"), 120)
                report("pong" in out.lower(), f"普通模式试调用，回复: {out[:60]!r}")
                out = await asyncio.wait_for(llm.ask("Reply with the single word: pong", think=True), 300)
                report("pong" in out.lower(), f"推理模式（think=True）试调用，回复: {out[:60]!r}")
                u = llm.global_usage
                print(f"       自检共用了 {u.calls} 次调用、{u.total_tokens} token")
            except Exception as e:  # noqa: BLE001
                fix = ("先用 scripts/serve_vllm.sh 起 vLLM；本机地址不要走代理" if llm.kind == "vllm"
                       else "检查 key、模型名、网络/代理（HTTPS_PROXY）")
                report(False, f"调用 {llm.base_urls} 失败: {e!r}"[:400], fix)

    print("\n全部通过，可以开始跑了。" if problems == 0 else f"\n有 {problems} 项没通过，按上面的提示修好再跑。")


asyncio.run(main())
