"""EDA helpers: compile/lint (iverilog, verilator), simulation, interface extraction (yosys),
random-stimulus testbench generation, and the official VerilogEval scoring rule.

Everything runs in a throwaway temp dir with a hard timeout. Nothing here ever sees the
reference solution except `score_verilogeval`, which is ONLY used by the evaluator.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass

# oss-cad-suite install dir (has bin/iverilog, vvp, verilator, yosys). If VFX_OSS_CAD is unset, the tools are
# looked up on PATH.
OSS = os.environ.get("VFX_OSS_CAD", "")


def _tool(name: str) -> str:
    return f"{OSS}/bin/{name}" if OSS else (shutil.which(name) or name)


IVERILOG = _tool("iverilog")
VVP = _tool("vvp")
VERILATOR = _tool("verilator")
YOSYS = _tool("yosys")

_SEM = asyncio.Semaphore(int(os.environ.get("VFX_EDA_JOBS", "48")))
_MISMATCH_RE = re.compile(r"Mismatches:\s*(\d+)\s+in\s+(\d+)\s+samples")


async def _run(cmd: list[str], cwd: str, timeout: float) -> tuple[int, str]:
    async with _SEM:
        p = await asyncio.create_subprocess_exec(*cmd, cwd=cwd, stdout=asyncio.subprocess.PIPE,
                                                 stderr=asyncio.subprocess.STDOUT)
        try:
            out, _ = await asyncio.wait_for(p.communicate(), timeout)
            return p.returncode, out.decode(errors="replace")
        except asyncio.TimeoutError:
            try:
                p.kill()
            except ProcessLookupError:
                pass
            await p.wait()
            return 124, "TIMEOUT"


def _clip(s: str, n: int = 3000) -> str:
    return s if len(s) <= n else s[: n // 2] + "\n...[truncated]...\n" + s[-n // 2:]


class _Tmp:
    def __enter__(self):
        self.d = tempfile.mkdtemp(prefix="vfx_", dir=os.environ.get("VFX_TMP"))
        return self.d

    def __exit__(self, *a):
        shutil.rmtree(self.d, ignore_errors=True)


# ----------------------------------------------------------------------------- lint
async def lint(code: str, top: str = "TopModule", verilator: bool = True) -> dict:
    """iverilog compile (errors) + optional verilator lint (warnings). ok == compiles under iverilog."""
    with _Tmp() as d:
        open(f"{d}/dut.sv", "w").write(code)
        rc, out = await _run([IVERILOG, "-g2012", "-Wall", "-Wno-timescale", "-s", top, "-o", "a.out", "dut.sv"], d, 30)
        errors = "\n".join(l for l in out.splitlines() if "error" in l.lower() or "syntax" in l.lower())
        ok = rc == 0 and not errors
        warnings = ""
        if verilator and ok:
            _, vout = await _run([VERILATOR, "--lint-only", "-Wall", "-Wno-DECLFILENAME", "-Wno-UNUSEDSIGNAL",
                                  "-Wno-UNUSEDPARAM", "-Wno-EOFNEWLINE", "--top-module", top, "dut.sv"], d, 30)
            warnings = "\n".join(l for l in vout.splitlines() if l.startswith("%"))
        return {"ok": ok, "errors": _clip(errors or (out if not ok else "")), "warnings": _clip(warnings, 2000)}


# ----------------------------------------------------------------------------- interface
@dataclass
class Port:
    name: str
    direction: str  # input | output | inout
    width: int


async def interface(code: str, top: str = "TopModule") -> list[Port] | None:
    """Port list via yosys JSON (robust to ANSI/non-ANSI/SV styles). None if yosys cannot parse."""
    with _Tmp() as d:
        open(f"{d}/dut.sv", "w").write(code)
        rc, out = await _run([YOSYS, "-q", "-p", f"read_verilog -sv dut.sv; hierarchy -top {top}; proc; write_json ports.json"], d, 60)
        if rc != 0 or not os.path.exists(f"{d}/ports.json"):
            return None
        mods = json.load(open(f"{d}/ports.json"))["modules"]
        m = mods.get(top) or next(iter(mods.values()))
        return [Port(n, p["direction"], len(p["bits"])) for n, p in m["ports"].items()]


_CLK_RE = re.compile(r"^(clk|clock|i_clk|clk_i)$", re.I)
_RST_RE = re.compile(r"(^|_)(a?s?reset|a?s?rst)(_?n|_b|_l)?($|_)", re.I)


def _is_active_low(name: str) -> bool:
    n = name.lower()
    return n.endswith("_n") or n.endswith("resetn") or n.endswith("rstn") or n.endswith("_b") or n.endswith("_l")


def classify_ports(ports: list[Port]):
    ins = [p for p in ports if p.direction == "input"]
    outs = [p for p in ports if p.direction != "input"]
    clk = next((p for p in ins if _CLK_RE.match(p.name) and p.width == 1), None)
    rsts = [p for p in ins if p is not clk and _RST_RE.search(p.name) and p.width == 1]
    data = [p for p in ins if p is not clk and p not in rsts]
    return ins, outs, clk, rsts, data


def _stim_tb(ports: list[Port], top: str, steps: int, drive: str, seed: int = 1) -> str:
    """Stimulus-only testbench. Each step: drive inputs (2ns after the negedge for clocked designs), sample outputs
    once before the clock edge (PRE, Mealy view) and once after it (POST).
    Line format: `@<i> IN <hex..> OUT <pre..> | <post..>`. No expected values are involved."""
    ins, outs, clk, rsts, data = classify_ports(ports)
    decl = [f"  reg [{p.width - 1}:0] {p.name};" for p in ins] + [f"  wire [{p.width - 1}:0] {p.name};" for p in outs]
    conns = ", ".join(f".{p.name}({p.name})" for p in ports)
    shown_in = data + rsts
    fmt_in = " ".join("%h" for _ in shown_in) or "-"
    fmt_out = " ".join("%h" for _ in outs) or "-"
    in_args = [p.name for p in shown_in]
    out_args = [p.name for p in outs]
    init = "\n".join(f"    {p.name} = 0;" for p in ins)
    if clk:
        # drive 2ns AFTER the negedge (not at it): negedge-triggered DUTs would otherwise race with the blocking
        # input assignments, and the sampled value would depend on the simulator's event order
        body = f"""      @(negedge {clk.name}); #2;
{drive}
      #1;
      pre = {{{", ".join(out_args) or "1'b0"}}};
      @(posedge {clk.name}); #1;"""
        clkgen = f"  always #5 {clk.name} = ~{clk.name};"
    else:
        body = f"""{drive}
      #4;
      pre = {{{", ".join(out_args) or "1'b0"}}};
      #1;"""
        clkgen = ""
    total_w = sum(p.width for p in outs) or 1
    pre_fmt = "%h"
    args = ", ".join(in_args + ["pre"] + out_args)
    show = f'      $display("@%0d IN {fmt_in} OUT {pre_fmt} | {fmt_out}", i, {args});'
    return f"""`timescale 1ns/1ps
module vfx_tb;
{chr(10).join(decl)}
  reg [{total_w - 1}:0] pre;
  integer i;
  integer seed;
  {top} dut({conns});
{clkgen}
  initial begin
    seed = {seed};
{init}
    #2;
    for (i = 0; i < {steps}; i = i + 1) begin
{body}
{show}
    end
    $finish;
  end
endmodule
"""


def random_tb(ports: list[Port], cycles: int = 200, seed: int = 1, top: str = "TopModule") -> str:
    """Random stimulus; reset asserted for the first 2 steps and then ~3% of the time."""
    _, _, _, rsts, data = classify_ports(ports)

    def rnd(w):
        k = (w + 31) // 32
        return "{" + ",".join(["$random(seed)"] * k) + "}" if k > 1 else "$random(seed)"

    drive = []
    for p in rsts:
        on, off = ("1'b0", "1'b1") if _is_active_low(p.name) else ("1'b1", "1'b0")
        drive.append(f"      {p.name} = (i < 2 || ($random(seed) & 31) == 0) ? {on} : {off};")
    for p in data:
        drive.append(f"      {p.name} = {rnd(p.width)};")
    return _stim_tb(ports, top, cycles, "\n".join(drive), seed)


def vector_tb(ports: list[Port], vectors: list[dict], top: str = "TopModule") -> str:
    """Replay explicit per-step input vectors ({port: int}); missing ports keep their previous value
    (0 initially). Used for counterexample replay and LLM-directed corner cases."""
    _, _, clk, _, _ = classify_ports(ports)
    names = {p.name: p.width for p in ports if p.direction == "input" and p is not clk}
    cases = []
    for i, v in enumerate(vectors):
        assigns = " ".join(f"{k} = {names[k]}'d{int(val) % (1 << names[k])};" for k, val in v.items()
                           if k in names and isinstance(val, int))
        if assigns:
            cases.append(f"        {i}: begin {assigns} end")
    drive = "      case (i)\n" + "\n".join(cases) + "\n        default: ;\n      endcase" if cases else ""
    return _stim_tb(ports, top, max(len(vectors), 1), drive)


async def simulate(code: str, tb: str, top_tb: str = "vfx_tb", timeout: float = 30) -> dict:
    """Compile DUT + testbench and run. Returns {ok, stage, log}."""
    with _Tmp() as d:
        open(f"{d}/dut.sv", "w").write(code)
        open(f"{d}/tb.sv", "w").write(tb)
        rc, out = await _run([IVERILOG, "-g2012", "-Wno-timescale", "-s", top_tb, "-o", "sim", "dut.sv", "tb.sv"], d, 30)
        if rc != 0:
            return {"ok": False, "stage": "compile", "log": _clip(out)}
        rc, out = await _run([VVP, "-n", "sim"], d, timeout)
        if rc == 124:
            return {"ok": False, "stage": "timeout", "log": "TIMEOUT"}
        return {"ok": rc == 0, "stage": "run", "log": out}


def _trace_lines(log: str) -> str | None:
    lines = [l for l in log.splitlines() if l.startswith("@")]
    return "\n".join(lines) if lines else None


async def trace(code: str, ports: list[Port], cycles: int = 200, seed: int = 1,
                top: str = "TopModule", vectors: list[dict] | None = None) -> str | None:
    """Output trace under random (or explicit `vectors`) stimulus. None on compile/run failure."""
    tb = vector_tb(ports, vectors, top) if vectors else random_tb(ports, cycles, seed, top)
    r = await simulate(code, tb)
    if not r["ok"]:  # compile error, timeout, or nonzero simulator exit ($fatal, crash): no partial traces
        return None
    return _trace_lines(r["log"])


# ----------------------------------------------------------------------------- official scoring (evaluator only)
_DUMP_RE = re.compile(r"^\s*\$dump(file|vars)\b.*$", re.M)


async def score_verilogeval(code: str, test_sv: str, ref_sv: str) -> dict:
    """Same flags and pass rule as NVlabs/verilog-eval (spec-to-rtl, iverilog).
    $dumpfile/$dumpvars are stripped: newer iverilog rejects their use-before-declare of tb_mismatch,
    and waves are not needed for scoring."""
    test_sv = _DUMP_RE.sub("", test_sv)
    with _Tmp() as d:
        open(f"{d}/samp.sv", "w").write(code)
        open(f"{d}/test.sv", "w").write(test_sv)
        open(f"{d}/ref.sv", "w").write(ref_sv)
        rc, clog = await _run([IVERILOG, "-Wall", "-Winfloop", "-Wno-timescale", "-g2012", "-s", "tb",
                               "-o", "sim", "samp.sv", "test.sv", "ref.sv"], d, 30)
        if rc != 0 or not os.path.exists(f"{d}/sim"):
            status = "syntax" if "syntax error" in clog else "compile"
            return {"pass": False, "status": status, "mismatches": None, "log": _clip(clog, 1500)}
        rc, rlog = await _run([VVP, "-n", "sim"], d, 30)
        if rc == 124:
            return {"pass": False, "status": "timeout", "mismatches": None, "log": "TIMEOUT"}
        m = _MISMATCH_RE.search(rlog)
        if not m:
            return {"pass": False, "status": "runtime", "mismatches": None, "log": _clip(rlog, 1500)}
        mm, total = int(m.group(1)), int(m.group(2))
        return {"pass": mm == 0, "status": "pass" if mm == 0 else "mismatch", "mismatches": mm,
                "samples": total, "log": _clip(rlog, 1500)}


def _rtllm_strip_comments(source: str) -> str:
    # Preserve strings, including filenames and diagnostic messages containing //.
    return re.sub(r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/',
                  lambda m: m[0] if m[0].startswith('"') else "\n" * m[0].count("\n") + " ",
                  source, flags=re.S)


def _rtllm_hierarchy(testbench: str) -> tuple[str, list[str]]:
    """Find the external DUT type and simulation roots in RTLLM testbenches.

    Supports their ordinary/parameterized instances and local checker modules;
    fail explicitly on ambiguity instead of guessing from the directory/spec.
    This is deliberately a reader for this corpus, not a full Verilog parser.
    """
    source = _rtllm_strip_comments(testbench)
    source = re.sub(r'"(?:\\.|[^"\\])*"', '""', source)
    declared = set(re.findall(r"\bmodule\s+(\w+)", source))
    instances = set(re.findall(
        r"^\s*([A-Za-z_]\w*)\s+(?:#\s*\((?:[^()]|\([^()]*\))*\)\s*)?"
        r"[A-Za-z_]\w*\s*\(", source, re.M
    )) - {"module", "if", "else", "begin", "end", "generate", "task", "function"}
    external = instances - declared
    roots = sorted(declared - instances)
    if len(external) != 1 or not roots:
        raise ValueError(f"Ambiguous RTLLM hierarchy: DUTs={sorted(external)}, roots={roots}")
    return external.pop(), roots


# All 50 supplied testbenches use this banner (with/without spaces around it).
# A completion message alone is not success; any failure overrides the banner.
_RTLLM_PASS_RE = re.compile(r"^\s*=*\s*Your\s+Design\s+Passed\s*=*\s*$", re.M)
_RTLLM_FAIL_RE = re.compile(
    r"^\s*(?:=+\s*)?(?:Error\b|Failed\b|Test failed\b|Test completed with\b|FATAL:|Timeout\b)",
    re.M | re.I,
)


async def score_rtllm(code: str, testbench_v: str, top: str,
                      input_files: dict[str, bytes] | None = None) -> dict:
    """Run the actual testbench with its inputs; require a clean success banner."""
    dut_top, roots = _rtllm_hierarchy(testbench_v)
    if dut_top != top:
        raise ValueError(f"RTLLM top {top!r} disagrees with testbench DUT {dut_top!r}")
    with _Tmp() as d:
        for name, contents in (input_files or {}).items():
            if os.path.isabs(name) or ".." in name.split("/") or name in {"dut.v", "testbench.v", "sim"}:
                raise ValueError(f"Invalid RTLLM input filename: {name}")
            path = os.path.join(d, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(contents)
        open(f"{d}/dut.v", "w").write(code)
        # Legacy benches use `checker` as an instance name, reserved since SV-2009.
        # Limit the older keyword set to the bench; candidate RTL stays SV-2012.
        open(f"{d}/testbench.v", "w").write(
            '`begin_keywords "1800-2005"\n' + testbench_v + '\n`end_keywords\n')
        root_flags = [flag for root in roots for flag in ("-s", root)]
        rc, clog = await _run([IVERILOG, "-g2012", "-Wno-timescale", *root_flags,
                               "-o", "sim", "dut.v", "testbench.v"], d, 30)
        if rc != 0:
            return {"pass": False, "status": "compile", "log": _clip(clog, 1500)}
        rc, rlog = await _run([VVP, "-n", "sim"], d, 60)
        if rc == 124:
            return {"pass": False, "status": "timeout", "log": "TIMEOUT"}
        ok = rc == 0 and bool(_RTLLM_PASS_RE.search(rlog)) and not _RTLLM_FAIL_RE.search(rlog)
        status = "pass" if ok else ("runtime" if rc else "mismatch")
        return {"pass": bool(ok), "status": status, "log": _clip(rlog, 1500)}
