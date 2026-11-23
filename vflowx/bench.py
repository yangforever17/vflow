"""Benchmarks + fixed splits. The search only ever sees `val`; `test` is for the final report."""
from __future__ import annotations

import glob
import json
import os
import random
import re
from dataclasses import dataclass, field

from . import eda

ROOT = os.environ.get("VFX_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VE_DIR = f"{ROOT}/data/verilog-eval/dataset_spec-to-rtl"
RTLLM_DIR = f"{ROOT}/data/RTLLM"
SPLIT_FILE = f"{ROOT}/data/splits.json"
N_VAL = 52  # 1/3 of VerilogEval-v2 spec-to-rtl for search, 2/3 held out
# Known-broken upstream: reference module and testbench disagree on port names (fails even with the
# reference solution).
BROKEN = {"Prob099_m2014_q6c"}

# Upstream RTLLM problems that still fail their own reference after adapter fixes.
RTLLM_BROKEN = {
    "radix2_div",  # Reference/testbench produce 3/8 result mismatches (including signed division).
    "clkgenerator",  # Testbench samples on the reference's blocking clock toggle: 20/20 failures.
    "ring_counter",  # Unsupported array initializer; after lowering it, racing checks miss completion.
    "adder_pipe_64bit",  # Testbench overrides DATA_WIDTH/STG_WIDTH parameters the spec never mentions.
}


@dataclass
class Problem:
    pid: str
    spec: str
    top: str
    bench: str
    _score_args: tuple = field(default=(), repr=False)

    async def score(self, code: str) -> dict:
        if self.bench == "verilogeval":
            return await eda.score_verilogeval(code, *self._score_args)
        return await eda.score_rtllm(code, *self._score_args)


def load_verilogeval() -> list[Problem]:
    probs = []
    for p in sorted(glob.glob(f"{VE_DIR}/*_prompt.txt")):
        pid = os.path.basename(p)[: -len("_prompt.txt")]
        if pid in BROKEN:
            continue
        base = p[: -len("_prompt.txt")]
        probs.append(Problem(pid, open(p).read(), "TopModule", "verilogeval",
                             (open(base + "_test.sv").read(), open(base + "_ref.sv").read())))
    return probs


def _rtllm_input_files(directory: str, testbench: str) -> dict[str, bytes]:
    """Read testbench inputs, including filenames supplied by string macros.

    Only input data is carried to the grader; verified RTL is never a dependency.
    Missing inputs fail at load time rather than allowing a zero-vector success.
    """
    source = eda._rtllm_strip_comments(testbench)
    macros = dict(re.findall(r'`define\s+(\w+)\s+"([^"\n]+)"', source))
    files = {}
    for m in re.finditer(
        r'\$(readmem[hb]|fopen)\s*\(\s*(?:"([^"\n]+)"|`(\w+))'
        r'(?:\s*,\s*"([^"\n]+)")?', source
    ):
        if m[1] == "fopen" and (m[4] is None or not m[4].startswith("r")):
            continue  # Output files are created by the simulation.
        name = m[2] if m[2] is not None else macros[m[3]]
        if os.path.isabs(name) or ".." in name.split("/"):
            raise ValueError(f"RTLLM input must be relative to its problem: {name}")
        with open(os.path.join(directory, name), "rb") as f:
            files[name] = f.read()
    return files


def load_rtllm(include_broken: bool = False) -> list[Problem]:
    probs = []
    for d in sorted(glob.glob(f"{RTLLM_DIR}/**/design_description.txt", recursive=True)):
        d = os.path.dirname(d)
        if "/_" in d:
            continue
        name = os.path.basename(d)
        if name in RTLLM_BROKEN and not include_broken:
            continue
        spec = open(f"{d}/design_description.txt").read()
        testbench = open(f"{d}/testbench.v").read()
        top, _ = eda._rtllm_hierarchy(testbench)
        names = re.findall(
            r"Module name:\s*(\w+)|\bmodule\s+(\w+)\s*(?:#\s*\(|\()", spec, re.I
        )
        if not names or any((a or b) != top for a, b in names):
            spec = spec.rstrip() + f"\n\nThe top module must be named {top}.\n"
        files = _rtllm_input_files(d, testbench)
        probs.append(Problem(f"rtllm/{name}", spec, top, "rtllm", (testbench, top, files)))
    return probs


def splits(seed: int = 2026) -> dict:
    if os.path.exists(SPLIT_FILE):
        return json.load(open(SPLIT_FILE))
    ids = [p.pid for p in load_verilogeval()]
    rng = random.Random(seed)
    rng.shuffle(ids)
    s = {"val": sorted(ids[:N_VAL]), "test": sorted(ids[N_VAL:]), "seed": seed}
    json.dump(s, open(SPLIT_FILE, "w"), indent=1)
    return s


RTLLM_SPLIT_FILE = f"{ROOT}/data/splits_rtllm.json"
N_RTLLM_VAL = 20


def rtllm_splits(seed: int = 2026) -> dict:
    """Stratified by RTLLM category (Arithmetic/Control/Memory/Miscellaneous): ~20 val / ~27 test."""
    if os.path.exists(RTLLM_SPLIT_FILE):
        return json.load(open(RTLLM_SPLIT_FILE))
    cat = {}
    for f in glob.glob(f"{RTLLM_DIR}/**/design_description.txt", recursive=True):
        d = os.path.dirname(f)
        cat[f"rtllm/{os.path.basename(d)}"] = os.path.relpath(d, RTLLM_DIR).split("/")[0]
    ids = [p.pid for p in load_rtllm()]
    rng = random.Random(seed)
    by_cat: dict[str, list[str]] = {}
    for pid in sorted(ids):
        by_cat.setdefault(cat.get(pid, "?"), []).append(pid)
    val = []
    for c, lst in sorted(by_cat.items()):
        rng.shuffle(lst)
        val += lst[: round(len(lst) * N_RTLLM_VAL / len(ids))]
    s = {"val": sorted(val), "test": sorted(set(ids) - set(val)), "seed": seed,
         "categories": {c: len(v) for c, v in by_cat.items()}}
    json.dump(s, open(RTLLM_SPLIT_FILE, "w"), indent=1)
    return s


# Benchmark descriptions shown to the optimizer.
DESCRIPTIONS = {
    "verilogeval": "VerilogEval-v2 spec-to-rtl (short human-written specs; combinational logic, small FSMs, "
                   "counters, shift registers; top module always `TopModule`)",
    "rtllm": "RTLLM-2.0 (medium-size realistic modules: multipliers, dividers, pipelined adders, FIFOs, "
             "frequency dividers, protocol/serial converters, FSMs; each spec names its own top module and "
             "the grading testbench checks multi-cycle timing and latency exactly as specified)",
}


def bench_of(split: str) -> str:
    return "rtllm" if split.startswith("rtllm") else "verilogeval"


FB_SIZE = {"val": 20, "rtllm_val": 0}  # 0 => feedback set == selection set (split too small to halve)


def fb_sel(val_split: str, seed: int = 7) -> dict:
    """Split a validation split into a FEEDBACK part (failures shown to the optimizer, smoke tests) and a
    SELECTION part (scores used for acceptance/selection, never shown to the optimizer)."""
    f = f"{ROOT}/data/fbsel_{val_split}.json"
    if os.path.exists(f):
        return json.load(open(f))
    ids = sorted(p.pid for p in load(val_split))
    k = FB_SIZE.get(val_split, len(ids) // 3)
    random.Random(seed).shuffle(ids)
    s = {"fb": sorted(ids[:k]) if k else sorted(ids), "sel": sorted(ids[k:]), "shared": k == 0}
    json.dump(s, open(f, "w"), indent=1)
    return s


def load(split: str) -> list[Problem]:
    """split in {val, test, all, rtllm, rtllm_val, rtllm_test} or '<val split>_fb' / '<val split>_sel'"""
    for suf in ("_fb", "_sel"):
        if split.endswith(suf):
            keep = set(fb_sel(split[: -len(suf)])[suf[1:]])
            return [p for p in load(split[: -len(suf)]) if p.pid in keep]
    if split == "rtllm":
        return load_rtllm()
    if split in ("rtllm_val", "rtllm_test"):
        keep = set(rtllm_splits()[split.split("_", 1)[1]])
        return [p for p in load_rtllm() if p.pid in keep]
    probs = load_verilogeval()
    if split == "all":
        return probs
    keep = set(splits()[split])
    return [p for p in probs if p.pid in keep]
