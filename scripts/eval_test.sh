#!/usr/bin/env bash
# 在测试集（103 题，搜索过程从没见过）上评估一个 workflow，结果写到 <workflow>/final_test/。
#   bash scripts/eval_test.sh runs/af_demo/best
#   bash scripts/eval_test.sh workflows/io            # 基线：直接让模型写一次
#   bash scripts/eval_test.sh workflows/io --executor openai/gpt-5.4-mini   # 换个执行器
set -euo pipefail
WF=${1:?用法: eval_test.sh <workflow 目录>}
cd "$(dirname "$0")/.." && source scripts/env.sh
python -m vflowx.evaluate --workflow "$WF" --split test --repeats ${REPEATS:-2} --out "$WF/final_test" "${@:2}"
