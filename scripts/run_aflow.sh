#!/usr/bin/env bash
# 跑一次 AFlow 搜索（后台运行，日志写到 logs/<名字>.log）。
#   bash scripts/run_aflow.sh af_demo            # 小规模试跑：8 轮
#   ROUNDS=48 PATIENCE=8 bash scripts/run_aflow.sh af27_repro   # 复现 af27 的设置
#   bash scripts/run_aflow.sh af_ds --executor deepseek/deepseek-chat --optimizer deepseek/deepseek-flash
#                                                # 后面的参数直接传给 vflowx.search，临时覆盖 aflow.toml
set -euo pipefail
NAME=${1:?用法: run_aflow.sh <实验名>}
cd "$(dirname "$0")/.." && source scripts/env.sh
mkdir -p logs
nohup python -m vflowx.search --run runs/$NAME --init workflows/io_staged \
  --rounds ${ROUNDS:-8} --width ${WIDTH:-2} --repeats ${REPEATS:-2} --patience ${PATIENCE:-4} \
  --effort ${EFFORT:-high} --val-split val "${@:2}" >> logs/$NAME.log 2>&1 < /dev/null &
echo "已在后台启动，PID=$!   查看进度: tail -f logs/$NAME.log"
