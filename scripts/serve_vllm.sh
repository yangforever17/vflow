#!/usr/bin/env bash
# 启动执行器 LLM（vLLM，OpenAI 兼容接口）。
#   bash scripts/serve_vllm.sh /path/to/Qwen3.8-27B-FP8            # 默认用 GPU 0,1，端口 8000
#   GPUS=2,3 PORT=8001 bash scripts/serve_vllm.sh /path/to/model    # 第二个副本
# 参数与 af27 实验一致：TP=2、32k 上下文、开启前缀缓存（同一题的重复采样会快很多）。
set -euo pipefail
MODEL=${1:?用法: serve_vllm.sh <模型目录>}
export CUDA_VISIBLE_DEVICES=${GPUS:-0,1}
TP=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l | tr -d ' ')
exec vllm serve "$MODEL" --served-model-name "${NAME:-qwen3.8-27b-fp8}" \
  --host 127.0.0.1 --port "${PORT:-8000}" --tensor-parallel-size "$TP" --max-model-len 32768 \
  --gpu-memory-utilization 0.88 --enable-prefix-caching --max-num-seqs 128
