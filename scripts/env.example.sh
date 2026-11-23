# 环境变量模板：只放路径和 API key。用哪个模型在 aflow.toml 里改。
#   cp scripts/env.example.sh scripts/env.sh   # 然后按你的机器修改 scripts/env.sh（它不会被提交）
#   source scripts/env.sh                      # 每开一个新终端都要 source 一次

export AFLOW_HOME=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # 本仓库根目录（一般不用改）
export VFX_OSS_CAD=/path/to/oss-cad-suite      # 含 bin/iverilog、vvp、verilator、yosys；工具都在 PATH 上可删掉这行

# 用到哪家的 API 就填哪家的 key（变量名和 aflow.toml 里的 api_key_env 对应）
# export OPENAI_API_KEY=sk-...
# export DEEPSEEK_API_KEY=sk-...

# ---- 一般不用改 -----------------------------------------------------------
export PYTHONPATH=$AFLOW_HOME                  # 让 `python -m vflowx.xxx` 找得到包
export VFX_ROOT=$AFLOW_HOME                    # 数据目录 = $VFX_ROOT/data
export VFX_TMP=$AFLOW_HOME/tmp                 # 仿真/编译的临时目录
mkdir -p "$VFX_TMP"
# 访问本机 vLLM 不走代理；访问外部 API 走系统代理（HTTPS_PROXY）。socks 代理会让 httpx 报错，所以去掉 ALL_PROXY
unset ALL_PROXY all_proxy
export NO_PROXY=localhost,127.0.0.1,::1 no_proxy=localhost,127.0.0.1,::1
