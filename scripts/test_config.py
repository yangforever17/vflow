"""离线测试：aflow.toml 的解析、各服务商的请求参数、优化器命令与 dsh 补丁。不联网、不花钱。

    PYTHONPATH=. python scripts/test_config.py
"""
import os
import tempfile

os.environ.update(OPENAI_API_KEY="sk-test-openai", DEEPSEEK_API_KEY="sk-test-deepseek")
for k in ("VFX_EXECUTOR", "VFX_OPTIMIZER", "VFX_HARNESS", "AFLOW_CONFIG", "VFX_CRED_FILE"):
    os.environ.pop(k, None)

from vflowx import config  # noqa: E402
from vflowx.llm import LLM  # noqa: E402
from vflowx.search import Search  # noqa: E402

# 1. 默认配置
assert config.executor_spec() == "local/qwen3.8-27b-fp8" and config.optimizer_spec() == ""
assert config.harness_name() == "codex"
m = config.resolve("local/qwen3.8-27b-fp8")
assert m.kind == "vllm" and m.base_urls == ["http://127.0.0.1:8000/v1"] and m.concurrency == 96

# 2. 执行器：各家的请求参数
def params(spec, think):
    os.environ["VFX_EXECUTOR"] = spec
    llm = LLM()
    return llm, llm._params(llm.model, 0.2, 1, think, None, 7)

llm, p = params("local/qwen3.8-27b-fp8", True)
assert p["extra_body"]["chat_template_kwargs"]["enable_thinking"] and p["max_tokens"] == 6144 and llm.api_key == "EMPTY"
llm, p = params("openai/gpt-5.4-mini", False)
assert p["reasoning_effort"] == "none" and p["temperature"] == 0.2 and "max_completion_tokens" in p
assert llm.api_key == "sk-test-openai"
_, p = params("openai/gpt-5.4-mini", True)
assert p["reasoning_effort"] == "medium" and "temperature" not in p
_, p = params("openai/gpt-4.1-mini", True)
assert "reasoning_effort" not in p and p["temperature"] == 0.2
llm, p = params("deepseek/deepseek-chat", False)
assert p["model"] == "deepseek-chat" and llm.api_key == "sk-test-deepseek"
_, p = params("deepseek/deepseek-chat", True)
assert p["model"] == "deepseek-reasoner"

# 3. 错误提示
for bad, msg in (("nosuch/x", "unknown provider"), ("openai", "write it as")):
    try:
        config.resolve(bad)
        raise AssertionError(bad)
    except ValueError as e:
        assert msg in str(e), e
os.environ.pop("DEEPSEEK_API_KEY")
os.environ["VFX_EXECUTOR"] = "deepseek/deepseek-chat"
try:
    LLM()
    raise AssertionError("missing key not detected")
except ValueError as e:
    assert "DEEPSEEK_API_KEY" in str(e)
os.environ["DEEPSEEK_API_KEY"] = "sk-test-deepseek"

# 4. 自定义服务商 + 配置文件
with tempfile.TemporaryDirectory() as d:
    open(f"{d}/aflow.toml", "w").write(
        'executor = "myapi/m1"\noptimizer = "myapi/m2"\n'
        '[providers.myapi]\nbase_url = "https://gw.example/v1"\napi_key_env = "MYAPI_KEY"\nconcurrency = 4\n')
    os.environ["AFLOW_CONFIG"] = f"{d}/aflow.toml"
    os.environ.pop("VFX_EXECUTOR")
    os.environ["MYAPI_KEY"] = "k"
    llm = LLM()
    assert llm.kind == "compatible" and llm.model == "m1" and llm.base_url == "https://gw.example/v1"
    assert llm._params("m1", 0.2, 1, True, None, None) == {"model": "m1", "max_tokens": 6144, "temperature": 0.2,
                                                             "top_p": 0.95}
    os.environ.pop("AFLOW_CONFIG")

# 5. 优化器 = harness + 模型：两个 harness 用同一种模型写法
from vflowx import harness  # noqa: E402

os.environ["VFX_EXECUTOR"] = "openai/gpt-5.4-mini"
with tempfile.TemporaryDirectory() as d:
    # 默认（af27）：codex + 自己的登录 -> 命令行和 af27 完全一致
    s = Search(f"{d}/r0", "workflows/io_staged", 1, 1, 4, "high", "val")
    assert config.harness_name() == "codex" and config.optimizer_spec() == ""
    assert s.harness.command("/w", "T") == (["codex", "exec", "--skip-git-repo-check", "-C", "/w", "-c",
                                             "model_reasoning_effort=high", "T"], None, None)
    # API 执行器 -> key 通过 600 权限的文件交给 agent
    cred = os.environ["VFX_CRED_FILE"]
    assert oct(os.stat(cred).st_mode & 0o777) == "0o600" and open(cred).read() == "sk-test-openai"

    os.makedirs(f"{d}/x")
    h = harness.make("codex", "gpt-5.5", "high", f"{d}/x")                 # codex 自己的登录 + 指定模型
    assert h.command("/w", "T")[0][-3:] == ["-m", "gpt-5.5", "T"]
    h = harness.make("codex", "openai/gpt-5.4-mini", "high", f"{d}/x")     # codex + 配置里的服务商
    argv = h.command("/w", "T")[0]
    assert "model_provider=aflow" in argv and "model_providers.aflow.wire_api=responses" in argv
    assert "model_providers.aflow.env_key=OPENAI_API_KEY" in argv and argv[-3:] == ["-m", "gpt-5.4-mini", "T"]
    try:                                                                    # codex 需要 Responses 接口
        harness.make("codex", "deepseek/deepseek-flash", "high", f"{d}/x")
        raise AssertionError("codex + deepseek should be refused")
    except ValueError as e:
        assert "Responses" in str(e) and 'harness = "dsh"' in str(e)

    h = harness.make("dsh", "deepseek/deepseek-flash", "medium", f"{d}/x")  # dsh + DeepSeek（内置服务商）
    patch = open(h.patch).read()
    assert "provider: deepseek-official" in patch and "model: deepseek-flash" in patch
    assert "reasoningEffort: high" in patch and "enabled: false" in patch
    argv, env, cwd = h.command("/w", "T")
    assert argv[:3] == ["dsh", "--profile", "headless"] and cwd == "/w" and env["DSH_PERMISSION_MODE"]
    h = harness.make("dsh", "openai/gpt-5.4-mini", "high", f"{d}/x")      # dsh + OpenAI
    patch = open(h.patch).read()
    assert "api: openai-responses" in patch and "apiKeyEnv: OPENAI_API_KEY" in patch and "reasoningEffort: high" in patch
    h = harness.make("dsh", "local/qwen3.8-27b-fp8", "high", f"{d}/x")    # dsh + 本地 vLLM
    patch = open(h.patch).read()
    assert "api: openai-completions" in patch and "apiKeyEnv: VFX_NO_KEY" in patch and "reasoningEffort: null" in patch
    h = harness.make("dsh", "", "high", f"{d}/x")                          # dsh 自己的默认
    assert "agent-default-model" not in open(h.patch).read()
    print("CONTEXT executor line:", s.executor_text())

print("test_config OK")
