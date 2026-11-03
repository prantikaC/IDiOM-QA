"""
Central LLM factory — returns (fast_llm, reasoning_llm) for a given backend.

Supported backends:
  openai  — gpt-5-mini (planner/critic/rewriter) + gpt-4o-mini (generator/KG/classifier)
  ollama  — qwen2.5:32b (reasoning) + qwen2.5:7b (fast); requires Ollama running locally.
            Both default to OLLAMA_HOST (single daemon). Set
            OLLAMA_FAST_BASE_URL / OLLAMA_REASONING_BASE_URL to split them
            across two daemons/GPUs instead — avoids the two models evicting
            each other on every fast<->reasoning cascade call when both
            share one daemon's memory budget (see 2026-09-01 incident:
            single-GPU daemon thrashed reload-vs-reload, ~40x slower than
            the historical per-question baseline).
  qwen    — Qwen models via local vLLM servers (OpenAI-compatible API).

            Default two-tier layout (recommended for L40S / A100):
              fast_llm      → Qwen2.5-7B-Instruct      port 8000  (~14 GB)
              reasoning_llm → Qwen2.5-32B-Instruct-AWQ  port 8001  (~18 GB)

            Start vLLM servers:
              vllm serve Qwen/Qwen2.5-7B-Instruct \\
                  --port 8000 --gpu-memory-utilization 0.85 --dtype auto
              vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ \\
                  --port 8001 --gpu-memory-utilization 0.85 --dtype auto

            Single-server fallback (set both URLs to the same port):
              QWEN_FAST_BASE_URL=http://localhost:8000/v1
              QWEN_REASONING_BASE_URL=http://localhost:8000/v1

            Configure via .env or environment variables:
              QWEN_FAST_BASE_URL       (default: http://localhost:8000/v1)
              QWEN_FAST_MODEL          (default: Qwen/Qwen2.5-7B-Instruct)
              QWEN_REASONING_BASE_URL  (default: http://localhost:8001/v1)
              QWEN_REASONING_MODEL     (default: Qwen/Qwen2.5-32B-Instruct-AWQ)

Usage:
    from src.core.llm_factory import get_llms
    fast_llm, reasoning_llm = get_llms("openai")
    fast_llm, reasoning_llm = get_llms("qwen")
"""

import os

from langchain_core.language_models import BaseChatModel


# ── Model identifiers ────────────────────────────────────────────────────────────

OPENAI_FAST      = "gpt-4o-mini"
OPENAI_REASONING = "gpt-5-mini"

OLLAMA_FAST      = "qwen2.5:7b"
OLLAMA_REASONING = "qwen2.5:32b"

# Qwen/vLLM defaults — override via env vars (see module docstring).
# Two-tier: 7B fast on port 8000 (~14 GB), 32B AWQ reasoning on port 8001 (~18 GB).
QWEN_FAST_DEFAULT_URL      = "http://localhost:8000/v1"
QWEN_FAST_DEFAULT_MODEL    = "Qwen/Qwen2.5-7B-Instruct"
QWEN_REASONING_DEFAULT_URL  = "http://localhost:8001/v1"
QWEN_REASONING_DEFAULT_MODEL = "Qwen/Qwen2.5-32B-Instruct-AWQ"


def get_llms(backend: str = "openai") -> tuple[BaseChatModel, BaseChatModel]:
    """
    Returns (fast_llm, reasoning_llm).

    fast_llm      — used for critic, generator, KG Cypher chain, baseline evaluators
    reasoning_llm — used for the planner (requires multi-step decomposition)
    """
    backend = backend.lower().strip()

    if backend == "openai":
        from langchain_openai import ChatOpenAI
        fast_llm = ChatOpenAI(
            model=OPENAI_FAST,
            temperature=0,
            timeout=60,
            max_retries=3,
        )
        reasoning_llm = ChatOpenAI(
            model=OPENAI_REASONING,
            temperature=1,   # gpt-5-mini ignores temperature but param is accepted
            timeout=60,
            max_retries=3,
        )

    elif backend == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError(
                "langchain-ollama is not installed. Run: pip install langchain-ollama"
            )
        # Per-tier host override — lets fast_llm and reasoning_llm run on
        # separate Ollama daemons/GPUs. Each falls back to OLLAMA_HOST
        # (single-daemon, prior default behavior) if its own var isn't set.
        _ollama_default_host = os.getenv("OLLAMA_HOST")
        fast_host      = os.getenv("OLLAMA_FAST_BASE_URL",      _ollama_default_host)
        reasoning_host = os.getenv("OLLAMA_REASONING_BASE_URL", _ollama_default_host)

        # If NONE of OLLAMA_FAST_BASE_URL / OLLAMA_REASONING_BASE_URL /
        # OLLAMA_HOST are set, base_url stays None and ChatOllama silently
        # falls back to the ollama-python client's hardcoded default
        # (http://127.0.0.1:11434) -- Ollama's system-level default daemon,
        # which has no CUDA_VISIBLE_DEVICES restriction and can load models
        # onto ANY GPU it chooses, including ones this project treats as
        # off-limits. This bit the project on 2026-09-05: a terminal missing
        # these exports silently loaded qwen2.5:7b/32b onto the default
        # daemon, which spread them across GPU1 and GPU2. Warn loudly here
        # instead of failing silently, since the model-name-only summary
        # print below gave zero visibility into which host was actually
        # used.
        if fast_host is None or reasoning_host is None:
            print("--- [LLM Factory] WARNING: OLLAMA_FAST_BASE_URL/OLLAMA_REASONING_BASE_URL/"
                  "OLLAMA_HOST are not all set. Falling back to Ollama's default "
                  "(http://127.0.0.1:11434) for whichever tier is unset -- this hits the "
                  "SYSTEM daemon, not your custom-GPU-pinned one, and it can load models "
                  "onto any GPU with no restriction. Export OLLAMA_FAST_BASE_URL and "
                  "OLLAMA_REASONING_BASE_URL before running if that's not what you want. ---")

        fast_llm = ChatOllama(
            model=OLLAMA_FAST,
            base_url=fast_host,
            temperature=0,
            num_predict=1024,
        )
        reasoning_llm = ChatOllama(
            model=OLLAMA_REASONING,
            base_url=reasoning_host,
            temperature=0,
            num_predict=2048,
        )

    elif backend == "qwen":
        from langchain_openai import ChatOpenAI

        fast_base_url = os.getenv("QWEN_FAST_BASE_URL", QWEN_FAST_DEFAULT_URL)
        fast_model    = os.getenv("QWEN_FAST_MODEL",    QWEN_FAST_DEFAULT_MODEL)
        # Reasoning uses its own URL/model (32B on port 8001 by default).
        # Override QWEN_REASONING_BASE_URL to point at the same port as fast
        # for a single-server setup.
        reasoning_base_url = os.getenv("QWEN_REASONING_BASE_URL", QWEN_REASONING_DEFAULT_URL)
        reasoning_model    = os.getenv("QWEN_REASONING_MODEL",    QWEN_REASONING_DEFAULT_MODEL)

        # vLLM requires a non-empty api_key but ignores its value.
        _vllm_key = os.getenv("QWEN_VLLM_API_KEY", "token-vllm")

        fast_llm = ChatOpenAI(
            model=fast_model,
            base_url=fast_base_url,
            api_key=_vllm_key,
            temperature=0,
            timeout=120,   # local inference can be slower than cloud APIs
            max_retries=2,
        )
        reasoning_llm = ChatOpenAI(
            model=reasoning_model,
            base_url=reasoning_base_url,
            api_key=_vllm_key,
            temperature=0,
            timeout=180,
            max_retries=2,
        )

    else:
        raise ValueError(
            f"Unknown backend '{backend}'. Choose 'openai', 'ollama', or 'qwen'."
        )

    def _describe(llm) -> str:
        name = llm.model_name if hasattr(llm, "model_name") else llm.model
        # base_url covers ChatOllama; openai_api_base covers ChatOpenAI-style
        # clients (qwen backend). Falls back to "(default)" for the openai
        # backend, which has no configurable host.
        host = getattr(llm, "base_url", None) or getattr(llm, "openai_api_base", None)
        return f"{name} @ {host or '(default)'}"

    print(f"[LLM Factory] Backend: {backend}")
    print(f"  fast_llm      → {_describe(fast_llm)}")
    print(f"  reasoning_llm → {_describe(reasoning_llm)}")

    return fast_llm, reasoning_llm


def get_llm_by_spec(spec: str) -> BaseChatModel:
    """
    Build a single LLM from a "<family>:<model>" spec string, independent of
    the fast/reasoning pairing get_llms() returns. Used for one-off swaps
    (e.g. config.CRITIC_REWRITER_LLM_OVERRIDE) where only one role's model
    needs to change, not the whole backend.

    Spec prefixes (same convention as scripts/evaluate_ragas_metrics.py's
    --judge flag):
      "ollama:<model>" — e.g. "ollama:deepseek-r1:32b", "ollama:llama3.1:70b".
                          Routes through OLLAMA_SPEC_BASE_URL if set (lets
                          this model run on a DIFFERENT daemon/GPU than
                          whatever get_llms() is using — important when
                          this spec is layered on top of an existing
                          fast_llm/reasoning_llm pair, e.g.
                          config.CRITIC_REWRITER_LLM_OVERRIDE, to avoid
                          multiple different-sized models fighting over one
                          daemon's memory and thrashing). Falls back to
                          OLLAMA_HOST, then OLLAMA_BASE_URL, if unset.
      "openai:<model>" — e.g. "openai:gpt-4o-mini", "openai:gpt-5-mini".
                          Requires OPENAI_API_KEY.
    """
    if ":" not in spec:
        raise ValueError(
            f"Invalid LLM spec '{spec}' — expected '<family>:<model>', "
            f"e.g. 'ollama:deepseek-r1:32b' or 'openai:gpt-4o-mini'."
        )
    family, model = spec.split(":", 1)
    family = family.lower().strip()

    if family == "ollama":
        from langchain_ollama import ChatOllama
        host = (os.getenv("OLLAMA_SPEC_BASE_URL")
                or os.getenv("OLLAMA_HOST")
                or os.getenv("OLLAMA_BASE_URL"))
        return ChatOllama(model=model, base_url=host, temperature=0, num_predict=2048)

    if family == "openai":
        from langchain_openai import ChatOpenAI
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                f"LLM spec '{spec}' requires OPENAI_API_KEY to be set."
            )
        return ChatOpenAI(model=model, temperature=0, timeout=60, max_retries=3)

    raise ValueError(
        f"Unknown LLM family '{family}' in spec '{spec}'. Choose 'ollama' or 'openai'."
    )
