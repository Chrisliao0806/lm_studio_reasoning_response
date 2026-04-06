# lm_studio_reasoning_response

A universal LangChain-compatible wrapper that properly surfaces `reasoning_content` (thinking tokens) from reasoning LLMs via the **OpenAI-compatible API** (`/v1/chat/completions`).  Works with DeepSeek-Reasoner, QwQ, vLLM, LM Studio, and any API that returns reasoning tokens in the delta.

---

## Table of Contents

- [Background](#background)
- [Why naive overrides fail](#why-naive-overrides-fail)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [LM Studio / vLLM (Gemma with thinking)](#lm-studio--vllm-gemma-with-thinking)
  - [DeepSeek-Reasoner](#deepseek-reasoner)
- [API Reference — ChatOpenAIReasoner](#api-reference--chatopenaireasononer)
  - [Constructor parameters](#constructor-parameters)
  - [Key method: set_abort_event](#key-method-set_abort_event)
- [Streaming vs Non-streaming](#streaming-vs-non-streaming)
- [Cooperative cancellation](#cooperative-cancellation)
- [Supported backends](#supported-backends)
- [LM Studio setup](#lm-studio-setup)
- [FAQ](#faq)

---

## Background

Modern reasoning LLMs (DeepSeek-R1, QwQ, Gemma with thinking, …) emit two separate token streams:

| Stream | Field | Description |
|---|---|---|
| Thinking | `reasoning_content` / `reasoning` | The model's internal chain-of-thought |
| Answer | `content` | The final response |

The standard LangChain `ChatOpenAI` silently **drops** these fields before they reach user code because Pydantic's `model_dump()` excludes `model_extra` fields.  This library fixes that.

---

## Why naive overrides fail

`BaseChatOpenAI._stream` calls `chunk.model_dump()` on every streaming chunk **before** passing it downstream.  Pydantic's `model_dump()` excludes `model_extra` fields, so `reasoning_content` is silently dropped before any hook can intercept it.

**Fix implemented here:** Override `_stream` / `_astream` to make the raw API call directly, read `reasoning_content` from the **live** delta object before serialisation, then inject it back into `gen_chunk.message.additional_kwargs["reasoning_content"]`.

---

## Features

- Drop-in replacement for `ChatOpenAI` — all LangChain usage patterns work unchanged
- Captures thinking tokens in both **streaming** and **non-streaming** modes
- Supports both `reasoning_content` (DeepSeek) and `reasoning` (vLLM Gemma-4) field names
- **OpenAI-compatible mode** works with LM Studio, DeepSeek-Reasoner, QwQ, vLLM, and any API returning reasoning tokens in the delta
- Pass model-specific thinking settings via `extra_body` (e.g. `{"chat_template_kwargs": {"enable_thinking": True}}` for vLLM/Gemma-4)
- Async support (`astream`, `ainvoke`, `abatch`)
- Cooperative cancellation via `threading.Event`

---

## Project Structure

```
lm_studio_reasoning_response/
├── chatopenai_reasoner.py   # Core wrapper — ChatOpenAIReasoner class
├── main.py                  # Usage example (LM Studio, Gemma model)
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Requirements

Python 3.12+

```
httpx==0.28.1
langchain==1.2.0
langchain-core==1.2.6
langchain-openai==1.1.6
openai==2.14.0
pydantic==2.12.5
requests==2.32.5
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Chrisliao0806/lm_studio_reasoning_response.git
cd lm_studio_reasoning_response
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Quick Start

### LM Studio / vLLM (Gemma with thinking)

Use this when running a reasoning model via LM Studio's OpenAI-compatible endpoint or a vLLM server.  Pass `extra_body` to enable the model's thinking mode.

```python
from langchain_core.prompts import ChatPromptTemplate
from chatopenai_reasoner import ChatOpenAIReasoner

llm = ChatOpenAIReasoner(
    model="google/gemma-4-e4b",
    openai_api_key="lm-studio",            # any non-empty string for LM Studio
    openai_api_base="http://localhost:1234/v1",
    streaming=True,
    stream_reasoning=True,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("user", "{input}")
])
messages = prompt.format_messages(input="What is 2+2? Think step by step.")

in_thinking = False
for chunk in llm.stream(messages):
    rc = chunk.additional_kwargs.get("reasoning_content", "")
    if rc:
        if not in_thinking:
            print("\n=== THINKING ===", flush=True)
            in_thinking = True
        print(rc, end="", flush=True)
    if chunk.content:
        if in_thinking:
            print("\n=== ANSWER ===", flush=True)
            in_thinking = False
        print(chunk.content, end="", flush=True)
```

### DeepSeek-Reasoner / QwQ

For DeepSeek-Reasoner, QwQ, or any API that returns `reasoning_content` in the standard `/v1/chat/completions` response:

```python
from chatopenai_reasoner import ChatOpenAIReasoner

llm = ChatOpenAIReasoner(
    model="deepseek-reasoner",
    openai_api_key="YOUR_API_KEY",
    openai_api_base="https://api.deepseek.com",
    stream_reasoning=True,
)

# Streaming
for chunk in llm.stream("Explain quantum entanglement"):
    rc = chunk.additional_kwargs.get("reasoning_content", "")
    if rc:
        print(rc, end="", flush=True)
    else:
        print(chunk.content, end="", flush=True)

# Non-streaming (invoke)
result = llm.invoke("What is 2 + 2?")
print(result.content)
print(result.additional_kwargs.get("reasoning_content"))
```

### Async usage

```python
import asyncio
from chatopenai_reasoner import ChatOpenAIReasoner

llm = ChatOpenAIReasoner(
    model="google/gemma-4-e4b",
    openai_api_key="lm-studio",
    openai_api_base="http://127.0.0.1:1234/v1",
    stream_reasoning=True,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)

async def main():
    async for chunk in llm.astream("Explain recursion"):
        rc = chunk.additional_kwargs.get("reasoning_content", "")
        if rc:
            print(rc, end="", flush=True)
        else:
            print(chunk.content, end="", flush=True)

asyncio.run(main())
```

---

## API Reference — ChatOpenAIReasoner

`ChatOpenAIReasoner` extends `ChatOpenAI` from `langchain-openai`.  All standard `ChatOpenAI` parameters are supported in addition to the ones below.

### Constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | — | Model name / ID |
| `openai_api_key` | `str` | — | API key (`"lm-studio"` for LM Studio) |
| `openai_api_base` | `str` | — | Base URL of the API server (e.g. `http://127.0.0.1:1234/v1`) |
| `stream_reasoning` | `bool` | `True` | Whether to inject reasoning tokens into `additional_kwargs` during streaming |
| `streaming` | `bool` | `False` | Enable streaming mode |
| `extra_body` | `dict` | `None` | Extra fields forwarded to the API (e.g. `{"chat_template_kwargs": {"enable_thinking": True}}`) |
| `temperature` | `float` | `0.7` | Sampling temperature |
| `max_tokens` | `int` | `None` | Max output tokens |

### Key method: set_abort_event

```python
import threading
abort = threading.Event()
llm.set_abort_event(abort)

# In another thread:
abort.set()   # gracefully stops the current stream
```

---

## Streaming vs Non-streaming

| Mode | Method | Reasoning available |
|---|---|---|
| Streaming | `llm.stream(...)` / `llm.astream(...)` | `chunk.additional_kwargs["reasoning_content"]` |
| Non-streaming | `llm.invoke(...)` / `llm.ainvoke(...)` | `result.additional_kwargs["reasoning_content"]` |

When `stream_reasoning=False`, thinking tokens are **not** injected during streaming but are still available after `invoke()` completes.

---

## Cooperative cancellation

You can stop an ongoing stream at any time without killing threads:

```python
import threading
from chatopenai_reasoner import ChatOpenAIReasoner

abort = threading.Event()
llm = ChatOpenAIReasoner(
    model="google/gemma-4-e4b",
    openai_api_key="lm-studio",
    openai_api_base="http://127.0.0.1:1234/v1",
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)
llm.set_abort_event(abort)

def run():
    for chunk in llm.stream("Write a very long essay"):
        print(chunk.content, end="", flush=True)

import threading
t = threading.Thread(target=run)
t.start()

# Stop after 3 seconds
import time; time.sleep(3)
abort.set()
t.join()
```

---

## Supported backends

| Backend | Notes |
|---|---|
| LM Studio | `openai_api_base="http://127.0.0.1:1234/v1"`, enable thinking via `extra_body` |
| DeepSeek-Reasoner | `openai_api_base="https://api.deepseek.com"` |
| QwQ (Alibaba Cloud) | `openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"` |
| vLLM | Point `openai_api_base` at your vLLM server |
| Ollama | `openai_api_base="http://localhost:11434/v1"` |
| Any OpenAI-compatible API | As long as `reasoning_content` or `reasoning` is in the delta |

---

## LM Studio setup

1. Download and launch [LM Studio](https://lmstudio.ai/)
2. Load a reasoning-capable model (e.g. Gemma 4, QwQ, DeepSeek-R1)
3. Start the local server (default port: **1234**)
4. Use `openai_api_base="http://127.0.0.1:1234/v1"` — the standard OpenAI-compatible endpoint
5. Enable thinking tokens by passing `extra_body={"chat_template_kwargs": {"enable_thinking": True}}` for models that support it (e.g. Gemma-4 via vLLM/LM Studio)

---

## FAQ

**Q: Why does `reasoning_content` come back empty with the standard `ChatOpenAI`?**  
A: LangChain calls `chunk.model_dump()` on every SSE chunk, which strips `model_extra` fields.  `ChatOpenAIReasoner` intercepts the raw delta before serialisation.

**Q: Can I use this in a LangChain chain (`|` operator)?**  
A: Yes. `ChatOpenAIReasoner` is a full `ChatOpenAI` subclass and works anywhere `ChatOpenAI` is accepted.

**Q: What happens if the model doesn't return `reasoning_content`?**  
A: `additional_kwargs` will simply not contain the key.  Normal `content` streaming is unaffected.

**Q: Which field name is used for thinking tokens?**  
A: `ChatOpenAIReasoner` checks both `reasoning_content` (DeepSeek/QwQ) and `reasoning` (vLLM Gemma-4) and always exposes the value under `additional_kwargs["reasoning_content"]`.

**Q: How do I disable thinking tokens to save tokens/time?**  
A: Set `stream_reasoning=False`, or pass the appropriate `extra_body` flag to the model (e.g. `{"chat_template_kwargs": {"enable_thinking": False}}`).
