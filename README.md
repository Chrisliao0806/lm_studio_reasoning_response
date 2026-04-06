# lm_studio_reasoning_response

A universal LangChain-compatible wrapper that properly surfaces `reasoning_content` (thinking tokens) from reasoning LLMs.  Supports both **OpenAI-compatible APIs** (DeepSeek-Reasoner, QwQ, vLLM, …) and the **LM Studio native API** (`/api/v1/chat`).

---

## Table of Contents

- [Background](#background)
- [Why naive overrides fail](#why-naive-overrides-fail)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [LM Studio native mode](#lm-studio-native-mode)
  - [OpenAI-compatible mode](#openai-compatible-mode)
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
| Thinking | `reasoning_content` | The model's internal chain-of-thought |
| Answer | `content` | The final response |

The standard LangChain `ChatOpenAI` silently **drops** `reasoning_content` before it reaches user code because Pydantic's `model_dump()` excludes `model_extra` fields.  This library fixes that.

---

## Why naive overrides fail

`BaseChatOpenAI._stream` calls `chunk.model_dump()` on every streaming chunk **before** passing it downstream.  Pydantic's `model_dump()` excludes `model_extra` fields, so `reasoning_content` is silently dropped before any hook can intercept it.

**Fix implemented here:** Override `_stream` / `_astream` to make the raw API call directly, read `reasoning_content` from the **live** delta object before serialisation, then inject it back into `gen_chunk.message.additional_kwargs["reasoning_content"]`.

---

## Features

- Drop-in replacement for `ChatOpenAI` — all LangChain usage patterns work unchanged
- Captures `reasoning_content` / thinking tokens in both **streaming** and **non-streaming** modes
- **LM Studio native mode** (`lmstudio_native=True`) calls `/api/v1/chat`, which natively exposes reasoning via SSE `reasoning.delta` / `message.delta` events
- **OpenAI-compatible mode** (default) works with DeepSeek-Reasoner, QwQ, vLLM, and any API returning `reasoning_content` in the delta
- Configurable reasoning budget for LM Studio: `"off"` | `"low"` | `"medium"` | `"high"` | `"on"`
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

### LM Studio native mode

Use this mode when running a reasoning model in **LM Studio**.  
The `/v1/chat/completions` shim in LM Studio does **not** expose reasoning tokens; you must use the native `/api/v1/chat` endpoint.

```python
from chatopenai_reasoner import ChatOpenAIReasoner

llm = ChatOpenAIReasoner(
    model="google/gemma-4-e4b",       # model ID as shown in LM Studio
    openai_api_key="lm-studio",        # any non-empty string
    openai_api_base="http://localhost:1234",
    lmstudio_native=True,
    lmstudio_reasoning="on",           # "off" | "on"
    stream_reasoning=True,
)

for chunk in llm.stream("Write a sorting algorithm"):
    rc = chunk.additional_kwargs.get("reasoning_content", "")
    if rc:
        print(rc, end="", flush=True)   # thinking tokens
    else:
        print(chunk.content, end="", flush=True)  # answer tokens
```

### OpenAI-compatible mode

For DeepSeek-Reasoner, QwQ, vLLM, or any API that returns `reasoning_content` in the standard `/v1/chat/completions` response:

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
    openai_api_base="http://localhost:1234",
    lmstudio_native=True,
    lmstudio_reasoning="on",
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
| `openai_api_base` | `str` | — | Base URL of the API server |
| `stream_reasoning` | `bool` | `True` | Whether to inject `reasoning_content` into `additional_kwargs` during streaming |
| `lmstudio_native` | `bool` | `False` | Enable LM Studio native `/api/v1/chat` mode |
| `lmstudio_reasoning` | `str` | `"on"` | Reasoning budget: `"off"` \| `"on"` |
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
    openai_api_base="http://localhost:1234",
    lmstudio_native=True,
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

| Backend | Mode | Notes |
|---|---|---|
| LM Studio | `lmstudio_native=True` | Requires LM Studio ≥ 0.3.x with a reasoning model loaded |
| DeepSeek-Reasoner | default | `openai_api_base="https://api.deepseek.com"` |
| QwQ (Alibaba Cloud) | default | `openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"` |
| vLLM | default | Point `openai_api_base` at your vLLM server |
| Ollama | default | Point `openai_api_base` at `http://localhost:11434/v1` |
| Any OpenAI-compatible API | default | As long as `reasoning_content` is in the delta |

---

## LM Studio setup

1. Download and launch [LM Studio](https://lmstudio.ai/)
2. Load a reasoning-capable model (e.g. Gemma 4, QwQ, DeepSeek-R1)
3. Start the local server (default port: **1234**)
4. Set `lmstudio_native=True` in `ChatOpenAIReasoner` — the standard `/v1/chat/completions` endpoint does **not** expose reasoning tokens
5. Control reasoning budget with `lmstudio_reasoning`: `"off"` | `"on"`

---

## FAQ

**Q: Why does `reasoning_content` come back empty with the standard `ChatOpenAI`?**  
A: LangChain calls `chunk.model_dump()` on every SSE chunk, which strips `model_extra` fields.  `ChatOpenAIReasoner` intercepts the raw delta before serialisation.

**Q: Can I use this in a LangChain chain (`|` operator)?**  
A: Yes. `ChatOpenAIReasoner` is a full `ChatOpenAI` subclass and works anywhere `ChatOpenAI` is accepted.

**Q: What happens if the model doesn't return `reasoning_content`?**  
A: `additional_kwargs` will simply not contain the key.  Normal `content` streaming is unaffected.

**Q: Does `lmstudio_native=True` work with non-reasoning models?**  
A: Yes, but the `reasoning.delta` SSE events will not be emitted.  Only `message.delta` events will appear.

**Q: How do I disable thinking tokens to save tokens/time?**  
A: Set `lmstudio_reasoning="off"` (LM Studio mode) or `stream_reasoning=False` (OpenAI mode).
