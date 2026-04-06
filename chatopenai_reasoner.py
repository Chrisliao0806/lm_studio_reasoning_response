"""
Reasoning LLM wrapper.

Provides ChatOpenAIReasoner — a universal drop-in replacement for ChatOpenAI
that properly surfaces ``reasoning_content`` from reasoning models.

Supported backends
------------------
1. **OpenAI-compatible APIs** (DeepSeek-Reasoner, QwQ, vLLM, …)  — default
   Uses the standard ``/v1/chat/completions`` endpoint via the OpenAI SDK.
   ``reasoning_content`` lives in ``model_extra`` on the live delta object and
   is captured *before* Pydantic's ``model_dump()`` can discard it.

2. **LM Studio native API**  (``lmstudio_native=True``)
   Uses LM Studio's ``/api/v1/chat`` endpoint, which natively exposes
   reasoning via SSE events (``reasoning.delta`` / ``message.delta``).
   The ``/v1/chat/completions`` shim does NOT support the reasoning parameter,
   so this mode is required to get thinking tokens from LM Studio.
   ``lmstudio_reasoning`` controls the reasoning budget:
   ``"off"`` | ``"on"``

Why naive overrides fail (OpenAI mode)
---------------------------------------
``BaseChatOpenAI._stream`` calls ``chunk.model_dump()`` on every streaming
chunk BEFORE passing it downstream.  Pydantic's ``model_dump()`` excludes
``model_extra`` fields, so ``reasoning_content`` is silently dropped before
any hook can see it.

Fix: override ``_stream``/``_astream`` to make the raw API call ourselves,
read ``reasoning_content`` from the LIVE delta object before serialisation,
then inject it back into ``gen_chunk.message.additional_kwargs``.

stream_reasoning=True  (default):
    reasoning injected into ``additional_kwargs["reasoning_content"]``
    → ReasoningAwareSSETokenCallback (in stream_handler.py) sees it.

stream_reasoning=False:
    reasoning NOT injected into additional_kwargs
    → only available in the final invoke() result.

Compatible with all LangChain usage patterns:
    llm.stream(...)  /  llm.invoke(...)
    async for c in llm.astream(...)  /  await llm.ainvoke(...)
    llm.batch([...])
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Iterator, List, Optional
import json
import logging
import threading

import httpx
import openai
import requests
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI


class ChatOpenAIReasoner(ChatOpenAI):
    """
    Universal ChatOpenAI subclass that surfaces ``reasoning_content`` from
    reasoning models.

    OpenAI-compatible mode (default, ``lmstudio_native=False``):
        Works with DeepSeek-Reasoner, QwQ, vLLM, and any API that returns
        ``reasoning_content`` in the delta's ``model_extra``.

    LM Studio native mode (``lmstudio_native=True``):
        Calls ``/api/v1/chat`` instead of ``/v1/chat/completions``.
        Set ``base_url`` to the LM Studio server (default ``http://localhost:1234``).
        Use ``lmstudio_reasoning`` to control the reasoning budget.
    """

    stream_reasoning: bool = True
    # ── LM Studio native mode ──────────────────────────────────────────────
    lmstudio_native: bool = False
    lmstudio_reasoning: str = "on"
    # ──────────────────────────────────────────────────────────────────────
    _abort_event: Optional[threading.Event] = None

    def set_abort_event(self, event: threading.Event) -> None:
        """Attach an abort event for cooperative cancellation."""
        object.__setattr__(self, "_abort_event", event)

    @staticmethod
    def _pick_reasoning(obj) -> Optional[str]:
        """
        Extract ``reasoning_content`` from a LIVE Pydantic object (before
        model_dump) or a plain dict (after model_dump, best-effort).
        """
        if isinstance(obj, dict):
            return obj.get("reasoning_content") or None
        val = getattr(obj, "reasoning_content", None)
        if not val:
            val = (getattr(obj, "model_extra", None) or {}).get("reasoning_content")
        return val or None

    # ── LM Studio helpers ──────────────────────────────────────────────────

    def _lmstudio_url(self) -> str:
        base = str(self.openai_api_base or "http://localhost:1234").rstrip("/")
        return f"{base}/api/v1/chat"

    def _lmstudio_headers(self) -> dict:
        # openai_api_key is SecretStr|None at runtime; Pylance may mis-infer the type.
        raw_key = self.openai_api_key  # noqa: SLF001
        secret_value = (
            getattr(raw_key, "_secret_value", None)
            or getattr(raw_key, "get_secret_value", lambda: None)()
        )
        key = secret_value or "lm-studio"
        return {"Authorization": f"Bearer {key}"}

    def _lmstudio_payload(self, messages: List[BaseMessage], stream: bool) -> dict:
        system_prompt: Optional[str] = None
        inputs: List[str] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_prompt = (
                    msg.content if isinstance(msg.content, str) else str(msg.content)
                )
            elif isinstance(msg, HumanMessage):
                inputs.append(
                    msg.content if isinstance(msg.content, str) else str(msg.content)
                )
        payload: dict[str, Any] = {
            "model": self.model_name,
            "input": inputs[-1] if inputs else "",
            "reasoning": self.lmstudio_reasoning,
            "temperature": self.temperature,
            "stream": stream,
        }
        if system_prompt:
            payload["system_prompt"] = system_prompt
        if self.max_tokens:
            payload["max_output_tokens"] = self.max_tokens
        return payload

    # ── LM Studio transports ───────────────────────────────────────────────

    def _lmstudio_stream(
        self,
        messages: List[BaseMessage],
        run_manager=None,
    ) -> Iterator[ChatGenerationChunk]:
        _log = logging.getLogger(__name__)
        current_event: Optional[str] = None
        with requests.post(
            self._lmstudio_url(),
            json=self._lmstudio_payload(messages, stream=True),
            headers=self._lmstudio_headers(),
            stream=True,
            timeout=None,
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if self._abort_event and self._abort_event.is_set():
                    _log.info(
                        "\U0001f6d1 [ChatOpenAIReasoner._lmstudio_stream] abort_event detected"
                    )
                    return
                if not raw_line:
                    current_event = None
                    continue
                line = raw_line.decode("utf-8")
                if line.startswith("event: "):
                    current_event = line[7:].strip()
                    continue
                if not line.startswith("data: "):
                    continue
                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                if current_event == "reasoning.delta" and self.stream_reasoning:
                    token = data.get("content", "")
                    chunk = AIMessageChunk(
                        content="",
                        additional_kwargs={"reasoning_content": token},
                    )
                    if run_manager:
                        run_manager.on_llm_new_token(token)
                    yield ChatGenerationChunk(message=chunk)

                elif current_event == "message.delta":
                    token = data.get("content", "")
                    chunk = AIMessageChunk(content=token)
                    if run_manager:
                        run_manager.on_llm_new_token(token)
                    yield ChatGenerationChunk(message=chunk)

    async def _lmstudio_astream(
        self,
        messages: List[BaseMessage],
        run_manager=None,
    ) -> AsyncIterator[ChatGenerationChunk]:
        _log = logging.getLogger(__name__)
        current_event: Optional[str] = None
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                self._lmstudio_url(),
                json=self._lmstudio_payload(messages, stream=True),
                headers=self._lmstudio_headers(),
                timeout=None,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if self._abort_event and self._abort_event.is_set():
                        _log.info(
                            "\U0001f6d1 [ChatOpenAIReasoner._lmstudio_astream] abort_event detected"
                        )
                        return
                    line = line.strip()
                    if not line:
                        current_event = None
                        continue
                    if line.startswith("event: "):
                        current_event = line[7:].strip()
                        continue
                    if not line.startswith("data: "):
                        continue
                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue

                    if current_event == "reasoning.delta" and self.stream_reasoning:
                        token = data.get("content", "")
                        chunk = AIMessageChunk(
                            content="",
                            additional_kwargs={"reasoning_content": token},
                        )
                        if run_manager:
                            await run_manager.on_llm_new_token(token)
                        yield ChatGenerationChunk(message=chunk)

                    elif current_event == "message.delta":
                        token = data.get("content", "")
                        chunk = AIMessageChunk(content=token)
                        if run_manager:
                            await run_manager.on_llm_new_token(token)
                        yield ChatGenerationChunk(message=chunk)

    def _lmstudio_generate(self, messages: List[BaseMessage]) -> ChatResult:
        resp = requests.post(
            self._lmstudio_url(),
            json=self._lmstudio_payload(messages, stream=False),
            headers=self._lmstudio_headers(),
            timeout=None,
        )
        resp.raise_for_status()
        data = resp.json()
        reasoning_text = "".join(
            i["content"] for i in data.get("output", []) if i["type"] == "reasoning"
        )
        message_text = "".join(
            i["content"] for i in data.get("output", []) if i["type"] == "message"
        )
        msg = AIMessage(
            content=message_text,
            additional_kwargs=(
                {"reasoning_content": reasoning_text} if reasoning_text else {}
            ),
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    # ── public overrides ───────────────────────────────────────────────────

    def _stream(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        messages: List[BaseMessage] = args[0] if args else kwargs.pop("messages")

        if self.lmstudio_native:
            run_manager = args[2] if len(args) > 2 else kwargs.get("run_manager")
            yield from self._lmstudio_stream(messages, run_manager)
            return

        stop = args[1] if len(args) > 1 else kwargs.pop("stop", None)
        run_manager = args[2] if len(args) > 2 else kwargs.pop("run_manager", None)
        self._ensure_sync_client_available()
        kwargs["stream"] = True
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        default_chunk_class = AIMessageChunk

        _log = logging.getLogger(__name__)
        with self.client.create(**payload) as response:
            for raw_chunk in response:
                if self._abort_event and self._abort_event.is_set():
                    _log.info(
                        "\U0001f6d1 [ChatOpenAIReasoner._stream] abort_event detected"
                    )
                    return

                reasoning: Optional[str] = None
                if getattr(raw_chunk, "choices", None):
                    reasoning = self._pick_reasoning(raw_chunk.choices[0].delta)

                chunk_dict = (
                    raw_chunk.model_dump()
                    if not isinstance(raw_chunk, dict)
                    else raw_chunk
                )
                gen_chunk = self._convert_chunk_to_generation_chunk(
                    chunk_dict, default_chunk_class, {}
                )
                if gen_chunk is None:
                    continue

                if reasoning and self.stream_reasoning:
                    gen_chunk.message.additional_kwargs["reasoning_content"] = reasoning

                default_chunk_class = gen_chunk.message.__class__

                if run_manager:
                    logprobs = (gen_chunk.generation_info or {}).get("logprobs")
                    run_manager.on_llm_new_token(
                        gen_chunk.text, chunk=gen_chunk, logprobs=logprobs
                    )

                yield gen_chunk

    async def _astream(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        messages: List[BaseMessage] = args[0] if args else kwargs.pop("messages")

        if self.lmstudio_native:
            run_manager = args[2] if len(args) > 2 else kwargs.get("run_manager")
            async for chunk in self._lmstudio_astream(messages, run_manager):
                yield chunk
            return

        stop = args[1] if len(args) > 1 else kwargs.pop("stop", None)
        run_manager = args[2] if len(args) > 2 else kwargs.pop("run_manager", None)
        kwargs["stream"] = True
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        default_chunk_class = AIMessageChunk

        _log = logging.getLogger(__name__)
        async with await self.async_client.create(**payload) as response:
            async for raw_chunk in response:
                if self._abort_event and self._abort_event.is_set():
                    _log.info(
                        "\U0001f6d1 [ChatOpenAIReasoner._astream] abort_event detected"
                    )
                    return

                reasoning: Optional[str] = None
                if getattr(raw_chunk, "choices", None):
                    reasoning = self._pick_reasoning(raw_chunk.choices[0].delta)

                chunk_dict = (
                    raw_chunk.model_dump()
                    if not isinstance(raw_chunk, dict)
                    else raw_chunk
                )
                gen_chunk = self._convert_chunk_to_generation_chunk(
                    chunk_dict, default_chunk_class, {}
                )
                if gen_chunk is None:
                    continue

                if reasoning and self.stream_reasoning:
                    gen_chunk.message.additional_kwargs["reasoning_content"] = reasoning

                default_chunk_class = gen_chunk.message.__class__

                if run_manager:
                    logprobs = (gen_chunk.generation_info or {}).get("logprobs")
                    await run_manager.on_llm_new_token(
                        gen_chunk.text, chunk=gen_chunk, logprobs=logprobs
                    )

                yield gen_chunk

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.lmstudio_native:
            return self._lmstudio_generate(messages)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def _create_chat_result(
        self,
        response: "dict | openai.BaseModel",
        generation_info: Optional[dict] = None,
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        if isinstance(response, openai.BaseModel) and getattr(
            response, "choices", None
        ):
            for i, choice in enumerate(response.choices):
                if i >= len(result.generations):
                    break
                reasoning = self._pick_reasoning(choice.message)
                if reasoning:
                    result.generations[i].message.additional_kwargs[
                        "reasoning_content"
                    ] = reasoning
        return result
