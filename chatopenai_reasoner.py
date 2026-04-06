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
import logging
import threading

import openai
from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
)
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI


class ChatOpenAIReasoner(ChatOpenAI):
    """
    Universal ChatOpenAI subclass that surfaces ``reasoning_content`` from
    reasoning models.

    Works with DeepSeek-Reasoner, QwQ, vLLM, and any OpenAI-compatible API
    that returns ``reasoning_content`` in the delta's ``model_extra``.
    """

    stream_reasoning: bool = True
    _abort_event: Optional[threading.Event] = None

    def set_abort_event(self, event: threading.Event) -> None:
        """Attach an abort event for cooperative cancellation."""
        object.__setattr__(self, "_abort_event", event)

    # ── shared helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _pick_reasoning(obj) -> Optional[str]:
        """
        Extract reasoning from a LIVE Pydantic object (before
        model_dump) or a plain dict (after model_dump, best-effort).
        Checks both ``reasoning_content`` (DeepSeek/QwQ) and
        ``reasoning`` (vLLM Gemma-4).
        """
        if isinstance(obj, dict):
            return obj.get("reasoning_content") or obj.get("reasoning") or None
        val = getattr(obj, "reasoning_content", None)
        if not val:
            val = getattr(obj, "reasoning", None)
        if not val:
            extras = getattr(obj, "model_extra", None) or {}
            val = extras.get("reasoning_content") or extras.get("reasoning")
        return val or None

    # ── public overrides ───────────────────────────────────────────────────

    def _stream(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        messages: List[BaseMessage] = args[0] if args else kwargs.pop("messages")

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