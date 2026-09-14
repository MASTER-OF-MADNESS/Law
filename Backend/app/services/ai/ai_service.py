"""Centralized AI service: Gemini primary, Groq automatic fallback.

This is the ONLY module in the application that should import a vendor SDK
(directly, or via gemini_provider/groq_provider). Every other service depends
on AIService, never on a specific provider — so swapping providers, or adding
a third, never touches business logic.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import prismtrace
from prismtrace import PRISMtrace

from app.config import Settings
from app.models.schemas import ProviderName
from app.services.ai.base import AIProvider, AIProviderError
from app.services.ai.gemini_provider import GeminiProvider
from app.services.ai.groq_provider import GroqProvider

logger = logging.getLogger("lawoud.ai_service")


class AllProvidersFailedError(Exception):
    """Both Gemini and Groq failed (or neither is configured)."""


class AIService:
    def __init__(self, settings: Settings):
        self._timeout = settings.ai_timeout_seconds
        self._providers: list[AIProvider] = []
        if settings.has_gemini:
            self._providers.append(GeminiProvider(settings.gemini_api_key, settings.gemini_model))
        if settings.has_groq:
            self._providers.append(GroqProvider(settings.groq_api_key, settings.groq_model))
        if not self._providers:
            logger.warning(
                "No AI provider configured (GEMINI_API_KEY and GROQ_API_KEY both empty). "
                "generate_text/generate_json will raise AllProvidersFailedError."
            )

        self._prismtrace: PRISMtrace | None = None
        if settings.has_prismtrace:
            try:
                self._prismtrace = PRISMtrace(
                    api_key=settings.prismtrace_api_key,
                    host=settings.prismtrace_host,
                    project_id=settings.prismtrace_project_id,
                )
                logger.info("PRISM tracing initialized for project %s", settings.prismtrace_project_id)
            except Exception as e:
                logger.warning("Failed to initialize PRISMtrace: %s", e)

    def _trace_call(
        self,
        *,
        model: str,
        input_messages: list[dict[str, Any]],
        output: str,
        latency_ms: int,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self._prismtrace:
            return
        try:
            self._prismtrace.trace_llm(
                model=model,
                input_messages=input_messages,
                output=output,
                latency_ms=max(1, latency_ms),
                session_id=session_id or prismtrace.current_session(),
                metadata=metadata,
            )
        except Exception as e:
            logger.debug("Failed to record PRISM trace: %s", e)

    def close(self) -> None:
        """Flush in-flight traces and close connection pool."""
        if self._prismtrace:
            try:
                self._prismtrace.flush(timeout=5.0)
                self._prismtrace.close()
            except Exception as e:
                logger.debug("Error closing PRISMtrace: %s", e)

    @property
    def configured_providers(self) -> list[str]:
        return [p.name for p in self._providers]

    async def generate_text(
        self, prompt: str, *, system: str | None = None, session_id: str | None = None
    ) -> tuple[str, ProviderName]:
        """Try each provider in order; return (text, provider_used)."""
        input_messages: list[dict[str, Any]] = []
        if system:
            input_messages.append({"role": "system", "content": system})
        input_messages.append({"role": "user", "content": prompt})

        last_error: Exception | None = None
        start = time.perf_counter()
        for provider in self._providers:
            prov_start = time.perf_counter()
            try:
                text = await asyncio.wait_for(
                    provider.generate_text(prompt, system=system), timeout=self._timeout
                )
                prov_latency = int((time.perf_counter() - prov_start) * 1000)
                self._trace_call(
                    model=getattr(provider, "_model", provider.name),
                    input_messages=input_messages,
                    output=text,
                    latency_ms=prov_latency,
                    session_id=session_id,
                    metadata={"provider": provider.name, "call_type": "generate_text"},
                )
                return text, ProviderName(provider.name)
            except (AIProviderError, TimeoutError, asyncio.TimeoutError) as e:
                logger.warning("Provider %s failed for generate_text: %s", provider.name, e)
                last_error = e
                continue
        total_latency = int((time.perf_counter() - start) * 1000)
        self._trace_call(
            model="all_providers",
            input_messages=input_messages,
            output=f"[ERROR] {last_error}",
            latency_ms=total_latency,
            session_id=session_id,
            metadata={"call_type": "generate_text", "error": str(last_error)},
        )
        raise AllProvidersFailedError(str(last_error) if last_error else "no provider configured")

    async def generate_json(
        self, prompt: str, *, schema: dict[str, Any], system: str | None = None, session_id: str | None = None
    ) -> tuple[str, ProviderName]:
        """Try each provider in order; return (raw_json_text, provider_used)."""
        input_messages: list[dict[str, Any]] = []
        if system:
            input_messages.append({"role": "system", "content": system})
        input_messages.append({"role": "user", "content": prompt})

        last_error: Exception | None = None
        start = time.perf_counter()
        for provider in self._providers:
            prov_start = time.perf_counter()
            try:
                text = await asyncio.wait_for(
                    provider.generate_json(prompt, schema=schema, system=system),
                    timeout=self._timeout,
                )
                prov_latency = int((time.perf_counter() - prov_start) * 1000)
                self._trace_call(
                    model=getattr(provider, "_model", provider.name),
                    input_messages=input_messages,
                    output=text,
                    latency_ms=prov_latency,
                    session_id=session_id,
                    metadata={"provider": provider.name, "call_type": "generate_json"},
                )
                return text, ProviderName(provider.name)
            except (AIProviderError, TimeoutError, asyncio.TimeoutError) as e:
                logger.warning("Provider %s failed for generate_json: %s", provider.name, e)
                last_error = e
                continue
        total_latency = int((time.perf_counter() - start) * 1000)
        self._trace_call(
            model="all_providers",
            input_messages=input_messages,
            output=f"[ERROR] {last_error}",
            latency_ms=total_latency,
            session_id=session_id,
            metadata={"call_type": "generate_json", "error": str(last_error)},
        )
        raise AllProvidersFailedError(str(last_error) if last_error else "no provider configured")

    async def stream_text(
        self, prompt: str, *, system: str | None = None, session_id: str | None = None
    ) -> AsyncIterator[tuple[str, ProviderName]]:
        """Yield (chunk, provider_used) tuples.

        Fallback policy: if a provider fails before yielding any chunk, the next
        provider is tried transparently. If a provider fails *after* it has
        already yielded chunks, we cannot silently restart mid-answer without
        risking a duplicated or garbled response — so we re-raise and let the
        caller (the orchestrator) emit an `error` event and close the stream
        with whatever was already delivered.
        """
        input_messages: list[dict[str, Any]] = []
        if system:
            input_messages.append({"role": "system", "content": system})
        input_messages.append({"role": "user", "content": prompt})

        last_error: Exception | None = None
        start = time.perf_counter()
        for provider in self._providers:
            yielded_any = False
            collected_chunks: list[str] = []
            prov_start = time.perf_counter()
            try:
                async for chunk in provider.stream_text(prompt, system=system):
                    yielded_any = True
                    collected_chunks.append(chunk)
                    yield chunk, ProviderName(provider.name)
                prov_latency = int((time.perf_counter() - prov_start) * 1000)
                self._trace_call(
                    model=getattr(provider, "_model", provider.name),
                    input_messages=input_messages,
                    output="".join(collected_chunks),
                    latency_ms=prov_latency,
                    session_id=session_id,
                    metadata={"provider": provider.name, "call_type": "stream_text"},
                )
                return
            except (AIProviderError, TimeoutError, asyncio.TimeoutError) as e:
                logger.warning("Provider %s failed for stream_text: %s", provider.name, e)
                last_error = e
                if yielded_any:
                    prov_latency = int((time.perf_counter() - prov_start) * 1000)
                    self._trace_call(
                        model=getattr(provider, "_model", provider.name),
                        input_messages=input_messages,
                        output="".join(collected_chunks) + f"\n[STREAM ERROR: {e}]",
                        latency_ms=prov_latency,
                        session_id=session_id,
                        metadata={"provider": provider.name, "call_type": "stream_text", "error": str(e)},
                    )
                    raise
                continue
        total_latency = int((time.perf_counter() - start) * 1000)
        self._trace_call(
            model="all_providers",
            input_messages=input_messages,
            output=f"[ERROR] {last_error}",
            latency_ms=total_latency,
            session_id=session_id,
            metadata={"call_type": "stream_text", "error": str(last_error)},
        )
        raise AllProvidersFailedError(str(last_error) if last_error else "no provider configured")
