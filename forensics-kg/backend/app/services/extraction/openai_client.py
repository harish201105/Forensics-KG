import json
import re
from typing import Dict, Any, Optional
from openai import AsyncOpenAI, BadRequestError, RateLimitError, APITimeoutError, APIConnectionError
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryCallState


def _strip_fences(content: str) -> str:
    """Remove surrounding markdown code fences if present."""
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", content, re.DOTALL)
    if fence_match:
        return fence_match.group(1)
    return content


def _extract_json_span(content: str) -> Optional[str]:
    """Extract the first balanced JSON object/array from a noisy string.

    Walks the text tracking brace/bracket depth (ignoring braces inside string
    literals) and returns the substring of the first complete top-level value.
    """
    start = None
    for i, ch in enumerate(content):
        if ch in "{[":
            start = i
            opener = ch
            break
    if start is None:
        return None

    closer = "}" if opener == "{" else "]"
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(content)):
        ch = content[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return content[start : i + 1]
    # Unbalanced (likely truncated) — return from start to end for repair
    return content[start:]


def _repair_truncated_json(snippet: str) -> str:
    """Best-effort repair of truncated JSON by closing open structures.

    Handles the common GPT failure mode where output is cut off at max_tokens:
    closes an unterminated string, drops trailing commas/partial keys, and
    appends the missing closing brackets in the correct order.
    """
    stack: list[str] = []
    in_str = False
    escape = False
    last_significant = -1
    for i, ch in enumerate(snippet):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
                last_significant = i
            continue
        if ch == '"':
            in_str = True
            last_significant = i
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
            last_significant = i
        elif ch in "}]":
            if stack:
                stack.pop()
            last_significant = i
        elif ch in "0123456789truefalsenull.-+eE":
            last_significant = i

    repaired = snippet
    if in_str:
        repaired += '"'
    else:
        # Trim a dangling comma or partial key/value after the last good token
        tail = repaired[last_significant + 1 :]
        if tail.strip() in (",", "") or tail.strip().endswith(","):
            repaired = repaired[: last_significant + 1]
        repaired = re.sub(r",\s*$", "", repaired)
    # Close any still-open containers, innermost first
    repaired += "".join(reversed(stack))
    return repaired


def _parse_json_safe(content: str) -> Dict[str, Any]:
    """Parse JSON from a model response, tolerating fences, prose, and truncation."""
    if not content or not content.strip():
        raise ValueError("Model returned an empty response.")

    # 1. Direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown fences and retry
    stripped = _strip_fences(content).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # 3. Extract the first balanced JSON span (handles leading/trailing prose)
    span = _extract_json_span(stripped)
    if span:
        try:
            return json.loads(span)
        except json.JSONDecodeError:
            # 4. Repair a likely-truncated span and retry
            try:
                repaired = _repair_truncated_json(span)
                result = json.loads(repaired)
                logger.warning(
                    "Recovered truncated/malformed JSON via repair (output was likely cut off)."
                )
                return result
            except json.JSONDecodeError:
                pass

    logger.error(f"Failed to parse JSON from model response: {content[:300]}...")
    raise ValueError(f"Model returned invalid JSON. Response starts with: {content[:120]}")


_RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError)


def _log_retry(rs: RetryCallState) -> None:
    exc = rs.outcome.exception() if rs.outcome else "unknown"
    sleep_time = rs.next_action.sleep if rs.next_action else "?"  # type: ignore[union-attr]
    logger.warning(f"OpenAI call failed ({exc}), retrying in {sleep_time}s...")


_retry_decorator = retry(
    retry=retry_if_exception_type(_RETRYABLE),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    before_sleep=_log_retry,
)


class OpenAIClient:
    """Wrapper for OpenAI API with structured output support."""

    def __init__(self, settings):
        self._client = AsyncOpenAI(api_key=settings.openai_api_key, timeout=120.0)
        self._model = settings.openai_model
        self._embedding_model = getattr(
            settings, "openai_embedding_model", "text-embedding-3-small"
        )

    @_retry_decorator
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per input (order preserved)."""
        if not texts:
            return []
        # OpenAI rejects empty strings; substitute a single space placeholder.
        cleaned = [t if (t and t.strip()) else " " for t in texts]
        response = await self._client.embeddings.create(
            model=self._embedding_model,
            input=cleaned,
        )
        # API returns items with an explicit index; sort to be safe.
        items = sorted(response.data, key=lambda d: d.index)
        return [item.embedding for item in items]

    async def embed_one(self, text: str) -> list[float]:
        """Embed a single text and return its vector."""
        vectors = await self.embed([text])
        return vectors[0] if vectors else []

    async def _call_with_schema_fallback(
        self,
        messages: list,
        response_schema: Dict[str, Any],
        schema_name: str,
        temperature: float,
        model: str,
    ) -> Dict[str, Any]:
        """Try json_schema (strict) first; fall back to json_object if unsupported."""
        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "schema": response_schema,
                        "strict": True,
                    },
                },
                temperature=temperature,
                max_tokens=16384,
            )
            if response.choices[0].finish_reason == "length":
                logger.warning("GPT response truncated (hit max_tokens). Output may be incomplete.")
            content = response.choices[0].message.content or ""
            return _parse_json_safe(content)
        except BadRequestError as e:
            error_msg = str(e).lower()
            if "json_schema" not in error_msg and "response_format" not in error_msg:
                raise
            logger.info(
                f"Model '{model}' does not support json_schema, falling back to json_object"
            )

        # Fallback: json_object with schema in system prompt
        schema_instruction = (
            "\n\nYou MUST respond with valid JSON matching this schema:\n"
            f"```json\n{json.dumps(response_schema, indent=2)}\n```\n"
            "Return ONLY the JSON object, no extra text."
        )
        fallback_messages = list(messages)
        fallback_messages[0] = {
            **fallback_messages[0],
            "content": fallback_messages[0]["content"] + schema_instruction,
        }
        response = await self._client.chat.completions.create(
            model=model,
            messages=fallback_messages,
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=16384,
        )
        content = response.choices[0].message.content or ""
        return _parse_json_safe(content)

    @_retry_decorator
    async def extract_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Dict[str, Any],
        temperature: float = 0.1,
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Call LLM with structured output, with automatic fallback for unsupported models."""
        model = model_override or self._model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return await self._call_with_schema_fallback(
            messages, response_schema, "extraction_result", temperature, model
        )

    @_retry_decorator
    async def analyze_image(
        self,
        system_prompt: str,
        user_prompt: str,
        image_base64: str,
        response_schema: Dict[str, Any],
        temperature: float = 0.1,
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Call LLM with an image for visual analysis, with automatic fallback."""
        model = model_override or self._model
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ]
        return await self._call_with_schema_fallback(
            messages, response_schema, "image_analysis", temperature, model
        )
