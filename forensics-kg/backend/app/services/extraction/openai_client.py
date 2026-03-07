import json
import re
from typing import Dict, Any, Optional
from openai import AsyncOpenAI, BadRequestError, RateLimitError, APITimeoutError, APIConnectionError
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryCallState


def _parse_json_safe(content: str) -> Dict[str, Any]:
    """Parse JSON from GPT response, handling markdown fences and malformed output."""
    # Try direct parse first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", content, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Return error structure instead of crashing
    logger.error(f"Failed to parse JSON from GPT response: {content[:200]}...")
    raise ValueError(f"GPT returned invalid JSON. Response starts with: {content[:100]}")


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
            )
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

    @_retry_decorator
    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        model_override: Optional[str] = None,
    ) -> str:
        """Simple chat completion without structured output."""
        response = await self._client.chat.completions.create(
            model=model_override or self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content or ""
