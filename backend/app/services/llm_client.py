"""
LLM Client Service

Provides a unified, provider-agnostic interface for communicating with Large Language Models.
Primary Provider: Google Gemini (Gemini 2.5 Flash via official `google-genai` SDK).
Optional / Fallback Provider: OpenAI and OpenAI-compatible endpoints.
Provides dependency injection / mock hooks for offline unit testing without real API calls.
"""

import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

from dotenv import load_dotenv

# Ensure root .env variables are loaded
ROOT_DIR = Path(__file__).resolve().parents[3]
ENV_PATH = ROOT_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

# Google GenAI SDK (Official modern SDK for Gemini 2.5 Flash)
try:
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_errors = None
    genai_types = None

# OpenAI SDK (Fallback / Alternative provider)
try:
    import openai
    from openai import OpenAI
except ImportError:
    openai = None
    OpenAI = None


class LLMClientError(Exception):
    """Exception raised when an LLM provider call fails or returns unparseable content."""

    pass


class LLMClient:
    """
    Client for interacting with LLMs for structured JSON generation.
    Supports Google Gemini 2.5 Flash (default) and OpenAI-compatible backends.
    """

    GEMINI_FLASH_MODELS: List[str] = [
        "gemini-3.6-flash",
        "gemini-3.7-flash",
        "gemini-3.8-flash",
        "gemini-3.5-flash",
        "gemini-3-flash",
    ]

    GEMINI_FLASH_LITE_MODELS: List[str] = [
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
    ]

    GEMINI_FALLBACK_MODELS: List[str] = [
        "gemini-3.6-flash",
        "gemini-3.7-flash",
        "gemini-3.8-flash",
        "gemini-3.5-flash",
        "gemini-3-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
    ]

    _mock_responder: Optional[Callable[..., Dict[str, Any]]] = None

    @classmethod
    def set_mock_responder(
        cls, responder: Optional[Callable[..., Dict[str, Any]]]
    ) -> None:
        """
        Inject a mock responder callable for unit tests.

        Args:
            responder: Callable taking (system_prompt, user_prompt, [image_parts]) and returning a dict or JSON string,
                       or None to restore live API calls.
        """
        cls._mock_responder = responder

    @classmethod
    def get_provider(cls) -> str:
        """Return the configured LLM provider name (default: 'gemini')."""
        return os.getenv("LLM_PROVIDER", "gemini").strip().lower()

    @classmethod
    def get_model(cls) -> str:
        """Return the configured model name (default: 'gemini-3.6-flash')."""
        provider = cls.get_provider()
        if provider == "gemini":
            return os.getenv("GEMINI_MODEL") or os.getenv("LLM_MODEL", "gemini-3.6-flash")
        return os.getenv("LLM_MODEL", "gpt-4o-mini")

    @classmethod
    def get_gemini_candidate_models(cls, primary_model: Optional[str] = None) -> List[str]:
        """
        Return ordered Gemini model candidates for generation and fallback.

        Ordering Rules:
        1. Regular Flash models are always prioritized first.
        2. Flash Lite models are only used as the fallback pool after regular Flash models are exhausted.
        3. Flash Lite should NOT be the primary model while regular Flash models are available.
        4. If a primary Flash model is configured, it is attempted first among regular Flash models.
        """
        if not primary_model:
            primary_model = cls.get_model()

        is_lite = bool(primary_model and "lite" in primary_model.lower())

        flash_candidates: List[str] = []
        if not is_lite and primary_model:
            flash_candidates.append(primary_model)
        for m in cls.GEMINI_FLASH_MODELS:
            if m not in flash_candidates:
                flash_candidates.append(m)

        lite_candidates: List[str] = []
        if is_lite and primary_model:
            lite_candidates.append(primary_model)
        for m in cls.GEMINI_FLASH_LITE_MODELS:
            if m not in lite_candidates:
                lite_candidates.append(m)

        return flash_candidates + lite_candidates

    @classmethod
    def _sanitize_log_message(cls, message: Optional[str]) -> str:
        """
        Redact sensitive information such as GEMINI_API_KEY and GOOGLE_API_KEY from log messages and errors.
        """
        if not message:
            return ""
        sanitized = str(message)
        for env_var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
            val = os.getenv(env_var)
            if val and len(val.strip()) > 3:
                sanitized = sanitized.replace(val.strip(), "[REDACTED_API_KEY]")

        # Redact Google API key patterns and URL/Bearer credentials
        sanitized = re.sub(r"AIza[0-9A-Za-z\-_]{35}", "[REDACTED_API_KEY]", sanitized)
        sanitized = re.sub(r"(key=)[A-Za-z0-9\-_]+", r"\1[REDACTED_API_KEY]", sanitized)
        sanitized = re.sub(r"(Bearer\s+)[A-Za-z0-9\-_.]+", r"\1[REDACTED_API_KEY]", sanitized)
        return sanitized

    @classmethod
    def _is_quota_exhausted_error(cls, err: Exception) -> bool:
        """Check if an error represents daily quota exhaustion or 429 quota limit."""
        err_str = str(err).lower()
        code = getattr(err, "code", None)
        if code == 429:
            return True
        quota_keywords = [
            "429",
            "generaterequestsperday",
            "generate_content_free_tier_requests",
            "quota exceeded",
            "quota_exceeded",
            "daily quota",
            "per_day",
            "per day",
            "perday",
            "resource_exhausted",
            "check quota",
            "insufficient_quota",
            "quota limit",
            "exceeded quota",
        ]
        return any(k in err_str for k in quota_keywords)

    @classmethod
    def _is_unsupported_or_not_found_error(cls, err: Exception) -> bool:
        """Check if an error indicates that Google rejected a model as unsupported or not found."""
        err_str = str(err).lower()
        code = getattr(err, "code", None)
        if code == 404:
            return True
        not_found_keywords = [
            "not found",
            "not_found",
            "404",
            "unsupported",
            "is not supported",
            "model not found",
            "unknown model",
            "does not exist",
            "is not found",
            "invalid model",
            "not supported for",
        ]
        return any(k in err_str for k in not_found_keywords)

    @classmethod
    def _is_transient_service_error(cls, err: Exception) -> bool:
        """Check if an error is a transient service error (503 / 502 / 504 / service unavailable)."""
        err_str = str(err).lower()
        code = getattr(err, "code", None)
        if code in (500, 502, 503, 504):
            return True
        transient_keywords = [
            "503",
            "502",
            "504",
            "unavailable",
            "service unavailable",
            "temporarily unavailable",
            "timeout",
            "deadline exceeded",
            "connection reset",
            "server error",
        ]
        return any(k in err_str for k in transient_keywords)

    @classmethod
    def get_temperature(cls) -> float:
        """Return the sampling temperature (default 0.0 for deterministic extraction)."""
        try:
            return float(os.getenv("LLM_TEMPERATURE", "0.0"))
        except ValueError:
            return 0.0

    @classmethod
    def _create_gemini_client(cls) -> Any:
        """Create and return an initialized Gemini client."""
        if genai is None:
            raise LLMClientError(
                "google-genai package is not installed. Please install it with 'pip install google-genai'."
            )

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise LLMClientError(
                "GEMINI_API_KEY environment variable is missing. "
                "Please configure GEMINI_API_KEY in your root .env file."
            )

        try:
            return genai.Client(api_key=api_key)
        except Exception as exc:
            safe_exc = cls._sanitize_log_message(str(exc))
            raise LLMClientError(f"Failed to initialize Google Gemini client: {safe_exc}") from exc

    @classmethod
    def _create_openai_client(cls) -> Any:
        """Create and return an initialized OpenAI client."""
        if OpenAI is None:
            raise LLMClientError(
                "openai package is not installed. Please install it with 'pip install openai'."
            )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMClientError(
                "OPENAI_API_KEY environment variable is missing. "
                "Please configure OPENAI_API_KEY in your root .env file."
            )

        base_url = os.getenv("OPENAI_BASE_URL")
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url.strip('"\'')

        try:
            return OpenAI(**client_kwargs)
        except Exception as exc:
            raise LLMClientError(f"Failed to initialize OpenAI client: {exc}") from exc

    @classmethod
    def _clean_json_text(cls, text: str) -> str:
        """Strip optional markdown code fences from LLM responses."""
        cleaned = (text or "").strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return cleaned.strip()

    @classmethod
    def _generate_gemini(
        cls,
        system_prompt: str,
        user_prompt: str,
        image_parts: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Generate structured JSON using Google Gemini via google-genai, supporting multimodal images."""
        client = cls._create_gemini_client()
        model_name = cls.get_model()
        temperature = cls.get_temperature()

        try:
            config = genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                temperature=temperature,
            )

            if image_parts:
                contents: List[Any] = []
                for img in image_parts:
                    img_data = img.get("data")
                    mime = img.get("mime_type") or "image/jpeg"
                    if img_data:
                        contents.append(genai_types.Part.from_bytes(data=img_data, mime_type=mime))
                contents.append(user_prompt)
            else:
                contents = user_prompt

            # Multi-model fallback loop prioritizing regular Flash models first, then Flash Lite
            model_candidates = cls.get_gemini_candidate_models(model_name)

            response = None
            last_exc = None

            for idx, active_model in enumerate(model_candidates):
                max_attempts = 3
                fallback_reason = None

                for attempt in range(1, max_attempts + 1):
                    try:
                        response = client.models.generate_content(
                            model=active_model,
                            contents=contents,
                            config=config,
                        )
                        break
                    except Exception as call_err:
                        last_exc = call_err
                        err_str = str(call_err).lower()

                        # 1. Model unsupported or not found (404 / NotFound) -> graceful immediate fallback
                        if cls._is_unsupported_or_not_found_error(call_err):
                            fallback_reason = f"Model unsupported or not found (404): {cls._sanitize_log_message(str(call_err))}"
                            break

                        # 2. Daily quota exhaustion / 429 quota error -> immediately fallback without wasting retries
                        if cls._is_quota_exhausted_error(call_err):
                            fallback_reason = f"Daily quota exhaustion / 429 quota error: {cls._sanitize_log_message(str(call_err))}"
                            break

                        # 3. Check for retry delay in transient/rate limit errors
                        wait_match = re.search(r"retry in (\d+(?:\.\d+)?)s", str(call_err), re.IGNORECASE)
                        retry_wait = float(wait_match.group(1)) if wait_match else 0.0

                        # If retry delay is too large (> 10s) and fallback candidates remain, retrying is not useful
                        if retry_wait > 10.0 and (idx + 1) < len(model_candidates):
                            fallback_reason = f"Rate limit delay requires {retry_wait:.1f}s delay; retrying not useful"
                            break

                        # 4. Transient 503 / service-unavailable error
                        if cls._is_transient_service_error(call_err) or "429" in err_str:
                            if attempt < max_attempts:
                                sleep_sec = min(retry_wait + 2.0, 15.0) if retry_wait > 0 else 3.0 * attempt
                                safe_err = cls._sanitize_log_message(str(call_err))
                                logger.warning(
                                    f"{active_model} transient error (attempt {attempt}/{max_attempts}). "
                                    f"Backing off {sleep_sec:.1f}s: {safe_err}"
                                )
                                time.sleep(sleep_sec)
                                continue
                            else:
                                fallback_reason = f"Transient error retries exhausted ({max_attempts}/{max_attempts}): {cls._sanitize_log_message(str(call_err))}"
                                break

                        # Fatal / non-transient error: re-raise immediately
                        raise

                if response is not None:
                    # Successful model stops the fallback chain immediately
                    break

                # active_model failed; log fallback transition
                has_next = (idx + 1) < len(model_candidates)
                next_model = model_candidates[idx + 1] if has_next else "None (all fallback models exhausted)"
                safe_reason = cls._sanitize_log_message(fallback_reason or str(last_exc))
                logger.warning(
                    f"Fallback triggered: attempted model='{active_model}', "
                    f"reason for fallback='{safe_reason}', "
                    f"next model selected='{next_model}'"
                )

            if response is None and last_exc is not None:
                raise last_exc

            if response is None:
                raise LLMClientError("Gemini returned an empty response.")

            raw_text = response.text
            if not raw_text:
                raise LLMClientError("Gemini returned an empty response.")

            cleaned = cls._clean_json_text(raw_text)
            parsed_data = json.loads(cleaned)
            if not isinstance(parsed_data, dict):
                raise LLMClientError("Gemini response did not parse into a dictionary.")

            return parsed_data

        except json.JSONDecodeError as exc:
            safe_exc = cls._sanitize_log_message(str(exc))
            raise LLMClientError(f"Failed to parse Gemini output as JSON: {safe_exc}") from exc
        except Exception as exc:
            if genai_errors and isinstance(exc, genai_errors.APIError):
                safe_msg = cls._sanitize_log_message(exc.message)
                raise LLMClientError(f"Gemini API error ({exc.code}): {safe_msg}") from exc
            safe_exc = cls._sanitize_log_message(str(exc))
            raise LLMClientError(f"Gemini generation failed: {safe_exc}") from exc

    @classmethod
    def _generate_openai(
        cls,
        system_prompt: str,
        user_prompt: str,
    ) -> Dict[str, Any]:
        """Generate structured JSON using OpenAI or OpenAI-compatible backend."""
        client = cls._create_openai_client()
        model_name = cls.get_model()
        temperature = cls.get_temperature()

        try:
            response = client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            raw_content = response.choices[0].message.content
            if not raw_content:
                raise LLMClientError("LLM returned an empty response.")

            cleaned = cls._clean_json_text(raw_content)
            parsed_data = json.loads(cleaned)
            if not isinstance(parsed_data, dict):
                raise LLMClientError("LLM response did not parse into a dictionary.")

            return parsed_data

        except json.JSONDecodeError as exc:
            safe_exc = cls._sanitize_log_message(str(exc))
            raise LLMClientError(f"Failed to parse LLM output as JSON: {safe_exc}") from exc
        except Exception as exc:
            safe_exc = cls._sanitize_log_message(str(exc))
            raise LLMClientError(f"OpenAI generation failed: {safe_exc}") from exc

    @classmethod
    def generate_json(
        cls,
        system_prompt: str,
        user_prompt: str,
        image_parts: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Send a prompt (and optional multimodal image parts) to the configured LLM and return parsed JSON.

        Args:
            system_prompt: High-level instructions and schema definitions.
            user_prompt: Document text and extraction request.
            image_parts: Optional list of image dicts with 'data' (bytes) and 'mime_type' (str).

        Returns:
            Dict[str, Any]: Parsed JSON dictionary returned by the LLM.

        Raises:
            LLMClientError: If provider communication fails or output is not valid JSON.
        """
        # 1. Use injected mock responder if present (for tests)
        if cls._mock_responder is not None:
            try:
                import inspect
                sig = inspect.signature(cls._mock_responder)
                if len(sig.parameters) >= 3:
                    response = cls._mock_responder(system_prompt, user_prompt, image_parts)
                else:
                    response = cls._mock_responder(system_prompt, user_prompt)
                if isinstance(response, str):
                    cleaned = cls._clean_json_text(response)
                    return json.loads(cleaned)
                return response
            except Exception as exc:
                raise LLMClientError(f"Mock LLM responder failed: {exc}") from exc

        # 2. Dispatch to configured provider
        provider = cls.get_provider()
        if provider == "gemini":
            return cls._generate_gemini(system_prompt, user_prompt, image_parts=image_parts)
        elif provider == "openai":
            return cls._generate_openai(system_prompt, user_prompt)
        else:
            raise LLMClientError(
                f"Unsupported LLM provider: '{provider}'. Supported providers are: 'gemini', 'openai'."
            )

