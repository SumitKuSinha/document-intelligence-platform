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
            raise LLMClientError(f"Failed to initialize Google Gemini client: {exc}") from exc

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

            # Resilient multi-model retry loop for transient demand spikes (503 / 429)
            model_candidates = [model_name]
            for fallback in ["gemini-3.7-flash", "gemini-3.5-flash"]:
                if fallback not in model_candidates:
                    model_candidates.append(fallback)

            response = None
            last_exc = None

            for active_model in model_candidates:
                max_attempts = 3
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
                        if "generaterequestsperday" in err_str:
                            logger.warning(f"{active_model} daily quota exceeded. Trying fallback model.")
                            break
                        if ("503" in err_str or "unavailable" in err_str or "429" in err_str or "resource_exhausted" in err_str) and attempt < max_attempts:
                            import re
                            import time
                            wait_match = re.search(r"retry in (\d+(?:\.\d+)?)s", str(call_err), re.IGNORECASE)
                            if wait_match:
                                sleep_sec = min(float(wait_match.group(1)) + 2.0, 75.0)
                            else:
                                sleep_sec = 3.0 * attempt
                            logger.warning(f"{active_model} transient error (attempt {attempt}/{max_attempts}). Backing off {sleep_sec:.1f}s: {call_err}")
                            time.sleep(sleep_sec)
                            continue
                        if ("503" in err_str or "unavailable" in err_str or "429" in err_str):
                            logger.warning(f"{active_model} exhausted retries. Trying fallback model.")
                            break
                        raise
                if response is not None:
                    break

            if response is None and last_exc is not None:
                raise last_exc

            raw_text = response.text
            if not raw_text:
                raise LLMClientError("Gemini returned an empty response.")

            cleaned = cls._clean_json_text(raw_text)
            parsed_data = json.loads(cleaned)
            if not isinstance(parsed_data, dict):
                raise LLMClientError("Gemini response did not parse into a dictionary.")

            return parsed_data

        except json.JSONDecodeError as exc:
            raise LLMClientError(f"Failed to parse Gemini output as JSON: {exc}") from exc
        except Exception as exc:
            if genai_errors and isinstance(exc, genai_errors.APIError):
                raise LLMClientError(f"Gemini API error ({exc.code}): {exc.message}") from exc
            raise LLMClientError(f"Gemini generation failed: {exc}") from exc

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
            raise LLMClientError(f"Failed to parse LLM output as JSON: {exc}") from exc
        except Exception as exc:
            raise LLMClientError(f"OpenAI generation failed: {exc}") from exc

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

