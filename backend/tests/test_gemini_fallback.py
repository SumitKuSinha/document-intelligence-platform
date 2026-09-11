"""
Automated Tests for Gemini LLM Model Fallback Strategy

Tests verify:
1. Primary Flash model is attempted first.
2. If primary Flash hits 429 daily quota, next Flash model is attempted.
3. Multiple Flash models can fail before reaching Flash Lite.
4. Flash Lite is eventually selected when all regular Flash candidates fail.
5. If gemini-3.5-flash-lite fails, gemini-3.1-flash-lite is attempted.
6. A successful Flash model stops the fallback chain immediately.
7. API key is never included in logs/errors.
8. Existing extraction tests continue to pass.
9. Fallback logs clearly show attempted model, reason for fallback, next model selected.
10. Google unsupported/not-found (404) model error gracefully moves to next candidate.
11. Flash Lite is not primary while regular Flash models are available.
"""

import logging
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure backend root is in sys.path
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.llm_client import LLMClient, LLMClientError
from google.genai import errors as g_errors


class TestGeminiFallbackStrategy(unittest.TestCase):
    """Test suite for Gemini model fallback hierarchy and error handling."""

    def setUp(self):
        """Ensure no mock responder pollutes fallback tests."""
        LLMClient.set_mock_responder(None)

    def tearDown(self):
        """Reset mock responder after each test."""
        LLMClient.set_mock_responder(None)

    # -------------------------------------------------------------------------
    # 1. Primary Flash model is attempted first
    # -------------------------------------------------------------------------
    def test_primary_flash_model_attempted_first_default(self):
        """Verify default primary Flash model (gemini-3.6-flash) is attempted first."""
        models_called = []
        mock_client = MagicMock()

        def mock_generate(model, contents, config):
            models_called.append(model)
            res = MagicMock()
            res.text = '{"success": true}'
            return res

        mock_client.models.generate_content.side_effect = mock_generate

        with patch.object(LLMClient, "_create_gemini_client", return_value=mock_client):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("LLM_MODEL", None)
                os.environ.pop("GEMINI_MODEL", None)

                result = LLMClient._generate_gemini("sys", "user")
                self.assertEqual(result, {"success": True})
                self.assertEqual(models_called, ["gemini-3.6-flash"])

    def test_primary_flash_model_attempted_first_configured(self):
        """Verify configured primary Flash model (e.g. gemini-3.7-flash) is attempted first."""
        models_called = []
        mock_client = MagicMock()

        def mock_generate(model, contents, config):
            models_called.append(model)
            res = MagicMock()
            res.text = '{"success": true}'
            return res

        mock_client.models.generate_content.side_effect = mock_generate

        with patch.object(LLMClient, "_create_gemini_client", return_value=mock_client):
            with patch.dict(os.environ, {"LLM_MODEL": "gemini-3.7-flash"}):
                result = LLMClient._generate_gemini("sys", "user")
                self.assertEqual(result, {"success": True})
                self.assertEqual(models_called, ["gemini-3.7-flash"])

    def test_fallback_candidates_order_default(self):
        """Verify fallback order prioritizes regular Flash models first, then Flash Lite models."""
        expected_order = [
            "gemini-3.6-flash",
            "gemini-3.7-flash",
            "gemini-3.8-flash",
            "gemini-3.5-flash",
            "gemini-3-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
        ]
        candidates = LLMClient.get_gemini_candidate_models("gemini-3.6-flash")
        self.assertEqual(candidates, expected_order)

    def test_flash_lite_not_primary_when_configured(self):
        """Verify Flash Lite is not the primary model while regular Flash models are available."""
        candidates = LLMClient.get_gemini_candidate_models("gemini-3.5-flash-lite")
        # Regular Flash models must still come first!
        self.assertEqual(candidates[0], "gemini-3.6-flash")
        # Verify all 5 regular Flash models precede any Flash Lite model
        first_lite_index = next(i for i, m in enumerate(candidates) if "lite" in m)
        self.assertEqual(first_lite_index, 5)

    # -------------------------------------------------------------------------
    # 2. If primary Flash hits 429 daily quota, next Flash model is attempted
    # -------------------------------------------------------------------------
    def test_primary_flash_429_quota_immediately_attempts_next_flash(self):
        """Verify 429 daily quota exhaustion moves immediately to the next Flash model without wasted retries."""
        models_called = []
        mock_client = MagicMock()

        def mock_generate(model, contents, config):
            models_called.append(model)
            if model == "gemini-3.6-flash":
                raise g_errors.APIError(429, {
                    "error": {
                        "code": 429,
                        "message": "Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests. Please retry in 45s.",
                        "status": "RESOURCE_EXHAUSTED",
                    }
                })
            res = MagicMock()
            res.text = '{"success": true}'
            return res

        mock_client.models.generate_content.side_effect = mock_generate

        with patch.object(LLMClient, "_create_gemini_client", return_value=mock_client):
            with patch.dict(os.environ, {"LLM_MODEL": "gemini-3.6-flash"}):
                result = LLMClient._generate_gemini("sys", "user")
                self.assertEqual(result, {"success": True})
                # gemini-3.6-flash was tried exactly once (no retry sleep waste), then gemini-3.7-flash
                self.assertEqual(models_called, ["gemini-3.6-flash", "gemini-3.7-flash"])

    # -------------------------------------------------------------------------
    # 3. Multiple Flash models can fail before reaching Flash Lite
    # -------------------------------------------------------------------------
    def test_multiple_flash_models_fail_before_reaching_flash_lite(self):
        """Verify multiple regular Flash models fail and fallback continues through regular Flash pool."""
        models_called = []
        mock_client = MagicMock()

        def mock_generate(model, contents, config):
            models_called.append(model)
            if model == "gemini-3.6-flash":
                raise g_errors.APIError(429, {
                    "error": {"code": 429, "message": "Daily quota exceeded", "status": "RESOURCE_EXHAUSTED"}
                })
            elif model == "gemini-3.7-flash":
                raise g_errors.APIError(404, {
                    "error": {"code": 404, "message": "Model gemini-3.7-flash not found"}
                })
            elif model == "gemini-3.8-flash":
                # Transient 503 with retry delay > 10s -> failover immediately
                raise g_errors.APIError(503, {
                    "error": {"code": 503, "message": "Service unavailable. Please retry in 20s."}
                })
            elif model == "gemini-3.5-flash":
                res = MagicMock()
                res.text = '{"status": "recovered_at_3.5_flash"}'
                return res
            raise RuntimeError(f"Unexpected model called: {model}")

        mock_client.models.generate_content.side_effect = mock_generate

        with patch.object(LLMClient, "_create_gemini_client", return_value=mock_client):
            with patch.dict(os.environ, {"LLM_MODEL": "gemini-3.6-flash"}):
                result = LLMClient._generate_gemini("sys", "user")
                self.assertEqual(result, {"status": "recovered_at_3.5_flash"})
                self.assertEqual(
                    models_called,
                    ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash", "gemini-3.5-flash"],
                )
                # Ensure Flash Lite models were not called
                self.assertNotIn("gemini-3.5-flash-lite", models_called)
                self.assertNotIn("gemini-3.1-flash-lite", models_called)

    # -------------------------------------------------------------------------
    # 4. Flash Lite is eventually selected when all regular Flash candidates fail
    # -------------------------------------------------------------------------
    def test_flash_lite_eventually_selected_when_all_flash_fail(self):
        """Verify Flash Lite is selected only after all 5 regular Flash models fail."""
        models_called = []
        mock_client = MagicMock()

        def mock_generate(model, contents, config):
            models_called.append(model)
            if model in [
                "gemini-3.6-flash",
                "gemini-3.7-flash",
                "gemini-3.8-flash",
                "gemini-3.5-flash",
                "gemini-3-flash",
            ]:
                raise g_errors.APIError(429, {
                    "error": {"code": 429, "message": "generaterequestsperday quota reached", "status": "RESOURCE_EXHAUSTED"}
                })
            elif model == "gemini-3.5-flash-lite":
                res = MagicMock()
                res.text = '{"fallback": "flash_lite_succeeded"}'
                return res
            raise RuntimeError(f"Unexpected model called: {model}")

        mock_client.models.generate_content.side_effect = mock_generate

        with patch.object(LLMClient, "_create_gemini_client", return_value=mock_client):
            with patch.dict(os.environ, {"LLM_MODEL": "gemini-3.6-flash"}):
                result = LLMClient._generate_gemini("sys", "user")
                self.assertEqual(result, {"fallback": "flash_lite_succeeded"})
                self.assertEqual(
                    models_called,
                    [
                        "gemini-3.6-flash",
                        "gemini-3.7-flash",
                        "gemini-3.8-flash",
                        "gemini-3.5-flash",
                        "gemini-3-flash",
                        "gemini-3.5-flash-lite",
                    ],
                )
                # gemini-3.1-flash-lite should not be called since gemini-3.5-flash-lite succeeded
                self.assertNotIn("gemini-3.1-flash-lite", models_called)

    # -------------------------------------------------------------------------
    # 5. If gemini-3.5-flash-lite fails, gemini-3.1-flash-lite is attempted
    # -------------------------------------------------------------------------
    def test_flash_lite_35_fails_flash_lite_31_attempted(self):
        """Verify when all Flash models and gemini-3.5-flash-lite fail, gemini-3.1-flash-lite is attempted."""
        models_called = []
        mock_client = MagicMock()

        def mock_generate(model, contents, config):
            models_called.append(model)
            if model != "gemini-3.1-flash-lite":
                raise g_errors.APIError(429, {
                    "error": {"code": 429, "message": "Daily quota limit reached", "status": "RESOURCE_EXHAUSTED"}
                })
            res = MagicMock()
            res.text = '{"fallback": "gemini-3.1-flash-lite_succeeded"}'
            return res

        mock_client.models.generate_content.side_effect = mock_generate

        with patch.object(LLMClient, "_create_gemini_client", return_value=mock_client):
            with patch.dict(os.environ, {"LLM_MODEL": "gemini-3.6-flash"}):
                result = LLMClient._generate_gemini("sys", "user")
                self.assertEqual(result, {"fallback": "gemini-3.1-flash-lite_succeeded"})
                self.assertEqual(
                    models_called,
                    [
                        "gemini-3.6-flash",
                        "gemini-3.7-flash",
                        "gemini-3.8-flash",
                        "gemini-3.5-flash",
                        "gemini-3-flash",
                        "gemini-3.5-flash-lite",
                        "gemini-3.1-flash-lite",
                    ],
                )

    # -------------------------------------------------------------------------
    # 6. A successful Flash model stops the fallback chain immediately
    # -------------------------------------------------------------------------
    def test_successful_flash_model_stops_fallback_immediately(self):
        """Verify a successful call stops the fallback chain and subsequent models are not called."""
        models_called = []
        mock_client = MagicMock()

        def mock_generate(model, contents, config):
            models_called.append(model)
            res = MagicMock()
            res.text = '{"invoice_id": "INV-001"}'
            return res

        mock_client.models.generate_content.side_effect = mock_generate

        with patch.object(LLMClient, "_create_gemini_client", return_value=mock_client):
            with patch.dict(os.environ, {"LLM_MODEL": "gemini-3.6-flash"}):
                result = LLMClient._generate_gemini("sys", "user")
                self.assertEqual(result, {"invoice_id": "INV-001"})
                self.assertEqual(models_called, ["gemini-3.6-flash"])

    # -------------------------------------------------------------------------
    # 7. API key is never included in logs/errors
    # -------------------------------------------------------------------------
    def test_api_key_never_included_in_logs_or_errors(self):
        """Verify GEMINI_API_KEY is redacted and never exposed in logs or exception messages."""
        secret_key = "AIzaSySecretApiKeyToNeverBeLeaked123"
        mock_client = MagicMock()

        def mock_generate(model, contents, config):
            raise g_errors.APIError(429, {
                "error": {
                    "code": 429,
                    "message": f"Daily quota limit exceeded for key {secret_key} url https://gemini.googleapis.com/v1beta?key={secret_key}",
                    "status": "RESOURCE_EXHAUSTED",
                }
            })

        mock_client.models.generate_content.side_effect = mock_generate

        with patch.object(LLMClient, "_create_gemini_client", return_value=mock_client):
            with patch.dict(os.environ, {"GEMINI_API_KEY": secret_key, "LLM_MODEL": "gemini-3.6-flash"}):
                with self.assertLogs("app.services.llm_client", level="WARNING") as log_ctx:
                    with self.assertRaises(LLMClientError) as exc_ctx:
                        LLMClient._generate_gemini("sys", "user")

                    # Check exception message
                    err_msg = str(exc_ctx.exception)
                    self.assertNotIn(secret_key, err_msg)
                    self.assertIn("[REDACTED_API_KEY]", err_msg)

                    # Check all logged messages
                    self.assertTrue(len(log_ctx.output) > 0)
                    for log_msg in log_ctx.output:
                        self.assertNotIn(secret_key, log_msg)
                        self.assertIn("[REDACTED_API_KEY]", log_msg)

    # -------------------------------------------------------------------------
    # 8. Fallback logs clearly show attempted model, reason, and next model
    # -------------------------------------------------------------------------
    def test_fallback_logs_show_attempted_model_reason_and_next_model(self):
        """Verify fallback logs clearly show attempted model, reason for fallback, and next model selected."""
        mock_client = MagicMock()

        def mock_generate(model, contents, config):
            if model == "gemini-3.6-flash":
                raise g_errors.APIError(429, {
                    "error": {"code": 429, "message": "Daily quota exceeded", "status": "RESOURCE_EXHAUSTED"}
                })
            res = MagicMock()
            res.text = '{"success": true}'
            return res

        mock_client.models.generate_content.side_effect = mock_generate

        with patch.object(LLMClient, "_create_gemini_client", return_value=mock_client):
            with patch.dict(os.environ, {"LLM_MODEL": "gemini-3.6-flash"}):
                with self.assertLogs("app.services.llm_client", level="WARNING") as log_ctx:
                    LLMClient._generate_gemini("sys", "user")

                    fallback_logs = [line for line in log_ctx.output if "Fallback triggered" in line]
                    self.assertTrue(len(fallback_logs) >= 1)
                    log_entry = fallback_logs[0]

                    # Verify required elements: attempted model, reason for fallback, next model selected
                    self.assertIn("attempted model='gemini-3.6-flash'", log_entry)
                    self.assertIn("reason for fallback=", log_entry)
                    self.assertIn("next model selected='gemini-3.7-flash'", log_entry)

    # -------------------------------------------------------------------------
    # 9. Unsupported / not found gracefully moves to next candidate
    # -------------------------------------------------------------------------
    def test_unsupported_or_not_found_moves_to_next_candidate(self):
        """Verify 404 / unsupported model errors gracefully move to the next candidate without crashing."""
        models_called = []
        mock_client = MagicMock()

        def mock_generate(model, contents, config):
            models_called.append(model)
            if model == "gemini-3.6-flash":
                raise g_errors.APIError(404, {
                    "error": {"code": 404, "message": "models/gemini-3.6-flash is not found for API version v1beta"}
                })
            res = MagicMock()
            res.text = '{"status": "ok"}'
            return res

        mock_client.models.generate_content.side_effect = mock_generate

        with patch.object(LLMClient, "_create_gemini_client", return_value=mock_client):
            with patch.dict(os.environ, {"LLM_MODEL": "gemini-3.6-flash"}):
                result = LLMClient._generate_gemini("sys", "user")
                self.assertEqual(result, {"status": "ok"})
                self.assertEqual(models_called, ["gemini-3.6-flash", "gemini-3.7-flash"])

    # -------------------------------------------------------------------------
    # 10. All models failing raises LLMClientError without exposing secrets
    # -------------------------------------------------------------------------
    def test_all_models_fail_raises_llm_client_error(self):
        """Verify when all candidates fail, LLMClientError is raised."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = g_errors.APIError(429, {
            "error": {"code": 429, "message": "Quota limit reached", "status": "RESOURCE_EXHAUSTED"}
        })

        with patch.object(LLMClient, "_create_gemini_client", return_value=mock_client):
            with patch.dict(os.environ, {"LLM_MODEL": "gemini-3.6-flash"}):
                with self.assertRaises(LLMClientError) as ctx:
                    LLMClient._generate_gemini("sys", "user")

                self.assertIn("Gemini API error (429)", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
