"""Tests for LLM-powered text processing (backend selection, chunking, cleaning)."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import text_processor
from text_processor import (
    CHUNK_SIZE_CHARS,
    DEFAULT_MODEL,
    ProcessingMode,
    _chunk_text,
    _is_truncated,
    clean_text_basic,
    get_backend,
    get_model,
    is_llm_available,
    process_chapter,
)


def _no_keys():
    """Patch environment so no LLM backend is configured."""
    return patch.dict(os.environ, {"OPENROUTER_API_KEY": "", "GEMINI_API_KEY": "", "LLM_MODEL": ""}, clear=False)


class TestBackendSelection:
    def test_no_keys_no_backend(self):
        with _no_keys():
            assert get_backend() is None
            assert is_llm_available() is False

    @pytest.mark.skipif(not text_processor.OPENAI_SDK_AVAILABLE, reason="openai SDK not installed")
    def test_openrouter_preferred_when_both_keys_set(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-key", "GEMINI_API_KEY": "g-key"}):
            assert get_backend() == "openrouter"

    @pytest.mark.skipif(not text_processor.GENAI_AVAILABLE, reason="google-genai not installed")
    def test_gemini_fallback_when_only_gemini_key(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "", "GEMINI_API_KEY": "g-key"}):
            assert get_backend() == "gemini"

    def test_default_model(self):
        with patch.dict(os.environ, {"LLM_MODEL": ""}):
            assert get_model().endswith(DEFAULT_MODEL.split("/")[-1])

    @pytest.mark.skipif(not text_processor.OPENAI_SDK_AVAILABLE, reason="openai SDK not installed")
    def test_model_override_via_env(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-key", "LLM_MODEL": "openai/gpt-5-nano"}):
            assert get_model() == "openai/gpt-5-nano"

    @pytest.mark.skipif(not text_processor.GENAI_AVAILABLE, reason="google-genai not installed")
    def test_google_prefix_stripped_for_gemini_backend(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "", "GEMINI_API_KEY": "g-key",
                                     "LLM_MODEL": "google/gemini-2.5-flash-lite"}):
            assert get_model() == "gemini-2.5-flash-lite"


class TestOpenRouterCleaning:
    """Clean-mode behavior against a mocked OpenRouter client."""

    def _response(self, content, finish_reason="stop"):
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=content), finish_reason=finish_reason,
        )])

    def _run_clean(self, text, response):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = response
        with (
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-key", "GEMINI_API_KEY": ""}),
            patch.object(text_processor, "OPENAI_SDK_AVAILABLE", True),
            patch.object(text_processor, "_get_client", return_value=mock_client),
        ):
            result = text_processor.clean_text_with_llm(text)
        return result, mock_client

    def test_cleaned_text_returned(self):
        original = "Sentence one stays. " * 20
        cleaned_body = original.strip()
        result, client = self._run_clean(original, self._response(cleaned_body))
        assert result == cleaned_body
        client.chat.completions.create.assert_called_once()
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == DEFAULT_MODEL
        assert kwargs["temperature"] == 0.1

    def test_truncated_single_paragraph_falls_back_to_basic(self):
        """finish_reason=length on an unsplittable paragraph → basic cleaning."""
        original = "One paragraph with a footnote[1] only. " * 10
        result, _ = self._run_clean(original.strip(), self._response("partial", "length"))
        assert "[1]" not in result
        assert "One paragraph" in result

    def test_short_output_triggers_retention_fallback(self):
        """Output far below retention ratio is treated as truncated."""
        original = "A full sentence that must be kept intact. " * 30
        result, _ = self._run_clean(original.strip(), self._response("way too short"))
        # single paragraph → falls back to basic cleaning of the original
        assert "must be kept intact" in result


class TestNoBackendFallbacks:
    def test_clean_mode_uses_basic_without_llm(self):
        with _no_keys():
            text = "Text with footnote[1] and URL https://example.com here."
            result = process_chapter(text, "Chapter 1", ProcessingMode.CLEAN)
        assert "[1]" not in result
        assert "https://example.com" not in result

    def test_summarize_returns_input_without_llm(self):
        with _no_keys():
            text = "Some text to summarize."
            assert text_processor.summarize_text_with_llm(text) == text

    def test_none_mode_returns_unchanged(self):
        text = "Original text[1] with artifacts."
        assert process_chapter(text, "Chapter 1", ProcessingMode.NONE) == text


class TestChunking:
    def test_short_text_single_chunk(self):
        text = "Short text that fits in one chunk."
        assert _chunk_text(text) == [text]

    def test_long_text_multiple_chunks(self):
        para = "Word " * 400  # ~2000 chars per paragraph
        text = "\n\n".join([para.strip()] * 30)  # ~60k chars total
        chunks = _chunk_text(text)
        assert len(chunks) > 1
        assert all(len(c) <= CHUNK_SIZE_CHARS for c in chunks)

    def test_content_preserved_across_chunks(self):
        para = "Sentence. " * 200
        text = "\n\n".join([para.strip()] * 40)
        chunks = _chunk_text(text)
        assert "\n\n".join(chunks) == text

    def test_empty_text(self):
        assert _chunk_text("") == [""]


class TestTruncationDetection:
    def test_full_output_not_truncated(self):
        text = "word " * 100
        assert _is_truncated(text, text) is False

    def test_short_output_is_truncated(self):
        original = "word " * 100
        assert _is_truncated(original, "word " * 50) is True

    def test_empty_original_not_truncated(self):
        assert _is_truncated("", "anything") is False


class TestProcessingMode:
    def test_mode_values(self):
        assert ProcessingMode.NONE == "none"
        assert ProcessingMode.CLEAN == "clean"
        assert ProcessingMode.SPEED_READ == "speed"
        assert ProcessingMode.SUMMARY == "summary"


class TestBasicCleaning:
    """Regex fallback cleaning (no LLM required)."""

    def test_removes_numeric_footnote_markers(self):
        text = "This is a sentence[1] with footnotes[2] and more[3]."
        result = clean_text_basic(text)
        assert "[1]" not in result and "[2]" not in result and "[3]" not in result
        assert result == "This is a sentence with footnotes and more."

    def test_removes_standalone_footnote_symbols(self):
        """Only standalone symbols are stripped — word-attached ones are left
        for the LLM, so emphasis like 2*3 or *word* is never mangled."""
        text = "A marker * on its own and a dagger † alone are removed."
        result = clean_text_basic(text)
        assert "*" not in result
        assert "†" not in result
        assert "marker" in result and "dagger" in result

    def test_preserves_editorial_bracket_insertions(self):
        """[sic] and similar editorial insertions must survive cleaning."""
        text = "He said it was there [sic] on the table."
        assert "[sic]" in clean_text_basic(text)

    def test_removes_standalone_page_numbers(self):
        text = "End of page.\n\n42\n\nStart of next page."
        result = clean_text_basic(text)
        assert "\n42\n" not in result
        assert "End of page." in result
        assert "Start of next page." in result

    def test_removes_urls(self):
        text = "Visit https://example.com for more info. Also check www.test.org please."
        result = clean_text_basic(text)
        assert "https://example.com" not in result
        assert "www.test.org" not in result
        assert "Visit" in result

    def test_removes_figure_references(self):
        text = "The data shows growth (see Figure 3.2) over time (see Table 1.5)."
        result = clean_text_basic(text)
        assert "(see Figure 3.2)" not in result
        assert "(see Table 1.5)" not in result

    def test_normalizes_whitespace(self):
        assert clean_text_basic("Paragraph one.\n\n\n\n\nParagraph two.") == "Paragraph one.\n\nParagraph two."
        assert clean_text_basic("Too    many   spaces    here.") == "Too many spaces here."
        assert "\t" not in clean_text_basic("Tabbed\t\tcontent\there.")

    def test_preserves_actual_content(self):
        text = "The quick brown fox jumps over the lazy dog."
        assert clean_text_basic(text) == text

    def test_unicode_handling(self):
        text = "Café résumé naïve 日本語 emoji 🎉"
        result = clean_text_basic(text)
        assert "Café" in result and "日本語" in result and "🎉" in result

    def test_handles_empty_and_whitespace(self):
        assert clean_text_basic("") == ""
        assert clean_text_basic("   \n\n   \t   ") == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
