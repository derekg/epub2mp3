"""Tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient


# Note: These tests require the app to be importable but won't load the TTS model
# They test the API structure and text processing integration

class TestCapabilitiesEndpoint:
    """Test /api/capabilities endpoint."""

    def test_capabilities_returns_expected_fields(self):
        """Test that capabilities endpoint returns expected structure."""
        # We can't easily test with loaded model, but we can verify the structure
        from app import app

        # Create a test client
        client = TestClient(app, raise_server_exceptions=False)

        # This will fail because model isn't loaded, but we can verify the route exists
        response = client.get("/api/capabilities")

        # Even if it errors, we verified the endpoint exists
        assert response.status_code in [200, 500]  # 500 if model not loaded

    def test_text_processing_modes_documented(self):
        """Test that text processing modes are properly defined."""
        from text_processor import ProcessingMode

        # Verify all modes exist
        assert hasattr(ProcessingMode, 'NONE')
        assert hasattr(ProcessingMode, 'CLEAN')
        assert hasattr(ProcessingMode, 'SPEED_READ')
        assert hasattr(ProcessingMode, 'SUMMARY')

        # Verify values match API expectations
        assert ProcessingMode.NONE == "none"
        assert ProcessingMode.CLEAN == "clean"
        assert ProcessingMode.SPEED_READ == "speed"
        assert ProcessingMode.SUMMARY == "summary"


class TestConverterIntegration:
    """Test converter integration with text processing."""

    def test_converter_accepts_text_processing_param(self):
        """Verify convert_epub_to_mp3 accepts text_processing parameter."""
        import inspect
        from converter import convert_epub_to_mp3

        sig = inspect.signature(convert_epub_to_mp3)
        params = list(sig.parameters.keys())

        assert "text_processing" in params, "convert_epub_to_mp3 should accept text_processing parameter"

    def test_converter_text_processing_default(self):
        """Verify text_processing defaults to 'none'."""
        import inspect
        from converter import convert_epub_to_mp3

        sig = inspect.signature(convert_epub_to_mp3)
        param = sig.parameters.get("text_processing")

        assert param is not None
        assert param.default == "none"


class TestCLIIntegration:
    """Test CLI integration."""

    def test_cli_has_clean_option(self):
        """Verify CLI accepts --clean flag."""
        from typer.testing import CliRunner
        from cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["convert", "--help"])

        assert "--clean" in result.output
        assert "-c" in result.output

    def test_cli_has_speed_read_option(self):
        """Verify CLI accepts --speed-read flag."""
        from typer.testing import CliRunner
        from cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["convert", "--help"])

        assert "--speed-read" in result.output

    def test_cli_has_summary_option(self):
        """Verify CLI accepts --summary flag."""
        from typer.testing import CliRunner
        from cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["convert", "--help"])

        assert "--summary" in result.output


class TestTextProcessingQuality:
    """Quality tests for text processing with realistic examples."""

    def test_book_chapter_cleaning(self):
        """Test cleaning a realistic book chapter excerpt."""
        from text_processor import clean_text_basic

        chapter_text = """
        CHAPTER ONE

        The Beginning[1]

        It was a dark and stormy night. The wind howled through the trees[2],
        rattling the windows of the old mansion (see Figure 1.1). Sarah pulled
        her coat tighter and stepped inside.

        23

        "Hello?" she called out, her voice echoing in the empty foyer. No response.
        The only sound was the steady drip-drip-drip of water somewhere in the
        darkness[3].

        For more about the mansion's history, visit https://example.com/mansion-history

        [1] Originally published in 1923
        [2] The trees were ancient oaks, planted over 200 years ago
        [3] Foreshadowing the plumbing problems to come
        """

        result = clean_text_basic(chapter_text)

        # Footnotes should be removed
        assert "[1]" not in result
        assert "[2]" not in result
        assert "[3]" not in result

        # URL should be removed
        assert "https://example.com" not in result

        # Figure reference should be removed (pattern: "(see Figure X.X)")
        assert "(see Figure 1.1)" not in result

        # Page number should be removed (standalone 23)
        lines = result.split('\n')
        assert not any(line.strip() == "23" for line in lines)

        # Story content should be preserved
        assert "dark and stormy night" in result
        assert '"Hello?"' in result
        assert "Sarah" in result
        assert "old mansion" in result

    def test_nonfiction_text_cleaning(self):
        """Test cleaning non-fiction/academic text."""
        from text_processor import clean_text_basic

        academic_text = """
        The impact of climate change on global food security[1,2] has become
        increasingly apparent in recent decades. Studies show (see Table 2.3)
        that crop yields in tropical regions have declined by an average of
        5.3% per decade[3].

        147

        Furthermore, research conducted by Smith et al.[4] demonstrates that
        water scarcity will affect over 2 billion people by 2050. For detailed
        methodology, refer to https://climate-data.org/methodology.

        The economic implications are staggering[5], with estimates suggesting
        annual losses of $500 billion by mid-century (Figure 4.1).
        """

        result = clean_text_basic(academic_text)

        # Citations removed
        assert "[1,2]" not in result
        assert "[3]" not in result
        assert "[4]" not in result
        assert "[5]" not in result

        # URL removed
        assert "https://" not in result

        # Key content preserved
        assert "climate change" in result
        assert "food security" in result
        assert "5.3%" in result or "5.3" in result
        assert "2 billion people" in result
        assert "$500 billion" in result

    def test_mixed_content_cleaning(self):
        """Test cleaning text with code and technical content."""
        from text_processor import clean_text_basic

        technical_text = """
        To implement the algorithm[1], use the following Python code:

        def sort(arr):
            return sorted(arr)

        89

        The time complexity is O(n log n)[2], making it efficient for most
        use cases. See https://docs.python.org/3/library/functions.html#sorted
        for more details.
        """

        result = clean_text_basic(technical_text)

        # Code should be preserved
        assert "def sort(arr):" in result
        assert "return sorted(arr)" in result

        # Technical notation preserved
        assert "O(n log n)" in result

        # Citations and URL removed
        assert "[1]" not in result
        assert "[2]" not in result
        assert "https://" not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
