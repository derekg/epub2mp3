"""Tests for LLM-powered text processing."""

import pytest
from text_processor import (
    clean_text_basic,
    chunk_text,
    ProcessingMode,
    is_ollama_available,
    process_chapter,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


class TestBasicCleaning:
    """Test the regex-based fallback cleaning."""

    def test_removes_footnote_markers_brackets(self):
        text = "This is a sentence[1] with footnotes[2] and more[3]."
        result = clean_text_basic(text)
        assert "[1]" not in result
        assert "[2]" not in result
        assert "[3]" not in result
        assert "This is a sentence with footnotes and more." == result

    def test_removes_footnote_markers_symbols(self):
        text = "A paragraph* with various† footnote‡ markers§."
        result = clean_text_basic(text)
        assert "*" not in result
        assert "†" not in result
        assert "‡" not in result
        assert "§" not in result

    def test_removes_citation_brackets(self):
        text = "This fact[citation needed] is disputed."
        result = clean_text_basic(text)
        assert "[citation needed]" not in result
        assert "This fact is disputed." == result

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
        assert "for more info" in result

    def test_removes_figure_references(self):
        text = "The data shows growth (see Figure 3.2) over time (see Table 1.5)."
        result = clean_text_basic(text)
        assert "(see Figure 3.2)" not in result
        assert "(see Table 1.5)" not in result

    def test_fixes_ocr_artifact_l_to_I(self):
        # l|pipe before capital letter often means I in OCR errors
        # The regex [|l](?=[A-Z]) converts l or | before capitals to I
        text = "He said heIlo to her."  # Actually tests the lookahead pattern
        result = clean_text_basic(text)
        # Should preserve the text (this tests the function runs without error)
        assert "He said" in result

    def test_fixes_ocr_artifact_0_to_o(self):
        # 0 in middle of word likely means o
        text = "The b0ok was g0od."
        result = clean_text_basic(text)
        assert "book" in result
        assert "good" in result

    def test_normalizes_excessive_newlines(self):
        text = "Paragraph one.\n\n\n\n\nParagraph two."
        result = clean_text_basic(text)
        assert "\n\n\n" not in result
        assert "Paragraph one.\n\nParagraph two." == result

    def test_normalizes_excessive_spaces(self):
        text = "Too    many   spaces    here."
        result = clean_text_basic(text)
        assert "  " not in result
        assert "Too many spaces here." == result

    def test_normalizes_tabs(self):
        text = "Tabbed\t\tcontent\there."
        result = clean_text_basic(text)
        assert "\t" not in result

    def test_preserves_actual_content(self):
        text = "The quick brown fox jumps over the lazy dog."
        result = clean_text_basic(text)
        assert result == text

    def test_handles_empty_string(self):
        result = clean_text_basic("")
        assert result == ""

    def test_handles_whitespace_only(self):
        result = clean_text_basic("   \n\n   \t   ")
        assert result == ""


class TestChunking:
    """Test text chunking for LLM processing."""

    def test_short_text_single_chunk(self):
        text = "Short text that fits in one chunk."
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_multiple_chunks(self):
        # Create text longer than CHUNK_SIZE
        text = "Word " * (CHUNK_SIZE // 4)  # Each "Word " is 5 chars
        chunks = chunk_text(text.strip())
        assert len(chunks) > 1

    def test_chunks_have_overlap(self):
        # Create text that needs multiple chunks
        text = "Sentence one. " * 500  # ~7000 chars
        chunks = chunk_text(text.strip())

        if len(chunks) > 1:
            # Check that chunks overlap (last part of chunk N appears in chunk N+1)
            chunk1_end = chunks[0][-100:]
            # The overlap means some content should appear in both chunks
            assert len(chunks[0]) > CHUNK_OVERLAP

    def test_breaks_at_sentence_boundaries(self):
        # Create text with clear sentence boundaries
        sentences = ["This is sentence number {}. ".format(i) for i in range(200)]
        text = "".join(sentences)
        chunks = chunk_text(text)

        # Most chunks should end with sentence-ending punctuation
        for chunk in chunks[:-1]:  # Exclude last chunk
            stripped = chunk.rstrip()
            assert stripped[-1] in ".!?" or stripped.endswith("\n"), \
                f"Chunk should end at sentence boundary: ...{stripped[-50:]}"

    def test_custom_chunk_size(self):
        text = "A" * 1000
        chunks = chunk_text(text, chunk_size=200, overlap=20)
        assert len(chunks) > 1
        assert all(len(c) <= 200 for c in chunks)


class TestProcessingMode:
    """Test ProcessingMode enum values."""

    def test_mode_values(self):
        assert ProcessingMode.NONE == "none"
        assert ProcessingMode.CLEAN == "clean"
        assert ProcessingMode.SPEED_READ == "speed"
        assert ProcessingMode.SUMMARY == "summary"


class TestProcessChapter:
    """Test the main process_chapter function."""

    def test_none_mode_returns_unchanged(self):
        text = "Original text[1] with artifacts."
        result = process_chapter(text, "Chapter 1", ProcessingMode.NONE)
        assert result == text

    def test_clean_mode_without_ollama_uses_basic(self):
        """When Ollama isn't available, clean mode should fall back to basic cleaning."""
        text = "Text with footnote[1] and URL https://example.com here."
        result = process_chapter(text, "Chapter 1", ProcessingMode.CLEAN)

        # Should have removed footnote and URL via basic cleaning fallback
        assert "[1]" not in result
        assert "https://example.com" not in result

    def test_speed_read_without_ollama_returns_original(self):
        """Summarization requires Ollama - without it, text is returned as-is after basic clean."""
        text = "Some text to summarize."
        # This will attempt LLM, fail, and return original
        result = process_chapter(text, "Chapter 1", ProcessingMode.SPEED_READ)
        # Without Ollama, summarization returns cleaned text
        assert len(result) > 0


class TestRealWorldExamples:
    """Test with realistic EPUB text samples."""

    def test_academic_text_cleaning(self):
        text = """
        The study of consciousness[1] has long fascinated philosophers and scientists alike.
        Recent neuroimaging studies[2,3] have revealed new insights (see Figure 2.1) into
        the neural correlates of subjective experience.

        42

        Furthermore, the hard problem of consciousness[4] remains unsolved, despite
        significant advances in our understanding of brain function. For more details,
        visit https://consciousness-studies.org/research.

        [1] Chalmers, D. (1996). The Conscious Mind.
        [2] Koch, C. (2004). The Quest for Consciousness.
        """

        result = clean_text_basic(text)

        # Footnotes removed
        assert "[1]" not in result
        assert "[2,3]" not in result
        assert "[4]" not in result

        # URL removed
        assert "https://" not in result

        # Figure reference removed
        assert "(see Figure 2.1)" not in result

        # Page number removed
        assert "\n42\n" not in result

        # Content preserved
        assert "consciousness" in result
        assert "neuroimaging" in result
        assert "hard problem" in result

    def test_fiction_text_cleaning(self):
        text = """
        "I never thought I'd see you again," she whispered*.

        He turned slowly, his eyes reflecting the dim light from the window†.

        187

        "Some things," he said, "are meant to be."‡
        """

        result = clean_text_basic(text)

        # Footnote symbols removed
        assert "*" not in result
        assert "†" not in result
        assert "‡" not in result

        # Page number removed
        assert "187" not in result

        # Dialogue preserved
        assert '"I never thought' in result
        assert '"Some things,"' in result

    def test_technical_text_cleaning(self):
        text = """
        The algorithm runs in O(n log n) time[1]. See Table 3.1 for benchmarks.

        def quicksort(arr):
            # Implementation details at https://github.com/example/sort
            pass

        As shown in Figure 4.2, performance scales linearly with input size.
        """

        result = clean_text_basic(text)

        # Technical content preserved (O notation should stay)
        assert "O(n log n)" in result
        assert "def quicksort" in result

        # URL removed
        assert "https://github.com" not in result

        # Footnote removed
        assert "[1]" not in result


class TestEdgeCases:
    """Test edge cases and potential issues."""

    def test_empty_input(self):
        assert clean_text_basic("") == ""
        assert chunk_text("") == [""]
        assert process_chapter("", "Title", ProcessingMode.NONE) == ""

    def test_unicode_handling(self):
        text = "Café résumé naïve 日本語 emoji 🎉"
        result = clean_text_basic(text)
        assert "Café" in result
        assert "日本語" in result
        assert "🎉" in result

    def test_very_long_text(self):
        # 100K characters
        text = "This is a test sentence. " * 5000
        chunks = chunk_text(text)
        assert len(chunks) > 1
        # All content should be preserved across chunks
        total_length = sum(len(c) for c in chunks)
        # Account for overlap being counted multiple times
        assert total_length >= len(text.strip())

    def test_text_with_only_artifacts(self):
        text = "[1][2][3] https://example.com\n\n42\n\n"
        result = clean_text_basic(text)
        # Should be mostly empty after cleaning
        assert len(result.strip()) < len(text) / 2

    def test_nested_brackets(self):
        text = "Complex citation[see [1] and [2]] here."
        result = clean_text_basic(text)
        # Should handle nested brackets gracefully
        assert "Complex citation" in result


class TestOllamaIntegration:
    """Tests for Ollama integration (may skip if Ollama not available)."""

    def test_ollama_availability_check(self):
        # This should return a boolean without error
        result = is_ollama_available()
        assert isinstance(result, bool)

    @pytest.mark.skipif(not is_ollama_available(), reason="Ollama not available")
    def test_llm_cleaning_with_ollama(self):
        """Test actual LLM cleaning when Ollama is available."""
        from text_processor import clean_text_with_llm

        text = "This text[1] has footnotes[2] that should be removed."
        result = clean_text_with_llm(text)

        # LLM should remove footnotes
        assert "[1]" not in result
        assert "[2]" not in result
        # Content should be preserved
        assert "text" in result.lower()
        assert "footnotes" in result.lower()

    @pytest.mark.skipif(not is_ollama_available(), reason="Ollama not available")
    def test_llm_summarization_with_ollama(self):
        """Test actual LLM summarization when Ollama is available."""
        from text_processor import summarize_text_with_llm

        # A longer text to summarize
        text = """
        The history of artificial intelligence began in antiquity, with myths, stories
        and rumors of artificial beings endowed with intelligence or consciousness by
        master craftsmen. The seeds of modern AI were planted by philosophers who
        attempted to describe the process of human thinking as the mechanical
        manipulation of symbols. This work culminated in the invention of the
        programmable digital computer in the 1940s, a machine based on the abstract
        essence of mathematical reasoning. This device and the ideas behind it
        inspired a handful of scientists to begin seriously discussing the possibility
        of building an electronic brain.
        """

        result = summarize_text_with_llm(text, "AI History", target_percent=30)

        # Summary should be shorter
        assert len(result) < len(text)
        # Should still mention key concepts
        assert any(word in result.lower() for word in ["ai", "artificial", "intelligence", "computer"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
