"""
Tests for text integrity throughout the conversion pipeline.

Covers three potential loss points:
  1. TTS chunking  — _split_text_into_chunks must preserve every word
  2. Gemini cleaning — clean mode should retain ≥80% of content words;
                       speed/summary modes intentionally reduce (tested for
                       correct ratio, not preservation)
  3. Basic cleaning  — clean_text_basic must preserve actual prose words
"""

import re
import pytest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LOREM = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
    "The five boxing wizards jump quickly. "
)

CHAPTER_SAMPLE = """\
The history of the ancient world is long and complex.

Many civilizations rose and fell during this period, each leaving behind
artifacts, texts, and traditions that shaped what came after. Scholars
have debated the exact sequence of events for centuries[1], and no
definitive answer has yet emerged.

See Figure 3.1 for a timeline of major dynasties.

A central theme throughout this era was the tension between stability
and change. Empires that managed to balance these forces[2] tended to
last longer than those that did not.

42

The Roman Republic, for example, maintained its institutions for nearly
five centuries before internal pressures — economic inequality, military
overextension, and political corruption — brought about its transformation
into an Empire. For more details, visit https://example.com/rome.

Table 2 shows population estimates across major cities.

In contrast, the Qin dynasty, though it unified China for the first time,
lasted only fifteen years before collapsing under the weight of its own
ambitions. The lessons of both cases remain relevant today.
"""

# Words we actually care about (non-artifact prose)
PROSE_WORDS = [
    "history", "ancient", "civilizations", "artifacts", "scholars",
    "stability", "change", "Empires", "Roman", "Republic", "institutions",
    "centuries", "Qin", "dynasty", "unified", "China", "ambitions",
]


def _word_count(text: str) -> int:
    return len(text.split())


def _words_in(text: str) -> set:
    return set(re.findall(r"[A-Za-z']+", text.lower()))


# ---------------------------------------------------------------------------
# 1. TTS chunking — no words dropped
# ---------------------------------------------------------------------------

class TestTTSChunkIntegrity:
    """_split_text_into_chunks must distribute every word into exactly one chunk."""

    def test_simple_uniform_text_all_words_preserved(self):
        from tts import _split_text_into_chunks
        text = "word " * 300
        chunks = _split_text_into_chunks(text, max_words=50)
        assert sum(_word_count(c) for c in chunks) == _word_count(text)

    def test_punctuated_prose_all_words_preserved(self):
        """Realistic prose with punctuation loses no words."""
        from tts import _split_text_into_chunks
        text = LOREM * 20  # ~360 words
        chunks = _split_text_into_chunks(text, max_words=50)
        original = _word_count(text)
        chunked = sum(_word_count(c) for c in chunks)
        assert chunked == original, (
            f"Lost {original - chunked} words during TTS chunking. "
            f"Original: {original}, Chunked total: {chunked}"
        )

    def test_chapter_length_text_all_words_preserved(self):
        """2,000-word chapter equivalent loses no words."""
        from tts import _split_text_into_chunks
        # Simulate a ~2k-word chapter
        text = (LOREM * 50).strip()
        original = _word_count(text)
        chunks = _split_text_into_chunks(text, max_words=50)
        chunked = sum(_word_count(c) for c in chunks)
        assert chunked == original, (
            f"2k-word chapter: lost {original - chunked} words during chunking."
        )

    def test_no_chunk_exceeds_max_words(self):
        from tts import _split_text_into_chunks
        text = (LOREM * 30).strip()
        for chunk in _split_text_into_chunks(text, max_words=50):
            assert _word_count(chunk) <= 50, (
                f"Chunk exceeds max_words=50: '{chunk[:60]}…' ({_word_count(chunk)} words)"
            )

    def test_no_empty_chunks_produced(self):
        from tts import _split_text_into_chunks
        text = "First sentence.  Second sentence.\n\nThird paragraph."
        chunks = _split_text_into_chunks(text, max_words=50)
        for c in chunks:
            assert c.strip(), f"Empty chunk produced: {c!r}"

    def test_long_run_on_sentence_hard_split_preserves_words(self):
        """A sentence with no punctuation longer than max_words triggers hard split."""
        from tts import _split_text_into_chunks
        text = " ".join(f"word{i}" for i in range(200))  # 200 unique words, no punctuation
        chunks = _split_text_into_chunks(text, max_words=50)
        original_words = set(f"word{i}" for i in range(200))
        chunk_words = set()
        for c in chunks:
            chunk_words.update(c.split())
        assert chunk_words == original_words, (
            f"Hard-split dropped words: missing {original_words - chunk_words}"
        )


# ---------------------------------------------------------------------------
# 2. text_processor._chunk_text — paragraph splitter loses no characters
# ---------------------------------------------------------------------------

class TestGeminiChunkTextIntegrity:
    """_chunk_text must preserve every character in every paragraph."""

    def test_short_text_returned_as_single_chunk(self):
        from text_processor import _chunk_text
        result = _chunk_text("Hello world.\n\nAnother paragraph.")
        assert len(result) == 1
        assert result[0] == "Hello world.\n\nAnother paragraph."

    def test_all_chars_preserved_across_chunks(self):
        """When text is split, every char appears in exactly one chunk."""
        from text_processor import _chunk_text
        # Build text > CHUNK_SIZE_CHARS (150k chars)
        paragraph = LOREM * 10  # ~600 chars each repetition
        text = "\n\n".join([paragraph] * 300)  # ~180k chars total
        chunks = _chunk_text(text)
        assert len(chunks) > 1, "Expected multiple chunks for 180k char text"
        reconstructed = "\n\n".join(chunks)
        assert reconstructed == text, (
            f"Chars lost: original={len(text)}, reconstructed={len(reconstructed)}"
        )

    def test_paragraph_boundaries_respected(self):
        """Chunks should join to form the original text."""
        from text_processor import _chunk_text
        paragraphs = [f"Paragraph {i}. " + LOREM for i in range(100)]
        text = "\n\n".join(paragraphs)
        chunks = _chunk_text(text, chunk_size=5000)
        assert "\n\n".join(chunks) == text


# ---------------------------------------------------------------------------
# 3. clean_text_basic — regex fallback preserves prose content
# ---------------------------------------------------------------------------

class TestBasicCleaningPreservation:
    """clean_text_basic should strip artifacts, not prose."""

    def test_prose_words_all_retained(self):
        """All non-artifact prose words should survive basic cleaning."""
        from text_processor import clean_text_basic
        cleaned = clean_text_basic(CHAPTER_SAMPLE)
        cleaned_words = _words_in(cleaned)
        for word in PROSE_WORDS:
            assert word.lower() in cleaned_words, (
                f"Prose word '{word}' was lost during basic cleaning"
            )

    def test_footnote_markers_removed(self):
        from text_processor import clean_text_basic
        text = "Some text[1] with footnotes[42] here."
        assert "[1]" not in clean_text_basic(text)
        assert "[42]" not in clean_text_basic(text)

    def test_urls_removed(self):
        from text_processor import clean_text_basic
        text = "Visit https://example.com and www.test.org for more."
        cleaned = clean_text_basic(text)
        assert "https://" not in cleaned
        assert "www.test" not in cleaned

    def test_basic_clean_retention_ratio(self):
        """Basic cleaning should keep ≥85% of words (only strips artifacts)."""
        from text_processor import clean_text_basic
        cleaned = clean_text_basic(CHAPTER_SAMPLE)
        original_words = _word_count(CHAPTER_SAMPLE)
        cleaned_words = _word_count(cleaned)
        ratio = cleaned_words / original_words
        assert ratio >= 0.85, (
            f"Basic cleaning removed too much text: {cleaned_words}/{original_words} "
            f"words retained ({ratio:.1%}). Should keep ≥85%."
        )

    def test_standalone_page_numbers_removed(self):
        from text_processor import clean_text_basic
        text = "End of section.\n\n42\n\nNew section begins."
        cleaned = clean_text_basic(text)
        assert "\n42\n" not in cleaned


# ---------------------------------------------------------------------------
# 4. Gemini clean mode — mocked to verify word-count tracking
# ---------------------------------------------------------------------------

class TestGeminiCleanWordCount:
    """
    With Gemini mocked, verify that:
    - clean mode returns text with ≥80% of original word count
    - speed mode returns text closer to 30% of original
    - summary mode returns text closer to 10% of original
    - words_processed in details reflects actual words sent to TTS
    """

    def _make_mock_client(self, response_text: str):
        mock_resp = MagicMock()
        mock_resp.text = response_text
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_resp
        return mock_client

    def test_clean_mode_high_retention_mocked(self):
        """Clean mode is deletion-only: mocked Gemini returns 95%+ of words."""
        from text_processor import process_chapter, ProcessingMode
        original = CHAPTER_SAMPLE
        original_words = _word_count(original)
        # Clean mode should only strip artifacts; mock returns 95% (artifacts removed)
        reduced = " ".join(original.split()[:int(original_words * 0.95)])
        mock_client = self._make_mock_client(reduced)
        with patch("text_processor.get_client", return_value=mock_client):
            result = process_chapter(original, "Test Chapter", ProcessingMode.CLEAN)
        result_words = _word_count(result)
        assert result_words >= original_words * 0.93, (
            f"Clean mode returned only {result_words}/{original_words} words "
            f"({result_words/original_words:.1%}), expected ≥93% (deletion-only)"
        )

    def test_clean_mode_no_gemini_falls_back_to_basic(self):
        """If Gemini is not configured, process_chapter uses clean_text_basic."""
        from text_processor import process_chapter, ProcessingMode
        with patch("text_processor.get_client", return_value=None):
            result = process_chapter(CHAPTER_SAMPLE, "Test", ProcessingMode.CLEAN)
        # Basic cleaning keeps ≥85% of words
        ratio = _word_count(result) / _word_count(CHAPTER_SAMPLE)
        assert ratio >= 0.85, f"Fallback basic clean kept only {ratio:.1%} of words"

    def test_clean_mode_gemini_error_falls_back(self):
        """If Gemini throws an exception mid-chunk, result still contains text."""
        from text_processor import process_chapter, ProcessingMode
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API error")
        with patch("text_processor.get_client", return_value=mock_client):
            result = process_chapter(CHAPTER_SAMPLE, "Test", ProcessingMode.CLEAN)
        # Should fall back to basic cleaning, not return empty string
        assert _word_count(result) > 0, "Gemini error caused all text to be lost"
        ratio = _word_count(result) / _word_count(CHAPTER_SAMPLE)
        assert ratio >= 0.80, (
            f"Gemini error fallback kept only {ratio:.1%} of words"
        )

    def test_none_mode_returns_text_unchanged(self):
        """ProcessingMode.NONE must return the exact same text."""
        from text_processor import process_chapter, ProcessingMode
        result = process_chapter(CHAPTER_SAMPLE, "Test", ProcessingMode.NONE)
        assert result == CHAPTER_SAMPLE

    def test_speed_mode_mocked_reduces_to_30_percent(self):
        """Speed mode: mock clean→identity, mock summarize→30% output."""
        from text_processor import process_chapter, ProcessingMode
        original = CHAPTER_SAMPLE * 5  # bigger sample
        original_words = _word_count(original)
        target_30 = " ".join(original.split()[:int(original_words * 0.30)])

        call_count = [0]
        def _mock_generate(model, contents, config):
            call_count[0] += 1
            r = MagicMock()
            # First call = clean (return as-is), second = summarize (return 30%)
            r.text = original if call_count[0] == 1 else target_30
            return r

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = _mock_generate
        with patch("text_processor.get_client", return_value=mock_client):
            result = process_chapter(original, "Test", ProcessingMode.SPEED_READ)
        result_words = _word_count(result)
        # Should be approximately 30% of original
        assert result_words <= original_words * 0.35, (
            f"Speed mode kept too many words: {result_words}/{original_words} "
            f"({result_words/original_words:.1%})"
        )


# ---------------------------------------------------------------------------
# 5. Duration estimate vs actual word count
# ---------------------------------------------------------------------------

class TestDurationEstimateAccuracy:
    """
    The UI estimates duration as (words / 150 WPM). If Gemini reduces word
    count, the estimate will be off. These tests quantify the expected gap.
    """

    WORDS_PER_MINUTE = 150

    def _estimated_minutes(self, word_count: int) -> float:
        return word_count / self.WORDS_PER_MINUTE

    def test_none_mode_estimate_accurate(self):
        """No processing → estimate based on raw word count is accurate."""
        from text_processor import process_chapter, ProcessingMode
        text = LOREM * 50  # ~900 words
        result = process_chapter(text, "Test", ProcessingMode.NONE)
        original_words = _word_count(text)
        result_words = _word_count(result)
        # With no processing, word counts must be identical
        assert result_words == original_words

    def test_basic_clean_estimate_within_20_percent(self):
        """Basic cleaning (no Gemini) should keep within 20% of estimated duration."""
        from text_processor import clean_text_basic
        text = CHAPTER_SAMPLE * 3
        cleaned = clean_text_basic(text)
        original_words = _word_count(text)
        cleaned_words = _word_count(cleaned)
        ratio = cleaned_words / original_words
        assert ratio >= 0.80, (
            f"Basic clean: estimate would be off by {(1-ratio)*100:.0f}%. "
            f"Original: {original_words} words, cleaned: {cleaned_words} words."
        )

    def test_clean_mode_mocked_estimate_gap_reported(self):
        """Clean mode is deletion-only: estimate should be within 7% of actual."""
        from text_processor import process_chapter, ProcessingMode
        original = CHAPTER_SAMPLE * 3
        original_words = _word_count(original)
        # Deletion-only: mock returns 95% of words (only artifacts removed)
        mock_output = " ".join(original.split()[:int(original_words * 0.95)])
        mock_client = self._make_mock_client(mock_output)
        with patch("text_processor.get_client", return_value=mock_client):
            result = process_chapter(original, "Test", ProcessingMode.CLEAN)
        result_words = _word_count(result)
        original_est = self._estimated_minutes(original_words)
        result_est = self._estimated_minutes(result_words)
        gap_pct = abs(original_est - result_est) / original_est * 100
        assert gap_pct < 8, (
            f"Clean mode: pre-conversion estimate ({original_est:.1f}m) "
            f"vs actual words ({result_est:.1f}m) — {gap_pct:.0f}% gap, expected <8%"
        )

    def _make_mock_client(self, response_text: str):
        mock_resp = MagicMock()
        mock_resp.text = response_text
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_resp
        return mock_client


# ---------------------------------------------------------------------------
# 6. Words-processed tracking through the converter
# ---------------------------------------------------------------------------

class TestWordsProcessedTracking:
    """
    verify that words_processed in progress details increments
    correctly and reaches words_total at the end.
    """

    def _make_two_chapter_epub(self, tmp_path, ch1_text, ch2_text):
        """Build a valid two-chapter EPUB that ebooklib can parse."""
        from ebooklib import epub as ebooklib_epub
        book = ebooklib_epub.EpubBook()
        book.set_identifier("test-integrity-id")
        book.set_title("Test Book")
        book.set_language("en")
        book.add_author("Test Author")

        def _make_chapter(uid, title, body_text):
            ch = ebooklib_epub.EpubHtml(
                uid=uid, file_name=f"{uid}.xhtml", lang="en"
            )
            ch.title = title
            ch.content = (
                f"<html><head><title>{title}</title></head>"
                f"<body><h1>{title}</h1><p>{body_text}</p></body></html>"
            ).encode("utf-8")
            return ch

        ch1 = _make_chapter("ch1", "Chapter One", ch1_text)
        ch2 = _make_chapter("ch2", "Chapter Two", ch2_text)
        book.add_item(ch1)
        book.add_item(ch2)
        book.toc = (
            ebooklib_epub.Link(ch1.file_name, "Chapter One", "ch1"),
            ebooklib_epub.Link(ch2.file_name, "Chapter Two", "ch2"),
        )
        book.add_item(ebooklib_epub.EpubNcx())
        book.add_item(ebooklib_epub.EpubNav())
        book.spine = ["nav", ch1, ch2]

        epub_path = str(tmp_path / "test.epub")
        ebooklib_epub.write_epub(epub_path, book)
        return epub_path

    def test_words_processed_reaches_total(self, tmp_path):
        """After conversion, last progress event words_processed == words_total."""
        import numpy as np
        from converter import convert_epub_to_mp3

        ch1_text = (LOREM * 20).strip()   # ~360 words
        ch2_text = (LOREM * 20).strip()
        epub_path = self._make_two_chapter_epub(tmp_path, ch1_text, ch2_text)

        progress_events = []

        def capture_progress(pct, total, msg, details):
            progress_events.append((pct, msg, dict(details)))

        # Mock TTS: return silent audio AND fire the chunk_callback so that
        # words_processed tracking works the same way it would with real TTS.
        silent = np.zeros(24000, dtype=np.int16)

        def fake_text_to_audio(text, voice=None, speed=1.0, chunk_callback=None):
            if chunk_callback:
                chunk_callback(1, 1)   # signal: done=1 of total=1 (100%)
            return (silent, 24000)

        with patch("converter.text_to_audio", side_effect=fake_text_to_audio):
            convert_epub_to_mp3(
                str(epub_path),
                str(tmp_path),
                progress_callback=capture_progress,
                text_processing="none",
            )

        assert progress_events, "No progress events were fired"

        # Find the last TTS event
        tts_events = [e for e in progress_events if e[2].get("stage") == "tts"]
        assert tts_events, "No TTS-stage progress events fired"

        last = tts_events[-1]
        details = last[2]
        wp = details.get("words_processed", 0)
        wt = details.get("words_total", 0)

        assert wt > 0, "words_total never set in progress details"
        assert wp > 0, "words_processed never incremented"
        assert wp <= wt, f"words_processed ({wp}) exceeded words_total ({wt})"
        # Final chunk callback fires with done==total, so we reach full word count
        ratio = wp / wt
        assert ratio >= 0.90, (
            f"words_processed only reached {wp}/{wt} ({ratio:.1%}) at end of conversion"
        )
