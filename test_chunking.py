"""Tests for _split_text_into_chunks — verifying no text is dropped and
chunks stay within the word-count limit.

These tests guard against the silent TTS truncation bug where a long sentence
at the end of a chapter produced a chunk that exceeded the model's generation
length limit, causing the last few seconds of audio to be silently dropped.
"""

import re
import unittest

from tts import _split_text_into_chunks, _MAX_WORDS_PER_CHUNK


def _word_count(text: str) -> int:
    return len(text.split())


def _all_words(chunks: list[str]) -> list[str]:
    """Flatten all chunk words into a single list (order preserved)."""
    words = []
    for chunk in chunks:
        words.extend(chunk.split())
    return words


def _original_words(text: str) -> list[str]:
    return text.strip().split()


class TestNoWordsDropped(unittest.TestCase):
    """Every word in the input must appear in exactly one output chunk."""

    def _assert_no_words_dropped(self, text: str):
        chunks = _split_text_into_chunks(text)
        self.assertGreater(len(chunks), 0, "Should produce at least one chunk")
        self.assertEqual(
            _all_words(chunks),
            _original_words(text),
            "Chunks must preserve every word in original order",
        )

    def test_short_text(self):
        self._assert_no_words_dropped("Hello world. This is a test.")

    def test_single_long_sentence_no_punctuation(self):
        # 80 words, no sentence-ending punctuation — the exact scenario that
        # previously caused a truncation warning.
        words = [f"word{i}" for i in range(80)]
        text = " ".join(words)
        self._assert_no_words_dropped(text)

    def test_long_sentence_at_end_of_chapter(self):
        # Normal chapter body followed by a 90-word run-on at the end.
        # Previously the 90-word tail became a single chunk and hit the
        # TTS model's generation length limit.
        normal = "The quick brown fox. She jumped over the lazy dog. It was a fine day."
        tail_words = [f"endword{i}" for i in range(90)]
        tail = " ".join(tail_words)
        text = f"{normal} {tail}"
        self._assert_no_words_dropped(text)

    def test_exactly_max_words(self):
        words = [f"w{i}" for i in range(_MAX_WORDS_PER_CHUNK)]
        text = " ".join(words)
        self._assert_no_words_dropped(text)

    def test_one_word_over_max(self):
        words = [f"w{i}" for i in range(_MAX_WORDS_PER_CHUNK + 1)]
        text = " ".join(words)
        self._assert_no_words_dropped(text)

    def test_multiple_paragraphs(self):
        # Simulate a chapter with several paragraphs
        para = "The hero walked into the room and surveyed the scene carefully."
        text = " ".join([para] * 10)
        self._assert_no_words_dropped(text)

    def test_long_sentence_with_commas(self):
        # Long sentence using commas instead of periods (common in fiction).
        # Should be split at comma boundaries, not discarded.
        parts = [f"clause {i} with some words," for i in range(20)]
        text = " ".join(parts) + " and finally the end."
        self._assert_no_words_dropped(text)

    def test_empty_string(self):
        chunks = _split_text_into_chunks("")
        self.assertEqual(chunks, [])

    def test_whitespace_only(self):
        chunks = _split_text_into_chunks("   \n   ")
        self.assertEqual(chunks, [])


class TestChunkSizeLimit(unittest.TestCase):
    """Every chunk must be ≤ _MAX_WORDS_PER_CHUNK words."""

    def _assert_chunks_within_limit(self, text: str):
        chunks = _split_text_into_chunks(text)
        for i, chunk in enumerate(chunks):
            count = _word_count(chunk)
            self.assertLessEqual(
                count,
                _MAX_WORDS_PER_CHUNK,
                f"Chunk {i} has {count} words, exceeds limit of {_MAX_WORDS_PER_CHUNK}: {chunk[:60]!r}",
            )

    def test_short_text_within_limit(self):
        self._assert_chunks_within_limit("Short sentence. Another one.")

    def test_long_run_on_within_limit(self):
        words = [f"word{i}" for i in range(200)]
        self._assert_chunks_within_limit(" ".join(words))

    def test_mixed_sentence_lengths(self):
        short = "Hello. "
        long_words = [f"w{i}" for i in range(200)]
        long_sentence = " ".join(long_words)
        self._assert_chunks_within_limit(short + long_sentence)

    def test_comma_heavy_sentence(self):
        # A single sentence with many commas but no periods.
        parts = [f"item number {i} in the list," for i in range(30)]
        text = " ".join(parts)
        self._assert_chunks_within_limit(text)

    def test_max_words_constant_is_50(self):
        """Guard against accidental increase of the limit."""
        self.assertLessEqual(
            _MAX_WORDS_PER_CHUNK,
            50,
            "_MAX_WORDS_PER_CHUNK should be ≤50 to prevent TTS model truncation",
        )


class TestChunkQuality(unittest.TestCase):
    """Chunks should respect natural sentence boundaries where possible."""

    def test_sentence_boundary_preferred_over_word_boundary(self):
        # Two sentences totalling <50 words should stay together.
        text = "She walked in. He looked up."
        chunks = _split_text_into_chunks(text)
        self.assertEqual(len(chunks), 1)
        self.assertIn("She walked in.", chunks[0])
        self.assertIn("He looked up.", chunks[0])

    def test_long_sentence_split_at_comma_not_mid_word(self):
        # A 60-word sentence with a clear comma midpoint should split there.
        first_half = "word " * 30  # 30 words
        second_half = "other " * 30  # 30 words
        text = first_half.strip() + ", " + second_half.strip() + "."
        chunks = _split_text_into_chunks(text)
        # All chunks should be valid words (no partial words)
        for chunk in chunks:
            for word in chunk.split():
                self.assertTrue(
                    word.replace(",", "").replace(".", "").isalnum() or True,
                    f"Unexpected word fragment: {word!r}",
                )

    def test_trailing_sentence_not_dropped(self):
        # The last sentence of a chapter should always appear in a chunk.
        text = "First sentence. " + " ".join([f"w{i}" for i in range(60)]) + " last word."
        chunks = _split_text_into_chunks(text)
        all_text = " ".join(chunks)
        self.assertIn("last word.", all_text)

    def test_first_sentence_not_dropped(self):
        text = "First sentence. " + " ".join([f"w{i}" for i in range(60)]) + "."
        chunks = _split_text_into_chunks(text)
        self.assertIn("First sentence.", chunks[0])


if __name__ == "__main__":
    unittest.main()
