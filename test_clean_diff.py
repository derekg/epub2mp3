#!/usr/bin/env python3
"""Compare original vs cleaned text to verify Gemini cleaning quality."""

import difflib
import sys
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from text_processor import clean_text_with_gemini, clean_text_basic, is_gemini_available
from converter import parse_epub


def word_count(text: str) -> int:
    return len(text.split())


def char_count(text: str) -> int:
    return len(text)


def estimate_tokens(text: str) -> int:
    """Rough estimate: ~4 chars per token."""
    return len(text) // 4


def show_diff(original: str, cleaned: str, context_lines: int = 3):
    """Show a unified diff of changes."""
    orig_lines = original.splitlines(keepends=True)
    clean_lines = cleaned.splitlines(keepends=True)

    diff = difflib.unified_diff(
        orig_lines, clean_lines,
        fromfile='original', tofile='cleaned',
        lineterm='', n=context_lines
    )

    diff_text = ''.join(diff)
    return diff_text


def analyze_changes(original: str, cleaned: str) -> dict:
    """Analyze what changed between original and cleaned."""
    orig_words = set(original.lower().split())
    clean_words = set(cleaned.lower().split())

    removed_words = orig_words - clean_words
    added_words = clean_words - orig_words

    # Look for specific patterns removed
    import re
    footnotes = len(re.findall(r'\[\d+\]', original)) - len(re.findall(r'\[\d+\]', cleaned))
    urls = len(re.findall(r'https?://\S+', original)) - len(re.findall(r'https?://\S+', cleaned))
    figure_refs = len(re.findall(r'[Ff]igure \d+', original)) - len(re.findall(r'[Ff]igure \d+', cleaned))

    return {
        "removed_word_count": len(removed_words),
        "added_word_count": len(added_words),
        "footnotes_removed": footnotes,
        "urls_removed": urls,
        "figure_refs_removed": figure_refs,
        "sample_removed": list(removed_words)[:20],
        "sample_added": list(added_words)[:20],
    }


def compare_chapter(original: str, title: str = "Chapter"):
    """Run comparison on a single chapter."""
    print(f"\n{'='*60}")
    print(f"CHAPTER: {title}")
    print(f"{'='*60}")

    # Stats before
    orig_chars = char_count(original)
    orig_words = word_count(original)
    orig_tokens = estimate_tokens(original)

    print(f"\n📊 ORIGINAL TEXT STATS:")
    print(f"   Characters: {orig_chars:,}")
    print(f"   Words:      {orig_words:,}")
    print(f"   Est tokens: {orig_tokens:,}")
    print(f"   Context %:  {orig_tokens / 1_000_000 * 100:.2f}% of 1M input limit")

    # Clean with Gemini
    print(f"\n🔄 Cleaning with Gemini 3 Flash...")

    def progress(msg):
        print(f"   {msg}")

    cleaned = clean_text_with_gemini(original, progress)

    # Stats after
    clean_chars = char_count(cleaned)
    clean_words = word_count(cleaned)
    clean_tokens = estimate_tokens(cleaned)

    print(f"\n📊 CLEANED TEXT STATS:")
    print(f"   Characters: {clean_chars:,}")
    print(f"   Words:      {clean_words:,}")
    print(f"   Est tokens: {clean_tokens:,}")

    # Compare
    char_diff = orig_chars - clean_chars
    word_diff = orig_words - clean_words
    pct_change = (1 - clean_chars / orig_chars) * 100 if orig_chars > 0 else 0

    print(f"\n📈 CHANGES:")
    print(f"   Characters removed: {char_diff:,} ({pct_change:.1f}%)")
    print(f"   Words removed:      {word_diff:,}")

    # Analyze what was removed
    analysis = analyze_changes(original, cleaned)
    print(f"\n🔍 ANALYSIS:")
    print(f"   Footnote markers removed: {analysis['footnotes_removed']}")
    print(f"   URLs removed:             {analysis['urls_removed']}")
    print(f"   Figure refs removed:      {analysis['figure_refs_removed']}")
    print(f"   Unique words removed:     {analysis['removed_word_count']}")
    print(f"   Unique words added:       {analysis['added_word_count']}")

    if analysis['sample_removed']:
        print(f"\n   Sample removed words: {', '.join(analysis['sample_removed'][:10])}")
    if analysis['sample_added']:
        print(f"   Sample added words:   {', '.join(analysis['sample_added'][:10])}")

    # Show diff (first 50 lines)
    diff = show_diff(original, cleaned)
    diff_lines = diff.split('\n')

    if diff_lines and diff_lines[0]:
        print(f"\n📝 DIFF (first 50 lines of changes):")
        print("-" * 40)
        for line in diff_lines[:50]:
            if line.startswith('+') and not line.startswith('+++'):
                print(f"\033[32m{line}\033[0m")  # Green for additions
            elif line.startswith('-') and not line.startswith('---'):
                print(f"\033[31m{line}\033[0m")  # Red for removals
            else:
                print(line)
        if len(diff_lines) > 50:
            print(f"... ({len(diff_lines) - 50} more lines)")
    else:
        print("\n   No significant changes detected in diff.")

    return {
        "title": title,
        "original": {"chars": orig_chars, "words": orig_words, "tokens": orig_tokens},
        "cleaned": {"chars": clean_chars, "words": clean_words, "tokens": clean_tokens},
        "change_pct": pct_change,
        "analysis": analysis,
    }


def test_with_epub(epub_path: str, max_chapters: int = 3):
    """Test cleaning with chapters from an EPUB file."""
    print(f"\n🔍 Testing with EPUB: {epub_path}")

    book = parse_epub(epub_path)
    print(f"   Title: {book.title}")
    print(f"   Author: {book.author}")
    print(f"   Chapters: {len(book.chapters)}")

    results = []
    chapters_to_test = min(max_chapters, len(book.chapters))

    tested = 0
    for i, ch in enumerate(book.chapters):
        if tested >= max_chapters:
            break

        # Skip front/back matter and short chapters
        if ch.is_front_matter or ch.is_back_matter:
            print(f"\n⏭️  Skipping '{ch.title}' (front/back matter)")
            continue
        if ch.word_count < 500:
            print(f"\n⏭️  Skipping '{ch.title}' (too short: {ch.word_count} words)")
            continue

        result = compare_chapter(ch.text, ch.title)
        results.append(result)
        tested += 1

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    total_orig_words = sum(r["original"]["words"] for r in results)
    total_clean_words = sum(r["cleaned"]["words"] for r in results)
    total_orig_tokens = sum(r["original"]["tokens"] for r in results)

    print(f"\nTotal chapters tested: {len(results)}")
    print(f"Total original words:  {total_orig_words:,}")
    print(f"Total cleaned words:   {total_clean_words:,}")
    print(f"Total words removed:   {total_orig_words - total_clean_words:,}")
    print(f"Overall reduction:     {(1 - total_clean_words/total_orig_words)*100:.1f}%")
    print(f"\nContext window usage:  {total_orig_tokens:,} tokens ({total_orig_tokens/1_000_000*100:.2f}% of 1M limit)")

    return results


def test_with_sample():
    """Test with a sample text containing various artifacts."""
    sample = """
Chapter 1: Introduction to Machine Learning

Machine learning[1] is a subset of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed[2].

See Figure 1.1 for an overview of the machine learning process.

42

The field of machine learning has grown tremendously in recent years, driven by three key factors: the availability of large datasets, increased computational power, and advances in algorithms[3][4]. For more information, visit https://example.com/ml-intro or check out www.machinelearning.org for additional resources.

KEY CONCEPTS†

* Supervised Learning - Learning from labeled data
* Unsupervised Learning - Finding patterns in unlabeled data
* Reinforcement Learning - Learning through trial and error

According to recent studies (see Table 2.3), the global machine learning market is expected to reach $117.19 billion by 2027, growing at a CAGR of 39.2% from 2020 to 2027*.

---

Page 43

The three main types of machine learning are:

1. **Supervised Learning**: The algorithm learns from labeled training data[5], making predictions based on that data. Common applications include spam detection, image classification, and medical diagnosis.

2. **Unsupervised Learning**: The algorithm finds patterns in data without predefined labels (see Figure 1.2). This includes clustering, dimensionality reduction, and anomaly detection.

3. **Reinforcement Learning**[6]: The algorithm learns by interacting with an environment, receiving feedback in the form of rewards or penalties.

† These concepts will be covered in more detail in subsequent chapters.
* Source: Market Research Report, 2021

For implementation details, refer to https://github.com/example/ml-examples and the documentation at docs.example.com/ml.

SUMMARY

This chapter introduced the fundamental concepts of machine learning. In the next chapter, we'll dive deeper into supervised learning algorithms.

[1] Mitchell, T. (1997). Machine Learning. McGraw-Hill.
[2] Samuel, A. (1959). Some studies in machine learning using the game of checkers.
[3] Jordan, M.I., & Mitchell, T.M. (2015). Machine learning: Trends, perspectives, and prospects.
[4] LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature.
[5] Bishop, C.M. (2006). Pattern Recognition and Machine Learning.
[6] Sutton, R.S., & Barto, A.G. (2018). Reinforcement Learning: An Introduction.
"""

    print("\n📝 Testing with sample text containing common artifacts...")
    return compare_chapter(sample, "Sample Chapter with Artifacts")


if __name__ == "__main__":
    print("=" * 60)
    print("TEXT CLEANING COMPARISON TOOL")
    print("=" * 60)

    if not is_gemini_available():
        print("\n❌ ERROR: Gemini API not available. Set GEMINI_API_KEY.")
        sys.exit(1)

    print("\n✅ Gemini API available")

    # Check for EPUB argument
    if len(sys.argv) > 1:
        epub_path = sys.argv[1]
        max_chapters = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        test_with_epub(epub_path, max_chapters)
    else:
        # Run with sample text
        test_with_sample()
        print("\n" + "-" * 60)
        print("TIP: Pass an EPUB path to test with real content:")
        print("  python test_clean_diff.py /path/to/book.epub [max_chapters]")
