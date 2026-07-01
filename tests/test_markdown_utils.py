"""Unit tests for markdown_utils.split_markdown_message (Discord Markdown-safe split).

Run directly (no pytest dependency needed):
    python tests/test_markdown_utils.py
or under pytest:
    python -m pytest tests/test_markdown_utils.py

Ports the Case0-Case8 verification strategy from Claude_Discordbot PR #25 (whose
standalone test file was not committed there). Each chunk is validated by feeding
it through the *same* fence state machine the splitter uses, to confirm it ends
*closed* (not merely an even count of ```). Also checks: non-fence content is
reconstructed in order, every chunk is <= max_length, language tags survive the
reopen, run-length edges are respected (``` inside a ```` block is not a close),
and the 1999/2000/2001 boundary produces no empty code block.
"""

import os
import sys

# Make the repo root importable whether run as a script (cwd/tests dir on path)
# or under pytest, so we exercise the REAL shipped markdown_utils.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from markdown_utils import (  # noqa: E402
    MAX_DISCORD_LENGTH,
    _cut_at,
    _fence_open,
    _is_fence_close,
    split_markdown_message,
)


# --- validation helpers -------------------------------------------------------

def _ends_closed(chunk):
    """True if `chunk` ends OUTSIDE any open fence (renders as balanced Markdown).

    Uses the splitter's own _fence_open / _is_fence_close, i.e. a real state
    machine rather than a naive "count of ``` is even" heuristic.
    """
    fence = None
    for line in chunk.split("\n"):
        if fence is None:
            opened = _fence_open(line)
            if opened:
                fence = opened
        elif _is_fence_close(line, fence[0]):
            fence = None
    return fence is None


def _content_lines(text):
    """Lines that are NOT structural fence markers, per the splitter's state
    machine. Original and injected (reopen / temp-close) markers are dropped;
    content — including backtick-looking lines inside a wider fence — is kept.
    """
    out = []
    fence = None
    for line in text.split("\n"):
        if fence is None:
            opened = _fence_open(line)
            if opened:
                fence = opened
                continue
            out.append(line)
        else:
            if _is_fence_close(line, fence[0]):
                fence = None
                continue
            out.append(line)
    return out


def _content_chars(text):
    """Content characters with structural fence markers and line breaks removed,
    so reconstruction survives hard-split lines (a mid-line cut just yields two
    adjacent pieces that concatenate back).
    """
    return "".join(_content_lines(text))


def _assert_valid(text, chunks, max_length, balanced=True):
    for c in chunks:
        assert c.strip(), "empty/whitespace-only chunk emitted"
        assert len(c) <= max_length, f"chunk len {len(c)} > max_length {max_length}"
        assert _content_chars(c).strip() != "", "empty code-block (marker-only) chunk emitted"
    if balanced:
        for c in chunks:
            assert _ends_closed(c), "chunk does not end with a balanced fence"
    else:
        for c in chunks[:-1]:
            assert _ends_closed(c), "non-final chunk not temp-closed at split"
    assert _content_chars(text) == _content_chars("\n".join(chunks)), \
        "content not reconstructed (order/characters lost)"


def _bare_fenced(total_len):
    """A ```-fenced block of exactly `total_len` chars: ```\\n<body>\\n``` ."""
    body_len = total_len - 8  # len("```\n") + len("\n```")
    assert body_len >= 0
    return "```\n" + ("x" * body_len) + "\n```"


# --- Case 0: helper detection -------------------------------------------------

def test_case0_helpers():
    assert _fence_open("```python") == (3, "```python")
    assert _fence_open("   ```") == (3, "   ```")       # leading whitespace allowed
    assert _fence_open("````") == (4, "````")
    assert _fence_open("no fence") is None
    assert _fence_open("``two") is None                 # only 2 backticks

    assert _is_fence_close("```", 3) is True
    assert _is_fence_close("````", 3) is True           # longer run also closes
    assert _is_fence_close("```", 4) is False           # shorter run does NOT close
    assert _is_fence_close("  ```  ", 3) is True         # surrounding whitespace ok
    assert _is_fence_close("``` python", 3) is False     # info text => not a close
    assert _is_fence_close("", 3) is False

    assert _cut_at("hello world", 100) == len("hello world")  # fits, no cut
    assert _cut_at("hello world", 8) == 5                     # cut at the space
    assert _cut_at("helloworld", 5) == 5                      # no space => hard cut


# --- Case 1: fenced code block split, language tag + indentation preserved ----

def test_case1_code_block():
    body = "\n".join(f"    code_line_{i}()" for i in range(60))  # 4-space indent
    text = "```python\n" + body + "\n```"
    max_length = 100
    chunks = split_markdown_message(text, max_length)
    assert len(chunks) > 1
    _assert_valid(text, chunks, max_length)
    for c in chunks:
        assert c.split("\n")[0] == "```python", "language tag not preserved on reopen"
        assert c.split("\n")[-1] == "```", "chunk not fence-closed"
    assert any("    code_line_" in c for c in chunks), "indentation was stripped"


# --- Case 2: inline formatting (no fences) across a split --------------------

def test_case2_inline_formatting():
    lines = [
        f"Paragraph {i}: here is `inline_code_{i}` and **bold_{i}** with _em{i}_ text."
        for i in range(30)
    ]
    text = "\n".join(lines)
    max_length = 100
    chunks = split_markdown_message(text, max_length)
    assert len(chunks) > 1
    _assert_valid(text, chunks, max_length)
    for c in chunks:                       # single backticks never open a fence
        for line in c.split("\n"):
            assert _fence_open(line) is None


# --- Case 3: hard split of one very long, space-less token -------------------

def test_case3_hard_split():
    text = "A" * 350  # one line, no spaces, no fences
    max_length = 100
    chunks = split_markdown_message(text, max_length)
    assert len(chunks) == 4  # 100 + 100 + 100 + 50
    _assert_valid(text, chunks, max_length)
    assert "".join(chunks) == text  # exact reconstruction, nothing added/lost


# --- Case 4: quad-fence (````), inner ``` is content, not a close ------------

def test_case4_quad_fence_inner_triple():
    inner = "\n".join(f"``` still-open {i}" for i in range(20))
    text = "````\n" + inner + "\n````"
    max_length = 40
    chunks = split_markdown_message(text, max_length)
    assert len(chunks) > 1
    _assert_valid(text, chunks, max_length)
    for c in chunks:
        assert c.split("\n")[0] == "````"   # reopened with the 4-backtick run
        assert c.split("\n")[-1] == "````"
    assert any("still-open" in c for c in chunks)  # inner ``` survived as content


# --- Case 5: short / empty / max_length <= 0 --------------------------------

def test_case5_short_and_empty():
    assert split_markdown_message("") == []
    assert split_markdown_message("   ") == []
    assert split_markdown_message("\n\n") == []
    assert split_markdown_message("hello") == ["hello"]
    assert split_markdown_message("hello world", 2000) == ["hello world"]
    assert split_markdown_message("x", 0) == ["x"]     # degenerate branch
    assert split_markdown_message("   ", 0) == []


# --- Case 6: realistic mixed prose + code around the 2000 boundary -----------

def test_case6_mixed_default_limit():
    prose1 = "\n".join(
        f"Intro line {i} with some explanatory text about the topic." for i in range(20)
    )
    code = (
        "```python\n"
        + "\n".join(f"    value_{i} = compute({i})" for i in range(40))
        + "\n```"
    )
    prose2 = "\n".join(
        f"Conclusion line {i} summarizing the measured results here." for i in range(20)
    )
    text = prose1 + "\n\n" + code + "\n\n" + prose2
    assert len(text) > MAX_DISCORD_LENGTH
    chunks = split_markdown_message(text, MAX_DISCORD_LENGTH)
    assert len(chunks) >= 2
    _assert_valid(text, chunks, MAX_DISCORD_LENGTH)


# --- Case 7: block that fits (1999/2000) stays a single unchanged chunk ------

def test_case7_boundary_single_chunk():
    for total in (1999, 2000):
        text = _bare_fenced(total)
        assert len(text) == total
        chunks = split_markdown_message(text, MAX_DISCORD_LENGTH)
        assert chunks == [text], f"len={total} should be one unchanged chunk"


# --- Case 8: block one char over (2001) => two chunks, no empty code block ---

def test_case8_boundary_two_chunks():
    text = _bare_fenced(2001)
    assert len(text) == 2001
    chunks = split_markdown_message(text, MAX_DISCORD_LENGTH)
    assert len(chunks) == 2, "2001-char block should split into exactly two chunks"
    _assert_valid(text, chunks, MAX_DISCORD_LENGTH)


# --- Case 9: unclosed fence mirrored faithfully; splits inject a temp close --

def test_case9_unclosed_fence_mirrored():
    # Fits in one chunk => returned unchanged, still unclosed (faithful mirror).
    text = "```python\nprint(1)\nprint(2)"
    chunks = split_markdown_message(text, MAX_DISCORD_LENGTH)
    assert chunks == [text]
    assert not _ends_closed(chunks[-1])

    # Forced to split => intermediate chunks temp-closed, last stays unclosed.
    long_unclosed = "```python\n" + "\n".join(f"row_{i} = {i}" for i in range(80))
    chunks = split_markdown_message(long_unclosed, 40)
    assert len(chunks) > 1
    _assert_valid(long_unclosed, chunks, 40, balanced=False)
    assert not _ends_closed(chunks[-1]), "final unclosed fence should be mirrored open"


_CASES = [
    ("Case0 helper detection", test_case0_helpers),
    ("Case1 code block", test_case1_code_block),
    ("Case2 inline formatting", test_case2_inline_formatting),
    ("Case3 hard split", test_case3_hard_split),
    ("Case4 quad-fence w/ inner ```", test_case4_quad_fence_inner_triple),
    ("Case5 short/empty", test_case5_short_and_empty),
    ("Case6 mixed @2000", test_case6_mixed_default_limit),
    ("Case7 fenced block at 1999/2000", test_case7_boundary_single_chunk),
    ("Case8 fenced block at 2001", test_case8_boundary_two_chunks),
    ("Case9 unclosed fence mirrored", test_case9_unclosed_fence_mirrored),
]


def _run_all():
    for name, fn in _CASES:
        fn()
        print(f"{name}: OK")
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    _run_all()
