"""Markdown-safe splitting of long messages for Discord's 2000-char limit.

Pure, side-effect-free helpers so they can be unit-tested without importing the
bot (which starts Discord / constructs Vertex clients at import time).

Ported verbatim from the sibling project Claude_Discordbot, PR #25
(https://github.com/yasuhiroinoue/Claude_Discordbot/pull/25), final head SHA
3ec8df88668d3ba4496d26699c5f5ec7b0fd42ac. The splitting logic is
provider-independent: Gemini output is also GitHub-flavored Markdown.
"""

import re

# Discord's per-message character cap. Bot callers pass MAX_DISCORD_LENGTH
# explicitly; this default only applies to direct/test use.
MAX_DISCORD_LENGTH = 2000


_FENCE_RE = re.compile(r'^\s*(`{3,})')


def _fence_open(line):
    """Return (run_len, marker_line) if `line` opens a code fence, else None.

    An opener is a line whose first non-whitespace content is a run of 3+
    backticks (optionally followed by a language/info string).
    """
    m = _FENCE_RE.match(line)
    if not m:
        return None
    return len(m.group(1)), line


def _is_fence_close(line, run_len):
    """True if `line` closes a fence opened with `run_len` backticks.

    A close is a line of only backticks (>= run_len), with optional surrounding
    whitespace and no language/info text. This is intentionally NOT a plain
    toggle: a shorter run (e.g. ``` inside a ```` block) does not close it.
    """
    s = line.strip()
    return len(s) >= run_len and bool(s) and all(c == '`' for c in s)


def _cut_at(line, limit):
    """Largest cut index in [1, limit] for splitting `line`, preferring a space
    boundary so words/tokens aren't broken; falls back to a hard cut at `limit`.
    """
    if len(line) <= limit:
        return len(line)
    sp = line.rfind(' ', 1, limit)
    return sp if sp != -1 else limit


def split_markdown_message(text, max_length=MAX_DISCORD_LENGTH):
    """Split `text` into <= `max_length`-char chunks without breaking fenced
    code blocks.

    When a split lands inside a ``` fence, the current chunk is closed with a
    matching fence and the next chunk reopens it with the same info line (e.g.
    ```python), so every message renders correctly on its own. Chunk interiors
    are preserved verbatim (no stripping, so code indentation survives), and
    whitespace-only chunks are dropped so Discord never receives an empty
    message.
    """
    if max_length <= 0:
        return [text] if text.strip() else []

    lines = text.split('\n')
    chunks = []
    cur = []              # buffered lines for the current chunk
    fence = None          # (run_len, marker_line) while inside an open fence
    has_content = False   # cur holds real content beyond a reopened fence marker

    def buffered_len():
        return sum(len(l) for l in cur) + max(0, len(cur) - 1)

    def room(close_reserve):
        # Chars available for one more line, keeping `close_reserve` chars free
        # (e.g. to inject a closing fence when a split lands inside a code block).
        sep = 1 if cur else 0
        return max_length - buffered_len() - sep - close_reserve

    def flush():
        nonlocal cur, has_content
        pieces = list(cur)
        if fence is not None:
            pieces.append('`' * fence[0])          # temporary close at the split
        chunk = '\n'.join(pieces)
        if chunk.strip():
            chunks.append(chunk)
        cur = [fence[1]] if fence is not None else []   # reopen on the next chunk
        has_content = False

    def add_line(line, close_reserve, is_marker=False):
        # `is_marker` lines (fence open/close) don't count as real content, so a
        # chunk holding only a fence marker is never flushed on its own (which
        # would emit an empty code block).
        nonlocal has_content
        if len(line) <= room(close_reserve):
            cur.append(line)
            has_content = has_content or not is_marker
            return
        if has_content:
            flush()                                 # start a fresh chunk first
        while len(line) > room(close_reserve):
            r = room(close_reserve)
            if r <= 0:                              # degenerate: markers exhaust budget
                break
            cut = _cut_at(line, r)
            cur.append(line[:cut])
            has_content = True                      # a split piece is real content
            flush()
            line = line[cut:]
        cur.append(line)
        has_content = has_content or not is_marker

    for line in lines:
        opened_now = _fence_open(line) if fence is None else None
        closing_now = fence is not None and _is_fence_close(line, fence[0])
        if opened_now:
            close_reserve = 1 + opened_now[0]       # reserve for its eventual close
            is_marker = True
        elif closing_now:
            close_reserve = 0                       # this line IS the real close
            is_marker = True
        elif fence is not None:
            close_reserve = 1 + fence[0]            # room to inject a close if we split
            is_marker = False
        else:
            close_reserve = 0
            is_marker = False
        add_line(line, close_reserve, is_marker)
        if opened_now:
            fence = opened_now
        elif closing_now:
            fence = None

    # Final chunk: emit only if it carries real content. A trailing chunk of
    # injected markers only (reopen marker +/- real close) would render as an
    # empty code block, so it is dropped — the content is already in prior chunks.
    # No closing fence is injected here: an original that ends inside an unclosed
    # fence is mirrored faithfully.
    tail = '\n'.join(cur)
    if has_content and tail.strip():
        chunks.append(tail)
    return chunks
