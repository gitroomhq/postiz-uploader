"""Word timings -> an ASS subtitle file for one clip. Pure: no I/O, no ffmpeg.

Words arrive on the source timeline. They are sliced to the clip window, shifted so
the clip starts at zero, grouped into short lines, and each line is emitted once per
word with that word highlighted, which is the usual short-form caption look.
"""

from __future__ import annotations

from dataclasses import dataclass

from postiz_uploader.schema import CaptionStyle, Word

# a pause this long always starts a new line, so captions never hang over silence
MAX_GAP_SECONDS = 0.8
# how long a line stays up after its last word, unless the next line starts sooner
HOLD_SECONDS = 0.25
SENTENCE_END = (".", "?", "!", "…")
_ALIGNMENT = {"bottom": 2, "middle": 5, "top": 8}


@dataclass(frozen=True)
class Line:
    words: tuple[Word, ...]  # times relative to the clip
    start: float
    end: float


def slice_words(words: tuple[Word, ...], start: float, end: float) -> list[Word]:
    """Words whose midpoint falls inside [start, end], re-timed to the clip and clamped."""
    duration = end - start
    out = []
    for word in words:
        text = word.text.strip()
        if not text or word.end < word.start:
            continue
        if not start <= (word.start + word.end) / 2 <= end:
            continue
        out.append(Word(text, max(word.start - start, 0.0), min(word.end - start, duration)))
    out.sort(key=lambda w: w.start)
    return out


def group_lines(words: list[Word], style: CaptionStyle, duration: float) -> list[Line]:
    groups: list[list[Word]] = []
    current: list[Word] = []
    for word in words:
        if current:
            chars = sum(len(w.text) for w in current) + len(current) + len(word.text)
            previous = current[-1]
            if (
                len(current) >= style.max_words
                or chars > style.max_chars
                or word.start - previous.end > MAX_GAP_SECONDS
                or previous.text.endswith(SENTENCE_END)
            ):
                groups.append(current)
                current = []
        current.append(word)
    if current:
        groups.append(current)

    lines = []
    for i, group in enumerate(groups):
        end = group[-1].end + HOLD_SECONDS
        if i + 1 < len(groups):
            end = min(end, groups[i + 1][0].start)
        end = min(max(end, group[-1].end), duration)
        lines.append(Line(tuple(group), group[0].start, end))
    return lines


def _colour(hex_rgb: str) -> str:
    """#RRGGBB -> ASS &HBBGGRR&."""
    r, g, b = hex_rgb[1:3], hex_rgb[3:5], hex_rgb[5:7]
    return f"&H{b}{g}{r}&".upper()


def _escape(text: str) -> str:
    # braces open override blocks and a backslash starts a tag; neither has an escape
    # in ASS, so they are swapped for lookalikes rather than dropped
    return text.replace("\\", "⧵").replace("{", "(").replace("}", ")").replace("\n", " ").replace("\r", " ")


def _timestamp(seconds: float) -> str:
    cs = max(int(round(seconds * 100)), 0)
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _text(line: Line, active: int | None, style: CaptionStyle) -> str:
    parts = []
    for i, word in enumerate(line.words):
        text = _escape(word.text.upper() if style.uppercase else word.text)
        if i == active and style.highlight_color:
            text = f"{{\\c{_colour(style.highlight_color)}}}{text}{{\\c{_colour(style.color)}}}"
        parts.append(text)
    return " ".join(parts)


def build_ass(words: tuple[Word, ...], style: CaptionStyle, *, start: float, end: float, width: int, height: int):
    """Returns (ass document, number of caption lines). Zero lines means nothing to burn."""
    duration = end - start
    lines = group_lines(slice_words(words, start, end), style, duration)

    font_size = style.font_size or max(round(height / 24), 8)
    margin_v = style.margin_v if style.margin_v is not None else round(height / 5)
    if style.position == "middle":
        margin_v = 0
    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
        f"Style: Default,{style.font},{font_size},{_colour(style.color)},{_colour(style.color)},"
        f"{_colour(style.outline_color)},&H80000000&,{-1 if style.bold else 0},0,0,0,100,100,0,0,1,"
        f"{style.outline:g},{style.shadow:g},{_ALIGNMENT[style.position]},{round(width * 0.06)},"
        f"{round(width * 0.06)},{margin_v},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    events = []
    for line in lines:
        if not style.highlight_color:
            spans = [(line.start, line.end, None)]
        else:
            spans = []
            for i, word in enumerate(line.words):
                until = line.words[i + 1].start if i + 1 < len(line.words) else line.end
                spans.append((line.start if i == 0 else word.start, until, i))
        for span_start, span_end, active in spans:
            if span_end <= span_start:
                continue
            events.append(
                f"Dialogue: 0,{_timestamp(span_start)},{_timestamp(span_end)},Default,,0,0,0,,"
                f"{_text(line, active, style)}"
            )

    return "\n".join(header + events) + "\n", len(lines)
