from __future__ import annotations

from postiz_uploader.captions import build_ass, group_lines, slice_words
from postiz_uploader.schema import CaptionStyle, Word


def _words(*items: tuple[str, float, float]) -> tuple[Word, ...]:
    return tuple(Word(*item) for item in items)


def test_slice_retimes_to_the_clip_and_drops_the_rest():
    words = _words(("before", 1.0, 1.4), ("in", 10.2, 10.5), ("edge", 19.9, 20.4), ("after", 25.0, 25.3))
    out = slice_words(words, 10.0, 20.0)
    assert [w.text for w in out] == ["in"]
    assert round(out[0].start, 3) == 0.2 and round(out[0].end, 3) == 0.5


def test_a_word_straddling_the_cut_is_clamped_not_negative():
    out = slice_words(_words(("hello", 9.8, 10.6)), 10.0, 20.0)
    assert out[0].start == 0.0 and round(out[0].end, 3) == 0.6


def test_lines_break_on_count_pause_and_sentence_end():
    style = CaptionStyle(max_words=3, max_chars=100)
    words = list(_words(("a", 0, 0.2), ("b", 0.2, 0.4), ("c", 0.4, 0.6), ("d", 0.6, 0.8)))
    assert [len(line.words) for line in group_lines(words, style, 10)] == [3, 1]

    paused = list(_words(("a", 0, 0.2), ("b", 2.0, 2.2)))
    assert [len(line.words) for line in group_lines(paused, style, 10)] == [1, 1]

    sentence = list(_words(("Done.", 0, 0.2), ("Next", 0.3, 0.5)))
    assert [len(line.words) for line in group_lines(sentence, style, 10)] == [1, 1]


def test_a_line_never_outlives_the_next_one_or_the_clip():
    style = CaptionStyle(max_words=1)
    lines = group_lines(list(_words(("a", 0, 0.5), ("b", 0.6, 1.0))), style, 1.05)
    assert lines[0].end == 0.6
    assert lines[1].end == 1.05


def test_ass_document_has_one_event_per_word_with_the_highlight_moving():
    doc, count = build_ass(
        _words(("Hello", 5.0, 5.4), ("world", 5.4, 5.9)), CaptionStyle(), start=5.0, end=8.0, width=1080, height=1920
    )
    assert count == 1
    events = [line for line in doc.splitlines() if line.startswith("Dialogue:")]
    assert len(events) == 2
    # #FFE600 in ASS byte order
    assert "{\\c&H00E6FF&}Hello{\\c&HFFFFFF&} world" in events[0]
    assert "Hello {\\c&H00E6FF&}world" in events[1]
    assert events[0].startswith("Dialogue: 0,0:00:00.00,0:00:00.40,")
    assert "PlayResX: 1080" in doc and "PlayResY: 1920" in doc


def test_no_highlight_means_one_event_per_line_and_uppercase_applies():
    style = CaptionStyle(highlight_color=None, uppercase=True)
    doc, _ = build_ass(_words(("hi", 0, 0.3), ("there", 0.3, 0.6)), style, start=0, end=5, width=1080, height=1920)
    events = [line for line in doc.splitlines() if line.startswith("Dialogue:")]
    assert len(events) == 1 and events[0].endswith(",HI THERE")


def test_override_syntax_in_the_transcript_cannot_inject_tags():
    doc, _ = build_ass(_words(("{\\fs200}big", 0, 0.5)), CaptionStyle(), start=0, end=5, width=1080, height=1920)
    event = [line for line in doc.splitlines() if line.startswith("Dialogue:")][0]
    assert "\\fs200" not in event and "(⧵fs200)big" in event


def test_nothing_in_the_window_means_nothing_to_burn():
    _, count = build_ass(_words(("late", 50, 51)), CaptionStyle(), start=0, end=5, width=1080, height=1920)
    assert count == 0
