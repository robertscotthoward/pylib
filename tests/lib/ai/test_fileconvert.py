import os

from lib.ai.fileconvert import clean_text, is_nonsense_line, needs_conversion


GIBBERISH = "qwrtpz lkjhgf mnbvcx zxcvbn asdfgh"
PROSE = "The Apple II went on sale in 1977 and proved personal computers could sell."


def test_clean_text_repairs_mojibake():
    damaged = "IBMâ€™s Boca Raton team shipped the PC in twelve months."
    assert clean_text(damaged) == "IBM's Boca Raton team shipped the PC in twelve months."


def test_clean_text_drops_gibberish_lines():
    result = clean_text(f"{PROSE}\n{GIBBERISH}\n{PROSE}")
    assert GIBBERISH not in result
    assert result.count(PROSE) == 2


def test_clean_text_keeps_prose():
    text = f"{PROSE}\nSoftalk covered the Tecmar ALLINONE board in October 1982."
    assert clean_text(text) == text


def test_clean_text_keep_nonsense_only_fixes_encoding():
    text = f"{PROSE}\n{GIBBERISH}"
    assert clean_text(text, drop_nonsense=False) == text


def test_clean_text_is_idempotent():
    once = clean_text(f"{PROSE}\n{GIBBERISH}\n{PROSE}")
    assert clean_text(once) == once


def test_is_nonsense_line_ignores_short_lines():
    """Markdown syntax and headings sit below the detector's reliable range."""
    assert not is_nonsense_line("# 1977")
    assert not is_nonsense_line("- 1980")
    assert not is_nonsense_line("| a | b |")


def test_needs_conversion(tmp_path):
    source = tmp_path / "doc.docx"
    target = tmp_path / "doc.md"
    source.write_text("source")

    assert needs_conversion(str(source), str(target)), "missing target is stale"

    target.write_text("target")
    os.utime(target, (0, 0))
    os.utime(source, (100, 100))
    assert needs_conversion(str(source), str(target)), "older target is stale"

    os.utime(target, (100, 100))
    assert not needs_conversion(str(source), str(target)), "equal mtimes are up to date"

    os.utime(target, (200, 200))
    assert not needs_conversion(str(source), str(target)), "newer target is up to date"
