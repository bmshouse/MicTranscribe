from __future__ import annotations

from datetime import date, datetime

from mictranscribe.writer import TranscriptWriter


def test_rotates_to_new_file_on_date_change(tmp_path):
    dates = iter([date(2026, 1, 1), date(2026, 1, 1), date(2026, 1, 2)])
    writer = TranscriptWriter(tmp_path, today=lambda: next(dates))

    writer.write_line(datetime(2026, 1, 1, 10, 0, 0), datetime(2026, 1, 1, 10, 0, 5), "hello")
    writer.write_line(datetime(2026, 1, 1, 10, 1, 0), datetime(2026, 1, 1, 10, 1, 5), "world")
    writer.write_line(datetime(2026, 1, 2, 0, 0, 1), datetime(2026, 1, 2, 0, 0, 5), "new day")
    writer.close()

    day1 = tmp_path / "2026-01-01.txt"
    day2 = tmp_path / "2026-01-02.txt"
    assert day1.exists()
    assert day2.exists()

    content1 = day1.read_text()
    assert "[10:00:00 - 10:00:05] hello" in content1
    assert "[10:01:00 - 10:01:05] world" in content1
    assert "new day" not in content1

    content2 = day2.read_text()
    assert "[00:00:01 - 00:00:05] new day" in content2


def test_close_leaves_no_open_handle(tmp_path):
    writer = TranscriptWriter(tmp_path, today=lambda: date(2026, 1, 1))
    writer.write_line(datetime(2026, 1, 1, 9, 0, 0), datetime(2026, 1, 1, 9, 0, 1), "hi")
    assert writer._fh is not None
    writer.close()
    assert writer._fh is None
