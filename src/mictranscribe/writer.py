from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Callable, TextIO


class TranscriptWriter:
    """Appends timestamped lines to a transcript file, rotating to a new
    `YYYY-MM-DD.txt` file whenever the calendar date changes. Rotation is
    checked lazily on each write, since utterances are infrequent and
    event-driven rather than on a fixed timer.
    """

    def __init__(self, output_dir: Path, today: Callable[[], date] = date.today) -> None:
        self.output_dir = output_dir
        self._today = today
        self._current_date: date | None = None
        self._fh: TextIO | None = None

    def write_line(self, start: datetime, end: datetime, text: str) -> None:
        self._rotate_if_needed()
        line = f"[{start:%H:%M:%S} - {end:%H:%M:%S}] {text}\n"
        self._fh.write(line)
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def _rotate_if_needed(self) -> None:
        today = self._today()
        if today != self._current_date:
            self.close()
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._fh = open(self._path_for(today), "a", encoding="utf-8")
            self._current_date = today

    def _path_for(self, day: date) -> Path:
        return self.output_dir / f"{day.isoformat()}.txt"
