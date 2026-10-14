"""Shared-timeline ROI start and end marks, one pair per trial."""

from __future__ import annotations

from pathlib import Path


_HEADER = "# participant\ttrial\tbound\ttime_s\tframe"
_BOUNDS = ("start", "end")


def load_marks(path: Path) -> dict[tuple[str, str], dict[str, dict]]:
    """Read tab-separated ROI marks. Later rows replace the same bound."""
    if not path.is_file():
        return {}
    marks: dict[tuple[str, str], dict[str, dict]] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split("\t")
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected participant, trial, bound, time_s, frame")
        participant, trial, bound, time_text, frame_text = (field.strip() for field in fields)
        if bound not in _BOUNDS:
            raise ValueError(f"{path}:{line_number}: bound must be start or end")
        if not trial:
            raise ValueError(f"{path}:{line_number}: trial is empty")
        try:
            time_s = float(time_text)
            frame = int(frame_text)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: time and frame must be numbers") from exc
        if time_s < 0 or frame < 0:
            raise ValueError(f"{path}:{line_number}: time and frame must be zero or positive")
        pair = marks.setdefault((participant, trial), {})
        pair[bound] = {"time_s": time_s, "frame": frame}
    return marks


def save_marks(path: Path, marks: dict[tuple[str, str], dict[str, dict]]) -> None:
    """Rewrite the marks file. Start is written before end for each trial."""
    lines = [_HEADER]
    for participant, trial in sorted(marks):
        pair = marks[(participant, trial)]
        for bound in _BOUNDS:
            mark = pair.get(bound)
            if not mark:
                continue
            lines.append(
                f"{participant}\t{trial}\t{bound}\t{float(mark['time_s']):.6f}\t{int(mark['frame'])}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)
