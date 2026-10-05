"""Application state and persisted camera offsets."""

from __future__ import annotations

import json
from pathlib import Path

from .discovery import discover_folder, discover_trial_list
from .pose import POSE_EDGES, load_pose


class ViewerState:
    def __init__(self, source: Path, *, trial_list: Path | None = None):
        self.warnings: list[str] = []
        self._list_order = trial_list is not None
        if trial_list is not None:
            self.source = Path.cwd().resolve()
            self.entries, self.warnings = discover_trial_list(trial_list)
        else:
            self.source = source.expanduser().resolve()
            self.entries = discover_folder(self.source)
        self.by_id = {entry["id"]: entry for entry in self.entries}
        self.offsets_path = self.source / "pose-video-offsets.json"
        self._pose_cache: dict[str, dict] = {}
        self._load_offsets()

    def _saved_rows(self) -> list[dict]:
        if not self.offsets_path.is_file():
            return []
        payload = json.loads(self.offsets_path.read_text())
        rows = payload.get("offsets", []) if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            raise ValueError(f"Invalid offsets file: {self.offsets_path}")
        return [row for row in rows if isinstance(row, dict) and row.get("video")]

    def _load_offsets(self) -> None:
        saved = {
            str(row["video"]): int(row.get("offset_frames", 0))
            for row in self._saved_rows()
        }
        for entry in self.entries:
            entry["hasSavedOffset"] = entry["videoPath"] in saved
            if entry["hasSavedOffset"]:
                entry["savedOffsetFrames"] = saved[entry["videoPath"]]

    def save_offsets(self, rows: object) -> dict:
        if not isinstance(rows, list) or not rows:
            raise ValueError("Offsets must be a non-empty list.")
        updates = {}
        for row in rows:
            entry_id = str(row.get("id", "")) if isinstance(row, dict) else ""
            if entry_id not in self.by_id:
                raise ValueError("Each offset must reference a known video id.")
            updates[entry_id] = int(row.get("offset_frames", 0))

        saved = {str(row["video"]): row for row in self._saved_rows()}
        for entry_id, offset in updates.items():
            entry = self.by_id[entry_id]
            entry["hasSavedOffset"] = True
            entry["savedOffsetFrames"] = offset
            saved[entry["videoPath"]] = {
                "trial": entry["trial"],
                "camera": entry["camera"],
                "video": entry["videoPath"],
                "offset_frames": offset,
            }

        payload = {
            "version": 1,
            "source": str(self.source),
            "offsets": [saved[key] for key in sorted(saved)],
        }
        temporary = self.offsets_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(self.offsets_path)
        return {"saved": len(updates), "path": str(self.offsets_path)}

    def trials(self) -> list[dict]:
        grouped: dict[str, list[dict]] = {}
        order: list[str] = []
        for entry in self.entries:
            trial = entry["trial"]
            if trial not in grouped:
                order.append(trial)
                grouped[trial] = []
            grouped[trial].append(entry)
        keys = order if self._list_order else sorted(grouped)
        return [
            {
                "id": trial,
                "trialType": grouped[trial][0]["trialType"],
                "cameraCount": len(grouped[trial]),
                "saveStatus": "saved" if all(entry["hasSavedOffset"] for entry in grouped[trial]) else "unsaved",
            }
            for trial in keys
        ]

    def trial_entries(self, trial: str) -> list[dict]:
        return sorted(
            (entry for entry in self.entries if entry["trial"] == trial),
            key=lambda entry: entry["camera"],
        )

    def entry(self, entry_id: str) -> dict:
        try:
            return self.by_id[entry_id]
        except KeyError as exc:
            raise KeyError(f"Unknown video id: {entry_id}") from exc

    def poses(self, entry_id: str) -> dict:
        if entry_id in self._pose_cache:
            return self._pose_cache[entry_id]
        entry = self.entry(entry_id)
        path = Path(entry["posePath"])
        if path.is_file():
            payload = {"available": True, "frames": load_pose(path), "edges": POSE_EDGES}
        else:
            payload = {"available": False, "message": f"Pose pickle not found: {path}", "frames": []}
        self._pose_cache[entry_id] = payload
        return payload
