"""Discover videos and pose pickles in OpenCap download folders."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import threading
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


VIDEO_EXTENSIONS = {".avi", ".m4v", ".mov", ".mp4"}
_INDEX_CACHE: Path | None = None
_INDEX_LOCK = threading.Lock()


def configure_index_cache(path: Path | None) -> None:
    """Save frame timestamps and sync offsets under *path* for later launches."""
    global _INDEX_CACHE
    if path is None:
        _INDEX_CACHE = None
        return
    cache = Path(path).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    _INDEX_CACHE = cache


def _file_identity(path: Path) -> tuple[str, int, int]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return str(resolved), int(stat.st_size), int(stat.st_mtime_ns)


def _cache_file(parts: tuple[object, ...]) -> Path | None:
    if _INDEX_CACHE is None:
        return None
    digest = hashlib.sha256("\n".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return _INDEX_CACHE / f"{digest}.json"


def _read_cache(parts: tuple[object, ...], expected: dict[str, object]) -> dict | None:
    dest = _cache_file(parts)
    if dest is None or not dest.is_file():
        return None
    try:
        payload = json.loads(dest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    for key, value in expected.items():
        if payload.get(key) != value:
            return None
    return payload


def _write_cache(parts: tuple[object, ...], payload: dict) -> None:
    dest = _cache_file(parts)
    if dest is None:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_name(dest.name + ".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(dest)


@lru_cache(maxsize=256)
def _probe_frame_timestamps(path_value: str, file_size: int, modified_ns: int) -> tuple[float, ...]:
    """Return normalized presentation timestamps; size and mtime invalidate the cache."""
    del file_size, modified_ns
    executable = shutil.which("ffprobe")
    if executable is None:
        raise RuntimeError("ffprobe is required. Update the Conda environment to install FFmpeg.")
    command = [
        executable,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_frames",
        "-show_entries", "frame=best_effort_timestamp_time",
        "-of", "json",
        path_value,
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or "unknown FFprobe error"
        raise ValueError(f"Could not index video frames: {path_value} ({message})") from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"Timed out while indexing video frames: {path_value}") from exc
    payload = json.loads(completed.stdout)
    timestamps = [
        float(frame["best_effort_timestamp_time"])
        for frame in payload.get("frames", [])
        if frame.get("best_effort_timestamp_time") not in (None, "N/A")
    ]
    if not timestamps:
        raise ValueError(f"FFprobe returned no video frame timestamps: {path_value}")
    first = timestamps[0]
    normalized = tuple(round(timestamp - first, 9) for timestamp in timestamps)
    if any(current < previous for previous, current in zip(normalized, normalized[1:])):
        raise ValueError(f"Video frame timestamps are not ordered: {path_value}")
    return normalized


def frame_timestamps(path: Path) -> list[float]:
    stat = path.stat()
    return list(_probe_frame_timestamps(str(path.resolve()), stat.st_size, stat.st_mtime_ns))


def _video_parts(identity: tuple[str, int, int]) -> tuple[object, ...]:
    return ("video", *identity)


def _video_expected(identity: tuple[str, int, int]) -> dict[str, object]:
    path, size, mtime_ns = identity
    return {"path": path, "size": size, "mtime_ns": mtime_ns}


def cached_video_metadata(path: Path) -> tuple[float, int, int, int]:
    """fps, width, height, and frame count, reused when the file is unchanged."""
    identity = _file_identity(path)
    with _INDEX_LOCK:
        cached = _read_cache(_video_parts(identity), _video_expected(identity))
    if cached and all(key in cached for key in ("fps", "width", "height", "frameCount")):
        return float(cached["fps"]), int(cached["width"]), int(cached["height"]), int(cached["frameCount"])
    fps, width, height, frame_count = video_metadata(path)
    with _INDEX_LOCK:
        current = _read_cache(_video_parts(identity), _video_expected(identity)) or {}
        current.update(_video_expected(identity))
        current.update({"fps": fps, "width": width, "height": height, "frameCount": frame_count})
        _write_cache(_video_parts(identity), current)
    return fps, width, height, frame_count


def cached_frame_timestamps(path: Path) -> list[float]:
    """Presentation timestamps, reused when the file is unchanged."""
    identity = _file_identity(path)
    with _INDEX_LOCK:
        cached = _read_cache(_video_parts(identity), _video_expected(identity))
    times = cached.get("frameTimes") if cached else None
    if isinstance(times, list) and times:
        return [float(value) for value in times]
    print(f"Indexing frames: {path.name}", flush=True)
    timestamps = frame_timestamps(path)
    with _INDEX_LOCK:
        current = _read_cache(_video_parts(identity), _video_expected(identity)) or {}
        current.update(_video_expected(identity))
        current["frameTimes"] = timestamps
        _write_cache(_video_parts(identity), current)
    return timestamps


def cached_sync_offset(original: Path, sync: Path) -> int:
    """Frame offset from the original video to the sync video, reused when both are unchanged."""
    original_id = _file_identity(original)
    sync_id = _file_identity(sync)
    parts = ("sync-offset", *original_id, *sync_id)
    expected = {
        "originalPath": original_id[0],
        "originalSize": original_id[1],
        "originalMtimeNs": original_id[2],
        "syncPath": sync_id[0],
        "syncSize": sync_id[1],
        "syncMtimeNs": sync_id[2],
    }
    with _INDEX_LOCK:
        cached = _read_cache(parts, expected)
    if cached is not None and "poseFrameOffset" in cached:
        return int(cached["poseFrameOffset"])
    print(f"Indexing sync offset: {sync.name}", flush=True)
    offset = infer_sync_frame_offset(original, sync)
    payload = dict(expected)
    payload["poseFrameOffset"] = int(offset)
    with _INDEX_LOCK:
        _write_cache(parts, payload)
    return offset


def video_metadata(path: Path) -> tuple[float, int, int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Video does not exist: {path}")
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if fps <= 0 or width <= 0 or height <= 0 or frame_count <= 0:
        raise ValueError(f"Could not read video metadata: {path}")
    return fps, width, height, frame_count


def _pose_details(path: Path) -> tuple[str, str, bool] | None:
    """Return trial, original video stem, and whether the pickle is processed."""
    if path.name.endswith("_keypoints.pkl"):
        trial = path.name.removesuffix("_keypoints.pkl")
        return trial, trial, True
    if path.name.endswith("_rotated_pp.pkl"):
        return path.parent.name, path.name.removesuffix("_rotated_pp.pkl"), True
    if path.name.endswith("_rotated.pkl"):
        return path.parent.name, path.name.removesuffix("_rotated.pkl"), False
    return None


def _is_neutral(trial: str) -> bool:
    normalized = trial.strip().casefold()
    return normalized == "neutral" or normalized.startswith(("neutral_", "neutral-", "neutral "))


def _video_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.casefold() in VIDEO_EXTENSIONS
    )


def _thumbnail(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (48, 64), interpolation=cv2.INTER_AREA)


def infer_sync_frame_offset(original_path: Path, sync_path: Path) -> int:
    """Find which original frame became frame zero of a synchronized video."""
    _, _, _, original_count = video_metadata(original_path)
    _, _, _, sync_count = video_metadata(sync_path)
    if original_count <= sync_count:
        return 0

    original_capture = cv2.VideoCapture(str(original_path))
    original_frames = []
    try:
        while True:
            ok, frame = original_capture.read()
            if not ok:
                break
            original_frames.append(_thumbnail(frame))
    finally:
        original_capture.release()
    if len(original_frames) < sync_count:
        return 0

    max_offset = len(original_frames) - sync_count
    originals = np.asarray(original_frames, dtype=np.float32)
    sample_indices = sorted({0, sync_count // 4, sync_count // 2, 3 * sync_count // 4, sync_count - 1})
    wanted_samples = set(sample_indices)
    sync_samples = {}
    sync_capture = cv2.VideoCapture(str(sync_path))
    try:
        frame_index = 0
        while frame_index <= sample_indices[-1]:
            ok, frame = sync_capture.read()
            if not ok:
                break
            if frame_index in wanted_samples:
                sync_samples[frame_index] = _thumbnail(frame).astype(np.float32)
            frame_index += 1
    finally:
        sync_capture.release()

    scores = np.zeros(max_offset + 1, dtype=float)
    for frame_index, target in sync_samples.items():
        candidates = originals[frame_index:frame_index + max_offset + 1]
        scores += np.mean((candidates - target) ** 2, axis=(1, 2))
    return int(np.argmin(scores)) if sync_samples else 0


def _pose_type(path: Path) -> str:
    return "HRNet" if any("mmpose" in part.casefold() for part in path.parts) else "OpenPose"


def discover_session(path: Path, *, only_trials: set[str] | None = None) -> list[dict]:
    """Discover trials in one OpenCap session.

    *only_trials* skips indexing videos whose trial folder is not in the set.
    """
    session_path = path.expanduser().resolve()
    videos_dir = session_path / "Videos"
    if not videos_dir.is_dir():
        raise FileNotFoundError(f"OpenCap session has no Videos directory: {session_path}")

    candidates: dict[tuple[str, str], tuple[int, Path, str]] = {}
    for camera_dir in sorted(videos_dir.glob("Cam*")):
        if not camera_dir.is_dir():
            continue
        for pose_path in sorted(camera_dir.glob("OutputPkl*/**/*.pkl")):
            details = _pose_details(pose_path)
            if details is None:
                continue
            trial, original_stem, processed = details
            key = (camera_dir.name, trial)
            priority = 2 if processed else 1
            if key not in candidates or priority > candidates[key][0]:
                candidates[key] = (priority, pose_path, original_stem)

    rows = []
    for (camera, trial), (_, pose_path, original_stem) in sorted(candidates.items()):
        if only_trials is not None and trial not in only_trials:
            continue
        input_dir = videos_dir / camera / "InputMedia" / trial
        if not input_dir.is_dir():
            continue
        videos = _video_files(input_dir)
        originals = [
            video for video in videos
            if "sync" not in video.stem.casefold() and "rotated" not in video.stem.casefold()
        ]
        preferred = [video for video in originals if video.stem in {trial, original_stem}]
        original = (preferred or originals or [None])[0]
        sync = next((video for video in videos if "sync" in video.stem.casefold()), None)
        video = sync or original
        if video is None:
            continue

        fps, width, height, _ = cached_video_metadata(video)
        timestamps = cached_frame_timestamps(video)
        frame_count = len(timestamps)
        pose_width, pose_height = width, height
        pose_frame_offset = 0
        if original is not None:
            _, pose_width, pose_height, _ = cached_video_metadata(original)
            if sync is not None:
                pose_frame_offset = cached_sync_offset(original, sync)
        rows.append({
            "trial": trial,
            "trialType": "neutral" if _is_neutral(trial) else "dynamic",
            "camera": camera,
            "videoPath": str(video.resolve()),
            "posePath": str(pose_path.resolve()),
            "poseType": _pose_type(pose_path),
            "poseFrameOffset": pose_frame_offset,
            "poseWidth": pose_width,
            "poseHeight": pose_height,
            "fps": fps,
            "frameTimes": timestamps,
            "frameTiming": "verified-pts",
            "width": width,
            "height": height,
            "numFrames": frame_count,
            "frameRange": [0, frame_count - 1],
            "durationSeconds": timestamps[-1],
        })

    if not rows:
        raise ValueError(
            "No OpenCap trials found. Expected Videos/Cam*/InputMedia/<trial> "
            "videos with matching OutputPkl* pose files."
        )
    for index, entry in enumerate(rows):
        entry["id"] = str(index)
    return rows


def discover_folder(path: Path) -> list[dict]:
    """Load one OpenCap session or recursively discover a collection."""
    source_path = path.expanduser().resolve()
    if (source_path / "Videos").is_dir():
        return discover_session(source_path)
    if not source_path.is_dir():
        raise FileNotFoundError(f"OpenCap folder does not exist: {source_path}")

    session_paths = sorted({videos_dir.parent for videos_dir in source_path.rglob("Videos")})
    entries = []
    errors = []
    for session_path in session_paths:
        try:
            session_entries = discover_session(session_path)
        except (FileNotFoundError, ValueError) as exc:
            errors.append(f"{session_path.name}: {exc}")
            continue
        for entry in session_entries:
            entry["trial"] = f"{session_path.name} / {entry['trial']}"
            entry["id"] = str(len(entries))
            entries.append(entry)
    if not entries:
        detail = f" ({'; '.join(errors)})" if errors else ""
        raise ValueError(f"No usable OpenCap sessions found below: {source_path}{detail}")
    return entries


def parse_trial_manifest(path: Path) -> list[tuple[str, str, str]]:
    """Read ``session_path``, ``trial``, ``label`` lines from a text manifest.

    Fields are separated by a tab, or by a run of two or more spaces. A single
    space inside a path or label is kept. Blank lines and ``#`` comments are
    ignored.
    """
    manifest = path.expanduser()
    if not manifest.is_file():
        raise FileNotFoundError(f"Trial list does not exist: {manifest}")
    rows: list[tuple[str, str, str]] = []
    for line_number, raw in enumerate(manifest.read_text(encoding="utf-8-sig").splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "\t" in stripped:
            fields = [field.strip() for field in stripped.split("\t")]
        else:
            fields = [field.strip() for field in re.split(r" {2,}", stripped)]
            fields = [field for field in fields if field]
        if len(fields) != 3 or not all(fields):
            raise ValueError(
                f"{manifest}:{line_number}: expected session_path, trial, and label"
            )
        rows.append((fields[0], fields[1], fields[2]))
    if not rows:
        raise ValueError(f"Trial list is empty: {manifest}")
    return rows


def participant_from_label(label: str, trial: str) -> str:
    """Return the participant when ``label`` is ``participant`` plus ``trial``.

    The trial name is the trailing field. A label that does not end with that
    trial name has no participant.
    """
    suffix = f" {trial}"
    if trial and label.endswith(suffix):
        return label[: -len(suffix)]
    return ""


def discover_trial_list(path: Path) -> tuple[list[dict], list[str]]:
    """Discover only the session/trial pairs in a manifest, in file order.

    Missing sessions or trials are warnings. Videos that are not listed are not
    indexed.
    """
    requested = parse_trial_manifest(path)
    warnings: list[str] = []
    by_session: dict[Path, set[str]] = {}
    for session_text, trial, _label in requested:
        session = Path(session_text).expanduser()
        by_session.setdefault(session, set()).add(trial)

    discovered: dict[Path, list[dict]] = {}
    for session, trials in by_session.items():
        if not session.is_dir():
            warnings.append(f"session not found: {session}")
            discovered[session] = []
            continue
        try:
            discovered[session] = discover_session(session, only_trials=trials)
        except (FileNotFoundError, ValueError) as exc:
            warnings.append(f"{session}: {exc}")
            discovered[session] = []

    entries: list[dict] = []
    for session_text, trial, label in requested:
        session = Path(session_text).expanduser()
        matches = [entry for entry in discovered.get(session, []) if entry["trial"] == trial]
        if not matches:
            warnings.append(f"trial not found: {trial} in {session}")
            continue
        for entry in sorted(matches, key=lambda item: item["camera"]):
            labeled = dict(entry)
            labeled["sourceTrial"] = trial
            labeled["participant"] = participant_from_label(label, trial)
            labeled["trial"] = label
            labeled["id"] = str(len(entries))
            entries.append(labeled)
    if not entries:
        detail = f" ({'; '.join(warnings)})" if warnings else ""
        raise ValueError(f"No trials from {path}{detail}")
    return entries, warnings
