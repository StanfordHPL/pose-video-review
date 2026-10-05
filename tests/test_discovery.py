import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pose_video_review.discovery import (
    discover_folder,
    discover_session,
    discover_trial_list,
    frame_timestamps,
)
from pose_video_review.state import ViewerState


METADATA = (60.0, 720, 1280, 120)
FRAME_TIMES = [index / 60 for index in range(120)]


@contextmanager
def discovery_metadata():
    with (
        patch("pose_video_review.discovery.video_metadata", return_value=METADATA),
        patch("pose_video_review.discovery.frame_timestamps", return_value=FRAME_TIMES),
    ):
        yield


def create_trial(
    session: Path,
    camera: str = "Cam0",
    trial: str = "squat",
    output_folder: str = "OutputPkl",
    pose_name: str | None = None,
) -> tuple[Path, Path]:
    video = session / "Videos" / camera / "InputMedia" / trial / f"{trial}.mp4"
    pose_name = pose_name or f"{trial}_keypoints.pkl"
    pose = session / "Videos" / camera / output_folder / pose_name
    video.parent.mkdir(parents=True, exist_ok=True)
    pose.parent.mkdir(parents=True, exist_ok=True)
    video.touch()
    pose.touch()
    return video, pose


class DiscoveryTests(unittest.TestCase):
    def test_discovers_legacy_openpose_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory) / "OpenCapData_session"
            video, pose = create_trial(session)

            with discovery_metadata():
                entries = discover_session(session)

            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["trial"], "squat")
            self.assertEqual(entries[0]["videoPath"], str(video.resolve()))
            self.assertEqual(entries[0]["posePath"], str(pose.resolve()))
            self.assertEqual(entries[0]["poseType"], "OpenPose")
            self.assertEqual(entries[0]["frameTimes"], FRAME_TIMES)
            self.assertEqual(entries[0]["frameTiming"], "verified-pts")

    def test_discovers_current_openpose_and_hrnet_layouts(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory) / "OpenCapData_session"
            _, openpose = create_trial(
                session, "Cam0", "jump", "OutputPkl_default/jump", "trial-id_rotated_pp.pkl"
            )
            _, hrnet = create_trial(
                session, "Cam1", "squat", "OutputPkl_mmpose_0.8/squat", "trial-id_rotated_pp.pkl"
            )

            with discovery_metadata():
                entries = discover_session(session)

            self.assertEqual({entry["posePath"] for entry in entries}, {str(openpose.resolve()), str(hrnet.resolve())})
            self.assertEqual({entry["poseType"] for entry in entries}, {"OpenPose", "HRNet"})

    def test_prefers_processed_pickle_over_raw_hrnet_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory) / "OpenCapData_session"
            _, raw = create_trial(
                session, "Cam0", "squat", "OutputPkl_mmpose_0.8/squat", "trial-id_rotated.pkl"
            )
            processed = raw.with_name("trial-id_rotated_pp.pkl")
            processed.touch()

            with discovery_metadata():
                entries = discover_session(session)

            self.assertEqual(entries[0]["posePath"], str(processed.resolve()))

    def test_discovers_and_labels_neutral_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory) / "OpenCapData_session"
            create_trial(session, trial="neutral")
            create_trial(session, trial="squat")

            with discovery_metadata():
                entries = discover_session(session)

            self.assertEqual(
                {(entry["trial"], entry["trialType"]) for entry in entries},
                {("neutral", "neutral"), ("squat", "dynamic")},
            )

    def test_discovers_multiple_sessions_and_uses_unique_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "downloads"
            create_trial(root / "OpenCapData_one")
            create_trial(root / "nested" / "OpenCapData_two")

            with discovery_metadata():
                entries = discover_folder(root)

            self.assertEqual([entry["id"] for entry in entries], ["0", "1"])
            self.assertEqual(
                {entry["trial"] for entry in entries},
                {"OpenCapData_one / squat", "OpenCapData_two / squat"},
            )

    def test_uses_sync_video_and_inferred_pose_frame_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory) / "OpenCapData_session"
            video, _ = create_trial(session)
            sync = video.with_name("squat_sync.mp4")
            sync.touch()

            with (
                discovery_metadata(),
                patch("pose_video_review.discovery.infer_sync_frame_offset", return_value=12),
            ):
                entry = discover_session(session)[0]

            self.assertEqual(entry["videoPath"], str(sync.resolve()))
            self.assertEqual(entry["poseFrameOffset"], 12)

    def test_trial_list_keeps_manifest_order_and_skips_other_videos(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session_a = root / "OpenCapData_a"
            session_b = root / "session with spaces"
            create_trial(session_a, trial="squat")
            create_trial(session_a, trial="lunge")
            create_trial(session_a, trial="other")
            create_trial(session_b, trial="jump")
            manifest = root / "manifest.txt"
            manifest.write_text(
                "\n".join([
                    f"{session_a}\tsquat\tLabel one",
                    f"{session_b}  jump  Label two",
                    f"{session_a}\tlunge\tLabel three",
                    f"{session_a}\tmissing\tLabel missing",
                    "# comment",
                    "",
                ]),
                encoding="utf-8",
            )
            indexed: list[str] = []

            def record_timestamps(path: Path):
                indexed.append(path.name)
                return FRAME_TIMES

            with (
                patch("pose_video_review.discovery.video_metadata", return_value=METADATA),
                patch("pose_video_review.discovery.frame_timestamps", side_effect=record_timestamps),
                patch("pose_video_review.state.Path.cwd", return_value=root),
            ):
                entries, warnings = discover_trial_list(manifest)
                state = ViewerState(root, trial_list=manifest)

            self.assertEqual([entry["trial"] for entry in entries], ["Label one", "Label two", "Label three"])
            self.assertEqual([trial["id"] for trial in state.trials()], ["Label one", "Label two", "Label three"])
            self.assertNotIn("other.mp4", indexed)
            self.assertTrue(any("missing" in warning for warning in warnings))
            self.assertEqual(state.offsets_path, root / "pose-video-offsets.json")
            self.assertEqual(state.warnings, warnings)

    def test_reads_and_normalizes_ffprobe_frame_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "video.mp4"
            video.touch()
            output = json.dumps({"frames": [
                {"best_effort_timestamp_time": "2.000000"},
                {"best_effort_timestamp_time": "2.016667"},
                {"best_effort_timestamp_time": "2.033333"},
            ]})
            completed = SimpleNamespace(stdout=output, stderr="")

            with (
                patch("pose_video_review.discovery.shutil.which", return_value="/usr/bin/ffprobe"),
                patch("pose_video_review.discovery.subprocess.run", return_value=completed) as run,
            ):
                timestamps = frame_timestamps(video)

            self.assertEqual(timestamps, [0.0, 0.016667, 0.033333])
            self.assertIn("-show_frames", run.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
