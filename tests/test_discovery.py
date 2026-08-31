import json
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pose_video_review.discovery import discover_folder, discover_session, frame_timestamps


METADATA = (60.0, 720, 1280, 120)
FRAME_TIMES = [index / 60 for index in range(120)]


@contextmanager
def discovery_metadata():
    with (
        patch("pose_video_review.discovery.video_metadata", return_value=METADATA),
        patch("pose_video_review.discovery.frame_timestamps", return_value=FRAME_TIMES),
        redirect_stdout(StringIO()),
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

            with discovery_metadata(), redirect_stdout(StringIO()) as stdout:
                entries = discover_folder(root)

            output = stdout.getvalue()
            self.assertEqual([entry["id"] for entry in entries], ["0", "1"])
            self.assertEqual(
                {entry["trial"] for entry in entries},
                {"OpenCapData_one / squat", "OpenCapData_two / squat"},
            )
            self.assertIn("Found 2 sessions", output)
            self.assertIn("Session 1/2:", output)
            self.assertIn("Session 2/2:", output)
            self.assertIn("OpenCapData_one", output)
            self.assertIn("OpenCapData_two", output)

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

    def test_reports_trial_index_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory) / "OpenCapData_session"
            create_trial(session, trial="neutral")
            create_trial(session, trial="squat")
            create_trial(session, "Cam1", "squat")

            with discovery_metadata(), redirect_stdout(StringIO()) as stdout:
                entries = discover_session(session)

            output = stdout.getvalue()
            self.assertEqual(len(entries), 3)
            self.assertIn("Scanning session OpenCapData_session...", output)
            self.assertIn("Found 2 trials (3 camera videos). Indexing...", output)
            self.assertIn("Indexing 1/2: neutral (Cam0)", output)
            self.assertIn("Completed 1/2 trials: neutral", output)
            self.assertIn("Indexing 2/2: squat (Cam0)", output)
            self.assertIn("Indexing 2/2: squat (Cam1)", output)
            self.assertIn("Completed 2/2 trials: squat", output)

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
