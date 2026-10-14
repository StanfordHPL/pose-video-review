import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pose_video_review.discovery import participant_from_label
from pose_video_review.roi_marks import load_marks, save_marks
from pose_video_review.state import ViewerState

from test_discovery import FRAME_TIMES, METADATA, create_trial


class RoiMarkTests(unittest.TestCase):
    def test_participant_is_the_label_prefix_before_the_trial_name(self):
        self.assertEqual(participant_from_label("ID76 sdj_r1", "sdj_r1"), "ID76")
        self.assertEqual(participant_from_label("Label one", "squat"), "")

    def test_round_trip_replaces_one_bound_and_clear_drops_the_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = root / "session"
            create_trial(session, trial="sdj_r1")
            manifest = root / "manifest.txt"
            manifest.write_text(f"{session}\tsdj_r1\tID76 sdj_r1\n", encoding="utf-8")
            marks_path = root / "roi_marks.txt"

            with (
                patch("pose_video_review.discovery.video_metadata", return_value=METADATA),
                patch("pose_video_review.discovery.frame_timestamps", return_value=FRAME_TIMES),
                patch("pose_video_review.state.Path.cwd", return_value=root),
            ):
                state = ViewerState(root, trial_list=manifest)
            state.use_roi_marks(marks_path)
            self.assertEqual(state.trials()[0]["roiStatus"], "none")
            state.set_roi_mark("ID76 sdj_r1", "start", 4.31, 258)
            self.assertEqual(state.trials()[0]["roiStatus"], "start")
            state.set_roi_mark("ID76 sdj_r1", "end", 4.89, 293)
            state.set_roi_mark("ID76 sdj_r1", "start", 4.4, 264)

            stored = load_marks(marks_path)
            self.assertEqual(stored[("ID76", "sdj_r1")]["start"], {"time_s": 4.4, "frame": 264})
            self.assertEqual(stored[("ID76", "sdj_r1")]["end"]["frame"], 293)
            payload = state.roi_marks_payload("ID76 sdj_r1")
            self.assertTrue(payload["enabled"])
            self.assertEqual(payload["start"]["frame"], 264)

            self.assertEqual(state.trials()[0]["roiStatus"], "complete")
            state.clear_roi_marks("ID76 sdj_r1")
            self.assertEqual(load_marks(marks_path), {})
            self.assertIsNone(state.roi_marks_payload("ID76 sdj_r1")["end"])
            self.assertEqual(state.trials()[0]["roiStatus"], "none")

    def test_save_marks_orders_start_before_end(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roi_marks.txt"
            save_marks(path, {
                ("ID10", "rc_r1"): {"end": {"time_s": 2.0, "frame": 20}, "start": {"time_s": 1.0, "frame": 10}},
            })
            text = path.read_text(encoding="utf-8")
            self.assertLess(text.index("\tstart\t"), text.index("\tend\t"))
            self.assertIn("ID10\trc_r1\tstart\t1.000000\t10", text)


if __name__ == "__main__":
    unittest.main()
