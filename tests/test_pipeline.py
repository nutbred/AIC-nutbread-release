from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from aic_config import Settings
from preview import PreviewResolver, timestamp_url
from retrieval import (
    GENERAL_SCOPE, Evidence, RetrievalPipeline, cluster_evidence, resolve_family_scope,
    temporal_nms, video_matches_scope,
)
from submission import build_kis_slate, build_temporal_slate
from l23_scenes import parse_query_constraints


class FusionTests(unittest.TestCase):
    def test_manual_scope_resolution(self) -> None:
        self.assertEqual(resolve_family_scope("bike", None), ("L23", "bike scope: L23 only"))
        self.assertEqual(resolve_family_scope("general", None), (GENERAL_SCOPE, "general scope: L23 excluded"))
        self.assertEqual(resolve_family_scope("global", None), (None, "global scope: all families"))

    def test_general_scope_excludes_l23_without_query_routing(self) -> None:
        self.assertTrue(video_matches_scope("L21_V001", restrict_family=GENERAL_SCOPE))
        self.assertFalse(video_matches_scope("L23_V001", restrict_family=GENERAL_SCOPE))

    def test_l23_query_constraints_keep_garment_colors_separate(self) -> None:
        constraints = parse_query_constraints("tay dua ao vang quan den quay flycam")
        self.assertEqual(constraints.top_colors, ("yellow",))
        self.assertEqual(constraints.bottom_colors, ("black",))
        self.assertEqual(constraints.shot_type, "aerial")
    def test_fusion_joins_only_same_video_and_time_window(self) -> None:
        rows = [
            Evidence("qwen3_random", "L21_V001", 10_000, 0.8, 1, "visual"),
            Evidence("asr_lexical", "L21_V001", 12_000, 3.0, 1, "spoken"),
            Evidence("representative_lexical", "L21_V002", 11_000, 2.0, 1, "visual", exact_btc=True),
        ]
        events = cluster_evidence(rows, radius_ms=3_000, rrf_k=60)
        self.assertEqual(len(events), 2)
        first = next(event for event in events if event["video_id"] == "L21_V001")
        self.assertEqual({row.source for row in first["evidence"]}, {"qwen3_random", "asr_lexical"})

    def test_temporal_nms_suppresses_adjacent_same_video_only(self) -> None:
        events = [
            {"video_id": "L21_V001", "timestamp_ms": 10_000, "score": 3.0},
            {"video_id": "L21_V001", "timestamp_ms": 11_000, "score": 2.0},
            {"video_id": "L21_V002", "timestamp_ms": 11_000, "score": 1.0},
        ]
        kept = temporal_nms(events, threshold_ms=3_000, limit=10)
        self.assertEqual([(row["video_id"], row["timestamp_ms"]) for row in kept], [("L21_V001", 10_000), ("L21_V002", 11_000)])


class RealArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.settings = Settings()
        cls.settings.enable_qwen = False
        cls.settings.enable_dense_asr = False
        cls.settings.enable_dense_representative = False
        cls.settings.enable_five_fps = False
        cls.settings.enable_btc_clip = False
        cls.pipeline = RetrievalPipeline(cls.settings)

    def test_real_representative_and_asr_lexical_search(self) -> None:
        result = self.pipeline.search("đồng bằng sông cửu long sụt lún", mode="hybrid")
        self.assertTrue(result["results"])
        self.assertIn("L21_V001", {row["video_id"] for row in result["results"][:20]})
        self.assertIn("asr_lexical", result["source_counts"])
        self.assertIn("representative_lexical", result["source_counts"])

    def test_bike_scope_is_applied_to_all_returned_events(self) -> None:
        result = self.pipeline.search("đua xe đạp flycam", mode="hybrid", scope="bike")
        self.assertEqual(result["scope"]["name"], "bike")
        self.assertEqual(result["scope"]["family"], "L23")
        self.assertTrue(result["results"])
        self.assertTrue(all(row["video_id"].startswith("L23_") for row in result["results"]))

    def test_general_scope_excludes_l23_from_returned_events(self) -> None:
        result = self.pipeline.search("đua xe đạp flycam", mode="hybrid", scope="general")
        self.assertEqual(result["scope"]["name"], "general")
        self.assertEqual(result["scope"]["excludes"], ["L23"])
        self.assertTrue(all(not row["video_id"].startswith("L23_") for row in result["results"]))

    def test_asr_evidence_includes_two_chunks_on_each_side(self) -> None:
        self.pipeline.asr.load()
        evidence = self.pipeline.asr._evidence(2, 1.0, 1, "asr_lexical")
        self.assertEqual(len(evidence.asr_context), 5)
        self.assertEqual(
            [row["relation"] for row in evidence.asr_context],
            ["before", "before", "match", "after", "after"],
        )
        self.assertEqual(evidence.asr_context[2]["text"], evidence.text)

    def test_real_btc_ocr_ngram_search(self) -> None:
        rows = self.pipeline.ocr.search("Herbalife", 10, "L30_V072")
        self.assertTrue(rows)
        self.assertTrue(all(row.video_id == "L30_V072" for row in rows))
        self.assertIn(3_000, {row.timestamp_ms for row in rows})

    def test_qwen_and_representative_provenance_are_distinct(self) -> None:
        self.pipeline.qwen.load()
        qwen_row = self.pipeline.qwen.metadata.rows([0])[0]
        self.pipeline.representative.load()
        representative = self.pipeline.representative._evidence(0, 1.0, 1, "representative_lexical")
        self.assertEqual(self.pipeline.qwen.index.d, 2048)
        self.assertNotIn("keyframe_id", qwen_row)
        self.assertFalse(qwen_row.get("source_jpg_available", False))
        self.assertTrue(representative.exact_btc)
        self.assertIsNotNone(representative.keyframe_id)

    def test_five_fps_global_inventory(self) -> None:
        self.pipeline.five_fps.load()
        self.assertGreaterEqual(self.pipeline.five_fps.state.records or 0, 18_000)
        timestamps = [row["timestamp_ms"] for row in self.pipeline.five_fps.frames["L21_V001"][:6]]
        self.assertIn(200, [right - left for left, right in zip(timestamps, timestamps[1:])])

    def test_five_fps_is_global_and_clip_fallback_excludes_coverage(self) -> None:
        import torch

        self.pipeline.five_fps.build_index()
        frame = self.pipeline.five_fps.flat_frames[321]
        vector = torch.load(frame["embedding"], map_location="cpu", weights_only=True).float().numpy().reshape(1, -1)
        vector /= np.linalg.norm(vector, axis=1, keepdims=True)
        hit = self.pipeline.five_fps.global_search(vector, 1, None)[0]
        self.assertEqual((hit.video_id, hit.frame_id), (frame["video_id"], frame["frame_id"]))
        self.pipeline.clip.load()
        clip_query = np.asarray(self.pipeline.clip.index.reconstruct(0)).reshape(1, -1)
        fallback = self.pipeline.clip.search(
            clip_query, 5, None, exclude_videos=self.pipeline.five_fps.covered_videos
        )
        self.assertTrue(fallback)
        self.assertTrue(all(row.video_id not in self.pipeline.five_fps.covered_videos for row in fallback))

    def test_preview_labels_qwen_as_never_exact(self) -> None:
        resolver = PreviewResolver(self.settings)
        event = {
            "video_id": "L21_V001",
            "timestamp_ms": 0,
            "rank": 1,
            "score": 1.0,
            "evidence": [
                Evidence("qwen3_random", "L21_V001", 0, 1.0, 1, "visual", exact_btc=False).public()
            ],
        }
        decorated = resolver.decorate(event)
        self.assertNotEqual(decorated["preview"]["label"], "EXACT BTC FRAME")
        self.assertEqual(decorated["submission_frame_id"], "0")

    def test_nearest_downloaded_btc_is_preferred_for_non_visual_hit(self) -> None:
        resolver = PreviewResolver(self.settings)
        event = {
            "video_id": "L21_V001",
            "timestamp_ms": 12_345,
            "rank": 1,
            "score": 1.0,
            "evidence": [
                Evidence("asr_lexical", "L21_V001", 12_345, 1.0, 1, "spoken", text="sample").public()
            ],
        }
        decorated = resolver.decorate(event)
        self.assertEqual(decorated["preview"]["kind"], "btc_nearest")
        self.assertLessEqual(abs(decorated["preview"]["offset_ms"]), self.settings.preview_near_ms)
        self.assertEqual(decorated["asr_context"][0]["relation"], "match")

    def test_all_downloaded_btc_batches_are_discovered(self) -> None:
        batches = {path.parent.name for path in Settings().btc_image_roots}
        self.assertTrue({"Keyframes_L21", "Keyframes_L22", "Keyframes_L23", "Keyframes_L24", "Keyframes_L27"} <= batches)

    def test_timestamp_url_preserves_query(self) -> None:
        url = timestamp_url("https://youtube.com/watch?v=abc", 12_400)
        self.assertIn("v=abc", url or "")
        self.assertIn("t=12s", url or "")


class FlaskSmokeTests(unittest.TestCase):
    def test_status_and_home(self) -> None:
        from app import app

        client = app.test_client()
        self.assertEqual(client.get("/").status_code, 200)
        response = client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        self.assertIn("sources", response.get_json())

    def test_kis_export_validates_selected_rows(self) -> None:
        from app import app

        client = app.test_client()
        response = client.post(
            "/export/kis",
            json={
                "query_num": "7",
                "selected": [
                    {"video_id": "L21_V001", "frame_id": "510"},
                    {"video_id": "../../bad", "frame_id": "x"},
                ],
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_data(as_text=True), "L21_V001,510\n")

    def test_qa_export_includes_video_frame_and_answer(self) -> None:
        from app import app

        client = app.test_client()
        response = client.post(
            "/export/qa",
            json={"selected": [{"video_id": "L30_V072", "frame_id": "700"}], "answer": "Giang Ly"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_data(as_text=True), "L30_V072,700,Giang Ly\n")

    def test_temporal_export_includes_every_frame(self) -> None:
        from app import app

        client = app.test_client()
        response = client.post(
            "/export/temporal",
            json={"hypotheses": [{"video_id": "L24_V033", "frame_ids": ["15930", "15990", "16350"]}]},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_data(as_text=True), "L24_V033,15930,15990,16350\n")


class SubmissionTests(unittest.TestCase):
    def test_kis_slate_preserves_early_diversity_and_uses_tail_jitter(self) -> None:
        events = [
            {
                "rank": rank,
                "video_id": f"L21_V{rank:03d}",
                "submission_frame_id": str(rank * 100),
                "frame_hypotheses": [str(rank * 100), str(rank * 100 - 1), str(rank * 100 + 1)],
            }
            for rank in range(1, 101)
        ]
        rows = build_kis_slate(events)
        self.assertEqual(len(rows), 100)
        self.assertTrue(all(row["variant"] == "primary" for row in rows[:50]))
        self.assertTrue(any(row["variant"] == "time-jitter" for row in rows[80:]))

    def test_temporal_slate_requires_same_video_coverage(self) -> None:
        moment_results = [
            [{"video_id": "L21_V001", "timestamp_ms": 1_000, "submission_frame_id": "10", "frame_hypotheses": ["10", "9"]}],
            [{"video_id": "L21_V001", "timestamp_ms": 2_000, "submission_frame_id": "20", "frame_hypotheses": ["20", "19"]}],
        ]
        rows = build_temporal_slate(moment_results)
        self.assertEqual(rows[0]["frame_ids"], ["10", "20"])
        self.assertEqual(len(rows[0]["moments"]), 2)

    def test_temporal_slate_uses_ordered_alternative_instead_of_repeated_top_hit(self) -> None:
        moment_results = [
            [
                {"video_id": "L21_V001", "timestamp_ms": 1_000, "submission_frame_id": "10"},
                {"video_id": "L21_V001", "timestamp_ms": 3_000, "submission_frame_id": "30"},
            ],
            [
                {"video_id": "L21_V001", "timestamp_ms": 1_000, "submission_frame_id": "10"},
                {"video_id": "L21_V001", "timestamp_ms": 2_000, "submission_frame_id": "20"},
            ],
        ]
        rows = build_temporal_slate(moment_results)
        self.assertEqual(rows[0]["frame_ids"], ["10", "20"])
        self.assertEqual(rows[0]["variant"], "primary")


if __name__ == "__main__":
    unittest.main()
