"""Preview provenance and media helpers."""
from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from aic_config import Settings


SAFE_VIDEO_ID = re.compile(r"^L\d{2}_V\d{3}$")
SAFE_FRAME_ID = re.compile(r"^\d+$")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".avi", ".mov")


def timestamp_url(url: str | None, timestamp_ms: int) -> str | None:
    if not url:
        return None
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["t"] = f"{max(0, round(timestamp_ms / 1000))}s"
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def format_timestamp(timestamp_ms: int) -> str:
    timestamp_ms = max(0, int(timestamp_ms))
    hours, remainder = divmod(timestamp_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    prefix = f"{hours:02d}:" if hours else ""
    return f"{prefix}{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


class PreviewResolver:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._maps: dict[str, list[dict[str, Any]]] = {}
        self._five_fps_maps: dict[str, list[dict[str, Any]]] = {}
        self._media: dict[str, dict[str, Any]] = {}
        self._videos: dict[str, Path | None] = {}
        (settings.cache_dir / "extracted").mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _valid_video(video_id: str) -> bool:
        return bool(SAFE_VIDEO_ID.fullmatch(video_id))

    def media_info(self, video_id: str) -> dict[str, Any]:
        if not self._valid_video(video_id):
            return {}
        if video_id not in self._media:
            path = self.settings.media_info_dir / f"{video_id}.json"
            try:
                self._media[video_id] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            except (OSError, ValueError):
                self._media[video_id] = {}
        return self._media[video_id]

    def btc_map(self, video_id: str) -> list[dict[str, Any]]:
        if not self._valid_video(video_id):
            return []
        if video_id in self._maps:
            return self._maps[video_id]
        path = self.settings.map_dir / f"{video_id}.csv"
        rows = []
        if path.is_file():
            try:
                with path.open(newline="", encoding="utf-8") as handle:
                    for row in csv.DictReader(handle):
                        rows.append(
                            {
                                "keyframe_id": int(row["n"]),
                                "timestamp_ms": round(float(row["pts_time"]) * 1000),
                                "frame_id": str(row["frame_idx"]),
                                "fps": float(row["fps"]),
                            }
                        )
            except (KeyError, TypeError, ValueError, OSError):
                rows = []
        self._maps[video_id] = rows
        return rows

    def nearest_map_frame(self, video_id: str, timestamp_ms: int) -> dict[str, Any] | None:
        rows = self.btc_map(video_id)
        if not rows:
            return None
        return min(rows, key=lambda row: abs(int(row["timestamp_ms"]) - int(timestamp_ms)))

    def five_fps_map(self, video_id: str) -> list[dict[str, Any]]:
        """Load timestamp metadata without touching the much larger embedding index."""
        if not self._valid_video(video_id):
            return []
        if video_id in self._five_fps_maps:
            return self._five_fps_maps[video_id]
        path = self.settings.five_fps_map_dir / f"{video_id}_map.csv"
        rows = []
        if path.is_file():
            try:
                with path.open(newline="", encoding="utf-8") as handle:
                    for row in csv.DictReader(handle):
                        rows.append(
                            {
                                "frame_id": str(row["FrameID"]),
                                "timestamp_ms": round(float(row["Seconds"]) * 1000),
                            }
                        )
            except (KeyError, TypeError, ValueError, OSError):
                rows = []
        self._five_fps_maps[video_id] = rows
        return rows

    def nearest_five_fps(self, video_id: str, timestamp_ms: int) -> dict[str, Any] | None:
        candidates = sorted(
            self.five_fps_map(video_id),
            key=lambda row: abs(int(row["timestamp_ms"]) - int(timestamp_ms)),
        )
        for row in candidates:
            path = self.five_fps_path(video_id, str(row["frame_id"]))
            if path is not None:
                return {**row, "path": path}
        return None

    def canonical_frame_id(
        self, video_id: str, timestamp_ms: int, evidence: list[dict[str, Any]], anchor_source: str | None
    ) -> tuple[str | None, float | None]:
        """Resolve an evaluator frame ID independently from preview availability."""

        five_fps = next(
            (
                row
                for row in evidence
                if row.get("source") == "siglip_5fps"
                and SAFE_FRAME_ID.fullmatch(str(row.get("frame_id") or ""))
            ),
            None,
        )
        if five_fps:
            nearest = self.nearest_map_frame(video_id, timestamp_ms)
            return str(five_fps["frame_id"]), float(nearest["fps"]) if nearest else None

        anchored = next(
            (
                row
                for row in evidence
                if row.get("source") == anchor_source
                and SAFE_FRAME_ID.fullmatch(str(row.get("frame_id") or ""))
            ),
            None,
        )
        if anchored:
            nearest = self.nearest_map_frame(video_id, timestamp_ms)
            return str(anchored["frame_id"]), float(nearest["fps"]) if nearest else None

        nearest = self.nearest_map_frame(video_id, timestamp_ms)
        if nearest is None:
            fallback = next(
                (row for row in evidence if SAFE_FRAME_ID.fullmatch(str(row.get("frame_id") or ""))), None
            )
            return (str(fallback["frame_id"]), None) if fallback else (None, None)
        if int(nearest["timestamp_ms"]) == int(timestamp_ms):
            return str(nearest["frame_id"]), float(nearest["fps"])
        frame_id = int(max(0, timestamp_ms) * float(nearest["fps"]) / 1000.0)
        return str(frame_id), float(nearest["fps"])

    def btc_path(
        self,
        video_id: str,
        keyframe_id: int | None = None,
        file_name: str | None = None,
    ) -> Path | None:
        if not self._valid_video(video_id):
            return None
        safe_name = Path(file_name).name if file_name else None
        names = []
        if safe_name:
            names.append(safe_name)
        if keyframe_id is not None:
            names.extend([f"{keyframe_id:03d}.jpg", f"{keyframe_id:06d}.jpg", f"{keyframe_id}.jpg"])
        for root in self.settings.btc_image_roots:
            for name in names:
                candidate = root / video_id / name
                if candidate.is_file():
                    return candidate
        return None

    def nearest_btc(self, video_id: str, timestamp_ms: int) -> dict[str, Any] | None:
        candidates = sorted(
            self.btc_map(video_id), key=lambda row: abs(int(row["timestamp_ms"]) - int(timestamp_ms))
        )
        for row in candidates:
            path = self.btc_path(video_id, keyframe_id=int(row["keyframe_id"]))
            if path is not None:
                return {**row, "path": path}
        return None

    def five_fps_path(self, video_id: str, frame_id: str) -> Path | None:
        if not self._valid_video(video_id) or not SAFE_FRAME_ID.fullmatch(str(frame_id)):
            return None
        path = self.settings.five_fps_image_dir / video_id / f"keyframe_{frame_id}.webp"
        return path if path.is_file() else None

    def video_path(self, video_id: str) -> Path | None:
        if not self._valid_video(video_id):
            return None
        if video_id not in self._videos:
            found = None
            for root in self.settings.video_roots:
                for extension in VIDEO_EXTENSIONS:
                    candidate = root / f"{video_id}{extension}"
                    if candidate.is_file():
                        found = candidate
                        break
                if found:
                    break
            self._videos[video_id] = found
        return self._videos[video_id]

    def extracted_path(self, video_id: str, timestamp_ms: int) -> Path | None:
        video = self.video_path(video_id)
        if video is None:
            return None
        timestamp_ms = max(0, int(timestamp_ms))
        output = self.settings.cache_dir / "extracted" / f"{video_id}_{timestamp_ms}.jpg"
        if output.is_file() and output.stat().st_size:
            return output
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp_ms / 1000:.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            "-y",
            str(output),
        ]
        try:
            completed = subprocess.run(command, capture_output=True, timeout=30, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return output if completed.returncode == 0 and output.is_file() and output.stat().st_size else None

    def decorate(self, event: dict[str, Any]) -> dict[str, Any]:
        video_id = str(event["video_id"])
        timestamp_ms = int(event["timestamp_ms"])
        evidence = list(event.get("evidence", []))
        media = self.media_info(video_id)
        watch_url = next((row.get("watch_url") for row in evidence if row.get("watch_url")), None)
        watch_url = watch_url or media.get("watch_url")
        title = next((row.get("title") for row in evidence if row.get("title")), None) or media.get("title")

        preview: dict[str, Any] | None = None
        # Prefer an exact BTC frame when the winning evidence has one.
        winner = max(evidence, key=lambda row: float(row.get("contribution", 0)), default={})
        if winner.get("exact_btc"):
            path = self.btc_path(
                video_id,
                keyframe_id=winner.get("keyframe_id"),
                file_name=winner.get("file_name"),
            )
            if path is not None:
                if winner.get("keyframe_id") is not None:
                    preview_url = f"/preview/btc/{video_id}/{int(winner['keyframe_id'])}"
                else:
                    preview_url = f"/preview/btc-file/{video_id}/{Path(str(winner['file_name'])).name}"
                preview = {
                    "kind": "btc",
                    "label": "EXACT BTC FRAME",
                    "timestamp_ms": int(winner["timestamp_ms"]),
                    "offset_ms": int(winner["timestamp_ms"]) - timestamp_ms,
                    "url": preview_url,
                    "frame_id": winner.get("frame_id"),
                    "keyframe_id": winner.get("keyframe_id"),
                }

        # Qwen records have no source JPG; use a nearby BTC frame when available.
        if preview is None:
            nearest = self.nearest_btc(video_id, timestamp_ms)
            if nearest is not None and abs(int(nearest["timestamp_ms"]) - timestamp_ms) <= self.settings.preview_near_ms:
                preview = {
                    "kind": "btc_nearest",
                    "label": "NEAREST BTC PREVIEW",
                    "timestamp_ms": int(nearest["timestamp_ms"]),
                    "offset_ms": int(nearest["timestamp_ms"]) - timestamp_ms,
                    "url": f"/preview/btc/{video_id}/{nearest['keyframe_id']}",
                    "frame_id": nearest.get("frame_id"),
                    "keyframe_id": nearest.get("keyframe_id"),
                }

        # Use the exact 5-FPS hit when available.
        if preview is None:
            five_fps = next((row for row in evidence if row.get("source") == "siglip_5fps"), None)
            if five_fps and self.five_fps_path(video_id, str(five_fps.get("frame_id"))) is not None:
                preview = {
                    "kind": "five_fps",
                    "label": "5-FPS PREVIEW",
                    "timestamp_ms": int(five_fps["timestamp_ms"]),
                    "offset_ms": int(five_fps["timestamp_ms"]) - timestamp_ms,
                    "url": f"/preview/5fps/{video_id}/{five_fps['frame_id']}",
                    "frame_id": five_fps.get("frame_id"),
                }

        # Other modalities may use a nearby 5-FPS image.
        if preview is None:
            nearest_five_fps = self.nearest_five_fps(video_id, timestamp_ms)
            if (
                nearest_five_fps is not None
                and abs(int(nearest_five_fps["timestamp_ms"]) - timestamp_ms) <= self.settings.preview_near_ms
            ):
                preview = {
                    "kind": "five_fps_nearest",
                    "label": "NEAREST 5-FPS PREVIEW",
                    "timestamp_ms": int(nearest_five_fps["timestamp_ms"]),
                    "offset_ms": int(nearest_five_fps["timestamp_ms"]) - timestamp_ms,
                    "url": f"/preview/5fps/{video_id}/{nearest_five_fps['frame_id']}",
                    "frame_id": nearest_five_fps.get("frame_id"),
                }

        # Extract local video frames lazily.
        if preview is None and self.video_path(video_id) is not None:
            preview = {
                "kind": "extracted",
                "label": "EXTRACTED AT MATCH TIME",
                "timestamp_ms": timestamp_ms,
                "offset_ms": 0,
                "url": f"/preview/extracted/{video_id}/{timestamp_ms}.jpg",
            }

        if preview is None:
            preview = {
                "kind": "none",
                "label": "NO LOCAL PREVIEW",
                "timestamp_ms": None,
                "offset_ms": None,
                "url": None,
            }

        badges = []
        source_badges = {
            "btc_clip": "BTC CLIP fallback",
            "btc_ocr_ngram": "OCR n-gram",
            "qwen3_random": "Qwen3",
            "representative": "Scene/OD",
            "representative_lexical": "Scene/OD",
            "representative_dense": "Scene/OD",
            "asr_lexical": "ASR",
            "asr_dense": "ASR",
            "siglip_5fps": "5-FPS SigLIP",
        }
        for row in evidence:
            badge = source_badges.get(str(row.get("source")))
            if badge and badge not in badges:
                badges.append(badge)
            if row.get("ocr") and "OCR" not in badges:
                badges.append("OCR")

        scene_row = next((row for row in evidence if row.get("scene")), {})
        ocr_row = next((row for row in evidence if row.get("ocr")), {})
        object_row = next((row for row in evidence if row.get("objects")), {})
        asr_row = next((row for row in evidence if str(row.get("source", "")).startswith("asr_")), {})
        asr_context = asr_row.get("asr_context") or []
        if asr_row.get("text") and not asr_context:
            asr_context = [
                {
                    "relation": "match",
                    "start_ms": int(asr_row.get("timestamp_ms", timestamp_ms)),
                    "end_ms": int(asr_row.get("timestamp_ms", timestamp_ms)),
                    "text": str(asr_row["text"]),
                }
            ]
        object_labels = []
        for item in object_row.get("objects") or []:
            label = str(item.get("label", "")).strip()
            count = item.get("count")
            if label:
                object_labels.append(f"{label} ×{count}" if count else label)

        export_frame_id, fps = self.canonical_frame_id(
            video_id, timestamp_ms, evidence, event.get("anchor_source")
        )
        frame_hypotheses: list[str] = []
        if export_frame_id is not None:
            base = int(export_frame_id)
            offsets = [0, -1, 1]
            if fps:
                short = max(2, round(fps * 0.2))
                offsets.extend([-short, short, -round(fps), round(fps)])
            frame_hypotheses = [
                str(base + offset) for offset in dict.fromkeys(offsets) if base + offset >= 0
            ]

        return {
            **event,
            "timestamp": format_timestamp(timestamp_ms),
            "title": title,
            "watch_url": watch_url,
            "watch_url_with_timestamp": timestamp_url(watch_url, timestamp_ms),
            "preview": preview,
            "badges": badges,
            "scene": scene_row.get("scene"),
            "ocr": ocr_row.get("ocr") or [],
            "objects": object_labels,
            "asr": asr_row.get("text"),
            "asr_context": asr_context,
            "submission_frame_id": export_frame_id,
            "export_frame_id": export_frame_id,
            "frame_hypotheses": frame_hypotheses,
        }
