"""Evaluator-aware KIS/VQA and temporal answer slates."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


VIDEO_RE = re.compile(r"^L\d{2}_V\d{3}$")
FRAME_RE = re.compile(r"^\d+$")


def _candidate(video_id: Any, frame_id: Any, **extra: Any) -> dict[str, Any] | None:
    video = str(video_id or "").upper()
    frame = str(frame_id or "")
    if not VIDEO_RE.fullmatch(video) or not FRAME_RE.fullmatch(frame):
        return None
    return {"video_id": video, "frame_id": frame, **extra}


def build_kis_slate(events: list[dict[str, Any]], limit: int = 100) -> list[dict[str, Any]]:
    """Keep cutoff ranks diverse, then spend the tail on temporal robustness."""

    limit = max(1, min(int(limit), 100))
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def append(video_id: Any, frame_id: Any, **extra: Any) -> None:
        row = _candidate(video_id, frame_id, **extra)
        if row is None or (row["video_id"], row["frame_id"]) in seen or len(output) >= limit:
            return
        seen.add((row["video_id"], row["frame_id"]))
        row["submission_rank"] = len(output) + 1
        output.append(row)

    # Limit repeated moments from one video at each scoring cutoff.
    ordered: list[dict[str, Any]] = []
    remaining = list(events)
    per_video: dict[str, int] = defaultdict(int)
    for cutoff, quota in ((5, 2), (20, 3), (50, 5), (80, 10_000)):
        index = 0
        while len(ordered) < min(cutoff, len(events)) and index < len(remaining):
            event = remaining[index]
            video_id = str(event.get("video_id") or "")
            if per_video[video_id] < quota:
                ordered.append(event)
                per_video[video_id] += 1
                remaining.pop(index)
            else:
                index += 1
    ordered.extend(remaining)

    primary_budget = min(len(ordered), 80, limit)
    for event in ordered[:primary_budget]:
        append(event.get("video_id"), event.get("submission_frame_id") or event.get("export_frame_id"), event_rank=event.get("rank"), variant="primary")

    # Add timestamp variants after the primary candidates.
    for event in events[:20]:
        hypotheses = list(event.get("frame_hypotheses") or [])[1:]
        for frame_id in hypotheses:
            append(event.get("video_id"), frame_id, event_rank=event.get("rank"), variant="time-jitter")

    for event in ordered[primary_budget:]:
        append(event.get("video_id"), event.get("submission_frame_id") or event.get("export_frame_id"), event_rank=event.get("rank"), variant="primary-tail")
    return output


def build_temporal_slate(
    moment_results: list[list[dict[str, Any]]], limit: int = 100, rrf_k: int = 60
) -> list[dict[str, Any]]:
    """Rank distinct, chronologically ordered paths with one frame per moment."""

    if not moment_results:
        return []
    by_video: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for moment_index, events in enumerate(moment_results):
        seen: set[tuple[str, str, int]] = set()
        for rank, event in enumerate(events, start=1):
            video_id = str(event.get("video_id") or "").upper()
            frame_id = str(event.get("submission_frame_id") or event.get("export_frame_id") or "")
            if not VIDEO_RE.fullmatch(video_id) or not FRAME_RE.fullmatch(frame_id):
                continue
            try:
                timestamp_ms = int(event.get("timestamp_ms", frame_id))
            except (TypeError, ValueError):
                timestamp_ms = int(frame_id)
            key = (video_id, frame_id, timestamp_ms)
            if key in seen:
                continue
            seen.add(key)
            candidates = by_video[video_id][moment_index]
            if len(candidates) < 16:
                candidates.append(
                    {
                        **event,
                        "moment_rank": rank,
                        "frame_id": frame_id,
                        "timestamp_ms": timestamp_ms,
                    }
                )

    complete: list[tuple[float, str, list[dict[str, Any]]]] = []
    for video_id, moments in by_video.items():
        if len(moments) != len(moment_results):
            continue
        groups = [moments[index] for index in range(len(moment_results))]
        states: list[tuple[float, list[dict[str, Any]]]] = [
            (1.0 / (rrf_k + int(row["moment_rank"])), [row]) for row in groups[0]
        ]
        for group in groups[1:]:
            next_states: list[tuple[float, list[dict[str, Any]]]] = []
            for row in group:
                valid = [
                    state
                    for state in states
                    if int(row["timestamp_ms"]) - int(state[1][-1]["timestamp_ms"]) >= 200
                    and str(row["frame_id"]) != str(state[1][-1]["frame_id"])
                ]
                if not valid:
                    continue
                prior_score, prior_path = max(valid, key=lambda state: state[0])
                next_states.append(
                    (prior_score + 1.0 / (rrf_k + int(row["moment_rank"])), [*prior_path, row])
                )
            states = next_states
            if not states:
                break
        if states:
            score, ordered = max(states, key=lambda state: state[0])
            complete.append((score, video_id, ordered))
    complete.sort(reverse=True)

    output: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()

    def preview_moment(row: dict[str, Any], moment_index: int) -> dict[str, Any]:
        return {
            "moment_index": moment_index,
            "moment_rank": int(row["moment_rank"]),
            "frame_id": str(row["frame_id"]),
            "timestamp_ms": int(row["timestamp_ms"]),
            "timestamp": row.get("timestamp"),
            "preview": row.get("preview") or {},
            "watch_url_with_timestamp": row.get("watch_url_with_timestamp"),
            "badges": list(row.get("badges") or []),
            "title": row.get("title"),
        }

    def append(
        video_id: str,
        frames: list[str],
        score: float,
        variant: str,
        moments: list[dict[str, Any]] | None = None,
    ) -> None:
        key = (video_id, tuple(frames))
        if key in seen or len(output) >= min(limit, 100):
            return
        seen.add(key)
        output.append(
            {
                "rank": len(output) + 1,
                "video_id": video_id,
                "frame_ids": frames,
                "score": round(score, 8),
                "variant": variant,
                "moments": moments or [],
            }
        )

    for score, video_id, moments in complete[:80]:
        append(
            video_id,
            [str(row["frame_id"]) for row in moments],
            score,
            "primary",
            [preview_moment(row, index) for index, row in enumerate(moments)],
        )
    for score, video_id, moments in complete[:20]:
        for hypothesis_index in range(1, 7):
            frames = []
            for row in moments:
                hypotheses = list(row.get("frame_hypotheses") or [row["frame_id"]])
                frames.append(str(hypotheses[min(hypothesis_index, len(hypotheses) - 1)]))
            append(video_id, frames, score, "time-jitter")
    return output
