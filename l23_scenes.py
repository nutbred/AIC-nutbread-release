"""Compact L23 cycling-scene metadata and query-aware reranking helpers."""
from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
COLOR_NAMES = (
    "black", "white", "gray", "red", "orange", "yellow", "green",
    "cyan", "blue", "navy", "purple", "pink", "brown",
)


def fold_text(value: str) -> str:
    value = " ".join(unicodedata.normalize("NFKC", value).casefold().split()).replace("đ", "d")
    normalized = unicodedata.normalize("NFD", value)
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


COLOR_ALIASES = {
    "black": ("black", "den"),
    "white": ("white", "trang"),
    "gray": ("gray", "grey", "xam"),
    "red": ("red", "do"),
    "orange": ("orange", "mau cam"),
    "yellow": ("yellow", "vang"),
    "green": ("green", "xanh la"),
    "cyan": ("cyan", "sky blue", "xanh nhat"),
    "blue": ("blue", "xanh duong"),
    "navy": ("navy", "dark blue", "xanh dam"),
    "purple": ("purple", "tim"),
    "pink": ("pink", "hong"),
    "brown": ("brown", "nau"),
}
AERIAL_TERMS = ("aerial", "overhead", "bird's eye", "bird eye", "drone", "helicopter", "flycam", "tren cao", "tu tren", "tren xuong")
GROUND_TERMS = ("ground level", "close up", "close-up", "tracking shot", "road level", "gan", "can canh")
TOP_TERMS = ("ao", "jersey", "top", "shirt")
BOTTOM_TERMS = ("quan", "shorts", "bottom", "bib")


@dataclass(frozen=True, slots=True)
class QueryConstraints:
    top_colors: tuple[str, ...] = ()
    bottom_colors: tuple[str, ...] = ()
    shot_type: str | None = None

    @property
    def active(self) -> bool:
        return bool(self.top_colors or self.bottom_colors or self.shot_type)


def _phrase_positions(text: str, phrases: tuple[str, ...]) -> list[int]:
    positions: list[int] = []
    for phrase in phrases:
        positions.extend(match.start() for match in re.finditer(rf"(?<!\\w){re.escape(phrase)}(?!\\w)", text))
    return positions


def _color_positions(text: str) -> list[tuple[int, str]]:
    positions: list[tuple[int, str]] = []
    for color, aliases in COLOR_ALIASES.items():
        matches = _phrase_positions(text, aliases)
        if matches:
            positions.append((min(matches), color))
    return positions


def _garment_positions(text: str, terms: tuple[str, ...]) -> list[int]:
    return _phrase_positions(text, terms)


def parse_query_constraints(query: str) -> QueryConstraints:
    """Extract only explicit color/camera hints; this is not query translation."""
    text = fold_text(query)
    color_positions = _color_positions(text)
    top_positions = _garment_positions(text, TOP_TERMS)
    bottom_positions = _garment_positions(text, BOTTOM_TERMS)
    top: list[str] = []
    bottom: list[str] = []
    for position, color in color_positions:
        nearest_top = min((abs(position - garment) for garment in top_positions), default=float("inf"))
        nearest_bottom = min((abs(position - garment) for garment in bottom_positions), default=float("inf"))
        if nearest_top < nearest_bottom:
            top.append(color)
        elif nearest_bottom < nearest_top:
            bottom.append(color)
    # A bare color description is treated as an upper-body hint, never a hard filter.
    if color_positions and not top and not bottom:
        top = [color for _, color in color_positions]
    shot_type = "aerial" if any(term in text for term in AERIAL_TERMS) else None
    if shot_type is None and any(term in text for term in GROUND_TERMS):
        shot_type = "normal"
    return QueryConstraints(tuple(dict.fromkeys(top)), tuple(dict.fromkeys(bottom)), shot_type)


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS scenes (
            video_id TEXT NOT NULL,
            scene_id INTEGER NOT NULL,
            start_ms INTEGER NOT NULL,
            end_ms INTEGER NOT NULL,
            representative_ms INTEGER NOT NULL,
            camera_type TEXT NOT NULL CHECK(camera_type IN ('aerial', 'normal', 'unknown')),
            camera_confidence REAL NOT NULL DEFAULT 0,
            cyclist_count INTEGER NOT NULL DEFAULT 0,
            top_colors TEXT NOT NULL DEFAULT '[]',
            bottom_colors TEXT NOT NULL DEFAULT '[]',
            rider_colors TEXT NOT NULL DEFAULT '[]',
            PRIMARY KEY (video_id, scene_id)
        );
        CREATE INDEX IF NOT EXISTS scenes_video_time
            ON scenes(video_id, start_ms, end_ms);
        """
    )
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    connection.commit()


def _decode_colors(value: str) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return []
    return [row for row in parsed if isinstance(row, dict) and row.get("color") in COLOR_NAMES]


def _color_match(requested: tuple[str, ...], observed: list[dict[str, Any]]) -> float:
    if not requested:
        return 0.0
    weights = {str(row["color"]): max(0.0, min(1.0, float(row.get("weight", 0.0)))) for row in observed}
    return sum(weights.get(color, 0.0) for color in requested) / len(requested)


class L23SceneStore:
    """Read-only SQLite lookup used to softly rerank already retrieved L23 events."""

    name = "l23_scene_metadata"

    def __init__(self, path: Path) -> None:
        self.path = path
        self.connection: sqlite3.Connection | None = None
        self.error: str | None = None
        self.records: int | None = None

    @property
    def configured(self) -> bool:
        return self.path.is_file()

    def load(self) -> None:
        if self.connection is not None or self.error is not None:
            return
        if not self.path.is_file():
            self.error = "database file is not configured"
            return
        try:
            connection = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True, check_same_thread=False)
            version = connection.execute("SELECT value FROM metadata WHERE key = 'schema_version'").fetchone()
            if version is None or int(version[0]) != SCHEMA_VERSION:
                raise RuntimeError("unsupported L23 scene database schema")
            self.records = int(connection.execute("SELECT COUNT(*) FROM scenes").fetchone()[0])
            self.connection = connection
        except (OSError, sqlite3.Error, RuntimeError, ValueError) as error:
            self.error = f"{type(error).__name__}: {error}"

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def status(self) -> dict[str, Any]:
        self.load()
        return {
            "name": self.name,
            "role": "offline L23 cycling shot and clothing-color metadata",
            "configured": self.configured,
            "loaded": self.connection is not None,
            "records": self.records,
            "error": self.error,
        }

    def scene_at(self, video_id: str, timestamp_ms: int) -> dict[str, Any] | None:
        self.load()
        if self.connection is None:
            return None
        row = self.connection.execute(
            """
            SELECT scene_id, start_ms, end_ms, representative_ms, camera_type,
                   camera_confidence, cyclist_count, top_colors, bottom_colors, rider_colors
            FROM scenes
            WHERE video_id = ? AND start_ms <= ? AND end_ms >= ?
            ORDER BY (end_ms - start_ms) ASC
            LIMIT 1
            """,
            (video_id, int(timestamp_ms), int(timestamp_ms)),
        ).fetchone()
        if row is None:
            return None
        return {
            "scene_id": int(row[0]), "start_ms": int(row[1]), "end_ms": int(row[2]),
            "representative_ms": int(row[3]), "camera_type": str(row[4]),
            "camera_confidence": round(float(row[5]), 3), "cyclist_count": int(row[6]),
            "top_colors": _decode_colors(row[7]), "bottom_colors": _decode_colors(row[8]),
            "rider_colors": _decode_colors(row[9]),
        }

    def rerank(self, events: list[dict[str, Any]], constraints: QueryConstraints) -> int:
        """Annotate L23 events and add at most a 30% score multiplier for explicit hints."""
        if not constraints.active:
            return 0
        matched = 0
        for event in events:
            if not str(event.get("video_id", "")).startswith("L23_"):
                continue
            scene = self.scene_at(str(event["video_id"]), int(event["timestamp_ms"]))
            if scene is None:
                continue
            top_match = _color_match(constraints.top_colors, scene["top_colors"])
            bottom_match = _color_match(constraints.bottom_colors, scene["bottom_colors"])
            shot_match = float(scene["camera_type"] == constraints.shot_type) if constraints.shot_type else 0.0
            bonus = min(0.30, 0.14 * top_match + 0.14 * bottom_match + 0.08 * shot_match)
            scene["query_match"] = {
                "top_color": round(top_match, 3), "bottom_color": round(bottom_match, 3),
                "shot_type": round(shot_match, 3), "boost": round(bonus, 3),
            }
            event["l23_scene"] = scene
            if bonus:
                event["score"] *= 1.0 + bonus
                matched += 1
        return matched
