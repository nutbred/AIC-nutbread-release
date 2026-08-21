"""aic-nutbread multimodal video-retrieval server."""
from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import time
from typing import Any

from flask import Flask, Response, jsonify, render_template, request, send_file
from flask.json.provider import DefaultJSONProvider

from aic_config import Settings
from preview import PreviewResolver, SAFE_FRAME_ID, SAFE_VIDEO_ID
from retrieval import GENERAL_SCOPE, RetrievalPipeline, resolve_family_scope
from submission import build_kis_slate, build_temporal_slate


class SafeJSONProvider(DefaultJSONProvider):
    """JSON provider that never emits NaN / Infinity (strict browsers reject them)."""

    def dumps(self, obj: Any, **kwargs: Any) -> str:
        kwargs.setdefault("allow_nan", False)
        try:
            return super().dumps(obj, **kwargs)
        except (ValueError, TypeError):
            # Coerce any remaining non-finite floats to a finite sentinel.
            def sanitize(value: Any) -> Any:
                if isinstance(value, float):
                    if not math.isfinite(value):
                        return 0.0 if math.isnan(value) else (-1e38 if value < 0 else 1e38)
                    return value
                if isinstance(value, dict):
                    return {key: sanitize(item) for key, item in value.items()}
                if isinstance(value, (list, tuple)):
                    return [sanitize(item) for item in value]
                return value

            return super().dumps(sanitize(obj), allow_nan=False, **kwargs)


settings = Settings()
pipeline = RetrievalPipeline(settings)
previews = PreviewResolver(settings)

app = Flask(__name__)
app.json = SafeJSONProvider(app)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024


def json_error(message: str, status: int) -> tuple[Response, int]:
    return jsonify(error=message), status


@app.get("/")
def home() -> str:
    return render_template("index.html")


@app.get("/api/status")
def status() -> Response:
    return jsonify(pipeline.status())


@app.post("/search")
def search() -> tuple[Response, int] | Response:
    started = time.perf_counter()
    payload = request.get_json(silent=True) or {}
    query = str(payload.get("query") or "").strip()
    visual_query = str(payload.get("visual_query") or "").strip() or None
    mode = str(payload.get("mode") or "hybrid").strip().lower()
    scope = str(payload.get("scope") or "general").strip().lower()
    restrict_video = str(payload.get("restrict_video") or "").strip().upper() or None
    if not query:
        return json_error("Enter a Vietnamese or English text query.", 400)
    if restrict_video and not SAFE_VIDEO_ID.fullmatch(restrict_video):
        return json_error("Video filter must look like L21_V001.", 400)
    try:
        output = pipeline.search(
            query, mode=mode, restrict_video=restrict_video,
            scope=scope, visual_query=visual_query
        )
    except ValueError as error:
        return json_error(str(error), 400)
    except Exception as error:
        app.logger.exception("Search failed")
        return json_error(f"Search failed: {type(error).__name__}: {error}", 500)
    output["results"] = [previews.decorate(event) for event in output["results"]]
    output["submission_candidates"] = build_kis_slate(output["results"])
    output["elapsed_time"] = round(time.perf_counter() - started, 3)
    return jsonify(output)


EVENT_LINE_RE = re.compile(r"^\s*[Ee]\s*\d+\s*[:.\-–—)\]]?\s*(.+)$")


def _parse_temporal_lines(lines: list[str]) -> tuple[str, list[str]]:
    """Split a TRAKE-style block into (context, [moment, ...]).

    - Lines that start with an event marker (E1, E2, e3., E-4, ...) become moments;
      the marker itself is stripped from the search text.
    - The first non-marker line is treated as the shared context description.
    - If no line has an event marker, fall back to the legacy behavior where
      every line is a moment and there is no context.
    """
    context = ""
    moments: list[str] = []
    for line in lines:
        match = EVENT_LINE_RE.match(line)
        if match:
            text = match.group(1).strip()
            if text:
                moments.append(text)
        elif not context:
            context = line
    if not moments:
        return "", list(lines)
    return context, moments[:8]


@app.post("/temporal-search")
def temporal_search() -> tuple[Response, int] | Response:
    started = time.perf_counter()
    payload = request.get_json(silent=True) or {}
    raw_queries = payload.get("queries") or []
    if isinstance(raw_queries, str):
        raw_queries = raw_queries.splitlines()
    context, queries = _parse_temporal_lines(
        [str(value).strip() for value in raw_queries if str(value).strip()]
    )
    mode = str(payload.get("mode") or "hybrid").strip().lower()
    scope = str(payload.get("scope") or "general").strip().lower()
    restrict_video = str(payload.get("restrict_video") or "").strip().upper() or None
    if not queries:
        return json_error("Enter one moment description per line.", 400)
    if restrict_video and not SAFE_VIDEO_ID.fullmatch(restrict_video):
        return json_error("Video filter must look like L21_V001.", 400)
    try:
        effective_family, route_reason = resolve_family_scope(scope, restrict_video)
    except ValueError as error:
        return json_error(str(error), 400)
    moment_results: list[list[dict[str, Any]]] = []
    warnings: list[str] = []
    try:
        for query in queries:
            # Prepend the shared context line (if any) so dense channels search
            # for the moment *within* the described scene.
            search_query = f"{context}: {query}" if context else query
            output = pipeline.search(
                search_query, mode=mode, restrict_video=restrict_video, scope=scope,
            )
            warnings.extend(output.get("warnings") or [])
            moment_results.append([previews.decorate(event) for event in output["results"]])
    except ValueError as error:
        return json_error(str(error), 400)
    except Exception as error:
        app.logger.exception("Temporal search failed")
        return json_error(f"Temporal search failed: {type(error).__name__}: {error}", 500)
    hypotheses = build_temporal_slate(moment_results)
    for hypothesis in hypotheses:
        for moment in hypothesis.get("moments") or []:
            index = int(moment.get("moment_index", -1))
            if 0 <= index < len(queries):
                moment["query"] = queries[index]
    return jsonify(
        queries=queries,
        context=context,
        scope={
            "name": scope,
            "video": restrict_video,
            "family": None if effective_family == GENERAL_SCOPE else effective_family,
            "excludes": ["L23"] if effective_family == GENERAL_SCOPE else [],
            "reason": route_reason,
        },
        moment_result_counts=[len(values) for values in moment_results],
        hypotheses=hypotheses,
        warnings=list(dict.fromkeys(warnings)),
        elapsed_time=round(time.perf_counter() - started, 3),
    )


@app.get("/preview/5fps/<video_id>/<frame_id>")
def five_fps_preview(video_id: str, frame_id: str) -> tuple[Response, int] | Response:
    path = previews.five_fps_path(video_id, frame_id)
    if path is None:
        return json_error("5-FPS preview not found", 404)
    return send_file(path, conditional=True, max_age=3600)


@app.get("/preview/btc/<video_id>/<int:keyframe_id>")
def btc_preview(video_id: str, keyframe_id: int) -> tuple[Response, int] | Response:
    path = previews.btc_path(video_id, keyframe_id=keyframe_id)
    if path is None:
        return json_error("BTC keyframe not found", 404)
    return send_file(path, conditional=True, max_age=3600)


@app.get("/preview/btc-file/<video_id>/<filename>")
def btc_file_preview(video_id: str, filename: str) -> tuple[Response, int] | Response:
    path = previews.btc_path(video_id, file_name=filename)
    if path is None:
        return json_error("BTC keyframe not found", 404)
    return send_file(path, conditional=True, max_age=3600)


@app.get("/preview/extracted/<video_id>/<int:timestamp_ms>.jpg")
def extracted_preview(video_id: str, timestamp_ms: int) -> tuple[Response, int] | Response:
    path = previews.extracted_path(video_id, timestamp_ms)
    if path is None:
        return json_error("Source video or extracted preview unavailable", 404)
    return send_file(path, conditional=True, max_age=86400)


@app.get("/media-info/<video_id>/<frame_id>")
def media_info(video_id: str, frame_id: str) -> tuple[Response, int] | Response:
    if not SAFE_VIDEO_ID.fullmatch(video_id) or not SAFE_FRAME_ID.fullmatch(frame_id):
        return json_error("Invalid video or frame ID", 400)
    rows = previews.btc_map(video_id)
    match = next((row for row in rows if str(row["frame_id"]) == frame_id), None)
    timestamp_ms = int(match["timestamp_ms"]) if match else 0
    info = previews.media_info(video_id)
    from preview import timestamp_url

    return jsonify(
        video=video_id,
        frameid=frame_id,
        timestamp=timestamp_ms / 1000,
        watch_url=info.get("watch_url"),
        watch_url_with_timestamp=timestamp_url(info.get("watch_url"), timestamp_ms),
        title=info.get("title"),
    )


def _selected_rows(payload: dict[str, Any]) -> list[tuple[str, str]]:
    output = []
    for row in payload.get("selected", [])[:100]:
        video_id = str(row.get("video_id") or row.get("video") or "").strip().upper()
        frame_id = str(row.get("frame_id") or row.get("frameid") or "").strip()
        if SAFE_VIDEO_ID.fullmatch(video_id) and SAFE_FRAME_ID.fullmatch(frame_id):
            output.append((video_id, frame_id))
    return output


def _submission_rows(payload: dict[str, Any]) -> list[tuple[str, str]]:
    ranked: list[tuple[str, str]] = []
    for row in payload.get("ranked", [])[:100]:
        video_id = str(row.get("video_id") or row.get("video") or "").strip().upper()
        frame_id = str(row.get("frame_id") or row.get("frameid") or "").strip()
        if SAFE_VIDEO_ID.fullmatch(video_id) and SAFE_FRAME_ID.fullmatch(frame_id):
            ranked.append((video_id, frame_id))
    selected = _selected_rows(payload)
    if not selected:
        return ranked[:100]
    # Safety-net ranking: human-verified picks first (max points when correct),
    # then the retrieval ranking as fallback. Any correct frame inside the
    # top 100 still scores; the lower the rank, the lower the score.
    chosen_keys = set(selected)
    tail = [row for row in ranked if row not in chosen_keys]
    return (selected + tail)[:100]


def _csv_response(rows: list[list[str]], filename: str) -> Response:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerows(rows)
    return Response(
        stream.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/export/kis")
def export_kis() -> Response:
    payload = request.get_json(silent=True) or {}
    number = str(payload.get("query_num") or "1").replace("/", "-").replace("\\", "-")
    return _csv_response([list(row) for row in _submission_rows(payload)], f"query-{number}-kis.csv")


@app.post("/export/qa")
def export_qa() -> tuple[Response, int] | Response:
    payload = request.get_json(silent=True) or {}
    answer = str(payload.get("answer") or "").strip()
    if not answer:
        return json_error("QA answer is empty", 400)
    number = str(payload.get("query_num") or "1").replace("/", "-").replace("\\", "-")
    rows = _submission_rows(payload)
    if not rows:
        return json_error("Select or retrieve at least one video/frame candidate", 400)
    return _csv_response([[video_id, frame_id, answer] for video_id, frame_id in rows], f"query-{number}-qa.csv")


@app.post("/export/temporal")
def export_temporal() -> tuple[Response, int] | Response:
    payload = request.get_json(silent=True) or {}
    rows: list[list[str]] = []
    for row in payload.get("hypotheses", [])[:100]:
        video_id = str(row.get("video_id") or "").strip().upper()
        frame_ids = [str(value).strip() for value in row.get("frame_ids") or []]
        variant = str(row.get("variant") or "")
        if not SAFE_VIDEO_ID.fullmatch(video_id) or not frame_ids:
            continue
        if variant.startswith("partial"):
            # Partial-coverage rows: blank fields are kept so missing moments
            # can be filled in by hand before submission.
            if all(value == "" or SAFE_FRAME_ID.fullmatch(value) for value in frame_ids):
                rows.append([video_id, *frame_ids])
        elif all(SAFE_FRAME_ID.fullmatch(value) for value in frame_ids):
            rows.append([video_id, *frame_ids])
    if not rows:
        return json_error("No valid temporal hypotheses to export", 400)
    number = str(payload.get("query_num") or "1").replace("/", "-").replace("\\", "-")
    return _csv_response(rows, f"query-{number}-temporal.csv")


if __name__ == "__main__":
    print(f"Workspace: {settings.workspace_root}")
    if settings.preload_text_models:
        print(f"Model warmup: {pipeline.warm_models()}")
    else:
        print("Dense query encoders will load lazily on first use.")
    print("Open /api/status to inspect configured branches and resident model devices.")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False, threaded=True)
