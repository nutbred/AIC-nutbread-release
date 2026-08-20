"""Optional retrieval adapters and event-level fusion.

The visual spaces deliberately remain separate:

* Qwen3-VL-Embedding random frames: global vector/timestamp evidence with no source JPG.
* BTC sparse keyframes: OCR timestamps and optional legacy CLIP evidence.
* 5-FPS SigLIP: optional candidate/refinement evidence for the partial local pool.
* BTC-derived representative frames captioned by Qwen2.5-VL: textual scene/OCR/OD evidence.

Only ``(video_id, timestamp_ms)`` is used to join them.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import threading
import unicodedata
from array import array
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from aic_config import Settings
from l23_scenes import L23SceneStore, parse_query_constraints


TOKEN_RE = re.compile(r"\w+", re.UNICODE)
FAMILY_RE = re.compile(r"^L\d{2}$")


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def fold_text(value: str) -> str:
    normalized = unicodedata.normalize("NFD", normalize_text(value).replace("đ", "d"))
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def tokens(value: str) -> list[str]:
    return TOKEN_RE.findall(fold_text(value))


GENERAL_SCOPE = "!L23"
SEARCH_SCOPES = {"bike", "general", "global"}


def video_matches_scope(
    video_id: str, restrict_video: str | None = None, restrict_family: str | None = None
) -> bool:
    if restrict_video:
        return video_id == restrict_video
    if restrict_family == GENERAL_SCOPE:
        return not video_id.startswith("L23_")
    return not restrict_family or video_id.startswith(f"{restrict_family}_")


def scope_candidate_limit(
    limit: int, total: int | None, restrict_video: str | None, restrict_family: str | None,
    *, unfiltered_multiplier: int = 1,
) -> int:
    """Over-fetch enough for restrictive scopes without penalizing General's L23 exclusion."""
    if restrict_video or (restrict_family and restrict_family != GENERAL_SCOPE):
        multiplier = 60
    elif restrict_family == GENERAL_SCOPE:
        multiplier = 2
    else:
        multiplier = unfiltered_multiplier
    value = max(limit * multiplier, limit)
    return min(value, total) if total is not None else value


def resolve_family_scope(
    scope: str, restrict_video: str | None
) -> tuple[str | None, str]:
    """Translate the user's manual collection choice to the existing family filter."""

    if restrict_video:
        family = restrict_video.split("_", 1)[0]
        if scope == "general" and family == "L23":
            raise ValueError("general scope excludes L23; choose bike or global for an L23 video")
        if scope == "bike" and family != "L23":
            raise ValueError("bike scope only accepts L23 videos")
        return family, "exact video filter"
    if scope == "bike":
        return "L23", "bike scope: L23 only"
    if scope == "general":
        return GENERAL_SCOPE, "general scope: L23 excluded"
    if scope == "global":
        return None, "global scope: all families"
    raise ValueError("scope must be bike, general, or global")


def top_indices(scores: np.ndarray, limit: int) -> np.ndarray:
    if scores.size == 0 or limit <= 0:
        return np.empty(0, dtype=np.int64)
    limit = min(limit, scores.size)
    picked = np.argpartition(scores, -limit)[-limit:]
    return picked[np.argsort(scores[picked])[::-1]]


def top_finite_scores(scores: np.ndarray, limit: int) -> list[tuple[int, float]]:
    """Return up to `limit` (index, score) pairs for finite scores only.

    Masked rows set to -inf (or any non-finite value) are never ranked, so they
    can never leak into JSON serialization as NaN / -Infinity (which browsers
    reject). Used by dense search paths that mask out-of-scope rows.
    """
    if scores.size == 0 or limit <= 0:
        return []
    finite = np.isfinite(scores)
    if not finite.any():
        return []
    idx = np.flatnonzero(finite)
    vals = scores[idx]
    order = np.argsort(vals)[::-1]
    top = idx[order[:limit]]
    return [(int(i), float(scores[i])) for i in top]


@dataclass(slots=True)
class Evidence:
    source: str
    video_id: str
    timestamp_ms: int
    score: float
    rank: int
    modality: str
    frame_id: str | None = None
    keyframe_id: int | None = None
    file_name: str | None = None
    text: str | None = None
    asr_context: list[dict[str, Any]] = field(default_factory=list)
    scene: str | None = None
    ocr: list[str] = field(default_factory=list)
    objects: list[dict[str, Any]] = field(default_factory=list)
    watch_url: str | None = None
    title: str | None = None
    exact_btc: bool = False
    contribution: float = 0.0

    def public(self) -> dict[str, Any]:
        row = asdict(self)
        row["score"] = round(float(self.score), 6)
        row["contribution"] = round(float(self.contribution), 8)
        return row


@dataclass(slots=True)
class SourceState:
    name: str
    role: str
    configured: bool
    loaded: bool = False
    records: int | None = None
    error: str | None = None

    def public(self) -> dict[str, Any]:
        return asdict(self)


class BM25Index:
    """Small in-memory BM25 index for CPU lexical evidence."""

    def __init__(self, texts: Iterable[str]) -> None:
        self.documents = [tokens(text) for text in texts]
        self.lengths = np.asarray([len(document) for document in self.documents], dtype=np.float32)
        self.average_length = float(self.lengths.mean()) if len(self.lengths) else 1.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for document_id, document in enumerate(self.documents):
            for term, frequency in Counter(document).items():
                self.postings[term].append((document_id, frequency))

    def search(self, query: str, limit: int) -> list[tuple[int, float]]:
        query_terms = list(dict.fromkeys(tokens(query)))
        if not query_terms or not self.documents:
            return []
        scores = np.zeros(len(self.documents), dtype=np.float32)
        k1, b = 1.2, 0.75
        for term in query_terms:
            posting = self.postings.get(term)
            if not posting:
                continue
            document_frequency = len(posting)
            inverse_frequency = math.log(1.0 + (len(self.documents) - document_frequency + 0.5) / (document_frequency + 0.5))
            for document_id, frequency in posting:
                denominator = frequency + k1 * (
                    1.0 - b + b * float(self.lengths[document_id]) / max(self.average_length, 1.0)
                )
                scores[document_id] += inverse_frequency * frequency * (k1 + 1.0) / denominator
        return [(int(index), float(scores[index])) for index in top_indices(scores, limit) if scores[index] > 0]


def character_ngrams(value: str) -> set[str]:
    """Word-boundary character n-grams resilient to OCR errors and lost accents."""

    output: set[str] = set()
    for word in tokens(value):
        padded = f"^{word}$"
        for size in (3, 4, 5):
            if len(padded) <= size:
                output.add(padded)
            else:
                output.update(padded[index : index + size] for index in range(len(padded) - size + 1))
    return output


OCR_STOPWORDS = {
    "cho", "cua", "dang", "duoc", "giua", "khong", "mot", "nguoi", "nhung",
    "phia", "sau", "theo", "tren", "trong", "truoc", "voi",
}


def ocr_match_quality(query: str, document: str) -> float:
    """Require a phrase match or a strong distinctive-token match."""

    query_grams = character_ngrams(query)
    document_grams = character_ngrams(document)
    if not query_grams or not document_grams:
        return 0.0
    overall = len(query_grams & document_grams) / len(query_grams)
    distinctive = 0.0
    for word in tokens(query):
        if word in OCR_STOPWORDS or (len(word) < 5 and not word.isdigit()):
            continue
        word_grams = character_ngrams(word)
        if word_grams:
            distinctive = max(distinctive, len(word_grams & document_grams) / len(word_grams))
    if overall >= 0.35:
        return overall
    if distinctive >= 0.85:
        return 0.5 + distinctive * 0.25
    return 0.0


class CharacterNgramIndex:
    """Compact inverted character index for noisy Vietnamese OCR."""

    def __init__(self, texts: Iterable[str]) -> None:
        self.postings: dict[str, array] = {}
        lengths: list[int] = []
        for document_id, text in enumerate(texts):
            grams = character_ngrams(text)
            lengths.append(len(grams))
            for gram in grams:
                self.postings.setdefault(gram, array("I")).append(document_id)
        self.lengths = np.asarray(lengths, dtype=np.float32)

    def search(self, query: str, limit: int) -> list[tuple[int, float]]:
        query_grams = character_ngrams(query)
        if not query_grams or not self.lengths.size:
            return []
        scores: dict[int, float] = defaultdict(float)
        matched: dict[int, int] = defaultdict(int)
        document_count = int(self.lengths.size)
        for gram in query_grams:
            posting = self.postings.get(gram)
            if not posting:
                continue
            weight = math.log1p(document_count / len(posting))
            for document_id in posting:
                scores[document_id] += weight
                matched[document_id] += 1
        ranked = sorted(
            (
                (
                    document_id,
                    score
                    * (matched[document_id] / max(len(query_grams), 1))
                    / math.sqrt(max(float(self.lengths[document_id]), 1.0)),
                )
                for document_id, score in scores.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[:limit]


class JsonlOffsets:
    """Random-access JSONL reader without retaining a large metadata file in RAM."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offsets: np.ndarray | None = None

    def load(self) -> None:
        if self.offsets is not None:
            return
        offsets: list[int] = []
        with self.path.open("rb") as handle:
            while True:
                start = handle.tell()
                line = handle.readline()
                if not line:
                    break
                offsets.append(start)
        self.offsets = np.asarray(offsets, dtype=np.uint64)

    def __len__(self) -> int:
        self.load()
        assert self.offsets is not None
        return int(self.offsets.size)

    def rows(self, indexes: Iterable[int]) -> list[dict[str, Any]]:
        self.load()
        assert self.offsets is not None
        output = []
        with self.path.open("rb") as handle:
            for index in indexes:
                if index < 0 or index >= self.offsets.size:
                    output.append({})
                    continue
                handle.seek(int(self.offsets[index]))
                output.append(json.loads(handle.readline().decode("utf-8")))
        return output


class ModelHub:
    """Resident query encoders sized for an 8-GB GPU."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._sentence_models: dict[str, Any] = {}
        self._sentence_devices: dict[str, str] = {}
        self._model_lock = threading.RLock()
        self._siglip_model: Any = None
        self._siglip_processor: Any = None

    def _device(self) -> str:
        import torch

        requested = self.settings.device
        if requested not in {"auto", "cpu", "cuda"}:
            raise ValueError("AIC_DEVICE must be auto, cpu, or cuda")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("AIC_DEVICE=cuda was requested, but CUDA is unavailable")
        return "cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu"

    @staticmethod
    def _release(model: Any) -> None:
        import gc
        import torch

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _new_sentence_model(self, model_id: str, device: str) -> Any:
        import torch
        from sentence_transformers import SentenceTransformer

        kwargs: dict[str, Any] = {
            "model_name_or_path": model_id,
            "device": device,
            "trust_remote_code": True,
            "local_files_only": self.settings.offline_models,
        }
        if device == "cuda":
            kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
        try:
            model = SentenceTransformer(**kwargs)
        except TypeError:
            kwargs.pop("local_files_only", None)
            kwargs.pop("model_kwargs", None)
            model = SentenceTransformer(**kwargs)
        model.eval()
        return model

    def _sentence_model(self, model_id: str, *, cpu_fallback: bool = False) -> Any:
        import gc
        import torch

        with self._model_lock:
            if model_id in self._sentence_models:
                return self._sentence_models[model_id]
            device = self._device()
            try:
                model = self._new_sentence_model(model_id, device)
            except (RuntimeError, torch.OutOfMemoryError):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if device != "cuda" or not cpu_fallback:
                    raise
                model = self._new_sentence_model(model_id, "cpu")
                device = "cpu"
            if self.settings.resident_text_models:
                self._sentence_models[model_id] = model
                self._sentence_devices[model_id] = device
            return model

    def sentence_vector(
        self,
        model_id: str,
        query: str,
        prompt: str | None = None,
        *,
        cpu_fallback: bool = False,
    ) -> np.ndarray:
        model = self._sentence_model(model_id, cpu_fallback=cpu_fallback)
        encode_kwargs: dict[str, Any] = {
            "batch_size": 1,
            "convert_to_numpy": True,
            "normalize_embeddings": True,
            "show_progress_bar": False,
        }
        if prompt:
            encode_kwargs["prompt"] = prompt
        result = model.encode([query], **encode_kwargs)
        if not self.settings.resident_text_models:
            self._release(model)
        return np.asarray(result, dtype=np.float32).reshape(1, -1)

    def qwen_vector(self, query: str) -> np.ndarray:
        return self.sentence_vector(
            self.settings.qwen_model,
            query,
            prompt="Retrieve relevant images for the text query.",
        )

    def bge_vector(self, query: str) -> np.ndarray:
        return self.sentence_vector(self.settings.bge_model, query, cpu_fallback=True)

    def runtime_status(self) -> dict[str, Any]:
        status: dict[str, Any] = {
            "resident_enabled": self.settings.resident_text_models,
            "loaded": [
                {"model": model_id, "device": self._sentence_devices.get(model_id, "transient")}
                for model_id in self._sentence_models
            ],
            "siglip": {
                "loaded": self._siglip_model is not None,
                "device": self.settings.siglip_device,
            },
        }
        try:
            import torch

            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                status["cuda"] = {
                    "device": torch.cuda.get_device_name(0),
                    "allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
                    "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 3),
                    "free_gib": round(free / 2**30, 3),
                    "total_gib": round(total / 2**30, 3),
                }
        except Exception:
            pass
        return status

    def clip_vector(self, query: str) -> np.ndarray:
        import open_clip
        import torch
        import torch.nn.functional as functional

        device = self._device()
        model, _, _ = open_clip.create_model_and_transforms(
            self.settings.clip_model,
            pretrained=self.settings.clip_pretrained,
            device=device,
        )
        tokenizer = open_clip.get_tokenizer(self.settings.clip_model)
        model.eval()
        try:
            tokenized = tokenizer([query]).to(device)
            with torch.inference_mode():
                vector = functional.normalize(model.encode_text(tokenized).float(), dim=-1).cpu().numpy()
        finally:
            self._release(model)
        return np.asarray(vector, dtype=np.float32).reshape(1, -1)

    def siglip_vector(self, query: str) -> np.ndarray:
        import torch
        import torch.nn.functional as functional
        from transformers import AutoModel, AutoProcessor

        device = self.settings.siglip_device
        if device not in {"cpu", "cuda"}:
            raise ValueError("AIC_SIGLIP_DEVICE must be cpu or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("AIC_SIGLIP_DEVICE=cuda but CUDA is unavailable")
        with self._model_lock:
            if self._siglip_processor is None:
                self._siglip_processor = AutoProcessor.from_pretrained(
                    self.settings.siglip_model,
                    use_fast=True,
                    local_files_only=self.settings.offline_models,
                )
            if self._siglip_model is None:
                self._siglip_model = AutoModel.from_pretrained(
                    self.settings.siglip_model,
                    local_files_only=self.settings.offline_models,
                ).to(device).eval()
            processor = self._siglip_processor
            model = self._siglip_model
        inputs = processor(
            text=[query], padding=True, truncation=True, max_length=64, return_tensors="pt"
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            vector = model.get_text_features(**inputs)
            if not isinstance(vector, torch.Tensor):
                vector = vector.pooler_output
            vector = functional.normalize(vector.float(), dim=-1).cpu().numpy()
        return np.asarray(vector, dtype=np.float32).reshape(1, -1)


class QwenRandomSource:
    name = "qwen3_random"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        configured = settings.enable_qwen and settings.qwen_index.is_file() and settings.qwen_metadata.is_file()
        self.state = SourceState(
            self.name,
            "Qwen3-VL-Embedding-2B random-frame recall; timestamp only, source JPG unavailable",
            configured,
        )
        self.index: Any = None
        self.metadata = JsonlOffsets(settings.qwen_metadata)

    def load(self) -> None:
        if self.index is not None:
            return
        import faiss

        self.index = faiss.read_index(str(self.settings.qwen_index))
        row_count = len(self.metadata)
        if self.index.ntotal != row_count:
            raise RuntimeError(f"Qwen index has {self.index.ntotal} vectors but metadata has {row_count} rows")
        if self.index.d != 2048:
            raise RuntimeError(f"Expected a 2048-d Qwen3 index, got {self.index.d}")
        self.state.loaded = True
        self.state.records = int(self.index.ntotal)

    def search(
        self, vector: np.ndarray, limit: int, restrict_video: str | None,
        restrict_family: str | None = None,
    ) -> list[Evidence]:
        self.load()
        if vector.shape[1] != self.index.d:
            raise RuntimeError(f"Qwen query dimension {vector.shape[1]} does not match index {self.index.d}")
        request_count = scope_candidate_limit(limit, self.index.ntotal, restrict_video, restrict_family)
        scores, indexes = self.index.search(np.ascontiguousarray(vector, dtype=np.float32), request_count)
        rows = self.metadata.rows(int(index) for index in indexes[0] if index >= 0)
        output: list[Evidence] = []
        for score, row in zip(scores[0], rows):
            video_id = str(row.get("video_id", ""))
            if not video_id or not video_matches_scope(video_id, restrict_video, restrict_family):
                continue
            output.append(
                Evidence(
                    source=self.name,
                    video_id=video_id,
                    timestamp_ms=int(row.get("frame_ms", 0)),
                    score=float(score),
                    rank=len(output) + 1,
                    modality="visual",
                    frame_id=str(row.get("frame_id")) if row.get("frame_id") is not None else None,
                    file_name=row.get("file_name"),
                    watch_url=row.get("watch_url"),
                    title=row.get("title"),
                    exact_btc=False,
                )
            )
            if len(output) >= limit:
                break
        return output


def _representative_text(row: dict[str, Any]) -> str:
    entity_parts: list[str] = []
    for entity in row.get("entities") or []:
        entity_parts.append(str(entity.get("label", "")))
        entity_parts.extend(str(value) for value in entity.get("attributes") or [])
        entity_parts.extend(str(value) for value in entity.get("actions") or [])
    return " ".join(
        [str(row.get("caption_en", "")), *(str(value) for value in row.get("visible_text") or []), *entity_parts]
    ).strip()


class RepresentativeSource:
    name = "representative"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.state = SourceState(
            self.name,
            "BTC exact keyframes selected as representatives and captioned by Qwen2.5-VL",
            settings.representative_captions.is_file(),
        )
        self.rows: list[dict[str, Any]] = []
        self.lexical: BM25Index | None = None
        self.embeddings: np.ndarray | None = None

    def load(self) -> None:
        if self.rows:
            return
        with self.settings.representative_captions.open(encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]
        self.lexical = BM25Index(_representative_text(row) for row in self.rows)
        if self.settings.representative_embeddings.is_file():
            embeddings = np.load(self.settings.representative_embeddings, mmap_mode="r")
            if embeddings.shape[0] != len(self.rows) or embeddings.shape[1] != 1024:
                raise RuntimeError("Representative BGE-M3 embeddings must be [caption rows, 1024]")
            self.embeddings = embeddings
        self.state.loaded = True
        self.state.records = len(self.rows)

    def _evidence(self, row_index: int, score: float, rank: int, source: str) -> Evidence:
        row = self.rows[row_index]
        return Evidence(
            source=source,
            video_id=str(row["video_id"]),
            timestamp_ms=round(float(row["pts_time"]) * 1000),
            score=score,
            rank=rank,
            modality="visual",
            keyframe_id=int(row["keyframe_id"]),
            file_name=Path(str(row.get("frame_uri", ""))).name or None,
            text=_representative_text(row),
            scene=row.get("caption_en"),
            ocr=[str(value) for value in row.get("visible_text") or []],
            objects=list(row.get("entities") or []),
            exact_btc=True,
        )

    def lexical_search(
        self, query: str, limit: int, restrict_video: str | None,
        restrict_family: str | None = None,
    ) -> list[Evidence]:
        self.load()
        assert self.lexical is not None
        output = []
        for row_index, score in self.lexical.search(
            query, scope_candidate_limit(limit, None, restrict_video, restrict_family)
        ):
            if not video_matches_scope(str(self.rows[row_index]["video_id"]), restrict_video, restrict_family):
                continue
            output.append(self._evidence(row_index, score, len(output) + 1, "representative_lexical"))
            if len(output) >= limit:
                break
        return output

    def dense_search(
        self, vector: np.ndarray, limit: int, restrict_video: str | None,
        restrict_family: str | None = None,
    ) -> list[Evidence]:
        self.load()
        if self.embeddings is None:
            return []
        if vector.shape[1] != self.embeddings.shape[1]:
            raise RuntimeError("Representative query/index dimension mismatch")
        scores = np.empty(self.embeddings.shape[0], dtype=np.float32)
        for start in range(0, len(scores), 4096):
            stop = min(start + 4096, len(scores))
            scores[start:stop] = np.asarray(self.embeddings[start:stop], dtype=np.float32) @ vector[0]
        if restrict_video or restrict_family:
            for row_index, row in enumerate(self.rows):
                if not video_matches_scope(str(row["video_id"]), restrict_video, restrict_family):
                    scores[row_index] = -np.inf
        output = []
        for row_index, score in top_finite_scores(scores, limit):
            output.append(self._evidence(row_index, score, len(output) + 1, "representative_dense"))
            if len(output) >= limit:
                break
        return output


class ASRSource:
    name = "asr"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        configured = settings.asr_records.is_file()
        self.state = SourceState(self.name, "Precomputed Vietnamese ASR chunks: BGE-M3 dense + CPU BM25", configured)
        self.rows: list[dict[str, Any]] = []
        self.video_rows: dict[str, list[int]] = {}
        self.lexical: BM25Index | None = None
        self.embeddings: np.ndarray | None = None

    def load(self) -> None:
        if self.rows:
            return
        import pandas as pd

        frame = pd.read_parquet(
            self.settings.asr_records,
            columns=["vector_row", "video_id", "start_sec", "end_sec", "text_raw"],
        ).sort_values("vector_row")
        self.rows = frame.to_dict("records")
        grouped_rows: dict[str, list[int]] = defaultdict(list)
        for row_index, row in enumerate(self.rows):
            grouped_rows[str(row["video_id"])].append(row_index)
        self.video_rows = {
            video_id: sorted(
                row_indexes,
                key=lambda index: (float(self.rows[index]["start_sec"]), float(self.rows[index]["end_sec"])),
            )
            for video_id, row_indexes in grouped_rows.items()
        }
        self.lexical = BM25Index(str(row.get("text_raw", "")) for row in self.rows)
        if self.settings.asr_embeddings.is_file():
            embeddings = np.load(self.settings.asr_embeddings, mmap_mode="r")
            if embeddings.shape != (len(self.rows), 1024):
                raise RuntimeError(f"ASR embedding shape {embeddings.shape} does not match ({len(self.rows)}, 1024)")
            self.embeddings = embeddings
        self.state.loaded = True
        self.state.records = len(self.rows)

    def context(self, row_index: int, radius: int = 2) -> list[dict[str, Any]]:
        """Return neighboring chunks from the same video, marking the retrieval hit."""
        row = self.rows[row_index]
        row_indexes = self.video_rows.get(str(row["video_id"]), [])
        try:
            position = row_indexes.index(row_index)
        except ValueError:
            return []
        start = max(0, position - radius)
        stop = min(len(row_indexes), position + radius + 1)
        output = []
        for neighbor_index in row_indexes[start:stop]:
            neighbor = self.rows[neighbor_index]
            relation = "match" if neighbor_index == row_index else ("before" if neighbor_index < row_index else "after")
            output.append(
                {
                    "relation": relation,
                    "start_ms": round(float(neighbor["start_sec"]) * 1000),
                    "end_ms": round(float(neighbor["end_sec"]) * 1000),
                    "text": str(neighbor.get("text_raw", "")),
                }
            )
        return output

    def _evidence(self, row_index: int, score: float, rank: int, source: str) -> Evidence:
        row = self.rows[row_index]
        return Evidence(
            source=source,
            video_id=str(row["video_id"]),
            timestamp_ms=round((float(row["start_sec"]) + float(row["end_sec"])) * 500),
            score=score,
            rank=rank,
            modality="spoken",
            text=str(row.get("text_raw", "")),
            asr_context=self.context(row_index),
        )

    def lexical_search(
        self, query: str, limit: int, restrict_video: str | None,
        restrict_family: str | None = None,
    ) -> list[Evidence]:
        self.load()
        assert self.lexical is not None
        output = []
        for row_index, score in self.lexical.search(
            query, scope_candidate_limit(limit, None, restrict_video, restrict_family)
        ):
            if not video_matches_scope(str(self.rows[row_index]["video_id"]), restrict_video, restrict_family):
                continue
            output.append(self._evidence(row_index, score, len(output) + 1, "asr_lexical"))
            if len(output) >= limit:
                break
        return output

    def dense_search(
        self, vector: np.ndarray, limit: int, restrict_video: str | None,
        restrict_family: str | None = None,
    ) -> list[Evidence]:
        self.load()
        if self.embeddings is None:
            return []
        if vector.shape[1] != self.embeddings.shape[1]:
            raise RuntimeError("ASR query/index dimension mismatch")
        scores = np.empty(self.embeddings.shape[0], dtype=np.float32)
        for start in range(0, len(scores), 4096):
            stop = min(start + 4096, len(scores))
            scores[start:stop] = np.asarray(self.embeddings[start:stop], dtype=np.float32) @ vector[0]
        if restrict_video or restrict_family:
            for row_index, row in enumerate(self.rows):
                if not video_matches_scope(str(row["video_id"]), restrict_video, restrict_family):
                    scores[row_index] = -np.inf
        output = []
        for row_index, score in top_finite_scores(scores, limit):
            output.append(self._evidence(row_index, score, len(output) + 1, "asr_dense"))
            if len(output) >= limit:
                break
        return output


class BTCClipSource:
    name = "btc_clip"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        configured = (
            settings.enable_btc_clip
            and settings.btc_clip_features.is_dir()
            and settings.map_dir.is_dir()
        )
        self.state = SourceState(
            self.name,
            f"BTC 512-D CLIP keyframes: {settings.clip_model}/{settings.clip_pretrained} (empirically verified)",
            configured,
        )
        self.index: Any = None
        self.rows: list[dict[str, Any]] = []

    def _map_rows(self, video_id: str) -> list[dict[str, Any]]:
        path = self.settings.map_dir / f"{video_id}.csv"
        if not path.is_file():
            return []
        output = []
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                output.append(
                    {
                        "video_id": video_id,
                        "timestamp_ms": round(float(row["pts_time"]) * 1000),
                        "frame_id": str(row["frame_idx"]),
                        "keyframe_id": int(row["n"]),
                    }
                )
        return output

    def load(self) -> None:
        if self.index is not None:
            return
        import faiss

        index = faiss.IndexFlatIP(512)
        metadata: list[dict[str, Any]] = []
        for path in sorted(self.settings.btc_clip_features.glob("*.npy")):
            video_id = path.stem.upper()
            mapped = self._map_rows(video_id)
            if not mapped:
                continue
            vectors = np.asarray(np.load(path), dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[1] != 512 or vectors.shape[0] != len(mapped):
                raise RuntimeError(
                    f"BTC CLIP/map mismatch for {video_id}: vectors={vectors.shape}, map={len(mapped)}"
                )
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors /= np.maximum(norms, 1e-12)
            index.add(np.ascontiguousarray(vectors))
            metadata.extend(mapped)
        if not metadata:
            raise RuntimeError("No valid BTC CLIP feature/map pairs found")
        self.index = index
        self.rows = metadata
        self.state.loaded = True
        self.state.records = len(metadata)

    def search(
        self,
        vector: np.ndarray,
        limit: int,
        restrict_video: str | None,
        restrict_family: str | None = None,
        exclude_videos: set[str] | None = None,
    ) -> list[Evidence]:
        self.load()
        if vector.shape[1] != 512:
            raise RuntimeError(f"BTC CLIP query dimension must be 512, got {vector.shape[1]}")
        excluded = exclude_videos or set()
        count = self.index.ntotal if excluded else scope_candidate_limit(
            limit, self.index.ntotal, restrict_video, restrict_family
        )
        scores, indexes = self.index.search(np.ascontiguousarray(vector, dtype=np.float32), count)
        output: list[Evidence] = []
        for score, index in zip(scores[0], indexes[0]):
            if index < 0:
                continue
            row = self.rows[int(index)]
            if not video_matches_scope(row["video_id"], restrict_video, restrict_family):
                continue
            if row["video_id"] in excluded:
                continue
            output.append(
                Evidence(
                    source=self.name,
                    video_id=row["video_id"],
                    timestamp_ms=int(row["timestamp_ms"]),
                    score=float(score),
                    rank=len(output) + 1,
                    modality="visual",
                    frame_id=row["frame_id"],
                    keyframe_id=row["keyframe_id"],
                    exact_btc=True,
                )
            )
            if len(output) >= limit:
                break
        return output


class BTCOCRSource:
    name = "btc_ocr"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        configured = settings.ocr_dir.is_dir() and any(settings.ocr_dir.rglob("*.jsonl"))
        self.state = SourceState(
            self.name,
            "BTC keyframe OCR (unknown detector/recognizer): accent-folded character 3-5 grams",
            configured,
        )
        self.rows: list[dict[str, Any]] = []
        self.ngrams: CharacterNgramIndex | None = None

    def _frame_map(self, video_id: str) -> dict[int, dict[str, Any]]:
        path = self.settings.map_dir / f"{video_id}.csv"
        output: dict[int, dict[str, Any]] = {}
        if not path.is_file():
            return output
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                timestamp_ms = round(float(row["pts_time"]) * 1000)
                output[timestamp_ms] = {
                    "keyframe_id": int(row["n"]),
                    "frame_id": str(row["frame_idx"]),
                }
        return output

    def load(self) -> None:
        if self.ngrams is not None:
            return
        rows: list[dict[str, Any]] = []
        for path in sorted(self.settings.ocr_dir.rglob("*.jsonl")):
            video_id = path.stem.upper()
            if not re.fullmatch(r"L\d{2}_V\d{3}", video_id):
                continue
            grouped: dict[int, list[str]] = defaultdict(list)
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    text = str(record.get("text") or "").strip()
                    if text and float(record.get("conf", 1.0)) >= 0.25:
                        grouped[int(record.get("ts_ms", 0))].append(text)
            frame_map = self._frame_map(video_id)
            for timestamp_ms, values in grouped.items():
                # Remove duplicate OCR strings at the same timestamp.
                originals = list(dict.fromkeys(values))
                joined = " ".join(originals)
                if not joined:
                    continue
                mapped = frame_map.get(timestamp_ms, {})
                rows.append(
                    {
                        "video_id": video_id,
                        "timestamp_ms": timestamp_ms,
                        "frame_id": mapped.get("frame_id"),
                        "keyframe_id": mapped.get("keyframe_id"),
                        "text": joined,
                        "ocr": originals,
                    }
                )
        self.rows = rows
        self.ngrams = CharacterNgramIndex(row["text"] for row in rows)
        self.state.loaded = True
        self.state.records = len(rows)

    def search(
        self, query: str, limit: int, restrict_video: str | None,
        restrict_family: str | None = None,
    ) -> list[Evidence]:
        self.load()
        assert self.ngrams is not None
        request_count = scope_candidate_limit(
            limit, None, restrict_video, restrict_family, unfiltered_multiplier=30
        )
        output: list[Evidence] = []
        for row_index, score in self.ngrams.search(query, request_count):
            row = self.rows[row_index]
            if not video_matches_scope(row["video_id"], restrict_video, restrict_family):
                continue
            quality = ocr_match_quality(query, row["text"])
            if quality <= 0:
                continue
            output.append(
                Evidence(
                    source="btc_ocr_ngram",
                    video_id=row["video_id"],
                    timestamp_ms=int(row["timestamp_ms"]),
                    score=float(score * quality),
                    rank=len(output) + 1,
                    modality="visual",
                    frame_id=row.get("frame_id"),
                    keyframe_id=row.get("keyframe_id"),
                    text=row["text"],
                    ocr=list(row["ocr"]),
                    exact_btc=True,
                )
            )
            if len(output) >= limit:
                break
        return output


class FiveFPSVisualSource:
    name = "siglip_5fps"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        configured = (
            settings.enable_five_fps
            and settings.five_fps_embedding_dir.is_dir()
            and settings.five_fps_image_dir.is_dir()
            and settings.five_fps_map_dir.is_dir()
        )
        self.state = SourceState(
            self.name,
            "Family-scoped dense 5-FPS SigLIP retrieval for locally preprocessed videos",
            configured,
        )
        self.frames: dict[str, list[dict[str, Any]]] = {}
        self.flat_frames: list[dict[str, Any]] = []
        self.index: Any = None

    def load(self) -> None:
        if self.frames:
            return
        import csv

        total = 0
        for map_path in sorted(self.settings.five_fps_map_dir.glob("*_map.csv")):
            video_id = map_path.stem.removesuffix("_map")
            rows = []
            with map_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    frame_id = str(row["FrameID"])
                    embedding = self.settings.five_fps_embedding_dir / video_id / f"keyframe_{frame_id}.pt"
                    image = self.settings.five_fps_image_dir / video_id / f"keyframe_{frame_id}.webp"
                    if embedding.is_file() and image.is_file():
                        rows.append(
                            {
                                "video_id": video_id,
                                "frame_id": frame_id,
                                "timestamp_ms": round(float(row["Seconds"]) * 1000),
                                "embedding": embedding,
                                "image": image,
                            }
                        )
            if rows:
                self.frames[video_id] = rows
                total += len(rows)
        if not total:
            raise RuntimeError("No matching 5-FPS embeddings/images/maps were found")
        self.state.loaded = True
        self.state.records = total

    @property
    def covered_videos(self) -> set[str]:
        self.load()
        return set(self.frames)

    def has_scope(self, restrict_video: str | None, restrict_family: str | None) -> bool:
        self.load()
        return any(
            video_matches_scope(video_id, restrict_video, restrict_family)
            for video_id in self.frames
        )

    def build_index(self) -> None:
        if self.index is not None:
            return
        import faiss
        import torch

        self.load()
        flat_frames = [frame for video_id in sorted(self.frames) for frame in self.frames[video_id]]
        cache_root = self.settings.cache_dir / "indexes"
        cache_root.mkdir(parents=True, exist_ok=True)
        index_path = cache_root / "siglip_5fps_flat_ip.faiss"
        manifest_path = cache_root / "siglip_5fps_manifest.json"
        latest_source_ns = max(
            int(frame["embedding"].stat().st_mtime_ns) for frame in flat_frames
        )
        expected_manifest = {
            "records": len(flat_frames),
            "dimension": 1152,
            "model": self.settings.siglip_model,
            "latest_source_ns": latest_source_ns,
        }
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            cached = faiss.read_index(str(index_path))
            if manifest == expected_manifest and cached.d == 1152 and cached.ntotal == len(flat_frames):
                self.index = cached
                self.flat_frames = flat_frames
                return
        except (OSError, ValueError, RuntimeError):
            pass

        def load_vector(frame: dict[str, Any]) -> np.ndarray:
            value = torch.load(frame["embedding"], map_location="cpu", weights_only=True)
            vector = value.detach().float().numpy().reshape(-1).astype(np.float32, copy=False)
            if vector.size != 1152:
                raise RuntimeError(f"5-FPS vector has {vector.size} dimensions, expected 1152")
            vector /= max(float(np.linalg.norm(vector)), 1e-12)
            return vector

        # Read per-frame tensors in parallel; later runs load the FAISS cache.
        with ThreadPoolExecutor(max_workers=min(16, max(4, (os.cpu_count() or 4)))) as executor:
            vectors = np.stack(list(executor.map(load_vector, flat_frames)))
        index = faiss.IndexFlatIP(1152)
        index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        if index.ntotal != len(flat_frames):
            raise RuntimeError("5-FPS index and metadata row counts differ")
        temporary = index_path.with_suffix(".tmp.faiss")
        faiss.write_index(index, str(temporary))
        temporary.replace(index_path)
        manifest_path.write_text(json.dumps(expected_manifest, indent=2), encoding="utf-8")
        self.index = index
        self.flat_frames = flat_frames

    def global_search(
        self, vector: np.ndarray, limit: int, restrict_video: str | None,
        restrict_family: str | None = None,
    ) -> list[Evidence]:
        self.build_index()
        if vector.shape[1] != self.index.d:
            raise RuntimeError(
                f"5-FPS query dimension {vector.shape[1]} does not match index {self.index.d}"
            )
        count = scope_candidate_limit(limit, self.index.ntotal, restrict_video, restrict_family)
        scores, indexes = self.index.search(np.ascontiguousarray(vector, dtype=np.float32), count)
        output: list[Evidence] = []
        for score, index in zip(scores[0], indexes[0]):
            if index < 0:
                continue
            frame = self.flat_frames[int(index)]
            if not video_matches_scope(frame["video_id"], restrict_video, restrict_family):
                continue
            output.append(
                Evidence(
                    source=self.name,
                    video_id=str(frame["video_id"]),
                    timestamp_ms=int(frame["timestamp_ms"]),
                    score=float(score),
                    rank=len(output) + 1,
                    modality="visual",
                    frame_id=str(frame["frame_id"]),
                    file_name=Path(frame["image"]).name,
                )
            )
            if len(output) >= limit:
                break
        return output

def source_family(source: str) -> str:
    if source.startswith("asr_"):
        return "asr"
    if source.startswith("representative_") or source == "representative":
        return "representative"
    if source.startswith("btc_ocr"):
        return "btc_ocr"
    return source


def cluster_evidence(evidence: list[Evidence], radius_ms: int, rrf_k: int) -> list[dict[str, Any]]:
    by_video: dict[str, list[Evidence]] = defaultdict(list)
    for item in evidence:
        item.contribution = 1.0 / (rrf_k + item.rank)
        by_video[item.video_id].append(item)
    events: list[dict[str, Any]] = []
    for video_id, rows in by_video.items():
        clusters: list[list[Evidence]] = []
        for row in sorted(rows, key=lambda item: item.timestamp_ms):
            if clusters:
                center = round(
                    sum(item.timestamp_ms * item.contribution for item in clusters[-1])
                    / sum(item.contribution for item in clusters[-1])
                )
            else:
                center = -10**18
            if clusters and abs(row.timestamp_ms - center) <= radius_ms:
                clusters[-1].append(row)
            else:
                clusters.append([row])
        for cluster in clusters:
            best_by_family: dict[str, Evidence] = {}
            support_by_family: dict[str, set[str]] = defaultdict(set)
            for row in cluster:
                family = source_family(row.source)
                support_by_family[family].add(row.source)
                if family not in best_by_family or row.contribution > best_by_family[family].contribution:
                    best_by_family[family] = row
            kept = list(best_by_family.values())
            base_score = sum(item.contribution for item in kept)
            modalities = {item.modality for item in kept}
            secondary_support = sum(max(0, len(values) - 1) for values in support_by_family.values())
            agreement_bonus = (
                0.002 * max(0, len(kept) - 1)
                + 0.0005 * secondary_support
                + (0.003 if len(modalities) > 1 else 0.0)
            )
            # Keep visual anchors precise when spoken evidence spans an interval.
            visual_anchors = [item for item in kept if item.modality == "visual"]
            anchor = max(visual_anchors or kept, key=lambda item: item.contribution)
            timestamp_ms = int(anchor.timestamp_ms)
            events.append(
                {
                    "video_id": video_id,
                    "timestamp_ms": timestamp_ms,
                    "anchor_source": anchor.source,
                    "score": base_score + agreement_bonus,
                    "evidence": sorted(kept, key=lambda item: item.contribution, reverse=True),
                }
            )
    events.sort(key=lambda event: event["score"], reverse=True)
    return events


def temporal_nms(events: list[dict[str, Any]], threshold_ms: int, limit: int) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for event in events:
        if any(
            prior["video_id"] == event["video_id"]
            and abs(int(prior["timestamp_ms"]) - int(event["timestamp_ms"])) <= threshold_ms
            for prior in kept
        ):
            continue
        kept.append(event)
        if len(kept) >= limit:
            break
    return kept


class RetrievalPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.models = ModelHub(settings)
        self.qwen = QwenRandomSource(settings)
        self.representative = RepresentativeSource(settings)
        self.asr = ASRSource(settings)
        self.ocr = BTCOCRSource(settings)
        self.clip = BTCClipSource(settings)
        self.five_fps = FiveFPSVisualSource(settings)
        self.l23_scenes = L23SceneStore(settings.l23_scenes_db)
        self.query_lock = threading.Lock()

    @property
    def states(self) -> list[SourceState]:
        return [
            self.clip.state,
            self.ocr.state,
            self.qwen.state,
            self.representative.state,
            self.asr.state,
            self.five_fps.state,
            SourceState(**self.l23_scenes.status()),
        ]

    def status(self) -> dict[str, Any]:
        return {
            "sources": [state.public() for state in self.states],
            "models": {
                "siglip": self.settings.siglip_model,
                "btc_clip": f"{self.settings.clip_model}/{self.settings.clip_pretrained}",
                "visual_translation": "disabled; optional English visual_query is user-supplied",
                "qwen3_embedding": self.settings.qwen_model,
                "representative_captioner_offline": "Qwen/Qwen2.5-VL-7B-Instruct",
                "text_dense": self.settings.bge_model,
                "runtime": self.models.runtime_status(),
            },
            "fusion": {
                "join_key": "video_id + timestamp_ms",
                "window_radius_ms": self.settings.window_radius_ms,
                "visual_priority": "global Qwen3-VL embedding; SigLIP and BTC CLIP disabled by default",
                "final_results": self.settings.final_results,
                "grading_cutoffs": [1, 5, 20, 50, 100],
            },
        }

    def warm_models(self) -> dict[str, Any]:
        """Load and exercise the active dense query encoders before serving traffic."""

        warnings: list[str] = []
        if self.settings.enable_qwen and self.qwen.state.configured:
            try:
                self.models.qwen_vector("warmup visual retrieval query")
            except Exception as error:
                warnings.append(f"Qwen warmup failed: {type(error).__name__}: {error}")
        if (
            self.settings.enable_dense_asr or self.settings.enable_dense_representative
        ) and (self.asr.state.configured or self.representative.state.configured):
            try:
                self.models.bge_vector("warmup semantic retrieval query")
            except Exception as error:
                warnings.append(f"BGE-M3 warmup failed: {type(error).__name__}: {error}")
        return {"models": self.models.runtime_status(), "warnings": warnings}

    @staticmethod
    def _attempt(label: str, operation: Any, warnings: list[str]) -> list[Evidence]:
        try:
            return operation()
        except Exception as error:
            warnings.append(f"{label} unavailable: {type(error).__name__}: {error}")
            return []

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        restrict_video: str | None = None,
        scope: str = "general",
        visual_query: str | None = None,
    ) -> dict[str, Any]:
        query = normalize_text(query)
        scope = normalize_text(scope)
        if not query:
            raise ValueError("Query is empty")
        if mode not in {"hybrid", "visual", "spoken"}:
            raise ValueError("mode must be hybrid, visual, or spoken")
        if scope not in SEARCH_SCOPES:
            raise ValueError("scope must be bike, general, or global")
        effective_family, route_reason = resolve_family_scope(scope, restrict_video)
        limit = self.settings.per_source_top_k
        warnings: list[str] = []
        evidence: list[Evidence] = []
        visual_text = normalize_text(visual_query or query)
        with self.query_lock:
            # Lexical branches remain available without neural query models.
            if mode in {"hybrid", "visual"} and self.representative.state.configured:
                evidence += self._attempt(
                    "Representative lexical search",
                    lambda: self.representative.lexical_search(
                        visual_text if visual_query else query, limit, restrict_video, effective_family
                    ),
                    warnings,
                )
            if mode in {"hybrid", "visual"} and self.ocr.state.configured:
                evidence += self._attempt(
                    "BTC OCR character n-gram search",
                    lambda: self.ocr.search(query, limit, restrict_video, effective_family),
                    warnings,
                )
            if mode in {"hybrid", "spoken"} and self.asr.state.configured:
                evidence += self._attempt(
                    "ASR lexical search",
                    lambda: self.asr.lexical_search(query, limit, restrict_video, effective_family),
                    warnings,
                )

            if mode in {"hybrid", "visual"} and self.qwen.state.configured:
                qwen_vector = self._attempt(
                    "Qwen3 text encoder", lambda: [self.models.qwen_vector(visual_text)], warnings
                )
                if qwen_vector:
                    evidence += self._attempt(
                        "Qwen3 random-frame search",
                        lambda: self.qwen.search(qwen_vector[0], limit, restrict_video, effective_family),
                        warnings,
                    )

            need_bge = mode in {"hybrid", "spoken"} and self.settings.enable_dense_asr and self.asr.state.configured
            need_bge = need_bge or (
                mode in {"hybrid", "visual"}
                and self.settings.enable_dense_representative
                and self.settings.representative_embeddings.is_file()
                and self.representative.state.configured
            )
            if need_bge:
                bge_vector = self._attempt("BGE-M3 text encoder", lambda: [self.models.bge_vector(query)], warnings)
                if bge_vector:
                    if mode in {"hybrid", "spoken"}:
                        evidence += self._attempt(
                            "ASR dense search",
                            lambda: self.asr.dense_search(
                                bge_vector[0], limit, restrict_video, effective_family
                            ),
                            warnings,
                        )
                    if (
                        mode in {"hybrid", "visual"}
                        and self.settings.enable_dense_representative
                        and self.settings.representative_embeddings.is_file()
                    ):
                        evidence += self._attempt(
                            "Representative dense search",
                            lambda: self.representative.dense_search(
                                bge_vector[0], limit, restrict_video, effective_family
                            ),
                            warnings,
                        )

            siglip_vector: np.ndarray | None = None
            need_siglip = (
                mode in {"hybrid", "visual"}
                and scope != "general"
                and self.five_fps.state.configured
                and self._attempt(
                    "5-FPS family coverage check",
                    lambda: [self.five_fps.has_scope(restrict_video, effective_family)],
                    warnings,
                ) == [True]
            )
            if need_siglip:
                encoded = self._attempt(
                    "SigLIP text encoder", lambda: [self.models.siglip_vector(visual_text)], warnings
                )
                if encoded:
                    siglip_vector = encoded[0]

            covered_videos: set[str] = set()
            if siglip_vector is not None and self.five_fps.state.configured:
                global_five_fps = self._attempt(
                    "Global 5-FPS SigLIP search",
                    lambda: self.five_fps.global_search(
                        siglip_vector, limit, restrict_video, effective_family
                    ),
                    warnings,
                )
                evidence.extend(global_five_fps)
                if self.five_fps.state.loaded:
                    covered_videos = set(self.five_fps.frames)

            clip_vector: np.ndarray | None = None
            should_search_clip = mode in {"hybrid", "visual"} and self.clip.state.configured
            if restrict_video and restrict_video in covered_videos:
                should_search_clip = False
            if should_search_clip:
                encoded = self._attempt(
                    "CLIP fallback text encoder", lambda: [self.models.clip_vector(visual_text)], warnings
                )
                if encoded:
                    clip_vector = encoded[0]
            if clip_vector is not None:
                evidence += self._attempt(
                    "BTC CLIP fallback search outside 5-FPS coverage",
                    lambda: self.clip.search(
                        clip_vector, limit, restrict_video, effective_family,
                        exclude_videos=covered_videos
                    ),
                    warnings,
                )
            events = cluster_evidence(evidence, self.settings.window_radius_ms, self.settings.rrf_k)
            constraints = parse_query_constraints(query)
            if self.settings.enable_l23_scene_rerank and scope in {"bike", "global"}:
                self.l23_scenes.rerank(events, constraints)
            events.sort(key=lambda event: float(event["score"]), reverse=True)

        results = temporal_nms(events, self.settings.temporal_nms_ms, self.settings.final_results)
        for rank, event in enumerate(results, start=1):
            event["rank"] = rank
            event["score"] = round(float(event["score"]), 8)
            event["evidence"] = [item.public() for item in sorted(event["evidence"], key=lambda x: x.contribution, reverse=True)]
        return {
            "query": query,
            "visual_query": visual_text,
            "visual_query_user_supplied": visual_query is not None,
            "mode": mode,
            "scope": {
                "name": scope,
                "video": restrict_video,
                "family": None if effective_family == GENERAL_SCOPE else effective_family,
                "excludes": ["L23"] if effective_family == GENERAL_SCOPE else [],
                "reason": route_reason,
            },
            "results": results,
            "warnings": warnings,
            "source_counts": dict(Counter(item.source for item in evidence)),
            "l23_constraints": {
                "top_colors": list(constraints.top_colors),
                "bottom_colors": list(constraints.bottom_colors),
                "shot_type": constraints.shot_type,
            },
        }
