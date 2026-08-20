"""Configuration and path discovery for the multimodal retrieval MVP."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent
# Use the historical workspace path locally; fall back to the checkout in containers.
_fallback_workspace = APP_ROOT.parents[2] if len(APP_ROOT.parents) > 2 else APP_ROOT
WORKSPACE_ROOT = Path(os.environ.get("AIC_WORKSPACE_ROOT", _fallback_workspace)).resolve()


def _path(env: str, default: Path) -> Path:
    return Path(os.environ.get(env, default)).expanduser().resolve()


def _paths(env: str, defaults: list[Path]) -> list[Path]:
    raw = os.environ.get(env)
    values = [Path(value) for value in raw.split(os.pathsep) if value] if raw else defaults
    return [value.expanduser().resolve() for value in values]


def _btc_image_roots() -> list[Path]:
    """Discover every downloaded BTC keyframe batch unless explicitly overridden."""

    raw = os.environ.get("AIC_BTC_IMAGE_ROOTS")
    if raw:
        return _paths("AIC_BTC_IMAGE_ROOTS", [])
    keyframes = WORKSPACE_ROOT / "Keyframes"
    discovered = sorted(
        path.resolve()
        for path in keyframes.glob("Keyframes_L*/keyframes")
        if path.is_dir()
    )
    return discovered or [
        (keyframes / "Keyframes_L21/keyframes").resolve(),
        (keyframes / "Keyframes_L27/keyframes").resolve(),
    ]


def _video_roots() -> list[Path]:
    raw = os.environ.get("AIC_VIDEO_ROOTS")
    if raw:
        return _paths("AIC_VIDEO_ROOTS", [])
    roots = [
        WORKSPACE_ROOT / "aic-nutbread/video",
        WORKSPACE_ROOT / "AIC-nonchalantUIT/video",  # old data bundle layout
    ]
    roots.extend(sorted(WORKSPACE_ROOT.glob("Bicycle-video/*/video")))
    return [path.resolve() for path in roots]


@dataclass(slots=True)
class Settings:
    app_root: Path = APP_ROOT
    workspace_root: Path = WORKSPACE_ROOT
    five_fps_data: Path = field(
        default_factory=lambda: _path("AIC_5FPS_DATA", APP_ROOT.parent / "data")
    )
    qwen_index: Path = field(
        default_factory=lambda: _path(
            "AIC_QWEN_INDEX",
            WORKSPACE_ROOT
            / "qwen-random/metadata/aic25_qwen_merged_search/index/qwen3vl_index_flat_ip.faiss",
        )
    )
    qwen_metadata: Path = field(
        default_factory=lambda: _path(
            "AIC_QWEN_METADATA",
            WORKSPACE_ROOT
            / "qwen-random/metadata/aic25_qwen_merged_search/index/qwen3vl_metadata.jsonl",
        )
    )
    asr_records: Path = field(
        default_factory=lambda: _path(
            "AIC_ASR_RECORDS", WORKSPACE_ROOT / "ASR/metadata/artifacts/text_dense/records.parquet"
        )
    )
    asr_embeddings: Path = field(
        default_factory=lambda: _path(
            "AIC_ASR_EMBEDDINGS", WORKSPACE_ROOT / "ASR/metadata/artifacts/text_dense/embeddings.npy"
        )
    )
    representative_captions: Path = field(
        default_factory=lambda: _path(
            "AIC_CAPTIONS", WORKSPACE_ROOT / "Hypersparse-Qwen2.5/captions/captions.jsonl"
        )
    )
    representative_embeddings: Path = field(
        default_factory=lambda: _path(
            "AIC_REPRESENTATIVE_EMBEDDINGS",
            APP_ROOT / "data/representative_bge_m3_embeddings.npy",
        )
    )
    map_dir: Path = field(
        default_factory=lambda: _path(
            "AIC_MAP_DIR", WORKSPACE_ROOT / "map-keyframes-aic25-b1/map-keyframes"
        )
    )
    media_info_dir: Path = field(
        default_factory=lambda: _path(
            "AIC_MEDIA_INFO_DIR", WORKSPACE_ROOT / "media-info-aic25-b1/media-info"
        )
    )
    ocr_dir: Path = field(
        default_factory=lambda: _path("AIC_OCR_DIR", WORKSPACE_ROOT / "OCR")
    )
    btc_clip_features: Path = field(
        default_factory=lambda: _path(
            "AIC_BTC_CLIP_FEATURES",
            WORKSPACE_ROOT / "clip-features-32-aic25-b1_extracted/clip-features-32",
        )
    )
    cache_dir: Path = field(
        default_factory=lambda: _path("AIC_CACHE_DIR", APP_ROOT / ".cache")
    )
    l23_scenes_db: Path = field(
        default_factory=lambda: _path("AIC_L23_SCENES_DB", APP_ROOT / "data/l23_scenes.sqlite")
    )
    btc_image_roots: list[Path] = field(
        default_factory=_btc_image_roots
    )
    video_roots: list[Path] = field(
        default_factory=_video_roots
    )
    siglip_model: str = field(
        default_factory=lambda: os.environ.get("SIGLIP_MODEL", "google/siglip-so400m-patch14-384")
    )
    qwen_model: str = field(
        default_factory=lambda: os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen3-VL-Embedding-2B")
    )
    bge_model: str = field(default_factory=lambda: os.environ.get("BGE_MODEL_ID", "BAAI/bge-m3"))
    clip_model: str = field(
        default_factory=lambda: os.environ.get("AIC_CLIP_MODEL", "ViT-B-32-quickgelu")
    )
    clip_pretrained: str = field(
        default_factory=lambda: os.environ.get("AIC_CLIP_PRETRAINED", "openai")
    )
    device: str = field(default_factory=lambda: os.environ.get("AIC_DEVICE", "auto").lower())
    global_top_k: int = field(default_factory=lambda: int(os.environ.get("AIC_GLOBAL_TOP_K", "250")))
    per_source_top_k: int = field(default_factory=lambda: int(os.environ.get("AIC_SOURCE_TOP_K", "100")))
    final_results: int = field(default_factory=lambda: int(os.environ.get("AIC_FINAL_RESULTS", "100")))
    window_radius_ms: int = field(default_factory=lambda: int(os.environ.get("AIC_WINDOW_RADIUS_MS", "10000")))
    temporal_nms_ms: int = field(default_factory=lambda: int(os.environ.get("AIC_TEMPORAL_NMS_MS", "3000")))
    preview_near_ms: int = field(default_factory=lambda: int(os.environ.get("AIC_PREVIEW_NEAR_MS", "10000")))
    rrf_k: int = field(default_factory=lambda: int(os.environ.get("AIC_RRF_K", "60")))
    enable_qwen: bool = field(
        default_factory=lambda: os.environ.get("AIC_ENABLE_QWEN", "1").lower() not in {"0", "false", "no"}
    )
    enable_btc_clip: bool = field(
        default_factory=lambda: os.environ.get("AIC_ENABLE_BTC_CLIP", "0").lower() not in {"0", "false", "no"}
    )
    enable_dense_asr: bool = field(
        default_factory=lambda: os.environ.get("AIC_ENABLE_DENSE_ASR", "1").lower() not in {"0", "false", "no"}
    )
    enable_dense_representative: bool = field(
        default_factory=lambda: os.environ.get("AIC_ENABLE_DENSE_REPRESENTATIVE", "1").lower()
        not in {"0", "false", "no"}
    )
    enable_five_fps: bool = field(
        default_factory=lambda: os.environ.get("AIC_ENABLE_5FPS", "1").lower() not in {"0", "false", "no"}
    )
    enable_l23_scene_rerank: bool = field(
        default_factory=lambda: os.environ.get("AIC_ENABLE_L23_SCENE_RERANK", "1").lower()
        not in {"0", "false", "no"}
    )
    siglip_device: str = field(
        default_factory=lambda: os.environ.get("AIC_SIGLIP_DEVICE", "cpu").lower()
    )
    resident_text_models: bool = field(
        default_factory=lambda: os.environ.get("AIC_RESIDENT_MODELS", "1").lower()
        not in {"0", "false", "no"}
    )
    preload_text_models: bool = field(
        default_factory=lambda: os.environ.get("AIC_PRELOAD_MODELS", "1").lower()
        not in {"0", "false", "no"}
    )
    offline_models: bool = field(
        default_factory=lambda: os.environ.get("AIC_OFFLINE_MODELS", "0").lower() in {"1", "true", "yes"}
    )

    @property
    def five_fps_embedding_dir(self) -> Path:
        return self.five_fps_data / "embedding"

    @property
    def five_fps_image_dir(self) -> Path:
        return self.five_fps_data / "keyframe"

    @property
    def five_fps_map_dir(self) -> Path:
        return self.five_fps_data / "maps"
