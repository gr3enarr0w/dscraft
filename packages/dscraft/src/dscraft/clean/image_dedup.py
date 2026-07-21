"""ONNX Runtime image embedding + near-duplicate detection (architecture doc Part 3,
"Module 2: LazyClean"), the image-modality counterpart of ``embeddings.py``/``dedup.py``.

**Why this file exists and what problem it resolves.** See
``docs/decisions/2026-07-image-dedup-evaluation.md`` for the full evaluation
this module implements the recommendation of. In short: `dscraft.clean` has
a hard, deliberate PyTorch-free constraint (ONNX Runtime only, to stay under
this package's ~100MB footprint target -- see ``embeddings.py``'s module
docstring), while CLIP-based image deduplication is conventionally
PyTorch-based (``open-clip-torch``). That evaluation found a Tier-1
(MIT-licensed-codebase), ONNX-exported, <100MB-class CLIP vision encoder
does exist -- ``Xenova/clip-vit-base-patch32``'s ``onnx/vision_model_int8.onnx``
(88.6MB, int8-quantized, re-exported from ``openai/clip-vit-base-patch32``,
whose own GitHub repository is MIT-licensed) -- so this capability belongs
in `dscraft.clean`, not `dscraft.vision`, exactly mirroring
``embeddings.py``'s existing "ONNX Runtime, never PyTorch" pattern rather
than requiring the PyTorch-based `dscraft.vision` stack.

Same hard constraint as ``embeddings.py``: embeddings are produced by
loading a ``.onnx`` model file directly via the ``onnxruntime`` Python
package. Do not import ``torch``, ``transformers``, or any CLIP-specific
Python package (e.g. ``open_clip``/``clip``) anywhere in this module.

**No Pillow dependency either.** Unlike a typical CLIP pipeline, this
module does not decode image files itself and does not depend on
``pillow`` (a `dscraft.vision`-specific dependency, not part of `clean`'s
footprint). Callers pass already-decoded ``(H, W, 3)`` ``uint8`` RGB numpy
arrays (e.g. from ``PIL.Image.open(path).convert("RGB")`` -> ``np.asarray``,
performed by the caller, or from `dscraft.vision`'s own preprocessing
pipeline if a caller already has one) -- see :func:`resize_and_normalize`
for the one preprocessing step this module does perform itself (a simple,
dependency-free nearest-neighbor resize + normalize, not a full CLIP
preprocessing pipeline).

Two embedding-model sources are provided, mirroring ``embeddings.py``
exactly:

1. :func:`build_synthetic_image_embedding_model` -- hand-builds a tiny ONNX
   graph on the fly (a linear projection + L2 normalization over a
   downsampled-pixel feature vector). Fully hermetic, used by tests and the
   example. **Not** a real CLIP model and not semantically meaningful --
   same scope boundary as ``build_synthetic_embedding_model`` in
   ``embeddings.py``.
2. :func:`download_recommended_clip_vision_model` -- documents (and, given
   network access, performs) the production wiring: fetching the Tier-1
   ``vision_model_int8.onnx`` checkpoint referenced in
   :data:`RECOMMENDED_IMAGE_MODEL_NAME`'s allowlist entry and caching it
   locally. Optional and lazy, exactly like
   ``embeddings.download_recommended_model`` -- never called by tests, the
   example, or any import-time code in this package.

:func:`detect_near_duplicate_images` is the one canonical entrypoint tying
this module's embedding path to ``dedup.py``'s existing, modality-agnostic
:func:`~dscraft.clean.dedup.find_near_duplicates` -- that function already
operates on any ``(n, dim)`` embedding array regardless of what produced it,
so this module does not duplicate any near-duplicate-detection logic, only
the image-specific embedding step.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from dscraft.core.licensing import ModelTier

from .dedup import DedupReport, find_near_duplicates
from .embeddings import MODEL_ALLOWLIST

__all__ = [
    "RECOMMENDED_IMAGE_MODEL_NAME",
    "ImageEmbeddingModel",
    "resize_and_normalize",
    "build_synthetic_image_embedding_onnx",
    "build_synthetic_image_embedding_model",
    "download_recommended_clip_vision_model",
    "detect_near_duplicate_images",
]

# ---------------------------------------------------------------------------
# Model licensing allowlist (architecture doc §2.10) -- registered into the
# SAME dscraft.clean.embeddings.MODEL_ALLOWLIST instance, not a second one:
# per dscraft.core.licensing's documented per-module (not per-file) ownership
# pattern, `clean` maintains exactly one allowlist for the whole subpackage.
# ---------------------------------------------------------------------------

RECOMMENDED_IMAGE_MODEL_NAME = "Xenova/clip-vit-base-patch32 (vision_model_int8.onnx)"

# A specific, pinned commit of the recommended checkpoint's HF repo -- see
# embeddings.py's _RECOMMENDED_MODEL_REVISION for why a pinned commit SHA
# (not "main") is required: "main" is a mutable branch ref that could be
# force-pushed to a different export at any time, silently invalidating the
# Tier-1/license review this module's allowlist entry documents.
_RECOMMENDED_IMAGE_MODEL_REVISION = "8557a67d0f43938c8628ee8db7b0f4fca8ecc603"

_RECOMMENDED_IMAGE_MODEL_ONNX_URL = (
    f"https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/"
    f"{_RECOMMENDED_IMAGE_MODEL_REVISION}/onnx/vision_model_int8.onnx"
)

MODEL_ALLOWLIST.register(
    name=RECOMMENDED_IMAGE_MODEL_NAME,
    tier=ModelTier.TIER_1,
    license_identifier="MIT (inherited; see notes)",
    notes=(
        "Recommended production checkpoint for "
        "dscraft.clean.image_dedup.ImageEmbeddingModel: an int8-quantized "
        "ONNX re-export of openai/clip-vit-base-patch32's vision tower, "
        "hosted at Xenova/clip-vit-base-patch32 (onnx/vision_model_int8.onnx, "
        "~88.6MB -- under this module's <100MB ONNX Runtime footprint "
        "target; the full fp32 vision_model.onnx is ~352MB and is NOT "
        "recommended). openai/CLIP's own GitHub repository (the code and, "
        "by long-standing community convention, the released pretrained "
        "weights) is MIT-licensed. Caveat, documented here per CLAUDE.md's "
        "'maintaining and re-verifying allowlists is an ongoing task' "
        "policy: OpenAI's Hugging Face model card for "
        "clip-vit-base-patch32 does not itself carry an explicit SPDX "
        "license tag, so this Tier-1 classification rests on the "
        "widely-held (but not HF-card-explicit) inheritance from the MIT-"
        "licensed openai/CLIP code repository, not a first-party SPDX "
        "declaration on the weights themselves -- re-verify before "
        "shipping this as a hard default in any downstream product, per "
        "the evaluation doc "
        "(docs/decisions/2026-07-image-dedup-evaluation.md). Not bundled "
        "with this package and not downloaded by default -- see "
        "download_recommended_clip_vision_model()."
    ),
)


# ---------------------------------------------------------------------------
# Preprocessing (dependency-free -- no Pillow, no torchvision)
# ---------------------------------------------------------------------------


def resize_and_normalize(image: np.ndarray, *, size: int = 8) -> np.ndarray:
    """Nearest-neighbor-resize ``image`` to ``(size, size, 3)`` and flatten to ``[0, 1]`` floats.

    This is the default preprocessor for the synthetic test/example image
    model -- the image-modality analogue of ``embeddings.py``'s
    ``hashing_bag_of_words_vectorizer``. It performs a simple, dependency-
    free nearest-neighbor downsample (via integer-strided NumPy indexing,
    no ``pillow``/``scipy``/``cv2``) and returns a flattened
    ``(size * size * 3,)`` ``float32`` vector normalized to ``[0.0, 1.0]``.
    It has no learned parameters and is not a real CLIP preprocessing
    pipeline (which additionally applies CLIP-specific mean/std
    normalization and bicubic resizing) -- intentionally simple, matching
    this package's existing scaffold depth (see ``dedup.py``'s documented
    naive-O(n^2) scope boundary).

    Args:
        image: an ``(H, W, 3)`` array (any numeric dtype; ``uint8`` RGB is
            the typical case). Grayscale ``(H, W)`` input is also accepted
            and is broadcast to 3 channels.
        size: the resized square side length. The returned vector has
            length ``size * size * 3``.

    Returns:
        A flattened ``(size * size * 3,)`` ``float32`` vector.

    Raises:
        ValueError: if ``image`` is not 2D or 3D, or its (only) trailing
            dimension is present but not exactly 3 channels.
    """
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(
            f"Expected an (H, W) or (H, W, 3) image array, got shape {arr.shape!r}."
        )

    height, width, _ = arr.shape
    row_idx = np.minimum((np.arange(size) * height) // size, height - 1)
    col_idx = np.minimum((np.arange(size) * width) // size, width - 1)
    resized = arr[np.ix_(row_idx, col_idx)]

    normalized = resized.astype(np.float32)
    if normalized.max() > 1.0:
        normalized = normalized / 255.0
    return normalized.reshape(-1)


# ---------------------------------------------------------------------------
# Synthetic ONNX graph builder (hermetic test/example fixture)
# ---------------------------------------------------------------------------


def build_synthetic_image_embedding_onnx(
    path: str | Path,
    *,
    feature_dim: int = 192,
    embedding_dim: int = 32,
    seed: int = 0,
) -> Path:
    """Hand-build a tiny ONNX graph and save it to ``path``.

    Image-modality analogue of ``embeddings.build_synthetic_embedding_onnx``
    -- same graph shape (``embedding = L2Normalize(input @ weight + bias)``),
    same hermetic, no-network-access, no-checked-in-model-file rationale.
    **A test/example fixture, not a real CLIP model.**

    ``feature_dim`` must match :func:`resize_and_normalize`'s output length
    for whatever ``size`` the caller intends to use (``size * size * 3``;
    the default ``feature_dim=192`` matches the default ``size=8``:
    ``8 * 8 * 3 == 192``).

    Returns the resolved ``Path`` the model was written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    weight = rng.normal(
        scale=1.0 / np.sqrt(feature_dim), size=(feature_dim, embedding_dim)
    ).astype(np.float32)
    bias = np.zeros((embedding_dim,), dtype=np.float32)
    eps = np.array([1e-12], dtype=np.float32)

    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, feature_dim])
    output_info = helper.make_tensor_value_info(
        "embedding", TensorProto.FLOAT, [None, embedding_dim]
    )

    weight_init = numpy_helper.from_array(weight, name="weight")
    bias_init = numpy_helper.from_array(bias, name="bias")
    eps_init = numpy_helper.from_array(eps, name="eps")

    nodes = [
        helper.make_node("MatMul", ["input", "weight"], ["linear_raw"], name="matmul"),
        helper.make_node("Add", ["linear_raw", "bias"], ["linear_out"], name="add_bias"),
        helper.make_node("Mul", ["linear_out", "linear_out"], ["squared"], name="square"),
        helper.make_node(
            "ReduceSum",
            ["squared"],
            ["sum_squared"],
            name="reduce_sum",
            axes=[1],
            keepdims=1,
        ),
        helper.make_node("Sqrt", ["sum_squared"], ["norm"], name="sqrt"),
        helper.make_node("Add", ["norm", "eps"], ["norm_eps"], name="add_eps"),
        helper.make_node("Div", ["linear_out", "norm_eps"], ["embedding"], name="l2_normalize"),
    ]

    graph = helper.make_graph(
        nodes,
        "lazyclean_synthetic_image_embedding",
        [input_info],
        [output_info],
        initializer=[weight_init, bias_init, eps_init],
    )
    model = helper.make_model(
        graph,
        producer_name="dscraft-clean",
        opset_imports=[helper.make_opsetid("", 11)],
    )
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


# ---------------------------------------------------------------------------
# ImageEmbeddingModel -- the one canonical image-embedding path
# ---------------------------------------------------------------------------


@dataclass
class ImageEmbeddingModel:
    """Wraps an ``onnxruntime.InferenceSession`` plus an image preprocessor.

    Image-modality analogue of ``embeddings.EmbeddingModel``. Unlike the
    text model, a real CLIP vision-tower ONNX graph is single-input
    (``pixel_values``) with no multi-input variant to support, so
    :meth:`embed`'s preprocessor is simpler than ``EmbeddingModel.embed``'s:
    it always returns one ``(feature_dim,)`` array per image, fed to the
    session's single named input.
    """

    session: ort.InferenceSession
    input_name: str
    output_name: str
    preprocessor: Callable[[np.ndarray], np.ndarray]
    embedding_dim: int

    @classmethod
    def from_onnx_file(
        cls,
        model_path: str | Path,
        *,
        preprocessor: Callable[[np.ndarray], np.ndarray],
        embedding_dim: int,
        input_name: str | None = None,
        output_name: str | None = None,
        providers: Sequence[str] | None = None,
    ) -> "ImageEmbeddingModel":
        """Load a ``.onnx`` image-embedding model via ``onnxruntime`` directly.

        No PyTorch, no CLIP-specific Python package -- ``onnxruntime.InferenceSession``
        is the only inference runtime this module ever touches, matching
        ``embeddings.EmbeddingModel.from_onnx_file``.
        """
        session = ort.InferenceSession(
            str(model_path), providers=list(providers) if providers else ["CPUExecutionProvider"]
        )
        resolved_input = input_name or session.get_inputs()[0].name
        resolved_output = output_name or session.get_outputs()[0].name
        return cls(
            session=session,
            input_name=resolved_input,
            output_name=resolved_output,
            preprocessor=preprocessor,
            embedding_dim=embedding_dim,
        )

    def embed(self, images: Iterable[np.ndarray]) -> np.ndarray:
        """Embed a batch of images, returning a ``(n, embedding_dim)`` float32 array."""
        rows = list(images)
        if not rows:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        features = np.stack([self.preprocessor(image) for image in rows]).astype(np.float32)
        (output,) = self.session.run([self.output_name], {self.input_name: features})
        return np.asarray(output, dtype=np.float32)


def build_synthetic_image_embedding_model(
    *,
    cache_dir: str | Path | None = None,
    size: int = 8,
    embedding_dim: int = 32,
    seed: int = 0,
) -> ImageEmbeddingModel:
    """Build (or reuse a cached) synthetic ONNX image-embedding model, ready to use.

    Convenience wrapper around :func:`build_synthetic_image_embedding_onnx`
    + :func:`resize_and_normalize` + :meth:`ImageEmbeddingModel.from_onnx_file`,
    mirroring ``embeddings.build_synthetic_embedding_model``. Fully
    hermetic: no network access, writes a small (a few KB) ``.onnx`` file to
    a temp/cache directory.
    """
    feature_dim = size * size * 3
    cache_dir_path = Path(cache_dir) if cache_dir is not None else Path(tempfile.gettempdir())
    onnx_path = (
        cache_dir_path / f"lazyclean_synthetic_image_v{feature_dim}x{embedding_dim}_seed{seed}.onnx"
    )
    if not onnx_path.exists():
        build_synthetic_image_embedding_onnx(
            onnx_path, feature_dim=feature_dim, embedding_dim=embedding_dim, seed=seed
        )

    def _preprocessor(image: np.ndarray) -> np.ndarray:
        return resize_and_normalize(image, size=size)

    return ImageEmbeddingModel.from_onnx_file(
        onnx_path, preprocessor=_preprocessor, embedding_dim=embedding_dim
    )


def download_recommended_clip_vision_model(
    *,
    cache_dir: str | Path | None = None,
    accept_restricted_licenses: bool = False,
) -> Path:
    """Lazily download the Tier-1 recommended CLIP vision-tower checkpoint (optional).

    Image-modality analogue of ``embeddings.download_recommended_model`` --
    same "never called by tests/examples/import-time code," same
    allowlist-check-before-network-access, same atomic-download-then-rename
    pattern. See :data:`RECOMMENDED_IMAGE_MODEL_NAME`'s allowlist entry
    (registered above) for the licensing rationale and its documented
    caveat.

    After downloading, wire the result into
    :meth:`ImageEmbeddingModel.from_onnx_file` together with a real CLIP
    image preprocessor (resize to 224x224, CLIP-specific mean/std
    normalization -- this function only fetches and caches the ``.onnx``
    graph itself, matching :func:`resize_and_normalize`'s documented scope
    boundary as a synthetic-fixture preprocessor, not a real CLIP
    preprocessing pipeline).
    """
    MODEL_ALLOWLIST.check(
        RECOMMENDED_IMAGE_MODEL_NAME, accept_restricted_licenses=accept_restricted_licenses
    )
    cache_dir_path = (
        Path(cache_dir) if cache_dir is not None else Path.home() / ".cache" / "dscraft" / "clean"
    )
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    dest = cache_dir_path / "clip-vit-base-patch32-vision_model_int8.onnx"
    if dest.exists():
        return dest

    import urllib.request

    fd, tmp_name = tempfile.mkstemp(dir=cache_dir_path, suffix=".onnx.part")
    tmp_path = Path(tmp_name)
    try:
        os.close(fd)
        urllib.request.urlretrieve(_RECOMMENDED_IMAGE_MODEL_ONNX_URL, tmp_path)  # noqa: S310
        tmp_path.replace(dest)  # atomic on the same filesystem
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return dest


# ---------------------------------------------------------------------------
# detect_near_duplicate_images -- the one canonical entrypoint tying this
# module's embedding path to dedup.py's existing, modality-agnostic
# find_near_duplicates.
# ---------------------------------------------------------------------------


def detect_near_duplicate_images(
    images: Iterable[np.ndarray],
    model: ImageEmbeddingModel,
    *,
    threshold: float = 0.92,
) -> tuple[np.ndarray, DedupReport]:
    """Embed ``images`` via ONNX Runtime and flag near-duplicate image pairs.

    Image-modality analogue of
    ``dscraft.clean.detect_near_duplicate_text``. Deliberately reuses
    :func:`dscraft.clean.dedup.find_near_duplicates` as-is rather than
    reimplementing any near-duplicate-detection logic -- that function
    already operates on any ``(n, dim)`` embedding array, regardless of the
    modality that produced it.

    Args:
        images: an iterable of ``(H, W, 3)`` (or ``(H, W)`` grayscale)
            numpy arrays, one per image. This module does not decode image
            files itself -- see the module docstring.
        model: an :class:`ImageEmbeddingModel` (e.g. from
            :func:`build_synthetic_image_embedding_model` for hermetic use,
            or a real production model wired via
            :meth:`ImageEmbeddingModel.from_onnx_file`).
        threshold: cosine-similarity cutoff in ``(0.0, 1.0]`` -- see
            :func:`dscraft.clean.dedup.find_near_duplicates`.

    Returns:
        ``(embeddings, report)`` -- the ``(n_images, embedding_dim)``
        float32 embedding array and the
        :class:`~dscraft.clean.dedup.DedupReport` of flagged pairs.
    """
    embeddings = model.embed(images)
    report = find_near_duplicates(embeddings, threshold=threshold)
    return embeddings, report
