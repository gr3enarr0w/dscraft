"""Tests for the PyTorch-free ONNX Runtime image-embedding + near-duplicate path.

Fully hermetic: uses build_synthetic_image_embedding_model() (a hand-built
ONNX graph via the `onnx` package's graph-builder API), never touches the
network, and never bundles a real multi-hundred-MB CLIP checkpoint. Mirrors
test_embeddings.py's/test_dedup.py's structure for the image modality.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

import dscraft.clean.image_dedup as image_dedup_module
from dscraft.clean import (
    MODEL_ALLOWLIST,
    RECOMMENDED_IMAGE_MODEL_NAME,
    ImageEmbeddingModel,
    build_synthetic_image_embedding_model,
    build_synthetic_image_embedding_onnx,
    detect_near_duplicate_images,
    resize_and_normalize,
)
from dscraft.clean.dedup import DedupReport
from dscraft.core.licensing import ModelTier


def _solid_color_image(color: tuple[int, int, int], size: int = 32) -> np.ndarray:
    """A tiny (size, size, 3) uint8 image filled with a single RGB color."""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :] = color
    return image


def test_no_pytorch_or_transformers_imported():
    """Same hard constraint as embeddings.py: dscraft.clean never imports
    torch/transformers, statically -- not just at the call sites this test
    file happens to exercise."""
    forbidden_import_re = re.compile(r"^\s*(import|from)\s+(torch|transformers)\b", re.MULTILINE)
    package_dir = Path(image_dedup_module.__file__).parent
    offending: list[str] = []
    for source_file in package_dir.rglob("*.py"):
        text = source_file.read_text(encoding="utf-8")
        if forbidden_import_re.search(text):
            offending.append(str(source_file))
    assert offending == [], (
        f"Found forbidden torch/transformers import(s) in: {offending!r}. "
        "dscraft.clean is deliberately PyTorch-free."
    )


def test_recommended_image_model_is_registered_tier_1():
    entry = MODEL_ALLOWLIST.check(RECOMMENDED_IMAGE_MODEL_NAME)
    assert entry.tier is ModelTier.TIER_1


# ---------------------------------------------------------------------------
# resize_and_normalize
# ---------------------------------------------------------------------------


def test_resize_and_normalize_output_shape_and_range():
    image = _solid_color_image((255, 0, 0), size=16)
    vector = resize_and_normalize(image, size=4)
    assert vector.shape == (4 * 4 * 3,)
    assert vector.dtype == np.float32
    assert vector.min() >= 0.0
    assert vector.max() <= 1.0


def test_resize_and_normalize_accepts_grayscale_input():
    grayscale = np.full((16, 16), 128, dtype=np.uint8)
    vector = resize_and_normalize(grayscale, size=4)
    assert vector.shape == (4 * 4 * 3,)


def test_resize_and_normalize_rejects_bad_shapes():
    with pytest.raises(ValueError):
        resize_and_normalize(np.zeros((4, 4, 4), dtype=np.uint8))  # wrong channel count
    with pytest.raises(ValueError):
        resize_and_normalize(np.zeros((4, 4, 4, 3), dtype=np.uint8))  # too many dims


def test_resize_and_normalize_solid_color_is_uniform():
    """A solid-color image resized to any smaller grid should still be a
    uniform vector -- every pixel sampled is the same color."""
    image = _solid_color_image((10, 20, 30), size=32)
    vector = resize_and_normalize(image, size=4).reshape(4, 4, 3)
    expected = np.array([10, 20, 30], dtype=np.float32) / 255.0
    np.testing.assert_allclose(vector[0, 0], expected, atol=1e-6)
    np.testing.assert_allclose(vector[-1, -1], expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Synthetic ONNX graph / ImageEmbeddingModel
# ---------------------------------------------------------------------------


def test_build_synthetic_image_embedding_onnx_writes_a_valid_model(tmp_path):
    onnx_path = tmp_path / "tiny_image.onnx"
    result = build_synthetic_image_embedding_onnx(onnx_path, feature_dim=48, embedding_dim=16, seed=1)

    assert result == onnx_path
    assert onnx_path.exists()
    assert onnx_path.stat().st_size < 100_000


def test_build_synthetic_image_embedding_model_returns_working_model(tmp_path):
    model = build_synthetic_image_embedding_model(cache_dir=tmp_path, size=4, embedding_dim=16)
    assert isinstance(model, ImageEmbeddingModel)
    assert model.embedding_dim == 16

    images = [_solid_color_image((255, 0, 0)), _solid_color_image((0, 255, 0))]
    embeddings = model.embed(images)
    assert embeddings.shape == (2, 16)
    assert embeddings.dtype == np.float32


def test_image_embed_output_is_l2_normalized(tmp_path):
    model = build_synthetic_image_embedding_model(cache_dir=tmp_path, size=4, embedding_dim=16)
    embeddings = model.embed([_solid_color_image((10, 20, 30)), _solid_color_image((200, 5, 90))])
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)


def test_image_embed_is_deterministic_across_calls(tmp_path):
    model = build_synthetic_image_embedding_model(cache_dir=tmp_path, size=4, embedding_dim=16)
    image = [_solid_color_image((77, 88, 99))]
    first = model.embed(image)
    second = model.embed(image)
    np.testing.assert_array_equal(first, second)


def test_image_embed_empty_input_returns_empty_array(tmp_path):
    model = build_synthetic_image_embedding_model(cache_dir=tmp_path, size=4, embedding_dim=16)
    embeddings = model.embed([])
    assert embeddings.shape == (0, 16)


# ---------------------------------------------------------------------------
# detect_near_duplicate_images
# ---------------------------------------------------------------------------


def test_detect_near_duplicate_images_flags_identical_and_near_identical_colors(tmp_path):
    model = build_synthetic_image_embedding_model(cache_dir=tmp_path, size=8, embedding_dim=32)

    red = _solid_color_image((200, 10, 10))
    near_red = _solid_color_image((202, 12, 9))  # visually near-identical to `red`
    blue = _solid_color_image((10, 10, 200))  # clearly distinct

    embeddings, report = detect_near_duplicate_images(
        [red, near_red, blue], model, threshold=0.999
    )

    assert isinstance(report, DedupReport)
    assert embeddings.shape == (3, 32)
    flagged = report.flagged_indices()
    assert {0, 1}.issubset(flagged)
    assert 2 not in flagged


def test_detect_near_duplicate_images_reuses_dedup_find_near_duplicates(tmp_path):
    """detect_near_duplicate_images must not reimplement near-duplicate
    scanning -- it should produce the exact same report dedup.py's
    find_near_duplicates would produce for the same embeddings."""
    from dscraft.clean.dedup import find_near_duplicates

    model = build_synthetic_image_embedding_model(cache_dir=tmp_path, size=8, embedding_dim=32)
    images = [_solid_color_image((5, 5, 5)), _solid_color_image((250, 250, 250))]

    embeddings, report = detect_near_duplicate_images(images, model, threshold=0.5)
    expected_report = find_near_duplicates(model.embed(images), threshold=0.5)

    assert report.threshold == expected_report.threshold
    assert report.num_rows == expected_report.num_rows
    assert [p.similarity for p in report.pairs] == [p.similarity for p in expected_report.pairs]
