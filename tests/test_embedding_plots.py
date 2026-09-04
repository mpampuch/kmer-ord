# tests/test_embedding_plots.py
"""Tests for embedding-plot column selection and the categorical memory guard.

Regression tests for the MemoryError where `sequence_id` (~880k unique
values) reached datashader's `ds.by()`, which allocates a
(height, width, n_categories) uint32 aggregation array — 526 GiB at 400x400.
ID-like columns must never reach the renderer, and the renderer itself must
refuse pathological cardinalities instead of attempting the allocation.
"""
import numpy as np
import pandas as pd
import pytest

from kmer_ord.vis.embedding_plots import (
    MAX_CATEGORICAL_AGG_BYTES,
    _plottable_feature_columns,
    _render_feature_to_image,
)


def _embedding_df(n_rows, categories):
    """Small embedding-like frame: coordinates plus one categorical column."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "x": rng.normal(size=n_rows),
        "y": rng.normal(size=n_rows),
        "feature": categories,
    })


class TestPlottableFeatureColumns:
    def test_excludes_id_columns_keeps_real_features(self):
        features_df = pd.DataFrame({
            "sequence_id": ["a", "b"],
            "header": ["h1", "h2"],
            "taxon": ["x", "y"],
            "GC_Content": [0.4, 0.6],
        })
        categorical, continuous = _plottable_feature_columns(features_df)
        assert categorical == ["taxon"]
        assert continuous == ["GC_Content"]

    def test_none_features_df_yields_empty_lists(self):
        # _load_features returns None when the table is missing; the loops
        # must not dereference it.
        assert _plottable_feature_columns(None) == ([], [])


class TestCategoricalMemoryGuard:
    def test_pathological_cardinality_is_skipped(self):
        # Enough unique categories that the (400, 400, n) uint32 aggregation
        # would exceed MAX_CATEGORICAL_AGG_BYTES.
        n = MAX_CATEGORICAL_AGG_BYTES // (400 * 400 * 4) + 2
        df = _embedding_df(n, [f"cat_{i}" for i in range(n)])
        img, color_key = _render_feature_to_image(
            df, "x", "y", "feature", mode="categorical")
        assert img is None
        assert color_key is None

    def test_normal_cardinality_still_renders(self):
        df = _embedding_df(300, [f"cat_{i % 3}" for i in range(300)])
        img, color_key = _render_feature_to_image(
            df, "x", "y", "feature", mode="categorical")
        assert img is not None
        assert set(color_key) == {"cat_0", "cat_1", "cat_2"}
