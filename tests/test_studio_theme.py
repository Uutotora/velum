"""Tests for the Studio design tokens (napari_app/studio/theme.py).

Pure strings — no Qt needed; runs under the light CI `test` group.
"""
import re

import pytest

from napari_app.studio import theme


def test_light_and_dark_share_identical_token_keys():
    # widgets style against token *names*; both themes must define the same set
    assert set(theme.LIGHT) == set(theme.DARK)


@pytest.mark.parametrize("palette", [theme.LIGHT, theme.DARK])
def test_every_token_is_a_valid_colour(palette):
    hex_re = re.compile(r"^#[0-9a-fA-F]{6}$")
    rgba_re = re.compile(r"^rgba\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*[\d.]+\s*\)$")
    for name, value in palette.items():
        assert hex_re.match(value) or rgba_re.match(value), f"{name}={value!r}"


def test_core_tokens_present():
    for key in ("bg", "surface", "border", "text", "text_muted",
                "primary", "signal", "success", "warning", "danger", "scope"):
        assert key in theme.LIGHT and key in theme.DARK


@pytest.mark.parametrize("palette", [theme.LIGHT, theme.DARK])
def test_build_qss_is_nonempty_and_styles_key_widgets(palette):
    qss = theme.build_qss(palette)
    assert len(qss) > 500
    for selector in ("QWidget", "QLineEdit", "QComboBox", "QCheckBox",
                     "QProgressBar", "QScrollBar", "QMenu"):
        assert selector in qss
    # no unsubstituted format braces left behind
    assert "{t[" not in qss


@pytest.mark.parametrize("kind", ["primary", "ghost", "success", "danger"])
def test_button_qss_builds_for_each_kind(kind):
    qss = theme.button_qss(theme.DARK, kind)
    assert "QPushButton" in qss and len(qss) > 60


def test_button_qss_rejects_unknown_kind():
    with pytest.raises(ValueError):
        theme.button_qss(theme.DARK, "sparkly")


def test_tokens_for_selects_palette_and_defaults_to_dark():
    assert theme.tokens_for("light") is theme.LIGHT
    assert theme.tokens_for("dark") is theme.DARK
    assert theme.tokens_for("nonsense") is theme.DARK


def test_viridis_ramp_endpoints_and_midpoint():
    assert theme.viridis_rgb(0.0) == (68, 1, 84)
    assert theme.viridis_rgb(1.0) == (253, 231, 37)
    assert theme.viridis_rgb(-5) == (68, 1, 84)      # clamps low
    assert theme.viridis_rgb(5) == (253, 231, 37)    # clamps high
    mid = theme.viridis_rgb(0.5)
    assert mid == (33, 145, 140)                      # exact control point


def test_viz_palette_has_six_distinct_hexes():
    assert len(theme.VIZ) == 6
    assert len(set(theme.VIZ)) == 6
