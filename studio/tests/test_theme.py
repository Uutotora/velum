"""Tests for the Studio design tokens (studio/theme.py).

Pure strings — no Qt needed; runs under the light CI `test` group.
"""
import re

import pytest

from studio import theme


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


# ── SCRIM contrast ───────────────────────────────────────────────────────────
def _parse_color(value: str) -> tuple[float, float, float, float]:
    """(r, g, b, a) for either a token's '#rrggbb' or SCRIM's 'rgba(...)' form."""
    if value.startswith("#"):
        return int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16), 1.0
    m = re.match(r"rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([\d.]+)\s*\)", value)
    assert m, f"not a recognised colour: {value!r}"
    r, g, b, a = m.groups()
    return float(r), float(g), float(b), float(a)


def _luminance(r: float, g: float, b: float) -> float:
    return 0.299 * r + 0.587 * g + 0.114 * b  # perceived brightness (ITU BT.601)


def _composite_over(fg: tuple[float, float, float, float],
                    bg: tuple[float, float, float, float]) -> tuple[float, float, float]:
    fr, fgg, fb, fa = fg
    br, bgg, bb, _ = bg
    return (fr * fa + br * (1 - fa), fgg * fa + bgg * (1 - fa), fb * fa + bb * (1 - fa))


@pytest.mark.parametrize("palette", [theme.LIGHT, theme.DARK])
@pytest.mark.parametrize("backdrop_key", ["bg", "surface", "surface2"])
def test_scrim_visibly_darkens_every_backdrop_token_in_both_themes(palette, backdrop_key):
    """Regression test: a scrim's whole job is to read as "darker than
    whatever's behind it" -- the previous value, rgba(8,10,20,0.34), was
    clearly tuned against light theme's white backdrop (a strong, obvious
    dim there) but never re-checked against dark theme, where its own RGB
    (8,10,20) nearly matches dark theme's own `bg` (#0d0f13) -- compositing
    barely changed anything, confirmed by pixel-sampling a real running
    dialog (#101318 -> #0e1017, a few units of drift, not a visible dim). A
    real modal read as a randomly-placed box on an undimmed page, not an
    overlay -- reported directly ("куда сьехало окно" -- where did the
    window slide off to).

    Asserts on *relative* luminance reduction, not an absolute unit delta:
    an already-near-black backdrop (dark theme's own `bg`) has little room
    to darken in absolute terms even at correct settings, but a fixed
    (theme-independent) pure-black scrim at a given alpha darkens *any*
    backdrop by exactly that alpha's fraction, regardless of how dark the
    backdrop already is -- the only way to compare fairly across both
    themes' very different starting lightness.
    """
    scrim = _parse_color(theme.SCRIM)
    backdrop = _parse_color(palette[backdrop_key])
    composited_lum = _luminance(*_composite_over(scrim, backdrop))
    backdrop_lum = _luminance(*backdrop[:3])
    relative_reduction = (backdrop_lum - composited_lum) / backdrop_lum
    assert relative_reduction >= 0.30, (
        f"scrim over {backdrop_key}={palette[backdrop_key]!r} only darkened luminance by "
        f"{relative_reduction:.0%} ({backdrop_lum:.1f} -> {composited_lum:.1f}) -- "
        f"not a visibly perceptible dim")
