"""The design tokens are defined once (gpl.theme) and mirrored exactly in config.toml."""

import tomllib
from pathlib import Path

from gpl import theme
from gpl.plots import SERIES_COLORS

CONFIG = Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml"


def test_config_toml_mirrors_the_theme_tokens() -> None:
    cfg = tomllib.loads(CONFIG.read_text())["theme"]
    assert cfg["primaryColor"] == theme.BRAND
    assert cfg["backgroundColor"] == theme.BG
    assert cfg["secondaryBackgroundColor"] == theme.SUBTLE
    assert cfg["textColor"] == theme.TEXT
    assert cfg["linkColor"] == theme.BRAND_BUTTON
    assert cfg["borderColor"] == theme.BORDER
    assert cfg["baseRadius"] == f"{theme.RADIUS_PX}px"
    assert cfg["font"] == f"{theme.FONT}:{theme.FONT_URL}"


def test_tokens_and_chart_palette() -> None:
    assert theme.space(1) == "8px" and theme.space(3) == "24px" and theme.space(0.5) == "4px"
    assert SERIES_COLORS[0] == theme.BRAND and SERIES_COLORS[1] == theme.ACCENT
    assert len(set(SERIES_COLORS)) == len(SERIES_COLORS)
    assert theme.GAIN not in SERIES_COLORS and theme.LOSS not in SERIES_COLORS  # kept for P&L
