"""Design tokens: the one place the app's colours, type, radius and spacing are defined.

Presentation only (no finance). Used by the plotly template (``gpl.plots``), the app's CSS
(``app/layout.py``) and mirrored in ``.streamlit/config.toml`` (a test keeps them equal).

Brand blue is the interface colour; accent pink is kept for one primary action per screen and
for "recommended". Green and red are only for gains/losses and pass/fail.
"""

BG = "#FCFFFF"  # page
CARD = "#FFFFFF"  # cards, tables, charts
BORDER = "#E2E8EC"  # subtle card borders
SUBTLE = "#F3F7F9"  # table headers, pill groups, quiet fills
TEXT = "#1A2530"  # charcoal
MUTED = "#5B6875"  # secondary text, captions
BRAND = "#0593C6"  # brand blue: header band, links, active states
BRAND_BUTTON = "#04759F"  # brand buttons (darker for contrast with white text)
BRAND_TINT = "#EAF6FB"  # active step card, selected pill
ACCENT = "#D83995"  # accent pink: the one primary action, "recommended"
ACCENT_BUTTON = "#C42C82"
ACCENT_TINT = "#FCEFF6"  # recommended column / card
GAIN = "#1E8A4C"  # gains, pass
GAIN_TINT = "#E6F4EC"
LOSS = "#C0392B"  # losses, fail
LOSS_TINT = "#FBEAE8"
WARN = "#B26B00"  # review, draft, rebalance suggested
WARN_TINT = "#FFF4E0"

FONT = "Inter"
FONT_URL = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
FONT_STACK = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
RADIUS_PX = 12
SPACE_PX = 8  # spacing scale: multiples of 8 px
MAX_WIDTH_PX = 1200

# Chart series, in order (never cycled): brand blue is "current"/"portfolio", pink is
# "recommended"/"proposed", then distinct hues. A benchmark is the grey dashed REFERENCE.
SERIES = [
    "#0593C6",  # brand blue
    "#D83995",  # accent pink
    "#1F3A5F",  # navy
    "#E3A72F",  # amber
    "#7CC4E4",  # light blue
    "#8C6BC8",  # violet
    "#5E7186",  # slate
    "#2BA3A3",  # teal
]
# Donut / composition slices: one blue family plus amber and greys (no pink: that means
# "recommended" elsewhere).
COMPOSITION = ["#1F3A5F", "#0593C6", "#7CC4E4", "#BFE3F2", "#E3A72F", "#8C96A0", "#2B6C8E",
               "#5E7186"]  # fmt: skip
REFERENCE = "#8C96A0"  # benchmark: grey, dashed
GRID = "#EDF1F4"


def space(n: float) -> str:
    """``n`` steps of the 8 px scale as CSS, e.g. space(2) -> '16px'."""
    return f"{n * SPACE_PX:g}px"
