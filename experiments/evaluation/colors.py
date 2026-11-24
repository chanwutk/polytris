"""Shared, editable categorical colors for Tetris's Altair charts."""
import altair as alt


# Start with observable10 so existing figures retain their colors.
TETRIS_COLORS = [
    '#4269d0',
    '#efb118',
    '#ff725c',
    '#6cc5b0',
    '#3ca951',
    '#a463f2',
    '#9c6b4e',
    '#97bbf5',
    '#ff8ab7',
    '#9498a0',
]


def categorical_color_scale(domain=None):
    """Return the shared palette, optionally fixing category order."""
    return alt.Scale(domain=alt.Undefined if domain is None else domain, range=TETRIS_COLORS)
