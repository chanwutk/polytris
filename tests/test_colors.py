import altair as alt
import pytest

from evaluation import colors


@pytest.mark.parametrize('domain', [None, [], ['second', 'first']])
def test_shared_palette_serializes_with_optional_category_order(domain):
    scale = colors.categorical_color_scale(domain).to_dict()
    assert scale['range'] == colors.TETRIS_COLORS
    if domain is None:
        assert 'domain' not in scale
    else:
        assert scale['domain'] == domain


def test_palette_edit_reaches_chart_spec(monkeypatch):
    monkeypatch.setattr(colors, 'TETRIS_COLORS', ['#112233', '#abcdef'])
    chart = alt.Chart(alt.Data(values=[{'category': 'a', 'value': 1}])).mark_bar().encode(
        x='category:N', y='value:Q',
        color=alt.Color('category:N', scale=colors.categorical_color_scale(['a', 'b'])),
    )
    assert chart.to_dict()['encoding']['color']['scale'] == {
        'domain': ['a', 'b'], 'range': ['#112233', '#abcdef'],
    }
