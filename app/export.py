"""PowerPoint tearsheet export.

Charts are written as native PowerPoint chart objects (not images), so an
analyst can recolor and restyle them to a house template after pasting.
"""
from __future__ import annotations

from datetime import date
from io import BytesIO

import pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE
from pptx.util import Inches, Pt

DISCLAIMER = ("Backtested / simulated research on historical data. Not live trading "
              "performance; not investment advice. Backtested results are subject to "
              "overfitting and will differ from live results.")


def _title_slide(prs: Presentation, title: str, subtitle: str) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = title
    slide.placeholders[1].text = f"{subtitle}\nGenerated {date.today():%d %b %Y}\n\n{DISCLAIMER}"


def _line_chart_slide(prs: Presentation, title: str, series: dict[str, pd.Series]) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[5])  # title-only layout
    slide.shapes.title.text = title
    data = CategoryChartData()
    first = next(iter(series.values()))
    data.categories = [d.strftime("%Y-%m-%d") for d in first.index]
    for name, s in series.items():
        data.add_series(name, tuple(None if pd.isna(v) else round(float(v), 5)
                                    for v in s.to_numpy()))
    chart = slide.shapes.add_chart(XL_CHART_TYPE.LINE, Inches(0.4), Inches(1.4),
                                   Inches(9.2), Inches(5.4), data).chart
    chart.has_title = False


def _table_slide(prs: Presentation, title: str, df: pd.DataFrame) -> None:
    """df must already contain display-formatted strings."""
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = title
    rows, cols = df.shape[0] + 1, df.shape[1] + 1
    table = slide.shapes.add_table(rows, cols, Inches(0.3), Inches(1.4),
                                   Inches(9.4), Inches(0.4 * rows)).table
    table.cell(0, 0).text = ""
    for j, col in enumerate(df.columns):
        table.cell(0, j + 1).text = str(col)
    for i, (idx, row) in enumerate(df.iterrows()):
        table.cell(i + 1, 0).text = str(idx)
        for j, value in enumerate(row):
            table.cell(i + 1, j + 1).text = str(value)
    for cell_row in table.rows:
        for cell in cell_row.cells:
            for para in cell.text_frame.paragraphs:
                para.font.size = Pt(11)


def build_tearsheet(subtitle: str,
                    charts: list[tuple[str, dict[str, pd.Series]]],
                    tables: list[tuple[str, pd.DataFrame]]) -> bytes:
    """Assemble the deck: title slide, one slide per chart (native, editable),
    one slide per formatted table. Returns the .pptx file as bytes."""
    prs = Presentation()
    _title_slide(prs, "Multi-Asset Systematic Strategy — Tearsheet", subtitle)
    for title, series in charts:
        _line_chart_slide(prs, title, series)
    for title, df in tables:
        _table_slide(prs, title, df)
    buf = BytesIO()
    prs.save(buf)
    return buf.getvalue()
