from io import BytesIO

import numpy as np
import pandas as pd
from pptx import Presentation

from app import export


def _series(n=100):
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(np.linspace(1.0, 1.5, n), index=idx)


def test_tearsheet_is_valid_pptx_with_expected_slides():
    charts = [("Equity", {"Strategy": _series(), "Benchmark": _series() * 0.9})]
    tables = [("Metrics", pd.DataFrame({"Sharpe": ["1.10"], "CAGR": ["8.0%"]},
                                       index=["EW Portfolio"]))]
    blob = export.build_tearsheet("test subtitle", charts, tables)

    assert blob[:2] == b"PK"  # pptx is a zip container
    prs = Presentation(BytesIO(blob))
    assert len(prs.slides) == 3  # title + 1 chart + 1 table


def test_tearsheet_handles_nans_in_series():
    s = _series()
    s.iloc[10:20] = np.nan
    blob = export.build_tearsheet("sub", [("Equity", {"Strategy": s})], [])
    assert blob[:2] == b"PK"


def test_disclaimer_on_title_slide():
    blob = export.build_tearsheet("sub", [], [])
    prs = Presentation(BytesIO(blob))
    text = " ".join(shape.text_frame.text for shape in prs.slides[0].shapes
                    if shape.has_text_frame)
    assert "not investment advice" in text.lower()
