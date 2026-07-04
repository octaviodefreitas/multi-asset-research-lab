import numpy as np
import pandas as pd
import pytest

from engine import factors


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(8)
    idx = pd.bdate_range("2015-01-01", periods=1500)
    ff = pd.DataFrame({
        "Mkt-RF": rng.normal(0.0003, 0.010, 1500),
        "SMB": rng.normal(0.0, 0.005, 1500),
        "HML": rng.normal(0.0, 0.005, 1500),
        "Mom": rng.normal(0.0002, 0.007, 1500),
        "RF": np.full(1500, 0.00005),
    }, index=idx)
    return ff


def test_regression_recovers_known_coefficients(synthetic):
    rng = np.random.default_rng(9)
    strat = (synthetic["RF"] + 0.0002 + 0.8 * synthetic["Mkt-RF"]
             - 0.3 * synthetic["HML"] + rng.normal(0, 1e-5, len(synthetic)))
    reg = factors.factor_regression(pd.Series(strat, index=synthetic.index), synthetic)
    assert reg["loadings"]["Mkt-RF"] == pytest.approx(0.8, abs=0.01)
    assert reg["loadings"]["HML"] == pytest.approx(-0.3, abs=0.01)
    assert reg["loadings"]["SMB"] == pytest.approx(0.0, abs=0.01)
    assert reg["alpha_ann"] == pytest.approx(0.0002 * 252, rel=0.05)
    assert reg["r2"] > 0.99
    assert abs(reg["alpha_t"]) > 2


def test_regression_requires_overlap(synthetic):
    strat = pd.Series([0.001] * 30, index=pd.bdate_range("1990-01-01", periods=30))
    with pytest.raises(ValueError):
        factors.factor_regression(strat, synthetic)


def test_ff_csv_parser_handles_preamble_and_annual_section():
    text = (
        "This file was created by CMPT_ME_BEME_RETS using something.\n"
        "\n"
        ",Mkt-RF,SMB,HML,RF\n"
        "19260701,0.10,-0.25,-0.27,0.009\n"
        "19260702,0.45,-0.33,-0.06,0.009\n"
        "19260706,0.17,,-0.04,0.009\n"
        "\n"
        "Annual Factors: January-December\n"
        ",Mkt-RF,SMB,HML,RF\n"
        "1927,29.47,-2.46,-3.75,3.12\n"
    )
    df = factors._parse_ff_csv(text)
    assert len(df) == 3  # annual rows excluded, daily rows kept
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert df.iloc[0]["Mkt-RF"] == pytest.approx(0.0010)  # percent -> decimal
    assert np.isnan(df.iloc[2]["SMB"])  # empty cell -> NaN, not a crash
