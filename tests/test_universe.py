"""Tests for the curated investable universe: tags, consistency with config, history check."""

from datetime import date

import pandas as pd
import yaml

import gpl.data as data
from gpl.data import load_investable, load_universe, verify_history

CFG = load_universe()
UNIVERSE = load_investable()
ASSET_CLASSES = {"Equity", "Bonds", "Sukuk", "Gold", "Cash", "Real estate"}
SHARIA = {"compliant", "not_compliant", "unknown"}
FACTORS = {"low_vol", "momentum", "quality"}


def test_every_security_is_fully_tagged() -> None:
    assert 60 <= len(UNIVERSE) <= 120
    for t, r in UNIVERSE.items():
        assert r["asset_class"] in ASSET_CLASSES, t
        assert r["sharia"] in SHARIA, t
        assert r["name"] and r["region"] and r["currency"] and r["domicile"], t
        assert r.get("factor") in (None, *FACTORS), t
        if r.get("ucits_of"):
            assert r["domicile"] == "Ireland" and r["ucits_of"] in UNIVERSE, t
        if r.get("gcc_stock"):
            assert r["domicile"] in {"Saudi Arabia", "UAE", "Qatar", "Kuwait", "Bahrain", "Oman"}
    assert {r["factor"] for r in UNIVERSE.values() if r.get("factor")} == FACTORS


def test_universe_agrees_with_the_existing_config_maps() -> None:
    for t, attrs in CFG["security_attributes"].items():
        if t in UNIVERSE:
            assert UNIVERSE[t]["sharia"] == attrs["sharia"], t  # one Sharia status per ticker
    for us, alt in CFG["ucits_equivalents"].items():
        assert UNIVERSE[alt["ticker"]]["ucits_of"] == us, us
    for t in CFG["cash_like"]:
        if t in UNIVERSE:
            assert UNIVERSE[t]["asset_class"] in {"Cash", "Sukuk"}, t
    for t, meta in CFG["exposure_meta"].items():
        if t in UNIVERSE:
            assert UNIVERSE[t]["region"] == meta["region"], t


def test_every_security_has_verified_history() -> None:
    for t, r in UNIVERSE.items():
        assert r["history_from"] and r["verified"], f"{t}: run scripts/verify_universe.py"
        assert date.fromisoformat(r["history_from"]) < date.fromisoformat(r["verified"])


def test_loader_merges_the_generated_history(tmp_path) -> None:
    uni = tmp_path / "u.yaml"
    hist = tmp_path / "h.yaml"
    uni.write_text(yaml.safe_dump({"securities": {"AAA": {"name": "A"}, "BBB": {"name": "B"}}}))
    hist.write_text(
        yaml.safe_dump(
            {"history": {"AAA": {"history_from": "2015-01-02", "verified": "2026-11-04"}}}
        )
    )
    out = load_investable(uni, hist)
    assert out["AAA"]["history_from"] == "2015-01-02" and out["AAA"]["verified"] == "2026-11-04"
    assert out["BBB"]["history_from"] is None  # not verified yet
    assert load_investable(uni, tmp_path / "missing.yaml")["AAA"]["verified"] is None


def test_verify_history_reports_first_last_and_problems(monkeypatch) -> None:
    idx = pd.bdate_range("2015-01-02", "2026-10-30")
    series = {
        "OK": pd.Series(1.0, index=idx),
        "OLD": pd.Series(1.0, index=pd.bdate_range("2015-01-02", "2020-01-31")),
    }
    monkeypatch.setattr(data, "download_series", lambda t, *a, **k: series.get(t))
    out = verify_history(["OK", "OLD", "NONE"], date(2026, 11, 4))
    assert out["OK"] == {"history_from": "2015-01-02", "last_price": "2026-10-30"}
    assert out["OLD"]["problem"] == "no prices since 2020-01-31"
    assert out["NONE"] == {"problem": "no prices on Yahoo Finance"}
