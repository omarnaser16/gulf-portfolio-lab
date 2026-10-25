"""Tests for the PDF documents (uncompressed in tests so their text can be searched)."""

from datetime import date

from gpl.client import EXAMPLE_CLIENTS, ClientProfile, build_ips
from gpl.data import load_universe
from gpl.report import FOOTER, esc, ips_pdf

CFG = load_universe()


def _text(pdf: bytes) -> str:
    return pdf.decode("latin-1")


def test_ips_pdf_contains_rrttllu_limits_and_footer() -> None:
    p = ClientProfile.from_dict(EXAMPLE_CLIENTS["Riyadh, conservative, 5 years, Sharia"])
    ips = build_ips(p, CFG, today=date(2026, 10, 25))
    pdf = ips_pdf(ips, compress=False)
    assert pdf.startswith(b"%PDF")
    text = _text(pdf)
    for heading in [
        "Return",
        "Risk",
        "Time horizon",
        "Taxes",
        "Liquidity",
        "Legal and regulatory",
        "Unique circumstances",
    ]:
        assert f"({heading})" in text, heading
    assert "(9.00%)" in text and "(15.00%)" in text  # band 2 limits
    assert "2026-10-25" in text and "EX-RUH" in text
    assert FOOTER[:40] in text
    assert len(ips_pdf(ips)) < len(pdf)  # compressed by default


def test_markup_is_escaped() -> None:
    assert esc("A & B <c>") == "A &amp; B &lt;c&gt;"
    p = ClientProfile(reference="R&D <1>", horizon_years=5)
    assert ips_pdf(build_ips(p, CFG)).startswith(b"%PDF")  # no markup error
