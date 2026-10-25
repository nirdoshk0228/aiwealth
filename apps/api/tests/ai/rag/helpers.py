"""Test helpers: a minimal text PDF writer and a tiny chunk corpus."""

from app.ai.rag.corpus import Chunk


def make_pdf(pages: list[str]) -> bytes:
    """A valid PDF with one Helvetica text line per page (enough for pypdf extraction)."""
    objs: list[bytes] = []
    n_pages = len(pages)
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n_pages))
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    font_id = 3 + 2 * n_pages
    for i, text in enumerate(pages):
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 50 700 Td ({safe}) Tj ET".encode()
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {4 + 2 * i} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode()
        )
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def chunk(cid: str, text: str, fund: str | None = None, page: int = 1) -> Chunk:
    doc = cid.split(":")[0]
    return Chunk(
        id=cid,
        doc_id=doc,
        title=f"Title of {doc}",
        doc_type="sid" if fund else "regulation",
        fund_code=fund,
        url=f"https://example.invalid/{doc}.pdf",
        page=page,
        text=text,
    )


CORPUS = [
    chunk("sid_alpha:p3:0", "The exit load is 1% if units are redeemed within 365 days.", "111", 3),
    chunk("sid_alpha:p7:0", "The benchmark of the scheme is the NIFTY 500 TRI index.", "111", 7),
    chunk(
        "sid_beta:p2:0",
        "Exit load: nil. The liquid scheme invests in money market paper.",
        "222",
        2,
    ),
    chunk("sid_beta:p9:0", "The benchmark is the NIFTY Liquid Index A-I.", "222", 9),
    chunk(
        "sebi_circ:p4:0",
        "The riskometer has six levels of risk, from low to very high risk.",
        None,
        4,
    ),
    chunk("sebi_circ:p5:0", "Large cap companies are the 1st to 100th by market capitalisation."),
]
