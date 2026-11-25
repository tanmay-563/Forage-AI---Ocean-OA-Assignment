"""Create compact, searchable synthetic supplier profile PDFs using the PDF standard library only."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1] / "demos" / "marine_procurement" / "documents"
PROFILES = {
    "supplier_northstar.pdf": [
        "SUPPLIER PROFILE: NORTHSTAR MARINE COMPONENTS LTD.",
        "Supplier code NSM-044. Fictional Blue Current vendor desk review: 18 February 2026.",
        "Northstar supplies filtration, gasket, and pump-service parts from Kochi, India. Its vendor record is active through 30 June 2026, subject to routine sanctions rescreening. Tax registration and bank account were verified on 12 January 2026; banking details are confidential.",
        "The supplier accepts purchase orders in USD or INR. Standard quotations remain valid for 14 calendar days unless a quotation states a shorter period. Export packaging is included above USD 1,000; smaller orders may incur a quotation-specific packaging fee.",
        "Dispatch estimates begin after purchase-order acceptance and required drawing approval. Catalogue stock is not reserved. Emergency dispatch may incur a fee and requires Procurement Manager approval under Blue Current policy.",
        "Certificates of conformity are available for products marked with a certificate flag. Batch traceability is available on request before shipment. Northstar does not guarantee compatibility of unlisted substitutes.",
        "Service hours are Monday through Friday, 08:30 to 17:00 India Standard Time. Weekend dispatch and guaranteed delivery dates are not stated. This synthetic profile is for demonstration only.",
    ],
    "supplier_pelagic.pdf": [
        "SUPPLIER PROFILE: PELAGIC INDUSTRIAL SUPPLY CO.",
        "Supplier code PIS-209. Fictional Blue Current vendor desk review: 19 February 2026.",
        "Pelagic supplies marine electrical, deck hardware, and pipe fittings from its Singapore distribution centre. Its vendor record is active through 31 December 2026, subject to quarterly compliance screening. Bank verification and tax registration were reviewed on 3 February 2026; account numbers remain restricted to Finance.",
        "Pelagic issues quotations in USD and SGD. Standard quotations remain valid for 21 calendar days unless stated otherwise. Export packing is included in standard catalogue unit prices. Dangerous-goods handling and temperature-controlled freight are quoted separately.",
        "Published lead times count business days after a valid purchase order is accepted. Catalogue stock is a snapshot and must be reconfirmed. Certificates of conformity are available for catalogue products marked as certified.",
        "Returns require written authorization within 15 calendar days after receipt. Special-order and cut-to-length products are excluded from standard returns. The supplier does not promise any particular vessel arrival date.",
        "Account service operates Monday through Friday, 09:00 to 18:00 Singapore Time. This synthetic profile is for demonstration only.",
    ],
}


def pdf_escape(text: str) -> str:
    ascii_text = text.encode("ascii", "replace").decode("ascii")
    return ascii_text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def create_pdf(lines: list[str], destination: Path) -> None:
    wrapped = []
    for line in lines:
        wrapped.extend([line[i:i + 88] for i in range(0, len(line), 88)] or [""])
    commands = ["BT", "/F1 10 Tf", "50 790 Td", "14 TL"]
    for index, line in enumerate(wrapped):
        if index: commands.append("T*")
        commands.append(f"({pdf_escape(line)}) Tj")
    commands.append("ET")
    stream = "\n".join(commands).encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, body in enumerate(objects, 1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]: output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    destination.write_bytes(output)


if __name__ == "__main__":
    for filename, content in PROFILES.items(): create_pdf(content, ROOT / filename)
