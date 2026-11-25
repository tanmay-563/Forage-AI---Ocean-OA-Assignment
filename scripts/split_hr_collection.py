"""Split the canonical synthetic handbook into a coherent multi-document HR collection."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1] / "demos" / "hr_policy"
SOURCE = ROOT / "source_handbook.md"
DOCUMENTS = ROOT / "documents"

if not SOURCE.exists():
    SOURCE.write_text((DOCUMENTS / "employee_handbook.md").read_text(encoding="utf-8"), encoding="utf-8")
text = SOURCE.read_text(encoding="utf-8")
heading = re.search(r"(?m)^##\s+", text)
intro = text[:heading.start()] if heading else text
parts = re.split(r"(?m)(?=^##\s+)", text[heading.start():] if heading else "")
sections = {match.group(1).strip(): part.strip() for part in parts if (match := re.match(r"(?m)^##\s+(.+)$", part))}
groups = {
    "employee_handbook.md": ("Northstar Tidal Systems Employee Handbook", ["Annual leave", "Sick leave", "Attendance and working hours", "Questions and policy conflicts"]),
    "benefits_remote_work.md": ("Benefits and Remote Work Policy", ["Health and wellbeing benefits", "Remote and hybrid work"]),
    "travel_and_expenses.md": ("Travel and Expense Policy", ["Business travel and reimbursement", "General expense claims"]),
}
DOCUMENTS.mkdir(parents=True, exist_ok=True)
for filename, (title, names) in groups.items():
    selected = [sections[name] for name in names if name in sections]
    if len(selected) != len(names): raise ValueError(f"Canonical handbook is missing a section for {filename}")
    content = intro.replace("# Northstar Tidal Systems Employee Handbook", f"# {title}", 1).strip()
    (DOCUMENTS / filename).write_text(content + "\n\n" + "\n\n".join(selected) + "\n", encoding="utf-8")
