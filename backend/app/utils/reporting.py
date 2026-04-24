from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib import colors

from app.core.config import settings
from app.core.schemas import PredictionResponse


def generate_prediction_report_pdf(
    patient_id: str,
    prediction: PredictionResponse,
    notes: str | None = None,
) -> tuple[str, Path]:
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    report_id = f"ecg-report-{uuid4().hex[:12]}"
    output_path = settings.reports_dir / f"{report_id}.pdf"

    doc = SimpleDocTemplate(str(output_path), pagesize=A4)
    styles = getSampleStyleSheet()
    content = []

    title = Paragraph("ECG Arrhythmia Detection Report", styles["Title"])
    content.append(title)
    content.append(Spacer(1, 16))

    meta = [
        ["Report ID", report_id],
        ["Patient ID", patient_id],
        ["Generated (UTC)", datetime.now(timezone.utc).isoformat()],
        ["Arrhythmia Flag", str(prediction.arrhythmia)],
        ["Risk Score", f"{prediction.risk_score:.3f}"],
        ["Confidence", f"{prediction.confidence:.3f}"],
        ["Signal Quality", prediction.signal_quality],
    ]
    table = Table(meta, colWidths=[180, 320])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    content.append(table)
    content.append(Spacer(1, 18))

    content.append(Paragraph("Top Class Probabilities", styles["Heading2"]))
    prob_rows = [["Class", "Probability"]] + [
        [entry.label, f"{entry.probability:.3f}"] for entry in prediction.top_classes
    ]
    prob_table = Table(prob_rows, colWidths=[350, 150])
    prob_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9ecff")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    content.append(prob_table)
    content.append(Spacer(1, 14))

    content.append(Paragraph("Model Explanation", styles["Heading2"]))
    content.append(Paragraph(prediction.explanation_text, styles["BodyText"]))
    content.append(Spacer(1, 10))

    if notes:
        content.append(Paragraph("Clinical Notes", styles["Heading2"]))
        content.append(Paragraph(notes, styles["BodyText"]))

    doc.build(content)
    return report_id, output_path

