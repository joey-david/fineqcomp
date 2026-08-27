from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib import colors


OUTPUT = "/Users/joey/Documents/admin/housing/preavis_depart_214_rue_saint_denis_2026-08-27.pdf"


def build_pdf() -> None:
    document = SimpleDocTemplate(
        OUTPUT,
        pagesize=A4,
        rightMargin=22 * mm,
        leftMargin=22 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        title="Congé - 214 rue Saint-Denis - 30 septembre 2026",
        author="Joey David L'homme de Prailles",
    )

    styles = getSampleStyleSheet()
    sender_style = ParagraphStyle(
        "Sender",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#243447"),
    )
    recipient_style = ParagraphStyle(
        "Recipient",
        parent=sender_style,
        alignment=TA_LEFT,
    )
    date_style = ParagraphStyle(
        "Date",
        parent=sender_style,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#536273"),
    )
    subject_style = ParagraphStyle(
        "Subject",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#102a43"),
        spaceAfter=9,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=17,
        textColor=colors.HexColor("#1f2933"),
        spaceAfter=10,
    )

    sender = Paragraph(
        "<b>Joey David L'homme de Prailles</b><br/>"
        "214 rue Saint-Denis<br/>75002 Paris<br/>"
        "+33 7 83 36 71 12<br/>joeydavid99@gmail.com",
        sender_style,
    )
    recipient = Paragraph(
        "<b>Loic Gnouoguia Tagne</b><br/>"
        "12 rue de Strasbourg<br/>92000 Nanterre",
        recipient_style,
    )
    address_table = Table([[sender, recipient]], colWidths=[88 * mm, 78 * mm])
    address_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )

    story = [
        address_table,
        Spacer(1, 19 * mm),
        Paragraph("Paris, le 27 août 2026", date_style),
        Spacer(1, 10 * mm),
        Paragraph(
            "Objet : Notification de congé - logement meublé situé au 214 rue Saint-Denis, 75002 Paris",
            subject_style,
        ),
        Paragraph("Monsieur,", body_style),
        Paragraph(
            "Je vous notifie par la présente mon congé concernant le logement meublé situé au "
            "214 rue Saint-Denis, 75002 Paris.",
            body_style,
        ),
        Paragraph(
            "Conformément aux stipulations de mon contrat de location et à l'article 25-8 de la loi "
            "du 6 juillet 1989, le préavis applicable est d'un mois à compter de la réception du congé. "
            "Je prévois donc de libérer les lieux et de restituer les clés le 30 septembre 2026.",
            body_style,
        ),
        Paragraph(
            "Je vous remercie de me confirmer la réception de ce congé et de me proposer des créneaux "
            "pour l'état des lieux de sortie, l'inventaire du mobilier et la remise des clés.",
            body_style,
        ),
        Paragraph(
            "Je vous communiquerai ma nouvelle adresse pour la restitution du dépôt de garantie.",
            body_style,
        ),
        Spacer(1, 4 * mm),
        Paragraph("Je vous prie d'agréer, Monsieur, l'expression de mes salutations distinguées.", body_style),
        Spacer(1, 9 * mm),
        Paragraph("Joey David L'homme de Prailles", body_style),
    ]

    def draw_page(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d9e2ec"))
        canvas.setLineWidth(0.7)
        canvas.line(22 * mm, 15 * mm, A4[0] - 22 * mm, 15 * mm)
        canvas.setFont("Helvetica", 8.5)
        canvas.setFillColor(colors.HexColor("#829ab1"))
        canvas.drawString(22 * mm, 10.5 * mm, "Congé - 214 rue Saint-Denis")
        canvas.drawRightString(A4[0] - 22 * mm, 10.5 * mm, "27 août 2026")
        canvas.restoreState()

    document.build(story, onFirstPage=draw_page)


if __name__ == "__main__":
    build_pdf()
