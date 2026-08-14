from reportlab.platypus import SimpleDocTemplate
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet


def create_pdf(filename, prediction, confidence):

    pdf = SimpleDocTemplate(filename)

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph(
            "<font size=22><b>Brain Tumor Detection Report</b></font>",
            styles["Title"],
        )
    )

    story.append(
        Paragraph("<br/><br/>", styles["Normal"])
    )

    story.append(
        Paragraph(
            f"<b>Prediction :</b> {prediction}",
            styles["Heading2"],
        )
    )

    story.append(
        Paragraph(
            f"<b>Confidence :</b> {confidence:.2f} %",
            styles["Heading2"],
        )
    )

    story.append(
        Paragraph("<br/><br/>", styles["Normal"])
    )

    story.append(
        Paragraph(
            "Generated using EfficientNet Deep Learning Model.",
            styles["BodyText"],
        )
    )

    pdf.build(story)