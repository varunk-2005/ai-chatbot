from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch


def main():
    c = canvas.Canvas("sample.pdf", pagesize=LETTER)
    width, height = LETTER

    text = c.beginText()
    text.setTextOrigin(inch, height - inch)
    text.setFont("Helvetica", 12)

    lines = [
        "Sample PDF for Q&A app",
        "",
        "This PDF contains extractable text. You can ask questions like:",
        "- What is this document about?",
        "- List three key topics mentioned.",
        "",
        "Content:",
        "Streamlit is a Python framework for building data apps.",
        "FAISS is a vector store for fast similarity search.",
        "LangChain helps orchestrate LLM pipelines and prompts.",
        "This sample is intended for testing purposes.",
    ]

    for line in lines:
        text.textLine(line)

    c.drawText(text)
    c.showPage()
    c.save()


if __name__ == "__main__":
    main()

