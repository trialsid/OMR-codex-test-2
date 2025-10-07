"""Convert PDF OMR sheet to image for testing."""
import fitz  # PyMuPDF
from pathlib import Path


def pdf_to_image(pdf_path: Path, output_path: Path, dpi: int = 200):
    """Convert first page of PDF to PNG image.

    Args:
        pdf_path: Input PDF file
        output_path: Output image file
        dpi: Resolution for conversion
    """
    # Open PDF
    doc = fitz.open(str(pdf_path))

    # Get first page
    page = doc[0]

    # Calculate zoom factor from DPI
    zoom = dpi / 72  # 72 is default DPI
    mat = fitz.Matrix(zoom, zoom)

    # Render page to image
    pix = page.get_pixmap(matrix=mat)

    # Save as PNG
    pix.save(str(output_path))

    doc.close()
    print(f"Converted {pdf_path} to {output_path}")


if __name__ == "__main__":
    pdf_path = Path("sheets/omr_sheet.pdf")
    output_path = Path("sheets/omr_sheet.png")

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
    else:
        pdf_to_image(pdf_path, output_path)
