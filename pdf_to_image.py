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
    sheets_dir = Path("sheets")

    if not sheets_dir.exists():
        print(f"Directory not found: {sheets_dir}")
    else:
        # Find all PDF files in the sheets directory
        pdf_files = list(sheets_dir.glob("*.pdf"))

        if not pdf_files:
            print(f"No PDF files found in {sheets_dir}")
        else:
            print(f"Found {len(pdf_files)} PDF file(s) to convert")
            for pdf_path in pdf_files:
                # Create output path with same name but .png extension
                output_path = pdf_path.with_suffix(".png")
                pdf_to_image(pdf_path, output_path)
