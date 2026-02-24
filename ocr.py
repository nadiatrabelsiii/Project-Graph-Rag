"""
Local OCR CLI using PaddleOCR-VL.

Example:
  python3 ocr.py \
    --input-pdf Data/Loi2025_17Arabe.pdf \
    --output-html OCR_Law/ocr_output.html \
    --output-json OCR_Law/ocr_output.json \
    --dpi 170
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.services.ocr_service import get_ocr_service


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PaddleOCR-VL on a PDF file.")
    parser.add_argument("--input-pdf", required=True, help="Path to input PDF")
    parser.add_argument("--output-html", default="", help="Optional output HTML path")
    parser.add_argument("--output-json", default="", help="Optional output JSON path")
    parser.add_argument("--dpi", type=int, default=170, help="Rasterization DPI")
    args = parser.parse_args()

    input_pdf = Path(args.input_pdf)
    if not input_pdf.exists():
        raise FileNotFoundError(f"Input PDF not found: {args.input_pdf}")

    service = get_ocr_service()
    result = service.process_pdf(str(input_pdf), dpi=args.dpi)

    default_html = f"{service.default_output_dir}/{input_pdf.stem}.html"
    default_json = f"{service.default_output_dir}/{input_pdf.stem}.json"
    output_html = args.output_html or default_html
    output_json = args.output_json or default_json

    html_path = service.write_ocr_html(result, output_html)
    json_path = service.write_ocr_json(result, output_json)
    print(f"Saved HTML: {html_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
