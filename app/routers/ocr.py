"""
/api/ocr — OCR processing endpoints backed by PaddleOCR-VL.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.models import OCRProcessRequest, OCRProcessResponse
from app.services.ocr_service import get_ocr_service

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["OCR"])


@router.post("/ocr/process", response_model=OCRProcessResponse)
async def process_ocr(req: OCRProcessRequest):
    input_pdf = Path(req.input_pdf_path)
    if not input_pdf.exists():
        raise HTTPException(404, f"PDF file not found: {req.input_pdf_path}")

    try:
        service = get_ocr_service()
        result = service.process_pdf(str(input_pdf), dpi=req.dpi)

        stem = input_pdf.stem
        output_html_path = req.output_html_path or f"{service.default_output_dir}/{stem}.html"
        output_json_path = req.output_json_path or f"{service.default_output_dir}/{stem}.json"

        output_html = service.write_ocr_html(result, output_html_path)
        output_json = service.write_ocr_json(result, output_json_path)
    except Exception as exc:
        log.exception("OCR processing failed")
        raise HTTPException(500, f"OCR processing error: {exc}")

    return OCRProcessResponse(
        source_file=result.get("source_file", str(input_pdf)),
        page_count=result.get("page_count", 0),
        total_text_blocks=result.get("total_text_blocks", 0),
        total_tables=result.get("total_tables", 0),
        output_html_path=output_html,
        output_json_path=output_json,
    )
