"""
PaddleOCR-VL document OCR service.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from app.config import load_environment

load_environment()


class PaddleOCRService:
    def __init__(self) -> None:
        self._pipeline: Any = None
        self.model_name = os.environ.get("PADDLEOCR_VL_MODEL_NAME", "PaddleOCR-VL-1.5")
        self.device = os.environ.get("PADDLEOCR_DEVICE", "cpu")
        self.default_dpi = int(os.environ.get("OCR_PDF_DPI", "170"))
        self.default_output_dir = os.environ.get("OCR_OUTPUT_DIR", "./OCR_Law")

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        from paddleocr import PaddleOCRVL

        kwargs: dict[str, Any] = {}
        if self.model_name:
            kwargs["model_name"] = self.model_name
        if self.device:
            kwargs["device"] = self.device

        try:
            self._pipeline = PaddleOCRVL(**kwargs)
            return self._pipeline
        except TypeError:
            # Handle API differences across PaddleOCR versions.
            pass

        try:
            self._pipeline = PaddleOCRVL(device=self.device)
            return self._pipeline
        except TypeError:
            self._pipeline = PaddleOCRVL()
            return self._pipeline

    @staticmethod
    def _strip_html(value: str) -> str:
        plain = re.sub(r"<[^>]+>", " ", value)
        return re.sub(r"\s+", " ", plain).strip()

    def _extract_page(self, raw_output: Any, page_number: int, width: int, height: int) -> dict[str, Any]:
        page: dict[str, Any] = {
            "page_number": page_number,
            "width": width,
            "height": height,
            "tables": [],
            "text_blocks": [],
            "html_content": "",
            "full_text": "",
        }

        html_parts: list[str] = []
        text_parts: list[str] = []

        if not isinstance(raw_output, list):
            return page

        for result in raw_output:
            result_json = getattr(result, "json", None)
            if not isinstance(result_json, dict):
                continue

            payload = result_json.get("res", result_json)
            parsing_list = payload.get("parsing_res_list", [])
            if not isinstance(parsing_list, list):
                continue

            for block in parsing_list:
                if not isinstance(block, dict):
                    continue

                block_type = block.get("block_label", "") or ""
                block_content = block.get("block_content", "") or ""
                block_bbox = block.get("block_bbox", []) or []
                block_id = block.get("block_id")

                if not block_content:
                    continue

                html_parts.append(block_content)
                plain_text = self._strip_html(str(block_content))
                if plain_text:
                    text_parts.append(plain_text)
                    page["text_blocks"].append(
                        {
                            "text": plain_text,
                            "type": block_type,
                            "bbox": block_bbox,
                            "block_id": block_id,
                        }
                    )

                if block_type == "table":
                    page["tables"].append(
                        {"html": block_content, "bbox": block_bbox, "block_id": block_id}
                    )

        page["html_content"] = "\n".join(html_parts)
        page["full_text"] = "\n".join(text_parts)
        return page

    def process_pdf(self, input_pdf_path: str, *, dpi: int | None = None) -> dict[str, Any]:
        from pdf2image import convert_from_path

        pipeline = self._load_pipeline()
        requested_dpi = dpi or self.default_dpi
        images = convert_from_path(input_pdf_path, dpi=requested_dpi, fmt="png")

        pages: list[dict[str, Any]] = []
        for index, image in enumerate(images, start=1):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                image_path = temp_file.name
                image.save(image_path, format="PNG")

            try:
                prediction = pipeline.predict(image_path)
                page = self._extract_page(prediction, index, image.width, image.height)
                pages.append(page)
            finally:
                try:
                    os.unlink(image_path)
                except OSError:
                    pass

        return {
            "source_file": str(input_pdf_path),
            "file_type": "pdf",
            "page_count": len(pages),
            "pages": pages,
            "full_text": "\n\n".join(page.get("full_text", "") for page in pages),
            "total_text_blocks": sum(len(page.get("text_blocks", [])) for page in pages),
            "total_tables": sum(len(page.get("tables", [])) for page in pages),
        }

    @staticmethod
    def write_ocr_json(result: dict[str, Any], output_json_path: str) -> str:
        target = Path(output_json_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(target)

    @staticmethod
    def write_ocr_html(result: dict[str, Any], output_html_path: str) -> str:
        target = Path(output_html_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []
        for page in result.get("pages", []):
            page_number = page.get("page_number", 0)
            lines.append(f"<!-- PageBreak [{page_number}] -->")
            lines.append(page.get("html_content", ""))
            lines.append("")

        target.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
        return str(target)


_service: PaddleOCRService | None = None


def get_ocr_service() -> PaddleOCRService:
    global _service
    if _service is None:
        _service = PaddleOCRService()
    return _service
