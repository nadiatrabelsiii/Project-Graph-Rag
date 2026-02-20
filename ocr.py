# modal_paddleocrvl_local_no_secrets.py
# Put table5.pdf next to this file.
# Run: modal run modal_paddleocrvl_local_no_secrets.py

import os
import json
import tempfile
import re
from typing import Dict, Any, List

import modal
from PIL import Image
from pydantic import BaseModel

app = modal.App("paddleocrvl-local-no-secrets")

PDF_LOCAL = "Loi2025_17Arabe_removed.pdf"
PDF_CONTAINER_PATH = "/root/Loi2025_17Arabe_removed.pdf"

# Persist PaddleX/PaddleOCR-VL downloaded models here:
# /root/.paddlex/official_models/...
models_vol = modal.Volume.from_name("paddlex-models", create_if_missing=True)

image = (
    modal.Image
    .from_registry("nvidia/cuda:12.1.1-devel-ubuntu20.04", add_python="3.11")
    .env({
        "DEBIAN_FRONTEND": "noninteractive",
        "PIP_DEFAULT_TIMEOUT": "1000",
        # Skip slow connectivity check each start
        "DISABLE_MODEL_SOURCE_CHECK": "True",
    })
    .apt_install(
        "wget", "curl",
        "libgl1-mesa-glx", "libglib2.0-0", "libgomp1", "libsm6", "libxext6", "libxrender-dev",
        "poppler-utils",
    )
    .run_commands(
        # Keep your Paddle install, but ensure NumPy is <2 to avoid the scalar conversion crash
        # reported for PaddleOCR-VL/PaddleX pipelines. :contentReference[oaicite:1]{index=1}
        "pip install --no-cache-dir numpy==1.26.4 --verbose",
        "pip install --no-cache-dir paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ --verbose",
        "pip uninstall -y safetensors || true",
        "pip install --no-cache-dir https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl",
        'pip install --no-cache-dir "paddleocr[doc-parser]" --verbose',
        "pip install --no-cache-dir pdf2image==1.17.0 --verbose",
    )
    .add_local_file(PDF_LOCAL, PDF_CONTAINER_PATH)
)

class OCRRequest(BaseModel):
    pdf_path: str = PDF_CONTAINER_PATH
    pdf_dpi: int = 170

@app.cls(
    image=image,
    gpu="A10",  # GPU for faster processing (can be changed to G40 if available)
    # Keep models across runs
    volumes={"/root/.paddlex": models_vol},
    # Keep at least one container alive (reduces cold starts)
    min_containers=1,
    # Keep it warm for longer between runs
    scaledown_window=60 * 60,
    timeout=600,
)
class OCRLocalServer:
    @modal.enter()
    def load_model(self):
        from paddleocr import PaddleOCRVL
        self.pipeline = PaddleOCRVL()

    def _pdf_to_images(self, pdf_path: str, dpi: int) -> List[Image.Image]:
        from pdf2image import convert_from_path
        return convert_from_path(pdf_path, dpi=dpi, fmt="png")

    def _vl_once(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        pages: List[Dict[str, Any]] = []

        for idx, img in enumerate(images):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                path = tf.name
                img.save(path, format="PNG")

            try:
                out = self.pipeline.predict(path)

                page: Dict[str, Any] = {
                    "page_number": idx + 1,
                    "width": img.width,
                    "height": img.height,
                    "tables": [],
                    "text_blocks": [],
                    "html_content": "",
                    "full_text": "",
                }

                html_parts: List[str] = []
                text_parts: List[str] = []

                for res in out:
                    if not hasattr(res, "json"):
                        continue
                    j = res.json
                    actual = j.get("res", j)
                    prl = actual.get("parsing_res_list", [])
                    if not isinstance(prl, list):
                        continue

                    for blk in prl:
                        if not isinstance(blk, dict):
                            continue

                        label = blk.get("block_label", "") or ""
                        content = blk.get("block_content", "") or ""
                        bbox = blk.get("block_bbox", []) or []
                        bid = blk.get("block_id")

                        if content:
                            html_parts.append(content)

                            plain = re.sub(r"<[^>]+>", " ", str(content))
                            plain = re.sub(r"\s+", " ", plain).strip()
                            if plain:
                                text_parts.append(plain)
                                page["text_blocks"].append(
                                    {"text": plain, "type": label, "bbox": bbox, "block_id": bid}
                                )

                        if label == "table":
                            page["tables"].append({"html": content, "bbox": bbox, "block_id": bid})

                page["html_content"] = "\n".join(html_parts)
                page["full_text"] = "\n".join(text_parts)
                pages.append(page)

            finally:
                try:
                    os.unlink(path)
                except Exception:
                    pass

        return pages

    @modal.method()
    def process_local_pdf(self, req: OCRRequest) -> Dict[str, Any]:
        images = self._pdf_to_images(req.pdf_path, req.pdf_dpi)
        pages = self._vl_once(images)
        return {
            "source_file": req.pdf_path,
            "file_type": "pdf",
            "page_count": len(pages),
            "pages": pages,
            "full_text": "\n\n".join(p.get("full_text", "") for p in pages),
            "total_text_blocks": sum(len(p.get("text_blocks", [])) for p in pages),
            "total_tables": sum(len(p.get("tables", [])) for p in pages),
        }

@app.local_entrypoint()
def main():
    server = OCRLocalServer()
    result = server.process_local_pdf.remote(OCRRequest())

    # Saved next to your script (on your local machine)
    with open("ocr_output.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("Saved: ocr_output.json")
