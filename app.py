# app.py
import os, sys, json, tempfile
import gradio as gr
import cv2
import numpy as np
from PIL import Image

# ── Auto-download detection weights từ HuggingFace Hub ──────
CHECKPOINT = "best.pt"
HF_REPO    = "phamha/drawing-model-weights"  # ← sửa thành username của bạn

def ensure_weights():
    if not os.path.exists(CHECKPOINT):
        print("[INFO] Downloading model weights...")
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id=HF_REPO,
            filename="best.pt",
            local_dir=".",
            local_dir_use_symlinks=False,
        )
        print("[INFO] Weights ready.")

ensure_weights()

sys.path.insert(0, ".")
from src.inference import run_pipeline


# ── Gradio handler ───────────────────────────────────────────
def process(image: Image.Image):
    if image is None:
        return None, "{}", "Chưa có ảnh."

    tmp_dir  = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, "input.jpg")
    image.save(tmp_path, quality=95)

    try:
        result, vis_path = run_pipeline(
            image_path  = tmp_path,
            output_dir  = tmp_dir,
            checkpoint  = CHECKPOINT,
            conf_thresh = 0.3,
        )
    except Exception as e:
        import traceback
        return None, "{}", f"Lỗi:\n{traceback.format_exc()}"

    # Ảnh visualize
    vis_bgr = cv2.imread(vis_path)
    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

    # JSON sạch
    clean_objs = []
    for obj in result["objects"]:
        clean_objs.append({
            "id":          obj["id"],
            "class":       obj["class"],
            "confidence":  obj["confidence"],
            "bbox":        obj["bbox"],
            "ocr_content": obj["ocr_content"],
        })
    json_str = json.dumps(
        {"image": result["image"], "objects": clean_objs},
        ensure_ascii=False, indent=2,
    )

    # OCR panel
    ocr_parts = []
    for obj in result["objects"]:
        content = obj.get("ocr_content")
        if not content:
            continue
        if isinstance(content, dict):
            content = content.get("text", "")
        if not str(content).strip():
            continue
        sep = "─" * 46
        ocr_parts.append(
            f"{sep}\n"
            f"[{obj['class']} #{obj['id']}]  conf={obj['confidence']}\n"
            f"{sep}\n{content}"
        )
    ocr_text = "\n\n".join(ocr_parts) or "Không phát hiện Note / Table."

    return vis_rgb, json_str, ocr_text


# ── UI ───────────────────────────────────────────────────────
with gr.Blocks(title="Engineering Drawing Analyzer", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🔧 Engineering Drawing Analyzer
    Tự động phát hiện và trích xuất văn bản từ bản vẽ kỹ thuật (tiếng Việt & tiếng Anh).

    | Class | Màu | Mô tả |
    |-------|-----|-------|
    | 🟢 PartDrawing | Xanh lá | Vùng bản vẽ chi tiết |
    | 🟠 Note | Cam | Ghi chú, chú thích |
    | 🔴 Table | Đỏ | Bảng dữ liệu kỹ thuật |
    """)

    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="pil", label="📁 Upload bản vẽ kỹ thuật")
            btn = gr.Button("🔍 Detect & OCR", variant="primary", size="lg")

        with gr.Column(scale=1):
            out_img = gr.Image(label="✅ Kết quả detection")

    with gr.Row():
        with gr.Column(scale=1):
            out_json = gr.Code(
                language="json",
                label="📋 JSON output",
                lines=25,
            )
        with gr.Column(scale=1):
            out_ocr = gr.Textbox(
                label="📝 OCR content (Note & Table)",
                lines=25,
                max_lines=60,
            )

    btn.click(
        fn      = process,
        inputs  = [inp],
        outputs = [out_img, out_json, out_ocr],
    )

    gr.Markdown("""
    ---
    **Detection:** RT-DETR-L · mAP50 = 0.942  
    **OCR:** TrOCR (microsoft/trocr-large-handwritten) + EasyOCR fallback  
    **Hỗ trợ:** Tiếng Việt · Tiếng Anh · Chữ viết tay · Chữ in
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)