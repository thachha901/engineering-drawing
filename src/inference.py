# src/inference.py

import torch
_orig_torch_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_load

import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import RTDETR
import re
from difflib import SequenceMatcher

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {DEVICE}")

CLASS_NAMES = ["note", "part-drawing", "table"]
CLASS_DISPLAY = {"note": "Note", "part-drawing": "PartDrawing", "table": "Table"}
COLORS = {"note": (0,165,255), "part-drawing": (0,200,0), "table": (0,0,220)}

_det_model = None
_ocr_paddle = None
_ocr_paddle_en = None
_ocr_easyocr = None
_ocr_vietocr = None


REALESRGAN_AVAILABLE = False
_esrgan_upsampler = None  # Thêm biến global

try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    REALESRGAN_AVAILABLE = True
    print("[INFO] Real-ESRGAN is available")
except ImportError:
    print("[WARN] Real-ESRGAN not installed. Install: pip install realesrgan basicsr")

def get_esrgan_upsampler():
    global _esrgan_upsampler
    if not REALESRGAN_AVAILABLE:
        return None
        
    if _esrgan_upsampler is None:
        try:
            print("[INFO] Loading Real-ESRGAN model...")
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            _esrgan_upsampler = RealESRGANer(
                scale=4,
                model_path='weights/RealESRGAN_x4plus_anime_6B.pth',
                model=model,
                device=DEVICE
            )
        except Exception as e:
            print(f"[WARN] Failed to load Real-ESRGAN: {e}")
            return None
            
    return _esrgan_upsampler

def upscale_if_needed(img_bgr, min_dim=300):
    """Upscale image using Real-ESRGAN if both dimensions are below threshold."""
    h, w = img_bgr.shape[:2]
    if h < min_dim or w < min_dim:
        upsampler = get_esrgan_upsampler()
        if upsampler is not None:
            try:
                output, _ = upsampler.enhance(img_bgr, outscale=2)
                return output
            except Exception as e:
                print(f"[WARN] ESRGAN upscale failed: {e}")
    return img_bgr
# ============================================================
# DOMAIN DICTIONARY — Từ điển bản vẽ kỹ thuật Việt Nam
# ============================================================

# Từ điển các từ thường gặp trong bảng kê bản vẽ kỹ thuật
TECH_DICTIONARY = {
    # Tên chi tiết
    "bọc táp": "Bọc táp",
    "boc tap": "Bọc táp",
    "bạc táp": "Bọc táp",
    "bọc tốp": "Bọc táp",
    "bọc tot": "Bọc táp",
    "vòng đệm": "Vòng đệm",
    "vong dem": "Vòng đệm",
    "vòng dệm": "Vòng đệm",
    "vong đệm": "Vòng đệm",
    "chốt trụ": "Chốt trụ",
    "chot tru": "Chốt trụ",
    "chốt trự": "Chốt trụ",
    "chot trụ": "Chốt trụ",
    "vít": "Vít",
    "vit": "Vít",
    "bu lông": "Bu lông",
    "bu long": "Bu lông",
    "bulong": "Bu lông",
    "bu-lông": "Bu lông",
    "bulông": "Bu lông",
    "vòng đệm vênh": "Vòng đệm vênh",
    "vong dem venh": "Vòng đệm vênh",
    "vòng dệm vênh": "Vòng đệm vênh",
    "then bằng": "Then bằng",
    "then bang": "Then bằng",
    "then bảng": "Then bằng",
    "ống dẫn": "Ống dẫn",
    "ong dan": "Ống dẫn",
    "ống chẫn": "Ống dẫn",
    "ống dần": "Ống dẫn",
    "ông dẫn": "Ống dẫn",
    "ông chẫn": "Ống dẫn",
    "ống chến": "Ống dẫn",
    "ông chến": "Ống dẫn",
    "chốt chặn": "Chốt chặn",
    "chot chan": "Chốt chặn",
    "chốt chắn": "Chốt chặn",
    "cnốt chến": "Chốt chặn",
    "chốt chén": "Chốt chặn",
    "bạc lót": "Bạc lót",
    "bac lot": "Bạc lót",
    "bọc lót": "Bạc lót",
    "bạc lốt": "Bạc lót",
    "bọc lết": "Bạc lót",
    "bọc lết": "Bạc lót",
    "giá đỡ": "Giá đỡ",
    "gia do": "Giá đỡ",
    "giá dở": "Giá đỡ",
    "giá đở": "Giá đỡ",
    "bánh răng": "Bánh răng",
    "banh rang": "Bánh răng",
    "bành răng": "Bánh răng",
    "bánh rằng": "Bánh răng",
    "bảnh răng": "Bánh răng",
    "bdnh răng": "Bánh răng",
    "bdình răng": "Bánh răng",
    "hộp bánh răng": "Hộp bánh răng",
    "hop banh rang": "Hộp bánh răng",
    "hộp bành răng": "Hộp bánh răng",
    "mộp bành răng": "Hộp bánh răng",
    "mộp bánh răng": "Hộp bánh răng",
    "nắp": "Nắp",
    "nap": "Nắp",
    "năp": "Nắp",
    "nốp": "Nắp",
    
    # Vật liệu
    "đồng nhôm": "Đồng nhôm",
    "dong nhom": "Đồng nhôm",
    "đồng thanh": "Đồng nhôm",
    "đồng thann": "Đồng nhôm",
    "đổng nhôm": "Đồng nhôm",
    "đống nhôm": "Đồng nhôm",
    "đống thanh": "Đồng nhôm",
    "thép ct3": "Thép CT3",
    "thep ct3": "Thép CT3",
    "thếp ct3": "Thép CT3",
    "tnép ct3": "Thép CT3",
    "thếp cts": "Thép CT3",
    "tnếp ct3": "Thép CT3",
    "thếp ctj": "Thép CT3",
    "tnép ctj": "Thép CT3",
    "tnếp ctj": "Thép CT3",
    "thep ctj": "Thép CT3",
    "thép 65": "Thép 65",
    "thep 65": "Thép 65",
    "thếp 65": "Thép 65",
    "tnếp 65": "Thép 65",
    "thếp 65f": "Thép 65",
    "tnép 65f": "Thép 65",
    "thép 6sf": "Thép 65",
    "thep 6sf": "Thép 65",
    "thếp 6sf": "Thép 65",
    "tnép 6sf": "Thép 65",
    "thép 45": "Thép 45",
    "thep 45": "Thép 45",
    "thếp 45": "Thép 45",
    "tnếp 45": "Thép 45",
    "sắt tây": "Sắt tây",
    "sat tay": "Sắt tây",
    "sắt tay": "Sắt tây",
    "sdi tay": "Sắt tây",
    "sdi day": "Sắt tây",
    "sdi đay": "Sắt tây",
    "gang 15-32": "Gang 15-32",
    "gang15-32": "Gang 15-32",
    "gong 15-32": "Gang 15-32",
    "gong15-32": "Gang 15-32",
    "gang 15.32": "Gang 15-32",
    "gang 15 32": "Gang 15-32",
    "gong 15.32": "Gang 15-32",
    "gang1532": "Gang 15-32",
    
    # Header
    "vị trí": "Vị trí",
    "vi tri": "Vị trí",
    "v.trí": "Vị trí",
    "tên chi tiết": "Tên chi tiết",
    "ten chi tiet": "Tên chi tiết",
    "tên chi tiết máy": "Tên chi tiết máy",
    "ten chi tiet may": "Tên chi tiết máy",
    "số lg": "Số lg",
    "so lg": "Số lg",
    "số lượng": "Số lg",
    "so luong": "Số lg",
    "s.lg": "Số lg",
    "số lý": "Số lg",
    "vật liệu": "Vật liệu",
    "vat lieu": "Vật liệu",
    "vat liéu": "Vật liệu",
    "ghi chú": "Ghi chú",
    "ghi chu": "Ghi chú",
    
    # Title block
    "bản vẽ số": "Bản vẽ số",
    "ban ve so": "Bản vẽ số",
    "bản gối": "Bản vẽ số",
    "bơm bánh răng": "BƠM BÁNH RĂNG",
    "bom banh rang": "BƠM BÁNH RĂNG",
    "bớm bánh răng": "BƠM BÁNH RĂNG",
    "bản vẽ lắp số": "Bản vẽ lắp số",
    "ban ve lap so": "Bản vẽ lắp số",
    "bản vể lắp số": "Bản vẽ lắp số",
    "bán vẽ lắp số": "Bản vẽ lắp số",
    "bán vể lắp số": "Bản vẽ lắp số",
    "tỷ lệ": "Tỷ lệ",
    "ty le": "Tỷ lệ",
    "tý lệ": "Tỷ lệ",
    "bộ môn hình hoạ": "Bộ môn Hình hoạ",
    "bộ môn hình họa": "Bộ môn Hình hoạ",
    "bo mon hinh hoa": "Bộ môn Hình hoạ",
    "bộ mốn hình hoạ": "Bộ môn Hình hoạ",
    "đại học bách khoa hà nội": "Đại học Bách khoa Hà Nội",
    "dai hoc bach khoa ha noi": "Đại học Bách khoa Hà Nội",
    "đại học bách khoa": "Đại học Bách khoa Hà Nội",
    "bại hoc bách khoa": "Đại học Bách khoa Hà Nội",
    "bại học bách khoa hà nội": "Đại học Bách khoa Hà Nội",
}

# Canonical part names for fuzzy matching
CANONICAL_PARTS = [
    "Bọc táp", "Vòng đệm", "Chốt trụ", "Vít", "Bu lông",
    "Vòng đệm vênh", "Then bằng", "Ống dẫn", "Chốt chặn",
    "Bạc lót", "Giá đỡ", "Bánh răng", "Hộp bánh răng", "Nắp",
]

CANONICAL_MATERIALS = [
    "Đồng nhôm", "Thép CT3", "Thép 65", "Thép 45",
    "Sắt tây", "Gang 15-32",
]

CANONICAL_HEADERS = [
    "Vị trí", "Tên chi tiết", "Tên chi tiết máy", "Số lg", 
    "Vật liệu", "Ghi chú",
]


def fuzzy_match(text, candidates, threshold=0.55):
    """Fuzzy match text against candidates, return best match if above threshold."""
    if not text or not candidates:
        return text
    
    text_lower = text.lower().strip()
    
    # Exact match in dictionary first
    if text_lower in TECH_DICTIONARY:
        return TECH_DICTIONARY[text_lower]
    
    # Fuzzy match
    best_match = None
    best_score = 0
    
    for candidate in candidates:
        score = SequenceMatcher(None, text_lower, candidate.lower()).ratio()
        if score > best_score:
            best_score = score
            best_match = candidate
    
    if best_score >= threshold:
        return best_match
    
    return text


def correct_technical_text(text, column_type="auto"):
    """
    Sửa lỗi OCR dựa trên domain knowledge.
    column_type: "position", "name", "quantity", "material", "note", "auto"
    """
    if not text or not text.strip():
        return text
    
    original = text.strip()
    text_lower = original.lower()
    
    # 1. Exact dictionary lookup
    if text_lower in TECH_DICTIONARY:
        return TECH_DICTIONARY[text_lower]
    
    # 2. Column-specific corrections
    if column_type == "position" or (column_type == "auto" and original.replace('.','').replace(',','').isdigit()):
        # Position column — should be a number
        cleaned = re.sub(r'[^0-9]', '', original)
        if cleaned:
            return cleaned
        return original
    
    if column_type == "quantity" or (column_type == "auto" and len(original) <= 2 and any(c.isdigit() for c in original)):
        cleaned = re.sub(r'[^0-9]', '', original)
        if cleaned:
            return cleaned
        return original
    
    if column_type == "name":
        # Try fuzzy match against known part names
        result = fuzzy_match(original, CANONICAL_PARTS, threshold=0.5)
        if result != original:
            return result
        # Also check headers
        result = fuzzy_match(original, CANONICAL_HEADERS, threshold=0.5)
        if result != original:
            return result
    
    if column_type == "material":
        result = fuzzy_match(original, CANONICAL_MATERIALS, threshold=0.5)
        if result != original:
            return result
    
    if column_type == "auto":
        # Try all categories
        for candidates in [CANONICAL_PARTS, CANONICAL_MATERIALS, CANONICAL_HEADERS]:
            result = fuzzy_match(original, candidates, threshold=0.55)
            if result != original:
                return result
    
    # 3. General corrections
    text_out = original
    
    # Fix common OCR character substitutions
    # M followed by digits (bolt/screw specs)
    text_out = re.sub(r'[Mm]\s*(\d)', r'M\1', text_out)
    
    # Fix: "5x8-35" style dimensions  
    text_out = re.sub(r'(\d+)\s*[xX×]\s*(\d+)\s*[-–]\s*(\d+)', r'\1x\2-\3', text_out)
    text_out = re.sub(r'(\d+)\s*[xX×]\s*(\d+)\s*[xX×]\s*(\d+)', r'\1x\2x\3', text_out)
    
    # Fix: "3n8.35" → "5x8-35" (common OCR error for handwriting)
    text_out = re.sub(r'(\d+)\s*n\s*(\d+)', r'\1x\2', text_out)
    
    # Fix: dimension specs like "4x6x14" or "4*6*14"  
    text_out = re.sub(r'(\d+)\s*[*]\s*(\d+)\s*[*]\s*(\d+)', r'\1x\2x\3', text_out)
    
    return text_out


def correct_table_row(row, num_columns=5):
    """
    Sửa lỗi cho toàn bộ 1 row, biết vị trí cột.
    Columns: [Vị trí, Tên chi tiết, Số lg, Vật liệu, Ghi chú]
    """
    if not row:
        return row
    
    corrected = list(row)
    
    # Pad to expected columns
    while len(corrected) < num_columns:
        corrected.append("")
    
    # Trim excess
    if len(corrected) > num_columns:
        corrected = corrected[:num_columns]
    
    # Column 0: Vị trí (number)
    if corrected[0]:
        corrected[0] = correct_technical_text(corrected[0], "position")
    
    # Column 1: Tên chi tiết (part name)
    if corrected[1]:
        corrected[1] = correct_technical_text(corrected[1], "name")
    
    # Column 2: Số lg (quantity - number)
    if corrected[2]:
        corrected[2] = correct_technical_text(corrected[2], "quantity")
    
    # Column 3: Vật liệu (material)
    if corrected[3]:
        corrected[3] = correct_technical_text(corrected[3], "material")
    
    # Column 4: Ghi chú (note - keep as-is mostly)
    if corrected[4]:
        corrected[4] = correct_technical_text(corrected[4], "auto")
    
    return corrected


# ============================================================
# MODEL LOADERS
# ============================================================

def get_det_model(checkpoint="best.pt"):
    global _det_model
    if _det_model is None:
        print(f"[INFO] Loading detection model: {checkpoint}")
        _det_model = RTDETR(checkpoint)
    return _det_model


# ============================================================
# SURYA OCR (optional)
# ============================================================

SURYA_AVAILABLE = False
try:
    from surya.ocr import run_ocr
    from surya.model.detection.model import load_det_processor, load_det_model
    from surya.model.recognition.model import load_rec_model
    from surya.model.recognition.processor import load_rec_processor
    SURYA_AVAILABLE = True
    print("[INFO] Surya OCR is available")
except ImportError:
    print("[WARN] Surya OCR not installed. Install with: pip install surya-ocr")


def ocr_with_surya(img_bgr, langs=["vi", "en"]):
    if not SURYA_AVAILABLE:
        raise ImportError("Surya OCR is not installed.")
    det_processor, det_model = load_det_processor(), load_det_model()
    rec_model, rec_processor = load_rec_model(), load_rec_processor()
    from PIL import Image
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    predictions = run_ocr([pil_img], [langs], det_model, det_processor,
                          rec_model, rec_processor)
    texts = [line.text for line in predictions[0].text_lines]
    return "\n".join(texts)


# ============================================================
# VietOCR (optional - tốt cho chữ viết tay tiếng Việt)
# ============================================================

VIETOCR_AVAILABLE = False
try:
    from vietocr.tool.predictor import Predictor
    from vietocr.tool.config import Cfg
    VIETOCR_AVAILABLE = True
    print("[INFO] VietOCR is available")
except ImportError:
    print("[WARN] VietOCR not installed. Install with: pip install vietocr")


def get_vietocr():
    global _ocr_vietocr
    if _ocr_vietocr is None and VIETOCR_AVAILABLE:
        try:
            config = Cfg.load_config_from_name('vgg_transformer')
            config['cnn']['pretrained'] = True
            config['device'] = DEVICE
            _ocr_vietocr = Predictor(config)
            print("[INFO] VietOCR loaded successfully")
        except Exception as e:
            print(f"[WARN] VietOCR load failed: {e}")
    return _ocr_vietocr


def ocr_line_vietocr(img_bgr):
    """OCR a single text line image using VietOCR."""
    predictor = get_vietocr()
    if predictor is None:
        return ""
    from PIL import Image
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    text = predictor.predict(pil_img)
    return text.strip()


# ============================================================
# PaddleOCR / EasyOCR
# ============================================================

def get_paddle_reader(lang='vi'):
    global _ocr_paddle, _ocr_paddle_en
    
    if lang == 'en':
        if _ocr_paddle_en is not None:
            return _ocr_paddle_en
    else:
        if _ocr_paddle is not None:
            return _ocr_paddle
    
    try:
        from paddleocr import PaddleOCR
        print(f"[INFO] Initializing PaddleOCR PP-OCRv4 (lang={lang})...")
        reader = PaddleOCR(
            lang=lang,
            use_angle_cls=True,
            use_gpu=(DEVICE == "cuda"),
            show_log=False,
            ocr_version='PP-OCRv4',
            det_db_thresh=0.15,
            det_db_box_thresh=0.2,
            det_db_unclip_ratio=2.0,
            use_dilation=True,
            det_db_score_mode='slow',
            rec_image_shape="3,48,320",
            max_text_length=80,
            rec_batch_num=6,
        )
        if lang == 'en':
            _ocr_paddle_en = reader
        else:
            _ocr_paddle = reader
        return reader
    except Exception as e:
        print(f"[WARN] PaddleOCR init failed: {e}")
        return None


def get_easyocr_reader():
    global _ocr_easyocr
    if _ocr_easyocr is None:
        import easyocr
        _ocr_easyocr = easyocr.Reader(
            ["vi", "en"], gpu=(DEVICE == "cuda"), verbose=False
        )
    return _ocr_easyocr


# ============================================================
# PREPROCESSING
# ============================================================

def enhance_faded_text(img_bgr):
    """Giải pháp 4: Unsharp Masking kết hợp Local Thresholding cho nét chữ mờ"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Unsharp Masking (Tăng cường cạnh/nét chữ)
    gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
    unsharp = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
    
    # 2. Ngưỡng cục bộ (Local Thresholding)
    try:
        from skimage.filters import threshold_sauvola
        window_size = 25
        thresh = threshold_sauvola(unsharp, window_size=window_size)
        binary = (unsharp > thresh) * 255
        binary = binary.astype(np.uint8)
    except ImportError:
        # Fallback về OpenCV nếu chưa cài scikit-image
        binary = cv2.adaptiveThreshold(unsharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 21, 10)
    
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

def preprocess_for_ocr(img_bgr, min_width=1500, mode="note"):
    h, w = img_bgr.shape[:2]
    
    if w < min_width:
        scale = min_width / w
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)
        h, w = img_bgr.shape[:2]
    
    if mode == "note":
        img_proc = cv2.bilateralFilter(img_bgr, 9, 75, 75)
        lab = cv2.cvtColor(img_proc, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img_proc = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        kernel = np.array([[0, -0.5, 0],
                           [-0.5, 3, -0.5],
                           [0, -0.5, 0]])
        img_proc = cv2.filter2D(img_proc, -1, kernel)
        return img_proc
    
    else:  # table
        img_proc = cv2.bilateralFilter(img_bgr, 11, 80, 80)
        lab = cv2.cvtColor(img_proc, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img_proc = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return img_proc


def preprocess_for_handwriting(img_bgr, min_width=1800):
    """
    Tiền xử lý đặc biệt cho chữ viết tay.
    Tăng contrast mạnh, loại bỏ đường kẻ bảng, giữ nét chữ.
    """
    h, w = img_bgr.shape[:2]
    
    if w < min_width:
        scale = min_width / w
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Remove horizontal and vertical lines (table borders)
    h_img, w_img = gray.shape
    
    # Detect and remove horizontal lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w_img // 10), 1))
    h_lines = cv2.morphologyEx(~gray, cv2.MORPH_OPEN, h_kernel, iterations=1)
    gray_no_lines = gray.copy()
    gray_no_lines[h_lines > 128] = 255
    
    # Detect and remove vertical lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, h_img // 10)))
    v_lines = cv2.morphologyEx(~gray, cv2.MORPH_OPEN, v_kernel, iterations=1)
    gray_no_lines[v_lines > 128] = 255
    
    # CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_no_lines)
    
    # Adaptive threshold — good for handwriting
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 21, 10)
    
    # Light morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def preprocess_grayscale_variant(img_bgr, min_width=1500):
    h, w = img_bgr.shape[:2]
    if w < min_width:
        scale = min_width / w
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


# ============================================================
# OCR FUNCTIONS
# ============================================================

def ocr_single_pass(reader, img_bgr):
    """Run OCR once, return (list_of_dicts, avg_confidence)."""
    if hasattr(reader, 'ocr'):  # PaddleOCR
        result = reader.ocr(img_bgr, cls=True)
        if not result or not result[0]:
            return [], 0.0
        items = []
        confs = []
        for line in result[0]:
            box, (text, conf) = line
            if conf >= 0.15 and text.strip():
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                items.append({
                    "text": text.strip(),
                    "conf": conf,
                    "x": np.mean(xs),
                    "y": np.mean(ys),
                    "x1": min(xs), "y1": min(ys),
                    "x2": max(xs), "y2": max(ys),
                    "box": box
                })
                confs.append(conf)
        avg_conf = np.mean(confs) if confs else 0.0
        return items, avg_conf
    else:  # EasyOCR
        results = reader.readtext(img_bgr, detail=1, paragraph=False)
        items = []
        confs = []
        for (pts, text, conf) in results:
            if conf >= 0.1 and text.strip():
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                items.append({
                    "text": text.strip(),
                    "conf": conf,
                    "x": np.mean(xs),
                    "y": np.mean(ys),
                    "x1": min(xs), "y1": min(ys),
                    "x2": max(xs), "y2": max(ys),
                    "box": pts
                })
                confs.append(conf)
        avg_conf = np.mean(confs) if confs else 0.0
        return items, avg_conf


def multi_pass_ocr(img_bgr, reader, ocr_type="note"):
    """Multi-pass OCR with different preprocessings."""
    best_items = []
    best_conf = 0.0
    
    # [NẾU ẢNH NHỎ LÀ DO CẮT TỪ GÓC, ÚP SCALE LUÔN TRƯỚC KHI LÀM GÌ ĐÓ]
    img_bgr = upscale_if_needed(img_bgr, min_dim=400)
    
    # Pass 1: Color preprocessing
    img_v1 = preprocess_for_ocr(img_bgr, min_width=1500, mode=ocr_type)
    items1, conf1 = ocr_single_pass(reader, img_v1)
    if conf1 > best_conf:
        best_conf = conf1
        best_items = items1
    
    # Pass 2: Handwriting-optimized preprocessing
    img_v2 = preprocess_for_handwriting(img_bgr, min_width=1800)
    items2, conf2 = ocr_single_pass(reader, img_v2)
    if conf2 > best_conf:
        best_conf = conf2
        best_items = items2
    
    # Pass 3: Extra upscale
    img_v3 = preprocess_for_ocr(img_bgr, min_width=2500, mode=ocr_type)
    items3, conf3 = ocr_single_pass(reader, img_v3)
    if conf3 > best_conf:
        best_conf = conf3
        best_items = items3
    
    # Pass 4: Grayscale Otsu
    img_v4 = preprocess_grayscale_variant(img_bgr, min_width=1500)
    items4, conf4 = ocr_single_pass(reader, img_v4)
    if conf4 > best_conf:
        best_conf = conf4
        best_items = items4

    # --- THÊM PASS 5: Giải quyết chữ bị mờ, lợt ---
    img_v5 = enhance_faded_text(img_bgr)
    items5, conf5 = ocr_single_pass(reader, img_v5)
    if conf5 > best_conf:
        best_conf = conf5
        best_items = items5
    
    print(f"      Multi-pass confidences: {conf1:.3f}, {conf2:.3f}, {conf3:.3f}, {conf4:.3f}, {conf5:.3f} → best={best_conf:.3f}")
    return best_items, best_conf


# ============================================================
# TABLE STRUCTURE — Intersection-based cell detection
# ============================================================

def detect_lines(gray, direction="horizontal", min_length_ratio=0.15):
    """Detect lines in image."""
    h, w = gray.shape
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    if direction == "horizontal":
        kernel_len = max(30, int(w * min_length_ratio))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    else:
        kernel_len = max(30, int(h * min_length_ratio))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    
    lines_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Dilate slightly to connect broken lines
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    lines_img = cv2.dilate(lines_img, dilate_kernel, iterations=1)
    
    return lines_img


def find_line_positions(lines_img, direction="horizontal", merge_distance=10):
    """Find positions of lines (y-coords for horizontal, x-coords for vertical)."""
    if direction == "horizontal":
        projection = np.sum(lines_img, axis=1)
    else:
        projection = np.sum(lines_img, axis=0)
    
    # Find peaks
    threshold = np.max(projection) * 0.3
    positions = np.where(projection > threshold)[0]
    
    if len(positions) == 0:
        return []
    
    # Merge close positions
    merged = [positions[0]]
    for pos in positions[1:]:
        if pos - merged[-1] > merge_distance:
            merged.append(pos)
        else:
            # Take average
            merged[-1] = (merged[-1] + pos) // 2
    
    return merged


def detect_table_cells_by_intersection(img_bgr):
    """
    Detect table cells by finding intersections of horizontal and vertical lines.
    Returns list of cells as (x1, y1, x2, y2) tuples, organized in grid.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Detect horizontal and vertical lines
    h_lines = detect_lines(gray, "horizontal", min_length_ratio=0.1)
    v_lines = detect_lines(gray, "vertical", min_length_ratio=0.1)
    
    # Find line positions
    y_positions = find_line_positions(h_lines, "horizontal", merge_distance=max(8, h//50))
    x_positions = find_line_positions(v_lines, "vertical", merge_distance=max(8, w//50))
    
    print(f"      Table grid: {len(y_positions)} horizontal × {len(x_positions)} vertical lines")
    
    if len(y_positions) < 2 or len(x_positions) < 2:
        # Fallback to contour-based detection
        return detect_table_structure(img_bgr), None
    
    # Generate cells from grid intersections
    cells = []
    grid = []
    for i in range(len(y_positions) - 1):
        row_cells = []
        for j in range(len(x_positions) - 1):
            x1, y1 = x_positions[j], y_positions[i]
            x2, y2 = x_positions[j + 1], y_positions[i + 1]
            
            # Filter tiny cells
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue
            
            cells.append((x1, y1, x2, y2))
            row_cells.append((x1, y1, x2, y2))
        
        if row_cells:
            grid.append(row_cells)
    
    return cells, grid


def detect_table_structure(img_bgr):
    """Fallback contour-based cell detection."""
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    h_kernel_len = max(40, w // 15)
    v_kernel_len = max(40, h // 15)
    
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    table_structure = cv2.add(horizontal_lines, vertical_lines)
    contours, _ = cv2.findContours(table_structure, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cells = []
    min_cell_area = (w * h) * 0.001
    max_cell_area = (w * h) * 0.85
    
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if min_cell_area < area < max_cell_area and cw > 15 and ch > 15:
            cells.append((x, y, x + cw, y + ch))
    
    cells = sorted(set(cells), key=lambda r: (r[1], r[0]))
    return cells


# ============================================================
# OCR TABLE — Grid-based approach
# ============================================================

def ocr_cell_improved(img_cell, backend="paddle"):
    if img_cell is None or img_cell.size == 0:
        return ""
    # Upscale very small cells with ESRGAN
    img_cell = upscale_if_needed(img_cell, min_dim=150)
    h, w = img_cell.shape[:2]
    if h < 5 or w < 5:
        return ""
    
    # Upscale small cells
    target_h = max(64, h)
    if h < target_h:
        scale = target_h / h
        img_cell = cv2.resize(img_cell, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
    
    target_w = max(200, w)
    if w < target_w:
        scale_w = target_w / w
        if scale_w > 1:
            img_cell = cv2.resize(img_cell, None, fx=scale_w, fy=scale_w,
                                  interpolation=cv2.INTER_CUBIC)
    
    best_text = ""
    best_conf = 0
    
    # Try VietOCR first (better for handwriting)
    if VIETOCR_AVAILABLE:
        try:
            vietocr_text = ocr_line_vietocr(img_cell)
            if vietocr_text:
                best_text = vietocr_text
                best_conf = 0.7  # Default confidence for VietOCR
        except Exception as e:
            pass
    
    # Try PaddleOCR / EasyOCR
    if backend == "paddle":
        reader = get_paddle_reader('vi')
    elif backend == "surya":
        text = ocr_with_surya(img_cell, langs=["vi", "en"])
        if text.strip():
            return text.strip()
        reader = get_paddle_reader('vi')
    else:
        reader = get_easyocr_reader()
    
    if reader is None:
        reader = get_easyocr_reader()
    
    if reader is None:
        return best_text
    
    # Variant 1: Color with CLAHE
    img_proc1 = cv2.bilateralFilter(img_cell, 5, 50, 50)
    lab = cv2.cvtColor(img_proc1, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    img_proc1 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    items1, conf1 = ocr_single_pass(reader, img_proc1)
    text1 = " ".join([it["text"] for it in items1])
    if conf1 > best_conf and text1.strip():
        best_conf = conf1
        best_text = text1
    
    # Variant 2: Handwriting preprocessing (remove lines)
    img_proc2 = preprocess_for_handwriting(img_cell, min_width=300)
    items2, conf2 = ocr_single_pass(reader, img_proc2)
    text2 = " ".join([it["text"] for it in items2])
    if conf2 > best_conf and text2.strip():
        best_conf = conf2
        best_text = text2
    
    # Variant 3: Binary Otsu
    gray = cv2.cvtColor(img_cell, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_proc3 = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    items3, conf3 = ocr_single_pass(reader, img_proc3)
    text3 = " ".join([it["text"] for it in items3])
    if conf3 > best_conf and text3.strip():
        best_conf = conf3
        best_text = text3

    # --- THÊM VARIANT 4: Dành cho nét chữ viết tay bị mờ/đứt nét ---
    img_proc4 = enhance_faded_text(img_cell)
    items4, conf4 = ocr_single_pass(reader, img_proc4)
    text4 = " ".join([it["text"] for it in items4])
    if conf4 > best_conf and text4.strip():
        best_conf = conf4
        best_text = text4
        
    # Also try English PaddleOCR for specs like "M6x50", "CT3"
    
    # Also try English PaddleOCR for specs like "M6x50", "CT3"
    if backend == "paddle":
        reader_en = get_paddle_reader('en')
        if reader_en:
            items_en, conf_en = ocr_single_pass(reader_en, img_proc1)
            text_en = " ".join([it["text"] for it in items_en])
            if conf_en > best_conf and text_en.strip():
                # Only prefer English if it looks like specs/numbers
                if re.search(r'[A-Z0-9]', text_en):
                    best_conf = conf_en
                    best_text = text_en
    
    return best_text


def ocr_table_grid(img, backend="paddle"):
    """
    OCR table using grid-based cell detection.
    Key improvement: detect grid structure first, then OCR each cell.
    """
    result = detect_table_cells_by_intersection(img)
    
    if isinstance(result, tuple):
        cells, grid = result
    else:
        cells = result
        grid = None
    
    if grid and len(grid) > 0:
        print(f"      Grid detected: {len(grid)} rows")
        
        # OCR each cell in grid order
        all_rows = []
        for row_idx, row_cells in enumerate(grid):
            row_texts = []
            for cell_box in row_cells:
                x1, y1, x2, y2 = cell_box
                
                # Extract cell with padding
                pad = 3
                cy1 = max(0, y1 + pad)  # +pad to skip the line itself
                cx1 = max(0, x1 + pad)
                cy2 = min(img.shape[0], y2 - pad)
                cx2 = min(img.shape[1], x2 - pad)
                
                if cy2 <= cy1 or cx2 <= cx1:
                    row_texts.append("")
                    continue
                
                cell_img = img[cy1:cy2, cx1:cx2]
                text = ocr_cell_improved(cell_img, backend=backend)
                row_texts.append(text.strip())
            
            if any(t for t in row_texts):  # Skip empty rows
                all_rows.append(row_texts)
        
        if all_rows:
            # Determine number of columns (use most common column count)
            col_counts = [len(r) for r in all_rows]
            if col_counts:
                expected_cols = max(set(col_counts), key=col_counts.count)
                
                # Normalize rows to same column count
                normalized_rows = []
                for row in all_rows:
                    if len(row) < expected_cols:
                        row = row + [""] * (expected_cols - len(row))
                    elif len(row) > expected_cols:
                        row = row[:expected_cols]
                    normalized_rows.append(row)
                
                # Apply domain correction
                corrected_rows = []
                for row in normalized_rows:
                    if expected_cols >= 4:
                        corrected = correct_table_row(row, num_columns=expected_cols)
                    else:
                        corrected = [correct_technical_text(cell) for cell in row]
                    corrected_rows.append(corrected)
                
                text = "\n".join(" | ".join(r) for r in corrected_rows)
                return {"rows": corrected_rows, "text": text}
    
    # Fallback if grid detection failed
    return None


def ocr_table(img_path, backend="paddle"):
    img = cv2.imread(img_path)
    if img is None:
        return {"rows": [], "text": ""}
    
    # Strategy 1: Grid-based cell detection + OCR
    print(f"      Trying grid-based table OCR...")
    result = ocr_table_grid(img, backend)
    if result and result.get("rows"):
        print(f"      Grid OCR: {len(result['rows'])} rows")
        return result
    
    # Strategy 2: PPStructure (if paddle backend)
    if backend == "paddle":
        pp_engine = get_pp_structure()
        if pp_engine is not None:
            try:
                h, w = img.shape[:2]
                if w < 1200:
                    scale = 1200 / w
                    img_scaled = cv2.resize(img, None, fx=scale, fy=scale,
                                            interpolation=cv2.INTER_CUBIC)
                else:
                    img_scaled = img
                
                result_pp = pp_engine(img_scaled)
                for item in result_pp:
                    if item.get('type') == 'table':
                        html = item.get('res', {}).get('html', '')
                        if html:
                            rows = parse_html_table(html)
                            if rows:
                                # Apply domain correction
                                corrected_rows = []
                                for row in rows:
                                    corrected = [correct_technical_text(cell) for cell in row]
                                    corrected_rows.append(corrected)
                                text = "\n".join(" | ".join(r) for r in corrected_rows)
                                print(f"      PPStructure: {len(corrected_rows)} rows")
                                return {"rows": corrected_rows, "text": text, "html": html}
            except Exception as e:
                print(f"      PPStructure error: {e}")
    
    # Strategy 3: Contour-based cell detection
    print(f"      Trying contour-based table OCR...")
    result = ocr_table_manual(img, img_path, backend)
    
    # Apply domain correction to final result
    if result.get("rows"):
        corrected_rows = []
        for row in result["rows"]:
            corrected = [correct_technical_text(cell) for cell in row]
            corrected_rows.append(corrected)
        result["rows"] = corrected_rows
        result["text"] = "\n".join(" | ".join(r) for r in corrected_rows)
    
    return result


def ocr_table_manual(img, img_path, backend="paddle"):
    cells = detect_table_structure(img)
    
    if cells:
        ocr_results = []
        for (x1, y1, x2, y2) in cells:
            cell_w, cell_h = x2 - x1, y2 - y1
            img_h, img_w = img.shape[:2]
            if cell_w > img_w * 0.9 and cell_h > img_h * 0.9:
                continue
            if cell_w < 15 or cell_h < 15:
                continue
            
            pad = 3
            cy1 = max(0, y1 - pad)
            cx1 = max(0, x1 - pad)
            cy2 = min(img.shape[0], y2 + pad)
            cx2 = min(img.shape[1], x2 + pad)
            cell_img = img[cy1:cy2, cx1:cx2]
            
            text = ocr_cell_improved(cell_img, backend=backend)
            if text:
                ocr_results.append({
                    "text": text.strip(),
                    "x": (x1 + x2) // 2,
                    "y": (y1 + y2) // 2,
                    "box": (x1, y1, x2, y2)
                })
        
        if ocr_results:
            rows = group_rows(ocr_results, vertical_thresh_ratio=0.5)
            return {
                "rows": rows,
                "text": "\n".join(" | ".join(r) for r in rows)
            }
    
    return ocr_table_fullimage(img, backend)


_pp_structure = None

def get_pp_structure():
    global _pp_structure
    if _pp_structure is not None:
        return _pp_structure
    try:
        from paddleocr import PPStructure
        print("[INFO] Initializing PPStructure...")
        _pp_structure = PPStructure(
            table=True, ocr=True, lang='vi',
            show_log=False, use_gpu=(DEVICE == "cuda"),
        )
        return _pp_structure
    except Exception as e:
        print(f"[WARN] PPStructure init failed: {e}")
        return None


def parse_html_table(html_str):
    rows = []
    tr_pattern = re.findall(r'<tr>(.*?)</tr>', html_str, re.DOTALL)
    for tr in tr_pattern:
        cells = re.findall(r'<td[^>]*>(.*?)</td>', tr, re.DOTALL)
        clean_cells = []
        for cell in cells:
            clean = re.sub(r'<[^>]+>', '', cell).strip()
            clean_cells.append(clean)
        if clean_cells:
            rows.append(clean_cells)
    return rows


def ocr_table_fullimage(img, backend="paddle"):
    if backend == "surya":
        text = ocr_with_surya(img, langs=["vi", "en"])
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        rows = [[line] for line in lines]
        return {"rows": rows, "text": text}
    
    reader = get_paddle_reader('vi') if backend == "paddle" else get_easyocr_reader()
    if reader is None:
        reader = get_easyocr_reader()
    
    img_proc = preprocess_for_ocr(img, min_width=1500, mode="table")
    items, _ = ocr_single_pass(reader, img_proc)
    
    if not items:
        # Try handwriting preprocessing
        img_hw = preprocess_for_handwriting(img, min_width=1800)
        items, _ = ocr_single_pass(reader, img_hw)
    
    if not items:
        return {"rows": [], "text": ""}
    
    # Apply corrections
    for item in items:
        item["text"] = correct_technical_text(item["text"])
    
    rows = group_rows(items, vertical_thresh_ratio=0.6)
    return {"rows": rows, "text": "\n".join(" | ".join(r) for r in rows)}


# ============================================================
# GROUP ROWS
# ============================================================

def group_rows(items, vertical_thresh_ratio=0.6):
    if not items:
        return []
    items_sorted = sorted(items, key=lambda x: x["y"])
    y_vals = [it["y"] for it in items_sorted]
    
    if len(y_vals) > 1:
        gaps = [y_vals[i+1] - y_vals[i] for i in range(len(y_vals)-1)]
        median_gap = np.median(gaps)
        thresh = max(8, median_gap * vertical_thresh_ratio)
    else:
        thresh = 12
    
    rows = []
    current_row = [items_sorted[0]]
    for it in items_sorted[1:]:
        if it["y"] - current_row[-1]["y"] < thresh:
            current_row.append(it)
        else:
            current_row.sort(key=lambda x: x["x"])
            rows.append(current_row)
            current_row = [it]
    current_row.sort(key=lambda x: x["x"])
    rows.append(current_row)
    
    return [[it["text"] for it in row] for row in rows]


# ============================================================
# POST-PROCESSING
# ============================================================

def post_process_ocr_text(text):
    if not text:
        return text
    
    text = re.sub(r'(?<=[0-9])O(?=[0-9])', '0', text)
    text = re.sub(r'(?<=M)O', '0', text)
    text = re.sub(r'(?<=Ø)O', '0', text)
    text = re.sub(r'(?<=[0-9])[lI](?=[0-9])', '1', text)
    text = re.sub(r'(\d+)\s*[xX]\s*(\d+)', r'\1×\2', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Domain correction
    text = correct_technical_text(text)
    
    return text


# ============================================================
# OCR NOTE
# ============================================================

def ocr_note(img_path, backend="paddle"):
    img = cv2.imread(img_path)
    if img is None:
        return ""
    
    if backend == "surya":
        text = ocr_with_surya(img, langs=["vi", "en"])
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        processed = [post_process_ocr_text(t) for t in lines]
        return "\n".join(processed)
    
    reader_vi = get_paddle_reader('vi') if backend == "paddle" else None
    reader_en = get_paddle_reader('en') if backend == "paddle" else None
    
    if reader_vi is None and reader_en is None:
        reader_vi = get_easyocr_reader()
    
    best_items = []
    best_conf = 0.0
    
    if reader_vi:
        items, conf = multi_pass_ocr(img, reader_vi, "note")
        if conf > best_conf:
            best_conf = conf
            best_items = items
    
    if reader_en:
        items, conf = multi_pass_ocr(img, reader_en, "note")
        if conf > best_conf:
            best_conf = conf
            best_items = items
    
    texts = [it["text"] for it in best_items]
    processed = [post_process_ocr_text(t) for t in texts]
    processed = [t for t in processed if t]
    
    return "\n".join(processed)


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(image_path, output_dir="outputs",
                 checkpoint="best.pt", conf_thresh=0.3,
                 ocr_backend="paddle"):
    image_path = str(image_path)
    img_name = Path(image_path).name
    stem = Path(image_path).stem
    crop_dir = Path(output_dir) / stem / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    
    model = get_det_model(checkpoint)
    results = model(image_path, imgsz=1024, conf=conf_thresh,
                    iou=0.5, device=DEVICE, verbose=False)
    
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Cannot read: {image_path}")
    
    objects = []
    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_idx = int(box.cls[0])
        conf_val = round(float(box.conf[0]), 4)
        cls_raw = CLASS_NAMES[cls_idx]
        cls_show = CLASS_DISPLAY[cls_raw]
        
        pad = 10
        crop = img_bgr[max(0, y1-pad):min(img_bgr.shape[0], y2+pad),
                       max(0, x1-pad):min(img_bgr.shape[1], x2+pad)]
        crop_path = str(crop_dir / f"{cls_show}_{i+1}.jpg")
        cv2.imwrite(crop_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 98])
        
        ocr_content = None
        if cls_raw == "note":
            print(f"[OCR] Note #{i+1} ({x2-x1}x{y2-y1}px)...")
            ocr_content = ocr_note(crop_path, backend=ocr_backend)
            print(f"      → {repr(ocr_content[:120]) if ocr_content else 'EMPTY'}")
        elif cls_raw == "table":
            print(f"[OCR] Table #{i+1} ({x2-x1}x{y2-y1}px)...")
            ocr_content = ocr_table(crop_path, backend=ocr_backend)
            preview = ocr_content.get("text", "")[:120]
            print(f"      → {repr(preview) if preview else 'EMPTY'}")
        
        objects.append({
            "id": i+1, "class": cls_show,
            "confidence": conf_val,
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "crop_path": crop_path,
            "ocr_content": ocr_content,
        })
        
        color = COLORS[cls_raw]
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_show} {conf_val:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_bgr, (x1, y1-th-10), (x1+tw+8, y1), color, -1)
        cv2.putText(img_bgr, label, (x1+4, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    vis_path = str(Path(output_dir) / stem / "result_vis.jpg")
    cv2.imwrite(vis_path, img_bgr)
    
    result = {"image": img_name, "objects": objects}
    json_path = str(Path(output_dir) / stem / "result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"[✓] {len(objects)} objects | {vis_path} | {json_path}")
    return result, vis_path


if __name__ == "__main__":
    import sys
    img = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    backend = sys.argv[2] if len(sys.argv) > 2 else "easyocr"
    result, _ = run_pipeline(img, ocr_backend=backend)
    print(json.dumps(result, ensure_ascii=False, indent=2))