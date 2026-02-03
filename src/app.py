# app.py
# pip install --upgrade pillow numpy opencv-python torch torchvision transformers scikit-image scipy pymatting pillow-heif rawpy

from pathlib import Path
import os
import numpy as np
import cv2
from PIL import Image, ImageFile
from skimage import io
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from transformers import AutoModelForImageSegmentation
from pymatting import estimate_alpha_cf
from time import perf_counter

# Allow truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Optional plugin imports
try:
    import pillow_heif
    pillow_heif.register_heif_opener()  # enables .avif, .heif, .heic support
except ImportError:
    pass

try:
    import rawpy  # for RAF/RAW image support
except ImportError:
    rawpy = None


# =====================
# Configuration
# =====================
TARGET = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load RMBG model (uses GPU if available)
MODEL = AutoModelForImageSegmentation.from_pretrained(
    "briaai/RMBG-1.4", trust_remote_code=True
).to(DEVICE).eval()


# =====================
# Utility Functions
# =====================
def first_tensor(obj):
    """Recursively find the first torch.Tensor in nested lists/tuples/dicts."""
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (list, tuple)):
        for x in obj:
            t = first_tensor(x)
            if t is not None:
                return t
    if isinstance(obj, dict):
        for v in obj.values():
            t = first_tensor(v)
            if t is not None:
                return t
    return None


def to_hwc3_uint8(arr):
    arr = np.asarray(arr)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[2] not in (3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] not in (1, 3, 4):
        raise ValueError(f"Unexpected image shape: {arr.shape}")
    if arr.shape[2] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        scale = 255.0 if arr.max() <= 1.0 else 1.0
        arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def pil_letterbox(img_np, size=TARGET):
    img_np = to_hwc3_uint8(img_np)
    h, w = img_np.shape[:2]
    s = min(size / float(h), size / float(w))
    nh = max(1, int(round(h * s)))
    nw = max(1, int(round(w * s)))
    im_pil = Image.fromarray(img_np)
    im_resized = im_pil.resize((nw, nh), resample=Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(im_resized, (0, 0))
    padded = np.asarray(canvas)
    return padded, (h, w), (nh, nw)


def preprocess_image(im_np):
    padded, orig_hw, new_hw = pil_letterbox(im_np, TARGET)
    x = torch.tensor(padded, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    x = normalize(x, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    return x.to(DEVICE), orig_hw, new_hw


def safe_resize_float(alpha_np, out_w, out_h):
    try:
        return cv2.resize(alpha_np, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    except cv2.error:
        a8 = (np.clip(alpha_np, 0, 1) * 255).astype(np.uint8)
        return np.asarray(Image.fromarray(a8).resize((out_w, out_h), Image.BICUBIC)).astype(np.float32) / 255.0


def postprocess_mask_to_alpha(pred_tensor, orig_hw, new_hw):
    if pred_tensor.ndim == 3:
        pred_tensor = pred_tensor.unsqueeze(0)
    if pred_tensor.ndim == 2:
        pred_tensor = pred_tensor.unsqueeze(0).unsqueeze(0)
    pred = torch.squeeze(
        F.interpolate(pred_tensor, size=(TARGET, TARGET), mode="bilinear", align_corners=False), 0
    )
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-6)
    alpha_1024 = pred[0].detach().cpu().numpy()
    nh, nw = new_hw
    alpha_unpadded = alpha_1024[:nh, :nw]
    H, W = orig_hw
    alpha = safe_resize_float(alpha_unpadded, W, H)
    return np.clip(alpha, 0.0, 1.0).astype(np.float32)


def mask_to_trimap(alpha, fg_th=0.97, bg_th=0.03, k=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    fg = (alpha >= fg_th).astype(np.uint8) * 255
    bg = (alpha <= bg_th).astype(np.uint8) * 255
    fg = cv2.erode(fg, kernel, 1)
    bg = cv2.erode(bg, kernel, 1)
    trimap = np.full(alpha.shape, 128, np.uint8)
    trimap[bg == 255] = 0
    trimap[fg == 255] = 255
    return trimap


def refine_with_pymatting(orig_rgb_u8, alpha_coarse, max_size=800):
    h, w = orig_rgb_u8.shape[:2]
    scale = min(1.0, max_size / max(h, w))
    if scale < 1.0:
        small_orig = cv2.resize(orig_rgb_u8, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        small_alpha = cv2.resize(alpha_coarse, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    else:
        small_orig = orig_rgb_u8
        small_alpha = alpha_coarse

    img_f64 = small_orig.astype(np.float64) / 255.0
    tri_f64 = (mask_to_trimap(small_alpha).astype(np.float64)) / 255.0
    alpha_refined_small = estimate_alpha_cf(img_f64, tri_f64)

    if scale < 1.0:
        alpha_refined = cv2.resize(alpha_refined_small, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        alpha_refined = alpha_refined_small

    return np.clip(alpha_refined, 0.0, 1.0).astype(np.float32)


def compose_png_rgba(img_rgb_u8, alpha, out_path):
    rgba = np.dstack([img_rgb_u8, (alpha * 255).astype(np.uint8)])
    Image.fromarray(rgba).save(out_path)


# =====================
# Main Background Removal Function
# =====================
def remove_background(path_in: str, path_out: str, refine=False):
    pin = Path(path_in)
    if not pin.is_file():
        raise FileNotFoundError(str(pin))

    ext = pin.suffix.lower()

    # --- Robust TIFF/RAW/HEIF-safe loading ---
    try:
        if ext in [".raf", ".raw"]:
            if rawpy is None:
                raise ImportError("Please install rawpy to read RAF/RAW files: pip install rawpy")
            with rawpy.imread(str(pin)) as raw:
                rgb = raw.postprocess()
            orig = rgb
        else:
            orig = io.imread(str(pin))
    except Exception as e:
        print(f"Primary load failed ({e}), trying fallback with Pillow...")
        with Image.open(str(pin)) as im:
            im = im.convert("RGB")
            orig = np.array(im)

    if orig.dtype != np.uint8:
        orig = np.clip(orig * 255, 0, 255).astype(np.uint8)

    if orig.ndim == 4:
        orig = orig[0]

    orig = to_hwc3_uint8(orig)

    # --- Preprocess + Model inference ---
    x, orig_hw, new_hw = preprocess_image(orig)
    with torch.no_grad():
        out = MODEL(x)

    pred_tensor = first_tensor(out)
    if pred_tensor is None:
        raise RuntimeError(f"Could not find prediction tensor in model output type: {type(out)}")

    alpha0 = postprocess_mask_to_alpha(pred_tensor, orig_hw, new_hw)

    # --- Optional refinement ---
    if refine:
        alpha = refine_with_pymatting(orig, alpha0)
    else:
        alpha = alpha0

    # Calculate percentage of background removed
    total_pixels = alpha.size
    background_pixels = np.sum(alpha <= 0.03)
    percent_background_removed = (background_pixels / total_pixels) * 100
    print(f"Background removed: {percent_background_removed:.2f}%")

    # --- Save RGBA PNG ---
    compose_png_rgba(orig, alpha, path_out)
    return path_out


# =====================
# Run Directly
# =====================
if __name__ == "__main__":
    input_path = r"static\images\women5.jpg"  # supports .jpg, .png, .tiff, .raf, .avif, etc.
    output_path = "output_logo.png"
    os.makedirs(Path(output_path).parent, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print("Processing started...")

    start_time = perf_counter()
    result = remove_background(input_path, output_path)
    end_time = perf_counter()

    latency = end_time - start_time
    print(f"Processing latency: {latency:.2f} seconds")
    print(f"Saved: {result}")
